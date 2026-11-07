"""NVIDIA GR00T N1.7 PolicyServer speaking the ROS 2 client's REQ/REP wire.

Implements the REQ/REP half of docs/SERVER_CLIENT_COMMUNICATION.md: one msgpack map per
request and per reply, two endpoints.

- ``get_config`` -- the one-time handshake: layout, joint names, units, rate, image keys.
- ``get_action`` -- one observation in, one complete ``(T, A)`` action chunk out, with
  ``action_id`` equal to the request's ``observation_id``.

PolicyServer owns REP socket setup and cleanup; Gr00tPolicy owns the complete model
pipeline, including relative-action decoding, so every chunk is absolute joint targets.
Only envelope validation, NPZ-to-modality adaptation, and reply building live here.

No server clock value is sent: the reply's stages are `time.monotonic_ns()` offsets from
the request's arrival, added to the echoed `observation_sent_ms`.
"""

import io
import logging
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import tyro
import zmq
from gr00t.policy.policy import BasePolicy
from gr00t.policy.server_client import MsgSerializer, PolicyServer

logger = logging.getLogger(__name__)

_RECV_TIMEOUT_MS = 200
_STATE_KEY = "joint_states"
_TASK_KEY = "task"


@dataclass
class ServerConfig:
    model_path: str
    # The client contract: must equal the client profile's layout_id. Joint names are the
    # client's own, mapped onto the checkpoint's modality keys by PROXY_KEYS; camera names
    # are the client's own, mapped onto the checkpoint's video keys by PROXY_IMAGE_KEYS.
    # Both are printed at startup. Units are not on the wire: the checkpoint must be
    # trained in the client's rollout.joint_unit.
    #
    # control_rate_hz is deliberately absent. This server returns one complete action
    # chunk per request and never paces anything, so the rate was only ever a number the
    # two sides had to keep equal for no behavioural reason. The client owns its own
    # playback rate and no longer sends it.
    layout_id: str
    embodiment_tag: str = "new_embodiment"
    device: str = "cuda:0"
    host: str = "0.0.0.0"
    port: int = 5555
    strict: bool = True
    # Directory of TRT engines from scripts/build_trt.sh (<output-dir>/engines). Unset serves
    # PyTorch. Engines bake a static batch of 1 and the GPU architecture they were built on.
    trt_engine_path: str | None = None


# Proxy between the checkpoint's modality keys and the client's joint names, used for both
# state (client -> policy) and action (policy -> client). Keys are policy elements:
# "<modality_key>.<i>" is element i of a multi-dim key, a bare "<modality_key>" is a 1-dim
# key. Values are client joint names; their order here is the wire order of the client's
# joint_states vector and of the reply's action columns.
PROXY_KEYS: dict[str, str] = {
    "right_arm.0": "right_joint_1.pos",
    "right_arm.1": "right_joint_2.pos",
    "right_arm.2": "right_joint_3.pos",
    "right_arm.3": "right_joint_4.pos",
    "right_arm.4": "right_joint_5.pos",
    "right_arm.5": "right_joint_6.pos",
    "right_arm.6": "right_joint_7.pos",
    "right_hand": "right_gripper.pos",
    "left_arm.0": "left_joint_1.pos",
    "left_arm.1": "left_joint_2.pos",
    "left_arm.2": "left_joint_3.pos",
    "left_arm.3": "left_joint_4.pos",
    "left_arm.4": "left_joint_5.pos",
    "left_arm.5": "left_joint_6.pos",
    "left_arm.6": "left_joint_7.pos",
    "left_hand": "left_gripper.pos",
}

# Proxy between the checkpoint's video keys and the client's camera names (the NPZ image
# keys). Must map every video key, one-to-one.
PROXY_IMAGE_KEYS: dict[str, str] = {
    "head_camera": "body_camera",
    "left_camera": "left_camera",
    "right_camera": "right_camera",
}


def build_joint_layout(
    proxy_keys: dict[str, str], modality_keys: list[str], *, modality: str
) -> dict[str, np.ndarray]:
    """For each policy modality key, the client wire indices of its elements, in element order.

    Raises ValueError unless the proxy covers every element of every modality key exactly once.
    """
    elements: dict[str, dict[int | None, int]] = {key: {} for key in modality_keys}
    for wire_index, policy_name in enumerate(proxy_keys):
        if policy_name in elements:
            key, element = policy_name, None
        else:
            key, _, suffix = policy_name.rpartition(".")
            if key not in elements or not suffix.isdigit():
                raise ValueError(
                    f"proxy key {policy_name!r} names no {modality} modality key "
                    f"(expected '<key>' or '<key>.<i>' with key in {modality_keys})"
                )
            element = int(suffix)
        if element in elements[key]:
            raise ValueError(f"proxy maps {modality} element {policy_name!r} more than once")
        elements[key][element] = wire_index

    layout = {}
    for key, by_element in elements.items():
        if not by_element:
            raise ValueError(f"proxy does not map {modality} modality key {key!r}")
        if None in by_element:
            if len(by_element) != 1:
                raise ValueError(f"proxy maps {key!r} both as a scalar and by element")
            layout[key] = np.array([by_element[None]])
            continue
        if sorted(by_element) != list(range(len(by_element))):
            raise ValueError(
                f"proxy elements of {modality} key {key!r} must be 0..{len(by_element) - 1}, "
                f"got {sorted(by_element)}"
            )
        layout[key] = np.array([by_element[i] for i in range(len(by_element))])
    return layout


def _npy(array: np.ndarray) -> dict:
    """The client's ndarray envelope. NVIDIA's MsgSerializer would emit msgpack_numpy."""
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return {"__ndarray_class__": True, "as_npy": buffer.getvalue()}


def _error(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_request(request: dict, *, layout_id: str) -> None:
    """Raise ValueError unless `request` is a get_action envelope for this server."""
    # The client's counter opens each run at 0.
    if not _is_int(request.get("observation_id")) or request["observation_id"] < 0:
        raise ValueError("request 'observation_id' must be a non-negative integer")
    if not _is_int(request.get("observation_sent_ns")) or request["observation_sent_ns"] <= 0:
        raise ValueError("request 'observation_sent_ns' must be a positive integer")
    # Milliseconds since the client's rollout started: the first observation may be 0.
    if not _is_int(request.get("observation_sent_ms")) or request["observation_sent_ms"] < 0:
        raise ValueError("request 'observation_sent_ms' must be a non-negative integer")
    if request.get("layout_id") != layout_id:
        raise ValueError(
            f"request layout_id {request.get('layout_id')!r} does not match this server's "
            f"layout_id {layout_id!r}"
        )
    if not isinstance(request.get("observations"), bytes):
        raise ValueError("request 'observations' must be NPZ bytes")


def build_reply(
    *,
    request: dict,
    received_ns: int,
    inference_start_ns: int,
    inference_done_ns: int,
    layout_id: str,
    actions: np.ndarray,
) -> dict:
    """The get_action reply. Every `*_ns` argument is `time.monotonic_ns()` on this server;
    `action_sent` is read here, right before sending."""
    sent_ms = request["observation_sent_ms"]

    def ms(stage_ns: int) -> int:
        # Arrival is observation_sent_ms on the client's timeline; add server time since.
        return sent_ms + round((stage_ns - received_ns) / 1e6)

    return {
        "action_id": request["observation_id"],
        "observation_sent_ns": request["observation_sent_ns"],
        "observation_sent_ms": sent_ms,
        "inference_start_ms": ms(inference_start_ns),
        "inference_done_ms": ms(inference_done_ns),
        "action_sent_ms": ms(time.monotonic_ns()),
        "layout_id": layout_id,
        "actions": _npy(np.ascontiguousarray(actions, dtype=np.float32)),
    }


class Gr00tSyncServer(PolicyServer):
    """Reuse NVIDIA's REP socket lifecycle; serve get_config and get_action."""

    def __init__(
        self,
        policy: BasePolicy,
        *,
        layout_id: str,
        proxy_keys: dict[str, str] = PROXY_KEYS,
        proxy_image_keys: dict[str, str] = PROXY_IMAGE_KEYS,
        host: str = "*",
        port: int = 5555,
    ) -> None:
        modalities = policy.get_modality_config()
        # The checkpoint's modality keys, in saved order.
        self._image_keys = list(modalities["video"].modality_keys)
        self._state_keys = list(modalities["state"].modality_keys)
        self._action_keys = list(modalities["action"].modality_keys)
        self._language_keys = list(modalities["language"].modality_keys)

        if not layout_id:
            raise ValueError("layout_id must be set to the client's layout_id")
        # The client sends one joint vector and receives the same joints as action columns,
        # both in proxy order, named by the client.
        self._joint_names = list(proxy_keys.values())
        if len(set(self._joint_names)) != len(self._joint_names):
            raise ValueError(f"proxy maps two policy elements onto one client joint: {proxy_keys}")
        # policy key -> client wire indices of its elements; ValueError on a bad proxy.
        self._state_layout = build_joint_layout(proxy_keys, self._state_keys, modality="state")
        self._action_layout = build_joint_layout(proxy_keys, self._action_keys, modality="action")
        # policy video key -> client camera name (NPZ image key).
        if set(proxy_image_keys) != set(self._image_keys):
            raise ValueError(
                f"image proxy keys {sorted(proxy_image_keys)} must be exactly the checkpoint's "
                f"video keys {sorted(self._image_keys)}"
            )
        if len(set(proxy_image_keys.values())) != len(proxy_image_keys):
            raise ValueError(f"image proxy maps two video keys onto one camera: {proxy_image_keys}")
        self._camera_names = {key: proxy_image_keys[key] for key in self._image_keys}

        # Validated before binding, so a bad config never holds the port.
        super().__init__(policy, host=host, port=port)
        self._layout_id = layout_id
        self._config = {
            "layout_id": layout_id,
            # Client joint names via the proxy, in wire order.
            "state_features": list(self._joint_names),
            "action_features": list(self._joint_names),
            # Client camera names via the image proxy.
            "image_keys": sorted(self._camera_names.values()),
        }
        logger.info("layout_id          : %s", layout_id)
        logger.info("image keys (NPZ)   : %s (client camera mappings must be exactly these)",
                    sorted(self._camera_names.values()))
        logger.info("joint names        : %s (state_features/action_features; wire order)",
                    self._joint_names)
        logger.info("joint proxy        : %s", proxy_keys)
        logger.info("image proxy        : %s", self._camera_names)

    def run(self) -> None:
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        while self.running:
            if not poller.poll(_RECV_TIMEOUT_MS):
                continue
            raw_request = self.socket.recv()
            received_ns = time.monotonic_ns()  # on arrival, before decoding
            self.socket.send(MsgSerializer.to_bytes(self._dispatch(raw_request, received_ns)))

    def _dispatch(self, raw_request: bytes, received_ns: int) -> dict:
        """Always return exactly one reply, so a bad request never wedges the REP socket."""
        try:
            request = MsgSerializer.from_bytes(raw_request)
        except Exception as exc:
            return _error("MALFORMED_REQUEST", f"undecodable request: {exc}")
        endpoint = request.get("endpoint") if isinstance(request, dict) else None
        if endpoint == "get_config":
            return self._config
        if endpoint != "get_action":
            return _error(
                "MALFORMED_REQUEST",
                f"endpoint must be 'get_action' or 'get_config', got {endpoint!r}",
            )

        try:
            validate_request(request, layout_id=self._layout_id)
            observation = self._decode_observation(request["observations"])
        except (ValueError, KeyError, OSError) as exc:
            return _error("MALFORMED_REQUEST", str(exc))

        inference_start_ns = time.monotonic_ns()
        try:
            # get_action validates inputs/outputs and preprocesses, predicts, decodes,
            # and unnormalizes the full chunk, including relative-action decoding.
            action, _ = self.policy.get_action(observation)
            inference_done_ns = time.monotonic_ns()
            chunk = self._encode_action(action)
        except Exception as exc:
            logger.exception(
                "GR00T inference failed: observation_id=%s", request["observation_id"]
            )
            return _error("INFERENCE_ERROR", str(exc))
        if not np.isfinite(chunk).all():
            return _error("INFERENCE_ERROR", "policy returned non-finite actions")

        reply = build_reply(
            request=request,
            received_ns=received_ns,
            inference_start_ns=inference_start_ns,
            inference_done_ns=inference_done_ns,
            layout_id=self._layout_id,
            actions=chunk,
        )
        logger.debug(
            "observation_id=%s received->done %.1f ms shape=%s",
            request["observation_id"], (inference_done_ns - received_ns) / 1e6, chunk.shape,
        )
        return reply

    def _decode_observation(self, payload: bytes) -> dict:
        """NPZ bytes -> NVIDIA's keyed observation dict. Raises ValueError."""
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            received = set(data.files) - {_STATE_KEY, _TASK_KEY}
            expected = set(self._camera_names.values())
            if received != expected:
                raise ValueError(
                    f"observation image keys {sorted(received)} do not match the server's "
                    f"cameras {sorted(expected)} (missing: {sorted(expected - received)}, "
                    f"unexpected: {sorted(received - expected)})"
                )
            if _STATE_KEY not in data.files or _TASK_KEY not in data.files:
                raise ValueError(f"observation NPZ must hold {_STATE_KEY!r} and {_TASK_KEY!r}")
            state = data[_STATE_KEY]
            if state.dtype != np.float32 or state.shape != (len(self._joint_names),):
                raise ValueError(
                    f"{_STATE_KEY} must be float32 ({len(self._joint_names)},) in order "
                    f"{self._joint_names}, got {state.dtype} {state.shape}"
                )
            if not np.isfinite(state).all():
                raise ValueError(f"{_STATE_KEY} contains NaN or Inf")
            task = str(data[_TASK_KEY])
            video = {}
            for key, camera in self._camera_names.items():
                image = cv2.imdecode(data[camera], cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"invalid JPEG for camera {camera!r}")
                # The client's image_channel_order=rgb passes RGB to imencode;
                # imdecode therefore restores those same RGB channel values.
                video[key] = image[None, None]
        return {
            "video": video,
            # Client joint vector -> per-key (B=1, T=1, D) arrays via the proxy.
            "state": {
                key: state[indices].reshape(1, 1, -1)
                for key, indices in self._state_layout.items()
            },
            "language": {key: [[task]] for key in self._language_keys},
        }

    def _encode_action(self, action: dict) -> np.ndarray:
        """Policy per-key (B, T, D) actions -> one (T, joints) chunk in client wire order."""
        horizon = action[self._action_keys[0]].shape[1]
        chunk = np.empty((horizon, len(self._joint_names)), dtype=np.float32)
        for key, indices in self._action_layout.items():
            values = np.asarray(action[key][0])
            if values.shape != (horizon, len(indices)):
                raise ValueError(
                    f"policy action {key!r} has shape {values.shape}, but the proxy maps "
                    f"{len(indices)} elements over horizon {horizon}"
                )
            chunk[:, indices] = values
        return chunk


def enable_trt(policy: BasePolicy, engine_path: str) -> None:
    """Swap the policy's backbone and action head for the full-pipeline TRT engines."""
    import gr00t
    from gr00t.deployment.modes import InferenceMode

    # trt_model_forward lives in the Isaac-GR00T checkout, outside the gr00t package.
    deploy_dir = Path(gr00t.__file__).resolve().parents[1] / "scripts" / "deployment"
    sys.path.insert(0, str(deploy_dir))
    from _trt_contract import resolve_batch_size
    from trt_model_forward import setup_tensorrt_engines

    resolve_batch_size(engine_path, 1, source="serve")  # fail now, not on the first request
    setup_tensorrt_engines(policy, engine_path, mode=InferenceMode.n17_full_pipeline)


def main(config: ServerConfig) -> None:
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.resolve(config.embodiment_tag),
        model_path=config.model_path, device=config.device, strict=config.strict,
    )
    if config.trt_engine_path is not None:
        enable_trt(policy, config.trt_engine_path)
        logger.info("TensorRT engines   : %s", config.trt_engine_path)
    with Gr00tSyncServer(
        policy,
        layout_id=config.layout_id,
        host=config.host,
        port=config.port,
    ) as server:
        try:
            server.run()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(ServerConfig))
