# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


# Human has one camera, robot has three (head + two wrist).
# Both share one action horizon (must be <= the model's action_horizon, 40).
ACTION_HORIZON = 32


def _action_config(rep: ActionRepresentation) -> ActionConfig:
    return ActionConfig(rep=rep, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)


def _make_human_config(state_keys: list[str], action_reps: list[ActionRepresentation]) -> dict:
    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=["head_camera"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=state_keys),
        "action": ModalityConfig(
            delta_indices=list(range(ACTION_HORIZON)),
            modality_keys=state_keys,
            action_configs=[_action_config(rep) for rep in action_reps],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task_description"]
        ),
    }


def _make_robot_config(state_keys: list[str], action_reps: list[ActionRepresentation]) -> dict:
    return {
        "video": ModalityConfig(
            delta_indices=[0], modality_keys=["head_camera", "left_camera", "right_camera"]
        ),
        "state": ModalityConfig(delta_indices=[0], modality_keys=state_keys),
        "action": ModalityConfig(
            delta_indices=list(range(ACTION_HORIZON)),
            modality_keys=state_keys,
            action_configs=[_action_config(rep) for rep in action_reps],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task_description"]
        ),
    }


# Human data (pick_and_place2_ee_v2.1): 24-d EEF pose (xyz + rot, 6 per arm) + 6-d hand per side.
# EEF poses are expressed relative to the current pose; hand joints are absolute.
human = _make_human_config(
    ["left_eef_pose", "right_eef_pose", "left_hand_joints", "right_hand_joints"],
    [
        ActionRepresentation.RELATIVE,
        ActionRepresentation.RELATIVE,
        ActionRepresentation.ABSOLUTE,
        ActionRepresentation.ABSOLUTE,
    ],
)

# Robot data (ego_vdrd_merged_v2.1): 26-d = 7-DoF joints per arm + 6-d hand per side.
# Arm joints are relative to the current joints; hand joints are absolute.
robot = _make_robot_config(
    ["left_arm", "right_arm", "left_hand", "right_hand"],
    [
        ActionRepresentation.RELATIVE,
        ActionRepresentation.RELATIVE,
        ActionRepresentation.ABSOLUTE,
        ActionRepresentation.ABSOLUTE,
    ],
)

# One tag per embodiment -> each gets its own state encoder / action encoder+decoder projector.
register_modality_config(human, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
register_modality_config(robot, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT_2)
