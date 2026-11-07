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


def _make_human_config() -> dict:
    state_keys = ["left_eef_pose", "right_eef_pose", "left_hand_joints", "right_hand_joints"]
    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=["head_camera"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=state_keys),
        "action": ModalityConfig(
            delta_indices=list(range(ACTION_HORIZON)),
            modality_keys=state_keys,
            action_configs=[
                ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.EEF, format=ActionFormat.XYZ_ROT6D),
                ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.EEF, format=ActionFormat.XYZ_ROT6D),
                ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
                ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task_description"]
        ),
    }


def _make_robot_config() -> dict:
    state_keys = ["left_arm", "right_arm", "left_hand", "right_hand"]
    return {
        "video": ModalityConfig(
            delta_indices=[0], modality_keys=["head_camera", "left_camera", "right_camera"]
        ),
        "state": ModalityConfig(delta_indices=[0], modality_keys=state_keys),
        "action": ModalityConfig(
            delta_indices=list(range(ACTION_HORIZON)),
            modality_keys=state_keys,
            action_configs=[
                ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
                ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
                ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
                ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),
            ]
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task_description"]
        ),
    }

human = _make_human_config()
robot = _make_robot_config()

# One tag per embodiment -> each gets its own state encoder / action encoder+decoder projector.
register_modality_config(human, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
register_modality_config(robot, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT_2)
