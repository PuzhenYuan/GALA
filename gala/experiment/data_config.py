# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from abc import ABC, abstractmethod


from dataclasses import dataclass


from typing import Optional


from gala.data.dataset import ModalityConfig


from gala.data.transform.base import ComposedModalityTransform, ModalityTransform


from gala.data.transform.concat import ConcatTransform


from gala.data.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
    CoordinateTransform,
    StateActionSubtractPosition,
    StateActionRotateEuler,
    DropKeys,
    HierarchicalRelativeTransform,
)


from gala.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
    VideoOffsetCrop,
    VideoHorizontalFlip,
)


from gala.model.transforms import GR00TTransform, GR00TTransformWithGoalImage


from gala.model.backbone.eagle_backbone import DEFAULT_EAGLE_PATH


@dataclass
class BaseDataConfig(ABC):
    eagle_path = DEFAULT_EAGLE_PATH
    use_bridge = False

    def __init__(
        self,
        eagle_path: str = DEFAULT_EAGLE_PATH,
        use_bridge: bool = False,
        ignore_lang_prefix: bool = False,
        enable_imagenet_preprocessing: bool = True,
        num_bridge_tokens: int = None,
        tokenizer_only: bool = False,
    ):
        self.eagle_path = eagle_path
        self.use_bridge = use_bridge
        self.ignore_lang_prefix = ignore_lang_prefix
        self.enable_imagenet_preprocessing = enable_imagenet_preprocessing
        self.tokenizer_only = tokenizer_only
        # num_bridge_tokens is purely a transport channel here. The authoritative
        # value lives in the model config (``bridge_cfg.num_bridge_tokens``);
        # training scripts read it from there and pass it in. ``None`` is only
        # valid when ``use_bridge=False`` (the transform layer enforces this).
        self.num_bridge_tokens = num_bridge_tokens

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.video_delta_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    @abstractmethod
    def transform(self) -> ModalityTransform:
        pass


def import_external_data_config(
    data_config_str: str,
    eagle_path: str = DEFAULT_EAGLE_PATH,
    use_bridge: bool = False,
    ignore_lang_prefix: bool = False,
    enable_imagenet_preprocessing: bool = False,
    num_bridge_tokens: int = None,
    tokenizer_only: bool = False,
) -> Optional[BaseDataConfig]:
    """
    Import and instantiate an external data configuration class.

    Format: "module_path:ClassName" (e.g., "my_configs:RobotConfig")
    Supports nested modules like "package.submodule:ClassName"
    """
    if ":" not in data_config_str:
        return None

    import importlib
    import os
    import sys
    from pathlib import Path

    # Add current working directory to Python path
    current_dir = str(Path(os.getcwd()).absolute())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    try:
        module_path, class_name = data_config_str.split(":", 1)
        if not module_path or not class_name:
            raise ValueError(f"Invalid format: '{data_config_str}'. Use 'module:ClassName'")

        print(f"Loading external config: {module_path}.{class_name}")

        module = importlib.import_module(module_path)
        if not hasattr(module, class_name):
            available = [
                n
                for n in dir(module)
                if not n.startswith("_") and isinstance(getattr(module, n), type)
            ]
            raise AttributeError(
                f"Class '{class_name}' not found in '{module_path}'. Available: {available}"
            )

        # assert if the class has 'transform' and 'modality_config' methods
        if not hasattr(getattr(module, class_name), "transform"):
            raise AttributeError(f"Class '{class_name}' does not have a 'transform' method")
        if not hasattr(getattr(module, class_name), "modality_config"):
            raise AttributeError(f"Class '{class_name}' does not have a 'modality_config' method")

        return getattr(module, class_name)(
            eagle_path=eagle_path,
            use_bridge=use_bridge,
            ignore_lang_prefix=ignore_lang_prefix,
            enable_imagenet_preprocessing=enable_imagenet_preprocessing,
            num_bridge_tokens=num_bridge_tokens,
            tokenizer_only=tokenizer_only,
        )

    except (ModuleNotFoundError, AttributeError, ValueError) as e:
        print(f"Config loading failed: {e}")
        print("Example: my_configs:MyConfig, package.submodule:ClassName")
        raise


def load_data_config(
    data_config_str: str,
    eagle_path: str = DEFAULT_EAGLE_PATH,
    use_bridge: bool = False,
    ignore_lang_prefix: bool = False,
    enable_imagenet_preprocessing: bool = False,
    num_bridge_tokens: int = None,
    tokenizer_only: bool = False,
) -> BaseDataConfig:
    """
    Get a data config class from a string.
    >>> load_data_config("so100")
    >>> get_data_config("dir.subdir.my_configs:RobotConfig")

    Args:
        num_bridge_tokens: Number of bridge tokens. Read from
            ``model_config.bridge_cfg['num_bridge_tokens']`` by the training
            script and forwarded here. Required (non-None) when
            ``use_bridge=True``; otherwise the transform layer raises.
        tokenizer_only: If True, the data transform skips eagle backbone-side
            processing (VLM tokenisation, bridge-token append, goal-image eagle
            preprocessing). Use this for tokenizer-only training/evaluation.
    """
    if data_config_str in DATA_CONFIG_MAP:
        return DATA_CONFIG_MAP[data_config_str](
            eagle_path=eagle_path,
            use_bridge=use_bridge,
            ignore_lang_prefix=ignore_lang_prefix,
            enable_imagenet_preprocessing=enable_imagenet_preprocessing,
            num_bridge_tokens=num_bridge_tokens,
            tokenizer_only=tokenizer_only,
        )
    data_config_cls = import_external_data_config(
        data_config_str,
        eagle_path=eagle_path,
        use_bridge=use_bridge,
        ignore_lang_prefix=ignore_lang_prefix,
        enable_imagenet_preprocessing=enable_imagenet_preprocessing,
        num_bridge_tokens=num_bridge_tokens,
        tokenizer_only=tokenizer_only,
    )
    if data_config_cls is not None:
        return data_config_cls
    # Yellow warning color
    yellow = "\033[93m"
    reset = "\033[0m"
    raise ValueError(
        f"{yellow}Invalid data_config '{data_config_str}'. "
        f"Available options: {list(DATA_CONFIG_MAP.keys())}, "
        f"or use 'module:ClassName' for external configs{reset}"
    )


class FourierGr1ArmsWaistGausNormCropCamEgoJointsOnlyDataConfig(BaseDataConfig):
    """GR1 joints-only (5 joint groups). SinCos on state, mean_std on action."""

    video_keys = ["video.ego_view"]
    state_keys = [
        "state.right_arm",
        "state.right_hand",
        "state.left_arm",
        "state.left_hand",
        "state.waist",
    ]
    action_keys = [
        "action.right_arm",
        "action.right_hand",
        "action.left_arm",
        "action.left_hand",
        "action.waist",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0]
    video_delta_indices = [0]
    action_indices = list(range(16))

    # After SinCos, state dims change; skip mean_std there. Actions use mean_std as usual.
    state_normalization_modes = {}
    action_normalization_modes = {k: "mean_std" for k in action_keys}

    def transform(self) -> ModalityTransform:
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoOffsetCrop(
                apply_to=self.video_keys,
                top=int(256 * 0.17),
                left=0,
                height=int(256 * 0.66),
                width=256,
            ),
            VideoCrop(apply_to=self.video_keys, scale=0.95, height=168, width=256),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransformWithGoalImage(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=128,
                max_action_dim=128,
                eagle_path=self.eagle_path,
                use_bridge=self.use_bridge,
                ignore_lang_prefix=self.ignore_lang_prefix,
                enable_imagenet_preprocessing=True,
                vision_model_type="dinov2",
                num_bridge_tokens=self.num_bridge_tokens,
                tokenizer_only=self.tokenizer_only,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


DATA_CONFIG_MAP = {"fourier_gr1_arms_waist_gausNorm_crop_cam_ego_joints_only": FourierGr1ArmsWaistGausNormCropCamEgoJointsOnlyDataConfig}
