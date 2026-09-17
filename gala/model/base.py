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

from dataclasses import dataclass, field


from typing import Tuple


import numpy as np


import torch


import tree


from huggingface_hub import snapshot_download


from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError


from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel


from transformers.feature_extraction_utils import BatchFeature


from .action_head.flow_matching_action_head import (
    FlowmatchingActionHeadGALA,
    FlowmatchingActionHeadGALAConfig,
)


from .backbone.eagle_backbone import EagleBackboneGALA, qwen_vl_visual_module


from torch import nn


BACKBONE_FEATURE_KEY = "backbone_features"
BACKBONE_SEQ_KEYS = frozenset({"backbone_features", "backbone_attention_mask"})
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


@dataclass
class GALAConfig(PretrainedConfig):
    model_type = "gala"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})

    bridge_cfg: dict = field(init=False, metadata={"help": "Bridge configuration."})

    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class GALABase(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GALAConfig
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GALAConfig,
        local_model_path: str,
        tokenizer_len: int=None,
        bridge_type: str="vision_lang_obs",
        compute_bridge_loss: bool=False,
        select_layer: int=None,
        bridge_loss_type: str="ce",
        use_image_type_embedding: bool=False,
        use_vl_mask: bool=False,
        use_correct_attn_mask: bool=False,  # Convert HF-style mask to SDPA-style
        action_only_one_obs: bool=False,

        noise_tau: float=0,
        omit_image_type_embedding_for_goal: bool=False,
        reweight_noise: bool=False,

        groot_tokenizer_path: str=None,
        action_loss_weight: float=1.0,  # Weight for action loss (0.0 to disable)
        bridge_loss_weight: float=0.1,  # Global scale on bridge_loss; total = (action_loss_weight*action_loss + bridge_loss_weight*bridge_loss) / 2
        unified_embodiment_id: int=None,  # If set, all samples use this embodiment ID
        detach_vl_for_action: bool=False,  # If True, detach VL features before passing to action head
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        if select_layer is not None:
            config.backbone_cfg['select_layer'] = select_layer

        config.action_head_cfg['use_vl_mask'] = use_vl_mask
        config.action_head_cfg['use_correct_attn_mask'] = use_correct_attn_mask

        super().__init__(config)
        self.local_model_path = local_model_path

        # tokenizer_len = Eagle backbone vocab size; for Qwen/Eagle this is fixed at 151729.
        if tokenizer_len is None:
            raise ValueError("tokenizer_len (Eagle vocab size) must be provided as parameter")

        print(f"[INFO] tokenizer_len (Eagle vocab size) for backbone: {tokenizer_len}")
        # num_bridge_tokens is the authoritative slice/registration size; the
        # backbone needs it to align its embedding-grad hook with the same
        # range of token ids that the data transform appended via get_bridge_str.
        if 'num_bridge_tokens' not in config.bridge_cfg:
            raise ValueError("config.bridge_cfg must contain 'num_bridge_tokens'")
        self.backbone = EagleBackboneGALA(
            **config.backbone_cfg,
            tokenizer_len=tokenizer_len,
            num_bridge_tokens=config.bridge_cfg['num_bridge_tokens'],
        )
        action_head_cfg = FlowmatchingActionHeadGALAConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHeadGALA(action_head_cfg)

        # When set, all samples share a single embodiment_id inside the action head,
        # tying per-embodiment parameters together. Required for cross-embodiment co-training.
        if unified_embodiment_id is not None:
            print(f"Unified embodiment ID: {unified_embodiment_id}")
            self.action_head.unified_embodiment_id = unified_embodiment_id
        self.unified_embodiment_id = unified_embodiment_id

        print(f"Use bridge: {self.config.bridge_cfg['use_bridge']}")
        self.use_bridge = self.config.bridge_cfg['use_bridge']
        self.groot_tokenizer_path = groot_tokenizer_path

        # bridge_type must be set before _init_bridge_modules_tokenizer_mode (used inside).
        self.bridge_type = bridge_type
        self.compute_bridge_loss = compute_bridge_loss
        self.bridge_loss_type = bridge_loss_type
        self.action_loss_weight = action_loss_weight
        self.bridge_loss_weight = bridge_loss_weight
        self.detach_vl_for_action = detach_vl_for_action
        if detach_vl_for_action:
            print("VL features will be detached before action head (action loss will not update VLM via VL).")

        if self.config.bridge_cfg['use_bridge']:
            self._init_bridge_modules_tokenizer_mode()

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

        # CE loss is the only supported bridge loss in tokenizer mode.
        if bridge_loss_type not in ["ce", "cross_entropy"]:
            print(f"WARNING: bridge_loss_type={bridge_loss_type} is ignored; CE loss is always used")

        self.use_image_type_embedding = use_image_type_embedding
        self.omit_image_type_embedding_for_goal = omit_image_type_embedding_for_goal
        if use_image_type_embedding:
            self.image_type_embedding = nn.Embedding(3, self.backbone.eagle_model.config.hidden_size)
            nn.init.normal_(self.image_type_embedding.weight, mean=0.0, std=0.02)

        self.action_only_one_obs = action_only_one_obs
        self.noise_tau = noise_tau
        self.reweight_noise = reweight_noise

    def validate_inputs(self, inputs):
        # NOTE: ideally enforced inside backbone/action_head themselves; kept here to avoid breaking
        # the existing public API.
        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)


    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        return self.forward(inputs=inputs, action_mode=True)


    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)
        target_dtype = getattr(torch, str(self.compute_dtype), None)
        if target_dtype is None:
            target_dtype = getattr(self.action_head, "dtype", torch.float32)

        def to_device_with_maybe_dtype(x):
            if not isinstance(x, torch.Tensor):
                return x
            # Cast to the action head's compute dtype only for floating tensors;
            # integer tensors (token ids, indices) keep their original dtype.
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=target_dtype)
            return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load a complete inference checkpoint; never initialize missing weights."""
        from gala.checkpoint import load_config, resolve_checkpoint
        path = resolve_checkpoint(pretrained_model_name_or_path)
        config = load_config(path)
        bridge = config.bridge_cfg
        # Older callers may still supply these inference-disabled options.
        if kwargs.pop("compute_bridge_loss", False):
            raise ValueError("Bridge supervision is not part of inference")
        kwargs.pop("dinov2_path_override", None)
        model, loading = super().from_pretrained(
            path, local_model_path=path, config=config,
            bridge_type=bridge.get("bridge_type", "vision_lang_obs"),
            use_image_type_embedding=bridge.get("use_image_type_embedding", False),
            action_only_one_obs=bridge.get("action_only_one_obs", False),
            unified_embodiment_id=bridge.get("unified_embodiment_id"),
            use_vl_mask=config.action_head_cfg.get("use_vl_mask", True),
            use_correct_attn_mask=config.action_head_cfg.get("use_correct_attn_mask", True),
            output_loading_info=True, **kwargs,
        )
        unexpected = [key for key in loading.get("unexpected_keys", [])
                      if not key.startswith("bridge_ce_predictors.")]
        if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs") or unexpected:
            raise RuntimeError(f"Incomplete or incompatible inference checkpoint: {loading}")
        model.requires_grad_(False)
        model.eval()
        return model
