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
    def from_pretrained(cls, pretrained_model_name_or_path: str, resume_pretrained_option: str="all", **kwargs):
        if resume_pretrained_option != "all":
            raise ValueError("Inference requires loading all checkpoint weights")
        from gala.checkpoint import load_config
        model_config = load_config(pretrained_model_name_or_path)
        bridge_cfg_overrides = kwargs.pop("bridge_cfg_overrides", None)
        if bridge_cfg_overrides is not None:
            model_config.bridge_cfg = bridge_cfg_overrides

        tune_visual = kwargs.pop("tune_visual", model_config.backbone_cfg['tune_visual'])
        tune_llm = kwargs.pop("tune_llm", model_config.backbone_cfg['tune_llm'])
        tune_bridge_embedding = kwargs.pop("tune_bridge_embedding", model_config.backbone_cfg['tune_bridge_embedding'])
        tokenizer_len = kwargs.pop("tokenizer_len", None)
        tune_projector = kwargs.pop("tune_projector", model_config.action_head_cfg['tune_projector'])
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", model_config.action_head_cfg['tune_diffusion_model'])

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune backbone bridge embedding: {tune_bridge_embedding}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        try:
            bridge_type = kwargs.pop("bridge_type", model_config.bridge_cfg.get('bridge_type', "vision_lang_obs"))
            compute_bridge_loss = kwargs.pop("compute_bridge_loss", model_config.bridge_cfg.get('compute_bridge_loss', False))
            bridge_loss_type = kwargs.pop("bridge_loss_type", model_config.bridge_cfg.get('bridge_loss_type', 'ce'))
            tune_all_llm_embedding = kwargs.pop("tune_all_llm_embedding", model_config.bridge_cfg.get('tune_all_llm_embedding', False))
            use_image_type_embedding = kwargs.pop("use_image_type_embedding", model_config.bridge_cfg.get('use_image_type_embedding', False))
            omit_image_type_embedding_for_goal = kwargs.pop("omit_image_type_embedding_for_goal", model_config.bridge_cfg.get('omit_image_type_embedding_for_goal', False))
            action_only_one_obs = kwargs.pop("action_only_one_obs", model_config.bridge_cfg.get('action_only_one_obs', False))
            noise_tau = kwargs.pop("noise_tau", model_config.bridge_cfg.get('noise_tau', 0))
            reweight_noise = kwargs.pop("reweight_noise", model_config.bridge_cfg.get('reweight_noise', False))
            groot_tokenizer_path = kwargs.pop("groot_tokenizer_path", model_config.bridge_cfg.get('groot_tokenizer_path', None))
            action_loss_weight = kwargs.pop("action_loss_weight", model_config.bridge_cfg.get('action_loss_weight', 1.0))
            bridge_loss_weight = kwargs.pop("bridge_loss_weight", model_config.bridge_cfg.get('bridge_loss_weight', 0.1))
            unified_embodiment_id = kwargs.pop("unified_embodiment_id", model_config.bridge_cfg.get('unified_embodiment_id', None))
            detach_vl_for_action = kwargs.pop("detach_vl_for_action", model_config.bridge_cfg.get('detach_vl_for_action', False))
            # Optional override of the DINOv2 weight path inside the loaded GR00T tokenizer (deployment-time).
            dinov2_path_override = kwargs.pop("dinov2_path_override", None)
        except Exception as e:
            print(kwargs)
            raise e
        print(f"Bridge type: {bridge_type}")
        print(f"Compute bridge loss: {compute_bridge_loss}")
        print(f"Bridge loss type: {bridge_loss_type}")
        print(f"Tune all llm token embeddings: {tune_all_llm_embedding}")
        print(f"Use image type embeddings: {use_image_type_embedding}")
        print(f"Omit image type embeddings for goal images: {omit_image_type_embedding_for_goal}")
        print(f"Action head using only one obs: {action_only_one_obs}")
        print(f"Noise Tau: {noise_tau}")
        print(f"Reweight Noise: {reweight_noise}")
        print(f"GR00T tokenizer path: {groot_tokenizer_path}")
        print(f"Action loss weight: {action_loss_weight}")
        print(f"Bridge loss weight: {bridge_loss_weight}")
        print(f"Unified embodiment ID: {unified_embodiment_id}")
        print(f"Detach VL for action: {detach_vl_for_action}")
        if dinov2_path_override is not None:
            print(f"DINOv2 path override (for tokenizer): {dinov2_path_override}")

        select_layer = kwargs.pop("select_layer", model_config.backbone_cfg.get('select_layer', None))

        tune_bridge_visual = kwargs.pop("tune_bridge_visual", model_config.bridge_cfg['tune_bridge_visual'])
        tune_image_type_embedding = kwargs.pop("tune_image_type_embedding", model_config.bridge_cfg.get('tune_image_type_embedding', True))
        print(f"Tune bridge vision model: {tune_bridge_visual}")
        print(f"Tune image type embeddings: {tune_image_type_embedding}")

        use_vl_mask = kwargs.pop("use_vl_mask", model_config.action_head_cfg.get('use_vl_mask', True))
        print(f"Use VL mask: {use_vl_mask}")

        # use_correct_attn_mask: convert HF-style attention mask to SDPA-style inside action head
        use_correct_attn_mask = kwargs.pop(
            "use_correct_attn_mask", model_config.action_head_cfg.get('use_correct_attn_mask', True)
        )
        print(f"Use correct attn mask format: {use_correct_attn_mask}")

        # snapshot_download returns local cache path under ~/.cache/huggingface/hub/;
        # falls back to treating the argument as a local path if it is not a valid hub repo id.
        try:
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        customized_kwargs = {
            "tokenizer_len": tokenizer_len,
            "bridge_type": bridge_type,
            "compute_bridge_loss": compute_bridge_loss,
            "select_layer": select_layer,
            "bridge_loss_type": bridge_loss_type,
            "use_image_type_embedding": use_image_type_embedding,
            "use_vl_mask": use_vl_mask,
            "use_correct_attn_mask": use_correct_attn_mask,
            "action_only_one_obs": action_only_one_obs,
            "noise_tau": noise_tau,
            "omit_image_type_embedding_for_goal": omit_image_type_embedding_for_goal,
            "reweight_noise": reweight_noise,
            "groot_tokenizer_path": groot_tokenizer_path,
            "action_loss_weight": action_loss_weight,
            "bridge_loss_weight": bridge_loss_weight,
            "unified_embodiment_id": unified_embodiment_id,
            "detach_vl_for_action": detach_vl_for_action,
        }

        try:
            import os

            pretrained_model = super().from_pretrained(
                local_model_path, local_model_path=local_model_path,
                config=model_config,
                **customized_kwargs,
                **kwargs
            )

        except Exception:
            raise  # Never silently evaluate randomly initialized weights.

        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm,
            tune_bridge_embedding=tune_bridge_embedding,
            tokenizer_len=tokenizer_len,
            tune_all_llm_embedding=tune_all_llm_embedding,
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )

        if pretrained_model.use_bridge:
            pretrained_model.set_trainable_parameters(
                tune_bridge_visual=tune_bridge_visual,
                tune_image_type_embedding=tune_image_type_embedding
            )

        pretrained_model.config.backbone_cfg['tune_visual'] = tune_visual
        pretrained_model.config.backbone_cfg['tune_llm'] = tune_llm
        pretrained_model.config.backbone_cfg['tune_bridge_embedding'] = tune_bridge_embedding
        pretrained_model.config.action_head_cfg['tune_projector'] = tune_projector
        pretrained_model.config.action_head_cfg['tune_diffusion_model'] = tune_diffusion_model
        pretrained_model.config.action_head_cfg['use_vl_mask'] = use_vl_mask
        pretrained_model.config.action_head_cfg['use_correct_attn_mask'] = use_correct_attn_mask
        pretrained_model.config.bridge_cfg['tune_bridge_visual'] = tune_bridge_visual
        pretrained_model.config.bridge_cfg['tokenizer_len'] = tokenizer_len
        pretrained_model.config.bridge_cfg['bridge_type'] = bridge_type
        pretrained_model.config.bridge_cfg['compute_bridge_loss'] = compute_bridge_loss
        pretrained_model.config.bridge_cfg['bridge_loss_type'] = bridge_loss_type
        pretrained_model.config.backbone_cfg['tune_all_llm_embedding'] = tune_all_llm_embedding
        pretrained_model.config.bridge_cfg['use_image_type_embedding'] = use_image_type_embedding
        pretrained_model.config.bridge_cfg['action_only_one_obs'] = action_only_one_obs
        pretrained_model.config.bridge_cfg['noise_tau'] = noise_tau
        pretrained_model.config.bridge_cfg['reweight_noise'] = reweight_noise
        pretrained_model.config.bridge_cfg['omit_image_type_embedding_for_goal'] = omit_image_type_embedding_for_goal
        pretrained_model.config.bridge_cfg['tune_image_type_embedding'] = tune_image_type_embedding
        pretrained_model.config.bridge_cfg['groot_tokenizer_path'] = groot_tokenizer_path
        pretrained_model.config.bridge_cfg['unified_embodiment_id'] = unified_embodiment_id
        pretrained_model.config.bridge_cfg['bridge_loss_weight'] = bridge_loss_weight

        return pretrained_model
