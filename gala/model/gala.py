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

from __future__ import annotations


import torch
import torch.nn as nn
from transformers.feature_extraction_utils import BatchFeature

from gala.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gala.model.base import GALABase


class GALAModel(GALABase):
    """GALA VLA with DexLAM embedding targets for bridge supervision."""

    _EMBODIMENT_ID_TO_TAG = {v: k for k, v in EMBODIMENT_TAG_MAPPING.items()}


    def _init_bridge_modules_tokenizer_mode(self):
        self._needs_obs_embeds = self.bridge_type in ["vision_lang_obs", "vision_lang_obs_e2e"]
        self._needs_vision_lang_features = self.bridge_type in ["vision_lang", "vision_lang_obs", "vision_lang_obs_e2e"]
        if self._needs_obs_embeds:
            from gala.model.backbone.eagle_backbone import qwen_vl_visual_module
            qwen_vl_visual_module(self.backbone.eagle_model)
        self.num_bridge_tokens = self.config.bridge_cfg["num_bridge_tokens"]

    def _pack_cached_obs_embeds(
        self, obs_embeds: torch.Tensor, image_token_counts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_size = obs_embeds.shape[-1]
        token_counts = image_token_counts.to(device=obs_embeds.device, dtype=torch.long)
        if int(token_counts.sum().item()) != obs_embeds.shape[0]:
            raise RuntimeError(
                "Cached image embed/token mismatch: "
                f"sum(image tokens)={int(token_counts.sum().item())}, obs_embeds={obs_embeds.shape[0]}"
            )
        max_tokens = int(token_counts.max().item())
        packed = obs_embeds.new_zeros((token_counts.shape[0], max_tokens, hidden_size))
        mask = torch.zeros((token_counts.shape[0], max_tokens), dtype=torch.bool, device=obs_embeds.device)
        offset = 0
        for i, count in enumerate(token_counts.tolist()):
            if count > 0:
                packed[i, :count] = obs_embeds[offset : offset + count]
                mask[i, :count] = True
                offset += count
        return packed, mask


    def forward(
        self,
        inputs: dict,
        action_mode: bool = True,
        compute_bridge_loss_override: bool | None = None,
    ):
        """Predict action chunks using the checkpoint's bridge context."""
        if not action_mode or compute_bridge_loss_override:
            raise NotImplementedError("This release supports inference only")
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        cached_image_embeds = backbone_outputs.get("cached_image_embeds")
        cached_image_token_counts = backbone_outputs.get("cached_image_token_counts")
        if "cached_image_embeds" in backbone_outputs:
            del backbone_outputs["cached_image_embeds"]
        if "cached_image_token_counts" in backbone_outputs:
            del backbone_outputs["cached_image_token_counts"]
        batch_size = inputs["state"].shape[0]

        output_dict = {}
        if self.use_bridge:
            nt = self.num_bridge_tokens
            backbone_outputs["backbone_features"] = torch.nan_to_num(
                backbone_outputs["backbone_features"], nan=0.0, posinf=1e4, neginf=-1e4
            )
            bridge_segment_features = backbone_outputs["backbone_features"][:, -nt:]

            include_bridge_in_action_context = self.bridge_type == "vision_lang_obs_e2e"
            if include_bridge_in_action_context:
                vl_features = backbone_outputs["backbone_features"]
                vl_attention_mask = backbone_outputs["backbone_attention_mask"]
            else:
                vl_features = backbone_outputs["backbone_features"][:, :-nt]
                vl_attention_mask = backbone_outputs["backbone_attention_mask"][:, :-nt]


            obs_embeds = None
            if self._needs_obs_embeds:
                if cached_image_embeds is None:
                    raise RuntimeError(
                        "Obs image embeds missing: expected backbone outputs['cached_image_embeds'] "
                        "from the Qwen visual forward hook."
                    )
                if cached_image_token_counts is None:
                    raise RuntimeError(
                        "Obs image token counts missing: expected backbone outputs['cached_image_token_counts']."
                    )
                obs_embeds, obs_attention_mask = self._pack_cached_obs_embeds(
                    cached_image_embeds, cached_image_token_counts
                )
                hidden_size = obs_embeds.shape[-1]
                if self.action_only_one_obs:
                    obs_embeds = obs_embeds.reshape(batch_size, -1, self.num_bridge_tokens, hidden_size)
                    obs_embeds = obs_embeds[:, -1]
                    obs_attention_mask = obs_attention_mask.reshape(batch_size, -1, self.num_bridge_tokens).all(dim=-1)
                obs_embeds = torch.nan_to_num(obs_embeds, nan=0.0, posinf=1e4, neginf=-1e4)

            if self.detach_vl_for_action:
                vl_features = vl_features.detach()

            if self.bridge_type == "vision_lang":
                action_context_features = vl_features
                action_context_mask = vl_attention_mask
            elif self.bridge_type in ("vision_lang_obs", "vision_lang_obs_e2e"):
                assert obs_embeds is not None, f"bridge_type={self.bridge_type} requires obs_embeds"
                if self.use_image_type_embedding:
                    obs_embeds = obs_embeds + self.image_type_embedding.weight[1]
                obs_attention_mask = obs_attention_mask.to(device=obs_embeds.device, dtype=vl_attention_mask.dtype)
                action_context_features = torch.cat([vl_features, obs_embeds], dim=1)
                action_context_mask = torch.cat([vl_attention_mask, obs_attention_mask], dim=1)
            else:
                raise NotImplementedError(f"Invalid bridge_type: {self.bridge_type}")

            bridge_outputs = BatchFeature(
                data={
                    "backbone_features": action_context_features,
                    "backbone_attention_mask": action_context_mask,
                }
            )
        else:
            bridge_outputs = backbone_outputs

        action_head_outputs = self.action_head.get_action(bridge_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        action_head_outputs.update(output_dict)
        return action_head_outputs
