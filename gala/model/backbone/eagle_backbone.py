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
import os

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature


DEFAULT_EAGLE_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"


def qwen_vl_visual_module(eagle_model: nn.Module) -> nn.Module:
    """Vision tower for Qwen2.5-VL loaded via ``AutoModel`` (`Qwen2_5_VLModel` or wrapped)."""
    if getattr(eagle_model, "visual", None) is not None:
        return eagle_model.visual
    inner = getattr(eagle_model, "model", None)
    if inner is not None and getattr(inner, "visual", None) is not None:
        return inner.visual
    raise ValueError(
        "EagleBackboneGALA is Qwen2.5-VL only: expected eagle_model.visual "
        "or eagle_model.model.visual."
    )


def _sample_visual_token_counts(
    visual_mod: nn.Module,
    image_grid_thw: torch.Tensor | None,
    sample_image_counts: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor | None:
    if image_grid_thw is None or sample_image_counts is None:
        return None
    merge_size = int(
        getattr(
            visual_mod,
            "spatial_merge_size",
            getattr(getattr(visual_mod, "config", None), "spatial_merge_size", 2),
        )
    )
    per_image_counts = image_grid_thw.to(device=device, dtype=torch.long).prod(dim=1) // (merge_size**2)
    sample_counts = []
    offset = 0
    for count in sample_image_counts.to("cpu", dtype=torch.long).tolist():
        next_offset = offset + count
        sample_counts.append(per_image_counts[offset:next_offset].sum())
        offset = next_offset
    return torch.stack(sample_counts).to(device=device)


class EagleBackboneGALA(nn.Module):

    def __init__(
        self,
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        eagle_path: str | None = None,
        project_to_dim: int = 1536,
        tune_bridge_embedding: bool = False,
        tokenizer_len: bool = None,
        tune_all_llm_embedding: bool = False,
        num_bridge_tokens: int = None,
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"

        attn_implementation = os.environ.get("ATTN_IMPLEMENTATION")
        if attn_implementation not in {None, "eager", "sdpa", "flash_attention_2"}:
            raise ValueError(
                "ATTN_IMPLEMENTATION must be eager, sdpa, or flash_attention_2, "
                f"got {attn_implementation!r}"
            )
        model_kwargs = {"trust_remote_code": True}
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        self.eagle_model = AutoModel.from_pretrained(eagle_path, **model_kwargs)
        print(
            "Qwen attention backend: "
            f"requested={attn_implementation or 'auto'}, "
            f"resolved={self.eagle_model.config._attn_implementation}"
        )

        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

        print(f"Selected LLM Layer: {select_layer}")
        if hasattr(self.eagle_model.language_model, "model"):
            while len(self.eagle_model.language_model.model.layers) > select_layer:
                self.eagle_model.language_model.model.layers.pop(-1)
        else:
            while len(self.eagle_model.language_model.layers) > select_layer:
                self.eagle_model.language_model.layers.pop(-1)

        self.select_layer = select_layer
        self.num_bridge_tokens = num_bridge_tokens
        self.set_trainable_parameters(
            tune_llm, tune_visual,
            tune_bridge_embedding, tokenizer_len,
            tune_all_llm_embedding,
        )

    def set_trainable_parameters(self,
            tune_llm: bool,
            tune_visual: bool,
            tune_bridge_embedding: bool,
            tokenizer_len: int = None,
            tune_all_llm_embedding: bool = False,
        ):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        if tune_visual:
            # Vision-tower finetuning is no longer supported. The previous code
            # path additionally required a DDP-unused-param workaround that is
            # now removed; re-enable it only after restoring proper DDP support.
            raise NotImplementedError(
                "EagleBackboneGALA no longer supports tune_visual=True. "
                "Set tune_visual=False (vision tower is always frozen)."
            )
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.eagle_model.language_model.requires_grad_(False)
        qwen_vl_visual_module(self.eagle_model).requires_grad_(False)
        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")

        # Ensure a slot for the gradient-hook handle exists across re-entries.
        if not hasattr(self, "_embed_tokens_hook_handle"):
            self._embed_tokens_hook_handle = None

        if tune_bridge_embedding and (not tune_all_llm_embedding):
            if hasattr(self.eagle_model.language_model, "model"):
                embed_tokens = self.eagle_model.language_model.model.embed_tokens
            else:
                embed_tokens = self.eagle_model.language_model.embed_tokens

            embed_tokens.weight.requires_grad = True

            # Remove any previous hook before registering a new one.
            if self._embed_tokens_hook_handle is not None:
                self._embed_tokens_hook_handle.remove()
                self._embed_tokens_hook_handle = None

            if self.num_bridge_tokens is None:
                raise ValueError(
                    "EagleBackboneGALA.tune_bridge_embedding=True requires "
                    "num_bridge_tokens to be set. Pass it from the model's "
                    "bridge_cfg.num_bridge_tokens at construction time."
                )
            bridge_token_ids = torch.arange(
                tokenizer_len - self.num_bridge_tokens, tokenizer_len,
                device=embed_tokens.weight.device,
            )
            print(f"start_bridge_token_id: {bridge_token_ids[0]}, end_bridge_token_id: {bridge_token_ids[-1]}")

            # [vocab_size, 1] mask broadcasts against embed_tokens.weight.grad.
            self._embed_tokens_hook_mask = torch.zeros(embed_tokens.weight.shape[0], device=embed_tokens.weight.device)
            self._embed_tokens_hook_mask[bridge_token_ids] = 1.0
            self._embed_tokens_hook_mask = self._embed_tokens_hook_mask.view(-1, 1)

            def grad_hook(grad):
                if self._embed_tokens_hook_mask.device != grad.device:
                    self._embed_tokens_hook_mask = self._embed_tokens_hook_mask.to(grad.device)
                return grad * self._embed_tokens_hook_mask

            self._embed_tokens_hook_handle = embed_tokens.weight.register_hook(grad_hook)

        else:
            # Drop a previously registered hook if any.
            if self._embed_tokens_hook_handle is not None:
                self._embed_tokens_hook_handle.remove()
                self._embed_tokens_hook_handle = None

        print(f"Tune backbone bridge embedding: {tune_bridge_embedding}")
        print(f"Tune all llm token embeddings: {tune_all_llm_embedding}")

        if not tune_llm and not tune_visual:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.eagle_model.language_model and not self.tune_llm:
                self.eagle_model.language_model.eval()
            if not self.tune_visual:
                qwen_vl_visual_module(self.eagle_model).eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward_eagle(
        self, vl_input: BatchFeature
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix)
        }
        if "image_sizes" in eagle_input:
            del eagle_input["image_sizes"]
        sample_image_counts = eagle_input.pop("sample_image_counts", None)

        cached_image_embeds = None
        image_token_counts = None
        if eagle_input.get("pixel_values") is not None:
            visual_mod = qwen_vl_visual_module(self.eagle_model)
            grid_thw = eagle_input.get("image_grid_thw", eagle_input.get("video_grid_thw"))
            with torch.no_grad():
                cached_image_embeds = visual_mod(eagle_input["pixel_values"], grid_thw=grid_thw).detach()
            image_token_counts = _sample_visual_token_counts(
                visual_mod, eagle_input.get("image_grid_thw"), sample_image_counts, cached_image_embeds.device
            )
            if image_token_counts is None:
                image_token_index = getattr(self.eagle_model.config, "image_token_index", 151669)
                image_token_counts = (eagle_input["input_ids"] == image_token_index).sum(dim=1).to(torch.long)

        eagle_output = self.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)

        # print(f"(eagle_output.hidden_states): {len(eagle_output.hidden_states)}")
        eagle_features = eagle_output.hidden_states[self.select_layer]

        eagle_features = self.eagle_linear(eagle_features)
        return eagle_features, eagle_input["attention_mask"], cached_image_embeds, image_token_counts

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        eagle_embeds, eagle_mask, cached_image_embeds, image_token_counts = self.forward_eagle(vl_input)

        out = {
            "backbone_features": eagle_embeds,
            "backbone_attention_mask": eagle_mask,
        }
        if cached_image_embeds is not None:
            out["cached_image_embeds"] = cached_image_embeds
        if image_token_counts is not None:
            out["cached_image_token_counts"] = image_token_counts
        return BatchFeature(data=out)  # [B, T2, hidden_size]
