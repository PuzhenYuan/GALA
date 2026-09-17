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


from dataclasses import dataclass


from typing import Any


@dataclass(frozen=True)
class DexLAMTokenLayout:
    image_tokens: int
    pointcloud_tokens: int
    pointcloud_layout: str
    encoder_variant: str

    @property
    def total_tokens(self) -> int:
        pointcloud_branches = 1 if self.pointcloud_layout == "shared" else 2
        return self.image_tokens + pointcloud_branches * self.pointcloud_tokens


def resolve_dexlam_token_layout(bridge_cfg: dict[str, Any]) -> DexLAMTokenLayout:
    tokenizer_cfg = bridge_cfg.get("tokenizer_cfg", {})
    total_tokens = int(bridge_cfg["num_bridge_tokens"])
    image_tokens = int(tokenizer_cfg.get("dexlam_image_tokens", 4))
    pointcloud_layout = str(tokenizer_cfg.get("dexlam_pointcloud_layout", "per_hand"))
    encoder_variant = str(tokenizer_cfg.get("dexlam_encoder_variant", "v2"))

    if pointcloud_layout not in {"shared", "per_hand"}:
        raise ValueError(
            "dexlam_pointcloud_layout must be 'shared' or 'per_hand', "
            f"got {pointcloud_layout!r}"
        )
    if encoder_variant not in {"v2", "v6", "v6_image_only", "metis", "native", "opfa"}:
        raise ValueError(
            "dexlam_encoder_variant must be 'v2', 'v6', 'v6_image_only', "
            "'metis', 'native', or 'opfa', "
            f"got {encoder_variant!r}"
        )

    branches = 1 if pointcloud_layout == "shared" else 2
    default_pointcloud_tokens, remainder = divmod(total_tokens - image_tokens, branches)
    if remainder:
        raise ValueError(
            f"num_bridge_tokens={total_tokens} cannot be split into image_tokens={image_tokens} "
            f"and {branches} pointcloud branch(es)"
        )
    pointcloud_tokens = int(
        tokenizer_cfg.get("dexlam_pointcloud_tokens", default_pointcloud_tokens)
    )
    layout = DexLAMTokenLayout(
        image_tokens=image_tokens,
        pointcloud_tokens=pointcloud_tokens,
        pointcloud_layout=pointcloud_layout,
        encoder_variant=encoder_variant,
    )
    if layout.image_tokens < 0 or layout.pointcloud_tokens < 0:
        raise ValueError(f"DexLAM token counts must be non-negative, got {layout}")
    if layout.encoder_variant == "v6_image_only":
        if layout.image_tokens <= 0 or layout.pointcloud_tokens != 0:
            raise ValueError(
                "v6_image_only requires positive image tokens and zero pointcloud tokens, "
                f"got {layout}"
            )
    elif layout.pointcloud_tokens <= 0:
        raise ValueError(f"DexLAM pointcloud token count must be positive, got {layout}")
    if layout.total_tokens != total_tokens:
        raise ValueError(
            f"DexLAM token layout totals {layout.total_tokens}, but num_bridge_tokens={total_tokens}: "
            f"{layout}"
        )
    return layout
