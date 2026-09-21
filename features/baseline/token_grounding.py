"""TokenGrounding baseline: concatenated ADS and CGC layer features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from features.ads import compute_ads
from features.cgc import compute_cgc


@dataclass(frozen=True)
class TokenGroundingFeatures:
    vector: np.ndarray
    ads_per_layer: np.ndarray
    cgc_per_layer: np.ndarray
    ads_score: float
    cgc_score: float

    def as_payload(self) -> dict[str, Any]:
        return {
            "vector": self.vector.astype(np.float32, copy=False),
            "ads_per_layer": self.ads_per_layer.astype(np.float32, copy=False),
            "cgc_per_layer": self.cgc_per_layer.astype(np.float32, copy=False),
            "ads_score": float(self.ads_score),
            "cgc_score": float(self.cgc_score),
            "definition": "concat_ads_per_layer_cgc_per_layer",
        }


def compute_token_grounding_features(
    *,
    text_to_patch_attn: torch.Tensor,
    token_hidden_states: torch.Tensor,
    patch_hidden_states: torch.Tensor,
    visual_grid: tuple[int, int] | None,
    config: Mapping[str, Any] | None = None,
) -> TokenGroundingFeatures:
    """Return the one TokenGrounding vector used by the independent detector."""

    cfg = dict(config or {})
    ads_cfg = dict(cfg.get("ads") or {})
    cgc_cfg = dict(cfg.get("cgc") or {})
    ads_score, ads_layers = compute_ads(
        text_to_patch_attn,
        top_patch_pct=float(ads_cfg.get("top_patch_pct", 0.10)),
        connectivity=int(ads_cfg.get("connectivity", 8)),
        min_blob_area=int(ads_cfg.get("min_blob_area", 3)),
        top_k_layers=int(ads_cfg.get("top_k_layers", 10)),
        per_head_min=bool(ads_cfg.get("per_head_min", False)),
        top_k_heads=int(ads_cfg.get("top_k_heads", 0)),
        scale_blob_area=bool(ads_cfg.get("scale_blob_area", True)),
        grid_shape=visual_grid,
    )
    cgc_score, cgc_layers = compute_cgc(
        token_hidden_states,
        patch_hidden_states,
        top_k_patches=int(cgc_cfg.get("top_k_patches", 5)),
        top_k_pct=float(cgc_cfg.get("top_k_pct", 0.05)),
        text_to_patch_attn=(
            text_to_patch_attn
            if bool(cgc_cfg.get("use_attn_weighting", False))
            else None
        ),
        mid_layer_pct=tuple(cgc_cfg.get("mid_layer_pct", (0.25, 0.75))),
    )
    ads_array = ads_layers.detach().cpu().numpy().astype(np.float32, copy=False)
    cgc_array = cgc_layers.detach().cpu().numpy().astype(np.float32, copy=False)
    if ads_array.shape != cgc_array.shape:
        raise ValueError(
            "TokenGrounding ADS/CGC layer shapes differ: "
            f"{ads_array.shape} != {cgc_array.shape}"
        )
    vector = np.concatenate((ads_array, cgc_array)).astype(np.float32, copy=False)
    if not np.isfinite(vector).all():
        raise ValueError("TokenGrounding features contain non-finite values")
    return TokenGroundingFeatures(
        vector=vector,
        ads_per_layer=ads_array,
        cgc_per_layer=cgc_array,
        ads_score=float(ads_score),
        cgc_score=float(cgc_score),
    )
