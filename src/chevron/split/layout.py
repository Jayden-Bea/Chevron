from __future__ import annotations


def default_layout(width: int, height: int) -> dict[str, list[int]]:
    return {
        "top": [0, 0, width, height],
    }


def get_layout(cfg: dict, width: int, height: int) -> dict[str, list[int]]:
    return cfg.get("split", {}).get("crops") or default_layout(width, height)
