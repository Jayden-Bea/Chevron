from __future__ import annotations

import re


def parse_clock_text(text: str) -> float | None:
    m = re.search(r"(\d{1,2})\s*[:.]\s*(\d{2})", text)
    if not m:
        return None
    mm = int(m.group(1))
    ss = int(m.group(2))
    if ss >= 60:
        return None
    return float(mm * 60 + ss)
