import re
from typing import Optional

_PAT_P_S = re.compile(r'[pP]\s*(\d+)\s*[sS]')
_PAT_S = re.compile(r'(\d+)\s*[sS]')

def extract_series_count(s: str) -> Optional[int]:
    """
    从字符串中提取 'P...S' 中的数字（如 "1P96S" -> 96）。
    返回 int 如果成功，失败返回 None。
    """
    if not s:
        return None
    m = _PAT_P_S.search(s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    m2 = _PAT_S.findall(s)
    if m2:
        try:
            return int(m2[-1])
        except Exception:
            return None
    return None

