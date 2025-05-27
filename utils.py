# utils.py
# 通用工具函数，供业务逻辑调用
import re

def safe_float(val):
    """
    安全转换为float，异常时返回0.0。
    Safe conversion to float, returns 0.0 on error.
    """
    try:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            val = val.replace('N/A', '').replace('-', '').strip()
            return float(val) if val else 0.0
    except Exception:
        return 0.0
    return 0.0

def parse_battery_size(size_str):
    """
    解析尺寸字符串，兼容x/X/×/*分隔，返回三元组或None。
    Parse battery size string, support x/X/×/* as separator, return tuple or None.
    """
    s = str(size_str or '').replace("×", "x").replace("X", "x").replace("*", "x")
    s = re.sub(r"(\d)\s*[xX×*]\s*(\d)", r"\1x\2", s)
    if s and "x" in s:
        try:
            t = tuple(float(x) for x in s.split("x"))
            if len(t) == 3:
                return t
        except Exception:
            pass
    return None

def size_within_limit(bat_size, limit_size):
    """
    判断电池尺寸是否在限制范围内（均为三元组）。
    Check if battery size is within the limit (both are 3-tuple).
    """
    try:
        bat_tuple = tuple(float(x) for x in str(bat_size).replace("×", "*").replace("x", "*").replace("X", "*").split("*"))
        if len(bat_tuple) != 3 or not limit_size:
            return True
        return all(b <= l for b, l in zip(sorted(bat_tuple), sorted(limit_size)))
    except Exception:
        return True
