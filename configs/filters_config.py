# configs/filters_config.py
from typing import List, Tuple

PHONE_PROFILES: List[Tuple[str, float, float, float, float, float, float, float, float]] = [
    ("iphone_warm_vivid", 1.02, 0.98, +0.010, 1.02, 1.10, 1.08, 1.10, 0.10),
    ("samsung_cool_sharp", 0.98, 1.02, -0.012, 1.00, 1.12, 1.15, 1.20, 0.00),
    ("pixel_neutral", 1.00, 1.00, +0.000, 1.00, 1.05, 1.02, 1.05, 0.00),
    ("xiaomi_punchy", 1.01, 0.99, +0.015, 1.00, 1.15, 1.18, 1.15, 0.00),
    ("huawei_smooth", 1.00, 1.01, +0.005, 1.02, 1.03, 1.08, 0.95, 0.00),
    ("oneplus_crisp", 1.00, 1.00, -0.008, 1.00, 1.10, 1.07, 1.12, 0.00),
    ("sony_true", 1.00, 1.00, +0.000, 0.98, 1.02, 0.98, 1.00, 0.20),
    ("oppo_vivid", 1.02, 0.99, +0.012, 1.01, 1.08, 1.12, 1.08, 0.00),
    ("vivo_warm_soft", 1.03, 0.99, +0.008, 1.00, 1.04, 1.06, 0.96, 0.10),
]
LOOK_STRENGTH = 2.25
