"""Kiểm định thống kê cho so sánh các phương pháp nén.

Vì sao cần. Với n=50, sai số chuẩn của F1 vào khoảng ±13 điểm, nên chênh lệch
vài điểm giữa hai cấu hình là nhiễu chứ không phải tín hiệu. Nếu chỉ nhìn con số
trung bình rồi tuyên bố "cấu hình A tốt hơn B" thì rất dễ kết luận sai.

Hai công cụ ở đây:

    bootstrap_ci        khoảng tin cậy không giả định phân phối chuẩn. Phù hợp
                        với F1 vì F1 bị chặn trong [0,1] và phân bố lệch, nên
                        xấp xỉ chuẩn không đáng tin ở cỡ mẫu nhỏ.

    paired_bootstrap    so sánh HAI phương pháp trên CÙNG tập câu hỏi. Mạnh hơn
                        hẳn so sánh độc lập vì loại được phương sai giữa các câu
                        (câu khó thì mọi phương pháp đều khó). Đây là kiểm định
                        đúng cho câu hỏi "chuẩn hoá boolean có cải thiện thật
                        không", vì cùng một dự đoán chỉ khác cách chấm điểm.

Dùng seed cố định để kết quả tái lập được.
"""

from __future__ import annotations

import random
from statistics import mean

DEFAULT_ITERS = 10_000
SEED = 20260808


def bootstrap_ci(scores: list[float], iters: int = DEFAULT_ITERS,
                 alpha: float = 0.05, seed: int = SEED) -> dict:
    """Khoảng tin cậy percentile bootstrap cho giá trị trung bình."""
    if not scores:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = random.Random(seed)
    n = len(scores)
    means = []
    for _ in range(iters):
        means.append(mean(rng.choices(scores, k=n)))
    means.sort()
    lo = means[int(alpha / 2 * iters)]
    hi = means[int((1 - alpha / 2) * iters) - 1]
    return {"mean": mean(scores), "lo": lo, "hi": hi, "n": n}


def paired_bootstrap(a: list[float], b: list[float],
                     iters: int = DEFAULT_ITERS,
                     seed: int = SEED) -> dict:
    """So sánh có ghép cặp giữa hai phương pháp trên cùng tập câu hỏi.

    Trả về chênh lệch trung bình (a − b), khoảng tin cậy 95% của chênh lệch, và
    giá trị p hai phía. p ở đây là tỉ lệ lần lấy mẫu lại mà dấu của chênh lệch
    đảo ngược so với dấu quan sát được — cách diễn giải trực tiếp, không cần giả
    định phân phối.
    """
    if len(a) != len(b):
        raise ValueError("hai danh sách phải cùng độ dài (ghép cặp theo câu hỏi)")
    if not a:
        return {"diff": 0.0, "lo": 0.0, "hi": 0.0, "p": 1.0, "n": 0}

    rng = random.Random(seed)
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    obs = mean(diffs)

    boot = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        boot.append(mean(diffs[i] for i in idx))
    boot.sort()
    lo, hi = boot[int(0.025 * iters)], boot[int(0.975 * iters) - 1]

    # p hai phía: tỉ lệ mẫu bootstrap nằm về phía đối diện của 0
    if obs >= 0:
        p = 2 * sum(1 for d in boot if d <= 0) / iters
    else:
        p = 2 * sum(1 for d in boot if d >= 0) / iters
    return {"diff": obs, "lo": lo, "hi": hi, "p": min(p, 1.0), "n": n}


def min_detectable_diff(scores: list[float], power: float = 0.8,
                        alpha: float = 0.05) -> float:
    """Chênh lệch nhỏ nhất phát hiện được với cỡ mẫu hiện tại.

    Trả lời câu hỏi thực tế: "với n câu hỏi, chênh bao nhiêu điểm mới đáng tin?"
    Nếu chênh quan sát được nhỏ hơn giá trị này thì không nên kết luận.
    """
    import math
    n = len(scores)
    if n < 2:
        return float("inf")
    m = mean(scores)
    var = sum((x - m) ** 2 for x in scores) / (n - 1)
    z_a, z_b = 1.96, 0.84          # alpha=0.05 hai phía, power=0.8
    return (z_a + z_b) * math.sqrt(2 * var / n)


def required_n(diff: float, scores: list[float],
               power: float = 0.8, alpha: float = 0.05) -> int:
    """Cỡ mẫu cần để phát hiện một chênh lệch cho trước."""
    import math
    if diff <= 0:
        return -1
    m = mean(scores)
    var = sum((x - m) ** 2 for x in scores) / max(len(scores) - 1, 1)
    z_a, z_b = 1.96, 0.84
    return math.ceil(2 * var * (z_a + z_b) ** 2 / diff ** 2)
