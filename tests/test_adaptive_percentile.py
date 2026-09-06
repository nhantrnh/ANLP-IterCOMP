"""Ba tính chất mà Adaptive Percentile BẮT BUỘC giữ.

Không phải test cho vui: cả ba đều là điều kiện để so sánh AP với paper gốc còn
có nghĩa. Phá một cái là mọi con số ablation thành vô giá trị, mà lỗi kiểu đó
không hiện ra thành exception — nó chỉ làm kết quả sai lặng lẽ.

    python -m pytest tests/ -q
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from itercomp.scorer import (adaptive_percentile, adaptive_percentile_filter,
                             percentile_filter, score_concentration)


def test_suy_bien_ve_paper_goc():
    """k_min = k_max phải trả về đúng hằng số đó.

    Đây là điều kiện để so sánh AP với k cố định là so sánh CÓ KIỂM SOÁT:
    cùng một đường mã, chỉ khác tham số.
    """
    for k in (70.0, 85.0, 90.0, 95.0):
        assert adaptive_percentile([1, 5, 2, 9, 3], k, k) == k


def test_bat_bien_theo_thang_diem():
    """Nhân toàn bộ điểm với hằng số dương không được đổi k.

    bge-m3 và BM25 cho điểm ở hai thang hoàn toàn khác nhau. Một độ đo tập
    trung nhạy thang sẽ cho k khác nhau chỉ vì đổi encoder, và ablation
    "AP vs cố định" sẽ đo nhầm hiệu ứng encoder thành hiệu ứng AP.
    """
    base = [0.1, 0.25, 0.3, 0.9, 0.15]
    for mult in (10, 1000, 0.001):
        assert math.isclose(score_concentration(base),
                            score_concentration([x * mult for x in base]),
                            rel_tol=1e-9)


def test_bien_do_k_ton_trong_khoang():
    """k luôn nằm trong [k_min, k_max], kể cả với đầu vào bệnh lý."""
    cases = [[5, 5, 5, 5], [0, 0, 0, 1], [1], [], [0, 0], [1e9, 1, 1]]
    for scores in cases:
        k = adaptive_percentile(scores, 70.0, 95.0)
        assert 70.0 <= k <= 95.0, f"k={k} ngoài khoảng với {scores}"


def test_phan_ung_dung_chieu_voi_do_tap_trung():
    """Điểm dồn vào ít segment -> k cao (cắt mạnh); dàn đều -> k thấp."""
    tap_trung = adaptive_percentile([1, 1, 1, 1, 100], 70.0, 95.0)
    dan_deu = adaptive_percentile([5, 5, 5, 5, 5], 70.0, 95.0)
    assert tap_trung > dan_deu


def test_khong_bao_gio_tra_ve_rong():
    """Prompt rỗng làm reader trả lời bừa — tệ hơn là nén kém."""
    for scores in ([0.0, 0.0, 0.0], [1.0], [3, 1, 4, 1, 5]):
        assert len(adaptive_percentile_filter(scores)) >= 1
    assert adaptive_percentile_filter([]) == []


def test_khop_percentile_filter_khi_cung_k():
    """AP chỉ được khác percentile_filter ở CHỖ CHỌN k, không ở chỗ lọc."""
    scores = [3, 1, 4, 1, 5, 9, 2, 6]
    k = adaptive_percentile(scores)
    assert adaptive_percentile_filter(scores) == percentile_filter(scores, k)


# ── --only cua run_ablation.py ────────────────────────────────────────
# Toan bo 26 cau hinh o n=500 ton ~10.5h GPU (do that: 2.90 s/cau), vuot
# mot phien Kaggle. --only cho chay co trong diem.

def _configs():
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_ablation.py").read_text()
    ns: dict = {}
    exec(src[src.index("CONFIGS = ("):src.index("def main()")], ns)
    return ns["CONFIGS"]


def test_only_loc_dung_tien_to():
    cfgs = _configs()
    pref = ("percentile", "max_iter")
    got = [n for n, _ in cfgs if n.startswith(pref)]
    assert len(got) == 13, f"doi 13 cau hinh (5 max_iter + 8 k), duoc {len(got)}: {got}"
    assert all(g.startswith(pref) for g in got)


def test_only_khong_khop_thi_rong():
    cfgs = _configs()
    assert [n for n, _ in cfgs if n.startswith(("khong-co-that",))] == []


def test_co_du_29_cau_hinh():
    # 26 + 3 k cao them cho quet khop-muc-nen tren MuSiQue
    assert len(_configs()) == 29
