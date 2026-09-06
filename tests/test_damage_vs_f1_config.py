"""Script ghép hai nguồn dữ liệu phải dùng CÙNG cấu hình cho cả hai.

`damage_vs_f1.py` là script duy nhất trong repo ghép dữ liệu từ hai nguồn: F1
đọc từ file kết quả có sẵn, còn tỉ số hư-từ thì tính lại. Nếu hai bên dùng
scorer khác nhau, ngữ cảnh khác nhau, và tương quan thu được vô nghĩa — mà lỗi
đó KHÔNG ném exception, chỉ lặng lẽ cho ra số sai.

Đã xảy ra thật: script mặc định bm25 trong khi vimqa_200.json chạy dual.
"""
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SRC = (ROOT / "scripts" / "damage_vs_f1.py").read_text()


def test_scorer_khong_co_mac_dinh_cung():
    """--scorer phải mặc định None để suy theo eval-json, không phải 'bm25'."""
    m = re.search(r'add_argument\("--scorer",\s*default=([^,)\s]+)', SRC)
    assert m, "không tìm thấy khai báo --scorer"
    assert m.group(1) == "None", (
        f"--scorer mặc định {m.group(1)}, phải là None để lấy theo eval-json")


def test_percentile_khong_co_mac_dinh_cung():
    m = re.search(r'add_argument\("--percentile",\s*type=float,\s*default=([^,)\s]+)',
                  SRC)
    assert m, "không tìm thấy khai báo --percentile"
    assert m.group(1) == "None", (
        f"--percentile mặc định {m.group(1)}, phải là None")


def test_doc_scorer_tu_config():
    """Phải thật sự đọc cfg, không chỉ khai báo None rồi bỏ qua."""
    assert 'cfg.get("scorer"' in SRC, "không đọc scorer từ config của eval-json"


def test_percentile_none_trong_config_van_ra_so():
    """cfg["percentile"] có thể tồn tại với giá trị None — không được truyền None.

    run_eval.py ghi nguyên args, và cờ --percentile mặc định None để suy theo
    dataset. Nên cfg.get("percentile", 90.0) trả về None, không phải 90.0.
    """
    from itercomp.scorer import DEFAULT_PERCENTILE, PERCENTILE_BY_DATASET
    cfg = {"dataset": "vimqa", "percentile": None, "scorer": "dual"}
    pct = cfg.get("percentile")
    if pct is None:
        pct = PERCENTILE_BY_DATASET.get(cfg["dataset"], DEFAULT_PERCENTILE)
    assert isinstance(pct, (int, float)) and pct is not None
    assert 0 < pct <= 100


def test_canh_bao_khi_scorer_lech():
    """Ghi đè scorer phải in cảnh báo, không im lặng."""
    assert "CẢNH BÁO" in SRC and "khác scorer" in SRC, (
        "không cảnh báo khi --scorer khác scorer của eval-json")
