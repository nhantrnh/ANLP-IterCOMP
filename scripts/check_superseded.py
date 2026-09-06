"""Bài báo còn trích số từ lần chạy ĐÃ BỊ THAY THẾ không?

VÌ SAO CẦN. Mọi lần chạy ở cỡ mẫu lớn hơn trong dự án này đều lật một kết luận
đã viết: `max_iter` bão hoà, đường cong k, khoảng cách MuSiQue-VimQA, IterCOMP
vượt Raw trên tiếng Anh, "số của ta cao hơn paper". Kết quả mới thì được viết
vào ngay — nhưng khẳng định cũ mà nó bác bỏ thì nằm im tới khi có người đọc ra.

Script này không đọc hộ được. Nó làm một việc hẹp hơn nhưng tự động: với mỗi
cặp file kết quả CÙNG dataset + reader + method mà khác cỡ mẫu, nó tìm xem bài
có đang trích con số của bản NHỎ hơn không, và báo để người viết quyết định —
số nhỏ có thể vẫn hợp lệ (mốc lịch sử, so sánh cỡ mẫu), nhưng phải là lựa chọn
có ý thức chứ không phải sót.

    python scripts/check_superseded.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = [ROOT / "report_en" / "report.tex", ROOT / "paper_submission" / "paper.tex"]

# Số nhỏ được trích CÓ Ý THỨC: khai ở đây kèm lý do, nếu không sẽ bị báo.
DELIBERATE: dict[str, str] = {
    "35.1": "IterCOMP MuSiQue n=50 — nêu để đối chiếu với n=500 (26.74)",
    "57.0": "Oracle MuSiQue n=50 — bảng tab:vspaper, có ghi n=50",
    "32.8": "Raw MuSiQue n=50 — cùng bảng",
    "13.5": "LLMLingua-2 MuSiQue n=50 — cùng bảng",
    "1.74": "Oracle/Raw n=50 — trích cùng 2.13 của n=500",
    "0.62": "IC/Or MuSiQue n=50 — bảng tab:decomp, có ghi n=50",
    "21.6": "IterCOMP-LLMLingua2 n=50 — nêu cùng +11.60 của n=500",
    "33.03": "IterCOMP k=90 n=50 — bảng ablation, có ghi n=50",
    "38.77": "k=70 n=50 — nêu de cho thay thu hang DAO o n=500",
    "36.60": "k=80 n=50 — cùng bảng ablation",
    "36.35": "k=85 n=50 — cùng bảng ablation",
    "34.56": "k=95 n=50 — cùng bảng ablation",
    "28.36": "max_iter=1 n=50 — bảng saturation, có ghi n=50",
    "43.0": "IterCOMP VimQA n=50 — nêu cùng +4.42 của n=1003",
    "43.8": "Raw VimQA n=50 — cùng chỗ",
    "53.3": "Oracle VimQA n=50 — bảng tab:decomp",
    "30.7": "LLMLingua-2 VimQA n=50 — bảng tab:vspaper",
    "1.22": "Oracle/Raw VimQA n=50 — trích cùng 1.30 của n=1003",
    "0.81": "IC/Or VimQA n=50",
    "12.4": "IterCOMP-LLMLingua2 VimQA n=50",
    "37.5": "F1 ablation top-k=3 (app:open, 16.3%% ratio) — TRUNG lam tron voi hotpotqa_500 LLML-2 37.52, nhung la dai luong KHAC; bang 4ds da dung 36.00 (n=1500)",
}


def summaries() -> dict[tuple, list[tuple[int, str, dict]]]:
    """Nhóm các lần chạy theo (dataset, reader), giữ cỡ mẫu để so."""
    g = defaultdict(list)
    for f in sorted((ROOT / "results").glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        c, s = d.get("config"), d.get("summary")
        if not isinstance(c, dict) or not isinstance(s, dict):
            continue
        n = c.get("limit")
        if not isinstance(n, int) or not c.get("reader_model"):
            continue
        g[(c["dataset"], c["reader_model"])].append((n, f.name, s))
    return g


def main() -> int:
    tex = "\n".join(p.read_text() for p in TEX if p.exists())
    found: list[str] = []
    for (ds, reader), runs in summaries().items():
        if len(runs) < 2:
            continue
        big = max(n for n, _, _ in runs)
        for n, name, s in runs:
            if n >= big:
                continue
            for m, v in s.items():
                f1 = v.get("f1_norm")
                if f1 is None or f1 != f1:
                    continue
                tok = f"{f1:.1f}"
                if tok in DELIBERATE:
                    continue
                # chi bao neu con so NAY xuat hien trong bai
                if re.search(rf"(?<![\d.]){re.escape(tok)}(?![\d])", tex):
                    bigger = next((x[2].get(m, {}).get("f1_norm")
                                   for x in runs if x[0] == big), None)
                    found.append(
                        f"  {ds}/{m}: bài trích {tok} (n={n}, {name}) "
                        f"nhưng có n={big} cho {bigger:.2f}" if bigger else
                        f"  {ds}/{m}: bài trích {tok} (n={n}, {name}); có n={big}")
    if not found:
        print("Không có số nào từ lần chạy bị thay thế mà chưa khai.")
        return 0
    print(f"{len(found)} con số từ lần chạy NHỎ hơn đang xuất hiện trong bài:\n")
    for f in sorted(set(found)):
        print(f)
    print("\nMỗi cái phải hoặc đổi sang lần chạy lớn nhất, hoặc khai trong "
          "DELIBERATE kèm lý do (vd: nêu làm mốc lịch sử).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
