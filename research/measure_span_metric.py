"""Đo tác động của f1_span trên các kết quả ĐÃ CÓ, không cần chạy lại mô hình.

per_row đã lưu dự đoán của từng phương pháp, nên chỉ việc tính lại độ đo. Chạy
trên cả tiếng Việt và tiếng Anh: nếu phép đo hợp lệ thì nó phải nâng điểm ở
tiếng Việt và gần như không đổi ở tiếng Anh — bộ dữ liệu tiếng Anh là NHÓM ĐỐI
CHỨNG tự có, không cần thiết kế thêm.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # để chạy được cả khi scripts/ nằm trong src/

from itercomp import f1_score, f1_span, paired_bootstrap  # noqa: E402
from itercomp.metrics import normalize  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
METH = ("raw", "oracle", "llmlingua2", "itercomp")
VI = {"vimqa"}


def rows_of(path: Path) -> list[dict]:
    d = json.load(open(path, encoding="utf-8"))
    return d.get("per_row") or []


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--show-changed", action="store_true",
                    help="in từng câu được nâng điểm để kiểm tra bằng mắt")
    args = ap.parse_args()

    files = []
    for ds in ("vimqa", "musique", "hotpotqa", "2wiki"):
        for f in sorted(glob.glob(str(RESULTS / f"{ds}_*.json"))):
            if any(k in f for k in ("ablation", "lambda", "exp", "boolnorm")):
                continue
            if rows_of(Path(f)):
                files.append((ds, Path(f)))

    print(f"{'file':34s} {'method':12s} {'F1':>7s} {'F1+span':>8s} {'Δ':>7s} {'#đổi':>5s}")
    print("-" * 78)
    per_lang: dict[str, list[float]] = {"vi": [], "en": []}

    for ds, f in files:
        rows = rows_of(f)
        for m in METH:
            cur = new = 0.0
            changed = []
            for r in rows:
                got = r["methods"].get(m)
                if not got:
                    continue
                p = str(got.get("pred_norm") or got.get("pred") or "")
                g = r["gold"]
                a, b = f1_score(p, g), f1_span(p, g)
                cur += a
                new += b
                if b > a + 1e-9:
                    changed.append((p, g))
            n = len(rows) or 1
            a, b = 100 * cur / n, 100 * new / n
            per_lang["vi" if ds in VI else "en"].append(b - a)
            print(f"{f.name:34s} {m:12s} {a:7.1f} {b:8.1f} {b-a:+7.1f} "
                  f"{len(changed):5d}")
            if args.show_changed and changed:
                for p, g in changed:
                    ok = normalize(p) in normalize(g)
                    print(f"      [{'OK' if ok else 'SAI'}] {p[:40]!r}")
                    print(f"           ⊂ {g[:56]!r}")

    print("\n=== NHÓM ĐỐI CHỨNG ===")
    for lang, xs in per_lang.items():
        if xs:
            print(f"  {lang}: trung bình {sum(xs)/len(xs):+.2f} F1 "
                  f"trên {len(xs)} phép đo")
    if per_lang["vi"] and per_lang["en"]:
        vi = sum(per_lang["vi"]) / len(per_lang["vi"])
        en = sum(per_lang["en"]) / len(per_lang["en"])
        print(f"\n  Tiếng Việt {vi:+.2f} vs tiếng Anh {en:+.2f}")
        if abs(en) < 1.0 < vi:
            print("  => Phép đo chỉ tác động lên tiếng Việt: sửa một lệch đặc")
            print("     thù, không phải thổi phồng điểm chung.")
        else:
            print("  => CẢNH BÁO: tiếng Anh cũng đổi đáng kể, cần xem lại.")


if __name__ == "__main__":
    main()
