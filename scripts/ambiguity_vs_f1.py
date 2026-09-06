"""Mật độ từ mơ hồ có dự đoán được độ khó câu hỏi không?

VÌ SAO CẦN. Appendix về tiếng Việt không dấu nói: bỏ dấu tạo mơ hồ từ vựng, và
ngữ cảnh rộng giúp khử mơ hồ đó. Ta đã đo được **lượng** mơ hồ (20.8% âm tiết),
nhưng chưa cho thấy nó **liên quan tới chất lượng trả lời**.

PHÉP KIỂM Ở ĐÂY, và giới hạn của nó. Trên văn bản CÓ DẤU, mơ hồ chưa xuất hiện —
dấu còn nguyên thì `má` và `mà` vẫn phân biệt được. Nên nếu mật độ mơ hồ vẫn dự
đoán được F1 trên dữ liệu có dấu, đó KHÔNG phải bằng chứng cho giả thuyết khử mơ
hồ: nó nói mật độ ấy tương quan với thứ khác (độ dài, thể loại câu hỏi, tần suất
từ).

Nói cách khác đây là **phép kiểm giả** (falsification test): kết quả ÂM ở đây làm
giả thuyết MẠNH hơn, vì nó loại trừ cách giải thích tầm thường. Kết quả dương lại
là dấu hiệu xấu — mật độ mơ hồ đang đo một biến gây nhiễu.

Chạy CPU: chỉ đọc `results/vimqa_full_7b.json` và đếm âm tiết.

    python scripts/ambiguity_vs_f1.py
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from itercomp.fertility import syllable_spans  # noqa: E402


def strip_tones(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text)
    base = "".join(c for c in nfd if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", base).replace("đ", "d").replace("Đ", "D")


def spearman(x: list[float], y: list[float]) -> float:
    """Tương quan hạng — F1 chặn ở 0 và 100 nên Pearson không phù hợp."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def boot_ci(x: list[float], y: list[float], n: int = 2000, seed: int = 0):
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        idx = [rng.randrange(len(x)) for _ in range(len(x))]
        v = spearman([x[i] for i in idx], [y[i] for i in idx])
        if v == v:
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-json", type=Path,
                    default=ROOT / "results" / "vimqa_full_7b.json")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ev = json.loads(args.eval_json.read_text())
    per_row = ev["per_row"]

    # ── ban do dong dang, ha thap TRUOC (xem homograph_analysis.py) ──
    from itercomp import load_dataset
    words: set[str] = set()
    for r in load_dataset(ev["config"]["dataset"], len(per_row)):
        for sents in r["context"]["sentences"]:
            for x in sents:
                words |= {x[a:b].lower() for a, b in syllable_spans(x)}
        words |= {r["question"][a:b].lower()
                  for a, b in syllable_spans(r["question"])}
    groups: dict[str, set[str]] = defaultdict(set)
    for w in words:
        groups[strip_tones(w)].add(w)
    homographs = {k: v for k, v in groups.items() if len(v) >= 2}

    # ── mat do mo ho tung cau hoi ───────────────────────────────────
    dens, f1 = defaultdict(list), defaultdict(list)
    methods = [m for m in per_row[0]["methods"]]
    for r in per_row:
        syls = [r["question"][a:b].lower()
                for a, b in syllable_spans(r["question"])]
        if not syls:
            continue
        d = sum(1 for s in syls
                if len(homographs.get(strip_tones(s), ())) >= 2) / len(syls)
        for m in methods:
            dens[m].append(d)
            f1[m].append(100 * r["methods"][m]["f1_norm"])

    print(f"eval={args.eval_json.name}  n={len(dens[methods[0]])}  "
          f"nhóm đồng dạng={len(homographs)}\n")
    print(f"{'phương pháp':12s} {'mật độ tb':>10s} {'F1 tb':>7s} "
          f"{'Spearman':>10s} {'CI 95%':>18s}")
    out = {}
    for m in methods:
        rho = spearman(dens[m], f1[m])
        lo, hi = boot_ci(dens[m], f1[m])
        out[m] = {"rho": rho, "ci_lo": lo, "ci_hi": hi,
                  "mean_density": st.mean(dens[m]), "mean_f1": st.mean(f1[m])}
        print(f"{m:12s} {100*st.mean(dens[m]):9.1f}% {st.mean(f1[m]):7.1f} "
              f"{rho:10.3f} {f'[{lo:+.2f}, {hi:+.2f}]':>18s}")

    # Mat do gan nhu hang so thi tuong quan khong doc duoc, bat ke dau va CI.
    spread = st.stdev(dens[methods[0]]) if len(dens[methods[0]]) > 1 else 0.0
    mean_d = st.mean(dens[methods[0]])
    print(f"\nĐỘ PHÂN TÁN của mật độ: sd {100*spread:.1f} điểm %, "
          f"trung bình {100*mean_d:.1f}%")
    if mean_d > 0.6:
        print("  ⚠ Mật độ quá cao để phân biệt được câu nào mơ hồ hơn câu nào.")
        print("    Nguyên nhân: mọi hư từ phổ biến tiếng Việt (là, có, ở, và,")
        print("    của) đều thuộc một nhóm đồng dạng, nên gần như MỌI câu đều")
        print("    có mật độ cao. Đây là hạn chế của PHÉP ĐO, không phải kết")
        print("    quả về dữ liệu — mọi tương quan dưới đây không đọc được.")

    sig = [m for m in methods if out[m]["ci_hi"] < 0 or out[m]["ci_lo"] > 0]
    print("\nĐỌC KẾT QUẢ — đây là phép kiểm GIẢ, kết quả âm mới là tốt")
    if not sig:
        print("  KHÔNG phương pháp nào có tương quan (mọi CI chứa 0).")
        print("  -> Mật độ mơ hồ KHÔNG dự đoán độ khó trên văn bản CÓ DẤU.")
        print("     Đúng như mong đợi: dấu còn nguyên thì mơ hồ chưa xuất hiện.")
        print("     Điều này LOẠI TRỪ cách giải thích tầm thường — rằng câu")
        print("     nhiều từ đồng dạng vốn đã khó vì lý do khác. Giả thuyết khử")
        print("     mơ hồ ở appendix vì thế đứng vững hơn.")
    else:
        print(f"  CÓ tương quan ở {', '.join(sig)} — dù dấu còn nguyên.")
        print("  -> DẤU HIỆU XẤU: mật độ mơ hồ đang đo một biến gây nhiễu (độ")
        print("     dài? thể loại? tần suất từ?), không phải mơ hồ do bỏ dấu.")
        print("     Phải kiểm soát biến đó trước khi dùng lập luận khử mơ hồ.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"config": vars(args) | {"out": str(args.out),
                                           "eval_json": str(args.eval_json)},
                   "n_homograph_groups": len(homographs),
                   "summary": out}, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2, default=str)
        print(f"\n→ ghi {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
