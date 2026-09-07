"""Adaptive Percentile có thật sự thích ứng, hay chỉ là k cố định trá hình?

CÂU HỎI. Paper nêu ở Limitations rằng cần "adaptive mechanisms that dynamically
adjust these settings based on question complexity" nhưng không hiện thực hoá.
Ta đã hiện thực (`scorer.adaptive_percentile`). Trước khi tốn GPU đo F1, phải
trả lời câu rẻ hơn và cơ bản hơn: **k có thật sự biến thiên theo câu hỏi không?**

Nếu k gần như hằng số thì AP chỉ là k cố định đội lốt, và mọi khác biệt F1 sau
này là nhiễu. Câu này trả lời được trên CPU, không cần reader.

CÁI ĐO. Với mỗi câu hỏi: k mà AP chọn, số segment giữ lại, tỉ lệ token. So với
k cố định 70/80/90/95. Phân tầng theo số câu gold cần (1/2/3 hop) — nếu AP đúng
thiết kế, câu cần nhiều bằng chứng phải nhận k THẤP hơn (giữ rộng hơn).

    python scripts/ablation_adaptive_k.py --limit 200
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from itercomp import load_dataset, make_scorer, tokenize  # noqa: E402
from itercomp.core import decompose, score_segments  # noqa: E402
from itercomp.scorer import (adaptive_percentile, percentile_filter,  # noqa: E402
                             score_concentration)

FIXED_K = (70.0, 80.0, 90.0, 95.0)


def n_gold(row) -> int:
    sf = row.get("supporting_facts") or {}
    return len(sf.get("title", []))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="vimqa",
                    choices=["vimqa", "hotpotqa", "2wiki", "musique"])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--scorer", default="bm25",
                    help="bm25 chạy CPU; dual sát báo cáo hơn nhưng cần tải bge-m3")
    ap.add_argument("--k-min", type=float, default=70.0)
    ap.add_argument("--k-max", type=float, default=95.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = load_dataset(args.dataset, args.limit)
    scorer = make_scorer(args.scorer)

    ks, conc, by_hop = [], [], defaultdict(list)
    kept = {"adaptive": [], **{k: [] for k in FIXED_K}}
    ratio = {"adaptive": [], **{k: [] for k in FIXED_K}}

    for i, r in enumerate(rows, 1):
        segs = decompose(r["context"])
        texts = [str(s) for s in segs]
        total = sum(len(tokenize(t)) for t in texts) or 1
        scores = score_segments(r["question"], texts, scorer)

        k_now = adaptive_percentile(scores, args.k_min, args.k_max)
        ks.append(k_now)
        conc.append(score_concentration(scores))
        by_hop[n_gold(r)].append(k_now)

        for name, k in [("adaptive", k_now)] + [(k, k) for k in FIXED_K]:
            idx = percentile_filter(scores, k)
            kept[name].append(len(idx))
            ratio[name].append(sum(len(tokenize(texts[j])) for j in idx) / total)
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    print(f"\ndataset={args.dataset}  n={len(rows)}  scorer={args.scorer}  "
          f"k in [{args.k_min}, {args.k_max}]\n")

    # ── cau hoi 1: k co bien thien khong? ───────────────────────────
    print("A. k CÓ BIẾN THIÊN KHÔNG?")
    print(f"  k: tb {st.mean(ks):.1f}  sd {st.stdev(ks):.2f}  "
          f"min {min(ks):.1f}  max {max(ks):.1f}")
    print(f"  độ tập trung: tb {st.mean(conc):.3f}  sd {st.stdev(conc):.3f}")
    uniq = len({round(k, 1) for k in ks})
    print(f"  số giá trị k khác nhau: {uniq}/{len(ks)}")
    if st.stdev(ks) < 1.0:
        print("  -> k gần như HẰNG SỐ. AP là k cố định đội lốt; không đáng chạy GPU.")
    else:
        print("  -> k biến thiên thật. AP đang phản ứng với từng câu hỏi.")

    # ── cau hoi 2: bien thien co DUNG CHIEU khong? ──────────────────
    print("\nB. k CÓ PHẢN ỨNG ĐÚNG CHIỀU VỚI ĐỘ KHÓ KHÔNG?")
    print("  thiết kế: câu cần NHIỀU bằng chứng -> k THẤP (giữ rộng hơn)")
    print(f"{'số câu gold':>12s} {'n':>5s} {'k trung bình':>14s}")
    hops = sorted(h for h in by_hop if h > 0)
    for h in hops:
        print(f"{h:12d} {len(by_hop[h]):5d} {st.mean(by_hop[h]):13.1f}")
    if len(hops) >= 2:
        a, b = by_hop[hops[0]], by_hop[hops[-1]]
        # Bootstrap: nhom nhieu-hop thuong rat nho (n=8 tren VimQA), nen mot
        # chenh lech trung binh khong kem CI thi khong doc duoc chieu nao ca.
        import random
        rng = random.Random(0)
        diffs = sorted(
            sum(b[rng.randrange(len(b))] for _ in b) / len(b)
            - sum(a[rng.randrange(len(a))] for _ in a) / len(a)
            for _ in range(5000))
        lo_ci, hi_ci = diffs[125], diffs[4875]
        d = st.mean(b) - st.mean(a)
        print(f"  hiệu {hops[-1]}hop − {hops[0]}hop: {d:+.2f}  "
              f"CI95 [{lo_ci:+.2f}, {hi_ci:+.2f}]")
        if hi_ci < 0:
            print(f"  -> ĐÚNG chiều: câu nhiều hop nhận k thấp hơn, có ý nghĩa.")
        elif lo_ci > 0:
            print(f"  -> NGƯỢC chiều, có ý nghĩa. Độ tập trung điểm KHÔNG thay")
            print(f"     cho độ khó câu hỏi được.")
        else:
            print(f"  -> KHÔNG kết luận được (CI chứa 0, n={len(b)} ở nhóm nhiều")
            print(f"     hop). AP biến thiên, nhưng không theo số hop.")

    # ── cau hoi 3: AP nam o dau tren duong cong danh doi? ───────────
    print("\nC. AP NẰM Ở ĐÂU TRÊN ĐƯỜNG CONG ĐÁNH ĐỔI?")
    print(f"{'cấu hình':>12s} {'segment giữ':>13s} {'tỉ lệ token':>13s}")
    out = {}
    for name in ["adaptive"] + list(FIXED_K):
        lbl = "adaptive" if name == "adaptive" else f"k={name:.0f}"
        out[lbl] = {"kept": st.mean(kept[name]), "ratio": st.mean(ratio[name])}
        print(f"{lbl:>12s} {st.mean(kept[name]):12.1f} "
              f"{100*st.mean(ratio[name]):12.1f}%")

    a_ratio = st.mean(ratio["adaptive"])
    near = min(FIXED_K, key=lambda k: abs(st.mean(ratio[k]) - a_ratio))
    print(f"\n  AP nén tương đương k={near:.0f} "
          f"({100*a_ratio:.1f}% vs {100*st.mean(ratio[near]):.1f}%).")
    print(f"  => Muốn biết AP có ĐÁNG không, phải so F1 của AP với F1 của k={near:.0f}")
    print("     ở CÙNG tỉ lệ nén này. Đó là phép so duy nhất có nghĩa, và nó cần GPU.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"config": vars(args) | {"out": str(args.out)},
                   "k_stats": {"mean": st.mean(ks), "sd": st.stdev(ks),
                               "min": min(ks), "max": max(ks), "n_unique": uniq},
                   "k_by_hop": {str(h): st.mean(v) for h, v in by_hop.items()},
                   "tradeoff": out, "matched_fixed_k": near},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
        print(f"\n→ ghi {args.out}")


if __name__ == "__main__":
    main()
