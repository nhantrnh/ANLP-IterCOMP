"""Hai bảng phụ lục của paper gốc mà bản tái hiện còn thiếu.

Paper gốc đo nhiều hơn EM/F1/ratio. Hai bảng phụ lục quan trọng:

  Table 4  — kết quả theo SỐ HOP (EM, F1, #vòng, #token)
  Table 6  — Cost Reduction (%) và Speedup trên 5 LLM qua API

Cả hai tính được từ `results/` có sẵn + dataset local, KHÔNG cần GPU. Đây là
lần thứ ba trong dự án bài học đó lặp lại: kiểm dữ liệu đã có trước khi xin
phiên GPU.

MỘT CẢNH BÁO VỀ SPEEDUP. Paper ghi rõ Table 6 chỉ đo *"the final QA step
performed by the reader LLM"* — tức KHÔNG tính chi phí nén của chính nó. Đo
end-to-end thì bức tranh khác, và script in cả hai để người đọc thấy khác biệt
nằm ở định nghĩa, không ở dữ liệu.

    python scripts/hop_and_efficiency.py --eval results/vimqa_full_7b.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def gold_counts(dataset: str, n: int | None) -> dict[str, int]:
    """Số câu vàng cho mỗi câu hỏi — proxy cho số hop."""
    from itercomp import load_dataset
    out = {}
    for r in load_dataset(dataset, n):
        sf = r.get("supporting_facts") or {}
        titles = sf.get("title") or []
        out[r["question"]] = len(titles)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--dataset", default=None, help="mặc định lấy từ eval-json")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ev = json.loads(args.eval.read_text())
    cfg, rows, summary = ev["config"], ev["per_row"], ev["summary"]
    ds = args.dataset or cfg["dataset"]
    methods = list(summary)

    print(f"eval={args.eval.name}  dataset={ds}  n={len(rows)}  "
          f"reader={cfg.get('reader_model')}\n")

    # ── A. theo so hop (Table 4) ──────────────────────────────────────
    gc = gold_counts(ds, cfg.get("limit"))
    buckets: dict[int, list] = defaultdict(list)
    missing = 0
    for r in rows:
        k = gc.get(r["question"])
        if k is None:
            missing += 1
            continue
        buckets[k].append(r)
    if missing:
        print(f"  (bỏ {missing} câu không khớp được với dataset)\n")

    print("A. THEO SỐ CÂU VÀNG (proxy cho số hop) — tương ứng Table 4 paper")
    hdr = f"{'#vàng':>6s} {'n':>5s}"
    for m in methods:
        hdr += f" {m[:9]:>10s}"
    print(hdr + "   (F1*)")
    per_hop = {}
    for k in sorted(buckets):
        rs = buckets[k]
        line = f"{k:6d} {len(rs):5d}"
        per_hop[k] = {"n": len(rs)}
        for m in methods:
            f1 = 100 * st.mean(x["methods"][m]["f1_norm"] for x in rs)
            tok = st.mean(x["methods"][m]["tokens"] for x in rs)
            per_hop[k][m] = {"f1": f1, "tokens": tok}
            line += f" {f1:10.2f}"
        print(line)
    print("       token TB:")
    for k in sorted(buckets):
        line = f"{k:6d} {'':5s}"
        for m in methods:
            line += f" {per_hop[k][m]['tokens']:10.0f}"
        print(line)

    # ── B. hieu qua (Table 6) ─────────────────────────────────────────
    raw = summary.get("raw")
    print("\nB. HIỆU QUẢ — tương ứng Table 6 paper")
    if raw is None:
        print("  (không có 'raw' trong eval-json; bỏ qua)")
        eff = {}
    else:
        print(f"{'phương pháp':13s} {'token':>9s} {'giảm token':>11s} "
              f"{'s/câu':>7s} {'speedup end-to-end':>20s}")
        eff = {}
        for m in methods:
            v = summary[m]
            red = 100 * (1 - v["avg_tokens"] / raw["avg_tokens"])
            sp = raw["sec_per_q"] / v["sec_per_q"] if v["sec_per_q"] else float("nan")
            eff[m] = {"avg_tokens": v["avg_tokens"], "token_reduction": red,
                      "sec_per_q": v["sec_per_q"], "speedup_e2e": sp}
            print(f"{m:13s} {v['avg_tokens']:9.1f} {red:10.1f}% "
                  f"{v['sec_per_q']:7.2f} {sp:19.2f}×")

        print("\n  ĐỌC HAI CỘT CUỐI CHO ĐÚNG. Paper Table 6 ghi rõ chỉ đo *bước")
        print("  đọc cuối*, KHÔNG tính chi phí nén. Cột trên là end-to-end, nên")
        print("  nó bao gồm cả việc chấm điểm và vòng lặp của IterCOMP.")
        ic = eff.get("itercomp")
        if ic and ic["speedup_e2e"] < 1:
            print(f"  -> IterCOMP giảm {ic['token_reduction']:.1f}% token nhưng "
                  f"end-to-end CHẬM hơn {1/ic['speedup_e2e']:.2f}× so với raw.")
            print("     Reader ở đây chạy local trên T4; paper đo LLM qua API, nơi")
            print("     độ trễ mạng lấn át nên cùng phép nén lại cho speedup thật.")
            print("     Khác biệt nằm ở CÁCH ĐO, không ở dữ liệu.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"config": vars(args) | {"eval": str(args.eval),
                                           "out": str(args.out)},
                   "n": len(rows), "per_hop": per_hop, "efficiency": eff},
                  args.out.open("w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
        print(f"\n→ ghi {args.out}")


if __name__ == "__main__":
    main()
