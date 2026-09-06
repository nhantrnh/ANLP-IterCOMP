"""Template so tuoi chiem bao nhieu VimQA, va no lam lech ket qua the nao.

VI SAO CAN SCRIPT NAY. Bao cao tung ghi "98 (12.8%)" voi mau so 763 — nhung
VimQA dev co 1.003 cau, nen ti le dung la 9.8%. Nam con so cua muc do (763,
12.8%, 665, 26.3%, 63.1) khong script nao tinh va verifier khong pin, nen chung
la nhung con so KHONG truy nguoc duoc — dung loai loi ma repo-freshness canh bao.

    python scripts/age_comparison_split.py --eval results/vimqa_full_7b.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

# "X lon/nho tuoi hon Y phai khong?" — khuon duy nhat chiem gan mot phan muoi tap
TEMPLATE = re.compile(r"(lớn|nhỏ)\s+tuổi\s+hơn", re.IGNORECASE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", type=Path, default=Path("results/vimqa_full_7b.json"))
    ap.add_argument("--also", type=Path, default=None,
                    help="eval thu hai de doi chieu reader yeu (vd 0.5B)")
    ap.add_argument("--method", default="itercomp")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ev = json.loads(args.eval.read_text())
    rows = ev["per_row"]
    groups: dict[str, list[float]] = {"age": [], "rest": []}
    for r in rows:
        f1 = 100 * r["methods"][args.method]["f1_norm"]
        groups["age" if TEMPLATE.search(r["question"]) else "rest"].append(f1)

    n_tot = len(rows)
    out = {"n_total": n_tot, "method": args.method, "groups": {}}
    print(f"eval={args.eval.name}  method={args.method}  n={n_tot}\n")
    print(f"{'nhom':16s} {'n':>5s} {'% tap':>7s} {'F1=0':>8s} {'F1 TB':>7s}")
    for k, name in (("age", "So tuoi"), ("rest", "Con lai")):
        v = groups[k]
        zero = 100 * sum(1 for x in v if x == 0) / len(v)
        out["groups"][k] = {"n": len(v), "pct_of_set": 100 * len(v) / n_tot,
                            "pct_f1_zero": zero, "mean_f1": st.mean(v)}
        print(f"{name:16s} {len(v):5d} {100*len(v)/n_tot:6.1f}% "
              f"{zero:7.1f}% {st.mean(v):7.1f}")

    a, r = out["groups"]["age"], out["groups"]["rest"]
    print(f"\nTemplate nay chiem {a['pct_of_set']:.1f}% cua tap "
          f"({a['n']}/{n_tot}), khong phai 12.8% — mau so 763 la sai.")
    print(f"F1=0 tren {a['pct_f1_zero']:.1f}% nhom nay so voi {r['pct_f1_zero']:.1f}% "
          f"phan con lai -> KHO HON theo ti le that bai hoan toan.")
    print(f"Nhung F1 trung binh lai CAO HON ({a['mean_f1']:.1f} vs {r['mean_f1']:.1f}): "
          f"dap an boolean, dung la duoc 1.0 tron.")
    print("Hai do do NGUOC CHIEU nhau — phai bao cao ca hai, khong chon mot.")

    if args.also and args.also.exists():
        ev2 = json.loads(args.also.read_text())
        m2 = args.method
        rs = [r for r in ev2["per_row"] if m2 in r.get("methods", {})]
        g2 = {"age": [], "rest": []}
        for r in rs:
            f1 = 100 * r["methods"][m2]["f1_norm"]
            g2["age" if TEMPLATE.search(r["question"]) else "rest"].append(f1)
        print(f"\nDOI CHIEU {args.also.name} "
              f"(reader {ev2['config'].get('reader_model')}):")
        out["compare"] = {}
        for k, name in (("age", "So tuoi"), ("rest", "Con lai")):
            v = g2[k]
            zero = 100 * sum(1 for x in v if x == 0) / len(v)
            out["compare"][k] = {"n": len(v), "pct_f1_zero": zero,
                                 "mean_f1": st.mean(v)}
            print(f"  {name:10s} n={len(v):4d}  F1=0 {zero:5.1f}%  TB {st.mean(v):5.1f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"config": vars(args) | {"eval": str(args.eval),
                                           "out": str(args.out)}, **out},
                  args.out.open("w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
        print(f"\n→ ghi {args.out}")


if __name__ == "__main__":
    main()
