"""Tỉ số hư-từ có bền theo mức nén, hay chỉ là tạo tác của siêu tham số?

VÌ SAO CẦN. `syllable_damage.py` so IterCOMP (giữ 24.3% âm tiết) với LLMLingua-2
(giữ 31.7%) trên VimQA — hai MỨC NÉN khác nhau. Một tỉ số đo ở hai mức nén khác
nhau thì không so được: nén càng mạnh càng phải bỏ nhiều thứ, kể cả hư từ.

Script này quét `rate` của LLMLingua-2 để lấy điểm KHỚP mức nén của IterCOMP,
và vẽ cả đường cong. Nếu tỉ số vẫn ~1,4x ở đúng mức nén của IterCOMP, kết luận
đứng vững. Nếu nó tụt về ~1,0x, kết luận cũ là tạo tác và phải rút lại.

Chạy CPU.

    python scripts/damage_curve.py --dataset vimqa --limit 50
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # để chạy được cả khi scripts/ nằm trong src/

from itercomp import load_dataset, make_scorer  # noqa: E402
from itercomp.core import decompose, score_segments  # noqa: E402
from itercomp.fertility import function_word_loss, syllable_spans  # noqa: E402
from itercomp.scorer import percentile_filter  # noqa: E402

RATES = (0.15, 0.20, 0.25, 0.33, 0.45)


def ctx_full(row) -> str:
    return "\n".join(f"{t}: {' '.join(s)}" for t, s
                     in zip(row["context"]["title"], row["context"]["sentences"]))


def ctx_itercomp(row, scorer, pct: float) -> str:
    segs = decompose(row["context"])
    scores = score_segments(row["question"], [str(s) for s in segs], scorer)
    keep = percentile_filter(scores, pct)
    return "\n".join(f"{segs[i].doc_title}: {segs[i].text}" for i in keep)


def _mean(xs):
    xs = [x for x in xs if x == x]
    return st.mean(xs) if xs else float("nan")


def summarise(recs: list[dict]) -> dict:
    fw = _mean([r["loss"]["hư từ"] for r in recs])
    ct = _mean([r["loss"]["nội dung"] for r in recs])
    return {"kept_pct": st.mean(100 * r["kept"] / r["orig"] for r in recs),
            "fw_loss": fw, "content_loss": ct,
            "ratio": fw / ct if ct else float("nan")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="vimqa", choices=["vimqa", "musique"])
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--percentile", type=float, default=90.0)
    ap.add_argument("--scorer", default="bm25")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    lang = "vi" if args.dataset == "vimqa" else "en"
    rows = load_dataset(args.dataset, args.limit)
    scorer = make_scorer(args.scorer)

    print("Nạp LLMLingua-2 (CPU) ...", flush=True)
    from llmlingua import PromptCompressor
    comp = PromptCompressor(
        model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        use_llmlingua2=True, device_map="cpu")

    ic_recs: list[dict] = []
    ll_recs: dict[float, list[dict]] = {r: [] for r in RATES}

    for i, r in enumerate(rows, 1):
        full = ctx_full(r)
        n_syl = len(syllable_spans(full)) or 1

        ic = ctx_itercomp(r, scorer, args.percentile)
        ic_recs.append({"loss": function_word_loss(ic, full, lang),
                        "kept": len(syllable_spans(ic)), "orig": n_syl})

        for rate in RATES:
            ll = comp.compress_prompt(full, rate=rate,
                                      force_tokens=["\n", "?", ".", ","],
                                      drop_consecutive=True)["compressed_prompt"]
            ll_recs[rate].append({"loss": function_word_loss(ll, full, lang),
                                  "kept": len(syllable_spans(ll)), "orig": n_syl})
        if i % 10 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    ic = summarise(ic_recs)
    curve = {rate: summarise(rec) for rate, rec in ll_recs.items()}

    print(f"\ndataset={args.dataset} ({lang})  n={len(rows)}\n")
    print(f"{'phương pháp':22s} {'giữ âm tiết':>12s} {'mất hư từ':>11s}"
          f" {'mất nội dung':>13s} {'tỉ số':>8s}")
    print(f"{'IterCOMP k=' + str(args.percentile):22s} {ic['kept_pct']:11.1f}%"
          f" {100*ic['fw_loss']:10.1f}% {100*ic['content_loss']:12.1f}%"
          f" {ic['ratio']:7.2f}x")
    for rate in RATES:
        c = curve[rate]
        print(f"{'LLMLingua-2 rate=' + f'{rate:.2f}':22s} {c['kept_pct']:11.1f}%"
              f" {100*c['fw_loss']:10.1f}% {100*c['content_loss']:12.1f}%"
              f" {c['ratio']:7.2f}x")

    # diem khop muc nen voi IterCOMP
    best = min(RATES, key=lambda r: abs(curve[r]["kept_pct"] - ic["kept_pct"]))
    b = curve[best]
    print(f"\nSO SÁNH CÔNG BẰNG — khớp mức nén (IterCOMP giữ {ic['kept_pct']:.1f}%)")
    print(f"  điểm khớp gần nhất: rate={best:.2f}, giữ {b['kept_pct']:.1f}%"
          f" (lệch {abs(b['kept_pct']-ic['kept_pct']):.1f} điểm)")
    print(f"  IterCOMP    {ic['ratio']:.2f}x")
    print(f"  LLMLingua-2 {b['ratio']:.2f}x")

    gap = b["ratio"] - ic["ratio"]
    print(f"\n  chênh lệch ở CÙNG mức nén: {gap:+.2f}")
    if gap > 0.15:
        print("  -> Kết luận ĐỨNG VỮNG: nén mức token huỷ ngữ pháp lệch hơn")
        print("     nén mức câu, ngay cả khi hai bên nén bằng nhau.")
    else:
        print("  -> Kết luận KHÔNG đứng vững: chênh lệch cũ là tạo tác của việc")
        print("     so ở hai mức nén khác nhau. PHẢI rút lại mục 2.3.")

    # ti so co tang theo do nen khong?
    ks = [curve[r]["kept_pct"] for r in RATES]
    rs = [curve[r]["ratio"] for r in RATES]
    print(f"\n  tỉ số LLMLingua-2 theo mức nén: "
          f"{min(rs):.2f}x .. {max(rs):.2f}x khi giữ {min(ks):.0f}%..{max(ks):.0f}%")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"config": vars(args) | {"out": str(args.out)}, "lang": lang,
                   "itercomp": ic, "llmlingua2_curve": {str(k): v for k, v in curve.items()},
                   "matched_rate": best},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
        print(f"\n→ ghi {args.out}")


if __name__ == "__main__":
    main()
