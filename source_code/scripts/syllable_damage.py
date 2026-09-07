"""Nén mức token huỷ cái gì của tiếng Việt — và tiếng Anh có bị như vậy không?

GIẢ THUYẾT BAN ĐẦU (ĐÃ BÁC BỎ). Ta đoán nén mức token cắt GIỮA âm tiết, phá
dấu thanh: "phở" -> "ph". Đo trên VimQA: LLMLingua-2 phá **0** âm tiết. Nó
tôn trọng biên âm tiết. Giả thuyết sai, và mục 2.3 PROPOSAL.md phải sửa theo.

CÁI ĐO ĐƯỢC THAY VÀO. Nhìn đầu ra thật thì hư hại nằm chỗ khác:

    "độ pH nhỏ hơn 7"          -> "pH nhỏ 7"        (mất "hơn": đảo so sánh)
    "sinh ngày 7 tháng 10 1988" -> "7 tháng 10 1988" (mất mốc "ngày")
    "Maradona sinh 30/10/1960"  -> "Maradona 30 10 1960"

Bộ nén cắt **hư từ** — âm tiết mang quan hệ ngữ pháp, không mang nội dung.

TẠI SAO ĐIỀU NÀY ĐẶC THÙ TIẾNG VIỆT. Tiếng Việt là ngôn ngữ ĐƠN LẬP: so sánh,
phủ định, thời gian nằm ở âm tiết ĐỘC LẬP ("hơn", "không", "năm"). Tiếng Anh
mã hoá cùng thông tin vào hình vị DÍNH LIỀN ("bigger", "-ed") mà bộ lọc token
không thể tách khỏi từ gốc. Nên cùng một bộ nén, cùng một tỉ lệ giữ token, huỷ
ngữ pháp tiếng Việt trong khi tiếng Anh vẫn an toàn — đây là hệ quả của LOẠI
HÌNH NGÔN NGỮ, không phải mô hình kém tiếng Việt.

Con số cần nhìn là TỈ SỐ mất-hư-từ / mất-nội-dung. Bằng 1,0 nghĩa là bộ nén
không thiên vị. Lớn hơn 1,0 nghĩa là nó huỷ ngữ pháp nhanh hơn huỷ thông tin.

Chạy CPU: LLMLingua-2 là BERT-base (~700MB), không cần GPU. Không nạp reader.

    python scripts/syllable_damage.py --dataset vimqa --limit 50
    python scripts/syllable_damage.py --dataset musique --limit 50   # đối chứng EN
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from itercomp import load_dataset, make_scorer  # noqa: E402
from itercomp.core import decompose, score_segments  # noqa: E402
from itercomp.fertility import (  # noqa: E402
    FUNCTION_WORDS, FUNCTION_WORDS_EN, count_broken_syllables,
    function_word_loss, syllable_spans)
from itercomp.scorer import percentile_filter  # noqa: E402


def ctx_full(row) -> str:
    return "\n".join(f"{t}: {' '.join(s)}" for t, s
                     in zip(row["context"]["title"], row["context"]["sentences"]))


def ctx_itercomp(row, scorer, pct: float) -> str:
    segs = decompose(row["context"])
    scores = score_segments(row["question"], [str(s) for s in segs], scorer)
    keep = percentile_filter(scores, pct)
    return "\n".join(f"{segs[i].doc_title}: {segs[i].text}" for i in keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="vimqa", choices=["vimqa", "musique"])
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--rate", type=float, default=0.33,
                    help="tỉ lệ giữ token của LLMLingua-2")
    ap.add_argument("--percentile", type=float, default=90.0)
    ap.add_argument("--scorer", default="bm25")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    # vimqa la tieng Viet; ba tap con lai tieng Anh -> danh sach hu tu khac
    lang = "vi" if args.dataset == "vimqa" else "en"
    fw_table = FUNCTION_WORDS if lang == "vi" else FUNCTION_WORDS_EN

    rows = load_dataset(args.dataset, args.limit)
    scorer = make_scorer(args.scorer)

    print(f"Nạp LLMLingua-2 (CPU, ~700MB) ...", flush=True)
    from llmlingua import PromptCompressor
    comp = PromptCompressor(
        model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        use_llmlingua2=True, device_map="cpu")

    stats = {"itercomp": [], "llmlingua2": []}
    for i, r in enumerate(rows, 1):
        full = ctx_full(r)
        n_syl = len(syllable_spans(full)) or 1

        ic = ctx_itercomp(r, scorer, args.percentile)
        ll = comp.compress_prompt(full, rate=args.rate,
                                  force_tokens=["\n", "?", ".", ","],
                                  drop_consecutive=True)["compressed_prompt"]

        for name, ctx in (("itercomp", ic), ("llmlingua2", ll)):
            stats[name].append({
                "broken": count_broken_syllables(ctx, full),
                "syl_orig": n_syl,
                "syl_kept": len(syllable_spans(ctx)),
                "loss": function_word_loss(ctx, full, lang),
                "n_syl_full": n_syl,
            })
        if i % 10 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    print(f"\ndataset={args.dataset} ({lang})  n={len(rows)}  "
          f"LLMLingua-2 rate={args.rate}  IterCOMP k={args.percentile}\n")

    def _mean(xs):
        xs = [x for x in xs if x == x]          # bo NaN
        return st.mean(xs) if xs else float("nan")

    out = {}
    for name, rec in stats.items():
        losses = [x["loss"] for x in rec]
        fw = _mean([l["hư từ"] for l in losses])
        ct = _mean([l["nội dung"] for l in losses])
        out[name] = {
            "broken_total": sum(x["broken"] for x in rec),
            "kept_pct_mean": st.mean(100 * x["syl_kept"] / x["syl_orig"]
                                     for x in rec),
            "fw_loss": fw, "content_loss": ct,
            "ratio": fw / ct if ct else float("nan"),
            "per_class": {c: _mean([l[c] for l in losses]) for c in fw_table},
            "rows": len(rec),
        }

    # ── phan 1: gia thuyet am tiet-vo ───────────────────────────────
    print("A. ÂM TIẾT BỊ CẮT DỞ (giả thuyết ban đầu)")
    print(f"{'phương pháp':13s} {'âm tiết vỡ':>12s} {'% âm tiết giữ':>16s}")
    for name, d in out.items():
        print(f"{name:13s} {d['broken_total']:12d} {d['kept_pct_mean']:15.1f}%")
    if all(d["broken_total"] == 0 for d in out.values()):
        print("  -> KHÔNG phương pháp nào phá âm tiết. Giả thuyết BỊ BÁC BỎ:")
        print("     LLMLingua-2 tôn trọng biên âm tiết tiếng Việt.")

    # ── phan 2: cai that su bi cat ──────────────────────────────────
    print("\nB. HƯ TỪ BỊ CẮT so với NỘI DUNG BỊ CẮT")
    print(f"{'phương pháp':13s} {'mất hư từ':>11s} {'mất nội dung':>14s} {'tỉ số':>9s}")
    for name, d in out.items():
        print(f"{name:13s} {100*d['fw_loss']:10.1f}% {100*d['content_loss']:13.1f}%"
              f" {d['ratio']:8.2f}x")
    print("  tỉ số 1,00x = cắt không thiên vị; >1,00x = huỷ ngữ pháp nhanh hơn")

    print("\nC. MẤT HƯ TỪ THEO LOẠI (%)")
    print(f"{'loại':12s} " + " ".join(f"{n:>13s}" for n in out))
    for cls in fw_table:
        cells = " ".join(f"{100*out[n]['per_class'][cls]:12.1f}%" for n in out)
        print(f"{cls:12s} " + cells)

    ll_r, ic_r = out["llmlingua2"]["ratio"], out["itercomp"]["ratio"]
    print("\nĐIỀU NÀY NÓI GÌ")
    if ll_r > ic_r * 1.15:
        print(f"  LLMLingua-2 cắt hư từ lệch {ll_r:.2f}x, IterCOMP {ic_r:.2f}x.")
        print("  Nén mức CÂU giữ nguyên ngữ pháp vì nó giữ trọn câu; nén mức")
        print("  TOKEN không giữ. Đây là lợi thế CẤU TRÚC trên ngôn ngữ đơn lập.")
    elif ic_r > ll_r * 1.15:
        print(f"  NGƯỢC dự đoán: IterCOMP lệch {ic_r:.2f}x, LLMLingua-2 {ll_r:.2f}x.")
    else:
        print(f"  Hai phương pháp lệch tương đương ({ic_r:.2f}x vs {ll_r:.2f}x);")
        print("  độ đo này không phân biệt được chúng.")
    print("  Chạy lại với --dataset musique để biết tiếng Anh có bị vậy không.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"config": vars(args) | {"out": str(args.out)},
                   "lang": lang, "summary": out,
                   # per-row de so sanh vi vs en co bootstrap o buoc sau
                   "per_row": {n: [x["loss"] for x in rec]
                               for n, rec in stats.items()}},
                  open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2, default=str)
        print(f"\n→ ghi {args.out}")


if __name__ == "__main__":
    main()
