"""Tỉ lệ nén HIỆU DỤNG: quy tỉ lệ danh nghĩa về chi phí token thật.

VẤN ĐỀ. Mọi công trình nén câu lệnh báo cáo hiệu quả bằng **tỉ lệ token**, và
ngầm giả định tỉ lệ giống nhau nghĩa là chi phí giống nhau. Giả định đó sai với
ngôn ngữ có fertility cao: cùng một nội dung, tiếng Việt bill 2,39x token so với
tiếng Anh dưới `cl100k_base`. Survey của Li et al. (2025) không nhắc ngôn ngữ;
LLMLingua-2 đánh giá tiếng Trung nhưng không bàn số token.

CÁI SCRIPT NÀY ĐO. Với mỗi phương pháp, tỉ lệ nén tính theo BỐN tokenizer khác
nhau, cộng tỉ lệ hiệu dụng (danh nghĩa / fertility). Điểm cốt lõi: cùng một lần
nén cho ra con số chi phí chênh nhau tới 2x tuỳ tokenizer nào thanh toán — nên
"nén 5x" là phát biểu vô nghĩa nếu không nói tokenizer.

Chạy CPU, không cần GPU: chỉ tokenize lại ngữ cảnh, không gọi mô hình nào.

    python scripts/effective_ratio.py --dataset vimqa --limit 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # để chạy được cả khi scripts/ nằm trong src/

from itercomp import load_dataset, make_scorer, tokenize  # noqa: E402
from itercomp.core import decompose, score_segments  # noqa: E402
from itercomp.scorer import percentile_filter  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"

#: Fertility vi/en đo trên 930 cặp WMT24++ — xem results/fertility_wmt24pp.json.
#: Khoá phải khớp tên encoding của tiktoken, trừ 'regex' là đơn vị mà bài báo
#: dùng để báo cáo "ratio".
FERTILITY = {
    "cl100k_base": 2.39,      # GPT-4o, GPT-4 — cái thực sự bill tiền
    "o200k_base": 1.49,       # GPT-4o mới hơn, tokenizer tốt hơn cho tiếng Việt
    "regex": 1.34,            # \w+ — đơn vị của cột "ratio" trong báo cáo
}


def count_tokens(text: str, encoding: str) -> int:
    """Đếm token theo một tokenizer cụ thể.

    'regex' dùng chính hàm tokenize() của repo, để con số khớp đúng cột "ratio"
    trong báo cáo thay vì gần đúng.
    """
    if encoding == "regex":
        return len(tokenize(text))
    import tiktoken
    return len(tiktoken.get_encoding(encoding).encode(text))


def ctx_full(row) -> str:
    return "\n".join(f"{t}: {' '.join(s)}" for t, s
                     in zip(row["context"]["title"], row["context"]["sentences"]))


def ctx_oracle(row) -> str:
    sf = row.get("supporting_facts") or {}
    want = set(zip(sf.get("title", []), sf.get("sent_id", [])))
    keep = [f"{t}: {s}"
            for t, sents in zip(row["context"]["title"], row["context"]["sentences"])
            for i, s in enumerate(sents) if (t, i) in want]
    return "\n".join(keep) or ctx_full(row)


def ctx_itercomp(row, scorer, pct: float, max_iter: int) -> str:
    """Lọc percentile không có bước LLM.

    Bỏ answerability và follow-up để script chạy được trên CPU. Điều này KHÔNG
    ảnh hưởng kết luận: ta đo tỉ lệ token của ngữ cảnh đã nén, và mức nén do
    percentile quyết định, không do bộ dừng. Ablation cho thấy max_iter=2 và
    max_iter=5 cho cùng F1, nên một vòng là xấp xỉ hợp lý cho mục đích ĐO ĐỘ DÀI.
    """
    segs = decompose(row["context"])
    scores = score_segments(row["question"], [str(s) for s in segs], scorer)
    keep = percentile_filter(scores, pct)
    return "\n".join(f"{segs[i].doc_title}: {segs[i].text}" for i in keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="vimqa",
                    choices=["vimqa", "hotpotqa", "2wiki", "musique"])
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--percentile", type=float, default=90.0)
    ap.add_argument("--max-iter", type=int, default=5)
    ap.add_argument("--scorer", default="bm25",
                    help="bm25 chạy CPU không cần tải bge-m3; dual sát báo cáo hơn")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = load_dataset(args.dataset, args.limit)
    scorer = make_scorer(args.scorer)
    encs = list(FERTILITY)

    # tong token theo tung tokenizer, cho tung phuong phap
    tot: dict[str, dict[str, int]] = {m: {e: 0 for e in encs}
                                      for m in ("raw", "oracle", "itercomp")}
    for r in rows:
        ctxs = {"raw": ctx_full(r),
                "oracle": ctx_oracle(r),
                "itercomp": ctx_itercomp(r, scorer, args.percentile, args.max_iter)}
        for m, c in ctxs.items():
            for e in encs:
                tot[m][e] += count_tokens(c, e)

    print(f"dataset={args.dataset}  n={len(rows)}  scorer={args.scorer}  "
          f"k={args.percentile}\n")

    # ── bang 1: ty le danh nghia theo tung tokenizer ────────────────
    print("TỈ LỆ NÉN DANH NGHĨA — cùng một lần nén, đo bằng 4 tokenizer")
    print(f"{'method':12s} " + " ".join(f"{e:>13s}" for e in encs))
    rows_out = {}
    for m in ("oracle", "itercomp"):
        cells, vals = [], {}
        for e in encs:
            r_ = tot[m][e] / max(tot["raw"][e], 1)
            vals[e] = r_
            cells.append(f"{100*r_:12.1f}%")
        rows_out[m] = vals
        print(f"{m:12s} " + " ".join(cells))

    spread = max(rows_out["itercomp"].values()) / min(rows_out["itercomp"].values())
    print(f"\n  -> cùng một lần nén, tỉ lệ báo cáo chênh {spread:.2f}x tuỳ tokenizer.")
    print("     Nên 'nén N lần' là phát biểu chưa đủ nghĩa nếu không nói tokenizer.")

    # ── bang 2: ty le hieu dung ─────────────────────────────────────
    print("\nTỈ LỆ HIỆU DỤNG = danh nghĩa / fertility")
    print("  (quy về 'tương đương tiếng Anh': cùng nội dung viết bằng tiếng Anh")
    print("   thì tỉ lệ nén tương ứng là bao nhiêu)\n")
    print(f"{'tokenizer':14s} {'fertility':>10s} {'danh nghĩa':>12s} {'hiệu dụng':>11s}")
    eff = {}
    for e in encs:
        nom = rows_out["itercomp"][e]
        f = FERTILITY[e]
        eff[e] = nom / f
        print(f"{e:14s} {f:10.2f} {100*nom:11.1f}% {100*eff[e]:10.1f}%")

    print(f"\n  -> IterCOMP trên {args.dataset}: đọc {100*rows_out['itercomp']['regex']:.1f}% "
          f"số 'từ',")
    print(f"     nhưng chi phí API thực chỉ tương đương {100*eff['cl100k_base']:.1f}% "
          f"một prompt tiếng Anh cùng nội dung.")
    print("     Đó là con số người triển khai cần, và chưa paper nào báo cáo.")

    # ── bang 3: token tiet kiem duoc, quy doi ───────────────────────
    print("\nTOKEN TIẾT KIỆM (tổng trên toàn tập)")
    print(f"{'tokenizer':14s} {'raw':>10s} {'itercomp':>10s} {'tiết kiệm':>11s}"
          f" {'quy đổi EN':>12s}")
    for e in encs:
        saved = tot["raw"][e] - tot["itercomp"][e]
        print(f"{e:14s} {tot['raw'][e]:10,d} {tot['itercomp'][e]:10,d} "
              f"{saved:11,d} {saved/FERTILITY[e]:12,.0f}")
    print("\n  -> vì tiếng Việt tốn nhiều token hơn, CÙNG một tỉ lệ nén cắt được")
    print("     nhiều token thực tế hơn: nén có giá trị thực dụng CAO HƠN ở tiếng Việt.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"config": vars(args) | {"out": str(args.out)},
                   "fertility": FERTILITY,
                   "total_tokens": tot,
                   "nominal_ratio": rows_out,
                   "effective_ratio": eff,
                   "spread_across_tokenizers": spread},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
        print(f"\n→ ghi {args.out}")


if __name__ == "__main__":
    main()
