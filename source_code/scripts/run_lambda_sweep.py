"""Quét λ và k riêng cho tiếng Việt — cải tiến số 3.

Bài báo gốc chọn λ=0.6 và k=90 tối ưu cho tiếng Anh. Câu hỏi: các giá trị đó có
chuyển được sang tiếng Việt không?

Cơ sở để nghi ngờ:
  - Wullach et al. (2025) cho thấy ngưỡng liên quan tối ưu phụ thuộc mức tài
    nguyên của ngôn ngữ: ngôn ngữ ít tài nguyên ưa ngưỡng THẤP hơn (bao dung hơn).
  - Litschko et al. (IRJ 2022) ghi nhận mô hình huấn luyện trên tiếng Anh bị
    "quá khớp đơn ngữ" với đặc trưng khớp từ vựng chính xác.
  - Tiếng Việt có fertility ~2x nên phân bố điểm khác, kéo theo ngưỡng phân vị
    tương ứng với lượng nội dung khác.
Chưa có công trình nào quét λ cho hợp nhất dense/sparse theo từng ngôn ngữ.

Chạy:
    python scripts/run_lambda_sweep.py --dataset vimqa --limit 100 --reader hf
    python scripts/run_lambda_sweep.py --dataset hotpotqa --limit 100 --reader hf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from itercomp import (clean_answer, exact_match, f1_score, itercomp,
                      load_dataset, make_llm, make_reader, make_scorer,
                      normalize_boolean, tokenize)
from itercomp.scorer import PERCENTILE_BY_DATASET

LAMBDAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)   # 0.6 là giá trị bài báo dùng
PERCENTILES = (70.0, 80.0, 85.0, 90.0, 95.0)


def evaluate(rows, llm, reader, scorer, percentile, max_iter):
    """Trả về EM/F1 (đã chuẩn hoá boolean) và tỉ lệ nén trung bình."""
    ems, f1s, ratios = [], [], []
    for row in rows:
        res = itercomp(llm, row["question"], row["context"],
                       max_iter=max_iter, scorer=scorer, percentile=percentile)
        full = "\n".join(
            f"{t}: {' '.join(s)}"
            for t, s in zip(row["context"]["title"], row["context"]["sentences"]))
        n_o = len(tokenize(full)) or 1
        ratios.append(len(tokenize(res.to_prompt())) / n_o)
        if reader is not None:
            gold = row["answer"]
            pred = normalize_boolean(clean_answer(reader(res.to_prompt(),
                                                         row["question"])), gold)
            ems.append(exact_match(pred, gold))
            f1s.append(f1_score(pred, gold))
    n = len(rows)
    return {
        "em": 100 * sum(ems) / n if ems else None,
        "f1": 100 * sum(f1s) / n if f1s else None,
        "ratio": sum(ratios) / n,
    }


def main():
    ap = argparse.ArgumentParser(description="Quét λ và k theo ngôn ngữ")
    ap.add_argument("--dataset", default="vimqa",
                    choices=["vimqa", "hotpotqa", "2wiki", "musique"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--reader", default="hf",
                    choices=["mock", "hf", "openai", "openrouter", "gemini"])
    ap.add_argument("--reader-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--encoder", default="BAAI/bge-m3")
    ap.add_argument("--llm", default="mock", choices=["mock", "openai", "openrouter"])
    ap.add_argument("--max-iter", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = load_dataset(args.dataset, args.limit)
    llm = make_llm(args.llm)
    reader = None if args.reader == "mock" else make_reader(
        args.reader, model=args.reader_model)
    base_pct = PERCENTILE_BY_DATASET.get(args.dataset, 90.0)

    print(f"dataset={args.dataset}  n={len(rows)}  reader={args.reader}")
    print(f"k của bài báo cho bộ này = {base_pct}\n")

    out = {"config": vars(args) | {"out": str(args.out)}, "lambda": {}, "percentile": {}}

    # ── trục λ, giữ k theo bài báo
    print(f"── Quét λ (k={base_pct} cố định)")
    print(f"{'λ':>5}{'EM':>8}{'F1':>8}{'nén':>9}")
    for lam in LAMBDAS:
        sc = make_scorer("dual", encoder=args.encoder, lam=lam)
        r = evaluate(rows, llm, reader, sc, base_pct, args.max_iter)
        out["lambda"][str(lam)] = r
        f1 = f"{r['f1']:8.1f}" if r["f1"] is not None else f"{'—':>8}"
        em = f"{r['em']:8.1f}" if r["em"] is not None else f"{'—':>8}"
        print(f"{lam:>5.1f}{em}{f1}{r['ratio']:>8.1%}", flush=True)

    # ── trục k, giữ λ theo bài báo
    print(f"\n── Quét k (λ=0.6 cố định, giá trị bài báo)")
    print(f"{'k':>5}{'EM':>8}{'F1':>8}{'nén':>9}")
    sc = make_scorer("dual", encoder=args.encoder, lam=0.6)
    for pct in PERCENTILES:
        r = evaluate(rows, llm, reader, sc, pct, args.max_iter)
        out["percentile"][str(pct)] = r
        f1 = f"{r['f1']:8.1f}" if r["f1"] is not None else f"{'—':>8}"
        em = f"{r['em']:8.1f}" if r["em"] is not None else f"{'—':>8}"
        print(f"{pct:>5.0f}{em}{f1}{r['ratio']:>8.1%}", flush=True)

    # ── kết luận: giá trị tối ưu có khác bài báo không?
    if reader is not None:
        best_lam = max(out["lambda"], key=lambda k: out["lambda"][k]["f1"])
        best_pct = max(out["percentile"], key=lambda k: out["percentile"][k]["f1"])
        print(f"\n{'='*54}")
        print(f"λ tối ưu = {best_lam}  (bài báo: 0.6)"
              f"{'  -> KHÁC' if float(best_lam) != 0.6 else '  -> khớp'}")
        print(f"k tối ưu = {best_pct}  (bài báo: {base_pct})"
              f"{'  -> KHÁC' if float(best_pct) != base_pct else '  -> khớp'}")
        out["best"] = {"lambda": float(best_lam), "percentile": float(best_pct),
                       "paper_lambda": 0.6, "paper_percentile": base_pct}

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
        print(f"\n→ ghi {args.out}")


if __name__ == "__main__":
    main()
