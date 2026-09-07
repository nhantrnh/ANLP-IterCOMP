"""EXP-1: trần của việc dừng đúng lúc.

Câu hỏi: nếu bộ dừng HOÀN HẢO — dừng đúng vòng mà bằng chứng vàng vừa đủ — thì
được thêm bao nhiêu điểm F1 so với luật dừng của bài báo?

Đây là cửa chặn của cả hướng nghiên cứu. Nếu trần chỉ 2-3 điểm thì không có gì
để cải thiện, và mọi thí nghiệm sau (đa ngôn ngữ, công thức ngưỡng) đều vô nghĩa.
Chạy nó TRƯỚC khi xây bất cứ thứ gì khác.

LƯU Ý VỀ ĐỌC KẾT QUẢ. "oracle" dừng khi bằng chứng vàng đã ĐỦ, nên nó giữ
NHIỀU token hơn "paper" (paper dừng sớm hơn). Vì thế chênh lệch F1 giữa hai chế
độ KHÔNG phải hiệu ứng thuần của thời điểm dừng: một phần đến từ việc oracle
được đọc nhiều bằng chứng hơn. Trần đo được ở đây là trần LẠC QUAN. Muốn tách
hai yếu tố, phải so ở cùng mức tỉ lệ nén (điều chỉnh percentile cho paper tới
khi tỉ lệ khớp oracle) -- chưa làm trong script này.

Bốn chế độ trên cùng tập câu hỏi:
  paper    luật dừng nhị phân của bài báo
  oracle   dừng ngay vòng đầu tiên mà bằng chứng vàng đã đủ  <- TRẦN
  never    không bao giờ dừng, chạy hết max_iter             <- sàn về nén
  gold     chỉ đưa đúng câu bằng chứng vàng                  <- trần tuyệt đối
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from itercomp import (load_dataset, make_llm, make_reader, make_scorer,  # noqa: E402
                      clean_answer, f1_score, exact_match, normalize_boolean,
                      tokenize, paired_bootstrap)
from itercomp.core import decompose, is_answerable, make_followup, score_segments  # noqa: E402
from itercomp.scorer import percentile_filter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_stop_error import gold_sentences, covered  # noqa: E402


def gold_context(row) -> str:
    """Chỉ các câu bằng chứng vàng, cùng quy ước "tiêu đề: câu" như trên."""
    sf = row.get("supporting_facts") or {}
    want = set(zip(sf.get("title", []), sf.get("sent_id", [])))
    keep = [f"{t}: {s}"
            for t, sents in zip(row["context"]["title"], row["context"]["sentences"])
            for i, s in enumerate(sents) if (t, i) in want]
    return "\n".join(keep)


def run_loop(llm, row, scorer, pct, max_iter, mode: str):
    """Chạy vòng lặp với một trong bốn chế độ dừng. Trả về (context, số vòng)."""
    segs = decompose(row["context"])
    gold = gold_sentences(row)
    remaining, selected, query = list(range(len(segs))), [], row["question"]
    iters = 0

    for it in range(1, max_iter + 1):
        iters = it
        if not remaining:
            break
        scores = score_segments(query, [segs[i] for i in remaining], scorer)
        picked = [remaining[j] for j in percentile_filter(scores, pct)]
        selected.extend(segs[i] for i in picked)
        remaining = [i for i in remaining if i not in set(picked)]

        if mode == "oracle":
            # Dừng ngay khi bằng chứng vàng đã đủ. Đây là thông tin ORACLE
            # (nhìn nhãn), không dùng được lúc suy luận -- chỉ để đo trần.
            if covered(selected, gold):
                break
        elif mode == "paper":
            if is_answerable(llm, row["question"], selected):
                break
            query = make_followup(llm, row["question"], selected)
        elif mode == "never":
            query = make_followup(llm, row["question"], selected)
        else:
            raise ValueError(mode)

    return "\n".join(f"{s.doc_title}: {s.text}" for s in selected), iters


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="vimqa",
                    choices=["vimqa", "hotpotqa", "2wiki", "musique"])
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--reader", required=True, choices=["mock", "hf"])
    ap.add_argument("--reader-model", default=None)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--scorer", default="dual")
    ap.add_argument("--percentile", type=float, default=90.0)
    ap.add_argument("--max-iter", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.reader == "hf" and not args.reader_model:
        ap.error("--reader hf cần --reader-model")

    kw = {"model": args.reader_model} if args.reader_model else {}
    if args.load_4bit and args.reader == "hf":
        kw["load_4bit"] = True
    reader = make_reader(args.reader, **kw)
    llm = (make_llm("hf", model=args.reader_model, load_4bit=args.load_4bit)
           if args.reader == "hf" else make_llm("mock"))
    scorer = make_scorer(args.scorer)
    rows = [r for r in load_dataset(args.dataset, args.limit)
            if gold_sentences(r)]          # cần nhãn vàng để đo oracle

    print(f"dataset={args.dataset} n={len(rows)} (chỉ câu có nhãn vàng)")
    print(f"reader={args.reader}{'/' + args.reader_model if args.reader_model else ''}\n")

    out: dict[str, dict] = {}
    for mode in ("paper", "oracle", "never", "gold"):
        f1s, tok_ctx, tok_full, iters = [], [], [], []
        for r in rows:
            # Mẫu số dùng ĐÚNG quy ước của tử số ("tiêu đề: câu" mỗi dòng,
            # như run_eval.ctx_oracle). Nếu mẫu số nối tiêu đề một lần còn tử số
            # lặp lại mỗi câu thì "never" (chọn hết) cho tỉ lệ 107% -- đo lỗi
            # định dạng chứ không đo mức nén. Quy ước này giữ tên thực thể cho
            # reader, vốn cần cho suy luận nhiều bước.
            full = "\n".join(f"{s.doc_title}: {s.text}"
                             for s in decompose(r["context"]))
            if mode == "gold":
                ctx, n_it = gold_context(r), 0
            else:
                ctx, n_it = run_loop(llm, r, scorer, args.percentile,
                                     args.max_iter, mode)
            tok_ctx.append(len(tokenize(ctx)))
            tok_full.append(len(tokenize(full)))
            iters.append(n_it)
            pred = normalize_boolean(clean_answer(reader(ctx, r["question"])),
                                     r["answer"])
            f1s.append(f1_score(pred, r["answer"]) * 100)
        n = len(rows)
        out[mode] = {"f1": sum(f1s) / n,
                     "ratio": sum(tok_ctx) / max(sum(tok_full), 1),
                     "iters": sum(iters) / n, "per_row": f1s}
        print(f"  {mode:8s} F1={out[mode]['f1']:5.1f}  "
              f"tỉ lệ={out[mode]['ratio']*100:5.1f}%  vòng={out[mode]['iters']:.2f}",
              flush=True)

    # ── kết luận cửa chặn ──────────────────────────────────────────
    head = out["oracle"]["f1"] - out["paper"]["f1"]
    t = paired_bootstrap(out["oracle"]["per_row"], out["paper"]["per_row"])
    print(f"\n{'='*62}")
    print(f"TRẦN của việc dừng đúng lúc: {head:+.1f} F1")
    print(f"  bootstrap ghép cặp: [{t['lo']:+.1f}, {t['hi']:+.1f}]  p={t['p']:.4f}")
    print(f"{'='*62}")
    if head < 3.0:
        print("=> TRẦN QUÁ THẤP (< 3 điểm). Cải thiện bộ dừng không đáng làm.")
        print("   Dừng hướng nghiên cứu này ở đây.")
    elif t["p"] >= 0.05:
        print(f"=> Trần {head:+.1f} nhưng KHÔNG có ý nghĩa ở n={len(rows)}.")
        print("   Cần cỡ mẫu lớn hơn trước khi kết luận.")
    else:
        print(f"=> Có {head:+.1f} điểm để giành. Đáng theo tiếp.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"config": vars(args) | {"out": str(args.out)},
                   "results": {k: {kk: vv for kk, vv in v.items() if kk != "per_row"}
                               for k, v in out.items()},
                   "headroom": head, "test": t},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
        print(f"\n→ ghi {args.out}")


if __name__ == "__main__":
    main()
