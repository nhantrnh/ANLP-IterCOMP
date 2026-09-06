"""Quét ngưỡng dừng: đo đường đánh đổi nén-độ chính xác.

Bài báo dùng một quyết định nhị phân, tức một điểm duy nhất trên đường cong.
Script này vẽ cả đường: mỗi ngưỡng cho một cặp (tỉ lệ nén, F1), nên có thể
chọn điểm vận hành thay vì nhận điểm mà LLM tình cờ đưa ra.

So ba chế độ:
  paper     quyết định nhị phân is_answerable (nguyên bản)
  conf-T    dừng khi tin cậy >= T, với T quét trong một dải
  gain-G    dừng khi vòng thêm ít hơn G tỉ lệ token mới (KHÔNG gọi LLM)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # để chạy được cả khi scripts/ nằm trong src/

from itercomp import (itercomp, load_dataset, make_llm, make_reader,  # noqa: E402
                      make_scorer, clean_answer, f1_score, exact_match,
                      normalize_boolean, tokenize)


def run_mode(rows, llm, reader, scorer, pct, max_iter, **stop_kw):
    """Chạy một chế độ dừng trên toàn bộ rows, trả về số tổng hợp."""
    f1s, ems, ratios, iters = [], [], [], []
    for r in rows:
        res = itercomp(llm, r["question"], r["context"], max_iter=max_iter,
                       scorer=scorer, percentile=pct, **stop_kw)
        full = "\n".join(f"{t}: {' '.join(s)}" for t, s
                         in zip(r["context"]["title"], r["context"]["sentences"]))
        n_full = max(len(tokenize(full)), 1)
        ratios.append(len(tokenize(res.to_prompt())) / n_full)
        iters.append(res.iterations)

        pred = clean_answer(reader(res.to_prompt(), r["question"]))
        gold = r["answer"]
        pn = normalize_boolean(pred, gold)
        f1s.append(f1_score(pn, gold) * 100)
        ems.append(exact_match(pn, gold) * 100)

    n = len(rows)
    return {"f1": sum(f1s) / n, "em": sum(ems) / n,
            "ratio": sum(ratios) / n, "iters": sum(iters) / n,
            "f1_per_row": f1s}


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
    ap.add_argument("--thresholds", default="0.3,0.5,0.7,0.9",
                    help="dải ngưỡng tin cậy")
    ap.add_argument("--gains", default="0.1,0.2,0.3",
                    help="dải ngưỡng lợi ích cận biên")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.reader == "hf" and not args.reader_model:
        ap.error("--reader hf cần --reader-model")

    kw = {"model": args.reader_model} if args.reader_model else {}
    if args.load_4bit and args.reader == "hf":
        kw["load_4bit"] = True
    reader = make_reader(args.reader, **kw)
    # Dùng chung một mô hình cho reader và suy luận, như bài báo
    llm = (make_llm("hf", model=args.reader_model, load_4bit=args.load_4bit)
           if args.reader == "hf" else make_llm("mock"))
    scorer = make_scorer(args.scorer)
    rows = load_dataset(args.dataset, args.limit)

    print(f"dataset={args.dataset} n={len(rows)} reader={args.reader}"
          f"{'/' + args.reader_model if args.reader_model else ''}")
    print(f"percentile={args.percentile} max_iter={args.max_iter}")
    if not hasattr(llm, "choose_prob"):
        print("CẢNH BÁO: backend không cho đọc logit, nên điểm tin cậy chỉ nhận"
              " 0 hoặc 1.\n  Mọi ngưỡng conf-* sẽ cho cùng kết quả — chạy với"
              " --reader hf để quét thật.")
    print()

    modes = [("paper", {})]
    modes += [(f"conf-{t}", {"stop_threshold": float(t)})
              for t in args.thresholds.split(",")]
    modes += [(f"gain-{g}", {"stop_min_gain": float(g)})
              for g in args.gains.split(",")]

    results = {}
    print(f"{'chế độ':12s}{'F1':>7}{'EM':>7}{'tỉ lệ':>8}{'vòng':>7}{'phút':>7}")
    print("-" * 48)
    for name, kwargs in modes:
        t0 = time.time()
        r = run_mode(rows, llm, reader, scorer, args.percentile,
                     args.max_iter, **kwargs)
        results[name] = r
        print(f"{name:12s}{r['f1']:7.1f}{r['em']:7.1f}"
              f"{r['ratio']*100:7.1f}%{r['iters']:7.2f}{(time.time()-t0)/60:7.1f}",
              flush=True)

    # ── điểm nào trên biên Pareto? ─────────────────────────────────
    print("\nBiên Pareto (không có chế độ nào vừa F1 cao hơn vừa nén mạnh hơn):")
    for name, r in results.items():
        dominated = any(o["f1"] >= r["f1"] and o["ratio"] <= r["ratio"]
                        and (o["f1"] > r["f1"] or o["ratio"] < r["ratio"])
                        for k, o in results.items() if k != name)
        if not dominated:
            print(f"  {name:12s} F1={r['f1']:5.1f}  tỉ lệ={r['ratio']*100:5.1f}%")

    # ── so với paper, có ý nghĩa thống kê không? ───────────────────
    from itercomp import paired_bootstrap
    base = results["paper"]["f1_per_row"]
    print("\nSo với chế độ paper (bootstrap ghép cặp):")
    for name, r in results.items():
        if name == "paper":
            continue
        t = paired_bootstrap(r["f1_per_row"], base)
        sig = "có ý nghĩa" if t["p"] < 0.05 else "chưa đủ bằng chứng"
        print(f"  {name:12s}{t['diff']:+7.1f}  [{t['lo']:+6.1f},{t['hi']:+6.1f}]"
              f"  p={t['p']:.4f}  {sig}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"config": vars(args) | {"out": str(args.out)},
                   "results": {k: {kk: vv for kk, vv in v.items()
                                   if kk != "f1_per_row"}
                               for k, v in results.items()}},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
        print(f"\n→ ghi {args.out}")


if __name__ == "__main__":
    main()
