"""Đo tỉ lệ DỪNG SAI của bộ kiểm tra đủ bằng chứng.

Câu hỏi: khi is_answerable() trả True, bằng chứng đã gom có thật sự chứa
đáp án chưa? Bốn trường hợp:

  dừng-đúng   : dừng, và bằng chứng vàng đã nằm trong tập đã chọn
  DỪNG-SỚM    : dừng, nhưng bằng chứng vàng CHƯA có  <- lỗi tốn kém nhất
  chạy-đúng   : chưa dừng, và bằng chứng vàng chưa đủ
  chạy-thừa   : chưa dừng, dù bằng chứng vàng đã có   <- tốn token vô ích

Bài báo không báo cáo phân tách này; họ chỉ báo F1 cuối cùng, nên không thấy
được cơ chế dừng sớm đang mất bao nhiêu điểm.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # để chạy được cả khi scripts/ nằm trong src/

from itercomp import load_dataset, make_llm, make_scorer          # noqa: E402
from itercomp.core import decompose, is_answerable, make_followup, score_segments  # noqa: E402
from itercomp.scorer import percentile_filter                      # noqa: E402


def gold_sentences(row: dict) -> set[str]:
    """Tập câu bằng chứng vàng, chuẩn hoá để so khớp."""
    sf = row.get("supporting_facts") or {}
    titles, sent_ids = sf.get("title", []), sf.get("sent_id", [])
    ctx = row["context"]
    out = set()
    for t, sid in zip(titles, sent_ids):
        try:
            i = list(ctx["title"]).index(t)
            out.add(" ".join(ctx["sentences"][i][sid].split()))
        except (ValueError, IndexError):
            continue
    return out


def covered(selected, gold: set[str]) -> bool:
    """Tập đã chọn có phủ HẾT bằng chứng vàng chưa?

    Chỉ tính là phủ khi câu vàng nằm TRỌN trong một segment đã chọn. Cho phép
    chiều ngược lại (segment nằm trong câu vàng) sẽ tính "Album được phát hành"
    là đã phủ "Album được phát hành vào ngày 29 tháng 3 năm 2019" -- tức mất
    đúng phần mang đáp án, mà vẫn báo là đủ.
    """
    if not gold:
        return True          # không có nhãn vàng -> không tính được, coi như đủ
    have = [" ".join(str(s).split()) for s in selected]
    return all(any(g in h for h in have) for g in gold)


def trace_one(llm, row, scorer, percentile, max_iter):
    """Chạy vòng lặp, ghi lại quyết định dừng ở TỪNG vòng."""
    segs = decompose(row["context"])
    gold = gold_sentences(row)
    remaining, selected, query = list(range(len(segs))), [], row["question"]
    events = []

    for it in range(1, max_iter + 1):
        if not remaining:
            break
        scores = score_segments(query, [segs[i] for i in remaining], scorer)
        local = percentile_filter(scores, percentile)
        picked = [remaining[j] for j in local]
        selected.extend(segs[i] for i in picked)
        remaining = [i for i in remaining if i not in set(picked)]

        has_gold = covered(selected, gold)
        says_stop = is_answerable(llm, row["question"], selected)
        events.append({"iter": it, "stop": says_stop, "gold_covered": has_gold})
        if says_stop:
            break
        query = make_followup(llm, row["question"], selected)

    return events, bool(gold)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="vimqa",
                    choices=["vimqa", "hotpotqa", "2wiki", "musique"])
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--llm", required=True, choices=["mock", "hf", "openai", "gemini"])
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--scorer", default="dual")
    ap.add_argument("--percentile", type=float, default=90.0)
    ap.add_argument("--max-iter", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.llm == "hf" and not args.llm_model:
        ap.error("--llm hf cần --llm-model")

    kw = {"model": args.llm_model, "load_4bit": args.load_4bit} if args.llm == "hf" else {}
    llm = make_llm(args.llm, **kw)
    scorer = make_scorer(args.scorer)
    rows = load_dataset(args.dataset, args.limit)

    tally = {"stop_correct": 0, "STOP_EARLY": 0, "run_correct": 0, "run_wasteful": 0}
    no_gold = 0
    per_row = []

    for i, row in enumerate(rows, 1):
        events, had_gold = trace_one(llm, row, scorer, args.percentile, args.max_iter)
        if not had_gold:
            no_gold += 1
            continue
        for e in events:
            if e["stop"]:
                tally["stop_correct" if e["gold_covered"] else "STOP_EARLY"] += 1
            else:
                tally["run_wasteful" if e["gold_covered"] else "run_correct"] += 1
        per_row.append({"question": row["question"], "events": events})
        if i % 10 == 0:
            print(f"  [{i}/{len(rows)}]", flush=True)

    total = sum(tally.values()) or 1
    print(f"\n{'='*66}")
    print(f"QUYẾT ĐỊNH DỪNG — {args.dataset}, n={len(per_row)}, llm={args.llm}")
    print(f"{'='*66}")
    for k, v in tally.items():
        mark = "  <-- lỗi tốn kém nhất" if k == "STOP_EARLY" else ""
        print(f"  {k:14s}{v:5d}{v/total:8.1%}{mark}")
    if no_gold:
        print(f"  ({no_gold} câu không có nhãn bằng chứng vàng, bỏ qua)")

    stops = tally["stop_correct"] + tally["STOP_EARLY"]
    if stops:
        print(f"\n  Trong các lần DỪNG: {tally['STOP_EARLY']/stops:.1%} là dừng SỚM"
              f" (bằng chứng vàng chưa đủ)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"config": vars(args) | {"out": str(args.out)},
                   "tally": tally, "n_no_gold": no_gold, "per_row": per_row},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
        print(f"\n→ ghi {args.out}")


if __name__ == "__main__":
    main()
