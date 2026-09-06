"""Khi nào nén VƯỢT gold context — và vì sao điều đó xảy ra cả khi còn dấu.

PHÁT HIỆN DẪN TỚI SCRIPT NÀY. EXP-2 cho thấy trên văn bản bỏ dấu, IterCOMP vượt
gold-context 3.78–5.82 F1, và ta giải thích bằng khử mơ hồ từ vựng. Nhưng đếm
trên `vimqa_full_7b.json` — văn bản CÒN NGUYÊN DẤU — thì IterCOMP vẫn thắng
Oracle ở **9.8%** câu hỏi.

Nên "nén vượt gold" KHÔNG phải hiện tượng riêng của văn bản mất dấu. Bỏ dấu chỉ
làm nó mạnh lên. Điều đó vừa làm yếu vừa làm mạnh lập luận ở EXP-2:

  yếu đi:  khử mơ hồ không phải cơ chế duy nhất, vì có dấu vẫn xảy ra
  mạnh lên: hiện tượng nền tồn tại thật, không phải tạo tác của việc gỡ dấu

Script này phân loại 98 câu đó để biết cơ chế nào đang hoạt động.

GIẢ THUYẾT CẦN PHÂN BIỆT:

  A. Oracle thiếu bằng chứng — `supporting_facts` không đủ trả lời, nên câu vàng
     tự nó đã không đủ. Đây là lỗi NHÃN, không phải công của nén.
  B. Ngữ cảnh rộng giúp reader — nén giữ thêm câu và chính chúng cứu câu trả lời.
  C. Nhiễu có lợi — reader trả lời đúng nhờ may, không nhờ bằng chứng.

Phân biệt A với B bằng độ dài: nếu là A thì gold-context rất ngắn (1–2 câu).

Chạy CPU, chỉ đọc `results/`.

    python scripts/beats_oracle_analysis.py
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-json", type=Path,
                    default=ROOT / "results" / "vimqa_full_7b.json")
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ev = json.loads(args.eval_json.read_text())
    pr = ev["per_row"]

    def f1(r, m):
        return 100 * r["methods"][m]["f1_norm"]

    win = [r for r in pr if f1(r, "itercomp") > f1(r, "oracle")]
    lose = [r for r in pr if f1(r, "itercomp") < f1(r, "oracle")]
    tie = len(pr) - len(win) - len(lose)

    print(f"eval={args.eval_json.name}  n={len(pr)}  reader="
          f"{ev['config']['reader_model'].split('/')[-1]}\n")
    print(f"{'nhóm':22s} {'n':>5s} {'%':>6s} {'F1 IterCOMP':>12s} {'F1 Oracle':>10s}")
    for lab, g in (("IterCOMP > Oracle", win), ("IterCOMP < Oracle", lose)):
        print(f"{lab:22s} {len(g):5d} {100*len(g)/len(pr):5.1f}% "
              f"{st.mean(f1(r,'itercomp') for r in g):11.1f} "
              f"{st.mean(f1(r,'oracle') for r in g):10.1f}")
    print(f"{'bằng nhau':22s} {tie:5d} {100*tie/len(pr):5.1f}%")

    # ── giả thuyết A: gold-context quá ngắn? ────────────────────────
    # Số câu gold là proxy cho lượng bằng chứng nhãn cấp. Nếu nhóm thắng có ít
    # câu gold hơn hẳn, nguyên nhân là NHÃN THIẾU, không phải nén giỏi.
    from itercomp import load_dataset
    rows = load_dataset(ev["config"]["dataset"], len(pr))
    n_gold = {}
    for r_ev, r_ds in zip(pr, rows):
        if r_ev["question"].strip() != r_ds["question"].strip():
            raise SystemExit("lệch thứ tự câu hỏi — không ghép được")
        n_gold[r_ev["question"]] = len(
            (r_ds.get("supporting_facts") or {}).get("title", []))

    print("\nGIẢ THUYẾT A — Oracle thắng ít vì NHÃN thiếu bằng chứng?")
    print(f"{'nhóm':22s} {'số câu gold tb':>16s}")
    gw = [n_gold[r["question"]] for r in win]
    gl = [n_gold[r["question"]] for r in lose]
    print(f"{'IterCOMP > Oracle':22s} {st.mean(gw):15.2f}")
    print(f"{'IterCOMP < Oracle':22s} {st.mean(gl):15.2f}")
    if st.mean(gw) < st.mean(gl) - 0.15:
        print("  -> Nhóm thắng có ÍT câu gold hơn. Ủng hộ A: nén thắng phần lớn")
        print("     ở những câu mà nhãn vàng vốn đã thiếu bằng chứng.")
    elif st.mean(gw) > st.mean(gl) + 0.15:
        print("  -> Nhóm thắng có NHIỀU câu gold hơn. NGƯỢC A.")
    else:
        print("  -> Số câu gold gần như bằng nhau. A không giải thích được;")
        print("     nghiêng về B (ngữ cảnh rộng giúp reader).")

    # ── giả thuyết C: nén thắng nhờ may? ────────────────────────────
    # Nếu là may thì F1 nhóm thắng phải thấp và rải rác. Nếu nén thật sự cứu
    # được câu trả lời thì F1 phải cao.
    perfect = sum(1 for r in win if f1(r, "itercomp") >= 99)
    zero_or = sum(1 for r in win if f1(r, "oracle") <= 1)
    print("\nGIẢ THUYẾT C — nén thắng nhờ may?")
    print(f"  IterCOMP đạt F1 = 100 : {perfect}/{len(win)} "
          f"({100*perfect/len(win):.0f}%)")
    print(f"  Oracle  đạt F1 = 0   : {zero_or}/{len(win)} "
          f"({100*zero_or/len(win):.0f}%)")
    if perfect / len(win) > 0.5:
        print("  -> Quá nửa là F1 tuyệt đối, không phải điểm rải rác. NGƯỢC C:")
        print("     nén cứu được câu trả lời, không phải đoán đúng tình cờ.")
    else:
        print("  -> Phần lớn không tuyệt đối; không loại được C.")

    # ── ví dụ cụ thể ───────────────────────────────────────────────
    print(f"\n{'='*72}\nCASE STUDY — {args.examples} câu IterCOMP thắng rõ nhất\n")
    best = sorted(win, key=lambda r: -(f1(r, "itercomp") - f1(r, "oracle")))
    for i, r in enumerate(best[:args.examples], 1):
        print(f"[{i}] {r['question'][:105]}")
        print(f"    vàng           : {str(r['gold'])[:70]}")
        print(f"    IterCOMP       : {str(r['methods']['itercomp']['pred'])[:70]}"
              f"  (F1 {f1(r,'itercomp'):.0f})")
        print(f"    Oracle         : {str(r['methods']['oracle']['pred'])[:70]}"
              f"  (F1 {f1(r,'oracle'):.0f})")
        print(f"    số câu gold    : {n_gold[r['question']]}")
        print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"config": vars(args) | {"out": str(args.out),
                                           "eval_json": str(args.eval_json)},
                   "n_total": len(pr), "n_win": len(win), "n_lose": len(lose),
                   "pct_win": 100 * len(win) / len(pr),
                   "gold_count_win": st.mean(gw), "gold_count_lose": st.mean(gl),
                   "win_f1_perfect": perfect, "win_oracle_zero": zero_or,
                   "examples": [{
                       "question": r["question"], "gold": r["gold"],
                       "pred_itercomp": r["methods"]["itercomp"]["pred"],
                       "pred_oracle": r["methods"]["oracle"]["pred"],
                       "f1_itercomp": f1(r, "itercomp"),
                       "f1_oracle": f1(r, "oracle"),
                       "n_gold_sentences": n_gold[r["question"]],
                   } for r in best[:args.examples]]},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
        print(f"→ ghi {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
