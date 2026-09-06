"""Nghiên cứu loại bỏ (ablation) cho IterCOMP — sinh dữ liệu cho Mục 5.2 của báo cáo.

Năm trục ablation (khớp với Mục 5.3 + Hình 4 của paper):
    A. max_iter    1..5              paper dùng 5
    B. percentile  70/80/85/90/95    paper dùng k=90 (MuSiQue/HotpotQA), 85 (2Wiki)
    B'. top_k      1,2,3             top-k cố định — ĐỐI CHỨNG với percentile
    C. scorer      bm25/dense/lexical/dual   text encoder có cần thiết không?
    D. lambda      0.0..1.0          paper dùng λ=0.6
    E. bỏ bước     no-answerability / no-followup / no-both

C và D khớp đúng ablation của paper, nơi họ kết luận semantic và lexical BỔ TRỢ
nhau — dùng riêng lẻ đều kém hơn. E cho biết hai bước "reasoning-aware" đóng góp
thật hay lợi ích chỉ đến từ khâu chấm điểm.

⚠️ LƯU Ý PHƯƠNG PHÁP LUẬN: percentile filtering luôn giữ ~(100−k)% segment, nên
đổi λ hay scorer gần như KHÔNG đổi tỉ lệ nén — chỉ đổi segment NÀO được chọn.
Muốn so sánh scorer thì BẮT BUỘC phải có --reader để đo EM/F1.

Chạy:
    # miễn phí, chỉ đo tỉ lệ nén
    python scripts/run_ablation.py --dataset vimqa --limit 30 --llm mock

    # có EM/F1 (bắt buộc để so sánh scorer)
    python scripts/run_ablation.py --dataset vimqa --limit 30 --llm mock \
        --reader hf --out results/ablation_vimqa.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # để chạy được cả khi scripts/ nằm trong src/

from itercomp import (IterCompResult, decompose, load_dataset, make_llm,
                      make_reader, make_scorer, clean_answer,
                      exact_match, f1_score, tokenize, percentile_filter)
from itercomp.core import is_answerable, make_followup, score_segments
from itercomp.scorer import (DEFAULT_LAMBDA, DEFAULT_PERCENTILE,
                             PERCENTILE_BY_DATASET)
import argparse
import json
from pathlib import Path



def itercomp_ablated(
    llm, question: str, context: dict, *,
    max_iter: int = 5, top_k_per_iter: int | None = None,
    percentile: float = DEFAULT_PERCENTILE,
    scorer=None,
    use_answerability: bool = True,
    use_followup: bool = True,
) -> IterCompResult:
    """Bản IterCOMP bật/tắt được từng thành phần — dùng cho mọi trục ablation.

    use_answerability=False  -> bỏ Bước 2, luôn chạy đủ max_iter vòng
    use_followup=False       -> bỏ Bước 3, mọi vòng dùng lại câu hỏi gốc
    top_k_per_iter != None   -> dùng top-k cố định thay percentile filtering
    scorer                   -> đổi cách chấm điểm (bm25/dense/lexical/dual)
    """
    segs = decompose(context)
    res = IterCompResult(question=question, selected=[])
    remaining = list(range(len(segs)))
    query = question
    if scorer is None:
        scorer = make_scorer("dual")

    for it in range(1, max_iter + 1):
        res.iterations = it
        if not remaining:
            res.stopped_because = "hết segment"
            break

        scores = score_segments(query, [segs[i] for i in remaining], scorer)
        if top_k_per_iter is not None:
            order = sorted(range(len(remaining)), key=lambda j: -scores[j])
            local = order[:top_k_per_iter]
        else:
            local = percentile_filter(scores, percentile)
        picked = [remaining[j] for j in local]
        res.selected.extend(segs[i] for i in picked)
        remaining = [i for i in remaining if i not in set(picked)]

        if use_answerability and is_answerable(llm, question, res.selected):
            res.stopped_because = "answerable"
            break

        if use_followup:
            fu = make_followup(llm, question, res.selected)
            res.followups.append(fu)
            query = fu
        # use_followup=False -> query giữ nguyên câu hỏi gốc
    else:
        res.stopped_because = "hết vòng lặp"

    return res


# Mỗi cấu hình: dict các tham số truyền cho itercomp_ablated.
# Nhóm A: số vòng lặp | B: cách lọc | C: cách chấm điểm (trục của paper) |
# D: λ sweep | E: bỏ từng bước reasoning-aware
CONFIGS = (
    [(f"max_iter={m}", dict(max_iter=m)) for m in (1, 2, 3, 4, 5)]
    # paper Hình 4 quét k và kết luận k=90 tốt nhất trên MuSiQue
    + [(f"percentile={p}", dict(percentile=p)) for p in (70.0, 80.0, 85.0, 90.0, 95.0)]
    # k cao: MuSiQue co 20 passage/cau nen k=90 chi giu xuong 0.349, trong khi
    # paper bao 0.14 o cung k. Quet cao hon de tim diem KHOP MUC NEN, roi moi
    # so F1 — cung phuong phap luan bai dung cho ti so hu-tu.
    + [(f"percentile={p}", dict(percentile=p)) for p in (93.0, 96.0, 97.5)]
    + [(f"top_k={k}", dict(top_k_per_iter=k)) for k in (1, 2, 3)]
    # trục quan trọng nhất: text encoder có cần thiết không?
    + [(f"scorer={s}", dict(scorer_kind=s)) for s in ("bm25", "dense", "lexical", "dual")]
    + [(f"lambda={l}", dict(scorer_kind="dual", lam=l))
       for l in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]   # paper dùng λ=0.6
    + [("no-answerability", dict(use_answerability=False)),
       ("no-followup", dict(use_followup=False)),
       ("no-both", dict(use_answerability=False, use_followup=False))]
)


def main():
    ap = argparse.ArgumentParser(description="Ablation cho IterCOMP")
    ap.add_argument("--dataset", default="vimqa",
                    choices=["vimqa", "hotpotqa", "2wiki", "musique"])
    ap.add_argument("--limit", type=int, default=None,
                    help="bỏ trống = chạy TOÀN BỘ tập")
    # 'hf' dùng chính mô hình đọc cho các bước suy luận, như bài báo. Bắt buộc
    # chọn: 'mock' luôn báo "đủ bằng chứng" ngay vòng 1 nên vòng lặp không chạy.
    ap.add_argument("--llm", required=True,
                    choices=["mock", "hf", "openai", "openrouter", "gemini"])
    ap.add_argument("--reader", default=None, choices=[None, "mock", "openai", "openrouter", "gemini", "hf"],
                    help="bỏ trống = không đo EM/F1, chỉ đo nén (miễn phí)")
    ap.add_argument("--reader-model", default=None,
                    help="tên mô hình đọc; bỏ trống thì dùng mặc định của backend")
    ap.add_argument("--load-4bit", action="store_true",
                    help="nạp mô hình đọc ở 4-bit; cần cho 7-8B trên T4 15GB")
    ap.add_argument("--scorer", default="dual", choices=["bm25", "dense", "lexical", "dual"],
                    help="scorer mặc định cho các cấu hình không chỉ định riêng")
    ap.add_argument("--encoder", default="BAAI/bge-m3",
                    help="đổi encoder để so sánh, vd AITeamVN/Vietnamese_Embedding")
    ap.add_argument("--lam", type=float, default=DEFAULT_LAMBDA)
    ap.add_argument("--only", default=None,
                    help="chỉ chạy các cấu hình có tên bắt đầu bằng một trong "
                         "các tiền tố này (phân tách bằng dấu phẩy), vd "
                         "'percentile,max_iter'. Toàn bộ 26 cấu hình ở n=500 "
                         "tốn ~10.5h GPU nên thường phải chọn.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.llm == "hf" and not args.reader_model:
        ap.error("--llm hf cần --reader-model để biết nạp mô hình nào")
    # Truyền model cho backend hf; nó cache theo tên nên dùng lại đúng instance
    # của reader, không nạp thêm mô hình thứ hai vào VRAM.
    lkw = ({"model": args.reader_model, "load_4bit": args.load_4bit}
           if args.llm == "hf" else {})
    llm = make_llm(args.llm, **lkw)
    default_pct = PERCENTILE_BY_DATASET.get(args.dataset, DEFAULT_PERCENTILE)
    rkw = {}
    if args.reader_model and args.reader != "mock":
        rkw["model"] = args.reader_model
    if args.load_4bit and args.reader == "hf":
        rkw["load_4bit"] = True
    reader = make_reader(args.reader, **rkw) if args.reader else None
    rows = load_dataset(args.dataset, args.limit)

    print(f"dataset={args.dataset}  n={len(rows)}  llm={args.llm}  "
          f"reader={args.reader or '(không đo EM/F1)'}"
          f"{'/' + args.reader_model if args.reader_model else ''}"
          f"{' 4bit' if args.load_4bit else ''}  encoder={args.encoder}")
    if args.llm == "mock":
        print("  CẢNH BÁO: llm=mock -> vòng lặp dừng ngay vòng 1,"
              " KHÔNG so được với bài báo")
    print()

    configs = CONFIGS
    if args.only:
        pref = tuple(x.strip() for x in args.only.split(",") if x.strip())
        configs = [(n, c) for n, c in CONFIGS if n.startswith(pref)]
        if not configs:
            ap.error(f"--only {args.only!r} không khớp cấu hình nào. "
                     f"Có: {sorted({n.split('=')[0] for n, _ in CONFIGS})}")
        print(f"--only {args.only}: chạy {len(configs)}/{len(CONFIGS)} cấu hình "
              f"({', '.join(n for n, _ in configs)})\n")

    table = {}
    scorer_cache = {}
    for name, cfg in configs:
        cfg = dict(cfg)
        # tạo (và cache) scorer theo cấu hình
        sk = cfg.pop("scorer_kind", args.scorer)
        lam = cfg.pop("lam", args.lam)
        key = (sk, args.encoder, lam)
        if key not in scorer_cache:
            scorer_cache[key] = make_scorer(sk, encoder=args.encoder, lam=lam)
        cfg["scorer"] = scorer_cache[key]
        cfg.setdefault("percentile", default_pct)

        f1s, ems, ratios, iters, n_segs = [], [], [], [], []

        for row in rows:
            res = itercomp_ablated(
                llm, row["question"], row["context"], **cfg,
            )
            full = "\n".join(
                f"{t}: {' '.join(s)}"
                for t, s in zip(row["context"]["title"], row["context"]["sentences"])
            )
            n_orig = len(tokenize(full)) or 1
            ratios.append(len(tokenize(res.to_prompt())) / n_orig)
            iters.append(res.iterations)
            n_segs.append(len(res.selected))

            if reader:
                pred = clean_answer(reader(res.to_prompt(), row["question"]))
                f1s.append(f1_score(pred, row["answer"]))
                ems.append(exact_match(pred, row["answer"]))

        n = len(rows)
        table[name] = {
            "ratio": sum(ratios) / n,
            "iters": sum(iters) / n,
            "segments": sum(n_segs) / n,
            "em": 100 * sum(ems) / n if ems else None,
            "f1": 100 * sum(f1s) / n if f1s else None,
        }
        r = table[name]
        line = (f"{name:<20} nén={r['ratio']:>6.1%}  vòng={r['iters']:>4.2f}  "
                f"seg={r['segments']:>4.1f}")
        if r["f1"] is not None:
            line += f"  EM={r['em']:>5.1f}  F1={r['f1']:>5.1f}"
        print(line)

    # ─────────────────────────────────────── bảng LaTeX dán thẳng vào báo cáo
    print(f"\n{'='*70}\nLaTeX (dán vào Mục 5.2 của report.tex)\n{'='*70}")
    has_f1 = any(v["f1"] is not None for v in table.values())
    cols = "lrrr" + ("rr" if has_f1 else "")
    print(f"\\begin{{tabular}}{{{cols}}}")
    print("\\toprule")
    hdr = "Cấu hình & Tỉ lệ nén & Số vòng & \\#segment"
    if has_f1:
        hdr += " & EM & F1"
    print(hdr + " \\\\")
    print("\\midrule")
    for name, v in table.items():
        safe = name.replace("_", r"\_")   # escape cho LaTeX (f-string không cho backslash)
        row = (f"{safe} & {v['ratio']*100:.1f}" + r"\% & "
               + f"{v['iters']:.2f} & {v['segments']:.1f}")
        if has_f1:
            row += f" & {v['em']:.1f} & {v['f1']:.1f}" if v["f1"] is not None else " & -- & --"
        print(row + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"config": vars(args) | {"out": str(args.out)}, "table": table},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
        print(f"\n→ ghi {args.out}")

    if not reader:
        print("\nLƯU Ý: chưa đo EM/F1 (không có --reader). Tỉ lệ nén thấp chưa chắc tốt —"
              "\nphải có EM/F1 mới biết nén nhiều có làm mất thông tin hay không.")


if __name__ == "__main__":
    main()
