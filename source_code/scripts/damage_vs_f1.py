"""Hư hại ngữ pháp có THẬT SỰ làm sụt F1 không, hay chỉ trùng hợp?

VÌ SAO CẦN. `syllable_damage.py` cho thấy LLMLingua-2 cắt hư từ lệch 1,47x còn
IterCOMP 1,02x. Nhưng đó mới là quan sát về HÌNH THỨC văn bản. Nó chưa chứng
minh việc cắt hư từ GÂY RA mất chất lượng — có thể bộ nén tệ vì lý do khác và
chuyện cắt hư từ chỉ đi kèm.

Phép kiểm: trong CÙNG một phương pháp, những câu hỏi bị cắt hư từ nặng có sụt
F1 nhiều hơn những câu bị cắt nhẹ không? Nếu có tương quan, lập luận đứng.

DỮ LIỆU. `results/vimqa_200.json` đã có F1 từng câu cho từng phương pháp
(n=200, reader Qwen2.5-0.5B). Ta tính lại tỉ số hư-từ trên đúng 200 câu đó và
ghép theo chỉ số dòng. Không cần GPU: reader đã chạy xong từ trước.

LƯU Ý VỀ SỨC MẠNH. Reader 0.5B cho F1 tuyệt đối thấp (IterCOMP 19.3). Điều đó
KHÔNG ảnh hưởng phép kiểm này, vì ta so các câu VỚI NHAU trong cùng một cấu
hình, không so cấu hình với nhau.

    python scripts/damage_vs_f1.py
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from itercomp import load_dataset, make_scorer  # noqa: E402
from itercomp.core import decompose, score_segments  # noqa: E402
from itercomp.fertility import function_word_loss, syllable_spans  # noqa: E402
from itercomp.scorer import percentile_filter  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def ctx_full(row) -> str:
    return "\n".join(f"{t}: {' '.join(s)}" for t, s
                     in zip(row["context"]["title"], row["context"]["sentences"]))


def ctx_itercomp(row, scorer, pct: float) -> str:
    segs = decompose(row["context"])
    scores = score_segments(row["question"], [str(s) for s in segs], scorer)
    keep = percentile_filter(scores, pct)
    return "\n".join(f"{segs[i].doc_title}: {segs[i].text}" for i in keep)


def spearman(x: list[float], y: list[float]) -> float:
    """Tương quan hạng Spearman.

    Dùng hạng chứ không dùng Pearson vì F1 chặn ở 0 và 100, phân bố lệch nặng
    (rất nhiều 0), nên giả định tuyến tính của Pearson không đứng.
    """
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):                       # xu ly hang bang nhau
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def boot_ci(x: list[float], y: list[float], n: int = 2000, seed: int = 0):
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        idx = [rng.randrange(len(x)) for _ in range(len(x))]
        vals.append(spearman([x[i] for i in idx], [y[i] for i in idx]))
    vals = [v for v in vals if v == v]
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-json", type=Path, default=RESULTS / "vimqa_200.json")
    ap.add_argument("--percentile", type=float, default=None,
                    help="mặc định: lấy theo eval-json")
    ap.add_argument("--scorer", default=None,
                    help="mặc định: LẤY THEO eval-json. Ghi đè chỉ khi biết "
                         "mình đang làm gì — xem kiểm tra bên dưới.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.eval_json.is_file():
        raise SystemExit(
            f"không thấy {args.eval_json}.\n"
            "Cần một file kết quả của run_eval.py có per_row (F1 từng câu) cho\n"
            "cả itercomp lẫn llmlingua2. Chạy trên Kaggle/Colab với reader 7B:\n"
            "  python scripts/run_eval.py --dataset vimqa --limit 200 \\\n"
            "    --reader hf --reader-model Qwen/Qwen2.5-7B-Instruct --load-4bit \\\n"
            "    --itercomp-llm hf --methods llmlingua2,itercomp \\\n"
            "    --out results/vimqa_200_7b.json")
    ev = json.loads(args.eval_json.read_text())
    per_row = ev["per_row"]
    missing = [m for m in ("itercomp", "llmlingua2")
               if m not in per_row[0].get("methods", {})]
    if missing:
        raise SystemExit(f"{args.eval_json} thiếu method: {', '.join(missing)}")
    cfg = ev["config"]
    n = len(per_row)
    rate = cfg.get("rate", 0.33)
    lang = "vi" if cfg["dataset"] == "vimqa" else "en"

    rows = load_dataset(cfg["dataset"], n)
    if len(rows) != n:
        raise SystemExit(f"lệch số dòng: dataset {len(rows)} vs eval {n}")

    # Scorer PHẢI khớp với lần chạy đã sinh ra F1. Nếu không, ta tính tỉ số
    # hư-từ trên ngữ cảnh do bm25 chọn rồi ghép với F1 của ngữ cảnh do dual
    # chọn — hai ngữ cảnh khác nhau, và tương quan thu được vô nghĩa. Đây là
    # loại lỗi không ném exception, chỉ lặng lẽ cho ra số sai.
    # cfg["percentile"] có thể TỒN TẠI với giá trị None (run_eval.py ghi
    # nguyên args, và mặc định của cờ đó là None để suy theo dataset). Nên
    # cfg.get(..., 90.0) trả về None chứ không phải 90.0 — phải kiểm tra rõ.
    pct = args.percentile
    if pct is None:
        pct = cfg.get("percentile")
    if pct is None:
        from itercomp.scorer import DEFAULT_PERCENTILE, PERCENTILE_BY_DATASET
        pct = PERCENTILE_BY_DATASET.get(cfg["dataset"], DEFAULT_PERCENTILE)
    eval_scorer = cfg.get("scorer", "dual")
    scorer_kind = args.scorer or eval_scorer
    if scorer_kind != eval_scorer:
        print(f"CẢNH BÁO: --scorer {scorer_kind} khác scorer của {args.eval_json.name} "
              f"({eval_scorer}).\n"
              f"  Tỉ số hư-từ sẽ tính trên ngữ cảnh KHÁC với ngữ cảnh đã sinh F1,\n"
              f"  nên tương quan không đọc được. Bỏ --scorer để dùng {eval_scorer}.",
              flush=True)

    kw = {}
    if eval_scorer == "dual" and cfg.get("encoder"):
        kw["encoder"] = cfg["encoder"]
    scorer = make_scorer(scorer_kind, **kw)
    print(f"scorer={scorer_kind} (khớp eval-json)", flush=True)
    print(f"Nạp LLMLingua-2 (CPU) — tính lại ngữ cảnh cho {n} câu ...", flush=True)
    from llmlingua import PromptCompressor
    comp = PromptCompressor(
        model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        use_llmlingua2=True, device_map="cpu")

    dmg: dict[str, list[float]] = {"itercomp": [], "llmlingua2": []}
    f1: dict[str, list[float]] = {"itercomp": [], "llmlingua2": []}
    skipped = 0

    for i, (r, pr) in enumerate(zip(rows, per_row), 1):
        # kiem tra ghep dung cau, khong tin thu tu mot cach mu quang
        if pr["question"].strip() != r["question"].strip():
            raise SystemExit(f"dòng {i} lệch câu hỏi — không ghép được")

        full = ctx_full(r)
        ctxs = {"itercomp": ctx_itercomp(r, scorer, pct),
                "llmlingua2": comp.compress_prompt(
                    full, rate=rate, force_tokens=["\n", "?", ".", ","],
                    drop_consecutive=True)["compressed_prompt"]}

        for m, ctx in ctxs.items():
            loss = function_word_loss(ctx, full, lang)
            fw, ct = loss["hư từ"], loss["nội dung"]
            if fw != fw or ct != ct or ct <= 0:
                skipped += 1
                continue
            dmg[m].append(fw / ct)
            f1[m].append(pr["methods"][m]["f1_norm"])
        if i % 25 == 0:
            print(f"  {i}/{n}", flush=True)

    print(f"\neval={args.eval_json.name}  n={n}  reader={cfg.get('reader_model')}")
    if skipped:
        print(f"bỏ {skipped} điểm không định nghĩa được tỉ số\n")

    print(f"{'phương pháp':13s} {'tỉ số tb':>10s} {'F1 tb':>8s} "
          f"{'Spearman(tỉ số, F1)':>21s} {'CI 95%':>18s}")
    out = {}
    for m in ("itercomp", "llmlingua2"):
        rho = spearman(dmg[m], f1[m])
        lo, hi = boot_ci(dmg[m], f1[m])
        out[m] = {"rho": rho, "ci_lo": lo, "ci_hi": hi,
                  "mean_ratio": st.mean(dmg[m]), "mean_f1": st.mean(f1[m]),
                  "n": len(dmg[m])}
        print(f"{m:13s} {st.mean(dmg[m]):9.2f}x {st.mean(f1[m]):8.1f} "
              f"{rho:20.3f} {f'[{lo:+.2f}, {hi:+.2f}]':>18s}")

    print("\nĐỌC KẾT QUẢ")
    print("  rho âm = cắt hư từ càng nặng thì F1 càng thấp (giả thuyết đúng)")
    for m, d in out.items():
        sig = d["ci_hi"] < 0 or d["ci_lo"] > 0
        if not sig:
            verdict = "KHÔNG có tương quan (CI chứa 0)"
        elif d["rho"] < 0:
            verdict = "CÓ tương quan âm — ủng hộ giả thuyết"
        else:
            verdict = "tương quan DƯƠNG — NGƯỢC giả thuyết"
        print(f"  {m:12s}: {verdict}")

    # so nhom: 1/3 bi cat nang nhat vs 1/3 nhe nhat
    print("\nSO NHÓM (tam phân vị theo mức cắt hư từ)")
    print(f"{'phương pháp':13s} {'F1 nhóm nhẹ':>13s} {'F1 nhóm nặng':>14s} {'chênh':>8s}")
    for m in ("itercomp", "llmlingua2"):
        pairs = sorted(zip(dmg[m], f1[m]))
        k = len(pairs) // 3
        lo_f1 = st.mean(f for _, f in pairs[:k])
        hi_f1 = st.mean(f for _, f in pairs[-k:])
        out[m]["tercile_low_damage_f1"] = lo_f1
        out[m]["tercile_high_damage_f1"] = hi_f1
        print(f"{m:13s} {lo_f1:12.1f} {hi_f1:13.1f} {hi_f1-lo_f1:+8.1f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"eval_json": str(args.eval_json), "config": vars(args) | {"out": str(args.out)},
                   "summary": out,
                   "per_row": {m: [{"ratio": d, "f1": f}
                                   for d, f in zip(dmg[m], f1[m])] for m in dmg}},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
        print(f"\n→ ghi {args.out}")


if __name__ == "__main__":
    main()
