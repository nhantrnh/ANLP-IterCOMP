"""Demo trực tiếp: lần theo vòng lặp IterCOMP trên MỘT câu hỏi (VimQA hoặc tiếng Anh).

In từng bước của Hình 1 trong báo cáo, bằng đúng mã của repo (không cài lại):
    pass 1: điểm S_dual của các segment, segment nào qua ngưỡng percentile
    đánh giá "đủ bằng chứng chưa?" -> câu hỏi phụ q'
    pass 2: điểm theo q', câu mang đáp án nhảy hạng thế nào
    P_comp cuối cùng, rồi đáp án + số token của Raw / Oracle / LLMLingua-2 / IterCOMP
    tỉ lệ nén danh nghĩa so với tỉ lệ hiệu dụng (nhân fertility, Mục 3.1 của bài)
    chuẩn hoá boolean trước/sau (câu hỏi đúng/sai)
    --strip-diacritics: chạy lại trên văn bản KHÔNG DẤU (chỉ có nghĩa với VimQA)
    --dataset hotpotqa|2wiki|musique: cùng vòng lặp trên câu tiếng Anh (φ = 1, boolean không kích hoạt)

Cách chạy (từ thư mục gốc repo):
    # không cần GPU, không cần mô hình — chỉ để kiểm tra luồng
    python src/scripts/demo_trace.py --reader mock --itercomp-llm mock

    # thật (GPU, ~15 GB T4), lưu lại để phát lại khi mất GPU/mạng
    python src/scripts/demo_trace.py --index 320 --reader hf --itercomp-llm hf \
        --reader-model Qwen/Qwen2.5-7B-Instruct --load-4bit \
        --cache src/results/demo_cache/trace_320.json

    # một câu tiếng Anh (HotpotQA #277) chạy cùng vòng lặp
    python src/scripts/demo_trace.py --dataset hotpotqa --index 277 --reader hf --itercomp-llm hf \
        --reader-model Qwen/Qwen2.5-7B-Instruct --load-4bit --cache src/results/demo_cache/trace_en_277.json

    # phát lại tức thì, không nạp mô hình
    python src/scripts/demo_trace.py --from-cache src/results/demo_trace_example.json
    python src/scripts/demo_trace.py --from-cache src/results/demo_trace_en_example.json

    # dựng lại vết ghi sẵn từ kết quả đầy đủ (chỉ đáp án/số từ, không có điểm từng vòng)
    python src/scripts/demo_trace.py --from-results --index 320 --cache src/results/demo_trace_example.json

Với --reader mock, đáp án luôn là "unknown" và LLM giả trả lời "đủ" ngay vòng 1:
đủ để thấy luồng chạy, KHÔNG phải số liệu. Số thật nằm ở src/results/*.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/

from itercomp import (clean_answer, f1_score, load_dataset, make_llm,
                      make_reader, make_scorer, normalize_boolean, tokenize)
from itercomp.core import decompose, itercomp
from itercomp.metrics import is_boolean_question
from itercomp.scorer import DEFAULT_LAMBDA, PERCENTILE_BY_DATASET, percentile_filter

RESULTS = Path(__file__).resolve().parents[1] / "results"
#: Câu mặc định: cầu nối 2 bước ("Quê hương của Johann Heinrich Lambert có bao
#: nhiêu bang?"). Trong vimqa_full_7b.json: Raw sai, LLMLingua-2 sai, IterCOMP
#: đúng (855 -> 122 từ).
DEFAULT_INDEX = 320
#: Câu tiếng Anh mặc định (HotpotQA #277, cầu nối 2 bước: nhà soạn nhạc Baroque Ý
#: có tập sonata Op. 5 sinh ở thành phố nào?). Trong hotpotqa_1500_7b.json: Raw
#: và LLMLingua-2 đều "unknown", IterCOMP đúng (595 -> 57 từ).
DEFAULT_INDEX_BY_DATASET = {"vimqa": DEFAULT_INDEX, "hotpotqa": 277, "2wiki": 476, "musique": 0}
RESULT_FILE = {"vimqa": "vimqa_full_7b.json", "hotpotqa": "hotpotqa_1500_7b.json",
               "2wiki": "2wiki_1500_7b.json", "musique": "musique_1500_7b.json"}
DATASET_LABEL = {"vimqa": "VimQA", "hotpotqa": "HotpotQA", "2wiki": "2Wiki", "musique": "MuSiQue"}
#: Câu đúng/sai: reader trả "yes", gold "đúng" -> chuẩn hoá cứu được.
POLAR_INDEX = 23

_DSTROKE = str.maketrans({"đ": "d", "Đ": "D"})


def strip_diacritics(text: str) -> str:
    """Gỡ dấu tiếng Việt; đ/Đ phải map tay vì NFD không tách được."""
    s = unicodedata.normalize("NFD", text.translate(_DSTROKE))
    return unicodedata.normalize("NFC", "".join(c for c in s if not unicodedata.combining(c)))


def strip_row(row: dict) -> dict:
    """Gỡ dấu ngữ cảnh + câu hỏi (người gõ không dấu thì gõ không dấu cả câu hỏi).
    Nhãn vàng giữ nguyên dấu, như chế độ `strip` của báo cáo."""
    ctx, sf = row["context"], row.get("supporting_facts") or {}
    return row | {
        "question": strip_diacritics(row["question"]),
        "context": {"title": [strip_diacritics(t) for t in ctx["title"]],
                    "sentences": [[strip_diacritics(s) for s in doc] for doc in ctx["sentences"]]},
        "supporting_facts": {**sf, "title": [strip_diacritics(t) for t in sf.get("title", [])]},
    }


# ─────────────────────────────────────────────── đếm token / fertility


def _cl100k():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def fertility_factor(dataset: str = "vimqa") -> float:
    """φ (cl100k_base) đo trên 930 cặp WMT24++ — đọc từ file kết quả của repo.
    Tiếng Anh là ngôn ngữ tham chiếu: φ = 1, Ratio_eff = Ratio_nom."""
    if dataset != "vimqa":
        return 1.0
    try:
        d = json.load(open(RESULTS / "fertility_wmt24pp.json", encoding="utf-8"))
        return float(d["tokenizers"]["cl100k_base"]["fertility"])
    except Exception:
        return 2.39


def n_words(text: str) -> int:
    """Đơn vị của cột "ratio" trong bài: \\w+ (âm tiết với tiếng Việt)."""
    return len(tokenize(text))


# ────────────────────────────────────────────── bọc scorer / LLM để ghi vết


class TraceScorer:
    def __init__(self, inner):
        self.inner, self.calls = inner, []
        self.name = getattr(inner, "name", type(inner).__name__)

    def __call__(self, query, texts):
        scores = self.inner(query, texts)
        self.calls.append({"query": query, "texts": list(texts), "scores": [float(s) for s in scores]})
        return scores


class TraceLLM:
    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def __call__(self, prompt, max_tokens=256):
        out = self.inner(prompt, max_tokens=max_tokens)
        kind = "answerable" if "ANSWERABLE" in prompt else "followup"
        self.calls.append({"kind": kind, "output": out})
        return out

    def __getattr__(self, name):          # calls, model, ...
        return getattr(self.inner, name)


# ─────────────────────────────────────────────────────────── dựng vết


def build_trace(args, row, index, reader, llm, scorer, k) -> dict:
    from scripts.run_eval import ctx_full, ctx_oracle  # dùng lại đúng hàm baseline
    segs = decompose(row["context"])
    by_text = {str(s): s for s in segs}
    sf = row.get("supporting_facts") or {}
    gold_ids = set(zip(sf.get("title", []), (int(i) for i in sf.get("sent_id", []))))

    ts, tl = TraceScorer(scorer), TraceLLM(llm)
    res = itercomp(tl, row["question"], row["context"], max_iter=args.max_iter,
                   scorer=ts, percentile=k)

    # mỗi lần chấm điểm là một pass; sau pass là (đánh giá đủ?) rồi (câu hỏi phụ)
    passes, ptr = [], 0
    import numpy as np
    for call in ts.calls:
        scores, texts = call["scores"], call["texts"]
        keep = set(percentile_filter(scores, k))
        order = sorted(range(len(texts)), key=lambda i: -scores[i])
        rank = {i: r + 1 for r, i in enumerate(order)}

        def is_gold(i):
            s = by_text.get(texts[i])
            return bool(s and (s.doc_title, s.sent_id) in gold_ids)
        verdict = followup = None
        if ptr < len(tl.calls) and tl.calls[ptr]["kind"] == "answerable":
            verdict = "YES" in tl.calls[ptr]["output"].upper()
            ptr += 1
            if ptr < len(tl.calls) and tl.calls[ptr]["kind"] == "followup":
                followup = tl.calls[ptr]["output"].strip()
                ptr += 1
        passes.append({
            "query": call["query"], "n_candidates": len(texts),
            "threshold": float(np.percentile(scores, k)), "k": k,
            "top": [{"rank": r + 1, "score": round(scores[i], 4), "passed": i in keep,
                     "gold": is_gold(i), "text": texts[i]} for r, i in enumerate(order[:args.top])],
            "gold_ranks": sorted(rank[i] for i in range(len(texts)) if is_gold(i)),
            "answerable": verdict, "followup": followup})

    contexts = {"raw": ctx_full(row), "oracle": ctx_oracle(row)}
    if args.llmlingua2:
        try:
            from scripts.run_eval import LLMLingua2Method
            contexts["llmlingua2"] = LLMLingua2Method()(row)
        except Exception as e:                     # llmlingua chưa cài / không có GPU
            print(f"  (bỏ qua LLMLingua-2: {type(e).__name__}: {e})", file=sys.stderr)
    contexts["itercomp"] = res.to_prompt()

    enc, gold = _cl100k(), row["answer"]
    raw_w, raw_t = n_words(contexts["raw"]), (len(enc.encode(contexts["raw"])) if enc else None)
    methods = {}
    for name, ctx in contexts.items():
        pred = clean_answer(reader(ctx, row["question"]))
        pn = normalize_boolean(pred, gold, row["question"])
        methods[name] = {"pred": pred, "pred_norm": pn, "f1": f1_score(pred, gold),
                         "f1_norm": f1_score(pn, gold), "words": n_words(ctx),
                         "cl100k": len(enc.encode(ctx)) if enc else None}

    ds = getattr(args, "dataset", "vimqa")
    phi = fertility_factor(ds)
    nominal = methods["itercomp"]["words"] / max(raw_w, 1)
    return {
        "meta": {"source": "live", "index": index, "dataset": ds, "reader": args.reader,
                 "reader_model": args.reader_model, "itercomp_llm": args.itercomp_llm,
                 "scorer": ts.name, "k": k, "max_iter": args.max_iter,
                 "strip_diacritics": bool(args.strip_diacritics)},
        "question": row["question"], "gold": gold, "type": row.get("type"),
        "n_segments": len(segs), "passes": passes,
        "selected": [str(s) for s in res.selected], "stopped_because": res.stopped_because,
        "iterations": res.iterations, "methods": methods,
        "ratio": {"nominal": nominal, "effective": nominal * phi, "fertility": phi,
                  "nominal_cl100k": (methods["itercomp"]["cl100k"] / raw_t) if enc and raw_t else None},
        "boolean": {"polar": is_boolean_question(row["question"]),
                    "pred": methods["itercomp"]["pred"], "pred_norm": methods["itercomp"]["pred_norm"],
                    "f1": methods["itercomp"]["f1"], "f1_norm": methods["itercomp"]["f1_norm"]},
    }


def trace_from_results(index: int, path: Path | None = None, dataset: str = "vimqa") -> dict:
    """Vết "ghi sẵn" từ per_row của lần chạy đầy đủ (src/results/vimqa_full_7b.json).

    Chỉ chứa những gì file kết quả thật sự lưu: đáp án, số từ, F1 của từng
    phương pháp. File kết quả KHÔNG lưu điểm từng vòng, câu hỏi phụ hay P_comp,
    nên các phần đó vắng mặt — muốn có thì chạy live với --cache.
    """
    path = path or RESULTS / RESULT_FILE[dataset]
    d = json.load(open(path, encoding="utf-8"))
    r = d["per_row"][index]
    cfg, phi = d["config"], fertility_factor(dataset)
    methods = {name: {"pred": v["pred"], "pred_norm": v["pred_norm"], "f1": v["f1"],
                      "f1_norm": v["f1_norm"], "words": v["tokens"], "cl100k": None}
               for name, v in r["methods"].items()}
    ic = methods["itercomp"]
    nominal = ic["words"] / r["orig_tokens"]
    # P_comp có ở exp2_undiacritised_500_7b.json (chế độ keep, cùng cấu hình), nhưng
    # chỉ dùng khi số từ khớp ĐÚNG với lần chạy chính: hai lần chạy có thể lệch nhau.
    selected = []
    try:
        if dataset != "vimqa":
            raise KeyError("P_comp chỉ có cho VimQA")
        e = json.load(open(RESULTS / "exp2_undiacritised_500_7b.json", encoding="utf-8"))["detail"]["keep"]
        if index < len(e) and e[index]["question"] == r["question"] \
                and n_words(e[index]["ctx_itercomp"]) == ic["words"]:
            selected = e[index]["ctx_itercomp"].split("\n")
    except Exception:
        pass
    return {
        "meta": {"source": f"ghi sẵn từ {path.name}", "index": index, "dataset": dataset,
                 "reader": cfg["reader"], "reader_model": cfg["reader_model"],
                 "itercomp_llm": cfg["itercomp_llm"], "scorer": cfg["scorer"],
                 "k": 90.0, "max_iter": cfg["max_iter"], "strip_diacritics": False},
        "question": r["question"], "gold": r["gold"], "type": r.get("type"),
        "passes": [], "selected": selected, "iterations": None, "methods": methods,
        "ratio": {"nominal": nominal, "effective": nominal * phi, "fertility": phi,
                  "nominal_cl100k": None},
        "boolean": {"polar": is_boolean_question(r["question"]), "pred": ic["pred"],
                    "pred_norm": ic["pred_norm"], "f1": ic["f1"], "f1_norm": ic["f1_norm"]},
    }


# ───────────────────────────────────────────────────────────────── in


def _c(code, s, on):
    return f"\033[{code}m{s}\033[0m" if on else s


def _clip(s, n):
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n - 1] + "…"


def render(t: dict, color: bool = True, width: int = 110):
    b = lambda s: _c("1", s, color)
    dim = lambda s: _c("2", s, color)
    grn = lambda s: _c("32", s, color)
    red = lambda s: _c("31", s, color)
    m, bar = t["meta"], "═" * 78
    print(bar)
    ds = m.get("dataset", "vimqa")
    print(b(f"{DATASET_LABEL.get(ds, ds)} #{m.get('index')}  [{t.get('type')}]"), dim(f"  nguồn: {m.get('source')}"
          f"  reader={m.get('reader_model') or m.get('reader')}  scorer={m.get('scorer')}"
          f"  k={m.get('k')}  max_iter={m.get('max_iter')}"))
    if m.get("strip_diacritics"):
        print(red("  >> CHẾ ĐỘ KHÔNG DẤU: ngữ cảnh + câu hỏi đã gỡ dấu, nhãn vàng giữ dấu"))
    print(bar)
    print(b("Câu hỏi q :"), t["question"])
    print(b("Nhãn vàng :"), t["gold"], dim(f"   ({t['n_segments']} segment sau khi phân rã)" if t.get("n_segments") else ""))

    passes = t.get("passes")
    if not passes:
        print(dim("\n(vết bản ghi sẵn này không lưu điểm từng vòng — chạy live với --cache để có)"))
    for i, p in enumerate(passes or [], 1):
        print("\n" + b(f"── PASS {i}") + f"   truy vấn = {_clip(p['query'], 90)!r}")
        print(dim(f"   {p['n_candidates']} ứng viên, ngưỡng percentile-{p['k']:g} = {p['threshold']:.4f}"
                  f"   hạng của các câu vàng: {p['gold_ranks']}"))
        for r in p["top"]:
            mark = grn("✔ giữ  ") if r["passed"] else dim("✘ bỏ   ")
            star = grn("★") if r["gold"] else " "
            print(f"   #{r['rank']:<2} {r['score']:>7.4f} {mark}{star} {_clip(r['text'], width - 28)}")
        if p["answerable"] is not None:
            print("   LLM: đủ bằng chứng? ->", grn("ANSWERABLE_YES") if p["answerable"] else red("ANSWERABLE_NO"))
        if p["followup"]:
            print("   LLM: câu hỏi phụ q' ->", b(repr(p["followup"])))
    if passes:
        print(dim(f"\n   (★ = câu vàng trong supporting_facts)  dừng vì: {t.get('stopped_because')}"
                  f"  sau {t.get('iterations')} vòng"))
    if t.get("selected"):
        print("\n" + b("── P_comp (ngữ cảnh nén đưa cho reader)"))
        for s in t["selected"]:
            print("   •", _clip(s, width - 6))

    print("\n" + b("── Đáp án và chi phí") + dim("   (từ = \\w+, đơn vị của cột ratio)"))
    names = {"raw": "Raw", "oracle": "Oracle", "llmlingua2": "LLMLingua-2", "itercomp": "IterCOMP"}
    print(f"   {'':12}{'đáp án':34}{'F1*':>6}{'từ':>7}{'cl100k':>8}")
    for k_, v in t["methods"].items():
        ok = v["f1_norm"] == 1.0
        ans = _clip(v["pred_norm"], 32)
        print(f"   {names.get(k_, k_):12}{(grn(ans) if ok else red(ans)) + ' ' * max(0, 34 - len(ans))}"
              f"{v['f1_norm'] * 100:>6.0f}{v['words']:>7}{(v['cl100k'] if v['cl100k'] is not None else '-'):>8}")

    r = t["ratio"]
    print("\n" + b("── Tỉ lệ nén: danh nghĩa vs hiệu dụng"))
    print(f"   Ratio_nom = {r['nominal']:.3f}   (số từ \\w+ còn lại / ban đầu)")
    eff = "%.3f" % r["effective"]
    print(f"   φ(cl100k_base) = {r['fertility']:.2f}   =>   Ratio_eff = Ratio_nom × φ = {b(eff)}")
    if r.get("nominal_cl100k") is not None:
        print(dim(f"   đo trực tiếp bằng token cl100k_base trên chính ngữ cảnh này: {r['nominal_cl100k']:.3f}"))
    if ds == "vimqa":
        print(dim("   cùng tỉ lệ danh nghĩa, tiếng Việt tốn nhiều token tính tiền hơn tiếng Anh nên tiết kiệm thật ít hơn"))
    else:
        print(dim("   tiếng Anh là ngôn ngữ tham chiếu (φ = 1): tỉ lệ hiệu dụng trùng tỉ lệ danh nghĩa"))

    bo = t["boolean"]
    print("\n" + b("── Chuẩn hoá boolean"))
    if bo["polar"]:
        print(f"   câu hỏi đúng/sai: IterCOMP trả {bo['pred']!r} -> sau chuẩn hoá {bo['pred_norm']!r}"
              f"   F1 {bo['f1'] * 100:.0f} -> {bo['f1_norm'] * 100:.0f}   (nhãn vàng: {t['gold']!r})")
    elif ds != "vimqa":
        print(dim("   tiếng Anh: bước này không kích hoạt (+0.0 F1 trên cả ba bộ tiếng Anh; chỉ VimQA được +8.67)"))
    else:
        print(dim("   không phải câu đúng/sai: bước này không kích hoạt (thử --index 23)"))
    print(bar)


# ─────────────────────────────────────────────────────────────── main


def main():
    ap = argparse.ArgumentParser(description="Vết một câu hỏi qua IterCOMP (VimQA hoặc tiếng Anh)")
    ap.add_argument("--dataset", default="vimqa", choices=sorted(RESULT_FILE),
                    help="mặc định vimqa; hotpotqa/2wiki/musique = câu tiếng Anh (φ = 1)")
    ap.add_argument("--index", type=int, default=None,
                    help=f"chỉ số trong tập dev (mặc định theo dataset: {DEFAULT_INDEX_BY_DATASET}; VimQA câu đúng/sai: {POLAR_INDEX})")
    ap.add_argument("--reader", choices=["mock", "hf"], default="mock")
    ap.add_argument("--reader-model", default=None)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--itercomp-llm", choices=["mock", "hf"], default="mock")
    ap.add_argument("--scorer", default=None, choices=["bm25", "dense", "lexical", "dual"],
                    help="mặc định: dual (bge-m3) khi chạy hf, bm25 khi chạy mock (không cần tải mô hình)")
    ap.add_argument("--max-iter", type=int, default=5)
    ap.add_argument("--percentile", type=float, default=None)
    ap.add_argument("--top", type=int, default=5, help="số segment điểm cao nhất in ra mỗi vòng")
    ap.add_argument("--llmlingua2", action="store_true", help="chạy thêm LLMLingua-2 (cần llmlingua)")
    ap.add_argument("--strip-diacritics", action="store_true", help="chạy trên văn bản không dấu")
    ap.add_argument("--cache", type=Path, default=None, help="ghi vết JSON ra đây (phát lại bằng --from-cache)")
    ap.add_argument("--from-cache", type=Path, default=None, help="phát lại vết đã lưu, không nạp mô hình")
    ap.add_argument("--from-results", action="store_true",
                    help="dựng vết ghi sẵn cho --index từ src/results/<dataset>_*.json (không cần mô hình)")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.index is None:
        args.index = DEFAULT_INDEX_BY_DATASET[args.dataset]
    if args.strip_diacritics and args.dataset != "vimqa":
        ap.error("--strip-diacritics chỉ có nghĩa với --dataset vimqa")
    color = not args.no_color and sys.stdout.isatty()

    if args.from_results:
        trace = trace_from_results(args.index, dataset=args.dataset)
        render(trace, color)
        if args.cache:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            json.dump(trace, open(args.cache, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"đã lưu vết: {args.cache}")
        return
    if args.from_cache:
        render(json.load(open(args.from_cache, encoding="utf-8")), color)
        return

    if "hf" in (args.reader, args.itercomp_llm) and not args.reader_model:
        ap.error("--reader/--itercomp-llm hf cần --reader-model (ví dụ Qwen/Qwen2.5-7B-Instruct)")
    row = load_dataset(args.dataset)[args.index]
    if args.strip_diacritics:
        row = strip_row(row)
    k = args.percentile if args.percentile is not None else PERCENTILE_BY_DATASET[args.dataset]
    scorer_kind = args.scorer or ("dual" if args.reader == "hf" or args.itercomp_llm == "hf" else "bm25")
    rkw = {"model": args.reader_model, "load_4bit": args.load_4bit} if args.reader == "hf" else {}
    reader = make_reader(args.reader, **rkw)
    lkw = {"model": args.reader_model, "load_4bit": args.load_4bit} if args.itercomp_llm == "hf" else {}
    llm = make_llm(args.itercomp_llm, **lkw)
    scorer = make_scorer(scorer_kind, lam=DEFAULT_LAMBDA)

    if "mock" in (args.reader, args.itercomp_llm):
        print("! mock: đáp án/phán đoán giả — chỉ kiểm tra luồng, KHÔNG phải số liệu\n", file=sys.stderr)
    trace = build_trace(args, row, args.index, reader, llm, scorer, k)
    render(trace, color)
    if args.cache:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        json.dump(trace, open(args.cache, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"đã lưu vết: {args.cache}")


if __name__ == "__main__":
    main()
