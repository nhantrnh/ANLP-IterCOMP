"""Live demo: trace the IterCOMP loop on ONE question (VimQA or English).

Prints every step of Figure 1 of the paper, using the repository's own code:
    pass 1: S_dual score of each segment, and which ones clear the percentile threshold
    answerability judgement ("is the evidence sufficient?") -> follow-up question q'
    pass 2: scores against q', and how the answer-bearing sentence moves up
    the final P_comp, then answer + length for Raw / Oracle / LLMLingua-2 / IterCOMP
    nominal vs effective retention (multiplied by fertility, Section 3.3 of the paper)
    boolean normalisation before/after (polar questions)
    --strip-diacritics: rerun on text WITHOUT diacritics (VimQA only)
    --dataset hotpotqa|2wiki|musique: the same loop on an English question (phi = 1, no boolean step)

Usage (from the repository root):
    # no GPU and no models - checks the control flow only
    python src/scripts/demo_trace.py --reader mock --itercomp-llm mock

    # real run (GPU, ~15 GB T4), saved for replay when GPU or network is unavailable
    python src/scripts/demo_trace.py --index 320 --reader hf --itercomp-llm hf \
        --reader-model Qwen/Qwen2.5-7B-Instruct --load-4bit \
        --cache src/results/demo_cache/trace_320.json

    # an English question (HotpotQA #277) through the same loop
    python src/scripts/demo_trace.py --dataset hotpotqa --index 277 --reader hf --itercomp-llm hf \
        --reader-model Qwen/Qwen2.5-7B-Instruct --load-4bit --cache src/results/demo_cache/trace_en_277.json

    # instant replay, no model loading
    python src/scripts/demo_trace.py --from-cache src/results/demo_trace_example.json
    python src/scripts/demo_trace.py --from-cache src/results/demo_trace_en_example.json

    # rebuild a pre-recorded trace from the full results (answers and lengths only, no per-pass scores)
    python src/scripts/demo_trace.py --from-results --index 320 --cache src/results/demo_trace_example.json

With --reader mock the answer is always "unknown" and the fake judge says "sufficient" in pass 1:
enough to see the control flow, NOT results. Real numbers live in src/results/*.json.
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
#: Default question: a two-hop bridge ("Quê hương của Johann Heinrich Lambert có bao
#: nhiêu bang?"). In vimqa_full_7b.json Raw and LLMLingua-2 are wrong and IterCOMP is
#: right (855 -> 122 words).
DEFAULT_INDEX = 320
#: Default English question (HotpotQA #277, two-hop bridge: in what city was the
#: Italian Baroque composer of the Op. 5 sonatas born?). In hotpotqa_1500_7b.json Raw
#: and LLMLingua-2 both answer "unknown" and IterCOMP is right (595 -> 57 words).
DEFAULT_INDEX_BY_DATASET = {"vimqa": DEFAULT_INDEX, "hotpotqa": 277, "2wiki": 476, "musique": 0}
RESULT_FILE = {"vimqa": "vimqa_full_7b.json", "hotpotqa": "hotpotqa_1500_7b.json",
               "2wiki": "2wiki_1500_7b.json", "musique": "musique_1500_7b.json"}
DATASET_LABEL = {"vimqa": "VimQA", "hotpotqa": "HotpotQA", "2wiki": "2Wiki", "musique": "MuSiQue"}
#: Polar question: the reader answers "yes", the gold answer is "đúng" -> normalisation fixes it.
POLAR_INDEX = 23

#: core.py records why the loop stopped in Vietnamese; print it in English.
STOP_REASON = {"answerable": "the judge reported sufficient evidence",
               "hết vòng lặp": "reached max_iter",
               "hết segment": "no segments left"}


def stop_reason(reason: str | None) -> str:
    if not reason:
        return "-"
    if reason.startswith("đạt trần tỉ lệ"):
        return "reached the 60% retention cap"
    if reason.startswith("đạt token budget"):
        return "reached the token budget"
    return STOP_REASON.get(reason, reason)


def source_label(source: str | None) -> str:
    """Older cached traces store their provenance in Vietnamese ('ghi sẵn từ X')."""
    if source and source.startswith("ghi sẵn từ "):
        return "pre-recorded from " + source[len("ghi sẵn từ "):]
    return source or "-"


_DSTROKE = str.maketrans({"đ": "d", "Đ": "D"})


def strip_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics; đ/Đ are mapped by hand because NFD does not split them."""
    s = unicodedata.normalize("NFD", text.translate(_DSTROKE))
    return unicodedata.normalize("NFC", "".join(c for c in s if not unicodedata.combining(c)))


def strip_row(row: dict) -> dict:
    """Strip diacritics from context and question (someone typing without diacritics types the
    question that way too). The gold answer keeps its diacritics, as in the paper's `strip` setting."""
    ctx, sf = row["context"], row.get("supporting_facts") or {}
    return row | {
        "question": strip_diacritics(row["question"]),
        "context": {"title": [strip_diacritics(t) for t in ctx["title"]],
                    "sentences": [[strip_diacritics(s) for s in doc] for doc in ctx["sentences"]]},
        "supporting_facts": {**sf, "title": [strip_diacritics(t) for t in sf.get("title", [])]},
    }


# ─────────────────────────────────────────────── token counting / fertility


def _cl100k():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def fertility_factor(dataset: str = "vimqa") -> float:
    """phi (cl100k_base) measured on 930 WMT24++ pairs, read from the repository's results.
    English is the reference language: phi = 1, so Ratio_eff = Ratio_nom."""
    if dataset != "vimqa":
        return 1.0
    try:
        d = json.load(open(RESULTS / "fertility_wmt24pp.json", encoding="utf-8"))
        return float(d["tokenizers"]["cl100k_base"]["fertility"])
    except Exception:
        return 2.39


def n_words(text: str) -> int:
    """The unit of the paper's retention column: \\w+ (syllables for Vietnamese)."""
    return len(tokenize(text))


# ────────────────────────────────────────────── scorer / LLM wrappers that record calls


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


# ─────────────────────────────────────────────────────────── building a trace


def build_trace(args, row, index, reader, llm, scorer, k) -> dict:
    from scripts.run_eval import ctx_full, ctx_oracle  # reuse the exact baseline functions
    segs = decompose(row["context"])
    by_text = {str(s): s for s in segs}
    sf = row.get("supporting_facts") or {}
    gold_ids = set(zip(sf.get("title", []), (int(i) for i in sf.get("sent_id", []))))

    ts, tl = TraceScorer(scorer), TraceLLM(llm)
    res = itercomp(tl, row["question"], row["context"], max_iter=args.max_iter,
                   scorer=ts, percentile=k)

    # each scoring call is one pass; after it come the answerability judgement and the follow-up
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
        except Exception as e:                     # llmlingua not installed / no GPU
            print(f"  (skipping LLMLingua-2: {type(e).__name__}: {e})", file=sys.stderr)
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
    """A pre-recorded trace built from the per_row data of a full run (e.g. vimqa_full_7b.json).

    It holds only what the results file stores: each method's answer, length and F1. The
    results file does NOT store per-pass scores, follow-up questions or P_comp, so those are
    absent - run live with --cache to get them.
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
    # exp2_undiacritised_500_7b.json (keep setting, same configuration) stores P_comp, but it is
    # used only when its word count matches the main run EXACTLY: the two runs can differ.
    selected = []
    try:
        if dataset != "vimqa":
            raise KeyError("P_comp is only available for VimQA")
        e = json.load(open(RESULTS / "exp2_undiacritised_500_7b.json", encoding="utf-8"))["detail"]["keep"]
        if index < len(e) and e[index]["question"] == r["question"] \
                and n_words(e[index]["ctx_itercomp"]) == ic["words"]:
            selected = e[index]["ctx_itercomp"].split("\n")
    except Exception:
        pass
    return {
        "meta": {"source": f"pre-recorded from {path.name}", "index": index, "dataset": dataset,
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


# ───────────────────────────────────────────────────────────────── printing


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
    print(b(f"{DATASET_LABEL.get(ds, ds)} #{m.get('index')}  [{t.get('type')}]"),
          dim(f"  source: {source_label(m.get('source'))}"
              f"  reader={m.get('reader_model') or m.get('reader')}  scorer={m.get('scorer')}"
              f"  k={m.get('k')}  max_iter={m.get('max_iter')}"))
    if m.get("strip_diacritics"):
        print(red("  >> NO-DIACRITICS MODE: context and question stripped, gold answer keeps its diacritics"))
    print(bar)
    print(b("Question q :"), t["question"])
    print(b("Gold answer:"), t["gold"],
          dim(f"   ({t['n_segments']} segments after decomposition)" if t.get("n_segments") else ""))

    passes = t.get("passes")
    if not passes:
        print(dim("\n(this pre-recorded trace has no per-pass scores - run live with --cache to get them)"))
    for i, p in enumerate(passes or [], 1):
        print("\n" + b(f"── PASS {i}") + f"   query = {_clip(p['query'], 90)!r}")
        print(dim(f"   {p['n_candidates']} candidates, percentile-{p['k']:g} threshold = {p['threshold']:.4f}"
                  f"   ranks of the gold sentences: {p['gold_ranks']}"))
        for r in p["top"]:
            mark = grn("✔ keep ") if r["passed"] else dim("✘ drop ")
            star = grn("★") if r["gold"] else " "
            print(f"   #{r['rank']:<2} {r['score']:>7.4f} {mark}{star} {_clip(r['text'], width - 28)}")
        if p["answerable"] is not None:
            print("   LLM: is the evidence sufficient? ->",
                  grn("ANSWERABLE_YES") if p["answerable"] else red("ANSWERABLE_NO"))
        if p["followup"]:
            print("   LLM: follow-up question q' ->", b(repr(p["followup"])))
    if passes:
        print(dim(f"\n   (★ = gold supporting sentence)  stopped because: {stop_reason(t.get('stopped_because'))}"
                  f"  after {t.get('iterations')} pass(es)"))
    if t.get("selected"):
        print("\n" + b("── P_comp (compressed context given to the reader)"))
        for s in t["selected"]:
            print("   •", _clip(s, width - 6))

    print("\n" + b("── Answers and cost") + dim("   (words = \\w+, the unit of the retention column)"))
    names = {"raw": "Raw", "oracle": "Oracle", "llmlingua2": "LLMLingua-2", "itercomp": "IterCOMP"}
    print(f"   {'':12}{'answer':34}{'F1*':>6}{'words':>7}{'cl100k':>8}")
    for k_, v in t["methods"].items():
        ok = v["f1_norm"] == 1.0
        ans = _clip(v["pred_norm"], 32)
        print(f"   {names.get(k_, k_):12}{(grn(ans) if ok else red(ans)) + ' ' * max(0, 34 - len(ans))}"
              f"{v['f1_norm'] * 100:>6.0f}{v['words']:>7}{(v['cl100k'] if v['cl100k'] is not None else '-'):>8}")

    r = t["ratio"]
    print("\n" + b("── Retention: nominal vs effective"))
    print(f"   Ratio_nom = {r['nominal']:.3f}   (\\w+ words kept / original)")
    eff = "%.3f" % r["effective"]
    print(f"   phi(cl100k_base) = {r['fertility']:.2f}   =>   Ratio_eff = Ratio_nom × phi = {b(eff)}")
    if r.get("nominal_cl100k") is not None:
        print(dim(f"   measured directly in cl100k_base tokens on this context: {r['nominal_cl100k']:.3f}"))
    if ds == "vimqa":
        print(dim("   same nominal retention, but Vietnamese costs more billed tokens than English, so the real saving is smaller"))
    else:
        print(dim("   English is the reference language (phi = 1): effective retention equals nominal retention"))

    bo = t["boolean"]
    print("\n" + b("── Boolean normalisation"))
    if bo["polar"]:
        print(f"   polar question: IterCOMP answered {bo['pred']!r} -> after normalisation {bo['pred_norm']!r}"
              f"   F1 {bo['f1'] * 100:.0f} -> {bo['f1_norm'] * 100:.0f}   (gold: {t['gold']!r})")
    elif ds != "vimqa":
        print(dim("   English: the step never fires (+0.0 F1 on all three English sets; only VimQA gains +8.67)"))
    else:
        print(dim("   not a polar question: the step does not fire (try --index 23)"))
    print(bar)


# ─────────────────────────────────────────────────────────────── main


def main():
    ap = argparse.ArgumentParser(description="Trace one question through IterCOMP (VimQA or English)")
    ap.add_argument("--dataset", default="vimqa", choices=sorted(RESULT_FILE),
                    help="default vimqa; hotpotqa/2wiki/musique = an English question (phi = 1)")
    ap.add_argument("--index", type=int, default=None,
                    help=f"index in the dev set (default per dataset: {DEFAULT_INDEX_BY_DATASET}; VimQA polar question: {POLAR_INDEX})")
    ap.add_argument("--reader", choices=["mock", "hf"], default="mock")
    ap.add_argument("--reader-model", default=None)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--itercomp-llm", choices=["mock", "hf"], default="mock")
    ap.add_argument("--scorer", default=None, choices=["bm25", "dense", "lexical", "dual"],
                    help="default: dual (bge-m3) for hf runs, bm25 for mock runs (no model download)")
    ap.add_argument("--max-iter", type=int, default=5)
    ap.add_argument("--percentile", type=float, default=None)
    ap.add_argument("--top", type=int, default=5, help="how many top-scoring segments to print per pass")
    ap.add_argument("--llmlingua2", action="store_true", help="also run LLMLingua-2 (needs llmlingua)")
    ap.add_argument("--strip-diacritics", action="store_true", help="run on text without diacritics")
    ap.add_argument("--cache", type=Path, default=None, help="write the trace as JSON here (replay with --from-cache)")
    ap.add_argument("--from-cache", type=Path, default=None, help="replay a saved trace, no model loading")
    ap.add_argument("--from-results", action="store_true",
                    help="build a pre-recorded trace for --index from src/results/<dataset>_*.json (no models)")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.index is None:
        args.index = DEFAULT_INDEX_BY_DATASET[args.dataset]
    if args.strip_diacritics and args.dataset != "vimqa":
        ap.error("--strip-diacritics only makes sense with --dataset vimqa")
    color = not args.no_color and sys.stdout.isatty()

    if args.from_results:
        trace = trace_from_results(args.index, dataset=args.dataset)
        render(trace, color)
        if args.cache:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            json.dump(trace, open(args.cache, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"trace saved: {args.cache}")
        return
    if args.from_cache:
        render(json.load(open(args.from_cache, encoding="utf-8")), color)
        return

    if "hf" in (args.reader, args.itercomp_llm) and not args.reader_model:
        ap.error("--reader/--itercomp-llm hf needs --reader-model (e.g. Qwen/Qwen2.5-7B-Instruct)")
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
        print("! mock: fake answers and judgements - checks the control flow only, NOT results\n", file=sys.stderr)
    trace = build_trace(args, row, args.index, reader, llm, scorer, k)
    render(trace, color)
    if args.cache:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        json.dump(trace, open(args.cache, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"trace saved: {args.cache}")


if __name__ == "__main__":
    main()
