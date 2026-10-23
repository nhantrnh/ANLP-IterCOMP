"""Helpers for notebooks/colab_reproduce.ipynb.

The notebook doubles as the slide deck, so each cell calls only a few lines and the
longer logic lives here. Every function runs with the repository root as the working
directory (the notebook calls `os.chdir(REPO)`) and reads the relative paths `data/`,
`results/` and `src/results/`.

    from itercomp import nbhelpers as nb
"""
from __future__ import annotations

import collections
import glob
import json
import os
import statistics
import subprocess
import sys
import textwrap
import time
import urllib.request
from dataclasses import dataclass, field

# ══════════════════════════════════════════════ data
DATA_FILES = {
    'vimqa/validation.parquet':    ('nguyenlab/vimqa', 'data/validation-00000-of-00001.parquet'),
    '2wiki/dev.parquet':           ('xanhho/2WikiMultihopQA', 'dev.parquet'),
    'hotpotqa/validation.parquet': ('hotpotqa/hotpot_qa', 'distractor/validation-00000-of-00001.parquet'),
    'musique/dev.jsonl':           ('dgslibisey/MuSiQue', 'musique_ans_v1.0_dev.jsonl'),
}


def download_data(all_sets: bool) -> None:
    """Download VimQA (+ HotpotQA for the demo's English question); `all_sets` adds 2Wiki and MuSiQue."""
    for dst, (repo, src) in DATA_FILES.items():
        if not all_sets and not dst.startswith(('vimqa', 'hotpotqa')):
            continue
        p = 'data/' + dst
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if not os.path.exists(p):
            urllib.request.urlretrieve(f'https://huggingface.co/datasets/{repo}/resolve/main/{src}', p)
        print(f'{dst:32s} {os.path.getsize(p) / 1e6:6.1f} MB')


def peek_datasets() -> None:
    """For each downloaded set: passages and gold supporting sentences of the first question."""
    from itercomp import load_dataset
    paths = {'hotpotqa': 'data/hotpotqa/validation.parquet', '2wiki': 'data/2wiki/dev.parquet',
             'musique': 'data/musique/dev.jsonl', 'vimqa': 'data/vimqa/validation.parquet'}
    for ds, path in paths.items():
        if os.path.exists(path):
            r = load_dataset(ds, 1)[0]
            n = len(r.get('supporting_facts', {}).get('title', []))
            print(f"{ds:9s} {len(r['context']['title']):2d} passages | supporting_facts={n}")


def eda_vimqa(n: int = 200) -> None:
    """Descriptive statistics of VimQA, answer types, sentence decomposition and a tokenizer check."""
    from itercomp import load_dataset
    from itercomp.core import decompose
    from itercomp.metrics import tokenize

    rows = load_dataset('vimqa', n)
    n_para = [len(r['context']['title']) for r in rows]
    n_tok = [len(tokenize('\n'.join(' '.join(s) for s in r['context']['sentences']))) for r in rows]
    n_gold = [len(r.get('supporting_facts', {}).get('title', [])) for r in rows]
    print(f'VimQA - {len(rows)} questions')
    for lab, v in (('passages / question', n_para), ('tokens / context', n_tok), ('gold sentences / question', n_gold)):
        print(f'  {lab:28s}: median {statistics.median(v):.0f}  (min {min(v)}, max {max(v)})')

    boolean = {'đúng', 'không', 'sai', 'có'}
    kind = collections.Counter('yes/no' if str(r['answer']).strip().lower() in boolean else 'extractive'
                               for r in rows)
    print('\nanswer type:')
    for k, v in kind.most_common():
        print(f'  {k:12s} {v:3d}  {v / len(rows):5.1%}')

    r = rows[0]
    segs = decompose(r['context'])
    print(f"\ndecomposing the first question: {len(r['context']['title'])} passages -> {len(segs)} sentence segments")
    for s in segs[:3]:
        print(f'  [{s.doc_title[:22]:24s}] {str(s)[:58]}…')
    assert tokenize('vào') == ['vào'], 'the tokenizer breaks Vietnamese diacritics'
    print('tokenizer keeps diacritics:', tokenize('má mà mã'))


# ══════════════════════════════════════════════ configuration
SCALES = {
    'demo':  {'vimqa': 30},
    'paper': {'vimqa': None, 'musique': 1500, 'hotpotqa': 1500, '2wiki': 1500},
    'max':   {'vimqa': None, 'musique': None},
}
SEC_PER_Q = {'vimqa': 9.6, 'musique': 15.0, 'hotpotqa': 6.5, '2wiki': 6.2}   # measured on a T4, four methods per question
FULL_N = {'vimqa': 1003, 'musique': 2417, 'hotpotqa': 7405, '2wiki': 12576}


def nlabel(n) -> str:
    return 'full' if n is None else str(n)


@dataclass
class Config:
    """Sample size and reader. The file suffix comes from the ACTUAL reader so 1.5B results are never filed as 7B."""
    scale: str
    reader: str = 'Qwen/Qwen2.5-7B-Instruct'
    load_4bit: bool = True
    tag: str = 'hf'
    plan: dict = field(init=False)
    mtag: str = field(init=False)

    def __post_init__(self):
        assert self.scale in SCALES, f'SCALE must be one of {list(SCALES)}'
        self.plan = SCALES[self.scale]
        self.mtag = self.reader.split('/')[-1].replace('Qwen2.5-', '').replace('-Instruct', '').lower()
        os.makedirs('results', exist_ok=True)

    def outfile(self, ds: str, n, prefix: str = '') -> str:
        return f'results/{prefix}{ds}_{nlabel(n)}_{self.mtag}.json'

    def small_n(self, n_full: int = 200) -> int:
        """Sample size of the extra experiments (ablation, EXP-1/2, λ/k sweep)."""
        return n_full if self.scale != 'smoke' else 50

    def print_plan(self) -> None:
        print(f'reader = {self.reader} | 4-bit = {self.load_4bit} | SCALE = {self.scale!r}\n')
        print(f"{'dataset':10s} {'n':>7s} {'estimate':>10s}")
        tot = 0.0
        for ds, n in self.plan.items():
            h = (FULL_N[ds] if n is None else n) * SEC_PER_Q[ds] / 3600
            tot += h
            print(f'{ds:10s} {nlabel(n):>7s} {h:9.2f}h')
        print(f"{'TOTAL':10s} {'':>7s} {tot:9.2f}h" + ('  exceeds one 12 h Colab session' if tot > 12 else ''))


def run_stream(cmd: list, logfile: str):
    """Run a subprocess, echo its output immediately (Colab may drop a silent cell) and keep a log."""
    tail = []
    with open(logfile, 'w') as f:
        pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in pr.stdout:
            print(line, end='', flush=True)
            f.write(line)
            tail.append(line)
            del tail[:-400]
        pr.wait()
    return pr.returncode, ''.join(tail)


def _reader_args(cfg: Config) -> list:
    return ['--reader', 'hf', '--reader-model', cfg.reader, *(['--load-4bit'] if cfg.load_4bit else [])]


def _banner(title: str) -> None:
    print('\n' + '=' * 70 + f'\n### {title}   started {time.strftime("%H:%M")}\n' + '=' * 70, flush=True)


def _run_script(cfg: Config, script: str, extra: list, out: str, log: str, label: str) -> None:
    """Run an experiment script unless `out` already exists; report clearly if the result file is missing."""
    if os.path.exists(out):
        print(f'[skip] {label} already done ({out})')
        return
    _banner(label)
    t0 = time.time()
    rc, _ = run_stream([sys.executable, '-u', script, *extra, '--out', out], log)
    if rc:
        print(f'!!! {label} FAILED (exit code {rc}) - see the log')
    elif not os.path.exists(out):
        print(f'!!! {label} finished but {out} is MISSING')
    print(f'--> {(time.time() - t0) / 60:.1f} min', flush=True)


def run_main_eval(cfg: Config, gpu: bool) -> None:
    """Evaluate Raw / Oracle / LLMLingua-2 / IterCOMP-X following `cfg.plan` (checkpointed after EVERY question)."""
    if not gpu:
        print('No GPU: skipping the live evaluation. Paper-scale numbers are in VI.1 (read from the committed JSON).')
        return
    for ds in [d for d in ('vimqa', 'musique', 'hotpotqa', '2wiki') if d in cfg.plan]:
        n = cfg.plan[ds]
        _run_script(cfg, 'src/scripts/run_eval.py',
                    ['--dataset', ds, *(['--limit', str(n)] if n is not None else []), *_reader_args(cfg),
                     '--methods', 'raw,oracle,llmlingua2,itercomp',
                     '--itercomp-llm', cfg.tag, '--scorer', 'dual'],   # a real LLM for every reasoning step
                    cfg.outfile(ds, n), f'results/log_{ds}_{nlabel(n)}_{cfg.mtag}.txt', f'{ds} n={nlabel(n)}')
    print('\nMain evaluation DONE.')


# ══════════════════════════════════════════════ result tables (read from JSON, never typed)
FILES_T3 = {'VimQA': 'vimqa_full_7b.json', 'MuSiQue': 'musique_1500_7b.json',
            '2Wiki': '2wiki_1500_7b.json', 'HotpotQA': 'hotpotqa_1500_7b.json'}
METHODS = (('oracle', 'Oracle'), ('raw', 'Raw'), ('itercomp', 'IterCOMP-X'), ('llmlingua2', 'LLMLingua-2'))


def _load(f: str) -> dict:
    return json.load(open(f'src/results/{f}', encoding='utf-8'))


def load_table3() -> dict:
    """Summaries of the four full runs (VimQA n=1003; the three English sets n=1500)."""
    S = {}
    for name, f in FILES_T3.items():
        d = _load(f)
        S[name] = d['summary']
        S[name]['n'] = len(d['per_row'])
    return S


def _row(S: dict, label: str, fn) -> None:
    print(f'{label:16}' + ''.join(f'{fn(S[k]):>11}' for k in FILES_T3))


def print_table3(S: dict) -> None:
    print('Main results of the paper (Qwen2.5-7B, 4-bit)')
    print(f'{"F1*":16}' + ''.join(f'{k:>11}' for k in FILES_T3))
    _row(S, 'n', lambda s: s['n'])
    for m, lab in METHODS:
        _row(S, lab, lambda s, m=m: f"{s[m]['f1_norm']:.2f}")
    _row(S, 'Δ vs Raw', lambda s: f"{s['itercomp']['f1_norm'] - s['raw']['f1_norm']:+.2f}")
    _row(S, 'Δ vs LLML-2', lambda s: f"{s['itercomp']['f1_norm'] - s['llmlingua2']['f1_norm']:+.2f}")
    _row(S, 'Ratio_nom', lambda s: f"{s['itercomp']['ratio']:.3f}")
    print('Confidence intervals (†) are in the paper\'s main table. HotpotQA: Raw beats IterCOMP-X - the negative result is kept.')


def print_live_result(cfg: Config) -> None:
    """Result of the live run above (small n: it only shows that the pipeline runs)."""
    path = cfg.outfile('vimqa', cfg.plan['vimqa'])
    if not os.path.exists(path):
        print('No live result yet (no GPU, or VI.4 was skipped) - see the main table in VI.1.')
        return
    d = json.load(open(path, encoding='utf-8'))
    print(f"VimQA live run - n={len(d['per_row'])}, reader {cfg.reader}  (below the detection threshold: do NOT read a ranking from it)")
    print(f"{'':14}{'EM':>7}{'F1':>8}{'F1*':>8}{'ratio':>8}")
    for m, lab in METHODS:
        r = d['summary'][m]
        print(f"{lab:14}{r['em']:7.1f}{r['f1']:8.2f}{r['f1_norm']:8.2f}{r['ratio']:8.3f}")


def print_effective_ratio(S: dict) -> None:
    """Ratio_eff = Ratio_nom × φ (MULTIPLIED by fertility; φ = 1 for English)."""
    fw = _load('fertility_wmt24pp.json')
    print(f"φ over {fw['n_pairs']} WMT24++ pairs (Vi/En):")
    for k, v in fw['tokenizers'].items():
        label = k.replace('(ratio trong bảng)', '(retention unit)')   # the JSON label is in Vietnamese
        print(f"  {label:32} {v['fertility']:.2f}×   [{v['ci_lo']:.2f}, {v['ci_hi']:.2f}]")
    phi = fw['tokenizers']['cl100k_base']['fertility']
    nom = S['VimQA']['itercomp']['ratio']
    print(f'\nVimQA: Ratio_nom = {nom:.3f}  →  Ratio_eff = {nom:.3f} × {phi:.2f} = {nom * phi:.3f}')
    print('English (φ = 1): ' + ', '.join(f"{k} {S[k]['itercomp']['ratio']:.3f} → {S[k]['itercomp']['ratio']:.3f}"
                                           for k in ('MuSiQue', '2Wiki', 'HotpotQA')))


def print_boolean_lift(S: dict) -> None:
    """ΔF1 = F1* − F1 over each full set: Vietnamese gains, English moves by exactly 0.0."""
    print('Boolean normalisation (ΔF1 = F1* − F1):')
    print(f'{"":14}' + ''.join(f'{k:>11}' for k in FILES_T3))
    for m, lab in (('raw', 'Raw'), ('itercomp', 'IterCOMP-X')):
        print(f'{lab:14}' + ''.join(f"{S[k][m]['f1_norm'] - S[k][m]['f1']:>+11.2f}" for k in FILES_T3))


def print_reader_dependence(S: dict) -> None:
    """IterCOMP-X minus Raw / LLMLingua-2 under different readers, and the share of Oracle on each set."""
    def delta(f, base):
        s = _load(f)['summary']
        return s['itercomp']['f1_norm'] - s[base]['f1_norm'], s['itercomp']['ratio']

    print(f"{'reader / dataset':34}{'n':>6}{'Ratio_nom':>11}{'Δ Raw':>9}{'Δ LLML-2':>10}")
    for lab, f, n in (('Qwen2.5-7B   / HotpotQA', 'hotpotqa_1500_7b.json', 1500),
                      ('Qwen2.5-0.5B / HotpotQA', 'hotpotqa_500_05b.json', 500),
                      ('Mistral-7B  / VimQA', 'vimqa_200_mistral7b.json', 200)):
        dr, ratio = delta(f, 'raw')
        print(f'{lab:34}{n:>6}{ratio:>11.3f}{dr:>+9.2f}{delta(f, "llmlingua2")[0]:>+10.2f}')
    dj, rj = delta('hotpotqa_500_05b_judge7b.json', 'raw')
    print(f"{'0.5B reader, 7B judge / HotpotQA':34}{500:>6}{rj:>11.3f}{dj:>+9.2f}{'-':>10}")
    print('\nIterCOMP-X as a share of Oracle:  ' + '  '.join(
        f"{k} {S[k]['itercomp']['f1_norm'] / S[k]['oracle']['f1_norm']:.0%}" for k in FILES_T3))


def print_vi_vs_en(S: dict) -> None:
    """Side-by-side table: what changes in Vietnamese (HotpotQA stands in for English)."""
    phi = _load('fertility_wmt24pp.json')['tokenizers']['cl100k_base']['fertility']
    e = _load('exp2_undiacritised_500_7b.json')

    def stat(k, fn):
        return fn(S[k]['itercomp'], S[k]['oracle'])
    en, vi = 'HotpotQA', 'VimQA'
    rows = [('φ (cl100k_base)', '1.00', f'{phi:.2f}'),
            ('Ratio_nom (IterCOMP-X)', f"{S[en]['itercomp']['ratio']:.3f}", f"{S[vi]['itercomp']['ratio']:.3f}"),
            ('Ratio_eff = Ratio_nom × φ', f"{S[en]['itercomp']['ratio']:.3f}", f"{S[vi]['itercomp']['ratio'] * phi:.3f}"),
            ('Boolean normalisation: ΔF1', *(f"{stat(k, lambda i, o: i['f1_norm'] - i['f1']):+.2f}" for k in (en, vi))),
            ('IterCOMP-X / Oracle', *(f"{stat(k, lambda i, o: i['f1_norm'] / o['f1_norm']):.0%}" for k in (en, vi))),
            ('No diacritics: ΔF1 IterCOMP-X / gold', 'n/a', f"{e['delta_itercomp']:+.1f} / {e['delta_gold']:+.1f}")]
    print(f"{'':32}{'English (HotpotQA)':>22}{'Vietnamese (VimQA)':>22}")
    for a, b, c in rows:
        print(f'{a:32}{b:>22}{c:>22}')
    print('\nNo diacritics (n=500): gold still leads in all three settings and both branches lose ~20 F1, so the loss is mainly the reader\'s.')
    print(f"{'':8}{'IterCOMP-X':>12}{'Gold':>8}{'Δ':>9}")
    for k, v in e['results'].items():
        print(f"{k:8}{v['f1_itercomp']:>12.2f}{v['f1_gold']:>8.2f}{v['f1_itercomp'] - v['f1_gold']:>+9.2f}")


def error_analysis(cfg: Config) -> None:
    """Categorise IterCOMP-X errors on VimQA from `per_row` (no new inference)."""
    live = cfg.outfile('vimqa', cfg.plan['vimqa'])
    path = live if os.path.exists(live) else 'src/results/vimqa_full_7b.json'
    rows = json.load(open(path, encoding='utf-8'))['per_row']
    n = len(rows)
    print(f'Analysing {n} VimQA questions  ({path})\n')
    m = lambda r: r['methods']['itercomp']
    buckets = collections.OrderedDict([('Fully correct (F1*=1)', 0), ('Partially correct (0<F1*<1)', 0),
                                       ('Fully wrong (F1*=0)', 0)])
    fmt = 0
    for r in rows:
        fn, fr = m(r)['f1_norm'], m(r)['f1']
        buckets[list(buckets)[0 if fn == 1 else 1 if fn > 0 else 2]] += 1
        fmt += fr == 0 and fn == 1                      # wrong raw, rescued by boolean normalisation
    for k, v in buckets.items():
        print(f'  {k:<26} {v:>4}  ({100 * v / n:4.0f}%)')
    print(f'\nYes/no format errors rescued by boolean normalisation (raw F1=0 → F1*=1): {fmt} ({100 * fmt / n:.0f}%)')
    pats = ('tuổi hơn', 'già hơn', 'trẻ hơn', 'lớn tuổi', 'nhỏ tuổi')
    age = [r for r in rows if any(p in r['question'].lower() for p in pats)]
    if age:
        rest = [r for r in rows if r not in age]
        zero = lambda rs: 100 * sum(m(r)['f1_norm'] == 0 for r in rs) / max(len(rs), 1)
        print(f'Age-comparison template: {len(age)}/{n} questions ({100 * len(age) / n:.1f}%); {zero(age):.1f}% score F1*=0 '
              f'against {zero(rest):.1f}% elsewhere - an arithmetic question, not a retrieval one.')


# ══════════════════════════════════════════════ demo: step-by-step trace
class Demo:
    """Trace one question through the IterCOMP-X loop, following MODE ('live' | 'cached' | 'mock').

    live: load reader/LLM/scorer ONCE and run for real (traces saved to src/results/demo_cache/);
    a saved trace is always the fallback if the live run fails.
    """
    CACHE_DIR = 'src/results/demo_cache'
    EXAMPLES = {('hotpotqa', 277): 'src/results/demo_trace_en_example.json',
                ('vimqa', 320): 'src/results/demo_trace_example.json',
                ('vimqa', 23): 'src/results/demo_trace_polar_example.json',
                ('vimqa', 2): 'src/results/demo_trace_vimqa_2_example.json',
                ('vimqa', 98): 'src/results/demo_trace_vimqa_98_example.json',
                ('vimqa', 458): 'src/results/demo_trace_vimqa_458_example.json'}

    def __init__(self, mode: str, cfg: Config):
        import scripts.demo_trace as dt
        self.mode, self.cfg, self.dt, self._live = mode, cfg, dt, {}
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    def _objs(self) -> dict:
        if not self._live:
            from itercomp import make_llm, make_reader, make_scorer
            from itercomp.scorer import DEFAULT_LAMBDA
            c = self.cfg
            self._live = {'reader': make_reader('hf', model=c.reader, load_4bit=c.load_4bit),
                          'llm': make_llm('hf', model=c.reader, load_4bit=c.load_4bit),
                          'scorer': make_scorer('dual', lam=DEFAULT_LAMBDA)}
        return self._live

    def _live_trace(self, index: int, dataset: str, strip: bool, cached: str) -> None:
        import types
        from itercomp import load_dataset
        o = self._objs()
        args = types.SimpleNamespace(max_iter=5, top=5, llmlingua2=True, strip_diacritics=strip, reader='hf',
                                     reader_model=self.cfg.reader, itercomp_llm='hf', dataset=dataset)
        row = load_dataset(dataset)[index]
        row = self.dt.strip_row(row) if strip else row
        t = self.dt.build_trace(args, row, index, o['reader'], o['llm'], o['scorer'],
                                self.dt.PERCENTILE_BY_DATASET[dataset])
        json.dump(t, open(cached, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        self.dt.render(t, color=False)

    def trace(self, index: int, dataset: str = 'vimqa', strip: bool = False) -> None:
        tag = ('trace_' if dataset == 'vimqa' else f'trace_{dataset}_') + str(index) + ('_strip' if strip else '')
        cached = f'{self.CACHE_DIR}/{tag}.json'
        fallback = cached if os.path.exists(cached) else (None if strip else self.EXAMPLES.get((dataset, index)))
        if self.mode == 'live':
            try:
                return self._live_trace(index, dataset, strip, cached)
            except Exception as e:
                print(f'! live run failed ({type(e).__name__}: {e}) - falling back to the saved trace\n')
        elif self.mode == 'mock':
            r = subprocess.run([sys.executable, 'src/scripts/demo_trace.py', '--no-color', '--dataset', dataset,
                                '--index', str(index), '--reader', 'mock', '--itercomp-llm', 'mock',
                                *(['--strip-diacritics'] if strip else [])], capture_output=True, text=True)
            return print(r.stdout + (r.stderr if r.returncode else ''))
        if fallback is None:
            return print("(a full no-diacritics trace needs MODE='live' - the measured numbers are in VI.3)" if strip
                         else "No saved trace for this case yet (run once with MODE='live' to create it).")
        print(f'[replaying saved trace: {fallback}]')
        self.dt.render(json.load(open(fallback, encoding='utf-8')), color=False)

    NEW_CONTEXT = {
        'title': ['Hồ Chí Minh', 'Hồ Chí Minh', 'Hà Nội', 'Đà Nẵng', 'Huế'],
        'sentences': [['Thành phố Hồ Chí Minh là thành phố lớn nhất Việt Nam về dân số.'],
                      ['Thành phố này trước năm 1976 có tên là Sài Gòn.'],
                      ['Hà Nội là thủ đô của Việt Nam, nằm ở miền Bắc.'],
                      ['Đà Nẵng là thành phố trực thuộc trung ương ở miền Trung.'],
                      ['Huế từng là kinh đô của triều Nguyễn.']]}
    NEW_QUESTION = 'Thành phố lớn nhất Việt Nam về dân số trước năm 1976 tên là gì?'
    NEW_ANSWER = 'Sài Gòn'

    def new_example(self, question: str | None = None, documents: dict | None = None,
                    answer: str = '') -> None:
        """Run the loop on a SELF-WRITTEN question and context (needs MODE='live').

        With no arguments it uses the built-in example. `documents` = {title: passage}; each
        passage is split into sentences at full stops. An empty `answer` skips F1.
        """
        if self.mode != 'live':
            return print("MODE != 'live': skipping the new-data demo (needs a GPU).")
        import re as _re
        from itercomp import itercomp
        from itercomp.metrics import f1_score, normalize_boolean
        if question is None:
            ctx, q, gold = self.NEW_CONTEXT, self.NEW_QUESTION, self.NEW_ANSWER
        else:
            ctx = {'title': list(documents),
                   'sentences': [[x.strip() for x in _re.split(r'(?<=[.!?])\s+', d) if x.strip()]
                                 for d in documents.values()]}
            q, gold = question, answer
        o = self._objs()
        print(f'QUESTION: {q}\nANSWER  : {gold or "(not given)"}\nCONTEXT : {sum(len(s) for s in ctx["sentences"])} sentences\n')
        res = itercomp(o['llm'], q, ctx, max_iter=5, scorer=o['scorer'], percentile=90)
        print(f'--- IterCOMP: {res.iterations} pass(es), stopped because: {self.dt.stop_reason(res.stopped_because)} ---')
        for k, fu in enumerate(res.followups, 2):
            print(f"  follow-up question for pass {k}: {fu.strip()[:90]}")
        for s in res.selected:
            print(f'  kept: {str(s)[:70]}')
        full = '\n'.join(f'{t}: {" ".join(s)}' for t, s in zip(ctx['title'], ctx['sentences']))
        for ten, c in (('raw (everything)', full), ('itercomp (compressed)', res.to_prompt())):
            pred = o['reader'](c, q)
            score = f'  F1={f1_score(normalize_boolean(pred, gold, q), gold):.2f}' if gold else ''
            print(f'\n{ten:20s} -> {pred[:50]!r}{score}')
        if question is None:
            print('\nMulti-hop: it must link "the largest city" to "its name before 1976".')


# ══════════════════════════════════════════════ appendix A: extra experiments (RUN_EXTRAS)
_EVID = textwrap.dedent("""
    import json, sys
    sys.path.insert(0, 'src')
    from itercomp import load_dataset, itercomp, make_llm, make_scorer
    from itercomp.metrics import tokenize
    tag, reader, four = sys.argv[1], sys.argv[2], sys.argv[3] == '1'
    sc = make_scorer('dual')
    llm = make_llm('mock') if tag == 'mock' else make_llm('hf', model=reader, load_4bit=four)
    it = nz = n = 0
    why = {}
    for r in load_dataset('vimqa', 10):
        res = itercomp(llm, r['question'], r['context'], max_iter=5, scorer=sc, percentile=90)
        full = '\\n'.join(f'{t}: {" ".join(s)}' for t, s in zip(r['context']['title'], r['context']['sentences']))
        it += res.iterations
        nz += len(tokenize(res.to_prompt())) / max(len(tokenize(full)), 1)
        why[res.stopped_because] = why.get(res.stopped_because, 0) + 1
        n += 1
    print('RESULT ' + json.dumps([tag, it / n, nz / n * 100, why]))
""")


def _loop_evidence(cfg: Config, gpu: bool) -> None:
    """Does the loop really iterate? The mock judge always says YES in pass 1, so it degenerates to one-pass filtering.

    Each tag runs in a SEPARATE PROCESS: bitsandbytes' 4-bit buffers release VRAM only when the process exits.
    """
    rows = []
    for tag in ('mock', 'hf'):
        print(f'--- {tag} ---', flush=True)
        p = subprocess.run([sys.executable, '-c', _EVID, tag, cfg.reader, '1' if cfg.load_4bit else '0'],
                           capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith('RESULT ')), None)
        if line is None:
            print(p.stdout[-2000:], p.stderr[-2000:])
            raise SystemExit(f'tag {tag} returned no result')
        rows.append(json.loads(line[len('RESULT '):]))
        print(f'  passes={rows[-1][1]:.2f}  retention={rows[-1][2]:.1f}%', flush=True)
    json.dump([{'tag': t, 'iters': i, 'ratio': z, 'stop_reason': w} for t, i, z, w in rows],
              open('results/loop_evidence.json', 'w'), ensure_ascii=False, indent=1)
    print(f"\n{'backend':10s}{'avg passes':>11s}{'retention':>11s}  stop reasons")
    for t, i, z, w in rows:
        print(f'{t:10s}{i:11.2f}{z:10.1f}%  {w}')
    assert rows[1][1] > rows[0][1], 'hf does not iterate more than mock -> the hf backend is not working'
    print('\nOK: the hf backend iterates more than mock.')


def _k_sweep() -> None:
    """Sweep k to match the original paper's 0.14 retention on MuSiQue. Runs on CPU (no reader needed)."""
    from itercomp import itercomp, load_dataset, make_llm, make_scorer
    from itercomp.metrics import tokenize
    sc, llm = make_scorer('dual'), make_llm('mock')
    for ds, target in (('musique', 0.14), ('vimqa', None)):
        rows = load_dataset(ds, 50)
        print(f'\n{ds}  (n={len(rows)})' + (f'  - target {target:.2f}' if target else ''))
        print(f"  {'k':>5}{'retention':>12}{'gap':>9}")
        best = None
        for k in (88, 90, 92, 94, 95, 96, 97):
            tot = 0.0
            for r in rows:
                res = itercomp(llm, r['question'], r['context'], max_iter=5, scorer=sc, percentile=k)
                full = '\n'.join(f'{t}: {" ".join(s)}' for t, s in zip(r['context']['title'], r['context']['sentences']))
                tot += len(tokenize(res.to_prompt())) / max(len(tokenize(full)), 1)
            ratio = tot / len(rows)
            gap = f'{(ratio - target) * 100:+8.1f}pp' if target else ''
            print(f'  {k:>5}{ratio * 100:11.1f}%{gap:>9}')
            if target and (best is None or abs(ratio - target) < best[1]):
                best = (k, abs(ratio - target))
        if target:
            print(f'  -> k={best[0]} is closest to {target:.2f}')


def extra(name: str, cfg: Config, run: bool, gpu: bool) -> None:
    """One appendix-A experiment: 'loop' | 'ksweep' | 'ablation' | 'exp1' | 'exp2' | 'lambda'."""
    if not run or (name != 'ksweep' and not gpu):
        return print(f'RUN_EXTRAS=False or no GPU: skipping {name}.')
    if name == 'loop':
        return _loop_evidence(cfg, gpu)
    if name == 'ksweep':
        return _k_sweep()
    n, ra, mt = cfg.small_n(), _reader_args(cfg), cfg.mtag
    if name == 'ablation':
        out = f'results/ablation_vimqa_{n}_{mt}_{cfg.tag}.json'
        _run_script(cfg, 'src/scripts/run_ablation.py',
                    ['--dataset', 'vimqa', '--limit', str(n), '--llm', cfg.tag, *ra], out,
                    f'results/log_ablation_vimqa_{n}_{mt}.txt', f'Ablation vimqa n={n} (26 configurations)')
    elif name == 'exp1':
        for ds in ('vimqa', 'musique'):
            _run_script(cfg, 'src/scripts/exp1_oracle_stopping.py',
                        ['--dataset', ds, '--limit', str(n), *ra, '--scorer', 'dual', '--percentile', '90'],
                        f'results/exp1_oracle_{ds}_{n}_{mt}.json', f'results/log_exp1_{ds}_{n}_{mt}.txt',
                        f'EXP-1 {ds} n={n}')
    elif name == 'exp2':
        _run_script(cfg, 'src/scripts/exp2_undiacritised.py',
                    ['--limit', str(n), *ra, '--scorer', 'dual', '--percentile', '90'],
                    f'results/exp2_undiacritised_{n}_{mt}.json', f'results/log_exp2_{n}_{mt}.txt',
                    f'EXP-2 no diacritics n={n}')
    elif name == 'lambda':
        _run_script(cfg, 'src/scripts/run_lambda_sweep.py', ['--dataset', 'vimqa', '--limit', str(n), *ra],
                    f'results/lambda_sweep_vimqa_{n}_{mt}.json', f'results/log_lambda_{n}_{mt}.txt',
                    f'λ/k sweep vimqa n={n}')
    else:
        raise ValueError(name)


# ══════════════════════════════════════════════ appendix B
def zip_results(base: str) -> None:
    """Zip results/ and download it (Colab: files.download; Kaggle: the file appears under Output)."""
    import shutil
    path = shutil.make_archive(os.path.join(base, 'results'), 'zip', 'results')
    print(f'{path}  ({os.path.getsize(path) / 1e6:.1f} MB)')
    try:
        from google.colab import files
        files.download(path)
    except ImportError:
        print(f'Not on Colab - the file is at:\n  {path}\nOn Kaggle: Save Version → Output → download results.zip')
