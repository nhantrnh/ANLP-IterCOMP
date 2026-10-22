"""Hàm phụ cho notebooks/colab_reproduce.ipynb.

Notebook dùng như bộ slide nên mỗi ô chỉ gọi vài dòng; phần logic dài nằm ở đây.
Mọi hàm chạy với thư mục làm việc là gốc repo (notebook đã `os.chdir(REPO)`) và
đọc đường dẫn tương đối `data/`, `results/`, `src/results/`.

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

# ══════════════════════════════════════════════ dữ liệu
DATA_FILES = {
    'vimqa/validation.parquet':    ('nguyenlab/vimqa', 'data/validation-00000-of-00001.parquet'),
    '2wiki/dev.parquet':           ('xanhho/2WikiMultihopQA', 'dev.parquet'),
    'hotpotqa/validation.parquet': ('hotpotqa/hotpot_qa', 'distractor/validation-00000-of-00001.parquet'),
    'musique/dev.jsonl':           ('dgslibisey/MuSiQue', 'musique_ans_v1.0_dev.jsonl'),
}


def download_data(all_sets: bool) -> None:
    """Tải VimQA (+ HotpotQA cho câu tiếng Anh của demo); `all_sets` tải cả 2Wiki, MuSiQue."""
    for dst, (repo, src) in DATA_FILES.items():
        if not all_sets and not dst.startswith(('vimqa', 'hotpotqa')):
            continue
        p = 'data/' + dst
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if not os.path.exists(p):
            urllib.request.urlretrieve(f'https://huggingface.co/datasets/{repo}/resolve/main/{src}', p)
        print(f'{dst:32s} {os.path.getsize(p) / 1e6:6.1f} MB')


def peek_datasets() -> None:
    """Mỗi bộ đã tải: số đoạn văn và số câu bằng chứng vàng của câu đầu tiên."""
    from itercomp import load_dataset
    paths = {'hotpotqa': 'data/hotpotqa/validation.parquet', '2wiki': 'data/2wiki/dev.parquet',
             'musique': 'data/musique/dev.jsonl', 'vimqa': 'data/vimqa/validation.parquet'}
    for ds, path in paths.items():
        if os.path.exists(path):
            r = load_dataset(ds, 1)[0]
            n = len(r.get('supporting_facts', {}).get('title', []))
            print(f"{ds:9s} {len(r['context']['title']):2d} đoạn | supporting_facts={n}")


def eda_vimqa(n: int = 200) -> None:
    """Thống kê mô tả VimQA, phân bố loại đáp án, phân rã segment, kiểm tra tokenizer."""
    from itercomp import load_dataset
    from itercomp.core import decompose
    from itercomp.metrics import tokenize

    rows = load_dataset('vimqa', n)
    n_para = [len(r['context']['title']) for r in rows]
    n_tok = [len(tokenize('\n'.join(' '.join(s) for s in r['context']['sentences']))) for r in rows]
    n_gold = [len(r.get('supporting_facts', {}).get('title', [])) for r in rows]
    print(f'VimQA - {len(rows)} câu hỏi')
    for lab, v in (('đoạn/câu', n_para), ('token/ngữ cảnh', n_tok), ('câu bằng chứng vàng/câu hỏi', n_gold)):
        print(f'  {lab:28s}: trung vị {statistics.median(v):.0f}  (min {min(v)}, max {max(v)})')

    boolean = {'đúng', 'không', 'sai', 'có'}
    kind = collections.Counter('đúng/sai' if str(r['answer']).strip().lower() in boolean else 'trích xuất'
                               for r in rows)
    print('\nloại đáp án:')
    for k, v in kind.most_common():
        print(f'  {k:12s} {v:3d}  {v / len(rows):5.1%}')

    r = rows[0]
    segs = decompose(r['context'])
    print(f"\nphân rã câu đầu: {len(r['context']['title'])} đoạn -> {len(segs)} segment câu")
    for s in segs[:3]:
        print(f'  [{s.doc_title[:22]:24s}] {str(s)[:58]}…')
    assert tokenize('vào') == ['vào'], 'tokenizer làm hỏng dấu tiếng Việt'
    print('tokenizer giữ dấu:', tokenize('má mà mã'))


# ══════════════════════════════════════════════ cấu hình
SCALES = {
    'demo':  {'vimqa': 30},
    'paper': {'vimqa': None, 'musique': 1500, 'hotpotqa': 1500, '2wiki': 1500},
    'max':   {'vimqa': None, 'musique': None},
}
SEC_PER_Q = {'vimqa': 9.6, 'musique': 15.0, 'hotpotqa': 6.5, '2wiki': 6.2}   # đo thật trên T4, 4 phương pháp/câu
FULL_N = {'vimqa': 1003, 'musique': 2417, 'hotpotqa': 7405, '2wiki': 12576}


def nlabel(n) -> str:
    return 'full' if n is None else str(n)


@dataclass
class Config:
    """Cỡ mẫu và mô hình đọc. Hậu tố file lấy từ mô hình THẬT để số 1.5B không bị chép thành số 7B."""
    scale: str
    reader: str = 'Qwen/Qwen2.5-7B-Instruct'
    load_4bit: bool = True
    tag: str = 'hf'
    plan: dict = field(init=False)
    mtag: str = field(init=False)

    def __post_init__(self):
        assert self.scale in SCALES, f'SCALE phải thuộc {list(SCALES)}'
        self.plan = SCALES[self.scale]
        self.mtag = self.reader.split('/')[-1].replace('Qwen2.5-', '').replace('-Instruct', '').lower()
        os.makedirs('results', exist_ok=True)

    def outfile(self, ds: str, n, prefix: str = '') -> str:
        return f'results/{prefix}{ds}_{nlabel(n)}_{self.mtag}.json'

    def small_n(self, n_full: int = 200) -> int:
        """Cỡ mẫu của các thí nghiệm phụ (ablation, EXP-1/2, quét λ/k)."""
        return n_full if self.scale != 'smoke' else 50

    def print_plan(self) -> None:
        print(f'reader = {self.reader} | 4-bit = {self.load_4bit} | SCALE = {self.scale!r}\n')
        print(f"{'dataset':10s} {'n':>7s} {'ước tính':>10s}")
        tot = 0.0
        for ds, n in self.plan.items():
            h = (FULL_N[ds] if n is None else n) * SEC_PER_Q[ds] / 3600
            tot += h
            print(f'{ds:10s} {nlabel(n):>7s} {h:9.2f}h')
        print(f"{'TỔNG':10s} {'':>7s} {tot:9.2f}h" + ('  vượt một phiên Colab 12h' if tot > 12 else ''))


def run_stream(cmd: list, logfile: str):
    """Chạy tiến trình con, in output NGAY (Colab hay ngắt khi ô im lặng quá lâu) và lưu log."""
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
    print('\n' + '=' * 70 + f'\n### {title}   bắt đầu {time.strftime("%H:%M")}\n' + '=' * 70, flush=True)


def _run_script(cfg: Config, script: str, extra: list, out: str, log: str, label: str) -> None:
    """Chạy một script thí nghiệm nếu chưa có `out`; báo lỗi rõ ràng nếu thiếu file kết quả."""
    if os.path.exists(out):
        print(f'[bỏ qua] {label} đã có ({out})')
        return
    _banner(label)
    t0 = time.time()
    rc, _ = run_stream([sys.executable, '-u', script, *extra, '--out', out], log)
    if rc:
        print(f'!!! {label} LỖI (mã {rc}) - xem log')
    elif not os.path.exists(out):
        print(f'!!! {label} chạy xong nhưng KHÔNG có {out}')
    print(f'--> {(time.time() - t0) / 60:.1f} phút', flush=True)


def run_main_eval(cfg: Config, gpu: bool) -> None:
    """Đánh giá Raw / Oracle / LLMLingua-2 / IterCOMP-X theo `cfg.plan` (checkpoint sau MỖI câu)."""
    if not gpu:
        print('Không có GPU: bỏ qua đánh giá thật. Số ở quy mô báo cáo xem VI.1 (đọc từ JSON đã commit).')
        return
    for ds in [d for d in ('vimqa', 'musique', 'hotpotqa', '2wiki') if d in cfg.plan]:
        n = cfg.plan[ds]
        _run_script(cfg, 'src/scripts/run_eval.py',
                    ['--dataset', ds, *(['--limit', str(n)] if n is not None else []), *_reader_args(cfg),
                     '--methods', 'raw,oracle,llmlingua2,itercomp',
                     '--itercomp-llm', cfg.tag, '--scorer', 'dual'],   # LLM thật cho mọi bước suy luận
                    cfg.outfile(ds, n), f'results/log_{ds}_{nlabel(n)}_{cfg.mtag}.txt', f'{ds} n={nlabel(n)}')
    print('\nXONG bảng chính.')


# ══════════════════════════════════════════════ bảng kết quả (đọc JSON, không gõ tay)
FILES_T3 = {'VimQA': 'vimqa_full_7b.json', 'MuSiQue': 'musique_1500_7b.json',
            '2Wiki': '2wiki_1500_7b.json', 'HotpotQA': 'hotpotqa_1500_7b.json'}
METHODS = (('oracle', 'Oracle'), ('raw', 'Raw'), ('itercomp', 'IterCOMP-X'), ('llmlingua2', 'LLMLingua-2'))


def _load(f: str) -> dict:
    return json.load(open(f'src/results/{f}', encoding='utf-8'))


def load_table3() -> dict:
    """Tóm tắt của bốn lần chạy đầy đủ (VimQA n=1003; ba bộ tiếng Anh n=1500)."""
    S = {}
    for name, f in FILES_T3.items():
        d = _load(f)
        S[name] = d['summary']
        S[name]['n'] = len(d['per_row'])
    return S


def _row(S: dict, label: str, fn) -> None:
    print(f'{label:16}' + ''.join(f'{fn(S[k]):>11}' for k in FILES_T3))


def print_table3(S: dict) -> None:
    print('Bảng chính của bài (Qwen2.5-7B 4-bit)')
    print(f'{"F1*":16}' + ''.join(f'{k:>11}' for k in FILES_T3))
    _row(S, 'n', lambda s: s['n'])
    for m, lab in METHODS:
        _row(S, lab, lambda s, m=m: f"{s[m]['f1_norm']:.2f}")
    _row(S, 'Δ vs Raw', lambda s: f"{s['itercomp']['f1_norm'] - s['raw']['f1_norm']:+.2f}")
    _row(S, 'Δ vs LLML-2', lambda s: f"{s['itercomp']['f1_norm'] - s['llmlingua2']['f1_norm']:+.2f}")
    _row(S, 'Ratio_nom', lambda s: f"{s['itercomp']['ratio']:.3f}")
    print('Khoảng tin cậy (†) xem Bảng 3 của bài. HotpotQA: Raw thắng IterCOMP-X - kết quả âm được giữ nguyên.')


def print_live_result(cfg: Config) -> None:
    """Kết quả của lần chạy thật vừa rồi (n nhỏ: chỉ chứng tỏ pipeline chạy)."""
    path = cfg.outfile('vimqa', cfg.plan['vimqa'])
    if not os.path.exists(path):
        print('Chưa có kết quả chạy thật (không GPU hoặc VI.4 bị bỏ qua) - xem bảng chính ở VI.1.')
        return
    d = json.load(open(path, encoding='utf-8'))
    print(f"VimQA vừa chạy - n={len(d['per_row'])}, reader {cfg.reader}  (dưới ngưỡng phát hiện: KHÔNG kết luận xếp hạng)")
    print(f"{'':14}{'EM':>7}{'F1':>8}{'F1*':>8}{'ratio':>8}")
    for m, lab in METHODS:
        r = d['summary'][m]
        print(f"{lab:14}{r['em']:7.1f}{r['f1']:8.2f}{r['f1_norm']:8.2f}{r['ratio']:8.3f}")


def print_effective_ratio(S: dict) -> None:
    """Ratio_eff = Ratio_nom × φ (NHÂN với fertility; tiếng Anh φ = 1)."""
    fw = _load('fertility_wmt24pp.json')
    print(f"φ trên {fw['n_pairs']} cặp WMT24++ (Vi/En):")
    for k, v in fw['tokenizers'].items():
        print(f"  {k:32} {v['fertility']:.2f}×   [{v['ci_lo']:.2f}, {v['ci_hi']:.2f}]")
    phi = fw['tokenizers']['cl100k_base']['fertility']
    nom = S['VimQA']['itercomp']['ratio']
    print(f'\nVimQA: Ratio_nom = {nom:.3f}  →  Ratio_eff = {nom:.3f} × {phi:.2f} = {nom * phi:.3f}')
    print('Tiếng Anh (φ = 1): ' + ', '.join(f"{k} {S[k]['itercomp']['ratio']:.3f} → {S[k]['itercomp']['ratio']:.3f}"
                                           for k in ('MuSiQue', '2Wiki', 'HotpotQA')))


def print_boolean_lift(S: dict) -> None:
    """ΔF1 = F1* − F1 trên toàn bộ tập: tiếng Việt được cộng điểm, tiếng Anh đúng 0.0."""
    print('Chuẩn hoá boolean (ΔF1 = F1* − F1):')
    print(f'{"":14}' + ''.join(f'{k:>11}' for k in FILES_T3))
    for m, lab in (('raw', 'Raw'), ('itercomp', 'IterCOMP-X')):
        print(f'{lab:14}' + ''.join(f"{S[k][m]['f1_norm'] - S[k][m]['f1']:>+11.2f}" for k in FILES_T3))


def print_reader_dependence(S: dict) -> None:
    """IterCOMP-X trừ Raw / LLMLingua-2 dưới các reader khác nhau, và % Oracle ở mỗi bộ."""
    def delta(f, base):
        s = _load(f)['summary']
        return s['itercomp']['f1_norm'] - s[base]['f1_norm'], s['itercomp']['ratio']

    print(f"{'reader / dữ liệu':34}{'n':>6}{'Ratio_nom':>11}{'Δ Raw':>9}{'Δ LLML-2':>10}")
    for lab, f, n in (('Qwen2.5-7B   / HotpotQA', 'hotpotqa_1500_7b.json', 1500),
                      ('Qwen2.5-0.5B / HotpotQA', 'hotpotqa_500_05b.json', 500),
                      ('Mistral-7B  / VimQA', 'vimqa_200_mistral7b.json', 200)):
        dr, ratio = delta(f, 'raw')
        print(f'{lab:34}{n:>6}{ratio:>11.3f}{dr:>+9.2f}{delta(f, "llmlingua2")[0]:>+10.2f}')
    dj, rj = delta('hotpotqa_500_05b_judge7b.json', 'raw')
    print(f"{'0.5B đọc, judge ghim 7B / HotpotQA':34}{500:>6}{rj:>11.3f}{dj:>+9.2f}{'-':>10}")
    print('\nIterCOMP-X đạt bao nhiêu % Oracle:  ' + '  '.join(
        f"{k} {S[k]['itercomp']['f1_norm'] / S[k]['oracle']['f1_norm']:.0%}" for k in FILES_T3))


def print_vi_vs_en(S: dict) -> None:
    """Bảng đặt cạnh nhau: điều gì khác ở tiếng Việt (HotpotQA làm đại diện tiếng Anh)."""
    phi = _load('fertility_wmt24pp.json')['tokenizers']['cl100k_base']['fertility']
    e = _load('exp2_undiacritised_500_7b.json')

    def stat(k, fn):
        return fn(S[k]['itercomp'], S[k]['oracle'])
    en, vi = 'HotpotQA', 'VimQA'
    rows = [('φ (cl100k_base)', '1.00', f'{phi:.2f}'),
            ('Ratio_nom (IterCOMP-X)', f"{S[en]['itercomp']['ratio']:.3f}", f"{S[vi]['itercomp']['ratio']:.3f}"),
            ('Ratio_eff = Ratio_nom × φ', f"{S[en]['itercomp']['ratio']:.3f}", f"{S[vi]['itercomp']['ratio'] * phi:.3f}"),
            ('Chuẩn hoá boolean: ΔF1', *(f"{stat(k, lambda i, o: i['f1_norm'] - i['f1']):+.2f}" for k in (en, vi))),
            ('IterCOMP-X / Oracle', *(f"{stat(k, lambda i, o: i['f1_norm'] / o['f1_norm']):.0%}" for k in (en, vi))),
            ('Bỏ dấu: ΔF1 IterCOMP-X / gold', 'không áp dụng', f"{e['delta_itercomp']:+.1f} / {e['delta_gold']:+.1f}")]
    print(f"{'':32}{'Tiếng Anh (HotpotQA)':>22}{'Tiếng Việt (VimQA)':>22}")
    for a, b, c in rows:
        print(f'{a:32}{b:>22}{c:>22}')
    print('\nBỏ dấu (n=500): gold vẫn dẫn ở cả ba thiết lập, cả hai nhánh mất ~20 F1 → suy giảm nằm ở reader.')
    print(f"{'':8}{'IterCOMP-X':>12}{'Gold':>8}{'Δ':>9}")
    for k, v in e['results'].items():
        print(f"{k:8}{v['f1_itercomp']:>12.2f}{v['f1_gold']:>8.2f}{v['f1_itercomp'] - v['f1_gold']:>+9.2f}")


def error_analysis(cfg: Config) -> None:
    """Phân loại lỗi IterCOMP-X trên VimQA từ `per_row` (không suy luận lại)."""
    live = cfg.outfile('vimqa', cfg.plan['vimqa'])
    path = live if os.path.exists(live) else 'src/results/vimqa_full_7b.json'
    rows = json.load(open(path, encoding='utf-8'))['per_row']
    n = len(rows)
    print(f'Phân tích {n} câu VimQA  ({path})\n')
    m = lambda r: r['methods']['itercomp']
    buckets = collections.OrderedDict([('Đúng hoàn toàn (F1*=1)', 0), ('Đúng một phần (0<F1*<1)', 0),
                                       ('Sai hoàn toàn (F1*=0)', 0)])
    fmt = 0
    for r in rows:
        fn, fr = m(r)['f1_norm'], m(r)['f1']
        buckets[list(buckets)[0 if fn == 1 else 1 if fn > 0 else 2]] += 1
        fmt += fr == 0 and fn == 1                      # thô sai, chuẩn hoá boolean cứu
    for k, v in buckets.items():
        print(f'  {k:<26} {v:>4}  ({100 * v / n:4.0f}%)')
    print(f'\nLỗi định dạng yes/no được boolean-norm cứu (F1 thô=0 → F1*=1): {fmt} ({100 * fmt / n:.0f}%)')
    pats = ('tuổi hơn', 'già hơn', 'trẻ hơn', 'lớn tuổi', 'nhỏ tuổi')
    age = [r for r in rows if any(p in r['question'].lower() for p in pats)]
    if age:
        rest = [r for r in rows if r not in age]
        zero = lambda rs: 100 * sum(m(r)['f1_norm'] == 0 for r in rs) / max(len(rs), 1)
        print(f'Mẫu so-sánh tuổi: {len(age)}/{n} câu ({100 * len(age) / n:.1f}%); {zero(age):.1f}% bị F1*=0 '
              f'so với {zero(rest):.1f}% ở phần còn lại - câu hỏi số học, không phải truy hồi.')


# ══════════════════════════════════════════════ demo: vết từng bước
class Demo:
    """Vết một câu hỏi qua vòng lặp IterCOMP-X, theo MODE ('live' | 'cached' | 'mock').

    live: nạp reader/LLM/scorer MỘT lần rồi chạy thật (lưu vết vào src/results/demo_cache/);
    luôn có đường lùi về vết đã lưu nếu chạy thật lỗi.
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
                print(f'! chạy thật lỗi ({type(e).__name__}: {e}) - lùi về vết đã lưu\n')
        elif self.mode == 'mock':
            r = subprocess.run([sys.executable, 'src/scripts/demo_trace.py', '--no-color', '--dataset', dataset,
                                '--index', str(index), '--reader', 'mock', '--itercomp-llm', 'mock',
                                *(['--strip-diacritics'] if strip else [])], capture_output=True, text=True)
            return print(r.stdout + (r.stderr if r.returncode else ''))
        if fallback is None:
            return print("(vết không dấu đầy đủ chỉ có khi chạy MODE='live' - số đo thật ở bảng VI.3)" if strip
                         else "Chưa có vết đã lưu cho trường hợp này (chạy MODE='live' một lần để tạo).")
        print(f'[phát lại vết đã lưu: {fallback}]')
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
        """Chạy vòng lặp trên câu hỏi + ngữ cảnh TỰ VIẾT (cần MODE='live').

        Không truyền gì: dùng ví dụ có sẵn. `documents` = {tiêu đề: đoạn văn}; mỗi
        đoạn được tách thành câu theo dấu chấm. `answer` để trống thì không chấm F1.
        """
        if self.mode != 'live':
            return print("MODE != 'live': bỏ qua demo dữ liệu mới (cần GPU).")
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
        print(f'CÂU HỎI : {q}\nĐÁP ÁN  : {gold or "(không cho)"}\nNGỮ CẢNH: {sum(len(s) for s in ctx["sentences"])} câu\n')
        res = itercomp(o['llm'], q, ctx, max_iter=5, scorer=o['scorer'], percentile=90)
        print(f'--- IterCOMP: {res.iterations} vòng, dừng vì "{res.stopped_because}" ---')
        for k, fu in enumerate(res.followups, 2):
            print(f"  câu hỏi phụ cho vòng {k}: {fu.strip()[:90]}")
        for s in res.selected:
            print(f'  giữ: {str(s)[:70]}')
        full = '\n'.join(f'{t}: {" ".join(s)}' for t, s in zip(ctx['title'], ctx['sentences']))
        for ten, c in (('raw (toàn bộ)', full), ('itercomp (đã nén)', res.to_prompt())):
            pred = o['reader'](c, q)
            score = f'  F1={f1_score(normalize_boolean(pred, gold, q), gold):.2f}' if gold else ''
            print(f'\n{ten:20s} -> {pred[:50]!r}{score}')
        if question is None:
            print('\nCâu multi-hop: phải nối "thành phố lớn nhất" với "tên trước 1976".')


# ══════════════════════════════════════════════ phụ lục A: thí nghiệm mở rộng (RUN_EXTRAS)
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
    """Vòng lặp có thật sự chạy không: mock luôn trả YES ở vòng 1 nên suy thoái thành lọc một lần.

    Mỗi tag chạy trong TIẾN TRÌNH RIÊNG: buffer 4-bit của bitsandbytes chỉ trả VRAM khi tiến trình thoát.
    """
    rows = []
    for tag in ('mock', 'hf'):
        print(f'--- {tag} ---', flush=True)
        p = subprocess.run([sys.executable, '-c', _EVID, tag, cfg.reader, '1' if cfg.load_4bit else '0'],
                           capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith('RESULT ')), None)
        if line is None:
            print(p.stdout[-2000:], p.stderr[-2000:])
            raise SystemExit(f'tag {tag} không trả kết quả')
        rows.append(json.loads(line[len('RESULT '):]))
        print(f'  vòng={rows[-1][1]:.2f}  tỉ lệ={rows[-1][2]:.1f}%', flush=True)
    json.dump([{'tag': t, 'iters': i, 'ratio': z, 'stop_reason': w} for t, i, z, w in rows],
              open('results/loop_evidence.json', 'w'), ensure_ascii=False, indent=1)
    print(f"\n{'backend':10s}{'vòng TB':>9s}{'tỉ lệ nén':>11s}  lý do dừng")
    for t, i, z, w in rows:
        print(f'{t:10s}{i:9.2f}{z:10.1f}%  {w}')
    assert rows[1][1] > rows[0][1], 'hf không lặp nhiều vòng hơn mock -> backend hf chưa hoạt động'
    print('\nOK: backend hf lặp nhiều vòng hơn mock.')


def _k_sweep() -> None:
    """Quét k để khớp tỉ lệ nén 0.14 của bài gốc trên MuSiQue. Chạy được trên CPU (không cần reader)."""
    from itercomp import itercomp, load_dataset, make_llm, make_scorer
    from itercomp.metrics import tokenize
    sc, llm = make_scorer('dual'), make_llm('mock')
    for ds, target in (('musique', 0.14), ('vimqa', None)):
        rows = load_dataset(ds, 50)
        print(f'\n{ds}  (n={len(rows)})' + (f'  - mục tiêu {target:.2f}' if target else ''))
        print(f"  {'k':>5}{'tỉ lệ nén':>12}{'lệch':>9}")
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
            print(f'  -> k={best[0]} bám sát {target:.2f} nhất')


def extra(name: str, cfg: Config, run: bool, gpu: bool) -> None:
    """Một thí nghiệm của phụ lục A: 'loop' | 'ksweep' | 'ablation' | 'exp1' | 'exp2' | 'lambda'."""
    if not run or (name != 'ksweep' and not gpu):
        return print(f'RUN_EXTRAS=False hoặc không GPU: bỏ qua {name}.')
    if name == 'loop':
        return _loop_evidence(cfg, gpu)
    if name == 'ksweep':
        return _k_sweep()
    n, ra, mt = cfg.small_n(), _reader_args(cfg), cfg.mtag
    if name == 'ablation':
        out = f'results/ablation_vimqa_{n}_{mt}_{cfg.tag}.json'
        _run_script(cfg, 'src/scripts/run_ablation.py',
                    ['--dataset', 'vimqa', '--limit', str(n), '--llm', cfg.tag, *ra], out,
                    f'results/log_ablation_vimqa_{n}_{mt}.txt', f'Ablation vimqa n={n} (26 cấu hình)')
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
                    f'EXP-2 không dấu n={n}')
    elif name == 'lambda':
        _run_script(cfg, 'src/scripts/run_lambda_sweep.py', ['--dataset', 'vimqa', '--limit', str(n), *ra],
                    f'results/lambda_sweep_vimqa_{n}_{mt}.json', f'results/log_lambda_{n}_{mt}.txt',
                    f'Quét λ,k vimqa n={n}')
    else:
        raise ValueError(name)


# ══════════════════════════════════════════════ phụ lục B
def zip_results(base: str) -> None:
    """Đóng gói results/ và tải về (Colab: files.download; Kaggle: file xuất hiện ở tab Output)."""
    import shutil
    path = shutil.make_archive(os.path.join(base, 'results'), 'zip', 'results')
    print(f'{path}  ({os.path.getsize(path) / 1e6:.1f} MB)')
    try:
        from google.colab import files
        files.download(path)
    except ImportError:
        print(f'Không phải Colab - lấy file ở:\n  {path}\nTrên Kaggle: Save Version → Output → tải results.zip')
