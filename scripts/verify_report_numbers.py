"""Đối chiếu MỌI con số trong report.tex VÀ PROPOSAL.md với results/ — tự động.

VÌ SAO CẦN. Báo cáo trích hàng chục con số từ `results/`. Sửa một thí nghiệm rồi
quên cập nhật báo cáo là lỗi không ai phát hiện bằng mắt, và nó không gây lỗi
biên dịch — PDF vẫn đẹp, chỉ là số sai. Script này chạy trong vài giây và nói rõ
chỗ nào lệch.

Đã bắt được thật: con số "81% Oracle headroom" hoá ra là IterCOMP/Oracle chứ
không phải phần khoảng-trống bắt được (đại lượng đó âm), nên phải sửa cách diễn
đạt trong báo cáo.

    python scripts/verify_report_numbers.py
"""
import json, re, sys, os
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[1])   # gốc repo, không cứng hoá
tex = open("report_en/report.tex", encoding="utf-8").read()

# Bản hội nghị (nếu có) — nó gitignore nên dễ bị bỏ ngoài mọi kiểm tra, mà
# abstract của nó lại là chỗ nhiều số viết lại nhất.
_paper = Path("paper_submission/paper.tex")
paper_tex = _paper.read_text() if _paper.is_file() else ""

def load(f):
    try: return json.load(open(f"results/{f}.json", encoding="utf-8"))
    except Exception: return None

def S(f, m, k="f1_norm"):
    d = load(f)
    return d["summary"][m][k] if d else None

checks = []
def chk(label, claimed, actual, tol=0.15):
    ok = actual is not None and abs(claimed - actual) <= tol
    checks.append((ok, label, claimed, actual))

# --- bang chinh vs paper (dong 250-262), n=50 ---
v50, m50 = load("vimqa_50_7b"), load("musique_50_7b")
for meth, mus, vim in (("oracle",57.0,53.3),("raw",32.8,43.8),
                       ("llmlingua2",13.5,30.7),("itercomp",35.1,43.0)):
    chk(f"tab:vspaper MuSiQue {meth}", mus, m50["summary"][meth]["f1_norm"])
    chk(f"tab:vspaper VimQA   {meth}", vim, v50["summary"][meth]["f1_norm"])

# ratio + Oracle/Raw
chk("tab:vspaper MuSiQue ratio", 0.32, m50["summary"]["itercomp"]["ratio"], 0.006)
chk("tab:vspaper VimQA   ratio", 0.19, v50["summary"]["itercomp"]["ratio"], 0.006)
chk("Oracle/Raw MuSiQue", 1.74, m50["summary"]["oracle"]["f1_norm"]/m50["summary"]["raw"]["f1_norm"], 0.02)
chk("Oracle/Raw VimQA",   1.22, v50["summary"]["oracle"]["f1_norm"]/v50["summary"]["raw"]["f1_norm"], 0.02)

# 81% vs 62% = IterCOMP/Oracle
chk("81% VimQA IterCOMP/Oracle", 81.0, 100*v50["summary"]["itercomp"]["f1_norm"]/v50["summary"]["oracle"]["f1_norm"], 0.6)
chk("62% MuSiQue IterCOMP/Oracle", 62.0, 100*m50["summary"]["itercomp"]["f1_norm"]/m50["summary"]["oracle"]["f1_norm"], 0.6)

# --- ablation n=50 ---
ab = load("ablation_vimqa_7b_hf")["table"]
for key, f1 in (("percentile=70.0",38.77),("percentile=80.0",36.60),
                ("percentile=85.0",36.35),("percentile=90.0",33.03),("percentile=95.0",34.56)):
    chk(f"quet k {key}", f1, ab[key]["f1"], 0.02)
chk("no-answerability F1", 36.64, ab["no-answerability"]["f1"], 0.02)
for s_, f1 in (("lexical",35.44),("dense",35.12),("dual",33.03)):
    chk(f"scorer {s_}", f1, ab[f"scorer={s_}"]["f1"], 0.02)

# --- EXP-1 (moi) ---
e1v, e1m = load("exp1_oracle_vimqa_200_7b"), load("exp1_oracle_musique_200_7b")
chk("EXP-1 vimqa paper",  50.08, e1v["results"]["paper"]["f1"], 0.02)
chk("EXP-1 vimqa oracle", 49.91, e1v["results"]["oracle"]["f1"], 0.02)
chk("EXP-1 musique paper",25.90, e1m["results"]["paper"]["f1"], 0.02)
chk("EXP-1 vimqa never ratio", 0.497, e1v["results"]["never"]["ratio"], 0.002)

# --- boolean 300/1003 ---
# Check duy nhat can data/ (116 MB, khong di kem goi nop). Bo qua co bao khi
# thieu, thay vi nem FileNotFoundError — nguoi cham bai se tuong repo hong.
sys.path.insert(0, "src")
if (Path("data") / "vimqa" / "validation.parquet").is_file():
    from itercomp import load_dataset
    rows = load_dataset("vimqa", 10000)
    nb = sum(1 for r in rows
             if str(r["answer"]).strip().lower() in {"đúng", "không", "sai", "có"})
    chk("so cau boolean VimQA", 300, nb, 5)
else:
    print("bỏ qua 1 check cần data/ (chạy ./run.sh để tải, 116 MB)\n")

# --- appendix moi: bang full n=1003 ---
vf = load("vimqa_full_7b")
for meth, em, f1n, ratio in (("raw",28.5,46.87,1.000),("oracle",36.4,61.03,0.057),
                             ("llmlingua2",17.2,35.31,0.316),("itercomp",26.9,51.29,0.206)):
    chk(f"app:full {meth} EM",    em,    vf["summary"][meth]["em"], 0.06)
    chk(f"app:full {meth} F1*",   f1n,   vf["summary"][meth]["f1_norm"], 0.02)
    chk(f"app:full {meth} ratio", ratio, vf["summary"][meth]["ratio"], 0.002)

import math, statistics as st, random
def paired(a,b,B=10000):
    d=[100*(x["methods"][a]["f1_norm"]-x["methods"][b]["f1_norm"]) for x in vf["per_row"]]
    rng=random.Random(0)
    bs=sorted(sum(d[rng.randrange(len(d))] for _ in d)/len(d) for _ in range(B))
    return st.mean(d), bs[int(.025*B)], bs[int(.975*B)]
for a,b,cl,lo_,hi_ in (("itercomp","llmlingua2",15.98,12.98,19.01),
                       ("itercomp","raw",4.42,1.73,7.17),
                       ("oracle","itercomp",9.74,7.08,12.42)):
    m,lo,hi = paired(a,b)
    chk(f"app:full {a}-{b} delta", cl, m, 0.02)
    chk(f"app:full {a}-{b} CI lo", lo_, lo, 0.05)
    chk(f"app:full {a}-{b} CI hi", hi_, hi, 0.05)
chk("app:full Oracle/Raw 1.30", 1.30,
    vf["summary"]["oracle"]["f1_norm"]/vf["summary"]["raw"]["f1_norm"], 0.006)
chk("app:full nguong 4.3", 4.3,
    2.8*st.stdev([100*(x["methods"]["itercomp"]["f1_norm"]-x["methods"]["llmlingua2"]["f1_norm"])
                  for x in vf["per_row"]])/math.sqrt(len(vf["per_row"])), 0.06)

# ════ PROPOSAL.md ════
vf = load("vimqa_full_7b")
e1v = load("exp1_oracle_vimqa_200_7b"); e1m = load("exp1_oracle_musique_200_7b")
x2 = load("exp2_undiacritised_200_7b")

# muc 4.4 EXP-1
for lab, d, keys in (("vimqa", e1v, ((50.08,"paper","f1"),(49.91,"oracle","f1"),
                                     (49.39,"never","f1"),(54.94,"gold","f1"),
                                     (0.181,"paper","ratio"),(0.497,"never","ratio"))),
                     ("musique", e1m, ((25.90,"paper","f1"),(25.94,"oracle","f1"),
                                       (28.06,"never","f1"),(44.63,"gold","f1")))):
    for cl, mode, k in keys:
        chk(f"PROPOSAL 4.4 {lab} {mode}.{k}", cl, d["results"][mode][k],
            0.02 if k == "f1" else 0.002)

# muc 4.5 EXP-2
for mode, ic, gold in (("keep",50.08,54.94),("strip",29.29,25.52),("both",30.25,24.43)):
    chk(f"PROPOSAL 4.5 {mode} itercomp", ic,   x2["results"][mode]["f1_itercomp"], 0.02)
    chk(f"PROPOSAL 4.5 {mode} gold",     gold, x2["results"][mode]["f1_gold"],     0.02)

# muc 6: tran Oracle — HAI dai luong khac nhau, de lan
v50, m50 = load("vimqa_50_7b"), load("musique_50_7b")
def ratio_to_ceiling(d): return 100*d["summary"]["itercomp"]["f1_norm"]/d["summary"]["oracle"]["f1_norm"]
chk("PROPOSAL 6 tran VimQA n=50",  81.0, ratio_to_ceiling(v50), 0.6)
chk("PROPOSAL 6 tran VimQA full",  84.0, ratio_to_ceiling(vf),  0.6)
chk("PROPOSAL 6 tran MuSiQue",     62.0, ratio_to_ceiling(m50), 0.6)

def zero_rate(d, m="oracle"):
    pr = d["per_row"]
    return 100*sum(1 for r in pr if r["methods"][m]["f1_norm"] == 0)/len(pr)
chk("PROPOSAL 6 Oracle F1=0 VimQA n=50", 28.0, zero_rate(v50), 0.6)
chk("PROPOSAL 6 Oracle F1=0 MuSiQue",    30.0, zero_rate(m50), 0.6)
chk("PROPOSAL 6 Oracle F1=0 VimQA full", 29.0, zero_rate(vf),  0.6)

# --- appendix EXP-2 (undiacritised) ---
for mode, ic, gold in (("keep", 50.08, 54.94), ("strip", 29.29, 25.52),
                       ("both", 30.25, 24.43)):
    chk(f"app:undia {mode} IterCOMP", ic,
        x2["results"][mode]["f1_itercomp"], 0.02)
    chk(f"app:undia {mode} gold", gold, x2["results"][mode]["f1_gold"], 0.02)

# --- appendix hu hai ngu phap: chenh lech IterCOMP vs LLMLingua-2 ---
# Bao cao trich "11.8--16.0 F1 gap"; hai dau mut phai khop hai co mau.
for f, claimed in (("vimqa_200_7b", 11.8), ("vimqa_full_7b", 16.0)):
    d = load(f)
    if d:
        gap = (d["summary"]["itercomp"]["f1_norm"]
               - d["summary"]["llmlingua2"]["f1_norm"])
        chk(f"app:funcwords khoang F1 gap ({f})", claimed, gap, 0.06)

# --- CAU HOI CO CAU: bao cao dung file nao, reader nao? ---
# Toi tung phat bieu sai hai lan ve dieu nay vi suy tu TEN FILE thay vi mo
# config. Ghi lai thanh check de khong lap.
_EXPECTED_READERS = {
    "vimqa_50_7b": "Qwen/Qwen2.5-7B-Instruct",
    "musique_50_7b": "Qwen/Qwen2.5-7B-Instruct",
    "vimqa_full_7b": "Qwen/Qwen2.5-7B-Instruct",
}
for name, want in _EXPECTED_READERS.items():
    d = load(name)
    got = (d or {}).get("config", {}).get("reader_model")
    checks.append((got == want,
                   f"reader cua {name} (bang chinh phai dung MOT reader)",
                   0.0, None if got == want else 1.0))

# --- so DAN XUAT: hieu giua hai phuong phap ---
# Bao cao trich chenh lech chu khong trich F1 tho, nen mot lan sua F1 mot ben
# se lam chenh lech sai ma khong check nao bat.
for f, pairs in (
    ("musique_50_7b", (("itercomp", "llmlingua2", 21.6),
                       ("itercomp", "oracle", -21.9))),
    ("vimqa_50_7b", (("itercomp", "llmlingua2", 12.4),
                     ("itercomp", "oracle", -10.3))),
):
    d = load(f)
    if not d:
        continue
    for a, b, claimed in pairs:
        got = d["summary"][a]["f1_norm"] - d["summary"][b]["f1_norm"]
        chk(f"so dan xuat {a}-{b} ({f})", claimed, got, 0.06)

# --- fertility tieng Viet: HAI lan do khac nhau, CI khong chong ---
# Bao cao dung 2.39 (930 cap); bang 4 ngon ngu dung 2.49 (400 cap). Ca hai
# dung, nhung tron chung ma khong noi la loai drift 3.
fw = load("fertility_wmt24pp")
if fw:
    chk("fertility vi 930 cap (report dung 2.39)", 2.39,
        fw["tokenizers"]["cl100k_base"]["fertility"], 0.006)
fm = load("fertility_multilingual")
if fm:
    chk("fertility vi 400 cap (bang 4 ngon ngu dung 2.49)", 2.49,
        fm["languages"]["vi_VN"]["tokenizers"]["cl100k_base"]["fertility"], 0.006)

# --- ban hoi nghi: so trong abstract phai co nguon ---
if paper_tex:
    fm = load("fertility_multilingual")
    if fm:
        L = fm["languages"]
        for lang, claimed in (("hi_IN", 4.97), ("th_TH", 3.89),
                              ("vi_VN", 2.49), ("zh_CN", 1.87)):
            got = L[lang]["tokenizers"]["cl100k_base"]["fertility"]
            chk(f"paper abstract fertility {lang}", claimed, got, 0.006)
            # va con so do phai THAT SU xuat hien trong abstract
            checks.append((f"{claimed}" in paper_tex,
                           f"paper abstract co trich {claimed} ({lang})",
                           claimed, None if f"{claimed}" in paper_tex else -1.0))
    er = load("effective_ratio_vimqa")
    if er:
        # HAI spread khac nhau, de lan:
        #   spread_across_tokenizers = ratio DANH NGHIA  -> 1.02x
        #   max/min cua effective_ratio = ratio HIEU DUNG -> 1.77x
        # Bai bao trich 1.77x, tuc cai thu hai. Khoa ca hai de khong lan nua.
        chk("spread ratio DANH NGHIA (1.02x)", 1.02,
            er["spread_across_tokenizers"], 0.006)
        eff = er["effective_ratio"]
        chk("spread ratio HIEU DUNG (paper trich 1.77x)", 1.77,
            max(eff.values()) / min(eff.values()), 0.006)
        # Bang moi trong app:fert liet ke tung o — neo het, va bang phai
        # co that trong CA HAI file (truoc day chi abstract dan 1.02/1.77 ma
        # khong file nao co bang nao chung minh).
        for _tok, _nomv, _effv in (("cl100k_base", 0.236, 0.099),
                                   ("o200k_base", 0.240, 0.161),
                                   ("regex", 0.234, 0.175)):
            chk(f"app:fert nominal {_tok}", _nomv,
                er["nominal_ratio"]["itercomp"][_tok], 0.0006)
            chk(f"app:fert effective {_tok}", _effv, eff[_tok], 0.0006)
        for _name, _doc in (("report_en", tex), ("paper", paper_tex)):
            checks.append(("Does the effective ratio discriminate" in _doc,
                           f"{_name}: co bang chung minh effective ratio",
                           "co", "co" if "Does the effective ratio discriminate"
                           in _doc else "THIEU"))

# --- phan tich dong dang (bang chung cho gia thuyet EXP-2) ---
hg = load("homograph_vimqa")
if hg:
    chk("app:undia so am tiet VimQA", 13165, hg["n_syllables"], 1)
    chk("app:undia % am tiet mo ho", 20.8, hg["pct_ambiguous"], 0.06)
    chk("app:undia so phan biet mat", 1983, hg["n_distinctions_lost"], 1)

av = load("ambiguity_vs_f1_vimqa")
if av:
    chk("app:undia mat do tb 75.7%", 75.7,
        100 * av["summary"]["raw"]["mean_density"], 0.06)

bo = load("beats_oracle_vimqa")
if bo:
    chk("app:undia % IterCOMP>Oracle", 9.8, bo["pct_win"], 0.06)
    chk("app:undia gold count win", 1.47, bo["gold_count_win"], 0.006)
    chk("app:undia gold count lose", 1.72, bo["gold_count_lose"], 0.006)
    chk("app:undia % win F1=100", 56.0,
        100 * bo["win_f1_perfect"] / bo["n_win"], 0.6)
    chk("app:undia % win Oracle=0", 70.0,
        100 * bo["win_oracle_zero"] / bo["n_win"], 0.6)

# --- max_iter bat buoc y HET nhau, khong chi gan ---
# Bao cao noi "agree to six decimal places"; day la khang dinh manh hon moi
# ket qua thong ke khac trong bai, nen phai khoa lai chinh xac.
_ab = load("ablation_vimqa_7b_hf")
if _ab:
    _t = _ab["table"]
    _f1 = [_t[f"max_iter={k}"]["f1"] for k in (2, 3, 4, 5)
           if f"max_iter={k}" in _t]
    if len(_f1) == 4:
        checks.append((max(_f1) - min(_f1) < 1e-6,
                       "max_iter 2..5 F1 y het (6 chu so)", 0.0,
                       max(_f1) - min(_f1)))
        chk("max_iter=2 F1", 33.033695, _f1[0], 0.000006)

# --- 1c'': tuong quan o n=1003 ---
_df = load("damage_vs_f1_vimqa_full")
if _df:
    _s = _df["summary"]
    chk("app:funcwords rho llmlingua2 n=1003", -0.067, _s["llmlingua2"]["rho"], 0.0006)
    chk("app:funcwords CI lo llmlingua2", -0.124, _s["llmlingua2"]["ci_lo"], 0.0006)
    chk("app:funcwords CI hi llmlingua2", -0.004, _s["llmlingua2"]["ci_hi"], 0.0006)
    chk("app:funcwords rho itercomp n=1003", -0.010, _s["itercomp"]["rho"], 0.0006)

# --- doi chung 4 ngon ngu (nay CA HAI ban deu dung) ---
_fm = load("fertility_multilingual")
if _fm:
    _L = _fm["languages"]
    for _lang, _want in (("hi_IN", 4.97), ("th_TH", 3.89), ("vi_VN", 2.49), ("zh_CN", 1.87)):
        chk(f"app:multiling {_lang} cl100k", _want,
            _L[_lang]["tokenizers"]["cl100k_base"]["fertility"], 0.006)

# --- 5 baseline them, n=200, cung mot phien Kaggle ---
_bp = load("baselines_paired_vimqa200")
if _bp:
    chk("baseline: IterCOMP F1 n=200", 50.17, _bp["itercomp_f1"], 0.006)
    _want = {"llmlingua2": (38.80, 11.36, 4.93, 17.84),
             "recomp-extractive": (31.22, 18.95, None, None),
             "selective-context": (18.91, 31.25, None, None),
             "longllmlingua": (18.71, 31.46, None, None),
             "recomp-abstractive": (8.58, 41.58, None, None),
             "llmlingua": (7.06, 43.11, None, None)}
    for _m, (_f1, _d, _lo, _hi) in _want.items():
        _v = _bp["methods"][_m]
        chk(f"baseline {_m} F1", _f1, _v["f1"], 0.006)
        chk(f"baseline {_m} chenh", _d, _v["diff"], 0.006)
        if _lo is not None:
            chk(f"baseline {_m} CI lo", _lo, _v["lo"], 0.006)
            chk(f"baseline {_m} CI hi", _hi, _v["hi"], 0.006)

# --- ablation n=500: 10 cau hinh ---
_ab = load("ablation_vimqa500_7b")
if _ab:
    _t = _ab["table"]
    for _k, _f1 in (("max_iter=1",31.7),("max_iter=2",34.9),("max_iter=3",35.6),
                    ("max_iter=4",36.2),("max_iter=5",36.1),
                    ("percentile=70.0",35.9),("percentile=80.0",38.3),
                    ("percentile=85.0",38.2),("percentile=90.0",36.1),
                    ("percentile=95.0",38.2)):
        chk(f"abl500 {_k} F1", _f1, _t[_k]["f1"], 0.06)
    chk("abl500 max_iter 2->5", 1.2, _t["max_iter=5"]["f1"]-_t["max_iter=2"]["f1"], 0.06)
    _pf = [_t[f"percentile={p}"]["f1"] for p in (70.0,80.0,85.0,90.0,95.0)]
    chk("abl500 khoang cach k", 2.3, max(_pf)-min(_pf), 0.06)

# --- theo so cau vang + hieu qua (2 bang phu luc paper goc) ---
_he = load("hop_efficiency_vimqa")
if _he:
    _h = _he["per_hop"]
    for _g, _w in (("1", {"raw":60.20,"oracle":64.06,"llmlingua2":41.00,"itercomp":60.80}),
                   ("2", {"raw":39.05,"oracle":58.79,"llmlingua2":31.96,"itercomp":46.14}),
                   ("3", {"raw":39.39,"oracle":68.65,"llmlingua2":33.84,"itercomp":32.26})):
        for _m, _v in _w.items():
            chk(f"hop{_g} {_m} F1", _v, _h[_g][_m]["f1"], 0.006)
    chk("hop: n gold=1", 374, _h["1"]["n"], 0.5)
    chk("hop: n gold=2", 605, _h["2"]["n"], 0.5)
    chk("hop: n gold=3", 22, _h["3"]["n"], 0.5)
    chk("hop2: IterCOMP hon Raw", 7.09,
        _h["2"]["itercomp"]["f1"] - _h["2"]["raw"]["f1"], 0.006)
    _e = _he["efficiency"]
    chk("eff: itercomp giam token", 79.4, _e["itercomp"]["token_reduction"], 0.06)
    chk("eff: itercomp speedup e2e", 0.64, _e["itercomp"]["speedup_e2e"], 0.006)
    chk("eff: llmlingua2 speedup", 2.01, _e["llmlingua2"]["speedup_e2e"], 0.006)
    chk("eff: itercomp cham hon", 1.57, 1/_e["itercomp"]["speedup_e2e"], 0.006)

# --- doi chung: khoang cach tren tieng Anh LON HON tieng Viet ---
_mq = load("musique_50_7b")
if _mq:
    _s = _mq["summary"]
    chk("MuSiQue IterCOMP-LLMLingua2 gap", 21.60,
        _s["itercomp"]["f1_norm"] - _s["llmlingua2"]["f1_norm"], 0.006)

# --- nhom dong dang lon nhat ---
_hg = load("homograph_vimqa")
if _hg:
    _tg = _hg["top_groups"]
    _sz = sorted((len(v) for v in _tg.values()), reverse=True)
    chk("homograph: nhom lon nhat", 17, _sz[0], 0.5)
    chk("homograph: nhom nhi", 16, _sz[1], 0.5)

# --- template so tuoi (truoc day 5 con so KHONG truy nguon duoc) ---
_ac = load("age_comparison_vimqa")
if _ac:
    _g, _c = _ac["groups"], _ac.get("compare", {})
    chk("agecmp: n", 98, _g["age"]["n"], 0.5)
    chk("agecmp: % cua tap", 9.8, _g["age"]["pct_of_set"], 0.06)
    chk("agecmp: n con lai", 905, _g["rest"]["n"], 0.5)
    chk("agecmp: F1=0 age", 44.9, _g["age"]["pct_f1_zero"], 0.06)
    chk("agecmp: F1=0 rest", 37.1, _g["rest"]["pct_f1_zero"], 0.06)
    chk("agecmp: TB age", 55.1, _g["age"]["mean_f1"], 0.06)
    chk("agecmp: TB rest", 50.9, _g["rest"]["mean_f1"], 0.06)
    if _c:
        chk("agecmp 0.5B: F1=0 age", 92.1, _c["age"]["pct_f1_zero"], 0.06)
        chk("agecmp 0.5B: TB age", 5.5, _c["age"]["mean_f1"], 0.06)
        chk("agecmp 0.5B: F1=0 rest", 51.2, _c["rest"]["pct_f1_zero"], 0.06)
        chk("agecmp 0.5B: TB rest", 29.4, _c["rest"]["mean_f1"], 0.06)

# --- ti le loi song sot qua truy hoi hoan hao ---
_vf = load("vimqa_full_7b")
if _vf:
    _s = _vf["summary"]
    _sur = 100 * (100 - _s["oracle"]["f1_norm"]) / (100 - _s["itercomp"]["f1_norm"])
    chk("loi song sot qua Oracle (n=1003)", 80, _sur, 0.6)
    chk("Oracle de lai loi F1", 39.0, 100 - _s["oracle"]["f1_norm"], 0.06)
    chk("IterCOMP loi F1", 48.7, 100 - _s["itercomp"]["f1_norm"], 0.06)

# --- MuSiQue n=500: doi chung tieng Anh dung co mau ---
_m5 = load("musique_500_7b")
if _m5:
    _s = _m5["summary"]
    for _m, _w in (("oracle",49.84),("raw",23.43),("itercomp",26.74),("llmlingua2",15.14)):
        chk(f"mq500 {_m} F1", _w, _s[_m]["f1_norm"], 0.006)
    chk("mq500 IterCOMP/Oracle %", 54,
        100*_s["itercomp"]["f1_norm"]/_s["oracle"]["f1_norm"], 0.6)
    chk("mq500 Oracle/Raw", 2.13,
        _s["oracle"]["f1_norm"]/_s["raw"]["f1_norm"], 0.006)
    import statistics as _st, random as _rd
    def _pb(a,b,B=10000):
        d=[100*(x["methods"][a]["f1_norm"]-x["methods"][b]["f1_norm"]) for x in _m5["per_row"]]
        g=_rd.Random(0)
        bs=sorted(sum(d[g.randrange(len(d))] for _ in d)/len(d) for _ in range(B))
        return _st.mean(d), bs[int(.025*B)], bs[int(.975*B)]
    for _a,_b,_d,_lo,_hi in (("itercomp","llmlingua2",11.60,7.71,15.65),
                             ("itercomp","raw",3.31,-0.52,7.22),
                             ("oracle","itercomp",23.09,19.22,27.02)):
        _m2,_l,_h = _pb(_a,_b)
        chk(f"mq500 {_a}-{_b} delta", _d, _m2, 0.02)
        chk(f"mq500 {_a}-{_b} CI lo", _lo, _l, 0.06)
        chk(f"mq500 {_a}-{_b} CI hi", _hi, _h, 0.06)
    chk("vimqa IterCOMP/Oracle %", 84,
        100*load("vimqa_full_7b")["summary"]["itercomp"]["f1_norm"]
           /load("vimqa_full_7b")["summary"]["oracle"]["f1_norm"], 0.6)

# --- thu tu phuong phap con khop paper o n=500? ---
if _m5:
    _o = sorted(_m5["summary"], key=lambda k: -_m5["summary"][k]["f1_norm"])
    checks.append((_o == ["oracle","itercomp","raw","llmlingua2"],
                   "mq500: thu tu khop paper", "oracle>itercomp>raw>llmlingua2",
                   " > ".join(_o)))

# --- CompAct: baseline thu 6/7 ---
_ca = load("compact_vimqa_200_7b")
if _ca:
    _s = _ca["summary"]
    chk("compact F1", 23.43, _s["compact"]["f1_norm"], 0.006)
    chk("compact giu %", 5.6, 100*_s["compact"]["ratio"], 0.06)
    chk("compact s/cau", 20.9, _s["compact"]["sec_per_q"], 0.06)
    chk("itercomp s/cau (cung phien)", 2.6, _s["itercomp"]["sec_per_q"], 0.06)
    _f = lambda m: [100*r["methods"][m]["f1_norm"] for r in _ca["per_row"]]
    _u = lambda m: 100*sum(1 for r in _ca["per_row"]
                           if str(r["methods"][m]["pred"]).strip().lower().startswith("unknown"))/len(_ca["per_row"])
    chk("compact % unknown", 45, _u("compact"), 0.6)
    chk("itercomp % unknown", 20, _u("itercomp"), 0.6)
    import statistics as _st2, random as _rd2
    _d = [a-b for a, b in zip(_f("itercomp"), _f("compact"))]
    _g = _rd2.Random(0)
    _bs = sorted(sum(_d[_g.randrange(len(_d))] for _ in _d)/len(_d) for _ in range(10000))
    chk("IterCOMP-CompAct delta", 26.74, _st2.mean(_d), 0.02)

# --- doi chieu truc tiep voi paper goc tren MuSiQue ---
if _m5:
    _s5 = _m5["summary"]
    for _m, _p in (("oracle",37.51),("raw",19.92),("itercomp",27.36),("llmlingua2",14.72)):
        chk(f"vs-paper {_m} delta", round(_s5[_m]["f1_norm"]-_p, 2),
            _s5[_m]["f1_norm"]-_p, 0.006)
    chk("vs-paper: IterCOMP THAP hon", -0.62, _s5["itercomp"]["f1_norm"]-27.36, 0.006)
    chk("vs-paper: Oracle CAO hon", 12.33, _s5["oracle"]["f1_norm"]-37.51, 0.006)

# ── README va PROPOSAL cung phai khop ─────────────────────────────
# Verifier truoc day CHI doi chieu hai file .tex, nen README va PROPOSAL
# giu so cu ma khong ai bat: README con bang vs-paper chi co n=50, va
# PROPOSAL con ghi "62% tren tieng Anh" khi n=500 da cho 54%. Cac con so
# HEADLINE phai xuat hien dung trong ca hai tai lieu.
_docs = {}
for _n in ("README.md", "PROPOSAL.md"):
    _p = Path(_n)
    if _p.exists():
        _docs[_n] = _p.read_text()

def _need(doc, token, what):
    """Tai lieu PHAI chua chuoi nay (con so hien hanh)."""
    if doc not in _docs:
        return
    checks.append((token in _docs[doc], f"{doc}: {what}", token,
                   "co" if token in _docs[doc] else "THIEU"))

if _m5 and _vf:
    _a, _b = _m5["summary"], _vf["summary"]
    for _doc in ("README.md",):
        _need(_doc, f"{_a['oracle']['f1_norm']:.2f}", "Oracle MuSiQue n=500")
        _need(_doc, f"{_a['itercomp']['f1_norm']:.2f}", "IterCOMP MuSiQue n=500")
        _need(_doc, f"{_b['itercomp']['f1_norm']:.2f}", "IterCOMP VimQA n=1003")
    _ic_or_vi = 100 * _b["itercomp"]["f1_norm"] / _b["oracle"]["f1_norm"]
    _ic_or_en = 100 * _a["itercomp"]["f1_norm"] / _a["oracle"]["f1_norm"]
    for _doc in ("PROPOSAL.md",):
        _need(_doc, f"{_ic_or_vi:.0f}%", "IterCOMP/Oracle tieng Viet")
        _need(_doc, f"{_ic_or_en:.0f}%", "IterCOMP/Oracle tieng Anh")

# ── so tham chieu HARDCODE trong notebook Kaggle ──────────────────
# Notebook in bang so sanh voi cac baseline da co. results/ doi ma dong
# print khong doi thi nguoi cham chay notebook se thay so sai — va truoc
# day khong phep kiem nao nhin vao .ipynb.
import json as _js
_nb = Path("notebooks/kaggle/itercomp-compact.ipynb")
_bpair = load("baselines_paired_vimqa200")
if _nb.exists() and _bpair:
    _src = "\n".join("".join(c["source"])
                      for c in _js.loads(_nb.read_text())["cells"])
    for _m in ("llmlingua2", "recomp-extractive", "selective-context",
               "longllmlingua", "recomp-abstractive", "llmlingua"):
        _v = _bpair["methods"].get(_m)
        if not _v:
            continue
        _tok = f"{_v['f1']:.2f}"
        checks.append((_tok in _src, f"notebook compact: {_m}", _tok,
                       "co" if _tok in _src else "THIEU"))

# --- MuSiQue o CUNG MUC NEN ---
_mk = load("ablation_musique200_k")
if _mk:
    _t = _mk["table"]
    for _k, _r, _f in (("90.0",0.330,26.65),("93.0",0.287,26.32),
                       ("96.0",0.160,29.04),("97.5",0.160,29.04)):
        chk(f"mqk k={_k} ratio", _r, _t[f"percentile={_k}"]["ratio"], 0.0006)
        chk(f"mqk k={_k} F1", _f, _t[f"percentile={_k}"]["f1"], 0.006)
    chk("mqk: hon paper o cung muc nen", 1.68,
        _t["percentile=96.0"]["f1"] - 27.36, 0.006)

# --- 4 dataset o n>=500 (doc TRUC TIEP tung file dataset — mot nguon su that) ---
# Truoc day khoi nay doc results/four_datasets_paired.json, mot ban tong hop tu
# n=500. Khi 2Wiki va HotpotQA len n=1500 file do thanh STALE, nen kiem van PASS
# trong khi BANG trong bai da khac (2Wiki -2.69 -> +1.00, O/R 1.53 -> 1.61). Gio
# kiem thang vao cac file ma bang thuc su dung, khong the lech am tham nua.
import statistics as _st
_4files = {"vimqa":"vimqa_full_7b","musique":"musique_500_7b",
           "2wiki":"2wiki_1500_7b","hotpotqa":"hotpotqa_1500_7b"}
_4d = {ds: load(fn) for ds, fn in _4files.items()}
def _mean(rows, m): return _st.mean(100*r["methods"][m]["f1_norm"] for r in rows)
if all(_4d.values()):
    _w = {"vimqa":   {"oracle":61.03,"raw":46.87,"itercomp":51.29,"llmlingua2":35.31},
          "musique": {"oracle":49.84,"raw":23.43,"itercomp":26.74,"llmlingua2":15.14},
          "2wiki":   {"oracle":57.65,"raw":35.72,"itercomp":36.72,"llmlingua2":19.24},
          "hotpotqa":{"oracle":71.40,"raw":59.24,"itercomp":54.58,"llmlingua2":36.00}}
    for _ds, _ms in _w.items():
        for _m, _v in _ms.items():
            chk(f"4ds {_ds} {_m}", _v, _4d[_ds]["summary"][_m]["f1_norm"], 0.02)
    _diffs = {_ds: {"ir": _mean(_4d[_ds]["per_row"],"itercomp")-_mean(_4d[_ds]["per_row"],"raw"),
                    "il": _mean(_4d[_ds]["per_row"],"itercomp")-_mean(_4d[_ds]["per_row"],"llmlingua2")}
              for _ds in _4files}
    for _ds, _d in (("vimqa",4.42),("musique",3.31),("2wiki",1.00),("hotpotqa",-4.66)):
        chk(f"4ds {_ds} IterC-Raw", _d, _diffs[_ds]["ir"], 0.02)
    for _ds, _d in (("vimqa",15.98),("musique",11.60),("2wiki",17.48),("hotpotqa",18.58)):
        chk(f"4ds {_ds} IterC-LLML2", _d, _diffs[_ds]["il"], 0.02)
    _po = ["oracle","itercomp","raw","llmlingua2"]
    for _ds, _same in (("vimqa",True),("musique",True),("2wiki",True),("hotpotqa",False)):
        _order = sorted(_po, key=lambda m: -_4d[_ds]["summary"][m]["f1_norm"])
        checks.append(((_order == _po) == _same,
                       f"4ds {_ds}: thu tu {'khop' if _same else 'KHAC'} paper",
                       "khop" if _same else "khac", " > ".join(_order)))
    for _ds, _r in (("2wiki",1.61),("hotpotqa",1.21)):
        _s=_4d[_ds]["summary"]
        chk(f"4ds {_ds} Oracle/Raw", _r, _s["oracle"]["f1_norm"]/_s["raw"]["f1_norm"], 0.02)
    _gaps = [_diffs[_ds]["il"] for _ds in _4files]
    chk("dai gap: min", 11.6, min(_gaps), 0.06)
    chk("dai gap: max", 18.58, max(_gaps), 0.06)

# --- doi chung hu-tu o n=500 (chot chuoi n=50 cuoi cung) ---
_dm = load("damage_matched_500")
if _dm:
    for _ds, _ic, _ll, _g in (("vimqa",1.01,1.17,0.16),("musique",0.99,1.17,0.17)):
        chk(f"dmg500 {_ds} IterCOMP", _ic, _dm[_ds]["itercomp_ratio"], 0.006)
        chk(f"dmg500 {_ds} LLML-2", _ll, _dm[_ds]["llml2_ratio"], 0.006)
        chk(f"dmg500 {_ds} gap", _g, _dm[_ds]["gap"], 0.006)
    # dau phai GIU NGUYEN: en >= vi
    checks.append((_dm["musique"]["gap"] >= _dm["vimqa"]["gap"],
                   "dmg500: en >= vi (dau giu nguyen)", "en>=vi",
                   f"en {_dm['musique']['gap']:.2f} vs vi {_dm['vimqa']['gap']:.2f}"))

# --- gio may chay: MOI so gio trong doan Compute cost phai khop file do duoc ---
# Doan nay tung sai HAI lan (uoc luong thap 1/3; roi tong do duoc bi hai kernel
# moi lam cu). Ca hai lan khong bi bat vi khong co pin nao o day. Gio co.
_rt = load("kernel_runtimes")
if _rt:
    _pk = _rt["per_kernel_hours"]
    for _k, _claimed in (
        ("itercomp-gpu-tasks-567", 6.58), ("itercomp-ablation-500", 3.91),
        ("itercomp-musique-500", 2.34), ("itercomp-3datasets", 2.05),
        ("itercomp-musique-k", 1.97), ("itercomp-compact", 1.40),
        ("itercomp-damage-vs-f1-7b", 1.00), ("itercomp-baselines", 0.81),
        ("itercomp-1c-double", 5.69), ("itercomp-damage-500-cpu", 5.66),
        ("itercomp-2wiki-1500", 2.66), ("itercomp-hotpotqa-1500", 2.81),
        ("itercomp-musique-1500", 5.65), ("itercomp-hotpotqa-readers", 1.67),
    ):
        chk(f"gio {_k}", _claimed, _pk.get(_k), 0.006)
    chk("tong gio GPU", 32.9, _rt["gpu_hours_total"], 0.06)
    chk("tong gio CPU", 11.4, _rt["cpu_hours_total"], 0.06)
    # Moi so gio duoc trich phai co that trong .tex, va nguoc lai: khong duoc
    # con so gio nao trong .tex ma khong pin o day.
    _in_tex = set(re.findall(r"(\d+\.\d{1,2}) h\b", tex))
    _pinned = {f"{v:.2f}" for v in _pk.values()}
    _orphan = sorted(_in_tex - _pinned)
    checks.append((not _orphan, "khong co so gio nao trong .tex ma chua pin",
                   "0 orphan", ", ".join(_orphan) or "0 orphan"))

# --- moi so trong abstract phai co cho chong lung trong THAN BAI cung file ---
# Loi that: abstract ban hoi nghi dan fertility bon ngon ngu (4.97 Hindi, 3.89
# Thai, 1.87 Chinese) lam dong gop so MOT, nhung bang tab:multiling da bi xoa
# khoi paper.tex luc to chuc lai phu luc. Moi so van khop results/ nen check cu
# pass sach — no doi chieu so voi du lieu, khong hoi so co duoc chung minh trong
# chinh bai hay khong. Day la cau hoi do.
def _abstract_orphans(doc: str, name: str):
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", doc, re.S)
    if not m:
        return None
    abstract, body = m.group(1), doc[:m.start()] + doc[m.end():]
    nums = set(re.findall(r"\d+\.\d+", abstract))
    # So nho (<=2 chu so thap phan kieu 0.5, 1.0) hay xuat hien ngau nhien; van
    # kiem, nhung mot chuoi con khop trong than bai la du.
    return sorted(n for n in nums if n not in body)

for _name, _doc in (("report_en", tex), ("paper", paper_tex)):
    if not _doc:
        continue
    _orph = _abstract_orphans(_doc, _name)
    if _orph is None:
        checks.append((False, f"{_name}: khong tim thay abstract", "co", "THIEU"))
    else:
        checks.append((not _orph,
                       f"{_name}: so trong abstract co trong than bai",
                       "0 orphan", ", ".join(_orph) or "0 orphan"))

# --- hai so do CUNG mot dai luong thi phai co cau hoa giai ---
# 2.39x (930 cap) va 2.49x (400 cap/ngon ngu) deu la fertility tieng Viet duoi
# cl100k_base, do tren hai mau khac nhau cua cung mot corpus. Ca hai deu dung va
# deu can: 930 cap la con so vi/en tot nhat, 400 cap/ngon ngu la de SO SANH duoc
# giua bon ngon ngu. Mot phan bien da doc thanh mau thuan va de nghi bo 2.39 —
# lam vay la trich lan chay NHO hon, dung thu check_superseded ton tai de chan.
# Nen giu ca hai, va bat buoc tai lieu phai co cau noi ro chung khac nhau o dau.
for _name, _doc in (("report_en", tex), ("paper", paper_tex)):
    if not _doc:
        continue
    _has_both = "2.49" in _doc and "2.39" in _doc
    # cau hoa giai = mot doan chua CA HAI so trong pham vi 400 ky tu
    _near = False
    for _m in re.finditer(r"2\.49", _doc):
        if "2.39" in _doc[max(0, _m.start() - 400):_m.start() + 400]:
            _near = True
            break
    checks.append((not _has_both or _near,
                   f"{_name}: co cau hoa giai 2.49 vs 2.39",
                   "co", "co" if _near else "THIEU"))

# --- bang boolnorm 7B (tab:boolnorm7b) — TRUOC DAY KHONG PIN ---
# Bang nay la "cleanest result in this work" nhung khong file nao luu no: gia tri
# la f1_norm - f1 tren vimqa_50_7b / musique_50_7b, tinh khi bien dich chu khong
# ghi ra summary. Nen mot review tuong no khong co nguon, va neu vimqa_50_7b doi
# thi bang drift am tham. Gio pin thang: delta = f1_norm - f1.
_v50b = load("vimqa_50_7b"); _m50b = load("musique_50_7b")
if _v50b:
    for _m, _want in (("raw", 6.0), ("oracle", 4.0), ("llmlingua2", 8.0), ("itercomp", 10.0)):
        _s = _v50b["summary"][_m]
        chk(f"boolnorm vi {_m}", _want, _s["f1_norm"] - _s["f1"], 0.06)
if _m50b:
    for _m in ("raw", "oracle", "llmlingua2", "itercomp"):
        _s = _m50b["summary"][_m]
        chk(f"boolnorm en {_m} = 0", 0.0, _s["f1_norm"] - _s["f1"], 0.06)

# --- reader-swap: HotpotQA reader yeu (0.5B) khoi phuc thu tu paper ---
# Bang chung truc tiep cho reader-dependence: cung pipeline nen, chi doi reader.
import statistics as _rst
_h05=load("hotpotqa_500_05b"); _h7=load("hotpotqa_500_7b")
if _h05 and _h7:
    _m=lambda d,x:_rst.mean(100*r["methods"][x]["f1_norm"] for r in d["per_row"])
    chk("readerswap 0.5B IterC-Raw", 7.66, _m(_h05,"itercomp")-_m(_h05,"raw"), 0.1)
    chk("readerswap 7B IterC-Raw", -7.99, _m(_h7,"itercomp")-_m(_h7,"raw"), 0.1)
    # 0.5B phai KHOP paper (IC>Raw), 7B phai lat
    checks.append((_m(_h05,"itercomp")>_m(_h05,"raw"), "readerswap 0.5B: IC>Raw (khop paper)","IC>Raw",
                   f"{_m(_h05,'itercomp')-_m(_h05,'raw'):+.2f}"))

# --- retention IterCOMP (% token) — chan drift n=200 vs n=1003, va range 19-35 ---
_ret={"vimqa_full_7b":20.6,"hotpotqa_1500_7b":19.4,"musique_500_7b":34.9,"2wiki_1500_7b":35.4}
_rv=[]
for _f,_w in _ret.items():
    _d=load(_f)
    if _d:
        _r=_d["summary"]["itercomp"]["ratio"]*100; _rv.append(_r)
        chk(f"retention {_f}", _w, _r, 0.1)
if _rv:
    chk("retention range min (19)", 19, min(_rv), 0.5)
    chk("retention range max (35)", 35, max(_rv), 0.5)

# --- HẰNG SỐ TRÍCH DẪN (chép tay từ paper gốc / MTEB, KHÔNG có file nguồn) ---
# Khoá literal để lần sửa sau không đổi nhầm mà không ai biết.
# 43.63 & 51.78: ĐÃ đối chiếu paper/itercomp_acl2026.pdf Table 3 (HotpotQA Raw
#   & IterCOMP F1) — khớp; cùng bảng đó cũng xác minh MuSiQue 19.92/14.72/27.36.
# 39.84 & 34.18: ĐÃ đối chiếu VN-MTEB (aclanthology 2026.findings-eacl.86) cột
#   Retr. — bge-m3 39.84, Vietnamese_Embedding 34.18; "18 models / 41 datasets".
_cited = {
    "43.63": "Raw HotpotQA — paper Table 3 (đã xác minh)",
    "51.78": "IterCOMP HotpotQA — paper Table 3 (đã xác minh)",
    "39.84": "bge-m3 Retr. — VN-MTEB (đã xác minh)",
    "34.18": "Vietnamese_Embedding Retr. — VN-MTEB (đã xác minh)",
}
for _lit, _why in _cited.items():
    checks.append((_lit in tex, f"cited literal {_lit} ({_why})", "có trong tex",
                   "có" if _lit in tex else "MẤT"))
# quan hệ suy ra: HotpotQA Raw của TA (n=1500) − 43.63 của paper ≈ +15.6
_hraw = S("hotpotqa_1500_7b", "raw")
if _hraw is not None:
    chk("cited: 59.24−43.63 ≈ +15.6", 15.6, _hraw - 43.63, 0.1)

bad = [c for c in checks if not c[0]]
print(f"{len(checks)-len(bad)}/{len(checks)} con so KHỚP\n")
if bad:
    print("LỆCH:")
    for _, lab, cl, ac in bad:
        # ac co the la so HOAC chuoi (check tren README/PROPOSAL bao "co"/"THIEU")
        shown = ac if not isinstance(ac, (int, float)) else round(ac, 3)
        print(f"  {lab:38s} mong={cl}  thực tế={shown}")
else:
    print("Không lệch chỗ nào.")
