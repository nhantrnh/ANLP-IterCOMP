"""Do tac dong cua loi normalize_boolean len ket qua da luu.

Chay lai phep chuan hoa tren per_row cua vimqa_50_7b.json, mot lan bang logic
CU (chuoi cung "dung"/"khong") va mot lan bang logic MOI, roi so F1 trung binh.
"""
import json
import re
import sys
from pathlib import Path

# Goc repo suy ra tu vi tri file nay, khong cung hoa duong dan may ca nhan.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from itercomp.metrics import _YES, _NO, normalize, tokenize, f1_score  # noqa: E402


def nb_old(pred, gold):
    g = normalize(gold)
    if g not in _YES | _NO:
        return pred
    toks = set(tokenize(pred))
    hy, hn = toks & _YES, toks & _NO
    if hy and not hn:
        return "đúng" if g in _YES else "không"
    if hn and not hy:
        return "không" if g in _YES else "đúng"
    return pred


def nb_new(pred, gold):
    g = normalize(gold)
    if g not in _YES | _NO:
        return pred
    toks = set(tokenize(pred))
    hy, hn = toks & _YES, toks & _NO
    opp = "không" if g in _YES else "đúng"
    if hy and not hn:
        return gold if g in _YES else opp
    if hn and not hy:
        return opp if g in _YES else gold
    return pred


d = json.load(open(ROOT / "results" / "vimqa_50_7b.json", encoding="utf-8"))
rows = d["per_row"]

methods = sorted({m for r in rows for m in r.get("methods", {})})
print(f"{'method':14s} {'F1 cu':>8s} {'F1 moi':>8s} {'chenh':>7s}   (n={len(rows)})")
for m in methods:
    old = new = 0.0
    n = 0
    neg = 0
    for r in rows:
        got = r.get("methods", {}).get(m)
        if not got:
            continue
        pred, gold = got.get("pred", ""), r.get("gold", "")
        n += 1
        if normalize(gold) in _NO:
            neg += 1
        old += f1_score(nb_old(pred, gold), gold)
        new += f1_score(nb_new(pred, gold), gold)
    if n:
        print(f"{m:14s} {100*old/n:8.2f} {100*new/n:8.2f} {100*(new-old)/n:+7.2f}")
print(f"\nso cau gold thuoc cuc phu dinh: {neg}/{n}")
