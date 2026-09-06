"""Vì sao nén VƯỢT gold trên văn bản không dấu — tìm bằng chứng, không chỉ giả thuyết.

VẤN ĐỀ. EXP-2 đo được: bỏ dấu thì IterCOMP vượt gold-context 3.78–5.82 F1. Ta
giải thích bằng "ngữ cảnh rộng giúp khử mơ hồ từ vựng". Đó là **giả thuyết**, và
reviewer sẽ đòi bằng chứng.

CÁCH KIỂM. Bỏ dấu biến nhiều từ khác nghĩa thành một chuỗi:

    may  <-  máy, mày, mãy, mảy, mạy
    ma   <-  má, mà, mã, mả, mạ

Nếu giả thuyết đúng thì ở những câu IterCOMP thắng gold, câu hỏi hoặc đáp án
phải chứa **nhiều từ đồng dạng-không-dấu hơn** so với các câu khác. Đó là dự
đoán kiểm được, và nếu sai thì giả thuyết phải rút lại.

Script đo ba thứ:

  1. Mật độ từ mơ hồ — bao nhiêu âm tiết trong câu hỏi có ≥2 nghĩa khi bỏ dấu
  2. So nhóm: câu IterCOMP thắng gold vs câu thua
  3. Trích ví dụ cụ thể để đưa vào bài dưới dạng case study

Chạy CPU, chỉ cần `results/exp2_undiacritised_*.json` có khoá `detail`.

    python scripts/homograph_analysis.py
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from itercomp.fertility import syllable_spans  # noqa: E402


def strip_tones(text: str) -> str:
    """Bỏ dấu thanh và dấu phụ, giữ nguyên chữ cái cơ sở."""
    import unicodedata
    nfd = unicodedata.normalize("NFD", text)
    base = "".join(c for c in nfd if not unicodedata.combining(c))
    # đ -> d: gỡ dấu kiểu Việt cũng bỏ gạch ngang
    return unicodedata.normalize("NFC", base).replace("đ", "d").replace("Đ", "D")


def build_homograph_map(corpus_words: set[str]) -> dict[str, set[str]]:
    """Chuỗi-không-dấu -> tập âm tiết có dấu cùng dạng.

    Dựng từ chính corpus thay vì từ điển ngoài: ta cần biết mơ hồ nào **thật sự
    xuất hiện trong dữ liệu này**, không phải mọi mơ hồ lý thuyết của tiếng Việt.

    ĐẦU VÀO PHẢI ĐÃ HẠ THẤP. Nếu không, `Hà` và `hà` tính là hai âm tiết khác
    nhau rồi gộp lại, và phần gộp đó bị quy cho việc bỏ dấu. Đo thử trên MuSiQue
    cho thấy sai lệch này lớn: 12% "mất phân biệt" ở tiếng Anh hoá ra gần như
    toàn bộ là chữ hoa/thường, vì tiếng Anh chỉ có 2% từ mang dấu.
    """
    groups: dict[str, set[str]] = defaultdict(set)
    for w in corpus_words:
        groups[strip_tones(w)].add(w)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def ambiguity_density(text: str, homographs: dict[str, set[str]]) -> float:
    """Tỉ lệ âm tiết trong `text` mà bỏ dấu sẽ trùng với âm tiết khác."""
    syls = [text[a:b].lower() for a, b in syllable_spans(text)]
    if not syls:
        return 0.0
    n = sum(1 for s in syls if len(homographs.get(strip_tones(s), ())) >= 2)
    return n / len(syls)


def boot_ci(a: list[float], b: list[float], n: int = 5000, seed: int = 0):
    """CI 95% cho hiệu hai trung bình."""
    if not a or not b:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    d = []
    for _ in range(n):
        ra = sum(a[rng.randrange(len(a))] for _ in a) / len(a)
        rb = sum(b[rng.randrange(len(b))] for _ in b) / len(b)
        d.append(ra - rb)
    d.sort()
    return d[int(0.025 * n)], d[int(0.975 * n)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp2", type=Path,
                    default=ROOT / "results" / "exp2_undiacritised_200_7b.json")
    ap.add_argument("--mode", default="strip", choices=["strip", "both"],
                    help="chế độ gỡ dấu cần phân tích")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = (json.loads(args.exp2.read_text()) if args.exp2.is_file() else {})
    detail = (data.get("detail") or {}).get(args.mode)

    # ── dung ban do dong dang tu chinh corpus ───────────────────────
    from itercomp import load_dataset
    words: set[str] = set()
    for r in load_dataset("vimqa", 1000):
        for t, sents in zip(r["context"]["title"], r["context"]["sentences"]):
            for s in sents:
                words |= {s[a:b].lower() for a, b in syllable_spans(s)}
        words |= {r["question"][a:b].lower()
                  for a, b in syllable_spans(r["question"])}
    homographs = build_homograph_map(words)
    print(f"Bản đồ đồng dạng-không-dấu: {len(homographs)} nhóm từ {len(words)} "
          f"âm tiết trong VimQA\n")

    top = sorted(homographs.items(), key=lambda kv: -len(kv[1]))[:6]
    print("Nhóm mơ hồ nhất:")
    for k, v in top:
        print(f"  {k:10s} <- {', '.join(sorted(v)[:7])}")

    # ── 0. do TRUC TIEP mo ho do bo dau, khong can EXP-2 ───────────
    # Phan nay la bang chung DOC LAP: no do bo dau xoa bao nhieu phan biet
    # nghia trong chinh du lieu, khong phu thuoc ket qua reader nao.
    total_syl = len(words)
    ambiguous = sum(len(v) for v in homographs.values())
    print(f"\n{'='*70}")
    print("BỎ DẤU XOÁ BAO NHIÊU PHÂN BIỆT NGHĨA?\n")
    print(f"  âm tiết phân biệt (có dấu)   : {total_syl:,}")
    print(f"  chuỗi phân biệt (bỏ dấu)     : {len(set(strip_tones(w) for w in words)):,}")
    print(f"  âm tiết bị gộp với ≥1 từ khác: {ambiguous:,} "
          f"({100*ambiguous/total_syl:.1f}%)")
    collisions = sum(len(v) - 1 for v in homographs.values())
    print(f"  số phân biệt nghĩa MẤT       : {collisions:,}")
    print()
    print(f"  -> Bỏ dấu làm {100*ambiguous/total_syl:.1f}% âm tiết trở nên mơ hồ.")
    print("     Đó là lượng thông tin reader phải suy ra từ ngữ cảnh, và là")
    print("     cơ chế mà giả thuyết 'ngữ cảnh rộng giúp khử mơ hồ' dựa vào.")

    if not detail:
        print(f"\n{'='*70}")
        print("PHẦN CASE STUDY BỊ BỎ QUA")
        print(f"  {args.exp2.name} không có khoá 'detail' — file cũ chỉ lưu số")
        print("  tổng hợp. Chạy lại research/exp2_undiacritised.py (bản mới ghi")
        print("  detail, cần GPU) để có ví dụ cụ thể cho case study.")
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            json.dump({"config": vars(args) | {"out": str(args.out),
                                               "exp2": str(args.exp2)},
                       "n_homograph_groups": len(homographs),
                       "n_syllables": total_syl,
                       "n_ambiguous": ambiguous,
                       "pct_ambiguous": 100 * ambiguous / total_syl,
                       "n_distinctions_lost": collisions,
                       "top_groups": {k: sorted(v) for k, v in top},
                       "case_study": None},
                      open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print(f"\n→ ghi {args.out}")
        return 0

    # ── 1. mat do mo ho, chia theo IterCOMP thang/thua gold ─────────
    win, lose = [], []
    for d in detail:
        dens = ambiguity_density(d["question"], homographs)
        (win if d["f1_itercomp"] > d["f1_gold"] else lose).append((dens, d))

    print(f"\n{'nhóm':28s} {'n':>5s} {'mật độ mơ hồ':>14s} {'F1 IC':>7s} {'F1 gold':>8s}")
    for lab, grp in (("IterCOMP thắng gold", win), ("còn lại", lose)):
        if not grp:
            continue
        print(f"{lab:28s} {len(grp):5d} {100*st.mean(x for x,_ in grp):13.1f}% "
              f"{st.mean(d['f1_itercomp'] for _,d in grp):7.1f} "
              f"{st.mean(d['f1_gold'] for _,d in grp):8.1f}")

    if win and lose:
        lo, hi = boot_ci([x for x, _ in win], [x for x, _ in lose])
        diff = st.mean(x for x, _ in win) - st.mean(x for x, _ in lose)
        print(f"\nHiệu mật độ (thắng − còn lại): {100*diff:+.1f} điểm %  "
              f"CI 95% [{100*lo:+.1f}, {100*hi:+.1f}]")
        if lo > 0:
            print("  -> ỦNG HỘ giả thuyết: câu IterCOMP thắng gold có nhiều từ")
            print("     mơ hồ hơn, đúng như 'ngữ cảnh rộng giúp khử mơ hồ' dự đoán.")
        elif hi < 0:
            print("  -> NGƯỢC giả thuyết: câu IterCOMP thắng lại ÍT mơ hồ hơn.")
        else:
            print("  -> KHÔNG kết luận được (CI chứa 0). Mật độ mơ hồ không")
            print("     giải thích được vì sao IterCOMP thắng; giả thuyết chưa")
            print("     có bằng chứng và phải viết là suy đoán.")

    # ── 2. vi du cu the cho case study ──────────────────────────────
    print(f"\n{'='*70}\nCASE STUDY — {args.examples} ví dụ IterCOMP thắng gold rõ nhất\n")
    best = sorted(win, key=lambda x: -(x[1]["f1_itercomp"] - x[1]["f1_gold"]))
    for i, (dens, d) in enumerate(best[:args.examples], 1):
        amb = [d["question"][a:b] for a, b in syllable_spans(d["question"])
               if len(homographs.get(strip_tones(d["question"][a:b].lower()),
                                     ())) >= 2]
        print(f"[{i}] câu hỏi: {d['question'][:110]}")
        print(f"    đáp án vàng     : {d['gold'][:70]}")
        print(f"    IterCOMP trả lời: {d['pred_itercomp'][:70]}  (F1 {d['f1_itercomp']:.0f})")
        print(f"    gold-ctx trả lời: {d['pred_gold'][:70]}  (F1 {d['f1_gold']:.0f})")
        print(f"    âm tiết mơ hồ   : {', '.join(amb[:8]) or '(không có)'}")
        print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({
            "config": vars(args) | {"out": str(args.out), "exp2": str(args.exp2)},
            "n_homograph_groups": len(homographs),
            "density_win": st.mean(x for x, _ in win) if win else None,
            "density_lose": st.mean(x for x, _ in lose) if lose else None,
            "n_win": len(win), "n_lose": len(lose),
            "examples": [d for _, d in best[:args.examples]],
        }, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
        print(f"→ ghi {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
