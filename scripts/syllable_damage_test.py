"""Chênh lệch mất-hư-từ giữa tiếng Việt và tiếng Anh có thật không?

`syllable_damage.py` đo LLMLingua-2 cắt hư từ lệch 1,47x trên VimQA và 1,37x
trên MuSiQue. Câu hỏi duy nhất đáng hỏi tiếp: 0,10 đó là hiệu ứng hay là nhiễu
của n=50? Không kiểm định thì không được viết chữ "đặc thù tiếng Việt" nào.

Bootstrap phân tầng: lấy mẫu lại độc lập trong từng tập, dựng phân bố của hiệu
`tỉ số(vi) - tỉ số(en)`, đọc khoảng tin cậy 95%. CI chứa 0 nghĩa là dữ liệu
hiện có KHÔNG phân biệt được hai ngôn ngữ.

    python scripts/syllable_damage_test.py
"""
from __future__ import annotations

import json
import random
import statistics as st
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
B = 5000


def ratios(path: Path, method: str) -> list[float]:
    """Tỉ số mất-hư-từ / mất-nội-dung cho từng câu hỏi.

    Bỏ những dòng mà mẫu số bằng 0 hoặc NaN — không định nghĩa được tỉ số,
    và giữ lại sẽ bịa ra tín hiệu.
    """
    rows = json.loads(path.read_text())["per_row"][method]
    out = []
    for r in rows:
        fw, ct = r.get("hư từ"), r.get("nội dung")
        if fw is None or ct is None:
            continue
        if fw != fw or ct != ct or ct <= 0:
            continue
        out.append(fw / ct)
    return out


def boot_diff(a: list[float], b: list[float], seed: int = 0):
    rng = random.Random(seed)
    diffs = []
    for _ in range(B):
        ra = sum(a[rng.randrange(len(a))] for _ in range(len(a))) / len(a)
        rb = sum(b[rng.randrange(len(b))] for _ in range(len(b))) / len(b)
        diffs.append(ra - rb)
    diffs.sort()
    return diffs[int(0.025 * B)], diffs[int(0.975 * B)]


def main():
    print(f"Bootstrap {B} lần, tỉ số = mất hư từ / mất nội dung\n")
    print(f"{'phương pháp':13s} {'vi (VimQA)':>13s} {'en (MuSiQue)':>14s} "
          f"{'hiệu':>8s} {'CI 95% của hiệu':>22s}")

    verdicts = {}
    for m in ("itercomp", "llmlingua2"):
        vi = ratios(RESULTS / "syllable_damage_vimqa.json", m)
        en = ratios(RESULTS / "syllable_damage_musique.json", m)
        lo, hi = boot_diff(vi, en)
        d = st.mean(vi) - st.mean(en)
        verdicts[m] = (lo, hi)
        print(f"{m:13s} {st.mean(vi):12.2f}x {st.mean(en):13.2f}x {d:+8.2f} "
              f"{f'[{lo:+.2f}, {hi:+.2f}]':>22s}")

    print("\nKẾT LUẬN")
    for m, (lo, hi) in verdicts.items():
        sig = lo > 0 or hi < 0
        print(f"  {m}: {'CÓ' if sig else 'KHÔNG'} khác biệt vi/en ở mức 95%"
              + ("" if sig else " — CI chứa 0"))

    lo, hi = verdicts["llmlingua2"]
    if not (lo > 0 or hi < 0):
        print("\n  => KHÔNG được viết 'LLMLingua-2 hại tiếng Việt hơn tiếng Anh'.")
        print("     Cái đo được là: nén mức TOKEN cắt hư từ lệch hơn nén mức CÂU,")
        print("     ở CẢ HAI ngôn ngữ. Đó vẫn là phát hiện, nhưng không phải")
        print("     phát hiện đặc thù ngôn ngữ.")


if __name__ == "__main__":
    main()
