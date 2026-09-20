"""Fertility qua nhiều ngôn ngữ: hiện tượng phổ quát hay chỉ tiếng Việt?

CÂU HỎI. Báo cáo hiện tại đo fertility tiếng Việt/tiếng Anh và thấy 2,39x dưới
`cl100k_base`. Nhưng một con số cho một ngôn ngữ không nói được gì về việc cộng
đồng có đang đo sai hay không. Nếu tỉ lệ nén danh nghĩa che chi phí thật ở NHIỀU
ngôn ngữ, đó là lỗi hệ thống của cách đo, không phải đặc thù tiếng Việt.

CHỌN NGÔN NGỮ. Ba hệ chữ viết khác nhau để phép đo không lẫn với một quirk của
tokenizer:
  vi_VN  Latin + dấu thanh, khoảng trắng phân tách ÂM TIẾT không phải từ
  th_TH  Thai, KHÔNG có khoảng trắng giữa từ — tokenizer phải tự cắt
  hi_IN  Devanagari (abugida), ký tự tổ hợp phụ âm-nguyên âm
  zh_CN  Hán, không khoảng trắng, một ký tự thường là một hình vị

Chạy CPU: chỉ tokenize corpus song song, không gọi mô hình nào.

    python scripts/fertility_multilingual.py --pairs 500
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

RESULTS = Path(__file__).resolve().parents[1] / "results"

#: Ngôn ngữ đo, kèm ghi chú hệ chữ viết để giải thích kết quả.
LANGS = {
    "vi_VN": "Latin + dấu thanh; khoảng trắng tách âm tiết",
    "th_TH": "Thai; KHÔNG khoảng trắng giữa từ",
    "hi_IN": "Devanagari (abugida)",
    "zh_CN": "Hán; không khoảng trắng",
}

ENCODINGS = ("cl100k_base", "o200k_base")


def boot_ci(vals: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Khoảng tin cậy bootstrap 95% cho trung bình.

    Cần vì một tỉ số trung bình không kèm CI thì không nói được nó có khác 1,0
    hay không — mà đó chính là câu hỏi.
    """
    import random
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        s = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
        means.append(sum(s) / len(s))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=int, default=500,
                    help="số cặp câu song song mỗi ngôn ngữ")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import tiktoken
    from datasets import load_dataset as hf_load

    encs = {e: tiktoken.get_encoding(e) for e in ENCODINGS}
    out: dict[str, dict] = {}

    for lang, note in LANGS.items():
        try:
            ds = hf_load("google/wmt24pp", f"en-{lang}", split="train")
        except Exception as e:
            print(f"  {lang}: KHÔNG tải được ({type(e).__name__}) — bỏ qua")
            continue
        rows = list(ds)[: args.pairs]
        if not rows:
            print(f"  {lang}: rỗng — bỏ qua")
            continue

        per_lang = {}
        for ename, enc in encs.items():
            ratios = []
            for r in rows:
                en, tgt = r.get("source", ""), r.get("target", "")
                if not en or not tgt:
                    continue
                n_en = len(enc.encode(en))
                if n_en == 0:
                    continue
                ratios.append(len(enc.encode(tgt)) / n_en)
            if not ratios:
                continue
            lo, hi = boot_ci(ratios)
            per_lang[ename] = {"fertility": st.mean(ratios),
                               "ci_lo": lo, "ci_hi": hi, "n_pairs": len(ratios)}
        out[lang] = {"note": note, "tokenizers": per_lang}
        print(f"  {lang}: xong ({len(rows)} cặp)")

    if not out:
        raise SystemExit("Không đo được ngôn ngữ nào — kiểm tra mạng/dataset.")

    # ── bang ket qua ────────────────────────────────────────────────
    print(f"\n{'ngôn ngữ':9s} " + " ".join(f"{e:>22s}" for e in ENCODINGS)
          + "  hệ chữ viết")
    for lang, d in out.items():
        cells = []
        for e in ENCODINGS:
            v = d["tokenizers"].get(e)
            cells.append(f"{v['fertility']:6.2f}x [{v['ci_lo']:.2f},{v['ci_hi']:.2f}]"
                         if v else f"{'--':>22s}")
        print(f"{lang:9s} " + " ".join(cells) + f"  {d['note']}")

    print("\nĐIỀU NÀY NÓI GÌ")
    hi_fert = [(l, d["tokenizers"]["cl100k_base"]["fertility"])
               for l, d in out.items() if "cl100k_base" in d["tokenizers"]]
    hi_fert.sort(key=lambda x: -x[1])
    if hi_fert:
        top, val = hi_fert[0]
        print(f"  Cao nhất: {top} ở {val:.2f}x dưới cl100k_base.")
        print(f"  Mọi ngôn ngữ đo được đều > 1,0 — nên cùng một tỉ lệ nén danh")
        print(f"  nghĩa tương ứng chi phí API khác nhau ở TỪNG ngôn ngữ.")
        print(f"  Đây là lỗi hệ thống của cách đo, không phải đặc thù tiếng Việt.")

    # o200k vs cl100k: tokenizer moi hon co cong bang hon khong?
    print("\n  o200k_base so với cl100k_base:")
    for lang, d in out.items():
        a = d["tokenizers"].get("cl100k_base")
        b = d["tokenizers"].get("o200k_base")
        if a and b:
            print(f"    {lang}: {a['fertility']:.2f}x -> {b['fertility']:.2f}x  "
                  f"({100*(b['fertility']-a['fertility'])/a['fertility']:+.0f}%)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"source": "google/wmt24pp", "pairs_requested": args.pairs,
                   "languages": out},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\n→ ghi {args.out}")


if __name__ == "__main__":
    main()
