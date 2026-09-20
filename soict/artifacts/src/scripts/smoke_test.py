"""Smoke-test môi trường trước khi bắt tay vào IterCOMP.

Chạy:  python scripts/smoke_test.py

Kiểm 4 thứ, mỗi thứ độc lập — cái nào fail thì báo rõ cái đó, không dừng cả script:
  1. Đọc được VimQA parquet (tiếng Việt)
  2. Import + load được LLMLingua-2 (multilingual, có tiếng Việt)
  3. Nén thật 1 câu VimQA, in ra tỉ lệ nén
  4. Đếm token — kiểm chứng luận điểm "tiếng Việt tốn nhiều token hơn tiếng Anh"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # để chạy được cả khi scripts/ nằm trong src/


import sys
from pathlib import Path

from itercomp import DATA  # dùng chung một nguồn path với run_eval (gốc repo /data)
# Model multilingual (có tiếng Việt). Bản bert-base nhẹ hơn xlm-roberta-large.
COMPRESSOR = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

results = {}


def run_step(name, fn):
    """Chạy 1 bước, bắt lỗi, ghi kết quả. Trả về giá trị của fn (hoặc None nếu fail)."""
    print(f"\n{'='*60}\n[{name}]\n{'='*60}")
    try:
        out = fn()
        results[name] = "OK"
        return out
    except Exception as e:
        results[name] = f"FAIL: {type(e).__name__}: {e}"
        print(f"  ✗ {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------- 1. data
def load_vimqa():
    import pandas as pd
    p = DATA / "vimqa" / "validation.parquet"
    if not p.exists():
        raise FileNotFoundError(f"chưa có {p} — xem README để tải")
    df = pd.read_parquet(p)
    r = df.iloc[0]
    print(f"  rows: {len(df)}  cols: {list(df.columns)}")
    print(f"  Q: {r['question']}")
    print(f"  A: {r['answer']}   type: {r['type']}")
    print(f"  n_context: {len(r['context']['title'])}")
    return df


df = run_step("1. Đọc VimQA", load_vimqa)


# ------------------------------------------------------- 2. load compressor
def load_compressor():
    from llmlingua import PromptCompressor
    import torch

    # MPS (Apple Silicon) chưa ổn với 1 số op của llmlingua -> ưu tiên CPU cho model nhỏ này.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}  (torch {torch.__version__})")
    print(f"  model: {COMPRESSOR}")
    print("  ... lần đầu sẽ tải ~700MB, chờ một chút")
    c = PromptCompressor(
        model_name=COMPRESSOR,
        use_llmlingua2=True,
        device_map=device,
    )
    print("  ✓ load xong")
    return c


compressor = run_step("2. Load LLMLingua-2", load_compressor)


# ------------------------------------------------------------- 3. nén thật
def compress_one():
    if df is None or compressor is None:
        raise RuntimeError("bước 1 hoặc 2 fail nên bỏ qua")

    r = df.iloc[0]
    ctx = r["context"]
    # Ghép 10 đoạn context thành 1 prompt dài (đúng như setting distractor)
    docs = [
        f"{t}: {' '.join(sents)}"
        for t, sents in zip(ctx["title"], ctx["sentences"])
    ]
    prompt = "\n\n".join(docs)

    print(f"  câu hỏi: {r['question']}")
    print(f"  prompt gốc: {len(prompt)} ký tự, {len(docs)} đoạn")

    out = compressor.compress_prompt(
        prompt,
        rate=0.33,                    # giữ ~1/3 token
        force_tokens=["\n", "?", ".", ","],
        drop_consecutive=True,
    )

    print(f"\n  --- kết quả nén ---")
    for k in ("origin_tokens", "compressed_tokens", "rate", "ratio"):
        if k in out:
            print(f"  {k}: {out[k]}")
    comp = out["compressed_prompt"]
    print(f"\n  prompt sau nén (200 ký tự đầu):\n    {comp[:200]}...")

    # Dấu tiếng Việt còn nguyên không? (quan trọng: nén không được phá dấu)
    has_diacritics = any(ch in comp for ch in "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệ"
                                             "ìíỉĩịòóỏõọôồốổỗộơờớởỡợ"
                                             "ùúủũụưừứửữựỳýỷỹỵđ")
    print(f"\n  dấu tiếng Việt còn nguyên: {'✓ có' if has_diacritics else '✗ MẤT DẤU — cần điều tra'}")
    return out


run_step("3. Nén 1 câu VimQA", compress_one)


# ------------------------------------------- 4. token fertility (VN vs EN)
def token_fertility():
    """Kiểm chứng luận điểm cải tiến: tiếng Việt tốn nhiều token hơn tiếng Anh.

    Nếu đúng -> cùng 1 tỉ lệ nén, tiếng Việt tiết kiệm nhiều token thực tế hơn
    -> đây là contribution định lượng được cho phần điểm cộng.
    """
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")   # tokenizer của GPT-4/4o

    pairs = [
        ("Diego Maradona sinh năm 1960 tại Argentina và là cầu thủ bóng đá nổi tiếng.",
         "Diego Maradona was born in 1960 in Argentina and was a famous football player."),
        ("Thành phố Hồ Chí Minh là trung tâm kinh tế lớn nhất của Việt Nam.",
         "Ho Chi Minh City is the largest economic center of Vietnam."),
    ]
    print(f"  {'':4s} {'VN tok':>7s} {'EN tok':>7s} {'tỉ lệ':>7s}   {'VN từ':>6s} {'tok/từ VN':>10s}")
    ratios = []
    for i, (vi, en) in enumerate(pairs, 1):
        nv, ne = len(enc.encode(vi)), len(enc.encode(en))
        words = len(vi.split())
        ratios.append(nv / ne)
        print(f"  #{i}   {nv:7d} {ne:7d} {nv/ne:7.2f}x  {words:6d} {nv/words:10.2f}")
    avg = sum(ratios) / len(ratios)
    print(f"\n  → tiếng Việt tốn TRUNG BÌNH {avg:.2f}x token so với tiếng Anh")
    print(f"  → luận điểm cải tiến {'ĐƯỢC củng cố ✓' if avg > 1.15 else 'cần đo trên tập lớn hơn'}")
    return avg


run_step("4. Token fertility VN vs EN", token_fertility)


# ----------------------------------------------------------------- summary
print(f"\n{'='*60}\nTỔNG KẾT\n{'='*60}")
for k, v in results.items():
    print(f"  {'✓' if v == 'OK' else '✗'} {k:32s} {v}")

failed = [k for k, v in results.items() if v != "OK"]
if failed:
    print(f"\n{len(failed)} bước fail. Sửa xong rồi chạy lại trước khi làm itercomp.py")
    sys.exit(1)
print("\nMôi trường OK — sẵn sàng chạy itercomp.py")
