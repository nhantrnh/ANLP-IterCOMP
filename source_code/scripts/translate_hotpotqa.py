"""Dịch máy HotpotQA (distractor) sang tiếng Việt -> dataset multi-hop VN thứ hai.

VÌ SAO. Dataset multi-hop QA tiếng Việt DO NGƯỜI biên soạn chỉ có VimQA. Đề cho
phép xây dữ liệu bằng DỊCH MÁY, nên ta dịch một tập con HotpotQA sang tiếng Việt,
giữ nguyên cấu trúc supporting-facts, tạo bộ đánh giá multi-hop tiếng Việt thứ
hai. Chất lượng phụ thuộc bản dịch — phải khai là caveat.

Giữ nguyên: title (định danh đoạn) và supporting_facts (title + sent_id).
Dịch: question, mọi câu trong context, và answer (yes/no -> có/không).

    python scripts/translate_hotpotqa.py --limit 500      # cần GPU
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
YESNO = {"yes": "có", "no": "không"}


def build_translator(model_name: str, batch: int):
    """Trả về hàm dịch batch En->Vi bằng VietAI/envit5-translate (mặc định)."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=dtype).to(dev).eval()

    def translate(texts: list[str]) -> dict[str, str]:
        # Dedupe: nhiều câu/tiêu đề lặp lại, dịch một lần là đủ.
        uniq = list(dict.fromkeys(t for t in texts if t and t.strip()))
        out: dict[str, str] = {}
        for i in range(0, len(uniq), batch):
            chunk = uniq[i:i + batch]
            enc = tok(["en: " + t for t in chunk], return_tensors="pt",
                      padding=True, truncation=True, max_length=512).to(dev)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=512, num_beams=1)
            for t, d in zip(chunk, tok.batch_decode(gen, skip_special_tokens=True)):
                out[t] = d.replace("vi:", "").replace("en:", "").strip()
            if (i // batch) % 20 == 0:
                print(f"  dịch {min(i + batch, len(uniq))}/{len(uniq)}", flush=True)
        return out

    return translate


def translate_dataset(src: pd.DataFrame, translate) -> pd.DataFrame:
    # Gom mọi chuỗi cần dịch rồi dịch một lượt (tận dụng batch + dedupe).
    pool: list[str] = []
    for _, r in src.iterrows():
        pool.append(r["question"])
        for sents in r["context"]["sentences"]:
            pool.extend(list(sents))
        if str(r["answer"]).lower() not in YESNO:
            pool.append(r["answer"])
    tr = translate(pool)

    rows = []
    for _, r in src.iterrows():
        ctx = r["context"]
        new_sents = [[tr.get(s, s) for s in list(sents)]
                     for sents in ctx["sentences"]]
        a = str(r["answer"])
        ans = YESNO.get(a.lower(), tr.get(a, a))
        sf = r["supporting_facts"]
        rows.append({
            "id": r["id"],
            "question": tr.get(r["question"], r["question"]),
            "answer": ans,
            "type": r.get("type", ""),
            "level": r.get("level", ""),
            "supporting_facts": {"title": list(sf["title"]),
                                 "sent_id": [int(x) for x in sf["sent_id"]]},
            "context": {"title": list(ctx["title"]), "sentences": new_sents},
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--model", default="VietAI/envit5-translation")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=str(DATA / "vihotpot" / "validation.parquet"))
    args = ap.parse_args()

    src = pd.read_parquet(DATA / "hotpotqa" / "validation.parquet").head(args.limit)
    print(f"Dịch {len(src)} câu HotpotQA bằng {args.model} ...", flush=True)
    translate = build_translator(args.model, args.batch)
    out = translate_dataset(src, translate)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_parquet(args.out)
    print(f"✓ Đã dịch {len(out)} câu -> {args.out}")


if __name__ == "__main__":
    main()
