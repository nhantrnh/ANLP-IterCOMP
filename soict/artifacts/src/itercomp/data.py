"""Nạp và chuẩn hoá bốn bộ dữ liệu hỏi đáp đa bước.

Ba định dạng lưu trữ khác nhau được đưa về một cấu trúc chung:

    HotpotQA, VimQA  dict-of-columns
    2WikiMQA         chuỗi JSON [[title, [sents]], ...]
    MuSiQue          list đoạn văn, gắn cờ is_supporting thay cho supporting_facts

Đây là nguồn lỗi âm thầm: sai định dạng khiến `oracle` lặng lẽ rơi về ngữ cảnh
đầy đủ và cho số vô nghĩa."""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"


def load_dataset(name: str, limit: int | None = None) -> list[dict]:
    """Nạp dataset về format thống nhất: question / answer / context / type.

    limit=None -> nạp TOÀN BỘ tập (dùng khi tái hiện đầy đủ như paper).
    """
    import pandas as pd

    if name == "vimqa":
        df = pd.read_parquet(DATA / "vimqa" / "validation.parquet")
    elif name == "hotpotqa":
        df = pd.read_parquet(DATA / "hotpotqa" / "validation.parquet")
    elif name == "vihotpot":
        # HotpotQA distractor dịch máy sang tiếng Việt (dataset multi-hop VN thứ
        # hai; xem scripts/translate_hotpotqa.py). Cùng schema với hotpotqa.
        df = pd.read_parquet(DATA / "vihotpot" / "validation.parquet")
    elif name == "2wiki":
        df = pd.read_parquet(DATA / "2wiki" / "dev.parquet")
    elif name == "musique":
        rows = [json.loads(l) for l in open(DATA / "musique" / "dev.jsonl", encoding="utf-8")]
        if limit is not None:
            rows = rows[:limit]
        out = []
        for r in rows:
            rec = {
                "question": r["question"],
                "answer": r["answer"],
                "type": "musique",
                "context": {
                    "title": [p["title"] for p in r["paragraphs"]],
                    "sentences": [[p["paragraph_text"]] for p in r["paragraphs"]],
                },
            }
            # MuSiQue không có trường supporting_facts riêng; nó gắn cờ
            # is_supporting lên từng đoạn -> dựng lại để phương pháp "oracle" chạy được.
            sup = [(p["title"], 0) for p in r["paragraphs"] if p.get("is_supporting")]
            if sup:
                rec["supporting_facts"] = {"title": [t for t, _ in sup],
                                           "sent_id": [i for _, i in sup]}
            out.append(rec)
        return out
    else:
        raise ValueError(f"dataset không biết: {name}")

    def _ctx(c):
        """Chuẩn hoá context về {"title": [...], "sentences": [[...], ...]}.

        HotpotQA/VimQA lưu dạng dict-of-columns; 2Wiki lưu dạng CHUỖI JSON
        [[title, [sents]], ...]. Xử lý cả hai.
        """
        if isinstance(c, str):
            c = json.loads(c)
        if isinstance(c, dict):
            return {"title": list(c["title"]),
                    "sentences": [list(x) for x in c["sentences"]]}
        return {"title": [t for t, _ in c],
                "sentences": [list(s) for _, s in c]}

    def _sf(sf):
        """Chuẩn hoá supporting_facts về {"title": [...], "sent_id": [...]}."""
        if sf is None:
            return None
        if isinstance(sf, str):
            sf = json.loads(sf)
        if isinstance(sf, dict):
            if len(sf.get("title", [])) == 0:
                return None
            return {"title": list(sf["title"]),
                    "sent_id": [int(x) for x in sf["sent_id"]]}
        if len(sf) == 0:
            return None
        return {"title": [t for t, _ in sf], "sent_id": [int(i) for _, i in sf]}

    out = []
    for _, r in (df if limit is None else df.head(limit)).iterrows():
        rec = {
            "question": r["question"],
            "answer": r["answer"],
            "type": r.get("type", ""),
            "context": _ctx(r["context"]),
        }
        # supporting_facts: cần cho phương pháp "oracle" trong eval.py
        sf = _sf(r.get("supporting_facts"))
        if sf:
            rec["supporting_facts"] = sf
        out.append(rec)
    return out
