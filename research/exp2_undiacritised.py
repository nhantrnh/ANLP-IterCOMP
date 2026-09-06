"""EXP-2: vòng lặp còn chạy được trên tiếng Việt KHÔNG DẤU không?

Phần Limitations của báo cáo tự nêu: chúng tôi chưa đánh giá tiếng Việt không
dấu, dù đó là dạng rất phổ biến trên web (bình luận, tin nhắn, truy vấn tìm
kiếm). Đây là thí nghiệm lấp chỗ đó.

Vì sao câu hỏi này không tầm thường. Gỡ dấu làm nhập nhằng hàng loạt âm tiết:
"má" (mẹ), "mà" (nhưng), "mã" (code), "mả" (mộ), "mạ" (lúa non) đều thành "ma".
Với bộ nén, hệ quả nằm ở *hai chỗ khác nhau*:

  1. TÍNH ĐIỂM  bge-m3 phải khớp truy vấn với segment đã mất dấu. Nếu điểm
     tương đồng sụp thì bước lọc chọn sai bằng chứng, và lỗi này xảy ra TRƯỚC
     khi reader nhìn thấy gì.
  2. ĐỌC ĐÁP ÁN  reader phải sinh đáp án CÓ DẤU từ ngữ cảnh KHÔNG DẤU, vì
     nhãn vàng của VimQA có dấu.

Nếu chỉ đo F1 cuối thì hai nguyên nhân trộn vào nhau. Nên có chế độ "gold" để
tách: gold bỏ qua hoàn toàn bộ nén (đưa thẳng câu vàng), nên F1 tụt ở gold là
lỗi của READER, còn phần tụt thêm ở itercomp so với gold là lỗi của BỘ NÉN.

Ba chế độ dữ liệu:
  keep    giữ dấu — đường cơ sở, bằng số đã có trong báo cáo
  strip   gỡ dấu ở NGỮ CẢNH, nhãn vàng giữ dấu  <- tình huống thực tế
  both    gỡ dấu ở cả ngữ cảnh và nhãn vàng     <- tách phần lỗi do sinh dấu

Chú ý khi đọc: "both" không phải một thiết lập thực tế, nó là công cụ chẩn
đoán. Nếu strip tụt mạnh mà both thì không, nguyên nhân là reader không sinh
được dấu chứ không phải bằng chứng bị chọn sai.
"""
from __future__ import annotations
import argparse, json, sys, unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # để chạy được cả khi scripts/ nằm trong src/

from itercomp import (load_dataset, make_llm, make_reader, make_scorer,  # noqa: E402
                      clean_answer, f1_score, exact_match, normalize_boolean,
                      tokenize, paired_bootstrap)
from itercomp.core import decompose, is_answerable, make_followup, score_segments  # noqa: E402
from itercomp.scorer import percentile_filter  # noqa: E402
from itercomp import metrics as _metrics  # noqa: E402


#: Đ/đ không phải dấu phụ Unicode — NFD không tách được, phải map tay.
_DSTROKE = str.maketrans({"đ": "d", "Đ": "D"})


def strip_diacritics(text: str) -> str:
    """Gỡ toàn bộ dấu tiếng Việt, giữ nguyên chữ cái cơ sở.

    Dùng NFD rồi bỏ ký tự tổ hợp. Riêng đ/Đ là chữ cái riêng trong bảng chữ
    cái tiếng Việt, không phải d + dấu phụ, nên Unicode không tách ra được;
    bỏ sót nó sẽ để lại "đ" lẫn trong văn bản đã gỡ dấu và làm sai thí nghiệm.
    """
    s = unicodedata.normalize("NFD", text.translate(_DSTROKE))
    return unicodedata.normalize(
        "NFC", "".join(c for c in s if not unicodedata.combining(c)))

# Ở chế độ "both" nhãn vàng cũng bị gỡ dấu, nên "đúng"/"không" thành
# "dung"/"khong" và rơi ra ngoài từ vựng của normalize_boolean -- phép chuẩn
# hoá tắt lặng lẽ, đúng trên 34% dữ liệu (lớp đáp án lớn nhất của VimQA). Khi
# đó "both" sẽ tụt vì mất +10.0 F1 chuẩn hoá, không phải vì gỡ dấu, và cả thí
# nghiệm quy sai nguyên nhân. Thêm dạng không dấu vào từ vựng để phép chuẩn
# hoá vẫn hoạt động; không sửa logic lõi, chỉ mở rộng tập từ.
_metrics._YES = _metrics._YES | {strip_diacritics(w) for w in _metrics._YES}
_metrics._NO = _metrics._NO | {strip_diacritics(w) for w in _metrics._NO}

_nb_orig = _metrics.normalize_boolean


def _normalize_boolean_scriptaware(pred: str, gold: str) -> str:
    """normalize_boolean, nhưng trả về đáp án CÙNG HỆ CHỮ với nhãn vàng.

    Hàm gốc trả chuỗi cứng "đúng"/"không" (luôn có dấu). Ở chế độ "both" nhãn
    vàng là "dung"/"khong", nên dự đoán ĐÚNG vẫn bị tính F1 = 0 vì so "đúng"
    với "dung". Lỗi này chỉ xuất hiện khi nhãn vàng không dấu, và nó nuốt trọn
    lớp boolean — 34% VimQA — khiến "both" tụt vì lý do không liên quan tới
    việc gỡ dấu.

    Ở đây gỡ dấu kết quả khi và chỉ khi nhãn vàng cũng không dấu, nên chế độ
    "keep"/"strip" giữ nguyên hành vi gốc.
    """
    out = _nb_orig(pred, gold)
    if out is not pred and gold == strip_diacritics(gold):
        return strip_diacritics(out)
    return out


_metrics.normalize_boolean = _normalize_boolean_scriptaware
normalize_boolean = _normalize_boolean_scriptaware


def strip_row(row: dict) -> dict:
    """Bản sao của row với ngữ cảnh đã gỡ dấu. Câu hỏi cũng gỡ, vì người dùng
    gõ không dấu thì gõ không dấu cả câu hỏi."""
    ctx = row["context"]
    return row | {
        "question": strip_diacritics(row["question"]),
        "context": {"title": [strip_diacritics(t) for t in ctx["title"]],
                    "sentences": [[strip_diacritics(s) for s in doc]
                                  for doc in ctx["sentences"]]},
    }


def gold_context(row) -> str:
    sf = row.get("supporting_facts") or {}
    want = set(zip(sf.get("title", []), sf.get("sent_id", [])))
    keep = [f"{t}: {s}"
            for t, sents in zip(row["context"]["title"], row["context"]["sentences"])
            for i, s in enumerate(sents) if (t, i) in want]
    return "\n".join(keep)


def run_itercomp(llm, row, scorer, pct, max_iter) -> str:
    """Vòng lặp IterCOMP nguyên bản, không đổi gì."""
    segs = decompose(row["context"])
    remaining, selected, query = list(range(len(segs))), [], row["question"]
    for _ in range(max_iter):
        if not remaining:
            break
        scores = score_segments(query, [segs[i] for i in remaining], scorer)
        picked = [remaining[j] for j in percentile_filter(scores, pct)]
        selected.extend(segs[i] for i in picked)
        remaining = [i for i in remaining if i not in set(picked)]
        if is_answerable(llm, row["question"], selected):
            break
        query = make_followup(llm, row["question"], selected)
    return "\n".join(f"{s.doc_title}: {s.text}" for s in selected)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--reader", required=True, choices=["mock", "hf"])
    ap.add_argument("--reader-model", default=None)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--scorer", default="dual")
    ap.add_argument("--percentile", type=float, default=90.0)
    ap.add_argument("--max-iter", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.reader == "hf" and not args.reader_model:
        ap.error("--reader hf cần --reader-model")

    kw = {"model": args.reader_model} if args.reader_model else {}
    if args.load_4bit and args.reader == "hf":
        kw["load_4bit"] = True
    reader = make_reader(args.reader, **kw)
    llm = (make_llm("hf", model=args.reader_model, load_4bit=args.load_4bit)
           if args.reader == "hf" else make_llm("mock"))
    scorer = make_scorer(args.scorer)
    rows = load_dataset("vimqa", args.limit)

    print(f"dataset=vimqa n={len(rows)}  scorer={args.scorer}")
    print(f"reader={args.reader}{'/' + args.reader_model if args.reader_model else ''}\n")

    out: dict[str, dict] = {}
    for mode in ("keep", "strip", "both"):
        f1_ic, f1_gold, tok_ic, tok_full, dropped = [], [], [], [], 0
        # Giữ chi tiết từng câu để làm case study: reviewer sẽ đòi bằng chứng
        # cho giả thuyết "ngữ cảnh rộng giúp khử mơ hồ", và giả thuyết đó chỉ
        # kiểm được khi biết CÂU NÀO IterCOMP thắng gold và nó trả lời gì.
        detail = []
        for r in rows:
            # Ngữ cảnh gỡ dấu ở strip/both; nhãn vàng chỉ gỡ ở both.
            src = r if mode == "keep" else strip_row(r)
            gold = r["answer"] if mode != "both" else strip_diacritics(r["answer"])

            ctx_ic = run_itercomp(llm, src, scorer, args.percentile, args.max_iter)
            ctx_gd = gold_context(src)
            if not ctx_gd:
                dropped += 1
            full = "\n".join(f"{s.doc_title}: {s.text}" for s in decompose(src["context"]))
            tok_ic.append(len(tokenize(ctx_ic)))
            tok_full.append(len(tokenize(full)))

            preds = {}
            for name, ctx, acc in (("itercomp", ctx_ic, f1_ic),
                                   ("gold", ctx_gd, f1_gold)):
                pred = normalize_boolean(clean_answer(reader(ctx, src["question"])), gold)
                acc.append(f1_score(pred, gold) * 100)
                preds[name] = pred
            detail.append({
                "question": src["question"],
                "gold": gold,
                "pred_itercomp": preds["itercomp"],
                "pred_gold": preds["gold"],
                "f1_itercomp": f1_ic[-1],
                "f1_gold": f1_gold[-1],
                # ngữ cảnh cắt ngắn: đủ để đọc case study, không phình file
                "ctx_itercomp": ctx_ic[:400],
                "ctx_gold": ctx_gd[:400],
            })

        n = len(rows)
        out[mode] = {"f1_itercomp": sum(f1_ic) / n, "f1_gold": sum(f1_gold) / n,
                     "ratio": sum(tok_ic) / max(sum(tok_full), 1),
                     "per_row_itercomp": f1_ic, "per_row_gold": f1_gold,
                     "detail": detail,
                     "rows_without_gold": dropped}
        print(f"  {mode:6s} IterCOMP F1={out[mode]['f1_itercomp']:5.1f}   "
              f"Gold F1={out[mode]['f1_gold']:5.1f}   "
              f"tỉ lệ={out[mode]['ratio']*100:5.1f}%", flush=True)

    # ── tách nguyên nhân ────────────────────────────────────────────
    d_ic = out["strip"]["f1_itercomp"] - out["keep"]["f1_itercomp"]
    d_gd = out["strip"]["f1_gold"] - out["keep"]["f1_gold"]
    t_ic = paired_bootstrap(out["strip"]["per_row_itercomp"],
                            out["keep"]["per_row_itercomp"])
    t_gd = paired_bootstrap(out["strip"]["per_row_gold"],
                            out["keep"]["per_row_gold"])

    print(f"\n{'=' * 66}")
    print(f"Gỡ dấu làm F1 thay đổi: IterCOMP {d_ic:+.1f}  (p={t_ic['p']:.3f})")
    print(f"                        Gold     {d_gd:+.1f}  (p={t_gd['p']:.3f})")
    print(f"{'=' * 66}")
    # Gold không qua bộ nén, nên phần tụt của nó là lỗi reader. Phần IterCOMP
    # tụt THÊM so với gold mới là phần quy được cho bộ nén.
    extra = d_ic - d_gd
    print(f"Quy trách nhiệm: reader mất {abs(d_gd):.1f} điểm, "
          f"bộ nén mất thêm {abs(extra):.1f}")
    if abs(extra) < 2.0:
        print("=> Bộ nén CHỊU ĐƯỢC việc gỡ dấu; thiệt hại nằm ở reader.")
    else:
        print("=> Bộ nén cũng suy giảm — bước tính điểm bị ảnh hưởng bởi nhập nhằng.")
    print(f"Chẩn đoán 'both' (nhãn vàng cũng gỡ dấu): "
          f"IterCOMP F1={out['both']['f1_itercomp']:.1f} "
          f"so với strip {out['strip']['f1_itercomp']:.1f} — "
          f"chênh {out['both']['f1_itercomp'] - out['strip']['f1_itercomp']:+.1f} "
          f"là phần do reader không sinh được dấu.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"config": vars(args) | {"out": str(args.out)},
                   "results": {k: {kk: vv for kk, vv in v.items()
                                   if not kk.startswith("per_row")
                                   and kk != "detail"}
                               for k, v in out.items()},
                   # detail tách riêng: nó là dữ liệu cho case study, không
                   # phải số tổng hợp, và người đọc summary không cần nó.
                   "detail": {k: v["detail"] for k, v in out.items()},
                   "delta_itercomp": d_ic, "delta_gold": d_gd,
                   "attributable_to_compressor": extra,
                   "test_itercomp": t_ic, "test_gold": t_gd},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
        print(f"\n→ ghi {args.out}")


if __name__ == "__main__":
    main()
