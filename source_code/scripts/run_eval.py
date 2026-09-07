"""Đánh giá: so sánh các phương pháp nén prompt trên cùng một reader.

Đây là script sinh ra BẢNG CHÍNH của báo cáo.

Phương pháp so sánh:
    raw         — Raw Document: nối toàn bộ tài liệu, không nén (paper gọi là "Raw Document")
    oracle      — Oracle: chỉ đưa gold supporting documents (TRẦN theo paper)
    llmlingua2  — LLMLingua-2 (Pan et al. 2024), đa ngữ nên chạy được tiếng Việt
    itercomp    — phương pháp của paper (tự cài đặt từ công thức)

Baseline paper dùng mà repo này CHƯA có: LLMLingua, LongLLMLingua, RECOMP,
Selective-Context, R2C. Đều có code công khai (xem README Mục 9).

Chạy:
    # kiểm thử luồng, không tốn tiền
    python scripts/run_eval.py --dataset vimqa --limit 5 --reader mock

    # số thật, miễn phí (reader chạy cục bộ)
    python scripts/run_eval.py --dataset vimqa --limit 50 --reader hf \
        --out results/vimqa_50.json
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# PHẢI đặt trước khi torch nạp, nếu không cờ bị bỏ qua. Giảm phân mảnh VRAM:
# ngữ cảnh multi-hop dài ngắn rất khác nhau nên mỗi câu xin một khối kích cỡ
# khác, bộ nhớ vụn dần và một câu dài giữa chừng sẽ OOM dù tổng vẫn còn chỗ.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from itercomp import (DATA, itercomp, load_dataset, make_llm,
                      exact_match, f1_score, tokenize, clean_answer, make_reader,
                      make_scorer, normalize_boolean,
                      bootstrap_ci, paired_bootstrap, min_detectable_diff)
from itercomp.scorer import (AP_K_MAX, AP_K_MIN, DEFAULT_LAMBDA,
                             DEFAULT_PERCENTILE, PERCENTILE_BY_DATASET)
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path


RESULTS = Path(__file__).parent / "results"


# ═════════════════════════════════════════════════ các phương pháp nén


def ctx_full(row) -> str:
    """Raw Document (paper) — nối toàn bộ tài liệu, không lọc gì."""
    return "\n".join(
        f"{t}: {' '.join(s)}"
        for t, s in zip(row["context"]["title"], row["context"]["sentences"])
    )


def ctx_oracle(row) -> str:
    """Oracle (paper) — chỉ gold supporting documents. Là TRẦN của mọi phương pháp nén.

    Cho biết TRẦN của mọi phương pháp nén: nếu oracle cũng thấp thì
    lỗi nằm ở reader, không phải ở compressor.
    """
    sf = row.get("supporting_facts")
    if not sf:
        return ctx_full(row)
    want = set(zip(sf["title"], sf["sent_id"]))
    keep = []
    for t, sents in zip(row["context"]["title"], row["context"]["sentences"]):
        for i, s in enumerate(sents):
            if (t, i) in want:
                keep.append(f"{t}: {s}")
    return "\n".join(keep) or ctx_full(row)


class LLMLingua2Method:
    """Lazy-load compressor (nặng ~700MB) — chỉ load khi thật sự dùng."""

    MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

    def __init__(self, rate: float = 0.33):
        self.rate = rate
        self._c = None

    def _compressor(self):
        if self._c is None:
            import torch
            from llmlingua import PromptCompressor
            self._c = PromptCompressor(
                model_name=self.MODEL, use_llmlingua2=True,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
            )
        return self._c

    def __call__(self, row) -> str:
        out = self._compressor().compress_prompt(
            ctx_full(row), rate=self.rate,
            force_tokens=["\n", "?", ".", ","], drop_consecutive=True,
        )
        return out["compressed_prompt"]




def _to_legacy(pkv):
    """Chuyển `Cache` của transformers về list-of-tuples cho llmlingua.

    LẦN VÁ THỨ SÁU, và lần này lỗi nằm ở CHÍNH HÀM NÀY. Bản cũ là:

        if hasattr(pkv, "to_legacy_cache"):
            return pkv.to_legacy_cache()
        return pkv                      # <-- IM LẶNG TRẢ VỀ Cache

    transformers mới đã BỎ `to_legacy_cache()`, nên `hasattr` sai và hàm trả
    về nguyên cái `Cache`. Mọi lớp bọc bên ngoài vẫn chạy đúng — traceback
    Kaggle cho thấy `_iter` CÓ được gọi — nhưng chúng chuyển tiếp một object
    không đổi. Thất bại im lặng, tốn một phiên GPU nữa.

    Nay thử đủ bốn API rồi NÉM nếu không xong. Ném ở giây thứ 5 tốt hơn vô
    vàn so với chết ở giây thứ 113 sau khi đã nạp mô hình 7B.
    """
    if pkv is None or isinstance(pkv, (list, tuple)):
        return pkv

    # 1. API cũ: transformers 4.45-4.5x
    if hasattr(pkv, "to_legacy_cache"):
        try:
            out = pkv.to_legacy_cache()
            if out is not None:
                return out
        except Exception:
            pass

    # 2. key_cache / value_cache song song
    ks, vs = getattr(pkv, "key_cache", None), getattr(pkv, "value_cache", None)
    if ks is not None and vs is not None:
        return [(k, v) for k, v in zip(ks, vs)]

    # 3. .layers, mỗi lớp có .keys/.values — transformers mới
    layers = getattr(pkv, "layers", None)
    if layers is not None:
        out = [(getattr(l, "keys", None), getattr(l, "values", None))
               for l in layers]
        if all(k is not None and v is not None for k, v in out):
            return out

    # 4. còn lập được thì lập
    try:
        return [(k, v) for k, v in pkv]
    except Exception:
        pass

    raise TypeError(
        f"Không chuyển được cache kiểu {type(pkv).__name__} về list-of-tuples "
        f"cho llmlingua (thuộc tính: "
        f"{[a for a in ('to_legacy_cache','key_cache','layers') if hasattr(pkv, a)]}). "
        f"transformers đã đổi API cache lần nữa — xem _to_legacy trong run_eval.py.")


def selftest_cache_patch() -> str:
    """Kiểm bản vá bằng `Cache` THẬT của transformers đang cài. Gọi TRƯỚC khi
    nạp mô hình: hỏng thì biết trong 5 giây, không phải sau 113 giây.
    """
    import torch, transformers
    from transformers.cache_utils import DynamicCache

    c = DynamicCache()
    for i in range(2):
        z = torch.zeros(1, 2, 3, 4)
        try:
            c.update(z, z, i)
        except TypeError:                       # chữ ký đổi ở bản mới
            c.update(z, z, layer_idx=i)

    legacy = _to_legacy(c)
    n = sum(1 for _k, _v in legacy)             # đúng vòng lặp của llmlingua
    assert n == 2, f"chuyển ra {n} lớp, đợi 2"
    return (f"transformers {transformers.__version__}, "
            f"{type(c).__name__} -> list-of-tuples OK ({n} lớp)")


def _force_legacy_cache(compressor) -> None:
    """Ép `past_key_values` về dạng tuple cũ cho llmlingua.

    llmlingua 0.2.2 lặp `for k, v in past_key_values`, giả định list các tuple.
    transformers >= 4.47 trả `Cache`, và vòng lặp đó vỡ với
    `ValueError: too many values to unpack`.

    BỐN LẦN VÁ SAI, ghi lại vì mỗi lần tốn một phiên GPU:

    1. Bọc `model.forward` — không đủ MỘT MÌNH: llmlingua gọi `self.model(...)`
       tức `__call__`, nhưng `__call__` của nn.Module lại gọi `forward`, nên
       bọc forward CÓ chặn được đầu ra. Cái nó không chặn là đầu VÀO.
    2. Bọc `get_ppl` — chuyển được đầu ra, nhưng `iterative_compress_prompt`
       truyền cache TRỞ LẠI ở vòng sau và mô hình lại trả `Cache`.
    3. Gán `mdl.__call__` — **Python tra dunder qua CLASS**, gán trên instance
       không có tác dụng. Lỗi cơ bản, test bắt được.
    4. Bọc `forward` + `get_ppl` — VẪN CHẾT trên Kaggle ở
       `iterative_compress_prompt` dòng 1659. `get_ppl` đọc
       `past_key_values[0][0].shape[2]` TRƯỚC khi gọi model, nên một `Cache`
       truyền vào đã vỡ ngay tại đó, trước khi lớp bọc đầu-ra kịp làm gì.
    5. Bản này bọc thêm chính `iterative_compress_prompt`, và chuyển cache ở
       CẢ HAI đầu của `get_ppl` (vào lẫn ra) thay vì chỉ đầu ra.

    Không kiểm được bằng cách chạy thật trên máy có transformers 4.45 — nó vốn
    trả tuple. `tests/test_legacy_cache_patch.py` giả lập `Cache` và mô phỏng
    vòng lặp nhiều bước, nên bắt được cả bốn lần sai ở trên.
    """
    mdl = getattr(compressor, "model", None)
    if mdl is None or getattr(mdl, "_itercomp_legacy_patched", False):
        return

    if hasattr(mdl, "config"):
        try:
            mdl.config.return_legacy_cache = True
        except Exception:
            pass

    orig_fwd = mdl.forward

    def _fwd(*a, **kw):
        if "past_key_values" in kw:
            kw["past_key_values"] = _to_legacy(kw["past_key_values"])
        out = orig_fwd(*a, **kw)
        pkv = getattr(out, "past_key_values", None)
        if pkv is not None:
            try:
                out.past_key_values = _to_legacy(pkv)
            except Exception:
                pass
        return out

    mdl.forward = _fwd

    # get_ppl cũng bọc: nó trả (loss, pkv) và llmlingua dùng pkv trực tiếp.
    if hasattr(compressor, "get_ppl"):
        orig_ppl = compressor.get_ppl

        def _ppl(*a, **kw):
            # get_ppl doc past_key_values[0][0].shape[2] TRUOC khi goi model,
            # nen mot Cache truyen vao vo ngay tai do -> phai chuyen dau VAO.
            if kw.get("past_key_values") is not None:
                kw["past_key_values"] = _to_legacy(kw["past_key_values"])
            out = orig_ppl(*a, **kw)
            if isinstance(out, tuple) and len(out) == 2:
                return out[0], _to_legacy(out[1])
            return out

        compressor.get_ppl = _ppl

    # iterative_compress_prompt (LongLLMLingua) tu lap `for k, v in pkv` o bon
    # cho. No nhan cache tu get_ppl — da vá — nhung vong DAU truyen None roi
    # nhan ve cache that, nen van phai chan o day cho chac.
    if hasattr(compressor, "iterative_compress_prompt"):
        orig_iter = compressor.iterative_compress_prompt

        def _iter(*a, **kw):
            if "past_key_values" in kw and kw["past_key_values"] is not None:
                kw["past_key_values"] = _to_legacy(kw["past_key_values"])
            return orig_iter(*a, **kw)

        compressor.iterative_compress_prompt = _iter

    mdl._itercomp_legacy_patched = True


class LLMLinguaMethod:
    """LLMLingua gốc (Jiang et al., 2023) — nén theo perplexity, KHÔNG học.

    Khác LLMLingua-2 ở chỗ căn bản: bản gốc dùng một LM nhân quả nhỏ chấm
    perplexity từng token rồi bỏ token dễ đoán, còn LLMLingua-2 huấn luyện một
    bộ phân loại token. Nên đây không phải "bản cũ hơn" mà là **họ phương pháp
    khác**, và bài báo IterCOMP liệt kê cả hai làm đường cơ sở riêng.

    Dùng GPT-2 thay vì LLaMA-7B của bài gốc: đây là bộ chấm perplexity, và
    GPT-2 chạy được trên T4 cùng lúc với reader 7B. Ghi rõ trong báo cáo vì
    nó làm baseline này yếu hơn bản gốc.
    """

    MODEL = "gpt2"

    def __init__(self, rate: float = 0.33, model: str | None = None):
        self.rate, self.model = rate, model or self.MODEL
        self._c = None

    def _compressor(self):
        if self._c is None:
            import torch
            from llmlingua import PromptCompressor
            self._c = PromptCompressor(
                model_name=self.model,
                device_map="cuda" if torch.cuda.is_available() else "cpu")
            _force_legacy_cache(self._c)
        return self._c

    def __call__(self, row) -> str:
        return self._compressor().compress_prompt(
            ctx_full(row), rate=self.rate,
            use_sentence_level_filter=False,
            use_context_level_filter=False,
            use_token_level_filter=True,
        )["compressed_prompt"]


class LongLLMLinguaMethod(LLMLinguaMethod):
    """LongLLMLingua (Jiang et al., 2024) — LLMLingua có ĐIỀU KIỆN theo câu hỏi.

    Điểm khác duy nhất mà cũng là điểm mấu chốt: nó xếp hạng từng đoạn theo
    mức liên quan tới câu hỏi (`rank_method='longllmlingua'`) rồi phân bổ ngân
    sách nén không đều — đoạn liên quan bị nén nhẹ hơn. Đó chính là ý tưởng
    IterCOMP đẩy xa hơn bằng vòng lặp, nên nó là đường cơ sở đáng so nhất.
    """

    def __call__(self, row) -> str:
        return self._compressor().compress_prompt(
            _ctx_docs(row), question=row["question"], rate=self.rate,
            rank_method="longllmlingua",
            condition_compare=True,
            dynamic_context_compression_ratio=0.3,
            context_budget="+100",
        )["compressed_prompt"]


class SelectiveContextMethod:
    """Selective-Context (Li et al., 2023) — bỏ đơn vị có self-information thấp.

    Không phụ thuộc câu hỏi: nó chỉ hỏi "câu này có bất ngờ không" theo một LM
    nhân quả, rồi giữ những câu bất ngờ nhất. Đó là đối chứng đúng để tách hai
    thứ: bao nhiêu phần hiệu quả đến từ **nén** nói chung, và bao nhiêu đến từ
    việc **biết câu hỏi**.

    Cài đặt trực tiếp thay vì dùng gói `selective-context`: gói đó ghim
    transformers cũ và xung đột với phần còn lại của repo. Thuật toán chỉ là
    self-information mức câu, nên viết thẳng ra rõ ràng hơn là gỡ phụ thuộc.
    """

    MODEL = "gpt2"

    def __init__(self, rate: float = 0.33, model: str | None = None):
        self.rate, self.model = rate, model or self.MODEL
        self._m = None

    def _model(self):
        if self._m is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            tok = AutoTokenizer.from_pretrained(self.model)
            mdl = AutoModelForCausalLM.from_pretrained(self.model).to(dev).eval()
            self._m = (tok, mdl, dev, torch)
        return self._m

    def _self_info(self, sent: str) -> float:
        """Self-information trung bình mỗi token: −(1/n)·Σ log p(token).

        Câu càng khó đoán thì giá trị càng cao, và Selective-Context giữ đúng
        những câu đó. Chuẩn hoá theo độ dài, nếu không thì câu dài luôn thắng.
        """
        tok, mdl, dev, torch = self._model()
        ids = tok(sent, return_tensors="pt", truncation=True,
                  max_length=512).input_ids.to(dev)
        if ids.shape[1] < 2:
            return 0.0
        with torch.no_grad():
            logits = mdl(ids).logits
        lp = torch.log_softmax(logits[0, :-1], dim=-1)
        tgt = ids[0, 1:]
        return float(-lp[range(len(tgt)), tgt].mean())

    def __call__(self, row) -> str:
        segs = [(t, s) for t, sents in zip(row["context"]["title"],
                                           row["context"]["sentences"])
                for s in sents]
        if not segs:
            return ctx_full(row)
        scored = sorted(range(len(segs)),
                        key=lambda i: -self._self_info(segs[i][1]))
        keep = sorted(scored[:max(1, round(len(segs) * self.rate))])
        return "\n".join(f"{segs[i][0]}: {segs[i][1]}" for i in keep)


def _ctx_docs(row) -> list[str]:
    """Ngữ cảnh dưới dạng DANH SÁCH đoạn — LongLLMLingua cần vậy để xếp hạng."""
    return [f"{t}: {' '.join(s)}" for t, s
            in zip(row["context"]["title"], row["context"]["sentences"])]



class RecompExtractiveMethod:
    """RECOMP-extractive (Xu et al., 2024) — chọn câu bằng bộ mã hoá ĐÃ HUẤN LUYỆN.

    Khác IterCOMP ở chỗ then chốt: bộ mã hoá này được huấn luyện có giám sát để
    dự đoán câu nào giúp trả lời đúng, thay vì dùng độ tương đồng chung chung.
    Nên nó là đường cơ sở kiểm tra chính xác một câu hỏi: **huấn luyện có ăn
    đứt training-free không?**

    Checkpoint huấn luyện trên NQ (một-hop) vì tác giả không phát hành bản
    extractive cho HotpotQA. Đó là lệch phân bố, và báo cáo phải ghi rõ — nó
    làm baseline này yếu đi trên dữ liệu multi-hop.
    """

    MODEL = "fangyuan/nq_extractive_compressor"

    def __init__(self, rate: float = 0.33, model: str | None = None):
        self.rate, self.model = rate, model or self.MODEL
        self._m = None

    def _encoder(self):
        if self._m is None:
            import torch
            from sentence_transformers import SentenceTransformer
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._m = SentenceTransformer(self.model, device=dev)
        return self._m

    def __call__(self, row) -> str:
        segs = [(t, x) for t, sents in zip(row["context"]["title"],
                                           row["context"]["sentences"])
                for x in sents]
        if not segs:
            return ctx_full(row)
        enc = self._encoder()
        import numpy as np
        qv = enc.encode([row["question"]], normalize_embeddings=True)
        sv = enc.encode([x for _, x in segs], normalize_embeddings=True)
        order = np.argsort(-(sv @ qv[0]))
        keep = sorted(order[:max(1, round(len(segs) * self.rate))])
        return "\n".join(f"{segs[i][0]}: {segs[i][1]}" for i in keep)


class CompActMethod:
    """CompAct (Yoon et al., EMNLP 2024) — nén CHỦ ĐỘNG theo vòng, có bộ dừng.

    Baseline gần IterCOMP nhất về ý tưởng: cả hai đều lặp và đều tự phán đoán
    khi nào đủ bằng chứng. Khác ở chỗ CompAct **sinh ra** bản tóm tắt (abstractive)
    còn IterCOMP **chọn câu** (extractive), và CompAct duyệt tài liệu theo lô
    thay vì chấm điểm lại toàn bộ mỗi vòng.

    PROMPT LẤY NGUYÊN VĂN từ `dmis-lab/CompAct/utils.py::create_prompt`, không
    phải em tự soạn. Điều này quan trọng: mục baseline của bài chỉ ra rằng năm
    trong sáu baseline bị handicap vì thành phần tiếng Anh, nên thêm một baseline
    thứ sáu với prompt ĐOÁN sẽ tạo ra đúng loại so sánh không công bằng mà bài
    đang cảnh báo. Tham số cũng theo `scripts/run_prompt.sh` của họ:
    segment_size=5, max_iteration=6, greedy, max_new_tokens=900.

    Model card trên HuggingFace là template tự sinh, không có tài liệu gì — chat
    template (`[INST] ... [/INST]`, Mistral-7B) và prompt đều phải truy từ repo.
    """

    MODEL = "cwyoon99/CompAct-7b"
    SEGMENT_SIZE = 5
    MAX_ITER = 6

    _INSTR_FIRST = (
        "1. Generate a summary of source documents to answer the question. "
        "Ensure the summary is under 200 words and does not include any "
        "pronouns. DO NOT make assumptions or attempt to answer the question; "
        "your job is to summarize only.\n\n2. Evaluate the summary based solely "
        "on the information of it, without any additional background context: "
        "if it lacks sufficient details to answer the question, print "
        "'[INCOMPLETE]'. If it provides all necessary details, print "
        "'[COMPLETE]'. You should provide the reason of evalution.")
    _INSTR_NEXT = (
        "1. Generate a summary of the previous summary and the source documents "
        "to answer the question based on the evaluation of the previous summary. "
        "The evaluation indicates the missing information needed to answer the "
        "question. Ensure the summary is under 200 words and does not include "
        "any pronouns. DO NOT make assumptions or attempt to answer the "
        "question; your job is to summarize only.\n\n2. Evaluate the summary "
        "based solely on the information of it, without any additional "
        "background context: if it lacks sufficient details to answer the "
        "question, print '[INCOMPLETE]'. If it provides all necessary details, "
        "print '[COMPLETE]'. You should provide the reason of evalution.")

    def __init__(self, model: str | None = None, load_4bit: bool = True):
        self.model, self.load_4bit = model or self.MODEL, load_4bit
        self._m = None

    def _model(self):
        if self._m is None:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            kw = {}
            if self.load_4bit and torch.cuda.is_available():
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4")
                kw["device_map"] = "auto"
            tok = AutoTokenizer.from_pretrained(self.model)
            mdl = AutoModelForCausalLM.from_pretrained(self.model, **kw).eval()
            self._m = (tok, mdl, torch)
        return self._m

    @staticmethod
    def _parse(text: str) -> tuple[str, str]:
        """Tách summary và eval — theo `parse_output_without_sentence` của họ."""
        m = re.search(r"(Summary:)(.*?)(?=Evaluation:|$)", text, re.DOTALL)
        summary = m.group(2).strip() if m else text.split("Evaluation:")[0].strip()
        e = re.search(r"(Evaluation:)(.*?)(?=Summary:|$)", text, re.DOTALL)
        return summary, (e.group(2).strip() if e else "")

    def __call__(self, row) -> str:
        tok, mdl, torch = self._model()
        docs = [f"{t} {' '.join(s)}" for t, s
                in zip(row["context"]["title"], row["context"]["sentences"])]
        summary, ev = "", ""
        for it in range(self.MAX_ITER):
            seg = docs[it * self.SEGMENT_SIZE:(it + 1) * self.SEGMENT_SIZE]
            if not seg:
                break
            doc_in = " ".join(seg)
            if it == 0:
                prompt = (f"{self._INSTR_FIRST}\n\nQuestion: {row['question']}"
                          f"\n\nSource documents: {doc_in}\n\nSummary:")
            else:
                prompt = (f"{self._INSTR_NEXT}\n\nQuestion: {row['question']}"
                          f"\n\nPrevious summary: {summary}"
                          f"\n\nEvaluation of previous summary: {ev}"
                          f"\n\nSource documents: {doc_in}\n\nSummary:")
            text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           tokenize=False,
                                           add_generation_prompt=False)
            ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=8192).to(mdl.device)
            with torch.no_grad():
                out = mdl.generate(**ids, max_new_tokens=900, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
            gen = tok.decode(out[0][ids["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            summary, ev = self._parse(gen)
            if "[COMPLETE]" in ev and "[INCOMPLETE]" not in ev:
                break
            ev = ev.replace("[INCOMPLETE]", "").strip()
        return summary


class RecompAbstractiveMethod:
    """RECOMP-abstractive — T5 SINH ra bản tóm tắt thay vì chọn câu.

    Đây là họ thứ ba, khác hẳn hai họ kia: nó không giữ lại chữ nào của ngữ
    cảnh gốc mà viết lại. Đưa vào vì nó cho thấy giới hạn của việc so bằng
    "tỉ lệ giữ" — một bản tóm tắt sinh ra không có tỉ lệ giữ theo nghĩa đó.

    Dùng checkpoint HotpotQA: đúng miền multi-hop, nên baseline này KHÔNG bị
    lệch phân bố như bản extractive.
    """

    MODEL = "fangyuan/hotpotqa_abstractive"

    def __init__(self, model: str | None = None, max_new_tokens: int = 128):
        self.model, self.max_new_tokens = model or self.MODEL, max_new_tokens
        self._m = None

    def _model(self):
        if self._m is None:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            tok = AutoTokenizer.from_pretrained(self.model)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(self.model).to(dev).eval()
            self._m = (tok, mdl, dev, torch)
        return self._m

    def __call__(self, row) -> str:
        tok, mdl, dev, torch = self._model()
        prompt = f"Question: {row['question']}\n Document: {ctx_full(row)}\n Summary: "
        ids = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=1024).input_ids.to(dev)
        with torch.no_grad():
            out = mdl.generate(ids, max_new_tokens=self.max_new_tokens)
        return tok.decode(out[0], skip_special_tokens=True)


class IterCompMethod:
    def __init__(self, llm_kind: str, max_iter: int = 5, scorer_kind: str = "dual",
                 encoder: str = "BAAI/bge-m3", lam: float = DEFAULT_LAMBDA,
                 percentile: float = DEFAULT_PERCENTILE, top_k: int | None = None,
                 llm_model: str | None = None, llm_4bit: bool = False,
                 adaptive_k: bool = False, k_min: float = AP_K_MIN,
                 k_max: float = AP_K_MAX):
        lkw = {}
        if llm_kind == "hf" and llm_model:
            lkw = {"model": llm_model, "load_4bit": llm_4bit}
        self.llm = make_llm(llm_kind, **lkw)
        self.scorer = make_scorer(scorer_kind, encoder=encoder, lam=lam)
        self.max_iter, self.percentile, self.top_k = max_iter, percentile, top_k
        self.adaptive_k, self.k_min, self.k_max = adaptive_k, k_min, k_max
        #: k thực tế mỗi câu hỏi, để báo cáo AP có thật sự biến thiên hay không
        self.k_log: list[float] = []

    def __call__(self, row) -> str:
        res = itercomp(self.llm, row["question"], row["context"],
                       max_iter=self.max_iter, scorer=self.scorer,
                       percentile=self.percentile, top_k_per_iter=self.top_k,
                       adaptive_k=self.adaptive_k,
                       k_min=self.k_min, k_max=self.k_max)
        self.k_log.extend(res.k_per_iter)
        return res.to_prompt()


# ══════════════════════════════════════════════════════════════ metric


# ══════════════════════════════════════════════════════════════ main


def main():
    ap = argparse.ArgumentParser(description="So sánh các phương pháp nén prompt")
    ap.add_argument("--dataset", default="vimqa",
                    choices=["vimqa", "hotpotqa", "2wiki", "musique", "vihotpot"])
    ap.add_argument("--limit", type=int, default=None,
                    help="bỏ trống = chạy TOÀN BỘ tập (như paper)")
    # Không có giá trị mặc định: 'mock' trả đáp án giả nên vẫn ra bảng số trông
    # hợp lệ mà chẳng đo gì. Bắt buộc chọn để không lỡ đo mock rồi đem so bài báo.
    ap.add_argument("--reader", required=True,
                    choices=["mock", "openai", "openrouter", "gemini", "hf"])
    ap.add_argument("--reader-model", default=None)
    ap.add_argument("--load-4bit", action="store_true",
                    help="nạp mô hình đọc ở 4-bit; cần cho 7-8B trên T4 15GB")
    ap.add_argument("--methods", default="raw,oracle,llmlingua2,itercomp",
                    help="raw|none,oracle,llmlingua2,llmlingua,longllmlingua,selective-context,recomp-extractive,recomp-abstractive,compact,itercomp (phân cách bằng dấu phẩy)")
    # Cũng không có mặc định. 'mock' luôn báo "đủ bằng chứng" ngay vòng 1, nên
    # vòng lặp KHÔNG chạy và IterCOMP suy thoái thành lọc percentile một lần —
    # đúng lỗi đã làm kết quả 7B thấp hơn bài báo rất nhiều.
    ap.add_argument("--itercomp-llm", required=True,
                    choices=["mock", "hf", "openai", "openrouter", "gemini"],
                    help="hf = dùng chính mô hình đọc cho các bước suy luận, như bài báo")
    ap.add_argument("--rate", type=float, default=0.33, help="tỉ lệ giữ token cho llmlingua2")
    ap.add_argument("--max-iter", type=int, default=5, help="paper Mục 5.1: 5")
    ap.add_argument("--scorer", default="dual", choices=["bm25", "dense", "lexical", "dual"])
    ap.add_argument("--encoder", default="BAAI/bge-m3")
    ap.add_argument("--lam", type=float, default=DEFAULT_LAMBDA, help="paper: 0.6")
    ap.add_argument("--percentile", type=float, default=None,
                    help="mặc định theo dataset: 90 (MuSiQue/HotpotQA), 85 (2Wiki)")
    ap.add_argument("--top-k", type=int, default=None, help="ABLATION: top-k thay percentile")
    ap.add_argument("--adaptive-k", action="store_true",
                    help="CẢI TIẾN: suy k từ độ tập trung điểm mỗi vòng thay vì "
                         "hằng số. Đặt --k-min = --k-max để suy biến về paper gốc.")
    ap.add_argument("--k-min", type=float, default=AP_K_MIN)
    ap.add_argument("--k-max", type=float, default=AP_K_MAX)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.itercomp_llm == "hf" and not args.reader_model:
        ap.error("--itercomp-llm hf cần --reader-model để biết nạp mô hình nào")
    if args.reader == "hf" and not args.reader_model:
        ap.error("--reader hf cần --reader-model")

    pct = args.percentile if args.percentile is not None else \
        PERCENTILE_BY_DATASET.get(args.dataset, DEFAULT_PERCENTILE)
    kw = {}
    if args.reader_model and args.reader != "mock":
        kw["model"] = args.reader_model
    if args.load_4bit and args.reader == "hf":
        kw["load_4bit"] = True
    reader = make_reader(args.reader, **kw)
    rows = load_dataset(args.dataset, args.limit)

    methods = {}
    for m in args.methods.split(","):
        m = m.strip()
        if m in ("none", "raw"):        # "raw" = tên trong paper
            methods[m] = ctx_full
        elif m == "oracle":
            methods[m] = ctx_oracle
        elif m == "llmlingua2":
            methods[m] = LLMLingua2Method(rate=args.rate)
        elif m == "llmlingua":
            methods[m] = LLMLinguaMethod(rate=args.rate)
        elif m == "longllmlingua":
            methods[m] = LongLLMLinguaMethod(rate=args.rate)
        elif m == "selective-context":
            methods[m] = SelectiveContextMethod(rate=args.rate)
        elif m == "recomp-extractive":
            methods[m] = RecompExtractiveMethod(rate=args.rate)
        elif m == "compact":
            methods[m] = CompActMethod(load_4bit=args.load_4bit)
        elif m == "recomp-abstractive":
            methods[m] = RecompAbstractiveMethod()
        elif m == "itercomp":
            methods[m] = IterCompMethod(
                args.itercomp_llm, args.max_iter, args.scorer,
                args.encoder, args.lam, pct, args.top_k,
                llm_model=args.reader_model, llm_4bit=args.load_4bit,
                adaptive_k=args.adaptive_k, k_min=args.k_min, k_max=args.k_max)
        else:
            raise ValueError(f"method không biết: {m}")

    print(f"dataset={args.dataset}  n={len(rows)}  reader={args.reader}"
          + (f" ({args.reader_model})" if args.reader_model else "")
          + (" 4-bit" if args.load_4bit else ""))
    print(f"methods={list(methods)}")
    if "itercomp" in methods:
        # In ra để mỗi log tự nói nó đã đo cấu hình nào; thiếu dòng này thì
        # không phân biệt được lần chạy mock với lần chạy thật.
        print(f"itercomp: llm={args.itercomp_llm}  scorer={args.scorer}"
              f"  lam={args.lam}  percentile={pct}  max_iter={args.max_iter}"
              + (f"  top_k={args.top_k} (ABLATION)" if args.top_k else ""))
        if args.itercomp_llm == "mock":
            print("  CẢNH BÁO: llm=mock -> vòng lặp dừng ngay vòng 1,"
                  " KHÔNG so được với bài báo")
    print()

    agg = defaultdict(lambda: {"em": [], "f1": [], "em_norm": [], "f1_norm": [],
                               "tok": [], "orig": [], "sec": 0.0})
    per_row = []

    # ── checkpoint theo từng câu ──────────────────────────────────────────
    # Một phiên Colab có thể đứt bất cứ lúc nào (đóng máy, hết hạn, mất mạng).
    # Không có checkpoint thì bộ đang chạy dở mất trắng, dù đã tốn hàng giờ.
    # Ghi sau MỖI câu, và khi chạy lại thì bỏ qua đúng những câu đã xong.
    ckpt = args.out.with_suffix(".partial.json") if args.out else None
    # Tạo thư mục đích NGAY, không đợi tới lúc ghi kết quả cuối: checkpoint
    # đầu tiên được ghi sau câu 1, nên thư mục phải có sẵn trước vòng lặp.
    # Bản clone/giải nén sạch không có sẵn results/.
    if ckpt:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if ckpt and ckpt.exists():
        try:
            done = {r["question"]: r for r in json.load(open(ckpt, encoding="utf-8"))}
            print(f"tiếp tục từ checkpoint: đã có {len(done)}/{len(rows)} câu")
        except Exception as e:
            print(f"checkpoint hỏng, chạy lại từ đầu ({e})")

    def _replay(rec):
        """Đưa một câu đã tính từ checkpoint vào bảng tổng hợp."""
        for name, m in rec["methods"].items():
            a = agg[name]
            a["em"].append(m["em"]); a["f1"].append(m["f1"])
            a["em_norm"].append(m["em_norm"]); a["f1_norm"].append(m["f1_norm"])
            a["tok"].append(m["tokens"]); a["orig"].append(rec["orig_tokens"])
        per_row.append(rec)

    for i, row in enumerate(rows, 1):
        if row["question"] in done:
            _replay(done[row["question"]])
            continue
        n_orig = len(tokenize(ctx_full(row)))
        rec = {"question": row["question"], "gold": row["answer"],
               "type": row["type"], "orig_tokens": n_orig, "methods": {}}

        for name, fn in methods.items():
            t0 = time.time()
            ctx = fn(row)
            pred = clean_answer(reader(ctx, row["question"]))
            dt = time.time() - t0

            gold = row["answer"]
            em, f1 = exact_match(pred, gold), f1_score(pred, gold)
            # CẢI TIẾN CHO TIẾNG VIỆT: chuẩn hoá đáp án đúng/sai trước khi so.
            # Báo cáo cả hai để thấy rõ phần nào là lỗi thật, phần nào là lỗi định dạng.
            pred_n = normalize_boolean(pred, gold)
            em_n, f1_n = exact_match(pred_n, gold), f1_score(pred_n, gold)
            n_tok = len(tokenize(ctx))

            a = agg[name]
            a["em"].append(em); a["f1"].append(f1)
            a["em_norm"].append(em_n); a["f1_norm"].append(f1_n)
            a["tok"].append(n_tok); a["orig"].append(n_orig); a["sec"] += dt

            rec["methods"][name] = {"pred": pred, "em": em, "f1": f1,
                                    "pred_norm": pred_n, "em_norm": em_n,
                                    "f1_norm": f1_n, "tokens": n_tok}

        per_row.append(rec)
        if ckpt:
            # Ghi qua file tạm rồi đổi tên: nếu bị giết đúng lúc đang ghi thì
            # checkpoint cũ vẫn nguyên vẹn thay vì thành file JSON cụt.
            tmp = ckpt.with_suffix(".tmp")
            json.dump(per_row, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, default=str)
            tmp.replace(ckpt)
        best = max(rec["methods"].items(), key=lambda x: x[1]["f1"])
        print(f"[{i}/{len(rows)}] {row['question'][:55]:55s} gold={row['answer'][:22]:22s} "
              f"best={best[0]}(F1 {best[1]['f1']:.2f})")

    # ─────────────────────────────────────────────────────── bảng kết quả
    print(f"\n{'='*78}")
    print(f"KẾT QUẢ — {args.dataset} (n={len(rows)}, reader={args.reader})")
    print(f"{'='*78}")
    print(f"{'method':<14}{'EM':>7}{'F1':>8}"
          f"{'EM*':>7}{'F1*':>8}{'tokens':>8}{'ratio':>7}{'s/q':>6}")
    print("  (* = sau khi chuẩn hoá đáp án đúng/sai cho tiếng Việt)")
    print("-" * 78)
    table = {}
    for name, a in agg.items():
        n = len(a["em"])
        em = 100 * sum(a["em"]) / n
        f1 = 100 * sum(a["f1"]) / n
        tok = sum(a["tok"]) / n
        ratio = sum(a["tok"]) / sum(a["orig"])
        sec = a["sec"] / n
        em_n = 100 * sum(a["em_norm"]) / n
        f1_n = 100 * sum(a["f1_norm"]) / n
        table[name] = {"em": em, "f1": f1, "em_norm": em_n, "f1_norm": f1_n,
                       "avg_tokens": tok, "ratio": ratio, "sec_per_q": sec}
        print(f"{name:<14}{em:>7.1f}{f1:>8.1f}{em_n:>7.1f}{f1_n:>8.1f}"
              f"{tok:>8.0f}{ratio:>7.1%}{sec:>6.2f}")
    print("-" * 78)

    # ── khoảng tin cậy bootstrap: với n nhỏ, chênh vài điểm F1 là nhiễu
    print(f"\nKhoảng tin cậy 95% (bootstrap, 10k lần lấy mẫu lại):")
    print(f"{'method':<14}{'F1*':>8}{'KTC 95%':>18}")
    for name, a in agg.items():
        f1s = [x * 100 for x in a["f1_norm"]]
        ci = bootstrap_ci(f1s)
        print(f"{name:<14}{ci['mean']:>8.1f}  [{ci['lo']:>5.1f}, {ci['hi']:>5.1f}]")
    ref = list(agg)[-1]
    mdd = min_detectable_diff([x * 100 for x in agg[ref]["f1_norm"]])
    print(f"\nVới n={len(per_row)}, chênh nhỏ nhất phát hiện được ~{mdd:.1f} điểm F1."
          f"\nChênh nhỏ hơn mức này KHÔNG kết luận được.")

    # ── so sánh có ghép cặp giữa itercomp và các đường cơ sở
    if "itercomp" in agg and len(agg) > 1:
        print(f"\nSo sánh ghép cặp (cùng câu hỏi, khác phương pháp):")
        ic = [x * 100 for x in agg["itercomp"]["f1_norm"]]
        for name, a in agg.items():
            if name == "itercomp":
                continue
            t = paired_bootstrap(ic, [x * 100 for x in a["f1_norm"]])
            mark = "có ý nghĩa" if t["p"] < 0.05 else "chưa đủ bằng chứng"
            print(f"  itercomp − {name:<12}{t['diff']:>+7.1f}  "
                  f"[{t['lo']:>+6.1f},{t['hi']:>+6.1f}]  p={t['p']:.4f}  {mark}")

        print(f"\nTác động của chuẩn hoá đáp án đúng/sai (ghép cặp):")
        for name, a in agg.items():
            t = paired_bootstrap([x * 100 for x in a["f1_norm"]],
                                 [x * 100 for x in a["f1"]])
            mark = "có ý nghĩa" if t["p"] < 0.05 else "chưa đủ bằng chứng"
            print(f"  {name:<14}{t['diff']:>+7.1f}  "
                  f"[{t['lo']:>+6.1f},{t['hi']:>+6.1f}]  p={t['p']:.4f}  {mark}")

    if hasattr(reader, "cost_usd"):
        print(f"chi phí reader: ${reader.cost_usd():.4f} ({reader.calls} calls)")

    # phân tích theo type (bridge/comparison) — trục phân tích cho báo cáo
    types = {r["type"] for r in per_row if r["type"]}
    if len(types) > 1:
        print(f"\nF1 theo type:")
        print(f"{'method':<14}" + "".join(f"{t:>14}" for t in sorted(types)))
        for name in agg:
            line = f"{name:<14}"
            for t in sorted(types):
                v = [r["methods"][name]["f1"] for r in per_row if r["type"] == t]
                line += f"{100*sum(v)/len(v):>13.1f} " if v else f"{'—':>14}"
            print(line)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"config": vars(args) | {"out": str(args.out)},
                   "summary": table, "per_row": per_row},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
        print(f"\n→ ghi {args.out}")
        # Xong hẳn thì bỏ checkpoint, để lần chạy sau không tưởng là còn dở.
        if ckpt and ckpt.exists():
            ckpt.unlink()


if __name__ == "__main__":
    main()
