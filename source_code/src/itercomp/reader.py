"""Reader — sinh đáp án từ prompt (đã nén hoặc chưa nén).

Tách riêng khỏi itercomp.py để có thể so sánh công bằng:
    cùng một reader, khác nhau chỉ ở cách nén prompt.

Backend:
    mock    — không tốn tiền, trả về heuristic (chỉ để test pipeline)
    openai  — gpt-4o-mini (paper dùng GPT-4o; khai báo rõ trong báo cáo)
    hf      — model local (Qwen2.5-1.5B/7B...) qua transformers
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Prompt đọc hiểu: yêu cầu trả lời NGẮN để EM/F1 có nghĩa.
# Song ngữ vì ta chạy cả dataset tiếng Anh và VimQA tiếng Việt.
READER_PROMPT = """Answer the question using ONLY the context below.
Reply with the SHORT answer only (a few words, no explanation, no full sentence).
If the context does not contain the answer, reply: unknown

Trả lời câu hỏi CHỈ dựa vào ngữ cảnh dưới đây.
Chỉ ghi đáp án NGẮN GỌN (vài từ, không giải thích, không viết thành câu).

CONTEXT:
{context}

QUESTION: {question}
ANSWER:"""


class MockReader:
    """Reader giả — KHÔNG dùng để lấy số liệu báo cáo."""

    def __init__(self):
        self.calls = 0

    def __call__(self, context: str, question: str) -> str:
        self.calls += 1
        return "unknown"


class OpenAIReader:
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("thiếu OPENAI_API_KEY")
        self.client = OpenAI()
        self.model = model
        self.calls = self.in_tokens = self.out_tokens = 0

    def __call__(self, context: str, question: str) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user",
                       "content": READER_PROMPT.format(context=context, question=question)}],
            max_tokens=64,
            temperature=0.0,
        )
        self.calls += 1
        if r.usage:
            self.in_tokens += r.usage.prompt_tokens
            self.out_tokens += r.usage.completion_tokens
        return (r.choices[0].message.content or "").strip()

    def cost_usd(self) -> float:
        return self.in_tokens / 1e6 * 0.15 + self.out_tokens / 1e6 * 0.60


class HFReader:
    """Model local — MIỄN PHÍ, không quota, không phụ thuộc mạng.

    Đây là cách lấy EM/F1 khi không có API key. Model 1.5B yếu hơn GPT-4o nên
    EM/F1 thấp hơn paper, nhưng XU HƯỚNG TƯƠNG ĐỐI giữa các phương pháp vẫn giữ
    — đó mới là điều cần chứng minh (xem Mục 3.3 của báo cáo).

    Chọn device: CUDA > MPS (Apple Silicon) > CPU.
    Trên T4 free dùng ≤1.5B; 7B cần 4-bit.
    """

    def __init__(self, model: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 load_4bit: bool = False, max_input_tokens: int = 3072):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.is_available():
            self.device, dtype = "cuda", torch.float16   # T4 không có bf16
        elif torch.backends.mps.is_available():
            self.device, dtype = "mps", torch.float16
        else:
            self.device, dtype = "cpu", torch.float32

        self.tok = AutoTokenizer.from_pretrained(model)
        if load_4bit:
            # Model lượng tử hoá 4-bit KHÔNG cho gọi .to(device) — transformers
            # sẽ raise ValueError. Phải giao việc đặt device cho device_map.
            from transformers import BitsAndBytesConfig
            # device_map="auto" tự chia model theo VRAM nó ƯỚC LƯỢNG còn trống,
            # và ước lượng đó chừa lại một phần dự phòng. Khi nó quyết đẩy vài
            # lớp sang CPU/disk thì bitsandbytes báo lỗi thẳng ("Some modules
            # are dispatched on the CPU or the disk"), vì 4-bit không offload
            # được. Qwen2.5-7B ở 4-bit chỉ ~5,5 GB nên thừa sức nằm trọn trên
            # T4 15 GB: ép hẳn lên GPU 0 thay vì để nó đoán.
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",      # nf4 chính xác hơn fp4 cùng cỡ
                bnb_4bit_use_double_quant=True,  # tiết kiệm thêm ~0,4 GB
            )
            if self.device == "cuda":
                free, total = torch.cuda.mem_get_info()
                need = 6.5e9   # 7B ở nf4 + double-quant ~5,5 GB, chừa chỗ KV-cache
                if free < need:
                    raise RuntimeError(
                        f"GPU chỉ còn {free/1e9:.1f} GB trống trên tổng "
                        f"{total/1e9:.1f} GB, cần ~{need/1e9:.1f} GB cho mô hình "
                        f"7B ở 4-bit.\n"
                        f"  - Nếu tổng < 12 GB: GPU này không đủ cho 7B. Đổi "
                        f"READER sang 'Qwen/Qwen2.5-1.5B-Instruct'.\n"
                        f"  - Nếu tổng đủ nhưng trống ít: Runtime → Restart "
                        f"session rồi chạy lại (phiên trước còn giữ VRAM).")
            dmap = {"": 0} if self.device == "cuda" else "auto"
            self.model = AutoModelForCausalLM.from_pretrained(
                model, quantization_config=qcfg, device_map=dmap,
            ).eval()
            self.device = str(next(self.model.parameters()).device)
        else:
            self.model = (AutoModelForCausalLM
                          .from_pretrained(model, torch_dtype=dtype)
                          .to(self.device).eval())
        self.max_input_tokens = max_input_tokens
        self.calls = 0

        # Đăng ký vào kho dùng chung của HFLocalLLM. Không có bước này thì
        # run_eval.py tạo HFReader ở đây, rồi IterCompMethod tạo HFLocalLLM và
        # nạp một HFReader THỨ HAI — hai bản 7B trong VRAM là hết T4 15 GB.
        from .llm import HFLocalLLM
        HFLocalLLM._shared.setdefault(f"{model}|{load_4bit}", self)

    def __call__(self, context: str, question: str) -> str:
        import torch

        msgs = [{"role": "user",
                 "content": READER_PROMPT.format(context=context, question=question)}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        # cắt bớt input quá dài (phương pháp "none" có thể vượt context window)
        ids = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=self.max_input_tokens).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=48, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        self.calls += 1
        return self.tok.decode(out[0][ids.input_ids.shape[1]:],
                               skip_special_tokens=True).strip()

    def raw(self, prompt: str, max_new_tokens: int = 48) -> str:
        """Sinh trực tiếp từ prompt, KHÔNG bọc thêm READER_PROMPT.

        Cần cho hai bước answerability và follow-up của IterCOMP, vì các bước đó
        đã có câu lệnh riêng; bọc thêm câu lệnh đọc hiểu vào sẽ làm mô hình trả
        lời sai việc.
        """
        import torch

        msgs = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=self.max_input_tokens).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=max_new_tokens,
                                      do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        self.calls += 1
        return self.tok.decode(out[0][ids.input_ids.shape[1]:],
                               skip_special_tokens=True).strip()

    def choose(self, prompt: str, options: list[str]) -> str:
        """Chọn 1 trong `options` bằng MỘT lượt forward, không sinh từng token.

        Bước kiểm tra đủ bằng chứng chỉ cần biết YES hay NO. Gọi generate() cho
        việc đó tốn một lượt forward cho MỖI token sinh ra, dù ta chỉ đọc chữ
        đầu tiên. So trực tiếp logit của token đầu mỗi lựa chọn cho cùng kết quả
        với 1/8 chi phí — và bước này chạy tới 5 lần mỗi câu hỏi.
        """
        import torch

        msgs = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=self.max_input_tokens).to(self.device)
        with torch.no_grad():
            logits = self.model(**ids).logits[0, -1]
        self.calls += 1

        best, best_score = options[0], float("-inf")
        for opt in options:
            # token đầu của mỗi lựa chọn; đủ để phân biệt vì chúng khác nhau
            # ngay ký tự đầu (YES/NO).
            tid = self.tok.encode(opt, add_special_tokens=False)[0]
            if logits[tid].item() > best_score:
                best, best_score = opt, logits[tid].item()
        return best

    def choose_prob(self, prompt: str, pos: str, neg: str) -> float:
        """Xác suất của `pos` so với `neg` ở token đầu, trong [0, 1].

        Softmax trên đúng hai logit thay vì cả bảng từ vựng: ta chỉ cần biết
        mô hình nghiêng về YES hay NO và nghiêng bao nhiêu. Đây là thứ biến
        quyết định dừng từ nhị phân thành có thể đặt ngưỡng.
        """
        import torch

        msgs = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=self.max_input_tokens).to(self.device)
        with torch.no_grad():
            logits = self.model(**ids).logits[0, -1].float()
        self.calls += 1

        def first_id(word: str) -> int:
            return self.tok.encode(word, add_special_tokens=False)[0]

        pair = torch.tensor([logits[first_id(pos)], logits[first_id(neg)]])
        return float(torch.softmax(pair, dim=0)[0])

    def top2_margin(self, prompt: str) -> float:
        """Biên log-prob giữa token khả dĩ nhất và nhì, tại vị trí kế tiếp.

        Đây là tín hiệu TASR (Agent4IR@KDD 2026) dùng làm điểm tin cậy: biên
        rộng nghĩa là mô hình cam kết một đáp án, biên hẹp nghĩa là đang do dự.
        Cài lại để so sánh, không phải đóng góp của chúng tôi.
        """
        import torch

        msgs = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=self.max_input_tokens).to(self.device)
        with torch.no_grad():
            logits = self.model(**ids).logits[0, -1].float()
        self.calls += 1
        top2 = torch.topk(torch.log_softmax(logits, dim=-1), k=2).values
        return float(top2[0] - top2[1])

    def cost_usd(self) -> float:
        return 0.0   # chạy local


class APIReader:
    """Reader dùng backend LLM bất kỳ trong itercomp.LLM_BACKENDS (gemini/openrouter/...).

    Tách riêng khỏi OpenAIReader để tái dùng các backend free tier.
    """

    def __init__(self, backend: str, cache: bool = True, **kw):
        from .llm import make_llm
        self.llm = make_llm(backend, cache=cache, **kw)

    def __call__(self, context: str, question: str) -> str:
        # 512 vì nhiều model free là reasoning model -> max_tokens nhỏ sẽ trả rỗng
        return self.llm(READER_PROMPT.format(context=context, question=question),
                        max_tokens=512)

    @property
    def calls(self) -> int:
        return getattr(self.llm, "calls", 0)

    def cost_usd(self) -> float:
        return self.llm.cost_usd() if hasattr(self.llm, "cost_usd") else 0.0


def make_reader(kind: str, **kw):
    if kind == "mock":
        return MockReader()
    if kind == "openai":
        return OpenAIReader(**kw)
    if kind == "hf":
        return HFReader(**kw)
    if kind in ("gemini", "openrouter"):
        return APIReader(kind, **kw)
    raise ValueError(kind)


def clean_answer(raw: str) -> str:
    """Gọt đáp án thô của LLM về dạng ngắn để so EM/F1.

    LLM hay trả 'ANSWER: x' hoặc 'Đáp án là x.' -> cắt phần thừa.
    """
    s = raw.strip()
    s = re.sub(r"^(answer|đáp án|trả lời)\s*[:：]\s*", "", s, flags=re.I)
    s = re.sub(r"^(the answer is|đáp án là)\s*", "", s, flags=re.I)
    s = s.split("\n")[0].strip()
    return s.rstrip(".").strip()
