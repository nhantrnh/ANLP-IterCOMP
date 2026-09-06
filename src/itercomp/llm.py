"""Các backend mô hình ngôn ngữ cho hai bước cần LLM của IterCOMP.

Mọi backend tuân theo Protocol `LLM`, nên đổi nhà cung cấp không phải sửa
logic thuật toán.

    mock        không gọi mạng, chỉ để kiểm thử luồng
    openai      gpt-4o-mini
    openrouter  model :free, tự xoay vòng khi một key hết hạn
    gemini      free tier

`CachedLLM` bọc bất kỳ backend nào và ghi cache ra đĩa — bắt buộc với free tier
vì quota chỉ 50 request/ngày."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

# .env: tìm ở gốc repo trước, rồi mới tới thư mục cha.
#
# parents[2] ĐÃ là gốc repo (llm.py nằm ở src/itercomp/), nên bản trước thêm
# .parent vào đó khiến nó chỉ tìm BÊN NGOÀI repo — hoạt động trên máy đặt .env
# ở đấy, và im lặng bỏ qua .env đặt đúng chỗ theo README. Thử cả hai, ưu tiên
# trong repo, và không ghi đè biến môi trường đã có.
try:
    from dotenv import load_dotenv
    _ROOT = Path(__file__).resolve().parents[2]      # .../<repo>
    for _cand in (_ROOT / ".env", _ROOT.parent / ".env"):
        if _cand.is_file():
            load_dotenv(_cand, override=False)
except ImportError:
    pass


class LLM(Protocol):
    def __call__(self, prompt: str, max_tokens: int = 256) -> str: ...


class MockLLM:
    """Backend giả — để test logic vòng lặp mà không tốn tiền API.

    Trả lời theo heuristic đơn giản. KHÔNG dùng để lấy số liệu báo cáo.
    """

    def __init__(self):
        self.calls = 0

    def __call__(self, prompt: str, max_tokens: int = 256) -> str:
        self.calls += 1
        if "ANSWERABLE" in prompt:
            # Giả định: trả lời được sau khi đã gom >=2 segment
            return "YES" if prompt.count("[SEG") >= 2 else "NO"
        if "FOLLOW-UP" in prompt:
            return "Thông tin còn thiếu là gì?"
        return "unknown"


class OpenAILLM:
    """GPT-4o-mini. Paper dùng GPT-4o; mini rẻ hơn ~15x, khai báo rõ trong báo cáo."""

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("thiếu OPENAI_API_KEY (đặt trong .env hoặc export)")
        self.client = OpenAI(timeout=60.0, max_retries=3)
        self.model = model
        self.calls = 0
        self.in_tokens = 0
        self.out_tokens = 0

    def __call__(self, prompt: str, max_tokens: int = 256) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        self.calls += 1
        if r.usage:
            self.in_tokens += r.usage.prompt_tokens
            self.out_tokens += r.usage.completion_tokens
        return (r.choices[0].message.content or "").strip()

    def cost_usd(self) -> float:
        # giá gpt-4o-mini: $0.15/1M input, $0.60/1M output
        return self.in_tokens / 1e6 * 0.15 + self.out_tokens / 1e6 * 0.60


class OpenRouterLLM:
    """OpenRouter — có nhiều model free (:free suffix). Dùng OPENROUTER_API_KEY từ .env.

    Model free đã TEST SỐNG (2026-08), trả lời tiếng Việt đúng:
        openai/gpt-oss-20b:free            (đã test ✓, ctx 131k)
        inclusionai/ling-3.0-flash:free    (đã test ✓, ctx 262k)
        poolside/laguna-xs-2.1:free        (đã test ✓, ctx 262k)

    ⚠️ Đa số model free hiện nay là REASONING MODEL: chúng tiêu token cho phần
    suy luận nội bộ trước khi trả lời. Với max_tokens quá nhỏ (vd 64) chúng trả
    về CHUỖI RỖNG. Vì vậy mặc định max_tokens ở đây là 512, không phải 64.
    """

    def __init__(self, model: str = "openai/gpt-oss-20b:free"):
        from openai import OpenAI

        # Thử lần lượt các key trong .env, BỎ QUA key đã hết hạn (401).
        # Cần thiết vì .env có nhiều key và không phải key nào cũng còn sống.
        import urllib.request
        key = None
        for name in ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY01",
                     "OPENROUTER_API_KEY_OLD"):
            cand = os.getenv(name)
            if not cand:
                continue
            try:
                req = urllib.request.Request(
                    "https://openrouter.ai/api/v1/key",
                    headers={"Authorization": f"Bearer {cand}"})
                with urllib.request.urlopen(req, timeout=15):
                    key = cand
                    self.key_name = name
                    break
            except Exception:
                continue      # key lỗi -> thử key kế tiếp
        if not key:
            raise RuntimeError(
                "không có OPENROUTER_API_KEY nào còn hiệu lực trong .env")
        # timeout BẮT BUỘC: free tier hay treo kết nối vô hạn nếu không đặt.
        self.client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1",
                             timeout=60.0, max_retries=3)
        self.model = model
        self.calls = self.in_tokens = self.out_tokens = 0

    def __call__(self, prompt: str, max_tokens: int = 512) -> str:
        # reasoning model tiêu token nội bộ -> không hạ dưới 256
        max_tokens = max(max_tokens, 256)
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        self.calls += 1
        if r.usage:
            self.in_tokens += r.usage.prompt_tokens
            self.out_tokens += r.usage.completion_tokens
        return (r.choices[0].message.content or "").strip()

    def cost_usd(self) -> float:
        return 0.0 if ":free" in self.model else float("nan")


class GeminiLLM:
    """Gemini API trực tiếp. Free tier: 15 req/phút với gemini-2.0-flash."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        key = next((os.getenv(k) for k in
                    ("GEMINI_API_KEY", "GEMINI_API_KEY_01", "GOOGLE_API_KEY")
                    if os.getenv(k)), None)
        if not key:
            raise RuntimeError("thiếu GEMINI_API_KEY trong .env")
        # SDK Gemini dùng giao diện tương thích OpenAI
        from openai import OpenAI
        self.client = OpenAI(
            api_key=key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=60.0, max_retries=3,
        )
        self.model = model
        self.calls = self.in_tokens = self.out_tokens = 0

    def __call__(self, prompt: str, max_tokens: int = 256) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        self.calls += 1
        if r.usage:
            self.in_tokens += r.usage.prompt_tokens
            self.out_tokens += r.usage.completion_tokens
        return (r.choices[0].message.content or "").strip()

    def cost_usd(self) -> float:
        return 0.0   # free tier


class CachedLLM:
    """Bọc một LLM backend, cache kết quả ra đĩa theo hash(prompt).

    Cực quan trọng với free tier: quota 50 req/ngày, nên KHÔNG được gọi lại
    cùng một prompt. Cache giúp chạy tiếp từ chỗ dừng sau khi hết quota.
    """

    def __init__(self, inner, path: str = "cache/llm_cache.json"):
        import json as _json
        self.inner = inner
        self.path = Path(__file__).parent / path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._c = _json.loads(self.path.read_text())
        except Exception:
            self._c = {}
        self.hits = 0

    @property
    def calls(self) -> int:
        return getattr(self.inner, "calls", 0)

    def cost_usd(self) -> float:
        return self.inner.cost_usd() if hasattr(self.inner, "cost_usd") else 0.0

    def _key(self, prompt: str, max_tokens: int) -> str:
        import hashlib
        model = getattr(self.inner, "model", type(self.inner).__name__)
        return hashlib.sha256(f"{model}|{max_tokens}|{prompt}".encode()).hexdigest()

    def flush(self):
        import json as _json
        self.path.write_text(_json.dumps(self._c, ensure_ascii=False))

    def __call__(self, prompt: str, max_tokens: int = 256) -> str:
        k = self._key(prompt, max_tokens)
        if k in self._c:
            self.hits += 1
            return self._c[k]
        out = self.inner(prompt, max_tokens)
        self._c[k] = out
        # Ghi ngay để mất quota giữa đường vẫn giữ được kết quả. flush() viết
        # lại cả file nên với backend API (chậm hàng trăm ms mỗi lần gọi) chi
        # phí này không đáng kể; backend cục bộ thì không đi qua đây.
        self.flush()
        return out


class HFLocalLLM:
    """Mô hình ngôn ngữ cục bộ cho hai bước cần LLM của IterCOMP.

    Bài báo dùng CÙNG một mô hình (LLaMA-3-8B) cho cả việc đọc và hai bước suy
    luận. Trước đây backend duy nhất chạy được cục bộ là `mock`, nên phần
    "nhận thức suy luận" bị vô hiệu: `is_answerable` luôn trả lời được ngay ở
    vòng đầu và `make_followup` trả về một câu cố định. Backend này khôi phục
    đúng thiết lập của bài báo.

    Dùng lại chính instance của HFReader nếu đã nạp, để không giữ hai bản model
    trong VRAM — trên T4 15GB thì hai bản 7B sẽ tràn bộ nhớ.
    """

    _shared: dict[str, object] = {}

    def __init__(self, model: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 load_4bit: bool = False):
        key = f"{model}|{load_4bit}"
        if key not in HFLocalLLM._shared:
            from .reader import HFReader
            HFLocalLLM._shared[key] = HFReader(model=model, load_4bit=load_4bit)
        self._r = HFLocalLLM._shared[key]
        self.model = model
        self.calls = 0

    def __call__(self, prompt: str, max_tokens: int = 256) -> str:
        # raw() sinh trực tiếp từ prompt, không bọc thêm câu lệnh đọc hiểu.
        # Giới hạn token đầu ra thấp vì hai bước này chỉ cần một từ hoặc một câu.
        self.calls += 1
        return self._r.raw(prompt, max_new_tokens=min(max_tokens, 64))

    def choose(self, prompt: str, options: list[str]) -> str:
        """Chuyển tiếp xuống reader nếu nó hỗ trợ chọn bằng một lượt forward."""
        self.calls += 1
        return self._r.choose(prompt, options)

    def choose_prob(self, prompt: str, pos: str, neg: str) -> float:
        """Chuyển tiếp xuống reader; xem HFReader.choose_prob."""
        self.calls += 1
        return self._r.choose_prob(prompt, pos, neg)

    def top2_margin(self, prompt: str) -> float:
        """Chuyển tiếp; xem HFReader.top2_margin (tín hiệu của TASR)."""
        self.calls += 1
        return self._r.top2_margin(prompt)

    def cost_usd(self) -> float:
        return 0.0


LLM_BACKENDS = {
    "mock": MockLLM,
    "openai": OpenAILLM,
    "openrouter": OpenRouterLLM,
    "gemini": GeminiLLM,
    "hf": HFLocalLLM,
}


#: Backend chạy cục bộ: không có quota để tiết kiệm nên cache chỉ gây hại.
LOCAL_BACKENDS = frozenset({"mock", "hf"})


def make_llm(kind: str, cache: bool = True, **kw) -> LLM:
    """cache=True bọc CachedLLM — nên bật với free tier để không đốt quota lặp lại.

    KHÔNG cache backend cục bộ: không có quota để tiết kiệm, mà CachedLLM ghi
    lại toàn bộ file JSON sau mỗi lần gọi. Với mô hình cục bộ, mỗi câu hỏi gọi
    LLM hơn 10 lần (5 vòng × kiểm tra đủ bằng chứng + sinh câu hỏi tiếp), nên
    chi phí ghi tăng bậc hai — khoảng 0,5 GB ghi đĩa cho một lần chạy 4 bộ dữ
    liệu, đổi lấy con số không.
    """
    inner = LLM_BACKENDS[kind](**kw)
    if cache and kind not in LOCAL_BACKENDS:
        return CachedLLM(inner)
    return inner
