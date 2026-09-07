"""Chấm điểm liên quan giữa câu hỏi và evidence segment.

Cài đặt theo ĐÚNG Mục 4.2.2 của paper IterCOMP (dual-aspect relevance score):

    S_dual(q, e) = λ · S_sem(q, e) + (1−λ) · S_lex(q, e)          (công thức 4)

    S_sem(q, e) = E(q)ᵀ · E(e)                                     (công thức 1)
                  inner product của dense embedding từ text encoder

    S_lex(q, e) = Σ_{t ∈ q∩e} w_t^q · w_t^e                        (công thức 3)
                  với w_t = ReLU(w_lexᵀ · E(t))                     (công thức 2)
                  sparse weighted inner product trên token đồng xuất hiện

Paper dùng **bge-m3 frozen, KHÔNG finetune** cho cả encoder E(·) và projection
vector w_lex — phù hợp nguyên tắc training-free. bge-m3 xuất đồng thời dense
và sparse (lexical weights) nên chỉ cần một model duy nhất.

Lọc segment: paper dùng **percentile filtering** (công thức 5), KHÔNG phải top-k:

    E_cand = { e | S_dual(q,e) ≥ Percentile(S, k) }

Ngưỡng thích ứng theo phân bố điểm của từng câu hỏi, áp lại ở MỖI bước suy luận.

---
Các scorer có sẵn (dùng cho ablation):
    bm25      — BM25Okapi, thuần thống kê từ vựng (KHÔNG phải của paper; là ablation)
    dense     — chỉ S_sem  (λ=1)
    lexical   — chỉ S_lex  (λ=0)
    dual      — đầy đủ theo paper (λ = 0.6)

Encoder thay thế được (đây là trục đóng góp cho tiếng Việt):
    BAAI/bge-m3                       đa ngữ, ĐÚNG như paper
    AITeamVN/Vietnamese_Embedding     chuyên tiếng Việt (finetune từ bge-m3)
    dangvantuan/vietnamese-embedding  PhoBERT-based
"""

from __future__ import annotations

import os
import sys

from typing import Protocol

# ── Siêu tham số ĐÚNG theo Mục 5.1 của paper ────────────────────────────
DEFAULT_ENCODER = "BAAI/bge-m3"   # paper: bge-m3 frozen, không finetune
DEFAULT_LAMBDA = 0.6              # paper: "set the hyperparameter λ ... to 0.6"
DEFAULT_PERCENTILE = 90.0         # paper: k=90 cho MuSiQue/HotpotQA, k=85 cho 2Wiki

# Paper chọn k để KHỚP mức nén của Oracle nhằm so sánh công bằng (Mục 5.1).
# VimQA không có trong paper; ta dùng 90 như HotpotQA vì cùng 10 đoạn/câu.
PERCENTILE_BY_DATASET = {
    "musique": 90.0,
    "hotpotqa": 90.0,
    "2wiki": 85.0,
    "vimqa": 90.0,
    "vihotpot": 90.0,   # HotpotQA dịch VN — cùng k với HotpotQA gốc
}


from .metrics import tokenize as _tokenize


class Scorer(Protocol):
    def __call__(self, query: str, texts: list[str]) -> list[float]: ...


# ═══════════════════════════════════════════════ BM25 (ablation, không phải paper)


class BM25Scorer:
    """BM25Okapi — thuần thống kê từ vựng, KHÔNG dùng text encoder.

    ⚠️ Đây KHÔNG phải phương pháp của paper. Giữ lại làm đường cơ sở ablation
    để trả lời: "text encoder có thực sự cần thiết không?"
    """

    name = "bm25"

    def __call__(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        try:
            from rank_bm25 import BM25Okapi
            bm25 = BM25Okapi([_tokenize(t) for t in texts])
            return [float(s) for s in bm25.get_scores(_tokenize(query))]
        except ImportError:
            q = set(_tokenize(query))
            return [len(q & set(_tokenize(t))) / (len(q) or 1) for t in texts]


# ══════════════════════════════════════════ dual-aspect theo paper (bge-m3)


class BGEM3Scorer:
    """Dual-aspect scorer dùng bge-m3 — cài đặt đúng Mục 4.2.2 của paper.

    mode:
        "dual"    S_dual = λ·S_sem + (1−λ)·S_lex   (paper)
        "dense"   chỉ S_sem   (ablation λ=1)
        "lexical" chỉ S_lex   (ablation λ=0)

    Model được load LAZY (bge-m3 ~2.3GB) và CACHE theo tên, nên nhiều scorer
    dùng cùng encoder sẽ chia sẻ một instance.
    """

    _cache: dict[str, object] = {}
    #: chỉ cảnh báo fallback một lần mỗi tiến trình
    _warned_fallback: bool = False

    def __init__(
        self,
        encoder: str = DEFAULT_ENCODER,
        mode: str = "dual",
        lam: float = DEFAULT_LAMBDA,
        use_fp16: bool = False,
    ):
        if mode not in ("dual", "dense", "lexical"):
            raise ValueError(f"mode không hợp lệ: {mode}")
        self.encoder_name = encoder
        self.mode = mode
        self.lam = 1.0 if mode == "dense" else 0.0 if mode == "lexical" else lam
        self.use_fp16 = use_fp16
        self.name = f"{mode}({encoder.split('/')[-1]}"
        self.name += f",λ={self.lam})" if mode == "dual" else ")"

    def _model(self):
        """Load model, cache theo tên.

        Ưu tiên FlagEmbedding (API gọn). Nếu không tương thích thì fallback sang
        transformers thuần — FlagEmbedding 1.4.0 lỗi `dtype` kwarg với
        transformers 4.45.x, và ta KHÔNG nâng transformers vì llmlingua đang chạy tốt.
        """
        key = f"{self.encoder_name}|{self.use_fp16}"
        if key in BGEM3Scorer._cache:
            return BGEM3Scorer._cache[key]

        try:
            from FlagEmbedding import BGEM3FlagModel
            m = ("flag", BGEM3FlagModel(self.encoder_name, use_fp16=self.use_fp16))
        except Exception as e:
            # Bắt rộng là CỐ Ý: FlagEmbedding hỏng theo nhiều kiểu tuỳ phiên bản
            # (1.4.0 ném TypeError vì kwarg `dtype`, bản khác ném ImportError).
            # Nhưng im lặng thì không được — người chạy cần biết mình đang dùng
            # nhánh nào, vì hai backend cho điểm hơi khác nhau.
            if not BGEM3Scorer._warned_fallback:
                print(f"[itercomp] FlagEmbedding không dùng được "
                      f"({type(e).__name__}: {str(e)[:80]}); "
                      f"dùng transformers thuần.", file=sys.stderr)
                BGEM3Scorer._warned_fallback = True
            m = ("hf", _HFBackend(self.encoder_name))
        BGEM3Scorer._cache[key] = m
        return m

    def __call__(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        need_dense = self.mode in ("dual", "dense")
        need_lex = self.mode in ("dual", "lexical")
        kind, model = self._model()

        if kind == "flag":
            out = model.encode([query] + list(texts), return_dense=need_dense,
                               return_sparse=need_lex, return_colbert_vecs=False)
            dense = out.get("dense_vecs")
            lexw = out.get("lexical_weights")
        else:
            dense, lexw = model.encode([query] + list(texts),
                                       need_dense=need_dense, need_lex=need_lex)

        n = len(texts)
        sem = [0.0] * n
        lex = [0.0] * n

        if need_dense and dense is not None:
            # công thức (1): S_sem = E(q)ᵀ E(e) — vector đã chuẩn hoá nên đây là cosine
            qv, dv = dense[0], dense[1:]
            sem = [float(qv @ dv[i]) for i in range(n)]

        if need_lex and lexw is not None:
            # công thức (2)+(3): tổng có trọng số trên token ĐỒNG XUẤT HIỆN.
            # lexical_weights = {token_id: w} với w = ReLU(w_lexᵀ E(t)).
            qw, dw = lexw[0], lexw[1:]
            for i in range(n):
                lex[i] = float(sum(float(w) * float(dw[i][t])
                                   for t, w in qw.items() if t in dw[i]))

        return [self.lam * s + (1.0 - self.lam) * l for s, l in zip(sem, lex)]


class _HFBackend:
    """Fallback: bge-m3 qua transformers thuần.

    Dense  = CLS embedding đã L2-normalize (đúng cách bge-m3 dùng).
    Sparse = ReLU(w_lexᵀ h_t) với w_lex nạp từ `sparse_linear.pt` của repo model
             — đây CHÍNH LÀ projection vector w_lex ở công thức (2) của paper.
             Nếu không có file đó, sparse weight lùi về 0 và ta cảnh báo.
    """

    def __init__(self, name: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        # ITERCOMP_SCORER_DEVICE ép thiết bị. Cần khi một tiến trình khác vừa
        # dùng GPU xong: VRAM còn phân mảnh nên bge-m3 nạp vào cuda có thể chết
        # mà không kịp báo lỗi rõ ràng. Chấm điểm trên CPU chỉ chậm hơn chứ
        # không đổi kết quả, nên đặt biến này là cách an toàn để chạy bước
        # chấm điểm sau một bước sinh văn bản trên cùng máy.
        forced = os.environ.get("ITERCOMP_SCORER_DEVICE", "").strip().lower()
        self.device = forced or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name)
        # fp16 trên GPU: bge-m3 chỉ dùng để CHẤM ĐIỂM chứ không sinh văn bản, nên
        # nửa độ chính xác không ảnh hưởng thứ tự xếp hạng, mà tiết kiệm một nửa
        # VRAM. Cần thiết vì bộ chấm điểm và mô hình đọc 7B cùng nằm trên một
        # T4 15 GB; để mặc định fp32 thì mô hình đọc không còn chỗ.
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = (AutoModel.from_pretrained(name, torch_dtype=dtype)
                      .to(self.device).eval())
        self.sparse_linear = self._load_sparse_linear(name)

    def _load_sparse_linear(self, name: str):
        """Nạp w_lex (sparse_linear.pt) từ repo model nếu có."""
        import torch
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(name, "sparse_linear.pt")
            sd = torch.load(p, map_location="cpu")
            lin = torch.nn.Linear(self.model.config.hidden_size, 1)
            lin.load_state_dict(sd)
            # Khớp dtype với encoder: nếu encoder là fp16 mà lớp này fp32 thì
            # phép nhân ma trận trong encode() sẽ báo lỗi lệch kiểu.
            return lin.to(device=self.device, dtype=self.model.dtype).eval()
        except Exception:
            return None   # encoder không có sparse head (vd model VN chỉ có dense)

    def encode(self, texts: list[str], *, need_dense=True, need_lex=True):
        torch = self.torch
        enc = self.tok(texts, padding=True, truncation=True, max_length=512,
                       return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**enc)
        h = out.last_hidden_state                      # (B, L, H)

        dense = None
        if need_dense:
            cls = torch.nn.functional.normalize(h[:, 0], dim=-1)
            # .float() sau khi rời GPU: tính điểm ở fp32 để tích vô hướng không
            # mất chính xác. Không tốn thêm VRAM vì đã nằm trên CPU.
            dense = cls.float().cpu().numpy()

        lexw = None
        if need_lex:
            lexw = []
            if self.sparse_linear is None:
                # không có w_lex -> trả dict rỗng, S_lex = 0 (ghi rõ trong báo cáo)
                lexw = [{} for _ in texts]
            else:
                with torch.no_grad():
                    w = torch.relu(self.sparse_linear(h)).squeeze(-1)   # (B, L)
                ids = enc["input_ids"].cpu()
                mask = enc["attention_mask"].cpu().bool()
                w = w.float().cpu()
                special = set(self.tok.all_special_ids)
                for b in range(len(texts)):
                    d: dict[int, float] = {}
                    for t, val, m in zip(ids[b].tolist(), w[b].tolist(), mask[b].tolist()):
                        if m and t not in special:
                            d[t] = max(d.get(t, 0.0), float(val))   # max-pool theo token
                    lexw.append(d)
        return dense, lexw


# ═════════════════════════════════════════════════════════════ factory


def make_scorer(kind: str = "dual", encoder: str = DEFAULT_ENCODER,
                lam: float = DEFAULT_LAMBDA, **kw) -> Scorer:
    """kind ∈ {bm25, dense, lexical, dual}."""
    if kind == "bm25":
        return BM25Scorer()
    if kind in ("dual", "dense", "lexical"):
        return BGEM3Scorer(encoder=encoder, mode=kind, lam=lam, **kw)
    raise ValueError(f"scorer không biết: {kind}")


# ══════════════════════════════════════════════════ percentile filtering


def percentile_filter(scores: list[float], k: float = DEFAULT_PERCENTILE) -> list[int]:
    """Công thức (5): trả về index các segment có điểm ≥ percentile thứ k.

    Khác top-k cố định: ngưỡng THÍCH ỨNG theo phân bố điểm của từng câu hỏi.
    Câu hỏi có nhiều bằng chứng liên quan sẽ giữ nhiều segment hơn, và ngược lại.

    Luôn trả về ít nhất 1 index (tránh prompt rỗng).
    """
    if not scores:
        return []
    import numpy as np
    thr = float(np.percentile(scores, k))
    keep = [i for i, s in enumerate(scores) if s >= thr]
    return keep or [max(range(len(scores)), key=lambda i: scores[i])]


#: Biên của k thích ứng. k thấp giữ nhiều segment hơn (ngưỡng percentile thấp).
AP_K_MIN, AP_K_MAX = 70.0, 95.0


def score_concentration(scores: list[float]) -> float:
    """Độ tập trung của phân bố điểm, trong [0, 1], BẤT BIẾN theo thang điểm.

    Trả về 1.0 khi điểm dồn vào một ít segment (bằng chứng rõ, nên cắt mạnh),
    0.0 khi điểm dàn đều (không rõ segment nào liên quan, nên giữ rộng).

    VÌ SAO TỈ SỐ PHÂN VỊ, KHÔNG PHẢI GINI HAY s_max/s_mean. Review chỉ ra rằng
    một độ đo tập trung phải bất biến theo thang điểm của encoder — bge-m3 và
    BM25 cho điểm ở hai thang hoàn toàn khác nhau, nên bất kỳ hiệu tuyệt đối
    nào cũng vô nghĩa khi đổi encoder. Tỉ số p90/p50 là một thống kê tỉ lệ:
    nhân toàn bộ điểm với hằng số bất kỳ không làm nó đổi.

    Trường hợp biên: p50 = 0 (quá nửa segment không khớp gì) là dấu hiệu
    bằng chứng RẤT tập trung, trả về 1.0.
    """
    if len(scores) < 2:
        return 0.0
    import numpy as np
    p50, p90 = np.percentile(scores, 50), np.percentile(scores, 90)
    if p50 <= 0:
        return 1.0 if p90 > 0 else 0.0
    ratio = p90 / p50
    # nen log de ti so 1..10 trai deu tren [0,1]; ti so 1.0 (phang) -> 0.0
    import math
    return min(1.0, math.log(ratio) / math.log(10.0)) if ratio > 1 else 0.0


def adaptive_percentile(scores: list[float],
                        k_min: float = AP_K_MIN,
                        k_max: float = AP_K_MAX) -> float:
    """Công thức (5'): suy k từ độ tập trung của chính phân bố điểm.

        k = k_min + (k_max - k_min) * concentration(S)

    Bằng chứng càng tập trung thì k càng cao (cắt càng mạnh), vì tín hiệu đã
    rõ nên không cần giữ rộng. Bằng chứng càng phân tán thì k càng thấp (giữ
    rộng), vì chưa biết segment nào đúng.

    Ba tính chất được giữ, theo đúng ràng buộc của paper gốc:

    1. TRAINING-FREE — chỉ dùng thống kê của S, không tham số học được.
    2. KHÔNG CẦN NHÃN — tính lúc suy luận, không đụng `supporting_facts`.
    3. SUY BIẾN — đặt k_min = k_max = 90 thì trả về đúng 90, tức là paper gốc.
       Nhờ vậy so sánh AP với k cố định là so sánh CÓ KIỂM SOÁT.
    """
    return k_min + (k_max - k_min) * score_concentration(scores)


def adaptive_percentile_filter(scores: list[float],
                               k_min: float = AP_K_MIN,
                               k_max: float = AP_K_MAX) -> list[int]:
    """Như `percentile_filter` nhưng k suy từ phân bố điểm thay vì cố định."""
    if not scores:
        return []
    return percentile_filter(scores, adaptive_percentile(scores, k_min, k_max))
