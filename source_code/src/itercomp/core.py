"""Thuật toán IterCOMP — vòng lặp nén có nhận thức suy luận.

Tái hiện từ mô tả trong ACL 2026 (https://aclanthology.org/2026.acl-long.1559/).

QUAN TRỌNG: bài báo gốc không công bố mã nguồn lẫn nội dung câu lệnh. Toàn bộ
prompt ở đây do chúng tôi tự thiết kế theo mô tả Mục 3, nên số tuyệt đối không
so trực tiếp được với bài báo — điều kiểm chứng là xu hướng tương đối.

Bốn bước mỗi vòng:
    1. phân rã tài liệu thành evidence segment (mức câu)
    2. chấm điểm dual-aspect, lọc theo percentile   -> xem scorer.py
    3. hỏi LLM: bằng chứng đã đủ chưa? đủ thì dừng sớm
    4. chưa đủ thì sinh câu hỏi phụ làm truy vấn vòng sau"""

from __future__ import annotations

from dataclasses import dataclass, field

from .llm import LLM, make_llm
from .metrics import tokenize
from .budget import budget_filter, split_long_segments
from .scorer import (AP_K_MAX, AP_K_MIN, DEFAULT_PERCENTILE,
                     adaptive_percentile, make_scorer, percentile_filter)


@dataclass
class Segment:
    """Một evidence segment — đơn vị nhỏ nhất mà IterCOMP gom dần."""

    doc_title: str
    sent_id: int
    text: str

    def __str__(self) -> str:
        return f"{self.doc_title}: {self.text}"


def decompose(context: dict) -> list[Segment]:
    """Bước 1 — tách documents thành evidence segments (mức câu).

    context theo format HotpotQA/VimQA: {"title": [...], "sentences": [[...], ...]}
    """
    segs = []
    for title, sents in zip(context["title"], context["sentences"]):
        for i, s in enumerate(sents):
            s = s.strip()
            if s:
                segs.append(Segment(doc_title=title, sent_id=i, text=s))
    return segs


# ═══════════════════════════════════════════════════ Bước 2: answerability

ANSWERABLE_PROMPT = """Bạn đang giải một câu hỏi multi-hop. Dưới đây là các bằng chứng đã thu thập.

CÂU HỎI: {question}

BẰNG CHỨNG ĐÃ CÓ:
{evidence}

Các bằng chứng trên đã ĐỦ để trả lời chính xác câu hỏi chưa?
Chỉ trả lời một từ: ANSWERABLE_YES hoặc ANSWERABLE_NO"""


def is_answerable(llm: LLM, question: str, selected: list[Segment]) -> bool:
    """Bước 2 — đánh giá đã đủ bằng chứng để trả lời chưa."""
    if not selected:
        return False
    evidence = "\n".join(f"[SEG{i}] {s}" for i, s in enumerate(selected))
    prompt = ANSWERABLE_PROMPT.format(question=question, evidence=evidence)
    # Backend cục bộ so được logit trong MỘT lượt forward thay vì sinh từng
    # token rồi chỉ đọc chữ đầu. Chưa bật mặc định: chưa đo được trên GPU nên
    # chưa biết có nhanh hơn thật không (generate cũng dừng sớm ở EOS, nên
    # khoảng cách hẹp hơn vẻ ngoài). Bật bằng ITERCOMP_FAST_ANSWERABLE=1.
    import os as _os
    if _os.getenv("ITERCOMP_FAST_ANSWERABLE") == "1" and hasattr(llm, "choose"):
        return llm.choose(prompt, ["YES", "NO"]) == "YES"
    return "YES" in llm(prompt, max_tokens=8).upper()


# ══════════════════════════════════════════════════ Bước 3: follow-up query

FOLLOWUP_PROMPT = """Bạn đang giải một câu hỏi multi-hop nhưng bằng chứng chưa đủ.

CÂU HỎI GỐC: {question}

BẰNG CHỨNG ĐÃ CÓ:
{evidence}

Hãy nêu MỘT câu hỏi phụ ngắn gọn, nhắm vào đúng thông tin còn THIẾU
để có thể trả lời câu hỏi gốc. Chỉ ghi câu hỏi phụ, không giải thích.

FOLLOW-UP:"""


def make_followup(llm: LLM, question: str, selected: list[Segment]) -> str:
    """Bước 3 — sinh câu hỏi phụ nhắm vào thông tin còn thiếu."""
    evidence = "\n".join(f"[SEG{i}] {s}" for i, s in enumerate(selected)) or "(chưa có)"
    return llm(FOLLOWUP_PROMPT.format(question=question, evidence=evidence), max_tokens=64)


# ══════════════════════════════════════════ Bước 4: retrieve + iterate


def score_segments(query: str, segs: list[Segment], scorer=None) -> list[float]:
    """Chấm điểm segment theo query.

    scorer=None -> mặc định dual-aspect bge-m3 (ĐÚNG như paper, Mục 4.2.2).
    Truyền scorer khác để ablation: make_scorer("bm25"|"dense"|"lexical"|"dual").
    """
    if not segs:
        return []
    if scorer is None:
        scorer = make_scorer("dual")
    return scorer(query, [str(s) for s in segs])


@dataclass
class IterCompResult:
    question: str
    selected: list[Segment]
    followups: list[str] = field(default_factory=list)
    iterations: int = 0
    #: Ghi lại quyết định dừng từng vòng khi dùng bộ dừng thích ứng.
    stop_info: list[dict] = field(default_factory=list)
    #: k thực tế dùng ở từng vòng khi bật adaptive_k — rỗng nếu dùng k cố định.
    k_per_iter: list[float] = field(default_factory=list)
    stopped_because: str = ""

    def to_prompt(self) -> str:
        return "\n".join(str(s) for s in self.selected)


def itercomp(
    llm: LLM,
    question: str,
    context: dict,
    *,
    max_iter: int = 5,          # paper Mục 5.1: "bound the maximum number of iterations at 5"
    max_ratio: float = 0.60,    # chặn an toàn: dừng nếu đã gom quá tỉ lệ này
    stop_threshold: float | None = None,  # None = dùng quyết định nhị phân của paper
    stop_min_gain: float | None = None,   # dừng khi vòng thêm quá ít token mới
    scorer=None,
    percentile: float = DEFAULT_PERCENTILE,
    adaptive_k: bool = False,
    k_min: float = AP_K_MIN,
    k_max: float = AP_K_MAX,
    top_k_per_iter: int | None = None,
    token_budget: int | None = None,
    budget_ratio: float | None = None,
    split_max_tokens: int | None = None,
) -> IterCompResult:
    """Vòng lặp IterCOMP đầy đủ.

    max_iter        số vòng tối đa
    scorer          None -> dual-aspect bge-m3 (đúng paper). Xem scorer.make_scorer()
    percentile      công thức (5) — giữ segment có điểm >= percentile thứ k.
                    ĐÂY LÀ CÁCH CỦA PAPER: ngưỡng thích ứng theo từng câu hỏi.
    adaptive_k      CẢI TIẾN của chúng tôi: suy k từ độ tập trung của phân bố
                    điểm ở MỖI VÒNG, thay vì một hằng số cho mọi câu hỏi. Paper
                    tự nêu ở Limitations rằng cần "adaptive mechanisms that
                    dynamically adjust these settings based on question
                    complexity" nhưng không hiện thực hoá. Đặt k_min = k_max
                    thì suy biến về paper gốc, nên so sánh là CÓ KIỂM SOÁT.
    top_k_per_iter  nếu != None thì dùng top-k cố định thay percentile (ABLATION).
    token_budget    dừng sớm nếu vượt ngân sách token (None = không giới hạn)
    """
    segs = decompose(context)                      # bước 1
    # CẢI TIẾN: chia nhỏ đoạn dài để độ hạt phân rã đồng đều giữa các bộ dữ liệu
    if split_max_tokens:
        segs = split_long_segments(segs, split_max_tokens)
    res = IterCompResult(question=question, selected=[])
    remaining = list(range(len(segs)))
    prev_selected: list = []
    query = question
    if scorer is None:
        scorer = make_scorer("dual")

    for it in range(1, max_iter + 1):
        res.iterations = it

        if not remaining:
            res.stopped_because = "hết segment"
            break

        # bước 4a — chấm điểm segment còn lại theo query hiện tại
        scores = score_segments(query, [segs[i] for i in remaining], scorer)

        if budget_ratio is not None:
            # CẢI TIẾN: lọc theo ngân sách token -> tỉ lệ nén so sánh được giữa
            # các bộ dữ liệu, không phụ thuộc độ hạt phân rã
            total = sum(tokenize(str(s)) .__len__() for s in segs)
            already = len(tokenize(res.to_prompt()))
            left = max(total * budget_ratio - already, 0)
            local = budget_filter(scores, [str(segs[i]) for i in remaining],
                                  budget_ratio=left / max(total, 1),
                                  total_tokens=total)
        elif top_k_per_iter is not None:
            # ablation: top-k cố định
            order = sorted(range(len(remaining)), key=lambda j: -scores[j])
            local = order[:top_k_per_iter]
        elif adaptive_k:
            # công thức (5'): k suy từ chính phân bố điểm của vòng này
            k_now = adaptive_percentile(scores, k_min, k_max)
            res.k_per_iter.append(k_now)
            local = percentile_filter(scores, k_now)
        else:
            # công thức (5): percentile filtering thích ứng (cách của paper)
            local = percentile_filter(scores, percentile)

        picked = [remaining[j] for j in local]

        # Chặn an toàn TRƯỚC khi gom: nếu mô hình không bao giờ báo "đủ bằng
        # chứng" (mô hình nhỏ thường trả ANSWERABLE_NO ở mọi vòng), vòng lặp sẽ
        # gom tới khi hết segment và bản "nén" có thể dài hơn ngữ cảnh gốc. Bài
        # báo không gặp tình huống này vì dùng mô hình 8B.
        if max_ratio is not None:
            total = sum(len(tokenize(str(x))) for x in segs) or 1
            used = len(tokenize(res.to_prompt()))
            keep = []
            for i in picked:
                add = len(tokenize(str(segs[i])))
                if res.selected and (used + add) / total > max_ratio:
                    res.stopped_because = f"đạt trần tỉ lệ ({max_ratio:.0%})"
                    break
                keep.append(i)
                used += add
            stop_now = len(keep) < len(picked)
            picked = keep
        else:
            stop_now = False

        prev_selected = list(res.selected)
        res.selected.extend(segs[i] for i in picked)
        remaining = [i for i in remaining if i not in set(picked)]
        if stop_now:
            break

        # dừng theo ngân sách token
        if token_budget is not None:
            n_tok = len(tokenize(res.to_prompt()))
            if n_tok >= token_budget:
                res.stopped_because = f"đạt token budget ({n_tok})"
                break

        # bước 2 — đủ bằng chứng chưa?
        # stop_threshold=None: đúng như bài báo, một quyết định nhị phân của LLM.
        # Đặt giá trị: dùng điểm tin cậy liên tục, biến ngưỡng dừng thành tham
        # số điều chỉnh được (xem stopping.py và ablation no-answerability, cho
        # thấy cơ chế dừng của bài báo đang mất 3.6 điểm F1).
        if stop_threshold is None and stop_min_gain is None:
            if is_answerable(llm, question, res.selected):
                res.stopped_because = "answerable"
                break
        else:
            from .stopping import should_stop
            stop, info = should_stop(
                llm, question, res.selected,
                prev_selected=prev_selected,
                threshold=stop_threshold if stop_threshold is not None else 0.5,
                min_gain=stop_min_gain)
            res.stop_info.append(info)
            if stop:
                res.stopped_because = info["reason"]
                break

        # bước 3 — chưa đủ thì sinh follow-up làm query cho vòng sau
        fu = make_followup(llm, question, res.selected)
        res.followups.append(fu)
        query = fu
    else:
        res.stopped_because = "hết vòng lặp"

    return res
