"""Bộ quyết định dừng cho vòng lặp suy luận.

IterCOMP hỏi LLM một câu nhị phân: "bằng chứng đã đủ chưa?" Quyết định đó
không thể hiệu chuẩn — không có núm nào để đánh đổi giữa dừng sớm (mất đáp án)
và chạy thừa (tốn token). Ablation của chúng tôi cho thấy cái giá: bỏ hẳn bước
kiểm tra làm F1 TĂNG từ 33.0 lên 36.6 trên VimQA, tức cơ chế dừng đang mất
3.6 điểm.

Ở đây thay bằng một điểm tin cậy liên tục, để ngưỡng dừng trở thành tham số
điều chỉnh được thay vì hành vi cố định của LLM.
"""
from __future__ import annotations

from .metrics import tokenize

#: Ngưỡng mặc định. Thấp hơn = dừng dễ hơn = nén mạnh hơn nhưng dễ mất đáp án.
DEFAULT_THRESHOLD = 0.5

ANSWERABLE_LOGIT_PROMPT = """Bằng chứng dưới đây có đủ để trả lời câu hỏi không?

CÂU HỎI: {question}

BẰNG CHỨNG:
{evidence}

Chỉ trả lời một từ: YES hoặc NO."""


def evidence_confidence(llm, question: str, selected: list) -> float:
    """Xác suất mô hình cho rằng bằng chứng đã đủ, trong [0, 1].

    Dùng logit của token đầu ("YES" vs "NO") thay vì đọc chuỗi sinh ra. Chuỗi
    chỉ cho một bit; logit cho một giá trị liên tục có thể đặt ngưỡng.
    """
    if not selected:
        return 0.0
    evidence = "\n".join(f"[{i}] {s}" for i, s in enumerate(selected))
    prompt = ANSWERABLE_LOGIT_PROMPT.format(question=question, evidence=evidence)

    if hasattr(llm, "choose_prob"):
        return llm.choose_prob(prompt, "YES", "NO")
    # Backend không hỗ trợ logit: lùi về nhị phân, 0/1
    out = llm(prompt, max_tokens=8).upper()
    return 1.0 if "YES" in out and "NO" not in out.split("YES")[0] else 0.0


def marginal_gain(prev_selected: list, selected: list) -> float:
    """Tỉ lệ token MỚI mà vòng vừa rồi thêm vào.

    Khi vòng lặp gần cạn thông tin hữu ích, mỗi vòng thêm càng ít. Giá trị này
    không cần gọi LLM nên miễn phí, và là tín hiệu dừng độc lập với việc mô
    hình tự đánh giá — hữu ích chính vì tự đánh giá là chỗ đang sai.
    """
    new = len(tokenize(" ".join(str(s) for s in selected)))
    old = len(tokenize(" ".join(str(s) for s in prev_selected)))
    return 0.0 if new == 0 else (new - old) / new


def should_stop(llm, question: str, selected: list, *,
                prev_selected: list | None = None,
                threshold: float = DEFAULT_THRESHOLD,
                min_gain: float | None = None) -> tuple[bool, dict]:
    """Quyết định dừng, kèm lý do để ghi log và phân tích.

    threshold  ngưỡng tin cậy; càng thấp càng dừng sớm
    min_gain   nếu đặt, dừng khi vòng vừa rồi thêm ít hơn tỉ lệ này token mới
    """
    conf = evidence_confidence(llm, question, selected)
    info = {"confidence": conf, "threshold": threshold}

    if min_gain is not None and prev_selected is not None:
        gain = marginal_gain(prev_selected, selected)
        info["marginal_gain"] = gain
        if gain < min_gain:
            info["reason"] = f"lợi ích cận biên {gain:.2f} < {min_gain}"
            return True, info

    stop = conf >= threshold
    info["reason"] = (f"tin cậy {conf:.2f} >= {threshold}" if stop
                      else f"tin cậy {conf:.2f} < {threshold}")
    return stop, info


# ══════════════════════════════════════════════ Các luật dừng từ công trình khác
#
# Khảo sát cho thấy "thay phán quyết nhị phân bằng điểm tin cậy hiệu chuẩn" đã
# được TASR (Agent4IR @ KDD 2026) công bố. Nên các luật dưới đây là ĐƯỜNG CƠ SỞ
# để so, không phải đóng góp của chúng tôi. Cài lại ở đây để so sánh có kiểm
# soát trên cùng một vòng lặp.

def tasr_margin(llm, question: str, selected: list) -> float:
    """Biên logit top1 − top2 tại token cam kết đáp án (TASR, KDD 2026 wksh).

    TASR đọc biên này ở token đầu tiên sau "Answer:" rồi hiệu chuẩn bằng hồi quy
    đẳng hướng. Ở đây trả về biên thô; phần hiệu chuẩn thuộc về script thí nghiệm
    vì nó cần tập dev.
    """
    if not selected:
        return 0.0
    evidence = "\n".join(str(s) for s in selected)
    prompt = (f"Dựa vào bằng chứng, trả lời câu hỏi.\n\n"
              f"BẰNG CHỨNG:\n{evidence}\n\nCÂU HỎI: {question}\nAnswer:")
    if not hasattr(llm, "top2_margin"):
        return 0.0
    return llm.top2_margin(prompt)


def tsss_repetition(prev_selected: list, selected: list) -> bool:
    """Dừng khi vòng vừa rồi không thêm segment nào (TSSS, NeurIPS 2025).

    Luật xác định, không gọi LLM. TASR dùng biến thể trên chuỗi đáp án; ở đây áp
    lên tập bằng chứng vì vòng lặp của IterCOMP gom bằng chứng chứ không sinh
    đáp án từng vòng.
    """
    return len(selected) == len(prev_selected)


def halt_coverage_proxy(scores: list[float], percentile: float) -> float:
    """Tỉ lệ segment vượt ngưỡng — proxy rẻ cho "độ phủ bằng chứng" (HALT, 2026).

    HALT đo độ phủ so với nhãn vàng, nên chỉ dùng được khi đánh giá. Proxy này
    thay bằng phần điểm nằm trên ngưỡng, tính được lúc suy luận.
    """
    if not scores:
        return 0.0
    import statistics
    thr = statistics.quantiles(scores, n=100)[int(percentile) - 1] \
        if len(scores) > 1 else scores[0]
    return sum(1 for s in scores if s >= thr) / len(scores)


#: Tên các luật dừng có thể so sánh. Giá trị là ghi nguồn để trích dẫn.
BASELINE_RULES = {
    "paper":     "IterCOMP: phán quyết nhị phân (Yun & Kim, ACL 2026)",
    "tasr":      "biên logit top1-top2 (TASR, Agent4IR@KDD 2026)",
    "tsss":      "phát hiện lặp lại (TSSS, NeurIPS 2025)",
    "conf":      "ngưỡng trên P(YES) — biến thể của TASR",
    "gain":      "lợi ích cận biên token, liên tục (của chúng tôi)",
}
