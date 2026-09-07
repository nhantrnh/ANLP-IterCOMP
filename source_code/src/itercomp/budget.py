"""Lọc theo ngân sách token thay vì theo phân vị điểm.

CẢI TIẾN ĐỀ XUẤT (không có trong bài báo gốc).

Vấn đề quan sát được. Bài báo lọc bằng ngưỡng phân vị trên điểm liên quan
(công thức 5), tức giữ lại khoảng $(100-k)\\%$ *số đoạn*. Nhưng đại lượng báo cáo
lại là tỉ lệ *token*. Hai đại lượng này chỉ trùng nhau khi các đoạn dài xấp xỉ
bằng nhau, và điều đó không đúng:

    bộ dữ liệu   số đoạn   từ/đoạn   độ lệch chuẩn
    MuSiQue           20      73,7            46,4
    2WikiMQA          31      22,4            10,5
    HotpotQA          41      26,6            13,4

MuSiQue phân rã thành ~20 đoạn văn dài, còn 2WikiMQA thành ~31 câu ngắn. Giữ 10%
của 20 đoạn dài cho ra rất ít token; giữ 15% của 31 câu ngắn lại cho ra nhiều hơn
theo tỉ lệ. Đây chính là lý do bài báo báo cáo tỉ lệ nén 0,14 trên MuSiQue nhưng
0,37 trên 2WikiMQA --- chênh lệch lớn hơn nhiều so với mức chênh k (90 so với 85)
có thể giải thích. Chúng tôi tái hiện được đúng hiện tượng này: 21,4% so với 35,1%.

Hệ quả: **tỉ lệ nén giữa các bộ dữ liệu không so sánh được với nhau**, vì cùng một
giá trị k cho ra lượng token khác nhau tuỳ độ hạt phân rã.

Giải pháp. Chọn đoạn theo thứ tự điểm giảm dần cho tới khi chạm ngân sách token
định trước. Cách này cho tỉ lệ nén *cùng một giá trị* trên mọi bộ dữ liệu, nên so
sánh giữa các bộ mới có nghĩa. Nó cũng bỏ được một siêu tham số: thay vì phải dò
k cho từng bộ, ta nêu thẳng ngân sách mong muốn.
"""

from __future__ import annotations

from .metrics import tokenize


def budget_filter(scores: list[float], texts: list[str],
                  budget_ratio: float = 0.15,
                  total_tokens: int | None = None,
                  min_keep: int = 1) -> list[int]:
    """Chọn đoạn theo điểm giảm dần cho tới khi đạt ngân sách token.

    budget_ratio  phần token muốn giữ lại so với toàn bộ ngữ cảnh
    total_tokens  tổng token của ngữ cảnh gốc; None thì tính từ `texts`
    min_keep      luôn giữ ít nhất ngần này đoạn, tránh câu lệnh rỗng

    Trả về chỉ số các đoạn được giữ, theo đúng thứ tự xuất hiện ban đầu để
    mạch văn không bị đảo.
    """
    if not scores:
        return []
    lens = [len(tokenize(t)) for t in texts]
    if total_tokens is None:
        total_tokens = sum(lens)
    budget = max(int(total_tokens * budget_ratio), 1)

    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    picked, used = [], 0
    for i in order:
        if picked and used + lens[i] > budget:
            continue                      # bỏ qua đoạn quá dài, thử đoạn kế tiếp
        picked.append(i)
        used += lens[i]
        if used >= budget:
            break
    while len(picked) < min_keep and len(picked) < len(scores):
        for i in order:
            if i not in picked:
                picked.append(i)
                break
    return sorted(picked)


def split_long_segments(segments: list, max_tokens: int = 60) -> list:
    """Chia nhỏ các đoạn quá dài để độ hạt phân rã đồng đều giữa các bộ dữ liệu.

    MuSiQue lưu mỗi tài liệu thành một đoạn văn duy nhất (trung bình 74 từ, có
    đoạn tới 288), trong khi HotpotQA và 2WikiMQA đã tách sẵn theo câu. Sự chênh
    lệch độ hạt này làm cho phép lọc theo phân vị hoạt động rất khác nhau giữa các
    bộ. Chia các đoạn dài theo ranh giới câu giúp phép so sánh công bằng hơn.

    Trả về danh sách Segment mới, giữ nguyên doc_title và sent_id gốc.
    """
    import re
    from dataclasses import replace

    out = []
    for seg in segments:
        if len(tokenize(seg.text)) <= max_tokens:
            out.append(seg)
            continue
        # tách theo dấu câu, giữ lại dấu để không mất thông tin ranh giới
        parts = re.split(r"(?<=[.!?])\s+", seg.text)
        buf = ""
        for part in parts:
            cand = (buf + " " + part).strip()
            if buf and len(tokenize(cand)) > max_tokens:
                out.append(replace(seg, text=buf))
                buf = part
            else:
                buf = cand
        if buf:
            out.append(replace(seg, text=buf))
    return out
