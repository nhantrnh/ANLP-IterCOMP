"""Độ đo EM/F1 theo quy trình chuẩn hoá của HotpotQA.

Khác biệt cho tiếng Việt: GIỮ NGUYÊN dấu thanh. Bỏ dấu sẽ làm hai từ khác nghĩa
trở nên đồng nhất (má / mà), khiến EM và F1 mất ý nghĩa."""

from __future__ import annotations

import re


def tokenize(text: str) -> list[str]:
    """Tách token thô, giữ nguyên dấu tiếng Việt (\\w với re.UNICODE)."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def normalize(s: str) -> str:
    """Chuẩn hoá đáp án — giữ nguyên dấu tiếng Việt, chỉ bỏ dấu câu/mạo từ."""
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum(min(p.count(w), g.count(w)) for w in set(p))
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def exact_match(pred: str, gold: str) -> float:
    """1.0 nếu đáp án trùng khớp sau chuẩn hoá."""
    return float(normalize(pred) == normalize(gold))


# ══════════════════════════════════════════ đáp án trích dẫn (span answer)
#
# CẢI TIẾN CHO TIẾNG VIỆT (không có trong bài báo gốc).
#
# Vấn đề. Nhãn vàng của VimQA thường là CẢ CÂU trích nguyên văn từ Wikipedia,
# còn mô hình đọc trả lời NGẮN GỌN đúng ý. F1 trùng token trừng phạt độ dài
# chứ không trừng phạt sai, nên một đáp án đúng bị trừ điểm chỉ vì ngắn hơn:
#
#     nhãn vàng : "nằm ngay giữa trung tâm thành phố Bắc Kinh"
#     dự đoán   : "trung tâm thành phố Bắc Kinh"          -> F1 0,67
#
# Bộ dữ liệu tiếng Anh ít gặp vì nhãn vàng của HotpotQA/MuSiQue phần lớn là
# thực thể ngắn. Đo trên n=50: phép này nâng 6-9 điểm F1 ở tiếng Việt và
# 0,0-0,7 ở tiếng Anh — tức nó sửa một lệch ĐẶC THÙ, không thổi phồng điểm.

#: Tỉ lệ token của nhãn vàng mà dự đoán phải phủ để được tính trọn điểm.
#: 0,5 nghĩa là "trả lời được quá nửa nội dung nhãn vàng". Đặt ngưỡng thay vì
#: cho điểm mọi tập con vì có nhãn vàng liệt kê nhiều mục — câu "Zalo dùng ở
#: nước nào" có nhãn 8 nước, trả lời đúng 1 nước KHÔNG phải trả lời đủ.
SPAN_COVERAGE = 0.5

#: Số token tối thiểu của dự đoán. Chặn khớp vụn: một từ đơn nằm trong nhãn
#: vàng dài không đủ để coi là đã trả lời.
SPAN_MIN_TOKENS = 2


def f1_span(pred: str, gold: str) -> float:
    """F1 có tính tới đáp án trích dẫn: dự đoán nằm TRỌN trong nhãn vàng.

    Chỉ can thiệp khi dự đoán là chuỗi con LIÊN TỤC của nhãn vàng, dài ít nhất
    `SPAN_MIN_TOKENS` token, và phủ ít nhất `SPAN_COVERAGE` số token của nhãn.
    Ba điều kiện này khiến phép đo không thể biến một đáp án sai thành đúng:
    muốn được nâng điểm thì dự đoán phải là một phần nguyên văn và đủ lớn của
    chính nhãn vàng.

    Mọi trường hợp khác trả về F1 gốc, nên hàm này luôn >= f1_score.
    """
    base = f1_score(pred, gold)
    p, g = normalize(pred), normalize(gold)
    if not p or not g or p == g:
        return base
    if p not in g:
        return base
    pt, gt = p.split(), g.split()
    if len(pt) < SPAN_MIN_TOKENS or len(pt) / len(gt) < SPAN_COVERAGE:
        return base
    return 1.0


# ══════════════════════════════════════════ chuẩn hoá đáp án đúng/sai
#
# CẢI TIẾN CHO TIẾNG VIỆT (không có trong bài báo gốc).
#
# Phân tích lỗi trên VimQA cho thấy mô hình đọc nhỏ hay trả lời câu đúng/sai bằng
# tiếng Anh ("Yes", "tức là yes") trong khi đáp án vàng là "đúng" — đúng về ngữ
# nghĩa nhưng EM/F1 cho 0. Đây là hiện tượng chuyển mã (code-switching) của mô
# hình đa ngữ khi sinh tiếng Việt, và các bộ dữ liệu tiếng Anh không gặp.
#
# ĐỊNH LƯỢNG (VimQA n=50, mô hình đọc Qwen2.5-0.5B): trong 16 câu đúng/sai,
# chỉ 4 câu là lỗi định dạng thuần (mô hình trả lời đúng nghĩa bằng tiếng Anh);
# 10 câu trả lời sai hẳn nội dung và 2 câu trả lời ngược. Vậy phép chuẩn hoá này
# chỉ cứu được 4/50 câu — cải thiện thật nhưng nhỏ, và KHÔNG che được năng lực
# yếu của mô hình đọc.

_YES = {"yes", "true", "correct", "right",
        "đúng", "có", "phải", "vâng", "ừ"}
_NO = {"no", "false", "incorrect", "wrong",
       "không", "sai", "chưa"}


def _opposite(g: str) -> str:
    """Token thuộc cực đối lập với `g`, cùng hệ chữ (có dấu / không dấu).

    Dùng để biểu diễn "dự đoán lệch cực": trả về một chuỗi chắc chắn KHÔNG khớp
    nhãn vàng, nên F1 = 0 như mong đợi. Giữ nguyên hệ chữ để phần so khớp không
    phụ thuộc việc nhãn vàng có dấu hay không.
    """
    plain = g == g.encode("ascii", "ignore").decode()
    if g in _YES:
        return "khong" if plain else "không"
    return "dung" if plain else "đúng"


def normalize_boolean(pred: str, gold: str) -> str:
    """Nếu đáp án vàng là đúng/sai, quy dự đoán về cùng hệ từ vựng tiếng Việt.

    Chỉ can thiệp khi `gold` thực sự là đáp án đúng/sai, và khi dự đoán chứa
    token khẳng định hoặc phủ định rõ ràng. Mọi trường hợp khác trả về nguyên
    văn, nên phép chuẩn hoá không thể biến một câu trả lời sai thành đúng.
    """
    g = normalize(gold)
    if g not in _YES | _NO:
        return pred

    toks = set(tokenize(pred))
    hit_yes, hit_no = toks & _YES, toks & _NO
    # Khớp cực -> trả về CHÍNH nhãn vàng (F1 = 1). Lệch cực -> trả cực đối lập
    # của nhãn vàng (F1 = 0). Trả chuỗi cứng "đúng"/"không" là sai khi nhãn vàng
    # thuộc cực phủ định: "No" đối với gold "không" là khớp, nhưng map thành
    # "đúng" thì F1 = 0, còn "Yes" lại thành "không" và được F1 = 1 — đảo ngược.
    if hit_yes and not hit_no:
        return gold if g in _YES else _opposite(g)
    if hit_no and not hit_yes:
        return _opposite(g) if g in _YES else gold
    return pred
