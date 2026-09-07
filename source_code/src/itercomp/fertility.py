"""Đo và hiệu chỉnh token fertility giữa các ngôn ngữ.

CẢI TIẾN CHO TIẾNG VIỆT (không có trong bài báo gốc).

Vấn đề. Các bộ tách token BPE phổ biến được huấn luyện chủ yếu trên tiếng Anh,
nên âm tiết tiếng Việt có dấu bị chia thành nhiều token con. Hệ quả là cùng một
lượng thông tin, văn bản tiếng Việt tiêu tốn nhiều token hơn — hiện tượng này
được gọi là *fertility* (Ács, 2019) và đã được đo trên nhiều ngôn ngữ
(Petrov et al., NeurIPS 2023; Ahia et al., EMNLP 2023; Lundin et al., 2025).

Hệ quả cho việc nén câu lệnh, và đây là điểm chưa được nêu trong tài liệu:
**tỉ lệ nén 2x ở tiếng Việt và tiếng Anh KHÔNG giữ lại cùng lượng thông tin.**
Khảo sát về nén câu lệnh của Li et al. (NAACL 2025) không đề cập tới ngôn ngữ,
và LLMLingua-2 đánh giá trên tiếng Trung nhưng không xét khác biệt số token.
Vì vậy so sánh hai ngôn ngữ ở *cùng tỉ lệ token* là so sánh lệch: ở tiếng Việt,
cùng một tỉ lệ tương ứng với ít nội dung thực chất hơn.

Module này cung cấp:
    measure_fertility   đo tỉ lệ token/từ và token VN so với token EN
    effective_ratio     quy tỉ lệ nén về "tỉ lệ hiệu dụng" đã hiệu chỉnh fertility
    syllable_safe_spans nhận diện biên âm tiết để không cắt giữa một âm tiết
"""

from __future__ import annotations

import re
import unicodedata

# Ngưỡng coi là "fertility cao" — trên mức này thì tỉ lệ nén danh nghĩa
# đánh giá quá cao lượng thông tin thực sự được giữ lại.
HIGH_FERTILITY = 1.3


def _encoder(name: str = "cl100k_base"):
    import tiktoken
    return tiktoken.get_encoding(name)


def measure_fertility(text: str, encoding: str = "cl100k_base") -> dict:
    """Đo fertility của một đoạn văn bản.

    tokens_per_word là số token trung bình mỗi từ (tách theo khoảng trắng).
    Với tiếng Việt cần lưu ý: khoảng trắng phân tách *âm tiết*, không phải *từ*
    ("học sinh" là một từ, hai âm tiết), nên đại lượng này thực chất là
    token/âm tiết — vẫn so sánh được giữa các ngôn ngữ nếu dùng câu song song.
    """
    enc = _encoder(encoding)
    n_tok = len(enc.encode(text))
    n_word = len(text.split()) or 1
    n_char = len(text) or 1
    return {
        "tokens": n_tok,
        "words": n_word,
        "chars": n_char,
        "tokens_per_word": n_tok / n_word,
        "chars_per_token": n_char / n_tok if n_tok else 0.0,
    }


def relative_fertility(vi_text: str, en_text: str,
                       encoding: str = "cl100k_base") -> float:
    """Fertility tương đối trên một cặp câu song song: token VN / token EN.

    Phải dùng câu SONG SONG (cùng nội dung) để loại nhiễu do độ dài nội dung.
    """
    enc = _encoder(encoding)
    n_en = len(enc.encode(en_text)) or 1
    return len(enc.encode(vi_text)) / n_en


def effective_ratio(compressed: str, original: str,
                    fertility: float,
                    encoding: str = "cl100k_base") -> dict:
    """Tỉ lệ nén danh nghĩa và tỉ lệ hiệu dụng đã hiệu chỉnh fertility.

    `nominal` là đại lượng mà bài báo gốc báo cáo: số token còn lại chia số token
    ban đầu. `effective` quy nó về "tương đương tiếng Anh" bằng cách chia cho
    fertility, để trả lời câu hỏi: *nếu cùng nội dung này viết bằng tiếng Anh thì
    tỉ lệ nén tương ứng là bao nhiêu?*

    Số token TIẾT KIỆM được cũng được quy đổi: vì tiếng Việt tốn nhiều token hơn,
    cùng một tỉ lệ nén cắt được nhiều token thực tế hơn — đó là lý do nén câu
    lệnh có giá trị thực dụng cao hơn với tiếng Việt.
    """
    enc = _encoder(encoding)
    n_c = len(enc.encode(compressed))
    n_o = len(enc.encode(original)) or 1
    nominal = n_c / n_o
    return {
        "nominal_ratio": nominal,
        "effective_ratio": nominal / fertility if fertility else nominal,
        "tokens_saved": n_o - n_c,
        "tokens_saved_en_equiv": (n_o - n_c) / fertility if fertility else (n_o - n_c),
        "fertility": fertility,
    }


# ─────────────────────────────────── biên âm tiết tiếng Việt

# Tiếng Việt là ngôn ngữ đơn lập theo âm tiết: khoảng trắng phân tách âm tiết, và
# mỗi âm tiết là một đơn vị mang nghĩa. Cắt giữa một âm tiết phá vỡ dấu thanh và
# biến từ này thành từ khác, nên có hại hơn ở tiếng Anh.
_VN_SYLLABLE = re.compile(r"[0-9A-Za-zÀ-ỹ]+", re.UNICODE)


def syllable_spans(text: str) -> list[tuple[int, int]]:
    """Trả về các khoảng [đầu, cuối) của từng âm tiết, dùng để tránh cắt giữa."""
    return [(m.start(), m.end()) for m in _VN_SYLLABLE.finditer(text)]


def has_diacritics(text: str) -> bool:
    """Văn bản có chứa dấu thanh/dấu phụ tiếng Việt hay không."""
    return any(unicodedata.combining(c) for c in unicodedata.normalize("NFD", text))


def count_broken_syllables(compressed: str, original: str) -> int:
    """Đếm số âm tiết bị cắt dở trong bản nén.

    Một âm tiết coi là bị cắt nếu nó xuất hiện trong bản nén dưới dạng tiền tố
    hoặc hậu tố THẬT SỰ của một âm tiết trong bản gốc — dấu hiệu bộ nén đã cắt
    giữa một đơn vị mang nghĩa.

    Đây là độ đo đặc thù tiếng Việt: các nghiên cứu về nén câu lệnh trên tiếng
    Anh không cần đo đại lượng này.
    """
    orig = {original[a:b] for a, b in syllable_spans(original)}
    broken = 0
    for a, b in syllable_spans(compressed):
        syl = compressed[a:b]
        if syl in orig:
            continue
        if any(o != syl and (o.startswith(syl) or o.endswith(syl)) for o in orig):
            broken += 1
    return broken


#: Hư từ tiếng Việt — mang quan hệ ngữ pháp chứ không mang nội dung từ vựng.
#:
#: Đây là điểm cốt lõi của phép đo. Tiếng Việt là ngôn ngữ ĐƠN LẬP: quan hệ
#: so sánh, sở hữu, thời gian, phủ định được mã hoá bằng những âm tiết ĐỘC LẬP
#: đứng cạnh từ nội dung. Tiếng Anh mã hoá cùng thông tin đó bằng hình vị DÍNH
#: LIỀN (bigger, -ed, -'s) mà không tokenizer nào tách rời được.
#:
#: Hệ quả: một bộ nén lọc token theo "độ thông tin" sẽ cắt hư từ tiếng Việt —
#: và cắt luôn quan hệ ngữ pháp — trong khi vẫn an toàn trên tiếng Anh. Không
#: phải vì mô hình kém tiếng Việt, mà vì loại hình ngôn ngữ khác nhau.
FUNCTION_WORDS: dict[str, frozenset[str]] = {
    # so sánh: mất "hơn" thì "nhỏ hơn 7" thành "nhỏ 7"
    "so sánh": frozenset("hơn kém nhất bằng như cùng khác".split()),
    # phủ định: mất "không" thì câu ĐẢO NGHĨA
    "phủ định": frozenset("không chưa chẳng đừng chớ".split()),
    # quan hệ/sở hữu
    "quan hệ": frozenset("của với và hoặc là do bởi tại vì nên mà thì".split()),
    # thời gian: mất "năm" thì "30 10 1960" hết là một ngày
    "thời gian": frozenset("năm tháng ngày khi trước sau đang đã sẽ từ đến".split()),
    # định vị
    "định vị": frozenset("ở trong ngoài trên dưới giữa tại về cho theo".split()),
}

#: Hư từ tiếng Anh, ĐỐI CHỨNG — cùng năm loại quan hệ ngữ nghĩa.
#:
#: Danh sách này cố ý bắt cặp theo CHỨC NĂNG chứ không dịch từng chữ, để phép
#: so sánh trả lời đúng câu hỏi: cùng một quan hệ ngữ pháp, ngôn ngữ nào để nó
#: ở âm tiết rời (cắt được) và ngôn ngữ nào gói nó vào hình vị dính liền?
#:
#: So sánh là ví dụ sạch nhất. Tiếng Việt: "nhỏ HƠN" — hai âm tiết, bỏ được
#: một. Tiếng Anh: "smaller" — một token, bộ lọc giữ hoặc bỏ cả cụm, không thể
#: bỏ riêng phần so sánh. Nên danh sách EN dưới đây thiếu hẳn phần hậu tố -er
#: /-est, và thiếu đó CHÍNH LÀ kết quả, không phải sơ suất.
FUNCTION_WORDS_EN: dict[str, frozenset[str]] = {
    "so sánh": frozenset("than as same different most least".split()),
    "phủ định": frozenset("not no never nor neither without".split()),
    "quan hệ": frozenset("of with and or is was by because so that".split()),
    "thời gian": frozenset(
        "year month day when before after during since until".split()),
    "định vị": frozenset("in on at out under between over from to for".split()),
}


def _fw_table(lang: str) -> dict[str, frozenset[str]]:
    return FUNCTION_WORDS_EN if lang == "en" else FUNCTION_WORDS


#: Tra ngược âm tiết -> loại, dựng một lần cho mỗi ngôn ngữ.
_FW_CLASS: dict[str, dict[str, str]] = {
    lang: {w: cls for cls, ws in _fw_table(lang).items() for w in ws}
    for lang in ("vi", "en")
}


def count_function_words(text: str, lang: str = "vi") -> dict[str, int]:
    """Đếm hư từ theo từng loại, không phân biệt hoa thường.

    Trả về dict {loại: số lần}, cộng khoá 'tổng'.
    """
    table, lookup = _fw_table(lang), _FW_CLASS[lang]
    out = {cls: 0 for cls in table}
    for a, b in syllable_spans(text):
        cls = lookup.get(text[a:b].lower())
        if cls:
            out[cls] += 1
    out["tổng"] = sum(out.values())
    return out


def function_word_loss(compressed: str, original: str,
                       lang: str = "vi") -> dict[str, float]:
    """Tỉ lệ hư từ bị bộ nén cắt, so với tỉ lệ âm tiết nội dung bị cắt.

    Trả về, cho mỗi loại, ``mất`` = 1 - (còn lại / ban đầu). Khoá 'nội dung' là
    cùng đại lượng đó tính trên các âm tiết KHÔNG phải hư từ.

    Con số cần nhìn là **tỉ số** giữa hai cái: nếu hư từ bị cắt với tỉ lệ CAO
    HƠN nội dung, bộ nén đang huỷ ngữ pháp nhanh hơn huỷ thông tin — chính là
    dạng hư hại mà độ đo âm tiết-vỡ không bắt được.
    """
    co = count_function_words(original, lang)
    cc = count_function_words(compressed, lang)
    out = {}
    for cls in _fw_table(lang):
        out[cls] = 1.0 - cc[cls] / co[cls] if co[cls] else float("nan")
    out["hư từ"] = 1.0 - cc["tổng"] / co["tổng"] if co["tổng"] else float("nan")

    n_o = len(syllable_spans(original)) - co["tổng"]
    n_c = len(syllable_spans(compressed)) - cc["tổng"]
    out["nội dung"] = 1.0 - n_c / n_o if n_o else float("nan")
    return out
