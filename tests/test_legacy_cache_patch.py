"""Vá `past_key_values` cho llmlingua — bọc `model.__call__`.

VÌ SAO KHÓ. Máy phát triển có transformers 4.45, vốn trả `tuple`, nên **không
thể kiểm bản vá bằng cách chạy thật**. Đã vá sai ba lần vì vậy:

  1. Bọc `model.forward` — llmlingua gọi `self.model(...)` tức `__call__`.
  2. Bọc `get_ppl` — chuyển được đầu ra, nhưng `iterative_compress_prompt`
     truyền cache TRỞ LẠI ở vòng sau và transformers biến nó thành `Cache`
     lần nữa.
  3. Bọc `model.__call__` — điểm duy nhất mọi cache đi qua, cả vào lẫn ra.

Test dưới đây giả lập `Cache` nên bắt được hồi quy mà không cần nâng cấp
transformers. Chúng mô phỏng đúng vòng lặp của llmlingua: gọi nhiều lần, mỗi
lần truyền lại cache của lần trước.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    """Lấy riêng hàm vá, không chạy main() của run_eval."""
    src = (ROOT / "scripts" / "run_eval.py").read_text()
    start = src.index("def _to_legacy")
    end = src.index("class LLMLinguaMethod")
    ns: dict = {}
    exec(src[start:end], ns)
    return ns["_force_legacy_cache"]


class _Cache:
    """Giả lập `Cache` của transformers >= 4.47."""

    def __init__(self, n=2):
        self.n = n

    def to_legacy_cache(self):
        return [(f"k{i}", f"v{i}") for i in range(self.n)]


class _Out:
    def __init__(self):
        self.past_key_values = _Cache()


class _Mdl:
    """Mô hình luôn trả `Cache`, kể cả khi nhận cache dạng cũ.

    Đó chính là hành vi khiến bản vá thứ hai thất bại.
    """

    def __init__(self):
        self.config = type("C", (), {})()
        self.seen_inputs = []

    def forward(self, *a, **kw):
        self.seen_inputs.append(kw.get("past_key_values"))
        return _Out()

    def __call__(self, *a, **kw):
        # nn.Module.__call__ gọi forward — mô phỏng đúng chuỗi đó, vì bản vá
        # bọc forward chứ không gán __call__ (dunder tra qua class).
        return self.forward(*a, **kw)


class _Comp:
    def __init__(self):
        self.model = _Mdl()


def test_dau_ra_thanh_tuple():
    c = _Comp()
    _load()(c)
    out = c.model(past_key_values=None)
    for k, v in out.past_key_values:        # dòng llmlingua vỡ ở đó
        assert k and v


def test_vong_lap_nhieu_lan():
    """Mô phỏng vòng lặp llmlingua: truyền lại cache của lần trước.

    Đây là kịch bản bản vá thứ hai không xử lý được.
    """
    c = _Comp()
    _load()(c)
    pkv = None
    for _ in range(3):
        out = c.model(past_key_values=pkv)
        pkv = out.past_key_values
        for k, v in pkv:                    # phải lặp được ở MỌI vòng
            assert k and v


def test_dau_vao_cung_duoc_chuyen():
    """Cache dạng `Cache` truyền VÀO cũng phải thành tuple.

    llmlingua cắt cache thành list-of-lists rồi truyền lại; nếu ta để nguyên
    `Cache` đi vào, transformers mới có thể không nhận dạng.
    """
    c = _Comp()
    _load()(c)
    c.model(past_key_values=_Cache())
    got = c.model.seen_inputs[-1]
    assert isinstance(got, list), f"đầu vào chưa chuyển: {type(got)}"
    for k, v in got:
        assert k and v


def test_khong_va_hai_lan():
    """Gọi vá hai lần không được bọc lồng nhau."""
    c = _Comp()
    patch = _load()
    patch(c)
    first = c.model.forward
    patch(c)
    assert c.model.forward is first


def test_giu_nguyen_tuple_san_co():
    """Cache đã ở dạng cũ thì để nguyên."""
    src = (ROOT / "scripts" / "run_eval.py").read_text()
    ns: dict = {}
    exec(src[src.index("def _to_legacy"):src.index("class LLMLinguaMethod")], ns)
    legacy = [("a", "b")]
    assert ns["_to_legacy"](legacy) is legacy
    assert ns["_to_legacy"](None) is None


def test_va_dung_call_khong_phai_forward():
    """Khoá lại bài học: forward và get_ppl đều không đủ."""
    src = (ROOT / "scripts" / "run_eval.py").read_text()
    body = src[src.index("def _force_legacy_cache"):src.index("class LLMLinguaMethod")]
    assert "mdl.forward = _fwd" in body, "phải bọc forward (dunder tra qua class)"
    assert "compressor.get_ppl = _ppl" in body, "phải bọc cả get_ppl"


# ── lan va thu NAM: iterative_compress_prompt (LongLLMLingua) ──────────
#
# Bon lan dau van chet tren Kaggle:
#   llmlingua/prompt_compressor.py:1659, in iterative_compress_prompt
#       for k, v in past_key_values
#   ValueError: too many values to unpack (expected 2)
#
# Nguyen nhan: get_ppl doc `past_key_values[0][0].shape[2]` TRUOC khi goi
# model. Mot Cache truyen VAO vo ngay tai do — lop boc dau-RA khong kip lam gi.


class _CompIter(_Comp):
    """Compressor co iterative_compress_prompt, nhu LongLLMLingua."""

    def __init__(self):
        super().__init__()
        self.seen_in_iter = []

    def iterative_compress_prompt(self, *a, past_key_values=None, **kw):
        # llmlingua lap `for k, v in past_key_values` — vo neu con la Cache
        if past_key_values is not None:
            for k, v in past_key_values:
                assert k and v
        self.seen_in_iter.append(past_key_values)
        return "compressed"


def test_va_boc_iterative_compress_prompt():
    """Cache truyen vao iterative_compress_prompt phai thanh tuple."""
    c = _CompIter()
    _load()(c)
    c.iterative_compress_prompt(past_key_values=_Cache())   # khong duoc nem
    got = c.seen_in_iter[-1]
    assert isinstance(got, list), f"van con {type(got).__name__}"
    for k, v in got:
        assert k and v


def test_get_ppl_chuyen_ca_dau_VAO():
    """get_ppl doc pkv[0][0].shape TRUOC khi goi model -> phai chan dau vao."""
    seen = {}

    class _C(_Comp):
        def get_ppl(self, *a, past_key_values=None, **kw):
            seen["in"] = past_key_values
            if past_key_values is not None:
                past_key_values[0][0]          # vo neu con la Cache
            return 0.0, _Cache()

    c = _C()
    _load()(c)
    c.get_ppl(past_key_values=_Cache())
    assert isinstance(seen["in"], list), "dau VAO cua get_ppl van la Cache"


def test_khoa_lai_ca_ba_diem_boc():
    """Bai hoc nam lan va: thieu bat ky diem nao la lai ton mot phien GPU."""
    src = (ROOT / "scripts" / "run_eval.py").read_text()
    body = src[src.index("def _force_legacy_cache"):src.index("class LLMLinguaMethod")]
    assert "mdl.forward = _fwd" in body
    assert "compressor.get_ppl = _ppl" in body
    assert "compressor.iterative_compress_prompt = _iter" in body, \
        "phai boc ca iterative_compress_prompt (lan va thu 5)"
    assert 'kw.get("past_key_values") is not None' in body, \
        "get_ppl phai chuyen dau VAO, khong chi dau ra"


# ── lan va thu SAU: _to_legacy tu no that bai im lang ─────────────────
#
# Bon lop boc deu chay dung (traceback Kaggle cho thay `_iter` DUOC goi),
# nhung `_to_legacy` tra ve nguyen cai Cache vi transformers moi da BO
# `to_legacy_cache()`, va ban cu ket thuc bang `return pkv`.


def _tl():
    src = (ROOT / "scripts" / "run_eval.py").read_text()
    ns: dict = {}
    exec(src[src.index("def _to_legacy"):src.index("class LLMLinguaMethod")], ns)
    return ns["_to_legacy"]


class _CacheKeyValue:
    """transformers ~4.47-4.5x: khong co to_legacy_cache, co key_cache/value_cache."""

    def __init__(self, n=2):
        self.key_cache = [f"k{i}" for i in range(n)]
        self.value_cache = [f"v{i}" for i in range(n)]


class _Layer:
    def __init__(self, i):
        self.keys, self.values = f"k{i}", f"v{i}"


class _CacheLayers:
    """transformers moi: chi co .layers, moi lop co .keys/.values."""

    def __init__(self, n=2):
        self.layers = [_Layer(i) for i in range(n)]


class _CacheOpaque:
    """Cache khong theo API nao — PHAI nem, khong duoc im lang."""
    pass


def test_cache_khong_co_to_legacy_cache():
    """Dung tinh huong lam chet phien GPU thu 6."""
    got = _tl()(_CacheKeyValue())
    assert isinstance(got, list), f"tra ve {type(got).__name__}"
    assert [(k, v) for k, v in got] == [("k0", "v0"), ("k1", "v1")]


def test_cache_chi_co_layers():
    got = _tl()(_CacheLayers())
    assert [(k, v) for k, v in got] == [("k0", "v0"), ("k1", "v1")]


def test_cache_la_khong_chuyen_duoc_thi_NEM():
    """Bai hoc trung tam: im lang tra ve Cache = mat mot phien GPU."""
    obj = _CacheOpaque()
    try:
        got = _tl()(obj)
    except TypeError as e:
        assert "khong chuyen duoc" in str(e).lower() or "Kh\u00f4ng chuy\u1ec3n" in str(e)
        return
    raise AssertionError(f"phai NEM, nhung tra ve {type(got).__name__}")


def test_selftest_chay_duoc_voi_transformers_that():
    """selftest_cache_patch() phai pass tren transformers dang cai."""
    src = (ROOT / "scripts" / "run_eval.py").read_text()
    ns: dict = {}
    exec(src[src.index("def _to_legacy"):src.index("class LLMLinguaMethod")], ns)
    msg = ns["selftest_cache_patch"]()
    assert "OK" in msg, msg
