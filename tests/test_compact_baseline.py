"""CompAct: prompt va tham so phai khop repo goc, khong phai tu soan.

VI SAO KHOA LAI. Muc baseline cua bai chi ra nam trong sau baseline bi handicap
vi thanh phan tieng Anh, va ket luan rang cac dong do KHONG do duoc phuong
phap. Them mot baseline thu sau voi prompt DOAN se tao ra dung loai so sanh
khong cong bang ma bai dang canh bao. Nen prompt lay nguyen van tu
`dmis-lab/CompAct/utils.py::create_prompt`, tham so tu `scripts/run_prompt.sh`,
va test nay khoa chung lai.

Model card tren HuggingFace la template tu sinh — khong co dinh dang prompt,
khong vi du, khong neu base model. Moi thu phai truy tu repo GitHub.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "scripts" / "run_eval.py").read_text()
BODY = SRC[SRC.index("class CompActMethod"):SRC.index("class RecompAbstractiveMethod")]
# BODY bo docstring cua class: docstring co noi ve "[INST]" nhu tai lieu, nen
# kiem "khong hardcode [INST]" phai chay tren MA, khong phai tren chu thich.
_q = '"""'
CODE = BODY.split(_q)[0] + _q.join(BODY.split(_q)[2:]) if BODY.count(_q) >= 2 else BODY
# gop khoang trang de so khop doan prompt bi wrap qua nhieu dong
FLAT = " ".join(BODY.replace("\\n", " ").split())


def test_prompt_vong_dau_nguyen_van():
    """Cau lenh vong 0, trich nguyen van tu create_prompt cua ho."""
    for frag in ("Generate a summary of source documents to answer the question",
                 "Ensure the summary is under 200 words and does not include any",
                 "DO NOT make assumptions or attempt to answer the question",
                 "your job is to summarize only",
                 "if it lacks sufficient details to answer the question, print",
                 "You should provide the reason of evalution"):
        assert frag in FLAT, f"thieu/sai doan prompt: {frag[:55]}"


def test_prompt_vong_sau_co_prev_summary_va_prev_eval():
    """Vong >0 phai truyen CA prev_summary va prev_eval — bo mot cai la doi
    thuat toan, vi eval chinh la thu chi ra thong tin con thieu."""
    assert "Previous summary:" in BODY
    assert "Evaluation of previous summary:" in BODY
    assert "based on the evaluation of the previous summary" in BODY.replace("\n", " ")


def test_tham_so_khop_script_cua_ho():
    """run_prompt.sh: iter=6, segment_size=5; run_prompt.py: max_new_tokens=900,
    temperature=0 (greedy)."""
    assert "SEGMENT_SIZE = 5" in BODY
    assert "MAX_ITER = 6" in BODY
    assert "max_new_tokens=900" in BODY
    assert "do_sample=False" in BODY, "phai greedy (temperature=0 o ban goc)"


def test_dung_khi_COMPLETE():
    assert '"[COMPLETE]" in ev' in BODY
    assert '"[INCOMPLETE]" not in ev' in BODY, \
        "'[INCOMPLETE]' chua '[COMPLETE]' nhu chuoi con -> phai loai tru"


def test_parse_tach_dung_summary_va_eval():
    # IMPORT MODULE THAT, khong exec mot khoi class trong namespace rong.
    # Lan dau em exec("import re\n" + BODY) — tu cung cap import ma module
    # thieu, nen test PASS trong khi Kaggle chet sau 239 giay voi
    # NameError: name 're' is not defined. Roi bo import di thi exec lai mat
    # ca import cap module. Ca hai deu la tao tac cua harness. Import that
    # (an toan: run_eval.py chi co `if __name__ == "__main__"` o cap module)
    # kiem dung thu se chay tren Kaggle.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_re_run_eval", ROOT / "scripts" / "run_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = mod.CompActMethod
    s, e = cls._parse("Summary: Ha Noi la thu do.\nEvaluation: du thong tin. [COMPLETE]")
    assert s == "Ha Noi la thu do."
    assert "[COMPLETE]" in e
    # khong co tien to "Summary:"
    s2, e2 = cls._parse("Chi co noi dung.\nEvaluation: [INCOMPLETE] thieu ngay sinh.")
    assert s2.startswith("Chi co noi dung")
    assert "[INCOMPLETE]" in e2


def test_dung_chat_template_khong_tu_ghep():
    """Model la Mistral ([INST]...[/INST]); phai dung apply_chat_template chu
    khong hardcode dau phan cach."""
    assert "apply_chat_template" in CODE
    assert "[INST]" not in CODE, "khong hardcode dinh dang chat trong MA"


# ── dispatch: moi nhanh phai gan vao `methods`, khong phai ten khac ──
#
# Ban dau em viet `fns[m] = CompActMethod(...)` trong khi cac nhanh khac dung
# `methods[m]`. NameError chi lo ra tren Kaggle sau 155 giay, vi:
#   - `--reader mock` KHONG ngan method tu nap model 7B cua no, nen dry-run
#     local bat dau tai 14 GB (da phai kill va don 2.8 GB tai do dang);
#   - khong test nao cham toi khoi dispatch.
# Test nay doc thang khoi dispatch, khong nap gi.

def test_moi_nhanh_dispatch_gan_vao_methods():
    import re
    blk = SRC[SRC.index('        if m in ("raw"'):SRC.index("        else:\n            raise ValueError")] \
        if '        if m in ("raw"' in SRC else \
        SRC[SRC.index('elif m == "oracle"') - 200:SRC.index('elif m == "itercomp"')]
    gans = re.findall(r"^\s*(\w+)\[m\]\s*=", blk, re.M)
    assert gans, "khong tim thay phep gan nao trong khoi dispatch"
    sai = sorted({g for g in gans if g != "methods"})
    assert not sai, f"nhanh dispatch gan vao bien sai: {sai} (phai la 'methods')"


def test_compact_co_trong_dispatch_va_help():
    assert 'elif m == "compact":' in SRC
    assert "methods[m] = CompActMethod(" in SRC
    assert "compact," in SRC, "phai liet ke trong --methods help"


def test_module_import_re():
    """`_parse` dung re.search; module PHAI import re o cap module.

    Bug that: run_eval.py khong co `import re`. Test dau tien cua em tu chen
    "import re" vao truoc khi exec, nen no PASS trong khi Kaggle chet. Mot test
    cung cap dependency ma code thieu thi khong test gi ca.
    """
    import re as _re
    assert _re.search(r"^import re$", SRC, _re.M), \
        "run_eval.py thieu `import re` o cap module"
