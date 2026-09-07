"""Cổng kiểm tra repo — bắt tài liệu lỗi thời TRƯỚC khi người đọc bắt được.

VÌ SAO TỒN TẠI. Dự án này đã gặp tám loại lỗi tài liệu, và **không loại nào gây
lỗi biên dịch hay lỗi chạy**. PDF vẫn đẹp, script vẫn chạy, chỉ là nội dung sai:

  1. README nói "chưa chạy" trong khi `results/` đã có file
  2. Số trong tài liệu lệch số trong `results/`
  3. Cùng một đại lượng, hai tài liệu ghi hai con số
  4. Số đúng nhưng mô tả sai ĐẠI LƯỢNG ("81% headroom" vs "81% điểm Oracle")
  5. Cấu hình ghi sai so với `config` trong chính file kết quả
  6. README liệt kê file đã xoá, hoặc thiếu file mới thêm
  7. `requirements.txt` thiếu package mà mã thực sự import
  8. Script ghép hai nguồn dữ liệu với cấu hình khác nhau

Loại 2 do `verify_report_numbers.py` lo. Loại 8 do `tests/` lo. Script này lo
phần còn lại — những thứ so được **tự động** giữa tài liệu và hiện trạng repo.

    python scripts/check_repo.py          # báo cáo
    python scripts/check_repo.py --strict # thoát mã 1 nếu có lỗi (dùng cho CI)
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ══════════════════════════════════════════════════════════════════
# CẤU HÌNH — chỉ phần này cần sửa khi dùng cho repo khác.
# ══════════════════════════════════════════════════════════════════

#: Thư mục chứa mã Python cần soi import và đối chiếu với README.
CODE_DIRS = ("src/itercomp", "scripts", "research", "tests")

#: Thư mục chứa kết quả thí nghiệm (JSON có khoá "config").
RESULTS_DIR = "results"

#: Các tài liệu mô tả repo. Dùng để bắt file mồ côi và tham chiếu chết.
DOCS = ("README.md", "research/README.md", "report_en/report.tex",
        "results/PROVENANCE.md")

#: Tài liệu nào chịu trách nhiệm liệt kê file mã.
INVENTORY_DOC = "README.md"

#: (mức, thông điệp). "loi" chặn; "canh_bao" chỉ nhắc.
Issue = tuple[str, str]
issues: list[Issue] = []


def loi(msg: str) -> None:
    issues.append(("loi", msg))


def canh_bao(msg: str) -> None:
    issues.append(("canh_bao", msg))


def read(rel: str) -> str:
    p = ROOT / rel
    return p.read_text() if p.is_file() else ""


# ── 6. README có khớp cây thư mục thật không ────────────────────────
def check_file_inventory() -> None:
    """Mọi script/module phải được README nhắc tới, và ngược lại.

    Bắt cả hai chiều: file mới thêm mà quên ghi, và file đã xoá mà README
    vẫn liệt kê — cái thứ hai nguy hiểm hơn vì người đọc sẽ đi tìm.
    """
    readme = read(INVENTORY_DOC)
    for d in CODE_DIRS:
        for f in sorted((ROOT / d).glob("*.py")):
            if f.name.startswith("_"):
                continue
            if f.name not in readme:
                canh_bao(f"{INVENTORY_DOC} không nhắc {d}/{f.name}")

    # chieu nguoc: README nhac file khong ton tai
    for m in re.finditer(r"[\w/]+\.(?:py|ipynb|tex|sh)", readme):
        rel = m.group(0)
        if rel.startswith(("http", "www")):
            continue
        hits = list(ROOT.rglob(Path(rel).name))
        if not hits:
            loi(f"{INVENTORY_DOC} nhắc `{rel}` nhưng file KHÔNG tồn tại")


# ── 1. Tài liệu nói "chưa chạy" mà results/ đã có ───────────────────
#: Câu chữ trong tài liệu -> file kết quả chứng minh nó đã chạy.
CLAIMS_VS_RESULTS = {
    "research/README.md": [
        ("exp1_oracle_stopping.py", "exp1_oracle_vimqa_200_7b.json"),
        ("exp2_undiacritised.py", "exp2_undiacritised_200_7b.json"),
    ],
}
STALE_WORDS = ("chưa chạy", "Chưa chạy", "chưa ai gọi", "cần chạy")


def check_stale_status() -> None:
    for doc, pairs in CLAIMS_VS_RESULTS.items():
        text = read(doc)
        for script, result in pairs:
            if not (ROOT / RESULTS_DIR / result).is_file():
                continue
            for line in text.splitlines():
                if script in line and any(w in line for w in STALE_WORDS):
                    loi(f"{doc}: dòng nói `{script}` chưa chạy, nhưng "
                        f"{RESULTS_DIR}/{result} đã tồn tại")


# ── 5. Cấu hình trong tài liệu vs config trong file kết quả ─────────
#: (file kết quả, khoá config, giá trị tài liệu ĐANG ghi ở đâu đó)
CONFIG_CLAIMS = [
    ("vimqa_full_7b.json", "reader_model", "Qwen/Qwen2.5-7B-Instruct"),
    ("vimqa_200_7b.json", "reader_model", "Qwen/Qwen2.5-7B-Instruct"),
    ("lambda_sweep_vimqa.json", "reader_model", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("exp1_oracle_vimqa_200_7b.json", "scorer", "dual"),
]


def check_result_configs() -> None:
    for fname, key, expected in CONFIG_CLAIMS:
        p = ROOT / RESULTS_DIR / fname
        if not p.is_file():
            canh_bao(f"{RESULTS_DIR}/{fname} không có — bỏ qua kiểm config")
            continue
        cfg = json.loads(p.read_text()).get("config", {})
        got = cfg.get(key)
        if got != expected:
            loi(f"{RESULTS_DIR}/{fname}: config[{key}] = {got!r}, "
                f"tài liệu giả định {expected!r}")


# ── 3. Cùng đại lượng, nhiều tài liệu — có khớp không ───────────────
#: nhãn -> (regex, các file phải đồng ý)
SHARED_NUMBERS = {
    "ngưỡng phát hiện ghép cặp n=50":
        (r"\b18[.,]0\b", ("README.md", "report_en/report.tex")),
    "ngưỡng phát hiện n=1003":
        (r"\b4[.,]3\b", ("README.md", "report_en/report.tex")),
}


def check_shared_numbers() -> None:
    for label, (pat, files) in SHARED_NUMBERS.items():
        missing = [f for f in files if not re.search(pat, read(f))]
        if missing:
            canh_bao(f"'{label}': không thấy trong {', '.join(missing)} "
                     f"— có thể một tài liệu còn số cũ")


# ── 7. requirements.txt vs import thật trong mã ─────────────────────
#: import name -> tên package trên PyPI, khi hai cái khác nhau.
PKG_ALIAS = {
    "sklearn": "scikit-learn", "yaml": "PyYAML", "PIL": "Pillow",
    "rank_bm25": "rank-bm25", "huggingface_hub": "huggingface-hub",
    "dotenv": "python-dotenv", "llmlingua": "llmlingua",
    "sentence_transformers": "sentence-transformers",
    "cv2": "opencv-python", "bs4": "beautifulsoup4",
}
STDLIB_OK = set(sys.stdlib_module_names) | {"itercomp"}


def _local_modules() -> set[str]:
    """Module do chính repo định nghĩa — không phải dependency ngoài.

    `research/exp1_oracle_stopping.py` làm `from measure_stop_error import ...`
    vì hai file cùng thư mục. Đó không phải package trên PyPI.
    """
    out = {"itercomp"}
    for d in CODE_DIRS:
        out |= {f.stem for f in (ROOT / d).glob("*.py")}
    return out


def _is_optional_import(node: ast.AST, tree: ast.AST) -> bool:
    """Import nằm trong try/except ImportError -> có fallback, không bắt buộc.

    `scorer.py` thử FlagEmbedding rồi lùi về transformers thuần khi phiên bản
    không tương thích. Bắt buộc nó trong requirements là sai.
    """
    for parent in ast.walk(tree):
        if not isinstance(parent, ast.Try):
            continue
        if not any(n is node for n in ast.walk(parent.body[0].parent
                                                if False else parent)):
            continue
        for h in parent.handlers:
            if h.type is None:                       # bare except
                return True
            names = ([h.type] if isinstance(h.type, ast.Name)
                     else h.type.elts if isinstance(h.type, ast.Tuple) else [])
            for n in names:
                # Exception cũng bắt ImportError, nên nó cũng là fallback hợp lệ
                if isinstance(n, ast.Name) and n.id in {
                        "ImportError", "ModuleNotFoundError", "Exception",
                        "BaseException"}:
                    return True
    return False


def check_requirements() -> None:
    req = read("requirements.txt").lower()
    skip = STDLIB_OK | _local_modules()
    seen: set[str] = set()
    for d in CODE_DIRS:
        for f in (ROOT / d).glob("*.py"):
            try:
                tree = ast.parse(f.read_text())
            except SyntaxError:
                loi(f"{d}/{f.name}: lỗi cú pháp Python")
                continue
            for node in ast.walk(tree):
                if _is_optional_import(node, tree):
                    continue
                if isinstance(node, ast.Import):
                    seen |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    seen.add((node.module or "").split(".")[0])
    for mod in sorted(seen - skip):
        if not mod:
            continue
        pkg = PKG_ALIAS.get(mod, mod)
        if pkg.lower() not in req:
            loi(f"requirements.txt thiếu `{pkg}` (mã có `import {mod}`)")


# ── Kết quả mới mà chưa tài liệu nào nhắc ───────────────────────────
#: Tài liệu nói "chưa có / chưa hiện thực" về thứ gì, và bằng chứng ngược lại.
#: Mỗi dòng: (tài liệu, từ khoá trong tài liệu, chuỗi chứng minh đã có, file mã)
CAPABILITY_CLAIMS = [
    ("README.md", "LLMLingua", 'elif m == "llmlingua"', "scripts/run_eval.py"),
    ("README.md", "Selective-Context", 'elif m == "selective-context"',
     "scripts/run_eval.py"),
    ("README.md", "RECOMP", 'elif m == "recomp-extractive"', "scripts/run_eval.py"),
]
NOT_YET = ("chưa có", "chưa hiện thực", "repo chưa", "còn thiếu", "chưa làm")


#: Kết luận đã bị thí nghiệm sau bác bỏ. Mỗi dòng: (chuỗi cũ, file chứng minh).
OVERTURNED = [
    ("33.0 → 36.6", "exp1_oracle_vimqa_200_7b.json"),
    ("mất 3.6 điểm F1", "exp1_oracle_vimqa_200_7b.json"),
]


def check_notebooks() -> None:
    """Notebook cũng là tài liệu, và cũng lỗi thời như README.

    Notebook nộp bài từng giữ khẳng định "bộ dừng mất 3.6 F1" sau khi EXP-1 đo
    được trần là −0.17. Không script nào đọc notebook, nên không gì bắt được
    ngoài việc mở ra đọc.

    Chỉ báo nếu chuỗi cũ xuất hiện MÀ KHÔNG kèm dấu hiệu đã sửa (chữ "bác bỏ",
    "ĐÃ ĐÓNG"), để một notebook giải thích lịch sử không bị báo sai.
    """
    import json as _json
    for nb_path in sorted(ROOT.glob("notebooks/**/*.ipynb")):
        try:
            nb = _json.loads(nb_path.read_text())
        except Exception as e:
            loi(f"{nb_path.relative_to(ROOT)}: không đọc được ({type(e).__name__})")
            continue
        text = " ".join("".join(c.get("source", [])) for c in nb.get("cells", []))
        for phrase, proof in OVERTURNED:
            if phrase not in text:
                continue
            if not (ROOT / RESULTS_DIR / proof).is_file():
                continue
            if any(w in text for w in ("bác bỏ", "ĐÃ ĐÓNG", "đã lật ngược")):
                continue
            loi(f"{nb_path.relative_to(ROOT)}: còn khẳng định `{phrase}`, "
                f"nhưng {RESULTS_DIR}/{proof} đã bác bỏ")


def check_capability_claims() -> None:
    """Tài liệu nói một tính năng CHƯA có, trong khi mã đã có.

    Khác `check_stale_status`: cái đó so với file kết quả, cái này so với **mã**.
    Đã xảy ra thật — README ghi năm đường cơ sở là "repo chưa có" sau khi bốn
    trong số đó đã được hiện thực.
    """
    for doc, keyword, proof, code_file in CAPABILITY_CLAIMS:
        text, code = read(doc), read(code_file)
        if proof not in code:
            continue                      # chưa hiện thực thật -> không sao
        for line in text.splitlines():
            if keyword in line and any(w in line for w in NOT_YET):
                loi(f"{doc}: dòng nói `{keyword}` chưa có, nhưng "
                    f"{code_file} đã hiện thực ({proof})")


def check_orphan_results() -> None:
    """File kết quả mà KHÔNG tài liệu nào nhắc VÀ không script nào đọc.

    Chỉ cảnh báo khi cả hai đều không — một file được `verify_numbers` đọc thì
    đã có ràng buộc rồi, dù tài liệu không gọi tên nó. Nếu cảnh báo cả những
    file đó thì danh sách dài tới mức không ai đọc, và cảnh báo thật bị chìm.
    """
    docs = " ".join(read(f) for f in DOCS)
    code = " ".join(f.read_text() for d in CODE_DIRS
                    for f in (ROOT / d).glob("*.py") if f.is_file())
    for p in sorted((ROOT / RESULTS_DIR).glob("*.json")):
        if p.stem not in docs and p.stem not in code:
            canh_bao(f"{RESULTS_DIR}/{p.name} không tài liệu nào nhắc, "
                     f"không script nào đọc — còn cần không?")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true",
                    help="thoát mã 1 nếu có lỗi (cho CI / pre-commit)")
    args = ap.parse_args()

    for fn in (check_file_inventory, check_stale_status, check_result_configs,
               check_shared_numbers, check_requirements, check_orphan_results,
               check_capability_claims, check_notebooks):
        fn()

    errs = [m for lv, m in issues if lv == "loi"]
    warns = [m for lv, m in issues if lv == "canh_bao"]

    if errs:
        print(f"LỖI ({len(errs)}):")
        for m in errs:
            print(f"  ✗ {m}")
    if warns:
        print(f"\nCẢNH BÁO ({len(warns)}):")
        for m in warns:
            print(f"  ! {m}")
    if not issues:
        print("Không phát hiện vấn đề nào.")
    else:
        print(f"\n{len(errs)} lỗi, {len(warns)} cảnh báo.")
    return 1 if (errs and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
