"""Kiểm layout PDF — những thứ tôi vẫn kiểm tay rồi bỏ sót.

Ba lỗi liên tiếp trong một phiên, mỗi lỗi vì kiểm thiếu một mục:

  1. Xoá `.log` sau mỗi lần dịch mà không đọc  -> 8 chỗ tràn lề, nặng nhất
     75.8pt, sống sót qua vô số lần build.
  2. Sắp lại phụ lục làm `\\end{document}` lạc lên giữa -> 3 mục rơi ra ngoài
     document. Build báo 15 trang thay vì 16, và tôi suýt đọc "giảm trang" là
     thắng lợi. **Số trang giảm tự nó không phải tin tốt.**
  3. Commit báo "17 trang, ref sạch" -> nhưng có trang chỉ 120 ký tự, một câu
     tràn sang. Đã kiểm số trang và số ref rồi dừng.

Nên script kiểm CẢ NĂM thứ một lượt, không cho chọn:
  số trang · ký tự tối thiểu mỗi trang · overfull hbox · undefined · '??'

Chạy trước khi nộp và sau mỗi lần sửa .tex:

    python scripts/check_layout.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [("report_en", "report"), ("paper_submission", "paper")]

MAX_PAGES = 18          # tran mem: vuot thi canh bao de xem lai do dai
MIN_CHARS_PER_PAGE = 1200   # duoi nguong = trang gan rong / widow


def build(d: Path, stem: str) -> str:
    """Dịch 3 lần và TRẢ VỀ log — không xoá, vì log là bằng chứng."""
    for f in (f"{stem}.aux", f"{stem}.log"):
        (d / f).unlink(missing_ok=True)
    for _ in range(3):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{stem}.tex"],
                       cwd=d, capture_output=True)
    log = (d / f"{stem}.log")
    txt = log.read_text(errors="ignore") if log.exists() else ""
    for f in (f"{stem}.aux", f"{stem}.out", f"{stem}.blg"):
        (d / f).unlink(missing_ok=True)
    log.unlink(missing_ok=True)
    return txt


def labels_before_caption(tex_path):
    """\label đặt TRƯỚC \caption trong float thì \ref ra SỐ SAI, im lặng.

    Đây là cơ chế duy nhất khiến một tham chiếu hợp lệ hiện ra sai số bảng:
    \label bắt giá trị counter tại chỗ nó đứng, mà counter chỉ tăng khi
    \caption chạy. Không có cảnh báo nào từ LaTeX, không undefined, không '??'
    — PDF vẫn dựng sạch, chỉ là bài trỏ nhầm bảng. Một phản biện từng báo lỗi
    này cho bản thảo (thực ra không có), nên từ nay có kiểm để trả lời bằng
    bằng chứng thay vì bằng lời.
    """
    import re as _re
    src = tex_path.read_text()
    bad = []
    for m in _re.finditer(r"\\begin\{(table|figure)\*?\}(.*?)\\end\{\1\*?\}", src, _re.S):
        blk = m.group(2)
        ci, li = blk.find("\\caption"), blk.find("\\label")
        if 0 <= li < ci:
            lab = _re.search(r"\\label\{([^}]*)\}", blk)
            bad.append(lab.group(1) if lab else "(không tên)")
    return bad


def main() -> int:
    bad: list[str] = []
    for sub, stem in DOCS:
        d = ROOT / sub
        if not (d / f"{stem}.tex").exists():
            print(f"bỏ qua {sub} (không có {stem}.tex)")
            continue
        log = build(d, stem)
        pdf = d / f"{stem}.pdf"
        pages = subprocess.run(["pdftotext", str(pdf), "-"],
                               capture_output=True, text=True).stdout.split("\f")
        dens = [len(p.strip()) for p in pages if p.strip()]

        over = re.findall(r"Overfull \\hbox \(([\d.]+)pt", log)
        undef = len(re.findall(r"undefined", log))
        qq = sum(p.count("??") for p in pages)
        font = len(re.findall(r"Font Warning|Missing character", log))
        misplaced = labels_before_caption(d / f"{stem}.tex")

        print(f"\n=== {sub}/{stem}.pdf ===")
        print(f"  trang                {len(dens)}")
        print(f"  ký tự / trang        min {min(dens)}  max {max(dens)}")
        print(f"  overfull hbox        {len(over)}"
              + (f"  (lớn nhất {max(float(x) for x in over):.1f}pt)" if over else ""))
        print(f"  undefined trong log  {undef}")
        print(f"  '??' trong PDF       {qq}")
        print(f"  cảnh báo font        {font}")
        print(f"  \\label trước \\caption  {len(misplaced)}")

        if over:
            bad.append(f"{sub}: {len(over)} chỗ tràn lề, lớn nhất "
                       f"{max(float(x) for x in over):.1f}pt")
        if undef:
            bad.append(f"{sub}: {undef} tham chiếu undefined")
        if qq:
            bad.append(f"{sub}: {qq} dấu '??' trong PDF")
        if font:
            bad.append(f"{sub}: {font} cảnh báo font")
        if misplaced:
            bad.append(f"{sub}: \\label đặt trước \\caption trong "
                       f"{len(misplaced)} float ({', '.join(misplaced)}) — "
                       f"\\ref sẽ ra SỐ SAI mà không báo lỗi")
        if min(dens) < MIN_CHARS_PER_PAGE:
            i = dens.index(min(dens)) + 1
            bad.append(f"{sub}: trang {i} chỉ {min(dens)} ký tự "
                       f"(widow / gần rỗng, ngưỡng {MIN_CHARS_PER_PAGE})")
        if len(dens) > MAX_PAGES:
            bad.append(f"{sub}: {len(dens)} trang, vượt {MAX_PAGES}")

    if bad:
        print("\n" + "\n".join("  ! " + b for b in bad))
        print(f"\n{len(bad)} vấn đề layout.")
        return 1
    print("\nLayout sạch: không tràn lề, không undefined, không '??', "
          "không trang gần rỗng.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
