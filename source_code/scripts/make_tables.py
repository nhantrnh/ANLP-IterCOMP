"""Sinh bảng LaTeX từ các file kết quả, dán thẳng vào báo cáo.

Mục đích: tránh chép số bằng tay. Mọi con số trong báo cáo phải truy được về một
file JSON trong results/, và script này là cầu nối duy nhất giữa hai bên.

Chạy:
    python scripts/make_tables.py                  # bảng chính + boolean
    python scripts/make_tables.py --n 200          # chỉ lấy kết quả n=200
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from itercomp import paired_bootstrap

RESULTS = Path(__file__).resolve().parents[1] / "results"
NAMES = {"raw": "Raw Document", "oracle": "Oracle",
         "llmlingua2": "LLMLingua-2", "itercomp": "IterCOMP"}
ORDER = ("hotpotqa", "2wiki", "musique", "vimqa")
LABEL = {"hotpotqa": "HotpotQA", "2wiki": "2WikiMQA",
         "musique": "MuSiQue", "vimqa": r"\textbf{VimQA (vi)}"}


def vn(x: float, d: int = 1) -> str:
    """Định dạng số theo kiểu Việt Nam (dấu phẩy thập phân)."""
    return f"{x:.{d}f}".replace(".", ",")


def _n_eff(d: dict) -> int:
    """Cỡ mẫu THẬT của một file kết quả.

    config.limit là None khi chạy toàn bộ tập, nên không dùng trực tiếp để so
    sánh được. Đếm per_row là cách duy nhất đúng cho cả hai trường hợp.
    """
    rows = d.get("per_row")
    if isinstance(rows, list) and rows:
        return len(rows)
    lim = (d.get("config") or {}).get("limit")
    return lim or 0


def _reader_rank(d: dict) -> int:
    """Xếp hạng mô hình đọc: 7B/8B > 1.5B > 0.5B > mock.

    Cần thiết vì cùng một cỡ mẫu có thể có nhiều file với reader khác nhau, và
    báo cáo phải lấy số của mô hình lớn nhất — trộn hai reader vào một bảng là
    so sánh lệch.
    """
    m = str((d.get("config") or {}).get("reader_model") or "").lower()
    for i, tag in enumerate(("mock", "0.5b", "1.5b", "3b", "7b", "8b")):
        if tag in m:
            return i
    return 1


def load(n: int | None) -> dict:
    """Nạp kết quả cho từng bộ dữ liệu.

    n=None nghĩa là "lấy tốt nhất có sẵn", không phải "lấy file bất kỳ": chọn
    theo (cỡ mẫu thật, hạng reader) giảm dần. Bản trước lặp qua danh sách đã
    sort theo TÊN và ghi đè `pick` mỗi vòng, nên nó lấy file cuối alphabet —
    tức vimqa_50_hf05b_dual.json (reader 0.5B, n=50) thắng cả vimqa_200.json.
    Với run full sắp tới thì lỗi này sẽ đưa số n=50 vào báo cáo.
    """
    out = {}
    for ds in ORDER:
        best = None
        for f in sorted(glob.glob(str(RESULTS / f"{ds}_*.json"))):
            if "ablation" in f or "lambda" in f or "exp" in f:
                continue
            try:
                d = json.load(open(f, encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if "summary" not in d:
                continue
            if n is not None and _n_eff(d) != n:
                continue
            # Hạng reader TRƯỚC cỡ mẫu: cùng bộ dữ liệu, n=200 với reader 0.5B
            # cho F1 24,8 còn n=50 với 7B cho 43,0 — chênh do MÔ HÌNH, không do
            # cỡ mẫu. Trộn hai reader vào một bảng là so sánh lệch, nên chọn
            # reader lớn nhất trước, rồi trong cùng reader mới lấy n lớn nhất.
            key = (_reader_rank(d), _n_eff(d))
            if best is None or key > best[0]:
                best = (key, d, f)
        if best:
            out[ds] = best[1]
            out[ds]["_source_file"] = Path(best[2]).name
            out[ds]["_n_eff"] = _n_eff(best[1])
    return out


def table_main(data: dict, decimal_comma: bool = True) -> str:
    """Bảng chính: F1 sau chuẩn hoá và tỉ lệ nén, mọi bộ dữ liệu."""
    order = [d for d in ORDER if d in data]
    fmt = vn if decimal_comma else (lambda x, d=1: f"{x:.{d}f}")
    lines = [r"\begin{tabular}{l" + "rr" * len(order) + "}", r"\toprule",
             r"\multirow{2}{*}{Phương pháp}" + "".join(
                 r" & \multicolumn{2}{c}{%s}" % LABEL[d] for d in order) + r" \\"]
    lines.append("".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i)
                         for i in range(len(order))))
    lines.append(" " + " & ".join(["", *["F1 & Tỉ lệ"] * len(order)]) + r" \\")
    lines.append(r"\midrule")
    for m, label in NAMES.items():
        cells = []
        for d in order:
            v = data[d]["summary"].get(m)
            # f1_norm chỉ có ở các file sinh sau khi thêm chuẩn hoá boolean;
            # file cũ hơn chỉ có f1. Lùi về f1 thay vì để KeyError giết cả bảng,
            # vì trên tiếng Anh hai giá trị này bằng nhau (phép chuẩn hoá không
            # bao giờ khớp) nên số vẫn đúng.
            f1 = v.get("f1_norm", v.get("f1")) if v else None
            cells.append(f"{fmt(f1)} & {fmt(v['ratio'] * 100)}\\%"
                         if f1 is not None else "-- & --")
        row = (r"\textbf{IterCOMP} (tái hiện)" if m == "itercomp" else
               r"Oracle (\textit{trần})" if m == "oracle" else label)
        lines.append(f"{row} & " + " & ".join(cells) + r" \\")
        if m == "oracle":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_boolean(data: dict, ds: str = "vimqa") -> str:
    """Bảng tác động của chuẩn hoá đáp án đúng/sai, kèm kiểm định ghép cặp."""
    if ds not in data or "per_row" not in data[ds]:
        return "% chưa có dữ liệu per_row cho " + ds
    rows = data[ds]["per_row"]
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Phương pháp & F1 thô & F1$^*$ & Chênh \\", r"\midrule"]
    for m, label in NAMES.items():
        raw = [r["methods"][m]["f1"] * 100 for r in rows if m in r["methods"]]
        nor = [r["methods"][m]["f1_norm"] * 100 for r in rows if m in r["methods"]]
        if not raw:
            continue
        t = paired_bootstrap(nor, raw)
        star = r"$^\dagger$" if t["p"] < 0.05 else ""
        bold = r"\textbf{%s}" % vn(t["diff"]) if m == "itercomp" else vn(t["diff"])
        lines.append(f"{label} & {vn(sum(raw)/len(raw))} & {vn(sum(nor)/len(nor))} "
                     f"& ${'+' if t['diff'] >= 0 else ''}{bold}${star} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def significance(data: dict, ds: str = "vimqa") -> str:
    """Các phép so sánh ghép cặp, để trích dẫn trong phần văn."""
    if ds not in data or "per_row" not in data[ds]:
        return ""
    rows = data[ds]["per_row"]
    ic = [r["methods"]["itercomp"]["f1_norm"] * 100 for r in rows]
    out = []
    for m in ("raw", "oracle", "llmlingua2"):
        o = [r["methods"][m]["f1_norm"] * 100 for r in rows if m in r["methods"]]
        if not o:
            continue
        t = paired_bootstrap(ic, o)
        mark = "có ý nghĩa" if t["p"] < 0.05 else "chưa đủ"
        out.append(f"  itercomp − {m:<11}{t['diff']:+7.1f}  "
                   f"[{t['lo']:+6.1f},{t['hi']:+6.1f}]  p={t['p']:.4f}  {mark}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Sinh bảng LaTeX từ results/")
    ap.add_argument("--n", type=int, default=None, help="chỉ lấy kết quả cỡ mẫu này")
    ap.add_argument("--dataset", default="vimqa", help="bộ dữ liệu cho bảng boolean")
    args = ap.parse_args()

    data = load(args.n)
    if not data:
        print("Chưa có kết quả nào trong results/")
        return

    desc = ", ".join("%s(n=%s)" % (k, (data[k].get("config") or {}).get("limit"))
                     for k in data)
    print("Nạp được:", desc, "\n")
    print("% ════════ BẢNG CHÍNH ════════")
    print(table_main(data))
    print(f"\n% ════════ CHUẨN HOÁ BOOLEAN ({args.dataset}) ════════")
    print(table_boolean(data, args.dataset))
    print(f"\n% ════════ KIỂM ĐỊNH GHÉP CẶP ({args.dataset}) ════════")
    print(significance(data, args.dataset))


if __name__ == "__main__":
    main()
