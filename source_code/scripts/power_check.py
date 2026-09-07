#!/usr/bin/env python3
"""Mỗi khẳng định trong báo cáo cần bao nhiêu câu hỏi?

Câu hỏi "n=500 trên 12,576 có ít quá không" không trả lời được ở mức dataset.
Cỡ mẫu không đủ hay thừa một cách trừu tượng — nó đủ *cho một chênh lệch cụ
thể*. 2Wiki là ví dụ rõ: cùng n=500, khẳng định trung tâm của paper (+16.84)
chỉ cần 37 câu, còn `IterCOMP − Raw` (−2.69) cần 1,304.

Nên script này không hỏi "n có lớn không" mà hỏi từng cặp phương pháp: với bề
rộng CI ghép cặp quan sát được, cần bao nhiêu câu để chênh lệch này rời khỏi 0?

    n_cần ≈ n × (nửa bề rộng CI / |chênh lệch|)²

Dùng chính bề rộng CI ghép cặp mà báo cáo công bố, chứ không dùng phương sai
không ghép cặp của `required_n` — hai phương pháp chấm cùng một câu hỏi tương
quan rất mạnh, nên bỏ ghép cặp đi sẽ đòi cỡ mẫu lớn hơn thực tế nhiều lần.

Khẳng định nào CI còn cắt qua 0 thì phải được KHAI trong UNRESOLVED kèm lý do,
giống cơ chế của check_superseded.py — để không ai lặng lẽ để lại một lỗ hổng.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from itercomp.stats import paired_bootstrap  # noqa: E402

# Các cặp báo cáo thật sự công bố. Thứ tự (a, b) = hiệu a − b.
CLAIMS = [
    ("itercomp", "llmlingua2"),   # khẳng định trung tâm: câu > token
    ("itercomp", "raw"),
    ("oracle", "raw"),
]

# dataset -> file kết quả lớn nhất hiện có
RUNS = {
    "vimqa":    "vimqa_full_7b.json",
    "musique":  "musique_500_7b.json",
    "2wiki":    "2wiki_1500_7b.json",
    "hotpotqa": "hotpotqa_1500_7b.json",
}

# Khẳng định CI còn cắt qua 0, kèm lý do vì sao chấp nhận để vậy.
# Khoá: (dataset, a, b)
UNRESOLVED: dict[tuple[str, str, str], str] = {
    ("musique", "itercomp", "raw"):
        "+3.31 F1 ở n=500 (bảng 4ds giữ n=500 làm thang so-với-paper). Đã KHÉP ở "
        "n=1500: +4.19 F1, CI [+2.01,+6.32] sạch (results/musique_1500_7b.json, "
        "§5). Giữ n=500 trong bảng vì §6.2/matched-compression/app:mq500 đều dùng "
        "n=500 để đối chiếu paper.",
    ("2wiki", "itercomp", "raw"):
        "+1.00 F1 ở n=1500 (đã chạy; n=500 cho −2.69, tức ĐỔI DẤU). Bề rộng CI "
        "thu từ ±4.35 xuống ±2.60 nhưng hiệu cũng nhỏ đi, nên vẫn cắt 0 — và "
        "đó chính là kết luận: hai phương pháp không phân biệt được trên 2Wiki, "
        "chứ không phải thiếu mẫu. Tách được +1.00 cần n≈10.000, gấp gần toàn "
        "tập 12.576, để đo một hiệu nhỏ hơn nhiễu đọc; không đáng.",
}


def f1s(rows: list[dict], method: str) -> list[float]:
    return [100 * r["methods"][method]["f1_norm"] for r in rows]


def main() -> int:
    strict = "--strict" in sys.argv
    problems: list[str] = []
    unused = set(UNRESOLVED)

    for ds, fname in RUNS.items():
        path = ROOT / "results" / fname
        if not path.exists():
            # Lùi về lần chạy nhỏ hơn nếu file lớn chưa có, và nói rõ.
            alt = ROOT / "results" / fname.replace("_1500_", "_500_")
            if not alt.exists():
                print(f"  {ds:9s} THIẾU {fname}")
                problems.append(f"{ds}: không có {fname}")
                continue
            path = alt
        data = json.loads(path.read_text())
        rows = data["per_row"]
        n = len(rows)
        print(f"\n  {ds}  n={n}  ({path.name})")
        for a, b in CLAIMS:
            if a not in rows[0]["methods"] or b not in rows[0]["methods"]:
                continue
            r = paired_bootstrap(f1s(rows, a), f1s(rows, b))
            diff, lo, hi = r["diff"], r["lo"], r["hi"]
            clear = lo * hi > 0
            half = (hi - lo) / 2
            need = round(n * (half / abs(diff)) ** 2) if diff else -1
            tag = "sạch" if clear else "CẮT QUA 0"
            print(f"    {a:9s}−{b:11s} {diff:+7.2f}  "
                  f"[{lo:+7.2f},{hi:+7.2f}]  {tag:10s} n cần ≈ {need:,}")
            key = (ds, a, b)
            if not clear:
                if key in UNRESOLVED:
                    print(f"      đã khai: {UNRESOLVED[key]}")
                    unused.discard(key)
                else:
                    problems.append(
                        f"{ds}: {a}−{b} = {diff:+.2f} còn cắt qua 0 "
                        f"(cần n≈{need:,}, đang có {n}) và CHƯA khai trong UNRESOLVED"
                    )
            elif key in UNRESOLVED:
                unused.discard(key)
                problems.append(
                    f"{ds}: {a}−{b} nay đã SẠCH — xoá khỏi UNRESOLVED "
                    f"(lý do cũ: {UNRESOLVED[key]})"
                )

    for key in sorted(unused):
        problems.append(f"khai thừa trong UNRESOLVED: {key} — không khớp cặp nào")

    print()
    if problems:
        for p in problems:
            print(f"  ! {p}")
        print(f"\n{len(problems)} vấn đề.")
        return 1 if strict else 0
    print("Mọi khẳng định đều sạch, hoặc đã khai kèm lý do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
