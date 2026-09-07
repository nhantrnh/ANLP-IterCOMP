#!/usr/bin/env python3
"""Ghi thời gian chạy ĐO ĐƯỢC của từng kernel Kaggle ra results/.

Đoạn "Compute cost" của báo cáo từng ghi ước lượng, bị phát hiện là đoán thấp
một phần ba, nên được sửa thành số đo từ log. Rồi hai kernel mới chạy sau đó và
con số 16 GPU-hours lại thành sai — vẫn nằm dưới câu "measured rather than
estimated". Tự nhận là đo mà lại lệch thì tệ hơn là ghi ước lượng.

Nên số giờ không được gõ tay vào .tex nữa. Script này tải log của mọi kernel
`itercomp-*`, lấy mốc thời gian lớn nhất trong log (Kaggle ghi `time` là giây
kể từ lúc kernel khởi động) và ghi ra results/kernel_runtimes.json để
verify_report_numbers.py neo vào.

    python scripts/kernel_runtimes.py          # cập nhật file
    python scripts/kernel_runtimes.py --check  # fail nếu file lệch thực tế

Kernel CPU tách riêng: chúng KHÔNG tiêu quota GPU, nên gộp vào tổng GPU là sai.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "kernel_runtimes.json"

# Kernel chạy trên CPU — không tiêu quota GPU.
CPU_KERNELS = {"itercomp-1c-double", "itercomp-damage-500-cpu"}

KERNELS = [
    "itercomp-damage-vs-f1-7b", "itercomp-gpu-tasks-567", "itercomp-1c-double",
    "itercomp-baselines", "itercomp-ablation-500", "itercomp-musique-500",
    "itercomp-compact", "itercomp-musique-k", "itercomp-3datasets",
    "itercomp-damage-500-cpu", "itercomp-2wiki-1500", "itercomp-hotpotqa-readers", "itercomp-hotpotqa-1500",
    "itercomp-musique-1500",
]


def runtime_hours(kernel: str, workdir: Path) -> float | None:
    """Số giờ wall-clock của kernel, hoặc None nếu chưa chạy / không có log."""
    dest = workdir / kernel
    r = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "output",
         f"nhantrnh/{kernel}", "-p", str(dest), "-q"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return None
    logs = sorted(dest.glob("*.log"))
    if not logs:
        return None
    try:
        entries = json.loads(logs[0].read_text())
    except json.JSONDecodeError:
        return None
    times = [e["time"] for e in entries
             if isinstance(e, dict) and e.get("time") is not None]
    return max(times) / 3600 if times else None


def measure() -> dict:
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        per = {}
        for k in KERNELS:
            h = runtime_hours(k, work)
            if h is not None:
                per[k] = round(h, 2)
    gpu = round(sum(v for k, v in per.items() if k not in CPU_KERNELS), 1)
    cpu = round(sum(v for k, v in per.items() if k in CPU_KERNELS), 1)
    return {
        "note": "Wall-clock đo từ log Kaggle. Sinh bởi scripts/kernel_runtimes.py "
                "— đừng sửa tay, và đừng gõ số giờ trực tiếp vào .tex.",
        "per_kernel_hours": dict(sorted(per.items(), key=lambda kv: -kv[1])),
        "gpu_hours_total": gpu,
        "cpu_hours_total": cpu,
        "cpu_kernels": sorted(CPU_KERNELS),
    }


def main() -> int:
    fresh = measure()
    if "--check" in sys.argv:
        if not OUT.exists():
            print(f"  ! thiếu {OUT.relative_to(ROOT)}")
            return 1
        old = json.loads(OUT.read_text())
        diffs = [
            f"{k}: file {old['per_kernel_hours'].get(k, '—')}h vs đo {v}h"
            for k, v in fresh["per_kernel_hours"].items()
            if old["per_kernel_hours"].get(k) != v
        ]
        gone = set(old["per_kernel_hours"]) - set(fresh["per_kernel_hours"])
        diffs += [f"{k}: có trong file, không đo lại được" for k in sorted(gone)]
        if diffs:
            for d in diffs:
                print(f"  ! {d}")
            print("\nChạy `python scripts/kernel_runtimes.py` để cập nhật.")
            return 1
        print(f"Runtime khớp: {fresh['gpu_hours_total']}h GPU "
              f"+ {fresh['cpu_hours_total']}h CPU trên "
              f"{len(fresh['per_kernel_hours'])} kernel.")
        return 0

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(fresh, ensure_ascii=False, indent=2) + "\n")
    for k, v in fresh["per_kernel_hours"].items():
        tag = " (CPU)" if k in CPU_KERNELS else ""
        print(f"  {k:28s} {v:5.2f} h{tag}")
    print(f"\n  GPU {fresh['gpu_hours_total']} h   CPU {fresh['cpu_hours_total']} h"
          f"   -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
