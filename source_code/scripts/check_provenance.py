"""Mọi file kết quả GPU có truy được về đúng một kernel Kaggle không?

VÌ SAO CẦN. Dự án chạy 10 kernel, nhiều cái fail rồi chạy lại, và ba kernel
khác nhau cùng sinh ra file tên `vimqa_200_7b.json` với nội dung khác nhau.
Không có phép kiểm này thì không ai biết con số trong bài đến từ lần chạy nào —
và đó đúng là cách hai giá trị 50.08 / 50.17 từng gây nhầm.

Script tải output của mọi kernel `itercomp-*`, băm từng file, rồi đối chiếu với
`results/`. Báo:
  - file local KHÔNG khớp byte với kernel nào  -> nguồn không rõ
  - file Kaggle CHƯA có trong results/          -> có thể đang thiếu
  - hai kernel cùng sinh một tên file           -> phải khai trong PROVENANCE.md

Cần mạng và `kaggle` CLI. Chạy trước khi nộp, không phải mỗi lần commit.

    python scripts/check_provenance.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNELS = [
    "itercomp-gpu-tasks-567", "itercomp-damage-vs-f1-7b", "itercomp-1c-double",
    "itercomp-baselines", "itercomp-ablation-500", "itercomp-musique-500",
    "itercomp-compact", "itercomp-musique-k", "itercomp-3datasets",
    "itercomp-damage-500-cpu", "itercomp-adaptive-k-dual",
    "itercomp-damage-full-cpu", "itercomp-vimqa-full-run",
]
# File local co nguon la KERNEL nay (con lai la CPU, khong can kiem).
GPU_FILES = {
    "vimqa_full_7b.json", "vimqa_200_7b.json", "vimqa_200_7b_pinned.json",
    "exp1_oracle_vimqa_200_7b.json", "exp1_oracle_musique_200_7b.json",
    "exp2_undiacritised_200_7b.json", "damage_vs_f1_vimqa200_7b.json",
    "damage_vs_f1_vimqa_full.json", "baselines_vimqa_200_7b.json",
    "ablation_vimqa500_7b.json", "musique_500_7b.json",
    "compact_vimqa_200_7b.json", "ablation_musique200_k.json",
    "2wiki_500_7b.json", "hotpotqa_500_7b.json",
    "damage_curve_vimqa_500.json", "damage_curve_musique_500.json",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="prov_"))
    remote: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for k in KERNELS:
        d = tmp / k
        d.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output",
                        f"nhantrnh/{k}", "-p", str(d)],
                       capture_output=True, text=True)
        for z in d.glob("*.zip"):
            subprocess.run(["unzip", "-o", "-q", str(z), "-d", str(d / "x")],
                           capture_output=True)
        for j in d.rglob("*.json"):
            if ".log" in j.name or "metadata" in j.name:
                continue
            remote[sha(j)].append((k, j.name))

    problems: list[str] = []
    for name in sorted(GPU_FILES):
        loc = ROOT / "results" / name
        if not loc.exists():
            problems.append(f"THIEU: results/{name} khong ton tai")
            continue
        hit = remote.get(sha(loc))
        if not hit:
            problems.append(
                f"NGUON KHONG RO: results/{name} khong khop byte voi kernel nao")
        else:
            print(f"  OK  {name:36s} <- {hit[0][0]}")

    # ten file trung nhau giua cac kernel -> phai khai
    by_name: dict[str, set[str]] = defaultdict(set)
    for hits in remote.values():
        for k, n in hits:
            by_name[n].add(k)
    prov = (ROOT / "results" / "PROVENANCE.md")
    prov_txt = prov.read_text() if prov.exists() else ""
    for n, ks in sorted(by_name.items()):
        if len(ks) > 1 and n not in prov_txt:
            problems.append(
                f"TRUNG TEN chua khai: {n} do {len(ks)} kernel sinh ra ({sorted(ks)})")

    if problems:
        print("\n" + "\n".join("  ! " + p for p in problems))
        print(f"\n{len(problems)} van de. Xem results/PROVENANCE.md.")
        return 1
    print("\nMoi file GPU truy duoc ve dung mot kernel; ten trung deu da khai.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
