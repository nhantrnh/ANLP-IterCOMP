#!/usr/bin/env bash
# Chay TAT CA phep kiem truoc khi nop. Khong chon, khong bo qua muc nao.
#
# Ly do co file nay: ba loi lien tiep trong mot phien deu vi kiem tay roi bo
# sot mot muc — doc log LaTeX, dem ky tu moi trang, doi chieu nguon goc. Mot
# lenh thi khong bo sot duoc.
set -u
cd "$(dirname "$0")/.."
fail=0
run() {
  printf '\n══ %s ══\n' "$1"; shift
  if "$@"; then :; else fail=1; fi
}
run "Số liệu trong báo cáo vs results/"  python scripts/verify_report_numbers.py
run "Tài liệu lỗi thời"                  python scripts/check_repo.py
run "Số từ lần chạy đã bị thay thế"      python scripts/check_superseded.py
run "Cỡ mẫu vs từng khẳng định"          python scripts/power_check.py --strict
run "Layout PDF"                         python scripts/check_layout.py
run "Test tính chất + hồi quy"           python tests/run_tests.py
printf '\n══ Nguồn gốc kết quả (cần mạng, chạy riêng) ══\n'
printf '  python scripts/check_provenance.py\n'
run "Runtime kernel đo được"              python scripts/kernel_runtimes.py --check
if [ "$fail" -eq 0 ]; then
  printf '\n✓ TẤT CẢ PHÉP KIỂM ĐỀU PASS\n'
else
  printf '\n✗ CÓ PHÉP KIỂM THẤT BẠI — xem ở trên\n'
fi
exit "$fail"
