#!/usr/bin/env bash
# Chạy toàn bộ pipeline bằng MỘT lệnh:  ./run.sh
#
# Tự lo: tạo venv, cài thư viện, tải dữ liệu, chạy đánh giá, sinh bảng LaTeX.
# Mặc định dùng mô hình đọc nhỏ và n=20 để chạy được trên máy cá nhân không GPU
# trong vài phút. Muốn số thật như báo cáo thì xem phần CẤU HÌNH bên dưới.
#
#   ./run.sh                    # kiểm tra nhanh, CPU, ~5 phút
#   READER_SIZE=7b ./run.sh     # số như báo cáo, cần GPU >=12 GB
#   LIMIT=200 ./run.sh          # đổi cỡ mẫu
#   LIMIT=full ./run.sh         # toàn bộ tập dev
#   METHODS=... ./run.sh        # thêm đường cơ sở, xem biến METHODS bên dưới
set -euo pipefail
cd "$(dirname "$0")"

# ─────────────────────────────────────────────────────────── CẤU HÌNH
READER_SIZE="${READER_SIZE:-0.5b}"   # 0.5b | 1.5b | 7b
LIMIT="${LIMIT:-20}"                 # số câu, hoặc 'full'
DATASETS="${DATASETS:-vimqa}"        # cách nhau bằng dấu cách
# Mặc định 4 phương pháp của bài báo. Đặt METHODS để thêm đường cơ sở khác:
#   METHODS=raw,oracle,llmlingua2,llmlingua,longllmlingua,selective-context,\
#           recomp-extractive,recomp-abstractive,itercomp ./run.sh
# Lưu ý: mỗi baseline thêm vào là một mô hình nữa phải nạp, nên n lớn thì tốn
# đáng kể — llmlingua đo được 25 s/câu trên CPU.
METHODS="${METHODS:-raw,oracle,llmlingua2,itercomp}"

case "$READER_SIZE" in
  0.5b) READER="Qwen/Qwen2.5-0.5B-Instruct" ;;
  1.5b) READER="Qwen/Qwen2.5-1.5B-Instruct" ;;
  7b)   READER="Qwen/Qwen2.5-7B-Instruct"   ;;
  *) echo "READER_SIZE phải là 0.5b, 1.5b hoặc 7b"; exit 1 ;;
esac

PY=python3
VENV=.venv

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# ─────────────────────────────────────────────────── 1. môi trường
say "1/5  Môi trường"
# Dùng venv đang hoạt động nếu có, rồi mới đến ./.venv, rồi ../.venv.
# Tạo mới là lựa chọn cuối: `python -m venv` có thể thất bại ở
# `ensurepip` trên một số bản cài Python, và khi đó thông báo mặc định
# không nói người dùng phải làm gì.
if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "dùng venv đang hoạt động: $VIRTUAL_ENV"
elif [ -d "$VENV" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
elif [ -d "../$VENV" ]; then
  echo "dùng ../$VENV"
  # shellcheck disable=SC1091
  source "../$VENV/bin/activate"
else
  echo "Tạo $VENV ..."
  if ! $PY -m venv "$VENV" 2>/dev/null; then
    cat >&2 <<'MSG'
Không tạo được venv. Hai cách:
  1. Tự tạo rồi chạy lại:   python3 -m venv .venv && source .venv/bin/activate
  2. Cài vào Python hiện tại: pip install -r requirements.txt
Rồi chạy lại ./run.sh — nó sẽ dùng môi trường đang hoạt động.
MSG
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt
echo "✓ thư viện đã sẵn sàng"

# 4-bit chỉ chạy được trên CUDA; tự tắt ở nơi khác thay vì để lỗi giữa chừng.
FOURBIT=""
if [ "$READER_SIZE" = "7b" ]; then
  if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python -m pip install -q "bitsandbytes>=0.43" && FOURBIT="--load-4bit"
    echo "✓ CUDA: bật 4-bit cho reader 7B"
  else
    echo "! Không có CUDA — 7B sẽ chạy fp32 trên CPU, RẤT chậm."
    echo "  Cân nhắc: READER_SIZE=0.5b ./run.sh"
  fi
fi

# ─────────────────────────────────────────────────── 2. dữ liệu
say "2/5  Dữ liệu"
python - <<'PY'
import os, urllib.request
FILES = {
 'vimqa/validation.parquet':    ('nguyenlab/vimqa', 'data/validation-00000-of-00001.parquet'),
 '2wiki/dev.parquet':           ('xanhho/2WikiMultihopQA', 'dev.parquet'),
 'hotpotqa/validation.parquet': ('hotpotqa/hotpot_qa', 'distractor/validation-00000-of-00001.parquet'),
 'musique/dev.jsonl':           ('dgslibisey/MuSiQue', 'musique_ans_v1.0_dev.jsonl'),
}
for dst, (repo, src) in FILES.items():
    p = 'data/' + dst
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.exists(p):
        print(f'  ✓ {dst}')
        continue
    print(f'  ↓ {dst} ...', flush=True)
    urllib.request.urlretrieve(
        f'https://huggingface.co/datasets/{repo}/resolve/main/{src}', p)
PY

# ─────────────────────────────────────────────────── 3. kiểm tra
say "3/5  Kiểm tra môi trường"
python scripts/smoke_test.py

# ─────────────────────────────────────────────────── 4. đánh giá
say "4/5  Đánh giá  (reader=$READER_SIZE, n=$LIMIT)"
LIMIT_ARG=()
[ "$LIMIT" != "full" ] && LIMIT_ARG=(--limit "$LIMIT")
TAG="${READER_SIZE//./}"

for ds in $DATASETS; do
  out="results/${ds}_${LIMIT}_${TAG}.json"
  if [ -f "$out" ]; then
    echo "  [bỏ qua] $out đã có"
    continue
  fi
  echo "  → $ds"
  python scripts/run_eval.py --dataset "$ds" "${LIMIT_ARG[@]}" \
    --reader hf --reader-model "$READER" $FOURBIT \
    --itercomp-llm hf --methods "$METHODS" \
    --scorer dual --out "$out"
done

# ─────────────────────────────────────────────────── 5. bảng
say "5/5  Bảng LaTeX"
python scripts/make_tables.py

say "XONG"
echo "Kết quả JSON: results/"
