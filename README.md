# Đề tài 32 — IterCOMP cho Hỏi–đáp đa bước tiếng Việt

Thực nghiệm lại phương pháp nén prompt **IterCOMP** (Yun & Kim, ACL 2026) và mở
rộng đánh giá sang **tiếng Việt** trên bộ **VimQA**. Môn *Xử lý Ngôn ngữ Tự nhiên
Nâng cao*, VNU-HCM University of Science.

> Bài báo gốc không công bố mã nguồn và prompt (chỉ công bố siêu tham số, Mục 5.1).
> Prompt *answerability* và *follow-up* do nhóm tự thiết kế; kết quả nhằm kiểm
> chứng **xu hướng tương đối** giữa các phương pháp, không so tuyệt đối với paper.

## Mục lục
1. [Bài nộp gồm những gì](#1-bài-nộp-gồm-những-gì)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Chạy nhanh](#3-chạy-nhanh) — [Kaggle](#31-kaggle) · [Colab](#32-google-colab) · [Local macOS/Linux](#33-local--macos--linux) · [Local Windows](#34-local--windows)
4. [Dữ liệu và mô hình](#4-dữ-liệu-và-mô-hình)
5. [Kết quả chính](#5-kết-quả-chính)
6. [Tính tái lập](#6-tính-tái-lập)

---

## 1. Bài nộp gồm những gì

| Yêu cầu | Vị trí |
|---|---|
| Báo cáo PDF (ACL short paper) | [`report.pdf`](report.pdf) — §1–9 chiếm 5 trang, References từ trang 6, 6 phụ lục (A–F), tổng 12 trang |
| Mã nguồn | [`src/`](src/) — `itercomp/` (cài đặt), `scripts/`, `requirements.txt`, `run.sh` |
| Jupyter Notebook (Colab/Kaggle) | [`notebooks/`](notebooks/) — `colab_reproduce.ipynb` (nộp) + kernel GPU từng thí nghiệm |
| README cách chạy | tệp này (Mục 3) |
| Đường dẫn tải dữ liệu | [`DATASET_LINKS.txt`](DATASET_LINKS.txt) — tự tải từ HuggingFace, không kèm data 116 MB |
| Mô hình / đường dẫn | [`MODEL_LINKS.txt`](MODEL_LINKS.txt) — **training-free, không có mô hình huấn luyện**; mọi mô hình tải tự động từ HuggingFace |

Báo cáo dùng định dạng ACL chính thức ([acl-org/acl-style-files](https://github.com/acl-org/acl-style-files)).

## 2. Cấu trúc thư mục

```
.
├── README.md · MODEL_LINKS.txt · DATASET_LINKS.txt · report.pdf
├── src/
│   ├── itercomp/       Cài đặt IterCOMP (core, scorer, reader, metrics, …)
│   ├── scripts/        run_eval.py (bảng chính), ablation, EXP-1/2, …
│   ├── results/        Kết quả JSON backing mọi con số trong báo cáo
│   ├── requirements.txt · run.sh
├── notebooks/
│   ├── colab_reproduce.ipynb   notebook nộp (đủ 8 phần, Colab/Kaggle)
│   └── kaggle/                 15 kernel GPU của từng thí nghiệm
```

## 3. Chạy nhanh

Notebook `notebooks/colab_reproduce.ipynb` tự tải mã nguồn, dữ liệu, mô hình —
**không phụ thuộc đường dẫn cục bộ**, đủ 8 phần (cài đặt → tải dữ liệu → EDA →
cấu hình mô hình → đánh giá → so mô hình cơ sở → phân tích lỗi → demo dữ liệu
tiếng Việt mới). Đây là cách khuyến nghị.

### 3.1. Kaggle
1. **Add → Upload Notebook** → chọn `notebooks/colab_reproduce.ipynb`.
2. Panel phải: **Settings → Accelerator → `GPU T4 x2`** và **Internet: On**.
3. **Run All**. Kết quả + bảng in ra trong notebook; tải về ở ô cuối.

### 3.2. Google Colab
1. colab.research.google.com → **File → Upload notebook** → `colab_reproduce.ipynb`.
2. **Runtime → Change runtime type → T4 GPU → Save**.
3. **Runtime → Run all**.

### 3.3. Local — macOS / Linux
Cần GPU cho mô hình đọc (Qwen2.5-7B, 4-bit). Phần CPU (fertility, damage) chạy được không cần GPU.

```bash
cd src && ./run.sh   # tạo .venv, cài thư viện, tải dữ liệu, chạy đánh giá, sinh bảng
# kiểm thử luồng (không tốn tài nguyên, không cần GPU):
python src/scripts/run_eval.py --dataset vimqa --limit 5 --reader mock --itercomp-llm mock
```

### 3.4. Local — Windows
`run.sh` là bash; trên Windows chạy thủ công (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r src\requirements.txt
# tải dữ liệu: xem DATASET_LINKS.txt (hoặc chạy các lệnh curl trong đó)
# kiểm thử luồng:
python src\scripts\run_eval.py --dataset vimqa --limit 5 --reader mock --itercomp-llm mock
# chạy thật (cần GPU): thay --reader mock bằng --reader hf --reader-model Qwen/Qwen2.5-7B-Instruct --load-4bit
```

> Notebook ghim `transformers==4.45.2` (Colab/Kaggle nay ship 5.0.0 xung đột cứng
> với `llmlingua 0.2.2`), cũng là phiên bản sinh ra mọi số trong báo cáo.

## 4. Dữ liệu và mô hình

- **Dữ liệu** (không kèm zip, 116 MB) — tải tự động, xem [`DATASET_LINKS.txt`](DATASET_LINKS.txt):
  VimQA (tiếng Việt), HotpotQA, 2WikiMultiHopQA, MuSiQue-Ans.
- **Mô hình** — [`MODEL_LINKS.txt`](MODEL_LINKS.txt). IterCOMP **training-free**:
  encoder `bge-m3`, compressor `LLMLingua-2`, reader `Qwen2.5` đều tải từ HuggingFace.

## 5. Kết quả chính

F1$^*$ (chuẩn hoá boolean), reader Qwen2.5-7B 4-bit — bảng đầy đủ ở §6 báo cáo:

| | VimQA (vi) | MuSiQue | 2Wiki | HotpotQA |
|---|---|---|---|---|
| | n=1003 | n=500 | n=1500 | n=1500 |
| Oracle | 61.03 | 49.84 | 57.65 | 71.40 |
| Raw | 46.87 | 23.43 | 35.72 | 59.24 |
| **IterCOMP** | **51.29** | **26.74** | **36.72** | 54.58 |
| LLMLingua-2 | 35.31 | 15.14 | 19.24 | 36.00 |
| IterCOMP − Raw | +4.42† | +3.31 | +1.00 | −4.66† |
| IterCOMP − LLMLingua-2 | +15.98† | +11.60† | +17.48† | +18.58† |

† khoảng tin cậy 95% (paired bootstrap) không chứa 0.

- **Khẳng định trung tâm (nén câu > nén token) đúng cả bốn** (+11.6 đến +18.6 F1,
  mọi khoảng tách 0).
- **Thứ tự đầy đủ phụ thuộc reader:** reproduces trên VimQA/MuSiQue, nhưng trên
  HotpotQA **Raw vượt IterCOMP** (−4.66, có ý nghĩa) — reader mạnh đọc ngữ cảnh dài
  tốt nên lợi thế *chính xác* của nén mất, chỉ còn lợi thế *chi phí*.
- **Tiếng Việt:** chuẩn hoá boolean +10.0 F1 (0.0 ở tiếng Anh); fertility làm tỉ lệ
  nén đánh giá thấp chi phí token thật 2.39×; văn bản không dấu — nén thắng cả gold.

## 6. Tính tái lập

- Greedy decoding → **kết quả tất định**; chạy lại cùng cấu hình cho cùng số.
- Mọi con số trong báo cáo truy nguồn về `src/results/` (JSON đầy đủ per-row).
- Thực nghiệm GPU chạy trên Kaggle T4; kernel lưu ở `notebooks/kaggle/`.
