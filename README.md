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
| Báo cáo PDF (ACL short paper) | [`report.pdf`](report.pdf) — §1–9 chiếm 5 trang, References từ trang 6, 6 phụ lục (A–F), tổng 13 trang; tiêu đề *Beyond the English Baseline: IterCOMP-X for Vietnamese Multi-Hop QA* |
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
│   ├── colab_reproduce.ipynb   notebook DUY NHẤT, dựng như slide (mục I-X theo thứ tự báo cáo) + demo, ~25 phút trên T4 (Colab/Kaggle)
│   └── kaggle/                 15 kernel GPU của từng thí nghiệm
```

## 3. Chạy nhanh

Notebook `notebooks/colab_reproduce.ipynb` tự tải mã nguồn, dữ liệu, mô hình —
**không phụ thuộc đường dẫn cục bộ**, đủ 8 phần (cài đặt → tải dữ liệu → EDA →
cấu hình mô hình → đánh giá → so mô hình cơ sở → phân tích lỗi → demo dữ liệu
tiếng Việt mới). Đây là cách khuyến nghị. Mặc định (`SCALE='demo'`, VimQA n=30)
chạy hết trong **~25 phút** trên T4 miễn phí; số ở quy mô báo cáo được đọc từ
`src/results/*.json`, không chạy lại. Ba công tắc ở ô đầu: `MODE`
(`live`/`cached`/`mock`, tự chọn theo GPU), `SCALE` (`demo`/`paper`/`max`),
`RUN_EXTRAS` (ablation, EXP-1/2, quét λ/k — mặc định tắt).

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

F1$^*$ (chuẩn hoá boolean), reader Qwen2.5-7B 4-bit — bảng đầy đủ ở §5.1 báo cáo:

| | VimQA (vi) | MuSiQue | 2Wiki | HotpotQA |
|---|---|---|---|---|
| | n=1003 | n=1500 | n=1500 | n=1500 |
| Oracle | 61.03 | 44.90 | 57.65 | 71.40 |
| Raw | 46.87 | 21.16 | 35.72 | **59.24** |
| **IterCOMP-X** | **51.29** | **25.35** | **36.72** | 54.58 |
| LLMLingua-2 | 35.31 | 14.63 | 19.24 | 36.00 |
| IterCOMP-X − Raw | +4.42† | +4.19† | +1.00 | −4.66† |
| IterCOMP-X − LLMLingua-2 | +15.98† | +10.72† | +17.48† | +18.58† |
| Ratio nominal / effective | 0.206 / 0.491 | 0.340 | 0.354 | 0.194 |

† khoảng tin cậy 95% (paired bootstrap) không chứa 0.

- **Nén câu > nén token đúng cả bốn** (+10.72 đến +18.58 F1, mọi khoảng tách 0).
- **Thứ tự đầy đủ phụ thuộc reader:** reproduces trên VimQA/MuSiQue, nhưng trên
  HotpotQA **Raw vượt IterCOMP-X** (−4.66, có ý nghĩa) — reader mạnh đọc ngữ cảnh dài
  tốt nên lợi thế *chính xác* của nén mất, chỉ còn lợi thế *chi phí*.
- **Effective ratio (đóng góp chính):** tiếng Việt tốn 2.39× token của tiếng Anh
  (`cl100k_base`), nên tỉ lệ nén danh nghĩa đánh giá thấp chi phí thật; 0.206 → 0.491.
- **Tiếng Việt:** chuẩn hoá boolean +8.67 F1 (0.0 ở tiếng Anh); bỏ dấu làm mất khoảng
  20 F1 ở cả gold lẫn ngữ cảnh nén (gold vẫn dẫn ở n=500); IterCOMP-X đạt 84% Oracle
  trên VimQA, cao hơn mọi bộ tiếng Anh.

## 6. Tính tái lập

- Greedy decoding → **kết quả tất định**; chạy lại cùng cấu hình cho cùng số.
- Mọi con số trong báo cáo truy nguồn về `src/results/` (JSON đầy đủ per-row).
- Thực nghiệm GPU chạy trên Kaggle T4; kernel lưu ở `notebooks/kaggle/`.

## Demo

Demo trực tiếp **một câu tiếng Anh (HotpotQA #277) và một câu tiếng Việt (VimQA #320)** qua cùng vòng lặp IterCOMP (điểm segment ở từng pass,
phán đoán "đủ bằng chứng", câu hỏi phụ, `P_comp`, đáp án + token của Raw / Oracle /
LLMLingua-2 / IterCOMP, tỉ lệ nén danh nghĩa so với hiệu dụng, chuẩn hoá boolean,
văn bản không dấu; VI.3c đặt hai ngôn ngữ cạnh nhau) nằm ở mục **VI.5-VI.6** của **cùng một notebook**
[`notebooks/colab_reproduce.ipynb`](notebooks/colab_reproduce.ipynb) (~5 phút sau khi
mô hình đã nạp; `MODE='live'` chạy thật, `'cached'` phát lại vết đã lưu — tự chọn khi
không có GPU, `'mock'` chỉ kiểm tra luồng). Nếu chạy thật lỗi, ô tự lùi về vết đã lưu.
Mã vết: [`src/scripts/demo_trace.py`](src/scripts/demo_trace.py), cũng chạy được riêng:

```bash
# Phát lại vết ghi sẵn — tức thì, không cần GPU/mạng/mô hình (dự phòng khi demo)
python src/scripts/demo_trace.py --from-cache src/results/demo_trace_en_example.json     # câu tiếng Anh HotpotQA #277, cầu nối 2 bước
python src/scripts/demo_trace.py --from-cache src/results/demo_trace_example.json        # câu tiếng Việt VimQA #320, cầu nối 2 bước
python src/scripts/demo_trace.py --from-cache src/results/demo_trace_polar_example.json  # câu #23, đúng/sai

# Kiểm tra luồng bằng LLM/reader giả (CPU; đáp án "unknown", KHÔNG phải số liệu)
python src/scripts/demo_trace.py --reader mock --itercomp-llm mock
python src/scripts/demo_trace.py --dataset hotpotqa --reader mock --itercomp-llm mock   # cần data/hotpotqa (notebook mục 0.4 tải)
python src/scripts/demo_trace.py --index 23 --reader mock --itercomp-llm mock --strip-diacritics

# Chạy thật (GPU T4) và lưu vết đầy đủ để phát lại sau
python src/scripts/demo_trace.py --index 320 --reader hf --itercomp-llm hf \
    --reader-model Qwen/Qwen2.5-7B-Instruct --load-4bit --llmlingua2 \
    --cache src/results/demo_cache/trace_320.json
python src/scripts/demo_trace.py --from-cache src/results/demo_cache/trace_320.json
# câu tiếng Anh: thêm --dataset hotpotqa --index 277
```

`demo_trace_example.json` / `demo_trace_en_example.json` dựng từ `src/results/vimqa_full_7b.json` /
`hotpotqa_1500_7b.json` bằng `--from-results [--dataset hotpotqa] --index 320|277`: chỉ có đáp án, số từ, F1 từng phương pháp (đúng như
file kết quả lưu). Điểm từng vòng và câu hỏi phụ **không** có trong file kết quả,
nên chỉ xuất hiện khi chạy thật với `--cache`. Mặc định mock dùng BM25 thay bge-m3
để khỏi tải mô hình.

## Prompts

The IterCOMP paper does not release its prompt text, so ours were written from
its method description and are given in full, Vietnamese and English, in
[PROMPTS.md](PROMPTS.md).
