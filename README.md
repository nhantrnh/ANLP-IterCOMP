# Xử lý Ngôn ngữ Tự nhiên Nâng cao

> **Kho mã nguồn:** https://github.com/nhantrnh/ANLP-IterCOMP
>
> **Cấu trúc gói nộp.** File này viết đường dẫn theo cây repo; trong gói nộp
> các file được sắp lại như sau:
>
> ```
> report.pdf            Báo cáo, định dạng ACL
> README.md             File này - cách chạy chương trình
> SOURCE_CODE_LINK.txt  Đường dẫn kho mã nguồn (GitHub)
> DATASET_LINKS.txt     Đường dẫn tải 4 bộ dữ liệu
> MODEL_LINKS.txt       Đường dẫn tải 4 checkpoint (không có model tự huấn luyện)
> notebooks/            Notebook chạy Colab/Kaggle
> source_code/          src/ scripts/ research/ requirements.txt run.sh report_latex/
> ```
>
> Quy đổi: `report_en/report.pdf` → `report.pdf`; phần còn lại của `report_en/`
> → `source_code/report_latex/`; `src/`, `scripts/`, `research/`,
> `requirements.txt`, `run.sh` → trong `source_code/`.
>
> Mọi lệnh trong Mục 3 chạy từ thư mục `source_code/`.

# Đề tài 32 - Thực nghiệm phương pháp **IterCOMP** (ACL 2026) và mở rộng đánh giá sang **tiếng Việt**.

## 1. Giới thiệu
### 1.1. Bài báo gốc
Phương pháp **IterCOMP** được giới thiệu trong bái báo:
> Yun, J. & Kim, Y. (2026). *IterCOMP: Reasoning-aware Adaptive Prompt Compression
> for Multi-hop Question Answering.* ACL 2026, San Diego.
> https://aclanthology.org/2026.acl-long.1559/

**Lưu ý:** 
- Bài báo **không công bố mã nguồn** và **không công bố nội dung câu lệnh** được sử dụng trong phân thực nghiệm.
- Bài báo **có công bố đầy đủ siêu tham số** (Mục 5.1) và nhóm dùng đúng các giá trị đó cho phần thực nghiệm.

**Do đó:**
- Toàn bộ prompt cho bước *answerability* và *follow-up* **do nhóm tự thiết kế**
  theo mô tả Mục 3, và được công bố đầy đủ trong phụ lục báo cáo.
- **Kết quả thực nghiệm của nhóm không so trực tiếp được với kết quả thực nghiệm của bài báo.** Mục tiêu là kiểm chứng *xu hướng tương đối* giữa các phương pháp có được bảo toàn hay không.
- `GPT-3.5-Turbo` mà bài báo dùng đã bị OpenAI ngừng cung cấp.

### 1.2. Yêu cầu bài nộp
Bài nộp sẽ gồm các nội dung sau, được nén lại thành tập tin .zip (đặt tên theo MSHV của tất cả thành viên nhóm):
| Yêu cầu | File |
|---|---|
| Báo cáo PDF định dạng ACL | [`report.pdf`](report.pdf) - tiếng Anh; §1–8 chiếm 5 trang và tràn ~15 dòng sang trang 6, sau đó là Limitations (ACL không tính vào giới hạn trang) rồi References + 10 phụ lục |
| Mã nguồn hoặc đường dẫn kho mã nguồn | `src/` + `scripts/` + `requirements.txt` + `run.sh`, nộp kèm trong zip |
| Jupyter Notebook | [`notebooks/colab_reproduce.ipynb`](notebooks/colab_reproduce.ipynb) - chạy Colab/Kaggle T4, xem Mục 3.3.4 |
| README mô tả cách chạy | file này; cách chạy notebook ở Mục 3.3.4, chạy local ở Mục 3.3 |
| Thông tin/đường dẫn tải dữ liệu | Mục 3.2 - tải tự động từ HuggingFace, **không kèm file dữ liệu trong zip** (116 MB) |
| Mô hình đã huấn luyện | **Không có - IterCOMP là training-free.** Mọi tham số đóng băng; `bge-m3`, `llmlingua-2` và mô hình đọc đều tải tự động từ HuggingFace |

Báo cáo dùng bộ định dạng chính thức ACL
([acl-org/acl-style-files](https://github.com/acl-org/acl-style-files)) - `report_en/acl.sty`.
Mọi con số trong báo cáo được `scripts/verify_report_numbers.py` đối chiếu tự
động với `results/`.

**Notebook nộp bài ghim `transformers==4.45.2`.** Colab/Kaggle nay có thể ship
5.0.0, vốn xung đột cứng với `llmlingua 0.2.2` (5.0 đòi `past_key_values` là
`Cache`, llmlingua lặp `for k, v in past_key_values`). 4.45.2 cũng là đúng phiên
bản mọi kết quả trong báo cáo được sinh ra - xem `notebooks/kaggle/README.md`.

### 1.3. Kết quả bốn dataset ở n≥500

| | VimQA 1003 | MuSiQue 500 | 2Wiki 500 | HotpotQA 500 |
|---|---|---|---|---|
| Oracle | 61.03 | 49.84 | 59.24 | **71.17** |
| Raw | 46.87 | 23.43 | **38.67** | **60.35** |
| IterCOMP | **51.29** | **26.74** | 35.98 | 52.36 |
| LLMLingua-2 | 35.31 | 15.14 | 19.14 | 37.52 |
| IterCOMP−Raw | +4.42 | +3.31 | −2.69 | **−7.99 ** |
| IterCOMP−LLMLingua-2 | +15.98 | +11.60 | +16.84 | +14.83 |

= CI 95% không chứa 0.

**Khẳng định trung tâm của paper (nén câu > nén token) đúng cả bốn.**

**Thứ tự đầy đủ thì không.** Paper báo `Oracle > IterCOMP > Raw > LLMLingua-2`;
mình tái hiện được ở MuSiQue và VimQA, nhưng ở 2Wiki và HotpotQA **Raw vượt
IterCOMP** - có ý nghĩa ở HotpotQA (−7.99).

**Vì sao:** reader của mình đọc ngữ cảnh dài tốt hơn. HotpotQA Raw đạt 60.35 so
với 43.63 của paper (+16.7 chỉ do reader), còn IterCOMP chỉ 51.78 → 52.36. Nên
lợi thế **chính xác** của nén mất, chỉ còn lợi thế **chi phí** (đọc 19.2% token).
Tiền đề "passage không liên quan làm giảm độ chính xác" **phụ thuộc reader**.

## 2. Thuật toán

IterCOMP là **một vòng lặp có điều khiển**, module *answerability judgment* (Mục 4.2.3) quyết định dừng hay tiếp tục ở mỗi vòng.

```
               ┌──────────────────────┐
               │ q^(0)  (câu hỏi gốc) │
               └──────────────────────┘
                           │
                           ▼
  ┌───────────────────────────────────────────────┐
  │ Bước 1 - Document Decomposition   [Mục 4.2.1] │
  │    documents  →  evidence segment (mức câu)   │
  └───────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────┐
│ Bước 2 - Relevant Evidence Filtering   [Mục 4.2.2] │
│      Chấm điểm dual-aspect  +  lọc percentile      │
└────────────────────────────────────────────────────┘
                           │
                           ▼
  ┌───────────────────────────────────────────────┐
  │ Bước 3 - Answerability Judgment   [Mục 4.2.3] │
  │  Evidence tích lũy đã đủ trả lời q^(0) chưa?  │
  └───────────────────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
          đủ                            chưa đủ
           │                               │
           ▼                               ▼
        P_comp                   Sinh follow-up q^(h)
           │                               │
           ▼                               ▼
       Reader M                     Quay lại Bước 2
           │               (evidence CỘNG DỒN, không thay thế)
           │                   Dừng khi đủ HOẶC hết max_iter
           │
           ▼
     Đáp án cuối y
```

### 2.1. Các bước

Tên các mục dưới đây khớp đúng nhãn "Bước 1/2/3" trong sơ đồ ở trên.

#### 2.1.1. Bước 1 - Document Decomposition (Mục 4.2.1)

Tách mỗi document thành evidence segment ở **mức câu** - paper chọn mức này vì
giữ được ngữ cảnh cục bộ, thay vì đoạn văn.

#### 2.1.2. Bước 2 - Relevant Evidence Filtering (Mục 4.2.2)

**Tín hiệu ngữ nghĩa (dense).** Đo độ gần nghĩa giữa câu hỏi và evidence segment
bằng tích trong của 2 dense embedding:

$$S_{sem}(q,e) = E(q)^\top E(e) \qquad (1)$$

- $q$: câu hỏi (query) đang xét
- $e$: evidence segment đang xét.
- $E(\cdot)$: text encoder bge-m3, biến văn bản thành dense embedding.
- $S_{sem}(q,e)$: giá trị càng lớn thì $q$ và $e$ càng gần nghĩa nhau.

**Trọng số từ vựng của 1 token.** Tính mức độ quan trọng của từng token, dùng
làm trọng số cho tín hiệu từ vựng ở công thức (3):

$$w_t = \text{ReLU}\!\left(w_{lex}^\top E(t)\right) \qquad (2)$$

- $t$: 1 token, trong câu hỏi hoặc trong evidence segment.
- $w_{lex}$: vector chiếu (projection), **lấy sẵn từ bge-m3** (M3-Embedding), không tự thiết kế.
- $w_t$: trọng số quan trọng của token $t$; áp dụng riêng cho token trong $q$ (ra $w_t^{q}$) và trong $e$ (ra $w_t^{e}$).

**Tín hiệu từ vựng (sparse).** Đo mức trùng khớp từ vựng giữa câu hỏi và
evidence segment, có trọng số theo công thức (2) - **KHÔNG phải BM25**:

$$S_{lex}(q,e) = \sum_{t \in q \cap e} w_t^{q} \cdot w_t^{e} \qquad (3)$$

- $q \cap e$ - tập token xuất hiện **đồng thời** ở cả $q$ và $e$.
- $w_t^{q}$, $w_t^{e}$ - trọng số của token $t$ trong $q$ và trong $e$ (từ công
  thức 2).
- $S_{lex}(q,e)$ - tổng có trọng số trên các token đồng xuất hiện; trọng số lấy
  từ bge-m3, không phải thống kê tần suất từ như BM25.

**Điểm dual-aspect tổng hợp.** Kết hợp 2 tín hiệu trên thành 1 điểm liên quan
duy nhất:

$$S_{dual}(q,e) = \lambda \cdot S_{sem}(q,e) + (1-\lambda) \cdot S_{lex}(q,e) \qquad (4)$$

- $S_{sem}(q,e)$ - tín hiệu ngữ nghĩa (công thức 1).
- $S_{lex}(q,e)$ - tín hiệu từ vựng (công thức 3).
- $\lambda \in [0,1]$ - trọng số cân bằng 2 tín hiệu.

$E(\cdot)$ và $w_{lex}$ cùng nằm trong 1 checkpoint bge-m3, **đóng băng hoàn
toàn** (training-free) - không huấn luyện hay tinh chỉnh gì thêm.

**Lọc theo percentile.** Giữ lại evidence có điểm liên quan cao, theo 1 ngưỡng
thích ứng với phân bố điểm của từng câu hỏi - không phải số lượng cố định:

$$\tau = \text{Percentile}(S, k), \qquad E_{cand} = \{\, e \mid S_{dual}(q,e) \geq \tau \,\} \qquad (5)$$

- $S$: tập điểm $S_{dual}$ (công thức 4) của mọi evidence đang xét ở vòng đó.
- $k$: **ngưỡng phần trăm** (paper: 90 hoặc 85) - **không phải số lượng
  evidence**.
- $\tau$: ngưỡng điểm tương ứng với percentile $k$ trên $S$.
- $E_{cand}$: evidence giữ lại (điểm $\geq \tau$).

Khác **top-k** (giữ số lượng cố định): số evidence giữ lại theo percentile
**thay đổi theo từng câu hỏi/vòng lặp**, vì $\tau$ phụ thuộc phân bố điểm mỗi
lần chấm.

#### 2.1.3. Bước 3 - Answerability Judgment & Missing Information Identification (Mục 4.2.3)

LLM đóng vai bộ điều khiển: xét evidence đã tích lũy $E_{cand}^{(h)}$ có đủ trả
lời **câu hỏi gốc $q^{(0)}$** chưa (không so với query hiện tại).

- Đủ → dừng, chốt $P_{comp} = E_{cand}^{(h)}$.
- Chưa đủ → LLM chỉ ra thông tin thiếu, sinh **follow-up question** $q^{(h)}$,
  dùng làm query cho vòng lọc kế tiếp.

#### 2.1.4. Vòng lặp & điều kiện dừng (tiếp theo Bước 3)

Lặp lại Bước 2–3 với $\text{query} = q^{(h)}$; evidence các vòng **cộng dồn**
($E_{cand}^{(h+1)} \supseteq E_{cand}^{(h)}$, không bị thay thế), tới khi đủ
bằng chứng (dừng sớm) hoặc đạt `max_iter` (paper: 5).

#### 2.1.5. Compressed prompt & Reader model (sau khi vòng lặp dừng)

$P_{comp}$ (evidence tích lũy tại vòng dừng) được đưa cho **reader model $M$**
- tách biệt khỏi LLM dùng ở Bước 3 - để sinh đáp án cuối:

$$y = M\left(P_{comp}, q^{(0)}\right)$$

### 2.2. Siêu tham số (Mục 5.1 của bài báo)

| Tham số | Giá trị |
|---|---|
| `max_iter` | 5 |
| $\lambda$ | 0.6 |
| $k$ (percentile) | 90 (MuSiQue, HotpotQA) · 85 (2WikiMultiHopQA) |
| Encoder + $w_{lex}$ | bge-m3, đóng băng hoàn toàn (training-free) |
| Reader $M$ | LLaMA-3-8B, dùng chung cho mọi phương pháp so sánh |

VimQA không có trong bài báo; repo dùng `k=90` như HotpotQA vì cùng 10 đoạn/câu.

### 2.3. Pseudocode

```text
segments    ← decompose(documents)            # Mục 4.2.1
accumulated ← ∅
query       ← q0

for h in 1..max_iter:
    scores  ← [S_dual(query, e) for e in remaining_segments]   # ct (1)-(4)
    picked  ← {e | scores[e] >= Percentile(scores, k)}          # ct (5)
    accumulated ← accumulated ∪ picked                          # tích lũy

    if LLM.is_answerable(q0, accumulated):   # so với câu hỏi GỐC
        break                                # dừng sớm
    query ← LLM.make_followup(q0, accumulated)

P_comp ← accumulated
y ← M(P_comp, q0)                            # reader model sinh đáp án cuối
```

### 2.4. Lưu ý

- **Không dùng BM25** - tín hiệu từ vựng lấy từ sparse head của `bge-m3`. BM25 chỉ
  xuất hiện trong repo như baseline ablation, không phải phương pháp của paper.
- **Percentile ≠ top-k** - $k$ là ngưỡng %, không phải số lượng evidence cố định.
- **Evidence tích lũy qua các vòng**, không bị thay thế; nhưng **answerability
  luôn so với câu hỏi gốc $q^{(0)}$**, dù query dùng để lọc đổi theo follow-up mới nhất.
- **Reader ($M$) và LLM suy luận (answerability/follow-up) là 2 vai trò tách
  biệt**, dù paper dùng cùng 1 checkpoint LLaMA-3-8B cho cả hai.
- Bài báo không công bố code/prompt cho answerability & follow-up; repo triển
  khai bằng 2 lệnh gọi LLM riêng (`is_answerable()` rồi `make_followup()`) - đây
  là lựa chọn triển khai, không phải trích dẫn trực tiếp từ paper.

## 3. Thực nghiệm

### 3.0. Chạy nhanh - một lệnh

```bash
./run.sh
```

Tự làm hết: tạo `.venv`, cài thư viện, tải 4 bộ dữ liệu, kiểm tra môi trường,
chạy đánh giá, in bảng LaTeX. Mặc định **reader 0.5B, n=20, chỉ VimQA** để chạy
được trên máy không GPU trong ~5 phút.

```bash
READER_SIZE=7b LIMIT=full ./run.sh        # số như báo cáo (cần GPU ≥12 GB)
LIMIT=200 DATASETS="vimqa musique" ./run.sh
```

| Biến | Mặc định | Giá trị |
|---|---|---|
| `READER_SIZE` | `0.5b` | `0.5b` · `1.5b` · `7b` |
| `LIMIT` | `20` | số câu, hoặc `full` |
| `DATASETS` | `vimqa` | `vimqa musique hotpotqa 2wiki` |

Script tự bật `--load-4bit` khi có CUDA và bỏ qua khi không - nên `READER_SIZE=7b`
trên máy không GPU sẽ chạy fp32 rất chậm và script cảnh báo trước.

**Windows**: `run.sh` cần bash (Git Bash hoặc WSL). Không có thì làm thủ công
theo Mục 3.1–3.3.

### 3.1. Cài đặt

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows - PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Nếu PowerShell chặn script (`cannot be loaded because running scripts is
disabled`), mở PowerShell rồi chạy một lần:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Windows - Command Prompt (cmd)**

```bat
py -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

Ba lưu ý cho Windows:

- Dùng `python` (không phải `python3`) sau khi kích hoạt venv.
- Lệnh nhiều dòng: bash dùng `\`, PowerShell dùng **`` ` ``**, cmd dùng **`^`**.
  Dễ nhất là **viết tất cả trên một dòng**.
- `bitsandbytes` (cần cho `--load-4bit`) không có bản Windows chính thức. Trên
  Windows hãy bỏ `--load-4bit` và dùng mô hình nhỏ hơn
  (`Qwen/Qwen2.5-1.5B-Instruct`), hoặc chạy Colab (Mục 3.3.4).

#### File `.env` - đặt ở đâu

API key **tuỳ chọn**, chỉ cần khi dùng backend API (`openai` / `openrouter` /
`gemini`). Chạy hoàn toàn cục bộ với `--reader hf` thì **không cần `.env`**.

Đặt `.env` ở **gốc repo** (cùng cấp `README.md`). Thư mục cha cũng được nhận -
tiện khi nhiều project dùng chung một key:

```
ANLP-IterCOMP/
├── .env          ← ưu tiên đọc ở đây
├── README.md
└── src/
.env              ← hoặc ở đây (cùng cấp thư mục ANLP-IterCOMP)
```

Nội dung:

```
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...
GEMINI_API_KEY=...
```

Cả hai vị trí đều đã gitignore. Biến môi trường đặt sẵn trong shell luôn được
ưu tiên hơn `.env`.

### 3.2. Tải dữ liệu

Không commit dữ liệu (116 MB). Tải lại:

```bash
cd data
dl() { mkdir -p "$(dirname "$2")"; curl -sL -o "$2" "https://huggingface.co/datasets/$1/resolve/main/$3"; }

# VimQA - tiếng Việt (10.047 câu: 8.041 train / 1.003 val / 1.003 test)
# Dùng nguyenlab/vimqa vì tải trực tiếp được. Bản SEACrowd/vimqa cùng dữ liệu
# nhưng yêu cầu xin quyền qua EULA.
dl nguyenlab/vimqa vimqa/train.parquet      data/train-00000-of-00001.parquet
dl nguyenlab/vimqa vimqa/validation.parquet data/validation-00000-of-00001.parquet
dl nguyenlab/vimqa vimqa/test.parquet       data/test-00000-of-00001.parquet

# Ba bộ tiếng Anh
dl xanhho/2WikiMultiHopQA 2wiki/dev.parquet           dev.parquet
dl hotpotqa/hotpot_qa     hotpotqa/validation.parquet distractor/validation-00000-of-00001.parquet
dl dgslibisey/MuSiQue     musique/dev.jsonl           musique_ans_v1.0_dev.jsonl
```

| Bộ dữ liệu | Số câu | Đoạn/câu | Ghi chú |
|---|---|---|---|
| **VimQA** validation | **1.003** | 10 | Tiếng Việt, cùng định dạng HotpotQA |
| 2Wiki dev | 12.576 | 10 | Có sẵn gold evidence triple |
| HotpotQA distractor val | 7.405 | 10 | |
| MuSiQue-Ans dev | 2.417 | 20 | Khó nhất |

VimQA dùng **bản mirror `nguyenlab/vimqa`** trên HuggingFace. Repo gốc
`github.com/vimqa/vimqa` bị khoá sau EULA qua email, chỉ có 10 dòng demo.

### 3.3. Chạy

#### 3.3.1. Kiểm tra môi trường

```bash
python scripts/smoke_test.py
```

Kiểm 4 thứ độc lập: đọc VimQA · nạp LLMLingua-2 · nén thật một câu · đo token
fertility. Bước nào hỏng thì báo riêng bước đó.

#### 3.3.2. Bảng chính

`--itercomp-llm` là **bắt buộc** - nó chọn LLM cho ba bước suy luận (phân rã /
kiểm tra đủ bằng chứng / sinh câu hỏi tiếp). Không có mặc định là có chủ ý: bỏ
sót nó thì script báo lỗi, chứ không âm thầm đo `mock` rồi để bạn mang số đó đi
so với bài báo. Thiếu tham số này sẽ gặp:

```
run_eval.py: error: the following arguments are required: --itercomp-llm
```

**macOS / Linux**

```bash
# kiểm thử luồng (F1 = 0 vì MockReader luôn trả "unknown")
python scripts/run_eval.py --dataset vimqa --limit 5 \
  --reader mock --itercomp-llm mock

# số thật, miễn phí - mô hình đọc chạy cục bộ
python scripts/run_eval.py --dataset vimqa --limit 50 \
  --reader hf --reader-model Qwen/Qwen2.5-1.5B-Instruct \
  --itercomp-llm hf --out results/vimqa_50.json

# bỏ --limit để chạy TOÀN BỘ tập phát triển
```

**Windows** (một dòng, không cần ký tự nối dòng)

```powershell
python scripts\run_eval.py --dataset vimqa --limit 5 --reader mock --itercomp-llm mock

python scripts\run_eval.py --dataset vimqa --limit 50 --reader hf --reader-model Qwen/Qwen2.5-1.5B-Instruct --itercomp-llm hf --out results\vimqa_50.json
```

Hai lỗi hay gặp khác, cùng nguyên nhân là thiếu tham số phụ thuộc:

| Báo lỗi | Cách sửa |
|---|---|
| `--reader hf cần --reader-model` | thêm `--reader-model Qwen/Qwen2.5-1.5B-Instruct` |
| `--itercomp-llm hf cần --reader-model` | như trên (`hf` dùng chính mô hình đọc) |

`--itercomp-llm hf` dùng **chính mô hình đọc** cho các bước suy luận, đúng như
bài báo. Dùng `mock` thì vòng lặp **không chạy** - nó luôn báo "đủ bằng chứng"
ngay vòng 1, nên IterCOMP suy biến thành lọc một lượt và **không so được với
bài báo**.

| Phương pháp | Ý nghĩa |
|---|---|
| `raw` | Raw Document - không nén |
| `oracle` | Chỉ gold supporting documents. **Trần trên** và là công cụ chẩn đoán: nếu Oracle cũng thấp thì điểm nghẽn ở mô hình đọc, không phải bộ nén |
| `llmlingua2` | Đường cơ sở nén, đa ngữ nên chạy được tiếng Việt |
| `itercomp` | Phương pháp của bài báo |

Đường cơ sở bài báo dùng: **4/5 đã hiện thực** trong `run_eval.py` -
`llmlingua`, `longllmlingua`, `selective-context`, `recomp-extractive`,
`recomp-abstractive`. Chỉ **R2C** chưa, vì không tìm thấy checkpoint công khai
và bài báo gốc cũng không trích dẫn nó đủ rõ để dựng lại.

Đã hiện thực nhưng **chưa có số**: cần một phiên GPU cho n=200 (~2h). Xem
`notebooks/kaggle/`.

#### 3.3.3. Ablation

```bash
python scripts/run_ablation.py --dataset vimqa --limit 30 \
  --llm hf --reader hf --reader-model Qwen/Qwen2.5-1.5B-Instruct \
  --out results/ablation_vimqa.json
```

Windows (một dòng):

```powershell
python scripts\run_ablation.py --dataset vimqa --limit 30 --llm hf --reader hf --reader-model Qwen/Qwen2.5-1.5B-Instruct --out results\ablation_vimqa.json
```

`--llm` cũng bắt buộc, cùng lý do như `--itercomp-llm` ở trên. `--reader hf` thì
phải kèm `--reader-model`.

26 cấu hình, 6 trục: `max_iter` 1–5 · `percentile` 70–95 · `top_k` 1–3 ·
**`scorer`** bm25/dense/lexical/dual · **λ** 0.0–1.0 · bỏ từng bước reasoning-aware.

**Bắt buộc có `--reader`.** Lọc percentile luôn giữ ~(100−k)% segment, nên đổi λ hay
scorer *không* làm đổi tỉ lệ nén - chỉ đổi *segment nào* được chọn. Không có EM/F1
thì không so sánh được các scorer.

Script in sẵn bảng LaTeX ở cuối để dán vào báo cáo.

#### 3.3.4. Chạy trên GPU (Colab **hoặc** Kaggle)

[`notebooks/colab_reproduce.ipynb`](notebooks/colab_reproduce.ipynb) - tái hiện với
mô hình đọc 7B. Cần **GPU T4**. Notebook tự phát hiện nền tảng (`/content` trên
Colab, `/kaggle/working` trên Kaggle) nên **cùng một file chạy được cả hai**.

**Ba bước để chạy:**

1. **Bật GPU T4** (xem bảng dưới). Colab/Kaggle không tự chọn GPU.
2. **Chọn quy mô** ở ô §3: biến `SCALE`. Mặc định `'paper'` = **8,6h GPU**, đúng
   quy mô sinh ra số trong báo cáo. Muốn kiểm tra chương trình chạy được thì đổi
   `SCALE = 'smoke'` (~0,4h, n=50) - nhưng ở n=50 ngưỡng phát hiện là 24,8 F1 nên
   số `smoke` in ra **không kết luận được**, đừng đối chiếu với báo cáo.
3. **Run all.**

Notebook **không cần upload gì bằng tay**: tự tải mã nguồn và cả 4 bộ dữ liệu, và
không tham chiếu đường dẫn cục bộ nào.

Thường Run all chạy một mạch tới hết. Ngoại lệ duy nhất phải cắt: ô §1 ghim
`transformers==4.45.2`, và nếu runtime đã nạp sẵn bản 5.x thì ghim chưa có hiệu
lực - ô §1 dừng ngay với `Ghim KHÔNG ăn`. Khi đó Restart session rồi Run all lại
từ đầu; các ô đều lặp lại được, không mất gì.

Phiên bị ngắt giữa chừng cũng không mất kết quả: `run_eval.py` ghi checkpoint sau
mỗi câu, chạy lại là tiếp từ chỗ dở.

| | Colab | Kaggle |
|---|---|---|
| Bật GPU | Runtime → Change runtime type → T4 | Settings → Accelerator → GPU T4 |
| Quota | ~12h/phiên, giới hạn theo ngày | **30h/tuần**, phiên tối đa 9h |
| Mạng | luôn bật | phải bật **Internet** trong Settings |
| Lấy kết quả | tự tải `results.zip` về | Save Version → tab Output |

**Đẩy lên Kaggle bằng CLI** (nếu không muốn upload tay):

```bash
kaggle kernels push -p . --accelerator NvidiaTeslaT4
```

`--accelerator` là **bắt buộc**. Kaggle cấp GPU ngẫu nhiên giữa **P100** và
**T4**, mà PyTorch cài sẵn ở đó chỉ hỗ trợ từ `sm_70` - P100 là `sm_60` nên
notebook dừng ngay ở ô kiểm tra GPU:

```
Tesla P100-PCIE-16GB | 17.1 GB | sm_60
AssertionError: GPU sm_60 không chạy được PyTorch - chọn T4
```

`kernel-metadata.json` **không có** tham số chọn loại GPU - chỉ `enable_gpu`
bật/tắt - nên phải truyền qua flag CLI. Trên web thì chọn
Settings → Accelerator → **GPU T4 x2**.

**Reader 7B KHÔNG vừa Kaggle T4.** Đã thử và thất bại - ghi lại để khỏi mất
thêm phiên:

Tiến trình chính của notebook giữ **1,68 GB** CUDA context ngay khi ô kiểm tra
GPU gọi `torch.cuda.get_device_properties()`. Context đó chỉ mất khi tiến trình
thoát, mà kernel Jupyter thì không thoát. Phần còn lại cho mỗi tiến trình con:

| | tổng | trừ context | reader 7B + bge-m3 + KV | kết quả |
|---|---|---|---|---|
| **Colab T4** | 15,00 GB | 13,32 GB | ~9,3 GB | vừa |
| **Kaggle T4** | 14,56 GB | 12,88 GB | ~9,3 GB | sát ngưỡng, OOM khi phân mảnh |

Chênh 0,44 GB giữa hai nền tảng là đủ để đổi kết quả. Trên Kaggle mọi job
`run_eval.py` chết mã 1 sau 0,3–0,5 phút, tức OOM lúc nạp model chứ không phải
chạy được rồi hết giờ.

Nếu buộc phải dùng Kaggle: chạy bge-m3 trên CPU (`make_scorer(..., device='cpu')`,
trả lại ~2,3 GB, chậm hơn ~25%) hoặc hạ reader xuống 1.5B - nhưng số liệu 1.5B
**không so được** với Bảng 1 vốn dùng 7B.

Kaggle vẫn hữu ích cho bậc `smoke` và các ô không cần reader 7B (khám phá dữ
liệu, quét `k`, đo fertility). Quota 30h/tuần, phiên tối đa 9h.

Với cả hai nền tảng: **VimQA chạy trước** và `run_eval.py` ghi checkpoint sau mỗi
câu, nên phiên bị ngắt thì chạy lại là tiếp từ chỗ dở.

Thứ tự chạy: ô cấu hình → **ô kiểm tra vòng lặp** (vài phút, có `assert` báo lỗi ngay
nếu backend suy luận không hoạt động) → bảng chính (nhiều giờ). Ô kiểm tra đứng trước
để không mất cả phiên mới phát hiện sai cấu hình.

Mặc định **Qwen2.5-7B-Instruct** (mở, không cần licence). Muốn đúng **LLaMA-3-8B** như
bài báo thì xin quyền tại HuggingFace rồi bỏ chú thích phần `login()` trong ô cấu hình.

Quy mô: notebook chọn qua biến `SCALE` trong ô cấu hình. Chi phí suy ra từ s/câu
**đã đo** trên T4 (Qwen2.5-7B 4-bit): VimQA 9,6 s/câu, MuSiQue 17,5 s/câu.

| `SCALE` | VimQA | MuSiQue | HotpotQA/2Wiki | tổng | ngưỡng phát hiện |
|---|---|---|---|---|---|
| `smoke` | 50 | 50 | – | 0,4h | 18,0 F1 |
| `paper` ← mặc định | **full 1.003** | 500 | 500 | **10,0h** | **4,3 F1** |
| `max` | full 1.003 | full 2.417 | – | 14,4h | 4,3 F1 |

*Ngưỡng trên là loại **ghép cặp** - phép so đúng vì mọi phương pháp chạy trên
cùng bộ câu hỏi. Ngưỡng không-ghép-cặp cao hơn 1,2× (24,8 ở n=50) và các bản
tài liệu cũ trích nhầm con số đó.*

Vì sao `paper` là mặc định: ở n=50 ngưỡng phát hiện là 18,0 F1, nên mọi so sánh
dưới ngưỡng đó **không kết luận được**. VimQA full hạ ngưỡng xuống **4,3 F1** -
và ở đó IterCOMP **vượt Raw +4,42 F1, KTC [+1,73, +7,17]**, điều n=50 không nói
được. Xem `results/vimqa_full_7b.json`; lần chạy này đã thực hiện, ~2,7h trên
Kaggle T4.

Vì sao **không** full cả 4 bộ: HotpotQA (7.405 câu) + 2Wiki (12.576) ≈ 97h
≈ 8 phiên Colab. Hai bộ đó chỉ dùng để xác nhận thứ tự phương pháp, mà n=500
đã đủ cho việc đó. `max` vượt một phiên 12h nên notebook tự cảnh báo.

`run_eval.py` ghi checkpoint sau **mỗi câu**, nên phiên bị ngắt giữa chừng chạy
lại vẫn tiếp từ chỗ dở - với VimQA full đây là điều kiện bắt buộc, không phải
tiện lợi.

## 4. Kết quả

### 4.1. So với bài báo gốc

Chạy 16/08/2026 trên Colab T4, Qwen2.5-7B-Instruct 4-bit, `--itercomp-llm hf`,
`scorer=dual`, λ=0.6, k=90, max_iter=5, n=50/bộ.

| Method | MuSiQue (paper) | MuSiQue n=50 | **MuSiQue n=500** | VimQA n=1003 |
|---|---|---|---|---|
| Oracle | 37.51 | 57.0 | **49.84** | 61.03 |
| Raw Documents | 19.92 | 32.8 | **23.43** | 46.87 |
| LLMLingua-2 | 14.72 | 13.5 | **15.14** | 35.31 |
| **IterCOMP** | **27.36** | **35.1** | **26.74** | **51.29** |
| *tỉ lệ nén* | *0.14* | *0.32* | *0.349* | *0.206* |
| *Oracle/Raw* | *1.88* | *1.74* | *2.13* | *1.30* |

**Bằng chứng cài đặt đúng:**

1. **Thứ tự phương pháp khớp hoàn toàn** - Oracle > IterCOMP > Raw > LLMLingua-2
   trên MuSiQue, ở **cả n=50 và n=500**, đúng thứ tự bài báo công bố.
2. **Oracle/Raw**: 1.74 ở n=50 và 2.13 ở n=500 so với 1.88 của paper. Tỉ số này
   đo "trần" mà chọn bằng chứng hoàn hảo mang lại, phụ thuộc dữ liệu và cách
   tính chứ **không** phụ thuộc reader. Khớp gần hơn ở mẫu nhỏ, nên trích cả hai
   thay vì chọn con số có lợi.

**Chặn trên của phép tái hiện.** Ở n=500, Oracle và Raw - hai thứ **không đụng
code nén** - cao hơn paper **+12.33** và **+3.51**, xác nhận reader mạnh hơn.
Nhưng IterCOMP, method **duy nhất phụ thuộc code này**, lại **thấp hơn 0.62**, dù
giữ ngữ cảnh gấp 2.5 lần (0.349 vs 0.14). Nghĩa là **bộ lọc của họ đạt cùng độ
chính xác trên 40% số token mình cần** - cơ chế tái hiện được, việc chọn bằng
chứng thì kém hiệu quả hơn bản gốc.

**Số của tôi cao hơn bài báo - đây KHÔNG phải cải tiến.** Tỉ lệ nén không khớp:
tôi giữ 0.32 ngữ cảnh còn bài báo giữ 0.14, nên một phần chênh lệch chỉ vì giữ
nhiều bằng chứng hơn. Ngưỡng percentile thích ứng theo từng câu (công thức 5) nên
tỉ lệ giữ lại phụ thuộc phân bố điểm của encoder.

**Điều KHÔNG kết luận được:** `IterCOMP − Raw` là +2.3 (MuSiQue) và −0.7 (VimQA),
p = 0.72 và 0.92. Với n=50, chênh nhỏ nhất phát hiện được là ~23–25 điểm F1, nên
**không thể nói IterCOMP thắng Raw**. Cái đứng vững: IterCOMP hơn LLMLingua-2
21.6 điểm trên MuSiQue (p=0.0002).

**Điều đáng chú ý về mặt thực dụng:** trên VimQA, IterCOMP đạt 43.0 F1 so với
43.8 của Raw trong khi chỉ đọc **19.3% số token**. Ngang độ chính xác với một
phần năm ngữ cảnh - đúng đánh đổi mà phương pháp nhắm tới.

### 4.2. Điều kiện để chạy trên tiếng Việt

**Không phải thay model nào** - bge-m3 và LLMLingua-2 vốn đã đa ngữ có tiếng Việt.
Ba tuỳ chỉnh **bắt buộc**, đều ở khâu xử lý văn bản:

| Tuỳ chỉnh | Không có thì sao |
|---|---|
| `re.UNICODE` khi tách token | ký tự có dấu bị tách rời → **EM/F1 vô nghĩa** |
| Giữ nguyên dấu thanh | `má` = `mà` = `mã` → gộp từ khác nghĩa |
| Chuẩn hoá đáp án đúng/sai | 30% dataset bị chấm sai (reader trả lời tiếng Anh) |

Thêm: VimQA không có trong paper nên không có `k` công bố → dùng `k=90` như
HotpotQA (cùng 10 đoạn/câu), đã verify bằng sweep.


### 4.3. Ablation trên VimQA (n=50, 7B 4-bit)

| Cấu hình | Tỉ lệ nén | Vòng TB | F1 |
|---|---|---|---|
| max_iter=1 | 13.4% | 1.00 | 28.4 |
| **max_iter=2** | **15.8%** | **1.18** | **33.0** |
| max_iter=3 | 17.0% | 1.30 | 33.0 |
| max_iter=4 | 18.1% | 1.40 | 33.0 |
| max_iter=5 *(paper)* | 18.9% | 1.50 | 33.0 |
| percentile k=70 | 43.0% | 1.40 | 38.8 |
| percentile k=85 | 24.5% | 1.42 | 36.4 |
| percentile k=90 *(paper)* | 18.9% | 1.50 | 33.0 |
| percentile k=95 | 11.8% | 1.62 | 34.6 |
| top-k=2 | 11.5% | 1.68 | 36.8 |
| top-k=3 | 16.3% | 1.52 | 37.5 |

**Ngân sách vòng lặp bão hoà ở 2.** F1 **giống hệt** 33.0 với max_iter ∈ {2,3,4,5}
- không phải "gần bằng" mà là cùng một con số. Trong khi đó tỉ lệ giữ lại tăng từ
15.8% lên 18.9% và số vòng thực thi trung bình từ 1.18 lên 1.50. Nâng ngân sách từ
2 lên 5 như bài báo tốn thêm ~20% token và ~27% lượt gọi LLM mà **không đổi lấy
điểm nào** trên bộ này. Vì F1 giống hệt chứ không chỉ tương tự, kết luận này
không cần kiểm định - nó là quan sát về **chi phí**, không phải suy luận về độ
chính xác.

Không suy rộng thành "bài báo đặt sai": điểm bão hoà nhiều khả năng phụ thuộc độ
sâu suy luận, mà VimQA chủ yếu 2-hop; ngân sách 5 có thể xứng đáng trên chuỗi
MuSiQue sâu hơn.

**Điều sweep ngưỡng KHÔNG cho phép kết luận.** Ở cùng mức nén, top-k cao hơn
percentile - 36.8 vs 34.6 tại ~11.5%, và 37.5 vs 33.0 tại 16–19%. So sánh này có
kiểm soát mức nén nên không bị nhiễu bởi "giữ nhiều thì điểm cao". Nhưng chênh
lệch 2.2 và 4.5 F1 nằm dưới ngưỡng phát hiện ghép cặp 18.0 ở n=50, nên **chỉ ghi
lại độ lớn, không rút kết luận**.

### 4.4. Phân tích theo loại suy luận (2Wiki)

| Phương pháp | bridge_comp | comparison | compositional | inference |
|---|---|---|---|---|
| raw | 64.3 | 37.6 | **2.9** | 21.5 |
| oracle | 57.1 | 34.3 | **41.6** | 36.8 |
| itercomp | 57.1 | 40.9 | 12.4 | 27.5 |

Với câu hỏi hợp thành, raw chỉ đạt 2.9 còn oracle 41.6 - gấp **14 lần**. Nhiễu gần
như phá hỏng hoàn toàn khả năng trả lời.

### 4.5. Phân tích lỗi (VimQA, n=50)

| Loại | reader 0.5B | **reader 7B** |
|---|---|---|
| Đúng hoàn toàn | 8% | **16%** |
| Đúng một phần | 34% | **32%** |
| **Sai định dạng ở câu đúng/sai** | **32%** | **10%** |
| Sai hoàn toàn | 26% | **42%** |

Lỗi định dạng **không phải lỗi bộ nén**: mô hình đọc trả `"yes"` thay vì `"đúng"`
→ F1 = 0 dù suy luận chính xác (mô hình trộn tiếng Anh khi sinh tiếng Việt).

Con số này **giảm 32% → 10%** khi đổi sang reader lớn hơn - bằng chứng trực tiếp
rằng đây là hạn chế của **mô hình đọc**, không phải của thuật toán nén. Nhưng nó
không về 0, nên vẫn cần chuẩn hoá ở tầng chỉ số.

### 4.6. Token fertility

| | Token VN | Token EN | Tỉ lệ |
|---|---|---|---|
| Trung bình | | | **1.98×** |

Tiếng Việt tốn gần gấp đôi token cho cùng nội dung (`cl100k_base`). Với **cùng một
tỉ lệ nén**, số token tiết kiệm trên văn bản tiếng Việt lớn hơn đáng kể - nén câu
lệnh không chỉ *áp dụng được* cho tiếng Việt mà *có lợi hơn*.

LLMLingua-2 nén 3.0× trên tiếng Việt (2.318 → 785 token) và **giữ nguyên dấu thanh**,
bảo toàn thực thể lẫn mốc thời gian.

## 5. Ba lỗi âm thầm đã gặp

Ghi lại vì cả ba đều **không báo lỗi**, chỉ cho kết quả sai:

1. **Mô hình suy luận trả chuỗi rỗng.** Model đời mới tiêu token cho suy luận nội bộ
   trước khi trả lời; với `max_tokens=64` chúng trả `''`, khiến F1 = 0 trông như trả
   lời sai. Đặt `max_tokens ≥ 256`.
2. **Gọi API treo vô hạn.** Thư viện OpenAI không đặt `timeout` mặc định - một lần
   chạy treo 30 phút ở 0% CPU với kết nối vẫn mở. Phải đặt `timeout` và `max_retries`.
3. **Loader sai định dạng.** MuSiQue không có trường `supporting_facts` mà gắn cờ
   `is_supporting` lên từng đoạn; nếu không xử lý, `oracle` lặng lẽ rơi về ngữ cảnh
   đầy đủ và cho F1 y hệt `raw`.

Ngoài ra, khi soạn báo cáo LaTeX tiếng Việt: phải dùng
`\usepackage[T1,T5]{fontenc}` - đảo thành `[T5,T1]` thì T1 thành bảng mã mặc định và
ký tự có dấu sẽ lỗi.

## 7. Cải tiến cho tiếng Việt

Ba cải tiến không có trong bài báo gốc, đều xuất phát từ quan sát trên dữ liệu
tiếng Việt của chúng tôi.

### 7.1. Chuẩn hoá đáp án đúng/sai - `metrics.normalize_boolean`

Mô hình đọc thường trả lời câu đúng/sai bằng **tiếng Anh** (`"Yes"`, `"tức là
yes"`) trong khi đáp án vàng là `"đúng"` - đúng nghĩa nhưng F1 = 0. Đây là chuyển
mã của mô hình đa ngữ nhỏ gặp điểm yếu của độ đo khớp từ vựng, trên bộ dữ liệu mà
**300/1.003 câu VimQA có đáp án `đúng`/`không`**.

VimQA chỉ chuẩn hoá Unicode và dấu thanh, không chuẩn hoá ngữ nghĩa đáp án.

| Phương pháp | F1 thô | F1* | Chênh | p |
|---|---|---|---|---|
| raw | 23.1 | 27.7 | +4.6 | **0.0004** |
| **oracle** | 28.3 | **30.1** | +1.8 | **0.036** |
| llmlingua2 | 15.0 | 17.0 | +2.0 | **0.034** |
| **itercomp** | 19.3 | 24.8 | **+5.5** | **0.0000** |

*(VimQA n=200, bootstrap ghép cặp 10.000 lần lấy mẫu lại)*

Chuẩn hoá có ý nghĩa với **cả bốn** phương pháp, và **IterCOMP hưởng lợi nhiều
nhất** (+5.5, p<0.001). Sau chuẩn hoá, IterCOMP vượt LLMLingua-2 **+7.8 điểm**
(KTC [+2.5, +13.3], p=0.004) - **khẳng định chính của paper, tái hiện được trên
tiếng Việt**.

**Bài học về cỡ mẫu - tôi đã kết luận sai một lần.** Ở n=50, chuẩn hoá nâng
IterCOMP +12.0 điểm và đưa nó lên **hạng 1, vượt cả Oracle**. Tôi đã viết đó là
"đảo thứ hạng". Với **n=200 thì điều đó KHÔNG còn đúng**: mức tăng thật chỉ
**+5.5**, và Oracle (30.1) trở lại dẫn đầu.

Chênh 12 điểm ở n=50 nằm **dưới ngưỡng phát hiện của chính cỡ mẫu đó** (21.9
điểm) → là nhiễu. Với n=200 ngưỡng giảm còn **10.3 điểm**.

### 7.2. Hiệu chỉnh theo token fertility - `fertility.py`

Fertility là hiện tượng **đã được đặt tên và đo đạc** (Petrov NeurIPS 2023, Ahia
EMNLP 2023) → không phải phát hiện của chúng tôi. Điều **chưa được nêu** là hệ
quả cho việc nén: tỉ lệ 2× ở tiếng Việt và tiếng Anh **không giữ lại cùng lượng
thông tin**, nên so hai ngôn ngữ ở cùng tỉ lệ token là lệch.

Khảo sát về nén câu lệnh (NAACL 2025) **không đề cập tới ngôn ngữ**; LLMLingua-2
có đánh giá tiếng Trung nhưng không xét khác biệt số token.

```
tỉ lệ danh nghĩa    58.1%    ← con số bài báo báo cáo
tỉ lệ hiệu dụng     37.5%    ← sau hiệu chỉnh fertility 1.55×
token tiết kiệm        13    → tương đương 8.4 token tiếng Anh
âm tiết bị cắt dở       3    ← độ đo đặc thù tiếng Việt
```

### 7.3 Ablation có EM/F1 (MuSiQue, n=50)

Bảng này đo với reader **Qwen2.5-0.5B** (lần chạy cũ), khác với bảng VimQA
phía trên vốn dùng 7B. Không so trực tiếp hai bảng.

Trích từ 25 cấu hình, đo với **reader 7B** (`results/ablation_vimqa_7b_hf.json`):

| Cấu hình | nén | vòng TB | F1 |
|---|---|---|---|
| BM25 (không dùng encoder) | 18.8% | 1.54 | 33.0 |
| chỉ ngữ nghĩa (λ=1) | 17.7% | 1.40 | 35.1 |
| chỉ từ vựng (λ=0) | 18.9% | 1.46 | 35.4 |
| **song diện, λ=0.6** *(paper)* | 18.9% | 1.50 | 33.0 |
| bỏ answerability | **52.9%** | 5.00 | 36.6 |
| bỏ cả hai bước | **54.4%** | 5.00 | 37.1 |

Với n=50, ngưỡng phát hiện ghép cặp là **±18.0 điểm F1** → **không** chênh
lệch F1 nào trong bảng đạt ý nghĩa thống kê. Bảng ghi nhận quy mô, **không rút
kết luận**.

Quan sát **không cần kiểm định**: bỏ bước answerability làm tỉ lệ giữ lại tăng
gấp ba (18.9% → 52.9%) vì vòng lặp không còn dừng sớm được. Đây là quan sát về
luồng điều khiển, xác nhận **dừng sớm là cơ chế nén chủ yếu**.

**EXP-1 đã trả lời câu hỏi bảng này để ngỏ** (`results/exp1_oracle_*_7b.json`,
n=200, reader 7B). So bốn chế độ dừng cho thấy bộ dừng **hoàn hảo** đạt 49.91 F1
so với 50.08 của bài báo trên VimQA - trần chỉ **−0.17 F1**, và +0.04 trên
MuSiQue. Chênh 3.6 F1 mà bảng n=50 gợi ý là **nhiễu**. Giá trị thật của bước
dừng nằm ở **tiết kiệm token**: không dừng thì giữ 49.7% ngữ cảnh thay vì 18.1%
ở cùng mức F1. Hệ quả: cải tiến bộ dừng **không đáng theo đuổi**.

Quan sát **không phụ thuộc thống kê**: bỏ bước answerability làm tỉ lệ nén tăng
10.0% → **55.9%** (gấp 5.6×) **trên MuSiQue**; trên VimQA là 18.9% → 52.9%
(gấp 2.8×). Hai con số thuộc hai tập khác nhau, không phải số vênh. Đây là quan sát về luồng điều khiển → xác nhận
**dừng sớm là cơ chế nén chủ yếu**. Về `max_iter`: ở n=50 các mức 2→5 cho F1 y hệt,
nhưng **ablation n=500 đã bác bỏ** điều đó (`ablation_vimqa500_7b.json`) - chúng
tách ra 34.9/35.6/36.2/36.1, vòng lặp TB tăng 1.18→1.58. Cái còn đúng: 2→5 chỉ
+1.2 F1, trong nhiễu, nên vẫn nên chọn budget nhỏ vì **chi phí**.

### 7.4. Quét λ và k riêng cho tiếng Việt - `run_lambda_sweep.py`

**Kết quả âm tính, báo cáo trung thực.** λ tốt nhất là 0.4 so với 0.6 của bài
báo, nhưng chênh chỉ **0.8 điểm F1** - để phát hiện cần **n ≈ 37.000** câu, VimQA
chỉ có 1.003. Với n=50, chênh nhỏ nhất phát hiện được là **21.9 điểm**.

→ **Không có bằng chứng** cho thấy siêu tham số của bài báo cần chỉnh cho tiếng
Việt. k=90 trùng bài báo.

### Về claim novelty

**LLMLingua-2 đã đánh giá trên tiếng Trung** (Appendix J). Nên:
- "đầu tiên trên ngôn ngữ ngoài tiếng Anh" - **SAI**
- "đầu tiên trên **tiếng Việt**" - đúng

### Độ mạnh thống kê

Với n=50, chênh nhỏ nhất phát hiện được ~**21.9 điểm F1**. Cỡ mẫu cần:

| Phát hiện chênh | Cần n |
|---|---|
| 10 điểm | 239 |
| 5 điểm | 956 |
| 0.8 điểm (λ) | 37.314 |

`run_eval.py` giờ tự in khoảng tin cậy bootstrap và kiểm định ghép cặp cho mọi bảng.

## 8. Việc còn lại

- [x] ~~Ablation có EM/F1~~ - xong trên MuSiQue (Phụ lục báo cáo)
- [x] Reader 7B - đã chạy trên Kaggle T4, xem `notebooks/kaggle/`
- [x] VimQA full **n=1003** - `results/vimqa_full_7b.json`
- [x] Hiện thực 4/5 đường cơ sở còn thiếu (`llmlingua`, `longllmlingua`,
      `selective-context`, `recomp-extractive`, `recomp-abstractive`)
- [ ] **Chạy** 4 đường cơ sở đó ở n=200 - cần ~2h GPU
- [ ] MuSiQue n≥500 để cân bảng đối chiếu (VimQA đã 1003, MuSiQue mới 50)
- [ ] Ablation ở n≥500 (hiện n=50, dưới ngưỡng ~18 F1)
- [ ] R2C - chưa tìm thấy checkpoint công khai

### Ghi chú về hạ tầng

Kaggle **không dùng được** cho việc này: tài khoản được cấp **P100 (sm_60)**, mà
PyTorch của Kaggle build cho `sm_70+` → buộc chạy CPU, mất **644 phút/dataset** và
bị cắt ở giới hạn 12h/phiên khi mới xong 1/4. Quota GPU 30h/tuần cũng đã hết.

Dùng **Colab** (cấp T4, `sm_75`) hoặc chạy local với reader 0.5B.

## 9. Giấy phép dữ liệu

| Nguồn | Giấy phép |
|---|---|
| VimQA | CC BY-NC-SA 4.0 (phi thương mại) |
| HotpotQA | CC BY-SA 4.0 |
| 2WikiMultiHopQA | Apache-2.0 |
| MuSiQue | CC BY 4.0 |
| LLMLingua | MIT |

Repo chỉ dùng cho mục đích học tập và nghiên cứu.

---

## 9. `results/` có gì

32 file JSON, sinh ra ở nhiều thời điểm và nhiều cấu hình. Bảng này để không
ai phải đoán file nào dùng cho bảng nào.

**Đọc cột `n` và `reader` trước khi so hai file bất kỳ.** Kết quả reader 0.5B
và 7B **không so trực tiếp được**, và đó là lỗi đã từng xảy ra thật trong dự
án này.

### Bảng chính - số vào báo cáo

| File | n | reader | Dùng ở đâu |
|---|---|---|---|
| `vimqa_full_7b.json` | **1003** | 7B | Phụ lục "Main Table at Full Scale" |
| `vimqa_200_7b.json` | 200 | 7B | Kiểm định ghép cặp, Mục 7.1 |
| `baselines_vimqa_200_7b.json` | 200 | 7B | **5 baseline thêm** - IterCOMP thắng cả 5, nhưng chỉ LLMLingua-2 là so sánh công bằng - 5 baseline còn lại dùng thành phần tiếng Anh, nên là **confound**, không phải lời giải thích |
| `baselines_paired_vimqa200.json` | 200 | 7B | CI ghép cặp IterCOMP vs từng baseline |
| `vimqa_200_7b_pinned.json` | 200 | 7B | Bảng chính chạy lại cùng phiên với baseline (transformers ghim 4.45.2); khớp file gốc trong nhiễu (+0.09 / +0.51) |
| `vimqa_50_7b.json` | 50 | 7B | Bảng đối chiếu với bài báo |
| `musique_50_7b.json` | 50 | 7B | Cột MuSiQue của bảng đó |
| `hotpotqa_50_hf05b.json` | 50 | 0.5B | Bảng 4 bộ dữ liệu |
| `2wiki_50_hf05b.json` | 50 | 0.5B | Bảng 4 bộ dữ liệu |
| `musique_50_hf05b.json` | 50 | 0.5B | Bảng 4 bộ dữ liệu |
| `vimqa_200.json` | 200 | 0.5B | Chuẩn hoá boolean, Mục 7.1 |

### Ablation

| File | n | reader | Đo gì |
|---|---|---|---|
| `ablation_vimqa_7b_hf.json` | 50 | 7B | 26 cấu hình - đường cong $k$, scorer, `max_iter` |
| `ablation_musique.json` | 50 | - | Ablation trên MuSiQue |
| `ablation_vimqa_scorers.json` | 20 | - | So riêng các scorer |
| `ablation_vimqa_mock.json` | 30 | - | Chạy thử, không vào báo cáo |
| `lambda_sweep_vimqa.json` | 50 | **0.5B** | Quét λ - reader khác nên **không so** với dòng 7B |
| `ablation_adaptive_k_vimqa.json` | 200 | - | Adaptive percentile có thích ứng thật không |
| `ablation_adaptive_k_musique.json` | 200 | - | Cùng phép đo, tập khác |
| `ablation_adaptive_k_vimqa_dual.json` | 200 | - | Kiểm lại bằng scorer `dual` - kết luận giữ nguyên |
| `ablation_adaptive_k_musique_dual.json` | 200 | - | Cùng phép kiểm, tập MuSiQue |

### Thí nghiệm bổ sung (ngoài bảng báo cáo)

| File | n | reader | Kết luận |
|---|---|---|---|
| `exp1_oracle_vimqa_200_7b.json` | 200 | 7B | Trần bộ dừng **−0,17 F1** |
| `exp1_oracle_musique_200_7b.json` | 200 | 7B | Trần **+0,04 F1** |
| `exp2_undiacritised_200_7b.json` | 200 | 7B | Bỏ dấu: nén **vượt** gold |
| `damage_vs_f1_vimqa200_7b.json` | 200 | 7B | ρ ≈ 0 - không tương quan |
| `damage_curve_vimqa_500.json` | 500 | - | **Đối chứng hư-từ n=500 (CPU)** - IterCOMP 1.01× vs LLMLingua-2 1.17×, chênh +0.16 |
| `damage_curve_musique_500.json` | 500 | - | Đối chứng **tiếng Anh** - 0.99× vs 1.17×, chênh **+0.17** → dấu giữ nguyên, hai chênh lệch bằng nhau |
| `damage_matched_500.json` | 500 | - | Tổng hợp: khẳng định granularity vững ở cỡ mẫu gấp mười |
| `2wiki_500_7b.json` | 500 | 7B | **2WikiMultiHopQA** (k=85) - Raw vượt IterCOMP 2.69, CI chứa 0 |
| `hotpotqa_500_7b.json` | 500 | 7B | **HotpotQA** - Raw vượt IterCOMP **7.99**, CI sạch → thứ tự paper **không** tái hiện |
| `four_datasets_paired.json` | - | 7B | Tổng hợp 4 dataset: khẳng định trung tâm đúng cả 4, thứ tự đầy đủ chỉ đúng 2 |
| `ablation_musique200_k.json` | 200 | 7B | **So ở CÙNG MỨC NÉN với paper** - k=96 cho ratio 0.160 (paper 0.14) và F1 **29.04** so với **27.36**, tức **+1.68** chứ không phải −0.62 |
| `compact_vimqa_200_7b.json` | 200 | 7B | **CompAct** (baseline 6/7) - F1 23.43, nén mạnh nhất (5.6% ≈ ngân sách Oracle) nhưng trả `unknown` 45%, chậm nhất 20.9 s/câu |
| `PROVENANCE.md` | - | - | **Nhật ký nguồn gốc** - kernel nào sinh file nào, cái nào bỏ và vì sao |
| `musique_500_7b.json` | 500 | 7B | **Đối chứng tiếng Anh đúng cỡ mẫu** - IterCOMP vượt LLMLingua-2 **+11.60** (có ý nghĩa) nhưng **KHÔNG** vượt Raw (+3.31, CI chứa 0); đạt **54%** điểm Oracle so với **84%** ở tiếng Việt |
| `age_comparison_vimqa.json` | 1003 | 7B | **Template so tuổi** - 9.8% của tập (không phải 12.8%); F1=0 44.9% vs 37.1% nhưng F1 TB **cao hơn**; ở 0.5B thì sụp 92.1% |
| `hop_efficiency_vimqa.json` | 1003 | 7B | **Hai bảng phụ lục của paper gốc** (Table 4 & 6): F1 theo số câu vàng; giảm token 79.4% nhưng end-to-end **chậm hơn 1.57×** |
| `ablation_vimqa500_7b.json` | 500 | 7B | **Ablation n=500** - bác bỏ kết quả bit-identical của `max_iter`; đường cong $k$ đảo thứ hạng so với n=50 |
| `damage_vs_f1_vimqa_full.json` | 1003 | 7B | ρ = −0.067, CI [−0.124, −0.004] - tách khỏi 0 nhưng **không đáng kể** |
| `damage_vs_f1_vimqa200_dual.json` | 200 | 0.5B | Bản chạy lại đúng scorer |
| `damage_vs_f1_vimqa200.json` | 200 | 0.5B | **Bản đầu, SAI scorer** - giữ để đối chiếu |

### Đo trên CPU, không cần reader

| File | Đo gì |
|---|---|
| `fertility_multilingual.json` | Fertility vi/th/hi/zh trên WMT24++ |
| `fertility_wmt24pp.json` | Fertility vi/en, 930 cặp |
| `fertility_vimqa_contexts.json` | Fertility trên chính ngữ cảnh VimQA |
| `effective_ratio_vimqa.json` | Tỉ lệ nén hiệu dụng theo 3 tokenizer |
| `syllable_damage_vimqa.json` | Hư hại ngữ pháp, tiếng Việt |
| `homograph_vimqa.json` | Bỏ dấu xoá bao nhiêu phân biệt nghĩa - bằng chứng cho EXP-2 |
| `ambiguity_vs_f1_vimqa.json` | Phép kiểm giả: mật độ mơ hồ **không** phân biệt được câu hỏi |
| `beats_oracle_vimqa.json` | 9,8% câu IterCOMP vượt Oracle **dù còn dấu** - phần lớn do nhãn thiếu |
| `syllable_damage_musique.json` | Cùng phép đo, đối chứng tiếng Anh |
| `damage_curve_vimqa.json` | Quét mức nén để so công bằng |
| `damage_curve_musique.json` | Cùng phép quét, tiếng Anh |
| `loop_evidence.json` | Vòng lặp có thật sự chạy không |
| `vimqa_50_boolnorm.json` | Ảnh hưởng của chuẩn hoá boolean |
| `vimqa_50_hf05b_dual.json` | Bản dual scorer, reader 0.5B |
| `vimqa_20_hf.json` | Chạy thử ban đầu |

Muốn kiểm mọi con số trong báo cáo khớp các file này:

```bash
python scripts/verify_report_numbers.py   # 73 số
python scripts/check_repo.py              # tài liệu có lỗi thời không
```
