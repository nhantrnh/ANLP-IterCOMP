# Artefacts cho bài SOICT 2026

Bản chụp *chỉ* những gì bài báo dùng: 22 file kết quả, ba prompt, và bản
`paper.tex` / `paper.pdf` tương ứng. Mọi con số in trong bài đều truy được về
đúng một file dưới `results/`.

Mã nguồn **không** nằm ở đây — nó ở repo chính
(<https://github.com/nhantrnh/IterCOMP-X>), thư mục `src/`, để một nơi duy nhất
là nguồn chuẩn thay vì hai bản dễ lệch nhau.

Chụp ngày 21/09/2026, tại commit của repo lúc tạo. File kết quả là output thô của
lần chạy, không chỉnh tay.

## Cấu hình chung

Reader `Qwen/Qwen2.5-7B-Instruct` ở 4-bit, giải mã greedy (`do_sample=False`),
scorer `dual` với bge-m3, `lambda=0.6`, `max_iter=5`. Ngoại lệ được ghi rõ ở bảng
dưới. Mỗi file tự mang `config` của nó — đọc khối đó thay vì tin bảng này.

## Bảng nào lấy từ file nào

| Vị trí trong bài | File | n | Ghi chú |
|---|---|---|---|
| Bảng 1 (fertility 4 tokeniser) | `fertility_wmt24pp.json` | 930 cặp | WMT24++ en-vi |
| Hình 2 (fertility 4 ngôn ngữ) | `fertility_multilingual.json` | 400 cặp/ngôn ngữ | Nguồn của 2.49x |
| §3.1 (context VimQA 1110 từ / 2364 token) | `fertility_vimqa_contexts.json` | 200 | |
| §3.1 (tỉ lệ hiệu dụng) | `effective_ratio_vimqa.json` | 50 | |
| **Bảng 5 (kết quả chính)** | `vimqa_full_7b.json` | 1003 | VimQA |
| | `musique_1500_7b.json` | 1500 | |
| | `hotpotqa_1500_7b.json` | 1500 | |
| | `2wiki_1500_7b.json` | 1500 | |
| Bảng 5 (hàng 6 baseline) | `baselines_vimqa_200_7b.json` | 200 | |
| Bảng 5 (CompAct) | `compact_vimqa_200_7b.json` | 200 | |
| **Bảng 6 (so bài gốc)** | `musique_500_7b.json` | 500 | cột `ours` đọc ở n=50 đầu |
| | `vimqa_200_7b.json` | 200 | cột VimQA |
| **Bảng 6 (mất dấu)** | `exp2_undiacritised_500_7b.json` | 500 | chạy lại sau khi sửa bug gold |
| **Bảng 8 (phân loại lỗi)** | `hotpotqa_500_05b.json` | 500 | cột 0.5B |
| | `hotpotqa_500_7b.json` | 500 | cột 7B |
| §5.2 (quét k khớp mức nén) | `ablation_musique200_k.json` | 200 | nguồn của 3.14 passes |
| §5.3 (latency, số vòng lặp) | `hop_efficiency_vimqa.json` | 1003 | VimQA 2.88s -> 4.52s |
| §5.3 (max_iter) | `ablation_vimqa500_7b.json` | 500 | nguồn của 1.58 passes |
| §5.4 (đảo chiều HotpotQA) | `hotpotqa_500_05b.json` | 500 | 37.88 vs 30.22 |
| §5.5 / §6 (dừng sớm) | `exp1_oracle_vimqa_200_7b.json` | 200 | |
| | `exp1_oracle_musique_200_7b.json` | 200 | |
| §6 (template so tuổi) | `age_comparison_vimqa.json` | 1003 | 9.8% tập, 44.9% zero |
| §5.1 (2Wiki n=500 phụ) | `2wiki_500_7b.json` | 500 | |

## Cảnh báo về hai file dễ nhầm

`exp2_undiacritised_500_7b.json` là lần chạy dùng cho bài: n=500, sau khi sửa
lỗi `strip_row` không strip `supporting_facts["title"]` (lỗi này từng khiến
48/200 ngữ cảnh gold rỗng). Kiểm bằng `rows_without_gold == 0`.

Hai bản n=200 (`..._200_7b.json` chưa sửa và `..._200_7b_fixed.json` đã sửa)
**không** dùng cho bảng nào: ở n=200 không khoảng tin cậy nào tách khỏi 0, kết
luận chỉ dứt khoát khi lên n=500.

`hotpotqa_500_3b.json` tồn tại trong repo nhưng **không dùng**: lần chạy đó cho
Oracle thấp hơn Raw, một thứ tự không hợp lệ, nên bị loại khỏi mọi bảng.

## Chạy lại

Lấy mã nguồn từ repo chính rồi:

```bash
cd src
pip install -r requirements.txt
python scripts/run_eval.py --dataset vimqa --limit 1003 \
  --reader hf --reader-model Qwen/Qwen2.5-7B-Instruct --load-4bit \
  --itercomp-llm hf --scorer dual --lam 0.6 --max-iter 5 \
  --methods raw,oracle,llmlingua2,itercomp --out results/vimqa_full_7b.json
```

Các script khác: `fertility_multilingual.py` (Hình 2), `effective_ratio.py`
(§3.1), `run_ablation.py` (Bảng ablation), `exp2_undiacritised.py` (Bảng 7),
`exp1_oracle_stopping.py` (§5.5). Mỗi script `--help` liệt kê cờ của nó.

Chạy inference cần GPU; các số fertility và phân tích lỗi chạy được trên CPU.
