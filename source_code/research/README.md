# `research/` - hướng phát triển sau đồ án

**Thư mục này KHÔNG sinh số nào cho báo cáo.** Mọi bảng trong
[`report_en/report.pdf`](../report_en/report.pdf) đến từ [`scripts/`](../scripts/).

Phân biệt cho rõ:

| | [`scripts/`](../scripts/) | `research/` (thư mục này) |
|---|---|---|
| Vai trò | Sinh số cho **bài nộp** | Khám phá hướng mới |
| Báo cáo dùng? | ✅ mọi bảng | ❌ không bảng nào |
| Đã có kết quả? | ✅ mọi script | **4/6** đã chạy; 2 cái còn lại không chạy nữa |

## Các script

| File | Câu hỏi nó trả lời | Trạng thái |
|---|---|---|
| `exp1_oracle_stopping.py` | Nếu bộ dừng **hoàn hảo** thì được thêm bao nhiêu F1? | ✅ đã chạy - trần **−0,17 F1** |
| `exp2_undiacritised.py` | Vòng lặp còn chạy được trên **tiếng Việt không dấu**? | ✅ đã chạy - nén **vượt** gold |
| `measure_stop_error.py` | Bộ `is_answerable` dừng sai bao nhiêu lần? | **không chạy nữa** - EXP-1 đóng hướng này |
| `sweep_stopping.py` | Quét ngưỡng dừng | **không chạy nữa** - EXP-1 đóng hướng này |
| `measure_span_metric.py` | `f1_span` đáng bao nhiêu điểm? | ✅ đã chạy: **+1,3 F1** |
| `measure_boolean_bug_impact.py` | Lỗi đảo cực boolean ảnh hưởng gì? | ✅ đã chạy |

## Vì sao hướng "bộ dừng thích ứng" bị hạ cấp

Ba script đầu (`measure_stop_error`, `sweep_stopping`, và [`stopping.py`](../src/itercomp/stopping.py))
ban đầu định làm **contribution**: thay phán quyết nhị phân của paper bằng điểm
tin cậy hiệu chuẩn được.

Tra tài liệu sau khi viết xong cho thấy **TASR** (Agent4IR @ KDD 2026) đã công
bố đúng thiết kế đó - đọc biên logit top1−top2, hiệu chuẩn bằng isotonic
regression, ghép với tín hiệu lặp lại - trước 4 tháng. **HALT** đo lỗi dừng sớm
so với gold, **S2G-RAG** kiểm toán ma trận nhầm lẫn của bộ phán quyết.

Nên `stopping.py` được viết lại thành **baseline để so sánh** (xem `BASELINE_RULES`
trong file đó), không phải đóng góp. Bài học ghi trong commit `0a2a835`: nên tra
tài liệu **trước** khi viết ba module, không phải sau.

## Phát hiện còn sống

Ablation cho thấy **bỏ hẳn bước kiểm tra đủ bằng chứng làm F1 TĂNG** từ 33,03
lên 36,64 trên VimQA - tức cơ chế dừng của paper đang *mất* 3,6 điểm, đổi lấy
việc đọc 16% ngữ cảnh thay vì 53%.

`exp1_oracle_stopping.py` được viết để trả lời câu chặn: nếu bộ dừng hoàn hảo
thì trần là bao nhiêu? Script tự tuyên bố phán quyết - trần dưới 3 F1 thì đóng
hướng, tốn 30 phút GPU thay vì trăm giờ. Kết quả này đã vào báo cáo ở
Phụ lục L ("What This Reimplementation Does Not Settle").

## Chạy

Cùng môi trường với `scripts/` - xem [README chính](../README.md) Mục 3.
Mọi script đều có `--help`.

```bash
python research/exp1_oracle_stopping.py --dataset vimqa --limit 200 \
  --reader hf --reader-model Qwen/Qwen2.5-7B-Instruct --load-4bit \
  --out results/exp1_oracle_vimqa.json
```

Notebook [`colab_reproduce.ipynb`](../notebooks/colab_reproduce.ipynb) đã nối
sẵn EXP-1 và EXP-2 ở hai ô cuối.
