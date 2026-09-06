# `research/` — thí nghiệm khám phá

**Thư mục này KHÔNG sinh bảng nào cho báo cáo** — mọi bảng trong
[`report_en/report.pdf`](../report_en/report.pdf) đến từ [`scripts/`](../scripts/).
Đây là các thí nghiệm phụ (một số kết quả đã vào phần thảo luận/phụ lục báo cáo).

| File | Câu hỏi | Trạng thái |
|---|---|---|
| `exp1_oracle_stopping.py` | Bộ dừng **hoàn hảo** thêm bao nhiêu F1? | ✅ trần **−0,17 F1** (vào phụ lục "What This Reimplementation Does Not Settle") |
| `exp2_undiacritised.py` | Vòng lặp chạy trên **tiếng Việt không dấu**? | ✅ nén **vượt** gold |
| `measure_span_metric.py` | `f1_span` đáng bao nhiêu? | ✅ **+1,3 F1** |
| `measure_boolean_bug_impact.py` | Lỗi đảo cực boolean ảnh hưởng gì? | ✅ đã chạy |
| `measure_stop_error.py`, `sweep_stopping.py` | Quét/đo ngưỡng dừng | ⏹ không dùng — EXP-1 đã đóng hướng này |

**Ghi chú:** hướng "bộ dừng thích ứng" ban đầu định làm đóng góp, nhưng tra tài
liệu thấy TASR (KDD 2026) đã công bố đúng thiết kế đó trước → hạ thành *baseline*
so sánh, không phải đóng góp.

## Chạy

Cùng môi trường với `scripts/` (xem [README chính](../README.md) Mục 3); mọi
script có `--help`.

```bash
python research/exp1_oracle_stopping.py --dataset vimqa --limit 200 \
  --reader hf --reader-model Qwen/Qwen2.5-7B-Instruct --load-4bit \
  --out results/exp1_oracle_vimqa.json
```

EXP-1 và EXP-2 đã được nối sẵn trong `../notebooks/colab_reproduce.ipynb`.
