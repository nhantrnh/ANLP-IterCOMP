# Notebook Kaggle

Các kernel chạy phần cần GPU cho từng thí nghiệm (bảng chính, ablation, EXP-1,
EXP-2, reader-swap, …). Chúng **không thay** `../colab_reproduce.ipynb` — đó mới
là notebook nộp bài, tái hiện đủ 8 phần; các kernel ở đây chỉ để chạy lại từng
thí nghiệm riêng lẻ.

## Đẩy lên Kaggle

```bash
python -m kaggle kernels push -p <thư-mục-chứa-notebook-và-kernel-metadata.json>
```

`kernel-metadata.json` cần `"enable_gpu": true`, `"enable_internet": true`, và
`"machine_shape": "NvidiaTeslaT4"` (nếu không Kaggle có thể cấp P100 — không chạy
bitsandbytes 4-bit).

## Mã nguồn

Kernel kéo mã từ **gist public** `nhantrnh/aeec3610fb9541b5f8b388cafd424eb9`.
**Sửa mã trong repo thì phải cập nhật gist**, nếu không kernel chạy bản cũ (dùng
`gh api -X PATCH gists/<id>`; `gh gist edit` báo thành công nhưng không đổi gì).

## Lưu ý vận hành

- **Ghim `transformers==4.45.2`** trong cell pip — Kaggle nay ship 5.0.0 xung đột
  cứng với `llmlingua 0.2.2`. Đây cũng là bản sinh ra mọi số trong báo cáo.
- **GPU vs CPU:** bảng chính / ablation / EXP-1 / EXP-2 cần reader 7B (GPU);
  các phép đo fertility / ambiguity chỉ cần CPU. Đặt `enable_gpu` đúng để khỏi
  phí quota (Kaggle: 2 phiên GPU, 30h/tuần, reset 00:00 UTC thứ Bảy).
