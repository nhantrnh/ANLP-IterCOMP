# Nguồn gốc kết quả - cái nào dùng, cái nào bỏ, vì sao

Đối chiếu **toàn bộ** output Kaggle với `results/` local ngày 03/09/2026.
**Cập nhật cuối cùng phải chạy lại sau khi mọi kernel kết thúc** - bảng này
từng lỗi thời chỉ vài phút sau khi viết, vì một kernel được đẩy sau đó.
Mỗi file GPU khớp **byte-to-byte** với đúng một kernel; không có file nào
mồ côi, không có file nào trộn giữa hai lần chạy.

## 1. Kernel Kaggle - trạng thái và số phận

| Kernel | Chạy lần cuối | Trạng thái | Kết quả |
|---|---|---|---|
| `itercomp-gpu-tasks-567` | 26/08 | ✅ COMPLETE | **DÙNG** - bảng chính n=1003, EXP-1, EXP-2 |
| `itercomp-damage-vs-f1-7b` | 26/08 | ✅ COMPLETE | **DÙNG** - 1c′ reader 7B |
| `itercomp-1c-double` | 27/08 | ✅ COMPLETE | **DÙNG** - 1c″ ρ ở n=1003 |
| `itercomp-baselines` | 01/09 | ✅ COMPLETE | **DÙNG** - 5 baseline (v6; v1–v5 fail) |
| `itercomp-ablation-500` | 02/09 | ✅ COMPLETE | **DÙNG** - ablation n=500 |
| `itercomp-musique-500` | 02/09 | ✅ COMPLETE | **DÙNG** - đối chứng tiếng Anh n=500 |
| `itercomp-compact` | 02/09 | ✅ COMPLETE | **DÙNG** - CompAct (v3; v1–v2 fail) |
| `itercomp-musique-k` | 03/09 | ✅ COMPLETE | **DÙNG** - quét $k$ MuSiQue, so ở cùng mức nén |
| `itercomp-3datasets` | 03/09 | ✅ COMPLETE | **DÙNG** - 2Wiki + HotpotQA n=500; thứ tự paper **không** tái hiện ở hai tập này |
| `itercomp-damage-500-cpu` | 03/09 | ✅ COMPLETE | **DÙNG** - đối chứng hư-từ n=500 (CPU); dấu giữ nguyên, hai chênh lệch bằng nhau |
| `itercomp-adaptive-k-dual` | 27/08 | ⚠️ CANCELLED | **BỎ** - số đã có từ CPU, xem §3.4 báo cáo |
| `itercomp-damage-full-cpu` | 27/08 | ⚠️ CANCELLED | **BỎ** - bản trùng của `itercomp-1c-double` |
| `itercomp-vimqa-full-run` | 24/08 | ❌ ERROR | **BỎ** - thay bằng `itercomp-gpu-tasks-567` |

## 2. File kết quả từ GPU (12) - khớp byte với kernel nào

| File | Kernel nguồn |
|---|---|
| `vimqa_full_7b.json` | `itercomp-gpu-tasks-567` |
| `vimqa_200_7b.json` | `itercomp-gpu-tasks-567` |
| `exp1_oracle_vimqa_200_7b.json` | `itercomp-gpu-tasks-567` |
| `exp1_oracle_musique_200_7b.json` | `itercomp-gpu-tasks-567` |
| `exp2_undiacritised_200_7b.json` | `itercomp-gpu-tasks-567` |
| `damage_vs_f1_vimqa200_7b.json` | `itercomp-damage-vs-f1-7b` |
| `damage_vs_f1_vimqa_full.json` | `itercomp-1c-double` |
| `baselines_vimqa_200_7b.json` | `itercomp-baselines` |
| `vimqa_200_7b_pinned.json` | `itercomp-baselines` |
| `ablation_vimqa500_7b.json` | `itercomp-ablation-500` |
| `musique_500_7b.json` | `itercomp-musique-500` |
| `compact_vimqa_200_7b.json` | `itercomp-compact` |
| `ablation_musique200_k.json` | `itercomp-musique-k` |
| `2wiki_500_7b.json` | `itercomp-3datasets` |
| `hotpotqa_500_7b.json` | `itercomp-3datasets` |
| `damage_curve_vimqa_500.json` | `itercomp-damage-500-cpu` |
| `damage_curve_musique_500.json` | `itercomp-damage-500-cpu` |

35 file còn lại tính trên **CPU** (fertility, hư hại hư từ, đồng âm, ablation
adaptive-k, hiệu quả, template so tuổi…) - không cần GPU, tái tạo bằng
`scripts/*.py` tương ứng.

## 3. Chỗ DUY NHẤT có hai bản cùng cấu hình

`vimqa_200_7b.json` được **ba** kernel sinh ra:

| Kernel | IterCOMP | LLMLingua-2 | Số phận |
|---|---|---|---|
| `itercomp-gpu-tasks-567` | 50.08 | 38.29 | **giữ** → `results/vimqa_200_7b.json` |
| `itercomp-damage-vs-f1-7b` | 50.08 | 38.29 | **bỏ** - cùng số, chỉ khác thứ tự khoá JSON |
| `itercomp-baselines` | **50.17** | **38.80** | **giữ riêng** → `vimqa_200_7b_pinned.json` |

Bản thứ ba **không** thay bản đầu và cũng **không** bị bỏ. Nó chạy lại cùng cấu
hình bên trong phiên baseline, vì **khoảng tin cậy ghép cặp bắt buộc mọi phương
pháp phải thấy cùng bộ câu hỏi trong cùng phiên**. Hai lần chạy lệch **0.09** và
**0.51** F1 - trong nhiễu (ngưỡng n=200 là 7.7). Bài trích **từng bảng theo lần
chạy của chính nó**, và caption bảng baseline giải thích vì sao có 50.17 ở đó
còn 50.08 ở chỗ khác.

## 4. Số n=50 còn trong bài - giữ CÓ Ý THỨC

`scripts/check_superseded.py` bắt mọi con số trích từ lần chạy nhỏ hơn khi đã
có bản lớn hơn. 20 số như vậy được khai kèm lý do (đều là mốc lịch sử nêu cạnh
số mới, ví dụ "35.1 ở n=50 → 26.74 ở n=500"). Số nào **không** khai sẽ bị báo.

## 5. Ba dataset của paper gốc

| Dataset | Đã chạy | Reader | So được với paper? |
|---|---|---|---|
| MuSiQue | n=50, **n=500** | 7B | ✅ có |
| 2WikiMultiHopQA | n=50 | 0.5B | ❌ khác reader |
| HotpotQA | n=50 | 0.5B | ❌ khác reader |

Đã chạy **đủ ba**, nhưng chỉ MuSiQue ở cấu hình so sánh được. Báo cáo ghi rõ
điều này ở mục Dataset.
| `2wiki_1500_7b.json` | `itercomp-2wiki-1500` | 2.66 h T4 | 2Wiki n=1500. Thay thế `2wiki_500_7b.json` trong bảng 4 dataset: ở n=500 hiệu IterCOMP−Raw là −2.69, ở n=1500 là +1.00 - ĐỔI DẤU. 500 câu đầu của lần này trùng đúng 500 câu cũ và tái hiện tới 0.00 F1 (greedy), nên khác biệt đến từ 1000 câu thêm, không phải nhiễu triển khai. File n=500 giữ lại để đối chiếu. |
| `hotpotqa_1500_7b.json` | `itercomp-hotpotqa-1500` | ~2.8 h T4 | HotpotQA n=1500. Thay `hotpotqa_500_7b.json` trong bang 4ds. Ca ba khang dinh GIU DAU va sach: IterCOMP-Raw -7.99->-4.66 (Raw van thang), IterCOMP-LLML2 +14.83->+18.58. 500 cau dau tai hien 0.00. File n=500 giu de doi chieu. |
| `musique_1500_7b.json` | `itercomp-musique-1500` | ~5.65 h T4 | MuSiQue n=1500. KHÉP khẳng định mở cuối: IterCOMP−Raw +3.31 (cắt 0 ở n=500) -> +4.19 CI [+2.01,+6.32] sạch. 500 câu đầu tái hiện 0.00. Bảng 4ds GIỮ n=500 (thang so-với-paper §6.2); file này dùng cho khẳng định §5 + power_check. |
| `hotpotqa_500_05b.json` | `itercomp-hotpotqa-readers` | ~0.5 h T4 | HotpotQA n=500, reader Qwen2.5-0.5B. Bằng chứng reader-dependence (§6.4): IterCOMP−Raw +7.66 (IC>Raw, KHỚP paper), đảo ngược −7.99 ở 7B; chỉ đổi reader. |
| `hotpotqa_500_3b.json` | `itercomp-hotpotqa-readers` | (cùng kernel) | HotpotQA n=500, reader Qwen2.5-3B. KHÔNG dùng: 55% câu Oracle trả 'unknown' (abstention bệnh lý ở 3B-4bit), Oracle<Raw vô lý - loại khỏi mọi khẳng định. |

## Archive - thí nghiệm KHÔNG đưa vào báo cáo

| File | Kernel | Số phận |
|---|---|---|
| `archive/vihotpot_500_7b.json` | `itercomp-vihotpot-500` | **ARCHIVE - không dùng trong báo cáo.** HotpotQA dịch máy sang tiếng Việt (envit5-translation), n=500, reader 7B - dataset multi-hop VN thứ hai (đề I.1 cho phép dịch máy). Kết quả F1*: Oracle 43.18, Raw 34.58, IterCOMP 29.71, LLMLingua-2 27.68. **Central claim KHÔNG tái lập rõ:** IterCOMP−LLMLingua-2 chỉ +2.03, CI [−2.03,+5.98] cắt 0 (bản tiếng Anh: +18.58). Mọi điểm rớt ~25 F1 so với bản Anh → dịch máy làm nhiễu nặng, hại IterCOMP hơn LLMLingua-2. Điều DUY NHẤT tái lập: Raw>IterCOMP (−4.88 sạch, đúng reader-dependence §6.4). Kết luận: giữ làm bằng chứng pipeline chạy được trên dataset multi-hop VN thứ hai (I.1/II.2), nhưng KHÔNG viết vào báo cáo vì tín hiệu bị nhiễu dịch máy lấn át; VimQA (tiếng Việt do người viết) vẫn là bằng chứng tiếng Việt chính (+15.98 sạch). |
