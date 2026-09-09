# Tiến độ thực nghiệm theo `Thucnghiem_tiep_CIR.docx`

> Cập nhật: 2026-09-09; v2 Dress seed42 đã hoàn tất và gate pass.
> Phạm vi kết quả hiện có: Dress seed42.
> Nguồn protocol chi tiết: `RELIABILITY_AWARE_CIR_EXPERIMENT_RUNBOOK.md`.

## 1. Mục tiêu nghiên cứu

- Hướng reliability-aware CIR đã được triển khai đúng mục tiêu: dự đoán
  `P(failure@10 | query)` thay cho variance/uncertainty cũ.
- Trạng thái: **đã hoàn thành ở mức pilot**, chưa có kết luận multi-seed/category.

## 2. Hiện tại

- Backbone là Point deterministic; toàn bộ Point được freeze khi học Reliability Head.
- Không tiếp tục Full HUG và không dùng uncertainty cũ để điều khiển fusion.
- V1 được giữ read-only. V2 dùng output root mới để không ghi đè artifact.

## 3. Dataset sử dụng

### Giai đoạn 1 — Xây dựng corruption benchmark

- Đã có đủ blur, occlusion, word deletion, spelling error, synonym replacement
  và semantic contradiction; chỉ query bị corrupt, target/gallery luôn sạch.
- JSONL/CSV lưu đủ query id, category, clean/corrupted text, corruption, severity,
  target, rank, hit@10, hit@50 và failure label.
- V1 hoàn tất nhưng synonym changed-rate chỉ `88.05%` train và `88.75%` val.
- V2 nâng schema lên 2, mở rộng lexical map và audit trước GPU. Kết quả audit:

| Split | Synonym changed-rate | Gate >=90% |
|---|---:|---|
| Train | 98.50% | Pass |
| Val | 98.76% | Pass |

### Giai đoạn 2 — Phân tích failure pattern

- Đã hoàn thành cho V1 Dress seed42: clean `47.99/70.90` R@10/R@50.
- Các nuisance text corruption làm recall giảm mạnh hơn hai image corruption;
  semantic contradiction được tách riêng vì làm đổi nghĩa.
- V2 đã build lại rank/failure labels: 149625 train rows, 50425 val rows.
  Clean 47.99/70.90. R@10 severity4: blur45.02, occlusion40.51,
  spelling17.40, deletion24.69, synonym37.58; contradiction21.12 báo riêng.

### Giai đoạn 3 — Reliability Estimator

- Đã triển khai query-only Reliability Head trên fusion representation của Point.
- V1 best checkpoint ở epoch 1; train tiếp gây overfit nên checkpoint selection
  và early stopping phải được giữ nguyên.
- V2 đã train xong, đủ best/last/final. Best epoch1, dừng epoch4;
  giữ lịch/selection hiện tại khi xác nhận seed7/123, không tăng epoch hậu nghiệm.

## 4. Loss function

- Đúng công thức đã khóa:
  `L_total = stopgrad(L_retrieval) + lambda_failure * BCE(failure)`.
- Failure label chính là `1[rank > 10]`; không trộn với failure@50.
- Trainer V2 từ chối manifest sai schema hoặc labels không đến từ cùng Point checkpoint.

## 5. Đánh giá Reliability

Kết quả V2 Dress seed42 (V1 xem runbook lịch sử):

| Metric | Kết quả | Gate |
|---|---:|---|
| AUROC | 0.7256 | Pass |
| AUPRC | 0.8114 (prevalence 0.6236) | Pass |
| ECE | 0.0680 | Pass |
| Risk reduction tại 50% coverage | 24.91% | Pass |

- V2 bổ sung nuisance-only, clean-only, modality/condition metrics, baseline chỉ
  dùng condition prevalence, baseline lặp clean score và risk-coverage CSV.
- Gate V2 chỉ pass nếu numeric gate, synonym coverage, backbone match và condition
  audit cùng pass. JSON V2 hiện có `adaptive_fusion_gate.passed=true`.
- Within-condition AUROC 0.7099; nuisance 0.7233. Contradiction ECE0.1375,
  báo riêng. Đây là validation được dùng chọn checkpoint, không phải test độc lập.

## 6. Adaptive Fusion

- **Chưa triển khai và chưa được phép mở.**
- Dress seed42 V2 đã có `adaptive_fusion_gate.passed=true`;
  bước kế tiếp được phép là xác nhận Dress seeds7/123 (chưa chạy).
- Chỉ cân nhắc fusion sau multi-seed và complement analysis; tuyệt đối không dùng
  uncertainty cũ.

## 7. Các file đầu ra cần gửi lại Cô

| Đầu ra | V1 | V2 hiện tại |
|---|---|---|
| Corruption dataset + CSV per-query | Có | Có |
| Baseline results + robustness tables | Có | Có |
| Reliability metrics | Có | Có; gate pass |
| Risk-coverage | Có trong JSON | Có JSON + CSV; chưa có hình báo cáo |
| Model checkpoint | Có best/last/final | Có best/last/final |
| Training logs | Có | Có 4 epochs |

## Note thay đổi code V2

- `modules/fiqc.py`: schema 2, synonym map mới, audit changed-rate.
- `modules/reliability.py`: AUPRC tie-safe.
- `eval/reliability.py`: fail-fast Gate 0, nuisance/shortcut/condition audit, gate
  tổng hợp và risk-coverage CSV.
- `train_reliability.py`: bắt buộc schema 2 và cùng Point label checkpoint.
- `scripts/run_reliability_cir.sh`: mặc định root `reliability_v2`, thêm `audit`
  và `gate`; workflow không ghi đè V1.
- `tests/test_reliability.py`: 8 CPU tests pass.

## Bước chạy tiếp theo

```bash
source scripts/project_env.sh
SEED=7 bash scripts/run_reliability_cir.sh pilot
SEED=123 bash scripts/run_reliability_cir.sh pilot
```

Lệnh trên xác nhận Dress V2 với hai seed còn lại, giữ nguyên protocol. Sau đó
tổng hợp mean±std/paired query-bootstrap CI, chưa mở fusion ngay. Khi chuyển máy,
khôi phục theo README và `docs/MIGRATION.md` trước khi chạy.
