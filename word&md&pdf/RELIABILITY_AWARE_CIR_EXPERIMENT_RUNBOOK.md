# Runbook kiểm soát thực nghiệm Reliability-Aware CIR

> Nguồn chỉ đạo: `word&md&pdf/Thucnghiem_tiep_CIR.docx`.
> Trạng thái xác minh 2026-09-09: v2 Dress seed42 hoàn tất, Gate 2 PASS;
> tiếp theo xác nhận Dress seeds7/123. V1 giữ read-only, fail Gate 0.
> Ngày khóa protocol v2: 2026-09-06.

## 1. Quyết định chuyển hướng

Hướng chính chuyển từ **uncertainty-guided retrieval** sang **reliability-aware
retrieval**. Không tiếp tục tối ưu Full HUG, variance head, dynamic uncertainty
weighting hoặc adaptive fusion từ uncertainty hiện tại. Backbone chuẩn là Point
deterministic đã ổn định trên Fashion-IQ.

Mục tiêu mới không phải làm recall tăng ngay, mà trả lời hai câu hỏi:

1. Query nào có khả năng làm Point retrieval thất bại?
2. Xác suất failure dự đoán được có calibrated đủ tốt để selective retrieval và,
   chỉ sau khi qua gate, điều khiển adaptive fusion hay không?

## 2. So sánh hướng cũ và hướng mới

| Thành phần | Hướng cũ: HUG/uncertainty | Hướng mới: reliability-aware |
|---|---|---|
| Backbone chính | Full HUG e2e hoặc Frozen-Point HUG | Point deterministic đã xác nhận |
| Tín hiệu phụ | Variance `sigma_r/sigma_t/sigma_m/sigma_q` | Xác suất `P(failure@k | query)` |
| Loss chính | HC + FC + Cord; các nhánh U1–U3/robust loss | Point InfoNCE được giữ cố định + BCE failure head |
| Mục tiêu đánh giá | Recall và calibration của uncertainty | Failure discrimination, calibration, risk–coverage |
| Trạng thái bằng chứng | HUG e2e collapse; Frozen uncertainty gần ngẫu nhiên | v2 seed42 AUROC 0.7256, gate pass; cần xác nhận seed7/123 |
| Adaptive fusion | Không được phép từ uncertainty chưa calibrated | Chỉ mở nếu reliability qua gate |

Hướng mới kế thừa toàn bộ nguyên tắc paired corruption, target/gallery sạch,
severity 0 identity và per-query audit. Nó không kế thừa giả định rằng variance
HUG là reliability.

## 3. Taxonomy và phạm vi v2

| ID | Định nghĩa | Vai trò |
|---|---|---|
| `point` | Mean-only symmetric InfoNCE, Q-Former trainable, B32 | Backbone và bộ sinh failure label |
| `point_reliability_probe` | Load Point best; freeze toàn bộ Point; học query-only Reliability Head | Mô hình chính của v2 |
| `point_reliability_joint` | Point + reliability head cùng cập nhật bằng retrieval + failure loss | Chưa chạy; chỉ là ablation sau probe |
| `adaptive_fusion` | Fusion dùng reliability probability | Chưa implement; chỉ mở sau gate |

V1 cố ý dùng `point_reliability_probe`. Với backbone frozen, failure label do
Point sinh ra là stationary và clean retrieval không thể drift. Tổng objective
được log là:

`L_total = stopgrad(L_retrieval) + lambda_failure * L_BCE`.

`L_retrieval` là invariant monitor; gradient chỉ cập nhật Reliability Head. Đây
là cách triển khai an toàn nhất cho chỉ đạo “dùng Point làm backbone ổn định”.
Joint training chỉ hợp lệ khi có cơ chế relabel sau mỗi vòng; không bật âm thầm
vì label cũ sẽ không còn mô tả failure của retrieval head đã thay đổi.

## 4. Fashion-IQ-C schema v2

Fashion-IQ-C là manifest mở rộng của Fashion-IQ, không phải dataset ảnh sao chép.
Ảnh nguồn không bị sửa. Mỗi corruption được tái tạo từ `query_index`, `seed`,
`corruption_type` và `severity`.

### 4.1 Bất biến

- Chỉ corrupt query: reference image hoặc modification text.
- Target image và toàn bộ gallery luôn sạch.
- Severity 0 là identity và chỉ ghi một lần cho mỗi query.
- Cùng query/seed/type phải tái tạo byte/text giống nhau.
- Severity cao hơn dùng cùng random priority; word deletion và vùng occlusion
  được thiết kế nested.
- Candidate/reference image vẫn bị loại khỏi gallery khi xếp hạng.

### 4.2 Corruption suite

| Modality | Type | Severity 1/2/3/4 | Nhóm diễn giải |
|---|---|---|---|
| Image | `gaussian_blur` | radius 0.5/1/2/4 | Nuisance, semantic-preserving |
| Image | `occlusion` | area 10/20/35/50% | Nuisance, semantic-preserving đến mức hợp lý |
| Text | `word_deletion` | 10/20/35/50% token | Information loss |
| Text | `spelling_error` | 5/10/20/30% vị trí có thể transpose | Surface noise |
| Text | `synonym_replacement` | 25/50/75/100% từ map được | Semantic-preserving lexical shift |
| Text | `semantic_contradiction` | 25/50/75/100% thuộc tính map được; fallback negation | Semantic-changing stress test |

`semantic_contradiction` không được gộp vô điều kiện với nuisance corruptions khi
claim robustness: nó có thể thay đổi truy vấn đúng về mặt ngữ nghĩa trong khi
target gốc vẫn giữ nguyên. Báo cáo riêng stress-test này và báo thêm pooled score
có/không có contradiction.

### 4.3 Schema bắt buộc mỗi row

`query_id`, `query_index`, `split`, `category`, `candidate_id`, `target_id`,
`target_image`, `clean_text`, `corrupted_text`, `modality`, `corruption_type`,
`severity`, `corruption_seed`, `corruption_metadata`, `label_checkpoint`,
`retrieval_rank`, `hit@10`, `hit@50`, `failure_k`, `failure_label`.

Nhãn chính: `failure_label = 1[retrieval_rank > 10]`. Chạy sensitivity `k=50`
trong output root khác; không trộn nhãn k=10 và k=50 trong cùng run.

## 5. Kiến trúc Reliability Head

Đầu vào là token fusion deterministic `mu_q` của Point, shape `[B,K,D]`. Head
không đọc target và không đọc gallery, nên probability có thể dùng trước retrieval.

1. Tính mean token và token standard deviation.
2. Ghép thành vector `2D`, LayerNorm.
3. MLP `2D -> 256 -> 1`, GELU, dropout 0.1.
4. Sigmoid logit thành xác suất failure.

Không dùng `sigma_*`, uncertainty estimator hoặc dynamic weighting của HUG trong
forward path.

## 6. Protocol theo gate

### Gate 0 — preflight/identity

- Compile toàn bộ file mới, Bash syntax và unit tests phải pass.
- Point checkpoint tồn tại và recipe thuộc họ Point.
- Fashion-IQ train/val caption và image split tồn tại.
- Severity 0 text/ảnh là identity; deterministic replay pass.
- `changed_rate` của synonym replacement phải được báo; nếu dưới 90%, mở rộng
  synonym map trước pilot chính và tăng schema version.

### Phase 1 — Dress seed42, tạo nhãn

Tạo hai manifest độc lập bằng cùng Point Dress seed42:

- `train`: dùng fit Reliability Head.
- `val`: tuyệt đối chỉ dùng chọn checkpoint/đánh giá.

Không sinh label từ `point_robust`, HUG hoặc checkpoint khác trong pilot v2.
Không dùng val row để tối ưu tham số, threshold hoặc temperature.

### Gate 1 — failure pattern

Trước khi train head, kiểm tra robustness CSV:

- Clean recall khớp JSON Point hiện có trong sai số số học.
- Recall/failure rate được báo theo corruption × severity.
- Báo riêng image, text, nuisance-only và semantic contradiction.
- Không kết luận “text là nguồn lỗi chính” chỉ từ một corruption; dùng paired
  query deltas và nhiều loại text corruption.

### Phase 2 — Reliability probe

Default pilot:

- Point backbone frozen.
- Batch 32; 10 epochs; AdamW; head LR `1e-4`; weight decay `1e-4`.
- `lambda_failure=1.0`; automatic positive-class weight từ train manifest.
- Model selection bằng validation AUROC; early stopping patience 3.
- Train và val dùng deterministic eval transform, không random augmentation.

### Gate 2 — reliability đủ dùng

Gate adaptive fusion chỉ pass khi đồng thời:

- pooled validation AUROC `> 0.70`;
- AUPRC cao hơn failure prevalence ít nhất `0.10` tuyệt đối;
- ECE `<= 0.10` (10 bins);
- risk tại coverage 50% thấp hơn risk toàn bộ ít nhất 20% tương đối;
- clean retrieval không đổi (bắt buộc đúng vì backbone frozen);
- không có dấu hiệu chỉ học corruption type: báo metric clean-only,
  nuisance-only, từng corruption và contradiction riêng.

Operational condition audit v2 được khóa trước GPU run v2:

- clean-only AUROC `>=0.60`;
- image và text AUROC đều `>=0.60`;
- nuisance-only AUROC `>0.70`;
- macro AUROC trong từng corruption × severity `>=0.65`;
- pooled AUROC hơn baseline chỉ dùng condition prevalence ít nhất `0.05`;
- train/val labels và Reliability checkpoint phải trỏ đúng cùng Point backbone.

Nếu pooled AUROC pass nhưng clean-only hoặc một modality gần ngẫu nhiên, kết luận
chỉ là corruption detector, chưa phải general query reliability; không mở fusion.

### Gate 3 — mở rộng

Chỉ sau Dress seed42 pass Gate 2:

1. Dress seeds `42,7,123` với manifest seed paired.
2. Báo mean±std và bootstrap CI paired.
3. Sau đó mới chạy Shirt/Toptee.
4. Adaptive fusion là phase riêng, có control Point và reliability probe; không
   sửa artifact v1 hay đổi gate hậu nghiệm.

Nếu Dress seed42 fail: thử calibration đơn giản trên val-calibration split
(temperature/isotonic phải fit ở split riêng), head capacity ablation và kiểm tra
label noise. Không quay lại Full HUG như một cách vá score.

## 7. Metrics và bảng bắt buộc

### Retrieval/robustness

- R@10, R@50 theo type × severity.
- Failure rate, rank distribution, changed rate.
- Paired delta so với clean cho cùng query.

### Reliability

- AUROC, AUPRC, ECE, Brier.
- Risk–coverage curve và AURC.
- Failure prevalence để diễn giải AUPRC.
- Overall, clean-only, modality, corruption type, severity; contradiction riêng.

Không chọn threshold rồi báo trên cùng val set. Nếu cần operating point, tách
validation thành calibration/tuning và final test protocol trước khi chạy.

## 8. Artifact và output root

Không ghi đè protocol cũ.

- V1 đã chạy giữ read-only tại `results/reliability_v1/`, `checkpoints/reliability_v1/`.
- Labels/CSV/tables v2: `results/reliability_v2/<category>/seed<seed>/`
- Checkpoint v2: `checkpoints/reliability_v2/<category>/seed<seed>/`
- Train log v2: `results/reliability_v2/<category>/seed<seed>/logs/`
- `checkpoint_final.pth` xác nhận run hoàn tất; `checkpoint_last.pth` để resume;
  `checkpoint_best.pth` để evaluate.

Đầu ra gửi GVHD:

1. Fashion-IQ-C JSONL và CSV per-query.
2. Baseline/robustness summary JSON + CSV.
3. Reliability metrics JSON + per-query probability CSV.
4. Checkpoint Point nguồn và reliability checkpoint.
5. Training log.
6. Bảng/đồ thị risk–coverage khi pilot đã chạy.

## 9. Lệnh chuẩn

```bash
source ref/LAVIS/.venv/bin/activate
./scripts/run_reliability_cir.sh preflight
./scripts/run_reliability_cir.sh audit

# Chạy tuần tự, có thể resume; không overwrite manifest/result có sẵn
./scripts/run_reliability_cir.sh build-train
./scripts/run_reliability_cir.sh build-val
./scripts/run_reliability_cir.sh train
./scripts/run_reliability_cir.sh evaluate
./scripts/run_reliability_cir.sh gate

# Hoặc toàn bộ Dress seed42 pilot
./scripts/run_reliability_cir.sh pilot
```

Smoke test giới hạn query phải dùng output root tạm riêng:

```bash
MAX_QUERIES=8 \
RESULT_ROOT=results/reliability_v2_smoke/dress/seed42 \
CHECKPOINT_ROOT=checkpoints/reliability_v2_smoke/dress/seed42 \
./scripts/run_reliability_cir.sh pilot
```

Không dùng smoke artifact để báo kết quả khoa học.

## 10. Thay đổi code v2

| File | Thay đổi | Lý do |
|---|---|---|
| `modules/fiqc.py` | Schema 2, synonym map theo caption thật, audit changed-rate deterministic | Vượt và chặn Gate 0 trước GPU |
| `modules/reliability.py` | AUPRC threshold-grouped khi score tie | Không làm AUPRC shortcut baseline lạc quan giả |
| `models/reliability_model.py` | Wrapper Point + query-only head; freeze Point | Giữ retrieval backbone ổn định |
| `eval/reliability.py` | Preflight audit, nuisance subsets, condition/clean baselines, gate tổng hợp, risk CSV | Chứng minh head không chỉ nhận diện corruption |
| `train_reliability.py` | Bắt buộc schema 2 và cùng Point checkpoint | Không trộn label/artifact v1-v2 |
| `scripts/run_reliability_cir.sh` | Root v2; workflow `audit -> build -> train -> evaluate -> gate` | Giữ v1 read-only và fail-fast |
| `tests/test_reliability.py` | Thêm test schema/coverage và tie-safe AUPRC | Chặn regression v2 |

## 11. Trạng thái xác minh

- V1 Dress seed42 hoàn tất: AUROC `0.7205`, AUPRC `0.8037`, ECE `0.0737`,
  risk reduction@50% `24.47%`; không promote vì synonym changed-rate dưới 90%.
- V2 `pytest -q tests/test_reliability.py`: 8 passed; `py_compile` và Bash syntax pass.
- V2 corruption audit thật: train `98.50%`, val `98.76%`, đều qua Gate 0.
- Audit JSON nằm trong `results/reliability_v2/dress/seed42/labels/`.
- V2 đã hoàn tất label/train/evaluate, đủ best/last/final. 149625 train rows,
  50425 val rows; clean 47.99/70.90. AUROC 0.7256, AUPRC 0.8114 (prevalence
  0.6236), ECE 0.0680, Brier 0.2091, AURC 0.4530, giảm risk@50% 24.91%.
- `adaptive_fusion_gate.passed=true`; numeric/condition/corruption audit đều pass.
  Within-condition AUROC 0.7099, nuisance 0.7233, clean 0.6965. Contradiction
  ECE 0.1375 cần báo riêng. Best epoch1, early stop epoch4, có overfit.
- Được xác nhận Dress seeds7/123, chưa có artifact hai seed này tại ngày kiểm tra.
  Adaptive fusion vẫn đóng đến khi multi-seed và complement analysis hoàn tất.
- Val dùng chọn checkpoint/đánh giá, không phải independent test. Không tune
  calibration trên cùng tập rồi báo test; khóa split riêng trước ablation mới.
- Chuyển máy: theo `README.md` và `docs/MIGRATION.md`; SHA-256/mapping root chỉ
  phục vụ vận hành, không thay đổi schema, corruption RNG hoặc gate đã khóa.
