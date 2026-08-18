# HUG-CIR project guide for Codex

File này là context lõi cho session mới. Giữ nó ngắn, cập nhật bằng cách thay
trạng thái cũ thay vì nối thêm nhật ký. Chi tiết thực nghiệm nằm trong runbook,
JSON, checkpoint và log được trỏ ở dưới.

## 1. Mục tiêu dự án

Dự án nghiên cứu Composed Image Retrieval (CIR) dựa trên HUG/BLIP-2, trước hết
trên Fashion-IQ (`dress`, `shirt`, `toptee`). Hai câu hỏi chính:

1. Tái lập retrieval của HUG một cách trung thực, đồng thời nói rõ những chỗ
   phải sửa/ổn định so với paper và public repo.
2. Kiểm tra heterogeneous uncertainty có thực sự dự báo query failure, phản ánh
   modality bị hỏng và giúp robustness hay không; chỉ xây reliability router nếu
   uncertainty đã calibrated.

Không tối ưu một con số đơn lẻ rồi gọi đó là exact reproduction. Mọi kết luận
chính phải dựa trên multi-seed, paired comparison và protocol cố định.

## 2. Thứ tự nguồn tin

Khi tài liệu mâu thuẫn, ưu tiên theo thứ tự:

1. Code/config đang chạy, JSON/checkpoint/log thực tế.
2. `word&md&pdf/SUPERVISOR_EXPERIMENT_RUNBOOK.md` — protocol v2 hiện hành.
3. `word&md&pdf/REPORT_TO_SUPERVISOR.md` và
   `word&md&pdf/WEEK2_WEEK3_FINDINGS.md` — kết quả/audit lịch sử.
4. `word&md&pdf/RESEARCH_PROPOSAL.md`, `U1_U3_RUNBOOK.md`,
   `PERSON2_ROBUSTNESS.md` — rationale và nhánh robustness.
5. `word&md&pdf/TRAINING_NOTES.md` — lịch sử debug; không mặc định là protocol
   mới nhất.
6. `word&md&pdf/2601.11393v2 (1).pdf` — paper gốc.

Thư mục `word&md&pdf/*`, `data/`, `checkpoints/`, `results/`, `wandb/` và `ref/`
đang bị `.gitignore` bỏ qua. Không suy ra “không tồn tại” chỉ từ `git status`, và
không giả định thay đổi tài liệu trong đó đã được Git lưu.

## 3. Taxonomy model/artifact

| ID | Ý nghĩa đúng |
|---|---|
| `legacy` | Pipeline lịch sử, mốc khoảng 6x R@50; chỉ dùng làm reference. |
| `point` | Mean-only symmetric InfoNCE, Q-Former trainable, physical batch 32. Baseline deterministic mạnh. |
| `point_matched` | Cùng Point nhưng batch/lịch train giống `hug_e2e`; control chính cho giới hạn GPU. |
| `hug_e2e` | BLIP-2 init, Q-Former/query tokens trainable, loss `HC + 0.5 FC + 0.1 Cord`; candidate gần mô tả paper nhất. |
| `hug_frozen_point` | Warm-start từ Point rồi freeze mean encoder, chỉ học uncertainty; controlled ablation, không phải full HUG. |

Tên checkpoint cũ `paper_*` thường chỉ `hug_frozen_point`, không được tự động gọi
là full-paper reproduction. Trong báo cáo dùng “corrected Point baseline”,
“corrected/stabilized end-to-end HUG reproduction” và “Frozen-Point HUG
controlled ablation”.

## 4. Các bất biến khoa học

- Paper báo Fashion-IQ Dress `R@10=48.37`, `R@50=71.56`.
- Artifact lịch sử đã xác nhận: Point seed42 khoảng `47.94/70.95`; Frozen-Point
  probabilistic khoảng `48.44/70.9x`. Số gần paper không chứng minh protocol khớp.
- Point dùng in-batch InfoNCE: batch 32 có 31 negatives, batch 4 chỉ có 3.
  HUG HC hiện cũng phụ thuộc số negative theo batch. Gradient accumulation thông
  thường không tạo physical negative pool tương đương batch 32.
- Vì vậy phải báo cả `hug_e2e - point` và `hug_e2e - point_matched`. Không so
  HUG batch nhỏ với Point batch 32 rồi quy toàn bộ chênh lệch cho kiến trúc.
- Với Eq.15 hiện tại, uncertainty của một query là hằng số đối với mọi gallery
  candidate nên không tự đổi ranking; thay đổi rank chủ yếu có thể đến từ gallery
  variance. Đánh giá uncertainty phải dùng AUROC/AUPRC, rank correlation,
  risk-coverage/AURC, severity monotonicity và mismatch, không chỉ Recall.
- Corruption chỉ tác động query; target/gallery phải giữ sạch. Severity 0 phải là
  identity. So sánh model phải dùng cùng manifest/seed.
- Không chạy multi-seed/category sweep nếu Dress seed42 pilot chưa qua gate.
- Không sửa loss/architecture chỉ để khớp con số paper mà không ghi thành ablation.

## 5. Bản đồ code

- `train.py`: CLI, training loop, U1/U2/U3 flags, validation, atomic checkpoint và
  resume model/optimizer/criterion/scheduler/RNG.
- `models/hug_model.py`: mean/uncertainty representation và fusion.
- `models/blip_backbone.py`: BLIP-2/Q-Former feature paths.
- `modules/losses.py`: Point InfoNCE, HC, FC, coordination loss.
- `modules/robustness_training.py`: monotonic calibration, modality dropout, KD.
- `eval.py`: clean retrieval, mean/probabilistic distance.
- `eval/robustness.py`: deterministic corruption, sweep, calibration metrics.
- `eval/modality_reliance.py`: image/text-only và shuffled-modality conditions.
- `eval/summarize_supervisor.py`: bảng CSV cuối và paired deltas.
- `scripts/run_supervisor_experiments.sh`: orchestrator duy nhất của protocol v2.
- `scripts/run_supervisor_person1.sh`: `point` + `hug_frozen_point`.
- `scripts/run_supervisor_person2.sh`: `point_matched` + `hug_e2e`.
- `scripts/run_calibrated_hug.sh`: nhánh U1/U2/U3 cũ; không trộn mặc định vào v2.

## 6. Trạng thái hiện tại — thay phần này, không append lịch sử

Last verified: **2026-08-18**, `main@2b61530` (working tree sạch trước khi tạo
file này).

- Protocol v2, two-person wrappers, resume/checkpoint, modality/robustness và
  summarizer đã implement; compile, Bash syntax, CPU smoke tests và paired-delta
  smoke test đã pass ở phiên trước.
- Chưa có artifact trong `checkpoints/supervisor_protocol_v2/` hoặc
  `results/supervisor_protocol_v2/`: bước tiếp theo là pilot Dress seed42.
- Blocker nhỏ trước pilot: preflight trong `scripts/run_supervisor_experiments.sh`
  vẫn kiểm tra PDF ở root, nhưng PDF đã chuyển vào `word&md&pdf/`. Sửa đường dẫn
  hoặc xác nhận preflight trước khi train.
- Protocol cũ batch 4 đã hoàn thành Point (`~42.19/64.55`) và HUG e2e
  (`~27.47/49.18`), Frozen-Point bị interrupt ở epoch 9. Không dùng các số này làm
  multi-seed conclusion; chúng chứng minh batch 4 làm thay đổi objective/overfit.
- Code và artifact seed42 cho U1/U2/U2-heads/U3 đã tồn tại dưới
  `results/person2/` và `checkpoints/u*`; một số báo cáo cũ ghi “chưa implement”
  là stale. Đây là nhánh cũ/secondary, chưa thay thế baseline audit của v2.
- Không có training process đang được coi là active tại lần verify này.

Next decision gate:

1. Sửa/check preflight PDF path.
2. Chạy hai pilot Dress seed42.
3. Point batch32 phải quay lại xấp xỉ `48/71`; HUG e2e không được OOM/NaN hoặc
   collapse liên tục sau các epoch đầu.
4. Chỉ sau đó khóa `E2E_BATCH_SIZE` và chạy Dress seeds `42,7,123`.

## 7. Lệnh chuẩn

```bash
source ref/LAVIS/.venv/bin/activate
./scripts/run_supervisor_experiments.sh preflight

# Hai người/máy; nếu một GPU thì chạy tuần tự
./scripts/run_supervisor_person1.sh pilot
./scripts/run_supervisor_person2.sh pilot

# Sau khi pilot pass
SEEDS=42,7,123 CATEGORIES=dress ./scripts/run_supervisor_person1.sh clean
SEEDS=42,7,123 CATEGORIES=dress ./scripts/run_supervisor_person2.sh clean

./scripts/run_supervisor_person1.sh modality
./scripts/run_supervisor_person2.sh modality
./scripts/run_supervisor_person1.sh robustness
./scripts/run_supervisor_person2.sh robustness
./scripts/run_supervisor_experiments.sh summarize
```

Default v2: Point `B32/10 epochs/warmup0`; Frozen `B32/20/warmup0`; E2E và
Point-matched `B8/30 epochs/warmup2`; seeds `42,7,123`; W&B off. Nếu E2E OOM,
restart cả cặp bằng `FORCE=1 E2E_BATCH_SIZE=6 ...person2.sh pilot`, rồi mới thử
batch 4. Khi batch đã khóa, không đổi giữa seed/category.

Validation nhẹ, không chạy GPU dài:

```bash
python -m py_compile train.py eval.py eval/robustness.py \
  eval/modality_reliance.py eval/summarize_supervisor.py
bash -n scripts/run_supervisor_experiments.sh \
  scripts/run_supervisor_person1.sh scripts/run_supervisor_person2.sh
PYTHONPATH=. python tests/test_robustness_training.py
git diff --check
```

## 8. Quy tắc vận hành và cập nhật

- Bắt đầu session bằng `git status --short --branch`, đọc phần trạng thái trên,
  rồi kiểm tra artifact thực tế; không dựa vào ký ức/log chat.
- Không tự chạy job GPU dài khi user chỉ yêu cầu giải thích, review hoặc diagnose.
- Không dùng `FORCE=1`, xóa checkpoint/result, hay dọn artifact lớn nếu user chưa
  yêu cầu rõ. Giữ nguyên thay đổi không liên quan trong dirty worktree.
- Run train chỉ hoàn tất khi có `checkpoint_final.pth`; run dở resume từ
  `checkpoint_last.pth`. `checkpoint_best.pth` chỉ để eval. Robustness hoàn tất
  khi có `.complete`. Mỗi run có `train.log`; W&B mặc định tắt.
- Artifact v2 nằm dưới `checkpoints/supervisor_protocol_v2/` và
  `results/supervisor_protocol_v2/`. Không trộn tự động với `supervisor_protocol/`
  hoặc checkpoint legacy.
- Khi hai máy chạy riêng, merge nguyên cấu trúc hai thư mục v2; model IDs tách
  biệt nên không đè nhau. Chỉ summarize sau khi merge.
- Sau một phase, chỉ cập nhật phần “Trạng thái hiện tại”: ngày/commit, phase đã
  pass, đường dẫn summary canonical, quyết định go/no-go và bước kế tiếp. Không
  thêm log từng epoch, danh sách mọi file hay bảng số dài vào `AGENTS.md`.
- Số liệu chi tiết đi vào JSON/CSV và tài liệu báo cáo. Nếu protocol/invariant
  thay đổi, cập nhật cả runbook và phần tương ứng ở đây; tránh tạo thêm runbook
  cạnh tranh.
