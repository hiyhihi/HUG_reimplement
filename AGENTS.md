# HUG-CIR project guide for Codex

File này là context lõi cho session mới. Giữ ngắn; thay trạng thái cũ thay vì
nối nhật ký. Số liệu chi tiết nằm trong runbook, JSON, checkpoint và log.

## 1. Mục tiêu

Nghiên cứu Composed Image Retrieval trên Fashion-IQ (`dress`, `shirt`,
`toptee`). Hướng chính hiện tại là **reliability-aware CIR**: giữ Point
deterministic làm backbone, xây Fashion-IQ-C bằng corruption chỉ trên query và
dự đoán xác suất retrieval failure. Full HUG/uncertainty là kết quả no-go lịch
sử; không tiếp tục tối ưu hay dùng uncertainty cũ để điều khiển fusion.

## 2. Thứ tự nguồn tin

Khi mâu thuẫn, ưu tiên code/config đang chạy và JSON/checkpoint/log thực tế,
rồi `word&md&pdf/RELIABILITY_AWARE_CIR_EXPERIMENT_RUNBOOK.md`, rồi
`word&md&pdf/SUPERVISOR_EXPERIMENT_RUNBOOK.md` và báo cáo lịch sử. Các
thư mục `word&md&pdf/`, `data/`, `checkpoints/`, `results/`, `wandb/`, `ref/`
có thể bị `.gitignore`; không suy ra không tồn tại từ `git status`.

## 3. Taxonomy model

| ID | Ý nghĩa đúng |
|---|---|
| `legacy` | Pipeline lịch sử, chỉ reference. |
| `point` | Mean-only symmetric InfoNCE, Q-Former trainable, B32; deterministic baseline. |
| `point_continued` | Warm-start Point; chỉ train tiếp clean InfoNCE; control cho `point_robust`. |
| `point_robust` | Warm-start Point; clean InfoNCE + token-dropout InfoNCE + ranking consistency; recall/robustness ablation. |
| `point_matched` | Cùng Point nhưng B8/lịch giống e2e; control công bằng. |
| `hug_e2e` | BLIP-2 init, Q-Former/query tokens trainable, `HC + .5FC + .1Cord`; candidate full HUG. |
| `hug_frozen_point` | Warm-start Point, freeze mean encoder, chỉ học uncertainty; controlled ablation, không phải full HUG. |
| `point_reliability_probe` | Warm-start Point, freeze toàn bộ Point, học query-only failure-probability head; hướng chính v2. |
| `point_reliability_joint` | Retrieval + failure joint training; chưa chạy, chỉ mở sau probe và relabel protocol. |

Không dùng tên checkpoint cũ `paper_*` để gọi full-paper reproduction. Phải báo
cả `hug_e2e - point` và `hug_e2e - point_matched`.

## 4. Bất biến khoa học

- Paper Dress: `48.37/71.56` R@10/R@50; số gần không tự chứng minh protocol
  khớp paper.
- Physical batch quyết định in-batch negatives; gradient accumulation không
  thay thế B32. Không so HUG batch nhỏ chỉ với Point B32.
- Reliability là `P(failure@k | query)`, không phải HUG variance. Đánh giá bằng
  AUROC, AUPRC, ECE, Brier, risk-coverage/AURC và per-condition metrics.
- Corruption chỉ tác động query; severity 0 identity; paired manifest/seed.
- Target/gallery luôn sạch. Semantic contradiction là semantic-changing stress
  test và phải báo riêng nuisance corruptions.
- Không multi-seed/category hoặc adaptive fusion nếu Dress seed42 reliability
  pilot chưa qua gate. Không đổi loss/architecture hậu nghiệm mà không ghi ablation.

## 5. Bản đồ code

- `train.py`: CLI, loop, resume/checkpoint/RNG, U1/U2/U3 flags.
- `models/hug_model.py`, `modules/losses.py`: representation, fusion, loss.
- `eval/robustness.py`, `eval/modality_reliance.py`: harness đánh giá.
- `eval/summarize_supervisor.py`: bảng và paired deltas.
- `scripts/run_supervisor_experiments.sh`: orchestrator protocol v2.
- `scripts/run_supervisor_person1.sh`: Point + Frozen-Point.
- `scripts/run_supervisor_person2.sh`: Point-matched + HUG e2e.
- `scripts/run_point_robust_recall.sh`: pilot tăng recall/độ bền text từ Point.
- `scripts/run_point_continued_control.sh`: control train tiếp Point chỉ với clean InfoNCE.
- `modules/fiqc.py`, `eval/reliability.py`: Fashion-IQ-C manifest, failure labels,
  robustness table và reliability evaluation.
- `modules/reliability.py`, `models/reliability_model.py`, `train_reliability.py`:
  query-only Reliability Head và head-only training trên frozen Point.
- `scripts/run_reliability_cir.sh`: orchestrator schema/protocol `reliability_v2`; v1 read-only.

## 6. Trạng thái hiện tại — verified 2026-09-09

- GVHD đã chuyển hướng sang reliability-aware CIR theo
  `word&md&pdf/Thucnghiem_tiep_CIR.docx`. Protocol chi tiết canonical mới:
  `word&md&pdf/RELIABILITY_AWARE_CIR_EXPERIMENT_RUNBOOK.md`.
- Pilot `reliability_v1` Dress seed42 đã hoàn tất đủ final/best/last: clean
  `47.99/70.90`, AUROC `0.7205`, AUPRC `0.8037` với prevalence `0.6226`, ECE
  `0.0737`, risk reduction tại 50% coverage `24.47%`. Numeric gate pass và
  within-condition AUROC `0.7031`, nhưng synonym changed-rate train/val chỉ
  `88.05/88.75%`, dưới Gate 0; giữ toàn bộ v1 read-only làm pilot, chưa mở
  multi-seed/category hoặc adaptive fusion.
- `reliability_v2` Dress seed42 đã hoàn tất: đủ best/last/final, 149625 train
  rows và 50425 val rows (2017 query × 25 conditions). AUROC `0.7256`, AUPRC
  `0.8114` (prevalence `0.6236`), ECE `0.0680`, risk reduction@50% `24.91%`;
  `adaptive_fusion_gate.passed=true`. Synonym train/val `98.50/98.76%`;
  within-condition AUROC `0.7099`. Clean `47.99/70.90`.
- Best epoch 1, early stop epoch 4: có overfit. Val dùng chọn checkpoint và
  đánh giá, không gọi là independent test. Contradiction ECE `0.1375`, báo riêng.
- Chưa có reliability v2 seed7/123. Bước tiếp theo: xác nhận Dress seeds 7/123
  cùng protocol, rồi mean±std/paired query-bootstrap CI. Không đổi loss/epoch
  hậu nghiệm. Adaptive fusion vẫn đóng đến khi Dress multi-seed và complement
  analysis hoàn tất; chưa mở Shirt/Toptee.

- Person1 hoàn tất Point và Frozen-Point: 3 seeds × 3 categories, clean,
  modality và robustness. Point mean: Dress `47.98±0.32/71.49±0.90`, Shirt
  `52.16±0.32/71.16±0.50`, Toptee `54.38±0.46/77.49±0.59`. Frozen mean trùng
  Point đúng thiết kế; Frozen probabilistic Dress `48.31±0.51/71.79±1.14`.
- Frozen uncertainty chưa calibrated: pooled severity-4 AUROC chỉ xấp xỉ
  `0.49–0.52`; text typo/dropout làm R@10 rơi mạnh. Không build router/U1–U3
  production claim.
- Person2 no-go: HUG e2e Shirt seed42 `37.78/57.51`, thấp hơn paired
  Point-matched `12.56/11.87`. Diagnostic Dress B8 xác nhận Point `47.00/71.10`
  nhưng HC-only/HC+FC/HC+Cord/full HUG chỉ `~29–30/51–53`; FC/Cord không phải
  nguyên nhân chính. Diagnostic tiếp theo cũng fail: HC mean-distance
  `29.60/51.56`, còn average-negatives collapse xuống `6.25/14.53`. Gradient
  Q-Former/query tokens hợp lệ; vấn đề nằm ở HC objective/reduction. Không sweep
  full HUG; audit công thức trước khi train thêm.
- `train.py` có instrumentation (trainability, pre-clip gradient, HC
  distance/variance/LR) và early stopping. Hai diagnostic HC đã không qua gate;
  không chạy lại hoặc mở rộng seed/category trước khi audit công thức.
- `point_robust` đã hoàn tất Dress 3 seed: `49.17±0.08/72.10±0.57`, cao hơn
  paired Point `+1.19/+0.61`. Delta trung bình mọi mức nhiễu text là
  `+0.79/+1.10`, nhưng severity 3–4 chỉ tăng khoảng `+0.61` R@10, chưa đạt gate
  robustness `+2.0` so với Point.
- `point_continued` hoàn tất Dress 3 seed: `49.03±0.39/72.05±0.31`, cao hơn
  Point `+1.06/+0.56`. Point Robust chỉ hơn control này `+0.13/+0.05` clean,
  `+0.19/+0.43` trung bình mọi mức nhiễu text và `+0.23/+0.67` ở severity 3–4;
  R@10 không tăng ở mọi seed. Phần lớn clean gain do train thêm; đóng góp riêng
  của robust objective nhỏ và chưa ổn định.
- Runbook canonical có bảng 4 kiến trúc, kết quả/gate và diagnostic plan:
  `word&md&pdf/SUPERVISOR_EXPERIMENT_RUNBOOK.md`.

## 7. Quy tắc vận hành

- Bắt đầu session: `git status --short --branch`, đọc trạng thái trên, rồi
  kiểm tra artifact thật. Không tự chạy GPU dài nếu user chỉ yêu cầu review.
- Run hoàn tất chỉ khi có `checkpoint_final.pth`; run dở resume từ
  `checkpoint_last.pth`; dùng `checkpoint_best.pth` để eval. Không dùng
  `FORCE=1`, xoá/ghi đè artifact nếu chưa được yêu cầu.
- Trước diagnostic mới, dùng `RESULT_ROOT` và `CHECKPOINT_ROOT` mới để giữ
  artifact dở; giữ paired model/batch/seed. Không mở rộng seed/category hoặc
  adaptive fusion trước Dress reliability pilot và calibration gates.
- `results/reliability_v1/` và `checkpoints/reliability_v1/` là pilot read-only.
  Nhánh mới chỉ dùng `results/reliability_v2/` và `checkpoints/reliability_v2/`;
  không ghi đè supervisor protocol v2. Label train/val phải sinh từ cùng Point
  checkpoint; không trộn `failure@10` và `failure@50`.

## 8. Chuyển máy và lưu trữ

- Điểm vào cho người mới: `README.md`; chi tiết `docs/MIGRATION.md`.
- Cách đơn giản mặc định: `python3 scripts/zip_project.py plan`, sau đó chạy
  `python3 scripts/zip_project.py pack`.
  ZIP gồm cả code, không bắt buộc
  có GitHub để chạy; giải nén vào thư mục mới rồi `verify`, setup và preflight.
- Hai profile ZIP: `essential` bỏ last của run đã có final và JSON lịch sử;
  `current-full` giữ đủ artifact v2. Cả hai giữ last của run dở, best/final của
  run hoàn tất và Point Dress 42/7/123. Không zip toàn bộ ~238 GB checkpoints cũ.
- Dataset Fashion-IQ đã được user backup riêng: ZIP mặc định KHÔNG có
  `data/fashion-iq/`, vẫn giữ `data/*.py` và failure manifests trong `results/`.
  Chỉ thêm dataset khi có `--include-dataset`; pack không yêu cầu dataset tồn tại
  nếu không bật cờ này. Dung lượng trước nén: essential ~17.22 GiB, current-full
  ~20.22 GiB (2026-09-09).
- ZIP mặc định: `<project>/migration_exports/cir-essential-no-data.zip`;
  profile current-full: `cir-current-full-no-data.zip`. `--output` tương đối
  được tính từ `--root`/root project, không từ thư mục shell đang đứng. Khi còn
  đuôi `.partial` thì chưa hoàn tất và không dùng để chuyển máy.
- Chỉ thiết kế/test script khi user chưa yêu cầu nén thật; không tự chạy backup
  hàng chục GB. Không tự push/upload trong yêu cầu chỉnh tài liệu/ZIP.
- Git giữ code (bao gồm `data/*.py`) và 3 runbook đã whitelist. Dataset,
  checkpoints/results, LAVIS/cache, archive và OAuth secrets không lên GitHub.
- Công cụ phụ cũ `scripts/migrate_project.py plan|pack|restore|verify`: TAR có
  SHA-256, giữ đúng cây thư mục; không chép `.venv`, không overwrite file khác.
- `scripts/upload_project_drive.sh` kiểm tra tài khoản `huyphan1610@gmail.com`
  trước upload bằng rclone. Không upload qua connector nếu email không khớp.
- `scripts/project_env.sh` cấu hình LAVIS/cache theo root mới;
  `scripts/setup_new_machine.sh` cài lại môi trường từ snapshot Python 3.8.
- `utils/artifact_paths.py` dùng `.migration/manifest.json` hoặc
  `HUG_SOURCE_ROOT` để ánh xạ root cũ. Không sửa byte checkpoint, nhãn/rank hay
  provenance gốc; không tự chọn checkpoint khác cùng basename.
- Chỉ load checkpoint tin cậy. Sau restore: verify checksum, CPU preflight,
  kiểm tra GPU/smoke root riêng trước run dài. Resume giữ epoch/optimizer;
  trainer hiện chưa lưu RNG nên không claim bitwise-equivalent resume.
- Git: commit migration trước đã lưu cục bộ (`7a86613`), push bị thiếu credential;
  không khẳng định clone từ remote đã có các thay đổi. ZIP lấy source hiện tại.

## 9. Cây thư mục đối chiếu trên máy mới

`[ZIP]` có trong gói, `[RIÊNG]` lấy từ backup dataset, `[TẠO]` sinh khi cài/chạy,
`[TÙY CHỌN]` không cần có trong essential. Cây này liệt kê cấu trúc phục vụ
reliability v2; danh sách TỪNG file + kích thước + SHA-256 nằm trong
`.migration/manifest.json`, dùng `verify` để kiểm tra đầy đủ (kể cả file con LAVIS).

```text
<project>/
├── AGENTS.md, README.md, LICENSE, requirements.txt              [ZIP]
├── train.py, train_reliability.py, eval.py                     [ZIP]
├── config/                                                   [ZIP]
│   ├── fashion_iq.yaml
│   └── cirr.yaml
├── data/
│   ├── __init__.py, dataset.py, transforms.py                 [ZIP: code, KHÔNG bỏ]
│   └── fashion-iq/                                           [RIÊNG: chép trước preflight]
│       ├── images/<image_id>.png
│       ├── captions/cap.dress.train.json
│       ├── captions/cap.dress.val.json
│       └── image_splits/split.dress.{train,val}.json
├── models/                                                   [ZIP]
│   ├── __init__.py, blip_backbone.py, hug_model.py
│   └── reliability_model.py, uncertainty_head.py
├── modules/                                                  [ZIP]
│   ├── __init__.py, fiqc.py, reliability.py, losses.py
│   └── metrics.py, dynamic_weighting.py, robustness_training.py
├── utils/                                                    [ZIP]
│   └── __init__.py, artifact_paths.py, checkpoint.py, logger.py
├── eval/                                                     [ZIP]
│   ├── reliability.py, robustness.py, robustness_legacy.py
│   └── modality_reliance.py, summarize_supervisor.py
├── scripts/                                                  [ZIP]
│   ├── zip_project.py, migrate_project.py                     # cả hai cần cho ZIP tool
│   ├── project_env.sh, setup_new_machine.sh
│   └── run_reliability_cir.sh, ...                            # giữ cả scripts lịch sử
├── tests/                                                    [ZIP: test_*.py]
├── assets/                                                   [ZIP]
├── docs/MIGRATION.md                                         [ZIP]
├── word&md&pdf/                                              [ZIP]
│   ├── Thucnghiem_tiep_CIR.docx
│   ├── RELIABILITY_AWARE_CIR_EXPERIMENT_RUNBOOK.md
│   ├── SUPERVISOR_EXPERIMENT_RUNBOOK.md
│   └── THUCNGHIEM_TIEP_CIR_PROGRESS.md
├── ref/LAVIS/                                               [ZIP: source/config/license]
│   ├── lavis/                                               # gồm models/processors/configs
│   └── pretrained_weights/blip2_pretrained.pth
├── checkpoints/
│   ├── supervisor_protocol_v2/point/dress/
│   │   ├── seed42/checkpoint_best.pth                        [ZIP]
│   │   ├── seed7/checkpoint_best.pth                         [ZIP]
│   │   └── seed123/checkpoint_best.pth                       [ZIP]
│   └── reliability_v2/dress/seed42/
│       ├── checkpoint_best.pth                              [ZIP: eval]
│       ├── checkpoint_final.pth                             [ZIP: đã hoàn tất]
│       └── checkpoint_last.pth                              [TÙY CHỌN: current-full]
├── results/reliability_v2/dress/seed42/                       [ZIP]
│   ├── reliability_metrics.json
│   ├── reliability_metrics.per_query.csv
│   ├── reliability_metrics.risk_coverage.csv
│   ├── logs/train_reliability.log
│   └── labels/                                              # mỗi split train VÀ val:
│       ├── fiqc_dress_{train,val}_seed42.jsonl
│       ├── fiqc_dress_{train,val}_seed42.csv
│       ├── fiqc_dress_{train,val}_seed42_summary.json
│       ├── fiqc_dress_{train,val}_seed42_robustness.csv
│       └── fiqc_dress_{train,val}_seed42_corruption_audit.json
├── .migration/                                              [ZIP: thư mục ẩn, phải giữ]
│   ├── manifest.json                                        # root gốc + inventory + SHA
│   ├── environment.txt                                      # snapshot package để cài lại
│   └── cache/
│       ├── torch/hub/checkpoints/eva_vit_g.pth
│       └── huggingface/hub/models--bert-base-uncased/
│           └── blobs/, refs/, snapshots/
├── .venv/                                                   [TẠO: setup_new_machine.sh]
└── migration_exports/                                       # nơi ZIP được tạo; không tự nhét vào ZIP
```

Checklist máy mới:

1. Chạy `python3 scripts/zip_project.py verify`: kiểm tra mọi file TRONG ZIP;
   **không kiểm tra dataset đã backup riêng**, GPU hay dependency hệ thống.
2. Chép dataset đúng `data/fashion-iq/` (đừng lồng thành `fashion-iq/fashion-iq/`).
   Dress cần đủ ảnh của cả captions và gallery splits train/val, không chỉ ảnh
   query. Shirt/Toptee có thể giữ từ backup riêng nhưng chưa mở thực nghiệm mới.
3. Chạy `bash scripts/setup_new_machine.sh`, rồi `source scripts/project_env.sh`,
   `bash scripts/run_reliability_cir.sh preflight` và `... gate` (seed42).
4. Chưa có `checkpoints/reliability_v2/dress/{seed7,seed123}/` và thư mục results
   tương ứng là bình thường: chúng chỉ sinh khi chạy hai seed tiếp theo. Không
   nhầm với Point seed7/123 bên trên: hai Point checkpoint đó BẮT BUỘC phải có.
5. Không có `.git`, `.venv` trước setup, `wandb/`, checkpoint HUG/v1 lịch sử
   hoặc `last` của seed42 trong essential KHÔNG phải thiếu file. Run dở luôn cần last.
