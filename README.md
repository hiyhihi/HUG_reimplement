# HUG-CIR → Reliability-Aware CIR

Project nghiên cứu **Composed Image Retrieval (CIR)** trên Fashion-IQ: đưa vào
ảnh tham chiếu và câu mô tả thay đổi, tìm ảnh đích phù hợp trong gallery.
Hướng hiện tại theo GVHD là **dự đoán query nào sẽ retrieval thất bại**, không
tiếp tục tối ưu Full HUG/uncertainty cũ.

## 1. Nhìn tổng quát trong một phút

```text
Fashion-IQ: ảnh tham chiếu + câu thay đổi
    ↓ corruption chỉ trên query (gallery/target luôn sạch)
Fashion-IQ-C: clean + 6 loại nhiễu × 4 severity
    ↓ Point deterministic, cố định trọng số
Rank / hit@10 / hit@50 → nhãn failure@10
    ↓ học query-only Reliability Head trên representation của Point
P(failure@10 | query) → AUROC / AUPRC / ECE / risk–coverage
    ↓ qua gate + xác nhận nhiều seed + phân tích bổ trợ giữa các nhánh
Cân nhắc adaptive fusion (CHƯA triển khai)
```

Head reliability không làm recall tự tăng: nó dự đoán rủi ro của Point để có
thể giữ lại các query đáng tin hơn. Contradiction làm đổi nghĩa, luôn báo riêng.

## 2. Đang ở đâu? (xác minh 2026-09-09)

| Hạng mục | Trạng thái |
|---|---|
| Fashion-IQ-C v2 / failure pattern | Dress seed42 đã có train/val manifests và robustness tables |
| Reliability probe v2 | Đã train xong, đủ `best/last/final`; best epoch 1, dừng epoch 4 |
| Gate seed42 | **PASS**: AUROC 0.7256; AUPRC 0.8114; ECE 0.0680; giảm risk@50% 24.91% |
| Clean retrieval | R@10/R@50 = 47.99/70.90 |
| Việc tiếp theo | Dress seed 7 và 123, giữ nguyên cấu hình |
| Chưa được mở | Shirt/Toptee, joint training, adaptive fusion trước các bước xác nhận |

Val hiện dùng cả chọn checkpoint và đánh giá; không gọi đây là test độc lập.
Head có dấu hiệu overfit sau epoch 1. V1 chỉ giữ làm pilot lịch sử vì fail
synonym coverage; không trộn artifact v1/v2.

## 3. Chuyển sang máy mới: làm theo thứ tự này

### Máy cũ: tạo backup và upload

Dừng các tiến trình đang train/build-label/evaluate trước khi đóng gói.
Script không lấy `.venv`, tài khoản, token hay toàn bộ cache của người dùng.

```bash
python3 scripts/migrate_project.py plan
python3 scripts/migrate_project.py pack --bundle migration_exports/cir-v2-20260909.tar
# Cài rclone; cấu hình remote tên huydrive, loại Google Drive,
# đăng nhập đúng huyphan1610@gmail.com. Chỉ cần làm một lần:
rclone config
bash scripts/upload_project_drive.sh migration_exports/cir-v2-20260909.tar huydrive
```

Gói hiện khoảng **21.2 GiB**, cần ít nhất **25 GB trống trên Drive**. Giữ riêng tư;
không chia sẻ công khai dataset/checkpoint. Script upload sẽ dừng nếu email không
khớp. Không đóng gói lại cùng tên: dùng tên mới khi có kết quả mới.

### Máy mới: clone → restore → cài môi trường → kiểm tra

Yêu cầu: Linux x86_64, GPU NVIDIA và driver phù hợp PyTorch CUDA 12.1;
cài `git`, `uv`, `rclone`. Dành khoảng 70 GB trống cho archive, giải nén và môi trường.

```bash
git clone https://github.com/hiyhihi/HUG_reimplement.git
cd HUG_reimplement
rclone config  # remote huydrive, đăng nhập đúng Gmail trên máy mới
rclone copy huydrive:AAAI26-HUG-migration/cir-v2-20260909.tar migration_exports/ --progress
python3 scripts/migrate_project.py restore --bundle migration_exports/cir-v2-20260909.tar
python3 scripts/migrate_project.py verify
bash scripts/setup_new_machine.sh
source scripts/project_env.sh
nvidia-smi
bash scripts/run_reliability_cir.sh gate
```

Backup chứa snapshot code để đối chiếu. Nếu clone đã có code mới khác backup,
restore sẽ **dừng**, không ghi đè: dùng checkout/clone đúng phiên bản của backup
trong thư mục mới. Không dùng `--force` để che xung đột.

Không cần sửa `/mnt/...` trong JSON/PTH: mã sẽ ánh xạ root gốc qua
`.migration/manifest.json`, giữ nguyên byte artifact. Nếu tự chép bằng tay,
xem [hướng dẫn chi tiết](docs/MIGRATION.md).

## 4. Chạy tiếp thực nghiệm (sau khi khôi phục)

```bash
source scripts/project_env.sh
# CPU preflight + đọc gate seed42, không train lại seed42:
bash scripts/run_reliability_cir.sh preflight
bash scripts/run_reliability_cir.sh gate

# Chạy tuần tự trên GPU; đổi CUDA_VISIBLE_DEVICES nếu cần:
CUDA_VISIBLE_DEVICES=0 CATEGORY=dress SEED=7 bash scripts/run_reliability_cir.sh pilot
CUDA_VISIBLE_DEVICES=0 CATEGORY=dress SEED=123 bash scripts/run_reliability_cir.sh pilot
```

`pilot` tự thực hiện: kiểm tra → audit corruption → tạo nhãn train/val → train
head → đánh giá best checkpoint → kiểm tra gate. Giữ B32, 10 epochs tối đa,
patience 3, LR 1e-4, failure@10. Không tự giảm batch hoặc tăng epochs.

Nếu bị ngắt: chạy lại đúng lệnh và đúng output root. Có `last` nhưng chưa có
`final` thì resume từ epoch/optimizer đã lưu; có `final` thì bỏ qua train.
Resume hiện chưa khôi phục RNG nên không bảo đảm bitwise giống run liên tục.
File label/evaluation đã có sẽ được bỏ qua, **không tự kiểm tra file dở**;
nếu lần trước ngắt lúc build-label, kiểm tra summary/CSV trước khi chạy tiếp.

Kết quả đọc tại `results/reliability_v2/dress/seed{42,7,123}/reliability_metrics.json`.
Sau đủ ba seed: tổng hợp mean±std và paired bootstrap CI theo query, rồi phân tích
khả năng bổ trợ giữa các nhánh trước khi đề xuất fusion. Gate script chỉ đọc seed
đang chọn; người chạy phải tuân thủ thứ tự trên, không tự động chặn mọi category.

## 5. Tìm gì ở đâu?

| Vị trí | Nội dung / nơi lưu |
|---|---|
| `AGENTS.md` | Context ngắn, quyết định và quy tắc cho người/agent tiếp quản; **Git** |
| `word&md&pdf/RELIABILITY_AWARE_CIR_EXPERIMENT_RUNBOOK.md` | Protocol, thresholds và gate canonical; **Git** |
| `word&md&pdf/THUCNGHIEM_TIEP_CIR_PROGRESS.md` | Tiến độ theo từng đầu mục của GVHD; **Git** |
| `word&md&pdf/SUPERVISOR_EXPERIMENT_RUNBOOK.md` | Kết quả/hướng cũ để đối chiếu; **Git** |
| `data/*.py`, `models/`, `modules/`, `eval/`, `scripts/`, `tests/` | Code; **Git** |
| `data/fashion-iq/`, `ref/LAVIS/`, `.migration/cache/` | Dataset, LAVIS tùy biến và trọng số/cache; **Drive** |
| `checkpoints/`, `results/` | Trọng số, nhãn, metrics, log; **Drive** |
| `docs/MIGRATION.md` | Chi tiết backup, kiểm thử, giới hạn và note thay đổi |

Chỉ tên `AGENTS.md` là canonical; không tạo thêm `agent.md` dễ lệch trạng thái.
Khi code/tài liệu mâu thuẫn, ưu tiên artifact thực tế rồi runbook canonical.
