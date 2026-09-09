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

## 3. Chuyển máy bằng một file ZIP

**Không nên zip toàn bộ project.** Riêng `checkpoints/` trên máy nguồn đã ~238 GB;
đa số là thí nghiệm lịch sử không cần cho hướng hiện tại. `.venv` cũng không nên
chép sang máy mới. Dùng [scripts/zip_project.py](scripts/zip_project.py), chỉ cần
Python chuẩn để đóng gói; không phải cài công cụ upload/đăng nhập Drive.

| Gói | Nội dung | Khi nào dùng |
|---|---|---|
| `essential` (mặc định, ~17.22 GiB trước nén) | Code/tài liệu, LAVIS + BLIP-2/EVA/BERT, Point Dress 42/7/123, kết quả v2; best/final của run xong, giữ last của run dở | **Khuyên dùng để chạy tiếp seed7/123** |
| `current-full` (~20.22 GiB trước nén) | Như trên + last/optimizer của run đã xong + JSON lịch sử nhỏ | Muốn lưu đầy đủ trạng thái thực nghiệm v2 hiện tại |

**Dataset Fashion-IQ đã lưu riêng nên mặc định không vào ZIP.** Code đọc dataset
`data/*.py` và nhãn thực nghiệm `results/.../labels/` vẫn được giữ. Nếu cần đổi
ý, thêm `--include-dataset` vào lệnh `plan`/`pack`.

Cả hai **không phải backup toàn bộ lịch sử nghiên cứu**: không mang các checkpoint
HUG cũ, `.git`, `.venv`, token hoặc cache không liên quan. `essential` không dùng
để cố ý train tiếp một run đã hoàn tất từ optimizer cũ; khi cần việc đó dùng
`current-full`. Giữ máy cũ cho đến khi máy mới đã kiểm thử thành công.

### Máy cũ — xem dung lượng rồi tạo ZIP

Dừng train/build-label/evaluate trước khi nén. Chạy trong project:

```bash
# Chỉ xem dung lượng dự kiến, chưa nén:
python3 scripts/zip_project.py plan

# Tạo gói khuyên dùng:
python3 scripts/zip_project.py pack

# Hoặc gói đầy đủ hơn (không cần tạo cả hai):
python3 scripts/zip_project.py pack --profile current-full
```

ZIP được lưu tại **`<project>/migration_exports/cir-essential-no-data.zip`**;
gói current-full có tên `cir-current-full-no-data.zip`. Trên máy hiện tại, mở
folder `migration_exports` trong VS Code/File Explorer để tải file sau khi nén xong:

```text
/mnt/data/users/quynhptit/huyptit/AAAI26-HUG/migration_exports/cir-essential-no-data.zip
```

Có thể đổi nơi lưu bằng `--output /duong/dan/backup.zip`. Đường dẫn tương đối
tính từ root project. Chỉ tải file `.zip` hoàn chỉnh, không tải `.zip.partial`.
`plan` chỉ in kế hoạch/đường dẫn; chưa tạo ZIP.

ZIP64 hỗ trợ file lớn hơn 4 GB; nén level1 để ưu tiên thời gian. `plan` báo dung
lượng **trước nén**, không hứa checkpoint sẽ nén nhỏ nhiều. Chép ZIP sang máy mới
bằng ổ đĩa/SCP hoặc tự upload Drive riêng tư. Không upload public dataset/checkpoint.
Script không ghi đè ZIP có sẵn; lần backup mới hãy chọn tên mới.

### Máy mới — giải nén → cài môi trường → chạy tiếp

Yêu cầu: Linux x86_64, GPU NVIDIA/driver phù hợp CUDA12.1, Python3 và `uv`.
Dành khoảng 70 GB trống cho archive, dữ liệu giải nén và môi trường.
ZIP có sẵn code nên **không bắt buộc chờ GitHub**. Với ZIP tin cậy do bạn tự tạo,
giải nén vào **thư mục mới, trống**:

```bash
mkdir HUG_reimplement
cd HUG_reimplement
python3 -m zipfile -e /duong/dan/cir-essential-no-data.zip .
python3 scripts/zip_project.py verify
# BƯỚC BẮT BUỘC: chép dataset đã lưu riêng vào data/fashion-iq/,
# gồm images/, captions/, image_splits/, rồi mới chạy setup bên dưới.
bash scripts/setup_new_machine.sh
source scripts/project_env.sh
nvidia-smi
bash scripts/run_reliability_cir.sh gate
```

Nếu muốn dùng Git: sau khi code đã push, clone đúng phiên bản có trong ZIP rồi
chạy `python3 scripts/zip_project.py restore --input /duong/dan/cir-essential-no-data.zip`.
Restore kiểm tra checksum và từ chối ghi đè file khác nội dung. Không ép overwrite
nếu clone không khớp phiên bản ZIP. Phiên bản chuẩn bị trước đã commit cục bộ,
nhưng push GitHub chưa thành công do thiếu đăng nhập; không mặc định remote đã mới.

Không cần sửa đường dẫn `/mnt/...` trong JSON/PTH: `.migration/manifest.json`
giữ root nguồn để mã tự ánh xạ sang root mới, không sửa nhãn/trọng số.
`setup_new_machine.sh` tạo lại môi trường Python3.8 từ snapshot package, cần mạng
để cài dependency. Preflight/gate chỉ là kiểm tra CPU/đọc kết quả đã lưu; kiểm tra
GPU trên root smoke riêng trước run dài, xem [chi tiết](docs/MIGRATION.md).
Đối chiếu đầy đủ cây thư mục, checkpoint nào bắt buộc/tùy chọn và các bước kiểm tra
tại [AGENTS.md, mục 9](AGENTS.md#9-cây-thư-mục-đối-chiếu-trên-máy-mới).

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
| `scripts/zip_project.py` | Lối vào chính: xem dung lượng, nén ZIP, restore và verify |
| `docs/MIGRATION.md` | Chi tiết kỹ thuật; công cụ TAR/Drive cũ là lựa chọn phụ |

Chỉ tên `AGENTS.md` là canonical; không tạo thêm `agent.md` dễ lệch trạng thái.
Khi code/tài liệu mâu thuẫn, ưu tiên artifact thực tế rồi runbook canonical.
