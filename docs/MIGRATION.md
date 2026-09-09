# Kiểm soát chuyển máy — 2026-09-09

## Phạm vi và nguyên tắc

Mục tiêu là tiếp tục **Dress reliability_v2 seeds 7/123** từ Point đã train,
đồng thời giữ đầy đủ bằng chứng pilot seed42. Đây không phải backup toàn bộ
lịch sử project: không có mọi checkpoint HUG/Frozen/Point Robust hoặc W&B.
Giữ máy cũ cho đến khi máy mới qua verify và smoke test.

GitHub lưu source và runbook đã review; Google Drive lưu artifact riêng tư.
Không đưa ảnh, trọng số, venv, token hoặc OAuth config lên GitHub.
Skill Google Drive đã kiểm tra connector: hiện là `tung.vuson.hau@gmail.com`,
khác tài khoản đích `huyphan1610@gmail.com`. Không upload qua connector này.

## Checklist artifact trong bundle

| Artifact | Vì sao phải giữ |
|---|---|
| Point Dress seed42/7/123 `checkpoint_best.pth` | Backbone chính xác cho label/train/eval từng seed |
| Toàn bộ `checkpoints/reliability_v2/` | Best để eval, last để resume, final xác nhận hoàn tất |
| Toàn bộ `results/reliability_v2/` | JSONL/CSV labels, summary, audit, probabilities, risk–coverage, log |
| `data/fashion-iq/` | Ảnh + captions + image splits; chỉ checkpoint không đủ chạy |
| `ref/LAVIS/` trừ `.git`, `.venv`, cache Python | Giữ source tùy biến, config, license và BLIP-2 pretrained |
| Cache EVA `eva_vit_g.pth`, cache `bert-base-uncased` | Model khởi tạo vẫn cần EVA/BERT dù sau đó load Point |
| Snapshot source hiện tại + 3 runbook Git | Có thể đối chiếu chính xác code với backup |
| `Thucnghiem_tiep_CIR.docx` | Nguồn chỉ đạo GVHD; giữ riêng tư trên Drive |
| JSON lịch sử nhỏ ở reliability_v1/supervisor_protocol_v2 | Tài liệu so sánh, không phải full historical backup |
| `.migration/environment.txt`, `.migration/manifest.json` | Version package, root gốc, SHA-256 và kích thước từng file |

Source LAVIS đang dùng xuất phát từ commit
`506965b9c4a18c1e565bd32acaccabe0198433f7`, có chỉnh config local và thông báo lỗi;
không thay nó bằng một bản pip LAVIS bất kỳ. Bundle giữ các thay đổi đó.

Môi trường nguồn: Python **3.8.20**, torch **2.4.1+cu121**, torchvision **0.19.1**,
transformers **4.33.2**, numpy **1.24.4**, timm **0.4.12**, huggingface-hub **0.21.4**.
`setup_new_machine.sh` cài snapshot toàn bộ dependency từ môi trường nguồn bằng
`--no-deps`, không giải lại dependency của LAVIS lịch sử. Không copy `.venv`
vì shebang, binary và đường dẫn của nó phụ thuộc máy.

## Thao tác đơn giản

Các lệnh chính nằm trong [README](../README.md). `plan` không ghi file;
`pack` tạo tar mới và ghi nhận checksum; `restore` giải nén có kiểm soát;
`verify` đọc lại SHA-256 mọi file. Chỉ dùng bundle/checkpoint do bạn tạo và tin cậy.
Checksum phát hiện hỏng dữ liệu, không phải chữ ký xác thực người gửi.

- Dừng mọi writer trước `pack`; không snapshot một epoch đang ghi.
- Gói đầy đủ hiện ~21.2 GiB; cần đủ quota Drive. Script không tự mua dung lượng.
  Dành ít nhất 25 GB trống trên Drive cho gói này.
- Không có quyền upload đúng account thì dừng bước upload, không đổi tài khoản đích.
- Rclone `copyto --immutable` không thay backup khác nội dung; không dùng `sync`.
- Rclone có retry nhưng không cam kết resume giữa chừng một file tar lớn qua mọi
  lần tắt process. Giữ archive máy cũ để thử lại.
- Không có SHA toàn archive riêng; restore kiểm tra SHA-256 từng payload theo
  manifest trong tar. Upload còn kiểm tra file qua `rclone check`.
- Không có restore overwrite: file có nội dung khác sẽ báo xung đột trước khi ghi.
- Nếu ngắt restore, chạy lại cùng lệnh; file khớp được bỏ qua. File `.restore-*`
  còn dở có thể còn trên đĩa; chỉ dọn các file đó sau khi xác minh thành công.
- Khi pack lỗi, `.tar.partial` được giữ để chẩn đoán; dùng tên bundle mới.

Nguồn CLI: [rclone userinfo](https://rclone.org/commands/rclone_config_userinfo/),
[rclone Google Drive](https://rclone.org/drive/).

## Khi tự chép artifact thay vì dùng bundle

Giữ đúng đường dẫn tương đối trong bảng trên. Trên máy mới:

```bash
# Giá trị này là root thực tế trên máy nguồn, không phải root máy mới:
export HUG_SOURCE_ROOT=/mnt/data/users/quynhptit/huyptit/AAAI26-HUG
source scripts/project_env.sh
```

`utils/artifact_paths.py` chỉ đổi prefix đã khai báo rồi kiểm tra file tồn tại;
không đoán theo basename và không nới lỏng kiểm tra cùng Point giữa train/val.
Manifest/checkpoint cũ được giữ nguyên: các chuỗi đường dẫn trong metrics là
provenance của run nguồn, không tự biến thành một lượt evaluate trên máy mới.
Bundle tự thiết lập mapping bằng metadata nên không cần export tay.
Nếu remigrate nhiều lần, giữ `HUG_SOURCE_ROOT` là root gốc của artifact.

Có thể override `LAVIS_ROOT`, `BLIP2_PRETRAINED`, `VENV`, `DATA_ROOT`,
`RESULT_ROOT`, `CHECKPOINT_ROOT`; không trỏ sang model/category/seed khác để vá lỗi.

## Checklist xác nhận máy mới

1. Restore + `verify` không báo missing/checksum/conflict.
2. `setup_new_machine.sh` và CPU preflight thành công; kiểm tra torch CUDA khả dụng.
3. Đọc gate seed42 vẫn true; đây chỉ là đọc kết quả đã lưu, chưa là GPU smoke.
4. Chạy smoke GPU trên **root tách riêng** bằng lệnh bên dưới. Dùng tên root mới
   cho lần kiểm tra mới; smoke không dùng để báo cáo metric.
5. Sau đó mới chạy Dress seed7 và123 theo README. Giữ nguyên schema2/failure@10.

```bash
source scripts/project_env.sh
CUDA_VISIBLE_DEVICES=0 MAX_QUERIES=8 \
  RESULT_ROOT=results/migration_smoke_20260909/dress/seed42 \
  bash scripts/run_reliability_cir.sh audit
CUDA_VISIBLE_DEVICES=0 MAX_QUERIES=8 \
  RESULT_ROOT=results/migration_smoke_20260909/dress/seed42 \
  bash scripts/run_reliability_cir.sh build-val
```

Smoke trên khởi tạo Point và retrieval trên gallery thật; 8 query vẫn cần VRAM
cho backbone/gallery. Nó không xác nhận full reliability head resume.
Muốn xác nhận head cũ trên GPU, evaluate lại best với train/val manifest đã restore
vào `--output_file` **mới** bằng `eval/reliability.py evaluate` (không dùng orchestrator
`evaluate` vì script sẽ skip metrics có sẵn). Đối chiếu metric với seed42 nguồn.

Trainer hiện lưu epoch, optimizer, best AUROC và stale count nhưng chưa lưu RNG;
resume có thể tiếp tục học, **không bảo đảm bitwise giống run liên tục**.
Phải dùng model checkpoint đáng tin vì PyTorch checkpoint có dữ liệu pickle.

## Note thay đổi lần chuyển máy

| File | Thay đổi | Không thay đổi |
|---|---|---|
| `.gitignore` | Giữ code `data/*.py`, whitelist 3 runbook; ignore artifact/secrets/archive | Không xóa dữ liệu thật |
| `models/blip_backbone.py`, `eval.py` | Bỏ hard-coded home/root cũ, cho override pretrained; restore monkey patch trong finally | Kiến trúc/weights/loss |
| `utils/artifact_paths.py`, model/trainer/evaluator reliability | Đổi root khi đọc artifact và kiểm tra provenance | Nhãn, rank, byte checkpoint, thresholds |
| `scripts/project_env.sh`, orchestrator | LAVIS/cache portable, ưu tiên `.venv` máy mới | Schema và lịch training |
| `scripts/migrate_project.py` | Inventory/pack/restore/verify, SHA-256, không overwrite | Artifact gốc |
| `scripts/upload_project_drive.sh` | Khóa email trước upload, immutable copy/check | Không share public, không sync/delete |
| `scripts/setup_new_machine.sh` | Tạo lại Python3.8 và dependency snapshot | Không nâng version để tiện cài |
| AGENTS/README/runbook/progress | Cập nhật seed42 pass, next seeds7/123 và hướng dẫn tiếp quản | Không mở fusion sớm |

Kiểm thử tại máy nguồn: **29 CPU tests pass**, compile + Bash syntax, preflight
8/8, gate seed42 true, import LAVIS/dataset và BERT offline thành công, validation
149625 train/50425 val rows thật pass; bài test restore
nhỏ xác nhận checksum, idempotence, chặn overwrite/traversal/symlink và mapping
root cũ. Chưa thay thế kiểm thử CUDA/driver trên máy mới. Không chạy train GPU
trong lượt chuẩn bị di chuyển này.
