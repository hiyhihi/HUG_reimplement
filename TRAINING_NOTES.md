# Kế hoạch tái lập và tiến tới 7x trên Fashion-IQ Dress

## Kết luận từ artifact hiện có

Mục tiêu đúng của paper trên **Dress** là `R@10=48.37`, `R@50=71.56`. Con số khoảng 75 là trung bình `R@50=74.73` của cả ba category, không phải riêng Dress.

| Artifact | Thiết lập suy ra | R@10 | R@50 | Nhận xét |
|---|---:|---:|---:|---|
| `run_reproduce_6x.sh` (seed 42, 24/7) | legacy, no warmup, loss scalars LR x100, 6 epochs | **37.68** | **62.07** | baseline 6x đã tái lập thành công |
| `run-20260530_173733-yjm5w2i1` + result epoch 6 | no warmup, loss scalars LR x100, 6 epochs | 35.99 | 61.68 | baseline lịch sử trước đó |
| result epoch 8 (31/5) | train dài hơn | 35.60 | 61.03 | đã plateau |
| result epoch 10 (31/5) | train dài hơn | 35.50 | 60.29 | bắt đầu giảm |
| `run-20260724_132159-oohnnc7h` | warmup 2 epochs, factor 0.01 | 2.28 | 7.88 | regression hiện tại |

Warmup hiện tại làm LR epoch đầu từ `3e-5` xuống `3e-7`. Dấu vết loss của run 26/6 và run 24/7 trùng nhau, cho thấy đây là nguyên nhân trực tiếp làm mất hành vi 6x. Không nên dùng warmup theo epoch cho dataset chỉ có 188 step/epoch; mặc định mới là `--warmup_epochs 0`.

## Sai khác quan trọng giữa paper và code public

1. Paper dùng `h([LQ], text, empty)` và thu 32 learnable-query tokens cho nhánh text. Code cũ lấy sentence tokens rồi cắt/pad 32.
2. Eq. 13 có negative theo cả query→target và target→query; code cũ chỉ có một chiều.
3. Eq. 16 học khoảng cách giữa các variance vector với `a'`, `b'`; code cũ thay bằng InfoNCE positive/negative khác công thức.
4. Paper gọi output uncertainty head là variance `sigma^2`; code cũ coi nó là standard deviation rồi bình phương lần nữa.
5. Paper khởi tạo holistic `a=1,b=0`; code cũ dùng `a=2,b=-5`.
6. Paper dùng AdamW `eps=1e-7`; code cũ dùng mặc định PyTorch.

Repo public của tác giả cũng không khớp paper ở các điểm trên. Vì vậy cần coi kết quả 71.56 là mục tiêu thực nghiệm, không phải kết quả có thể đạt chỉ bằng cách chạy nguyên repo public.

## Ba recipe đã tách

- `legacy`: giữ đường code tạo mốc 61.68 để có baseline tái lập.
- `point`: symmetric InfoNCE chỉ trên mean Q-Former; dùng để xác minh encoder/data/eval trước khi đưa uncertainty vào.
- `paper`: 32 query tokens cho text, variance đúng nghĩa, Eq. 13 hai chiều, Eq. 16 vectorized, coordination sigmoid đúng chiều mong muốn.

Point baseline mới đã đạt `R@10=47.94`, `R@50=70.95`. Trong recipe
`point`, cột `Loss HC` đang chứa symmetric mean-only InfoNCE; `Loss FC` và
`Loss Cord` được tắt có chủ đích nên luôn bằng 0. Paper recipe mới kích hoạt
cả ba loss. Khi warm-start paper từ point, backbone được freeze để giữ nguyên
mean retrieval đã đạt 70.95 và chỉ học uncertainty/calibration trước.

Mỗi run dùng output directory riêng và validation lưu `checkpoint_best.pth` theo `(R@10+R@50)/2`. Eval pairwise đã vector hóa, không còn vòng lặp Python trên từng query-gallery pair. `eval.py --distance_mode mean` giúp kiểm tra uncertainty có thật sự cải thiện ranking hay chỉ làm target bias.

## Thứ tự chạy khuyến nghị

1. Khi GPU trống, chạy `./run_reproduce_6x.sh`. Tiêu chí pass: R@50 khoảng 60–62. Run này tắt validation giữa epoch để giữ RNG/lịch train gần run lịch sử nhất.
2. Chạy `./run_point_baseline.sh`. Point validation/eval dùng mean-only và bỏ qua hoàn toàn uncertainty heads. Tiêu chí go/no-go: tối thiểu khoảng `R@10>=39`, `R@50>=62`. Nếu không đạt, chưa nên tune uncertainty; cần kiểm tra preprocessing/Q-Former trước.
3. Chạy `./run_paper_from_point.sh`. Script warm-start từ best point checkpoint, train full paper recipe 20 epoch, đánh giá cả probabilistic và mean-only.
4. Chỉ giữ nhánh uncertainty nếu probabilistic tốt hơn mean-only. Nếu mean-only tốt hơn, ưu tiên sweep `lambda_fc` trong `{0.1,0.25,0.5}` và `lambda_cord` trong `{0.05,0.1}`; không tăng số epoch mù quáng.

Các run nên dừng theo validation, không theo train loss. Log lịch sử cho thấy train loss tiếp tục giảm trong khi recall epoch 8–10 không tăng.
