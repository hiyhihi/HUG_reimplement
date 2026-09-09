# Runbook thực nghiệm HUG theo hướng GVHD

> **Snapshot kiểm chứng: 2026-08-24, `main@cfe7b44`.** Số liệu dưới đây chỉ
> dùng artifact thật trong `results/checkpoints/supervisor_protocol_v2/`. Không
> diễn giải checkpoint thiếu `checkpoint_final.pth` là một run hoàn tất. Tại lúc
> kiểm tra, `hug_e2e/shirt/seed7` vẫn đang train; không dừng job này chỉ để đổi
> hướng, nhưng cũng không khởi tạo run HUG e2e mới cho đến khi qua các gate ở
> Mục 7.

## 1. Câu hỏi nghiên cứu và nguyên tắc diễn giải

Hướng của cô không phải là tối ưu một Recall đơn lẻ. Cần trả lời tuần tự:

1. Mean-retrieval baseline có ổn định và tái lập được không?
2. Khi thêm uncertainty, chất lượng clean/robustness có tốt hơn một cách công
   bằng không?
3. Uncertainty có thật sự dự báo failure, modality hỏng và mismatch không?
4. Chỉ khi câu 3 đúng mới thử reliability router/adaptive fusion.

Vì vậy, một số gần paper không đủ để gọi là reproduction. Paper báo Fashion-IQ
Dress `R@10/R@50 = 48.37/71.56`; khác batch, khởi tạo, số negative hoặc trạng
thái freeze đều là khác protocol cần ghi rõ.

Các bất biến:

- Corruption chỉ tác động query; gallery/target phải sạch và severity 0 phải là
  identity. So sánh dùng cùng manifest và seed.
- Không dùng gradient accumulation để tuyên bố tương đương physical batch 32:
  InfoNCE/HC vẫn chỉ nhìn negative pool của micro-batch.
- Báo cáo HUG e2e phải có cả `hug_e2e - point` và
  `hug_e2e - point_matched`. So sánh e2e batch 8 với Point batch 32 một mình là
  không công bằng.
- Trong Eq.15 hiện hành, \(v_q\) là hằng số với mọi gallery candidate của một
  query. Vì vậy query uncertainty không thể tự thay đổi thứ hạng; khác biệt
  mean/probabilistic chủ yếu có thể đến từ gallery variance. Do đó phải dùng
  AUROC/AUPRC, Spearman, risk-coverage/AURC, severity monotonicity và mismatch,
  không chỉ Recall.

## 2. Bốn kiến trúc/recipe đang chạy và phân công

| Model ID | Khởi tạo và phần trainable | Objective / retrieval | Batch, lịch | Người phụ trách | Mục đích đúng |
|---|---|---|---|---|---|
| `point` | BLIP-2; vision encoder frozen, Q-Former + query tokens trainable | Symmetric mean-only InfoNCE; không có FC/Cord; mean distance | B32, 10 epoch, warmup 0 | Person1 | Corrected deterministic baseline mạnh |
| `point_matched` | **Cùng kiến trúc/loss Point** | Như `point` | B8, 30 epoch, warmup 2 (trùng e2e) | Person2 | Control cho tác động batch/lịch/GPU, không phải HUG |
| `hug_e2e` | BLIP-2 init trực tiếp; vision encoder frozen, Q-Former + query tokens + uncertainty heads trainable; **không** init Point, **không** `--freeze_backbone` | `HC + 0.5 FC + 0.1 Cord`; mean và probabilistic | B8, 30 epoch, warmup 2 | Person2 | Corrected/stabilized end-to-end HUG candidate gần paper nhất |
| `hug_frozen_point` | Load `point/checkpoint_best`; freeze BLIP/Q-Former/query tokens, chỉ train uncertainty heads | Cùng loss HUG; mean và probabilistic | B32, 20 epoch, warmup 0; loss-scalar multiplier 10 | Person1 | Controlled ablation của uncertainty, **không** phải full HUG |

Điểm khác biệt quan trọng giữa hai người:

- Person1 trả lời “với mean encoder đã tốt, uncertainty head tự nó thêm được gì?”
  Vì mean encoder bị freeze, `point` và Frozen-Point phải trùng mean ranking nếu
  load đúng checkpoint. Đây là check hợp lệ, không phải gain của HUG.
- Person2 trả lời “objective HUG end-to-end có tốt hơn Point khi điều kiện
  compute giống nhau không?” Chỉ cặp `hug_e2e`–`point_matched` mới trả lời trực
  tiếp câu này.
- Bốn model không được gộp thành một bảng “HUG vs non-HUG”: `point_matched` là
  control, còn Frozen-Point là ablation; chỉ `hug_e2e` là candidate full HUG.

## 3. Kết quả hiện có và cách đọc

### 3.1 Person1 — hoàn tất đầy đủ clean + modality + robustness

Person1 đã có `checkpoint_final.pth` cho Point và Frozen-Point, đủ 3 seed
`42,7,123` × 3 category; 18 robustness completion markers cũng đã có. Trung
bình ± sample std của clean retrieval:

| Category | Point mean R@10 / R@50 | Frozen-Point mean R@10 / R@50 | Frozen-Point probabilistic R@10 / R@50 | Đọc đúng |
|---|---:|---:|---:|---|
| Dress | 47.98 ± 0.32 / 71.49 ± 0.90 | 47.98 ± 0.32 / 71.49 ± 0.90 | **48.31 ± 0.51 / 71.79 ± 1.14** | Prob. `+0.33/+0.30`; gần paper nhưng protocol khác |
| Shirt | 52.16 ± 0.32 / 71.16 ± 0.50 | 52.16 ± 0.32 / 71.16 ± 0.50 | **52.72 ± 0.32 / 71.44 ± 0.26** | Gain R@10 `+0.56`; nhỏ, chưa đủ nói robust hơn |
| Toptee | 54.38 ± 0.46 / 77.49 ± 0.59 | 54.38 ± 0.46 / 77.49 ± 0.59 | **54.46 ± 0.40 / 77.46 ± 0.31** | Gần như hoà (`+0.09/-0.03`) |

Điều khả quan:

- Point rất ổn định qua seed và category; Dress đã đạt gate baseline khoảng
  `48/71`.
- Mean của Frozen-Point trùng Point ở tất cả seed/category đúng như thiết kế;
  checkpoint init/freeze và evaluator đang nhất quán.
- Bài test modality xác nhận bài toán có composition: image-only thấp, text-only
  cao hơn image-only, và shuffle text làm R@10 gần 1–5%. Ví dụ Dress seed42:
  `image+text=47.99`, image-only `5.16`, text-only `27.66`, shuffled text
  `1.88`, shuffled image `20.23`.

Điều chưa khả quan:

- Lợi ích probabilistic nhỏ và đổi dấu ở Toptee; không có bằng chứng nó cải
  thiện retrieval một cách ổn định.
- Gộp 9 run/category/seed của Frozen-Point: R@10 severity 0 → 4 lần lượt là
  blur `51.50→48.48` (94.1%), occlusion `51.50→41.40` (80.4%), token dropout
  `51.50→28.74` (55.8%), typo `51.50→19.98` (38.8%). Probabilistic gần như y
  hệt (`51.83→48.78/41.75/28.74/20.16`). Text corruption là failure mode ưu
  tiên, chưa phải một gain robustness của HUG.
- Failure AUROC ở severity 4 chỉ khoảng `0.49–0.52`; gần random. Đây khớp audit
  Week 2–3: uncertainty chưa calibrated để nhận biết query khó/sai. Không xây
  router hay claim uncertainty hữu ích ở giai đoạn này.

Nguồn canonical: `results/supervisor_protocol_v2/clean/`, `modality/`,
`robustness/{point,hug_frozen_point}/`.

### 3.2 Person2 — chỉ có partial, và full HUG đang no-go

| Artifact hiện có | Trạng thái | Clean best checkpoint | So sánh hợp lệ | Nhận xét |
|---|---|---:|---:|---|
| `point_matched/shirt/seed42` | hoàn tất | 50.34 / 69.38 | — | Best ở epoch 2; train tiếp đến 30 epoch làm validation giảm |
| `point_matched/shirt/seed7` | hoàn tất | 51.47 / 69.97 | — | Best ở epoch 2; cùng pattern overfit/lịch quá dài |
| `hug_e2e/shirt/seed42` | hoàn tất | mean 37.78 / 57.51; prob. 37.98 / 58.54 | so Point-matched seed42: mean `−12.56/−11.87` | **No-go rõ ràng**; checkpoint best thực tế là epoch 1 |
| `hug_e2e/shirt/seed7` | đang chạy tại snapshot | chưa dùng để kết luận | cần đợi completion | 4 epoch đầu lặp lại collapse: R@10 `39.01→28.36→21.93→24.29` |
| `point_matched/dress/seed42` | chỉ có `train.log`, không checkpoint | — | — | Không hoàn tất/không evaluable; không coi là pilot |

Với HUG e2e seed42, HC `3.0168→1.8173`, FC `0.5730→0.0954`, Cord
`0.6938→0.0738` đều giảm, trong khi validation R@10 `37.98→18.35`. Điều đó cho
thấy code có forward/backward, không NaN/OOM, Q-Former không bị freeze; nhưng
objective hiện tại tối ưu train loss theo hướng phá hỏng retrieval validation.
Đây là lỗi/không tương thích của **recipe optimization**, chưa phải bằng chứng
“full HUG kém hơn Point” ở mọi setting, nhưng đủ mạnh để dừng sweep hiện tại.

Diagnostic mới trên Dress seed42/B8 cho cùng kết luận: Point đạt `47.00/71.10`,
nhưng HC-only, HC+FC, HC+Cord và full HUG chỉ khoảng `29–30/51–53`. Vì HC-only
cũng hỏng, FC và Cord không phải nguyên nhân chính. Vấn đề cần kiểm tra là HC
và cách nó dùng uncertainty, không phải đổi batch hay chạy nhiều seed.

Vì vậy, Person2 **chưa chạy tiếp full HUG**. Chỉ chạy hai kiểm tra nhỏ ở output
root mới: (1) HC dùng mean distance, không cộng uncertainty; (2) HC lấy trung
bình thay vì cộng dồn các negative. Hai kiểm tra này giúp phân biệt “uncertainty
làm hỏng distance” với “cách tính HC làm gradient quá mạnh”.

Kết luận ngắn: **Person1 đã đủ phần baseline/Frozen-Point cô giao**, nhưng chưa
đủ để kết luận uncertainty hữu ích vì các metric dự báo lỗi vẫn gần ngẫu nhiên.
**Person2 không chạy sweep tiếp**; chỉ chạy hai ablation dưới đây rồi quyết định.

## 4. Mốc kỳ vọng và đối chiếu hiện tại

Đây là target/gate vận hành, không phải con số đã đạt hoặc lời hứa kết quả.

| Khâu | Kỳ vọng hợp lý trước khi đi tiếp | Hiện tại | Quyết định |
|---|---|---|---|
| Point B32, Dress | R@10 khoảng 47.5–48.5; R@50 khoảng 70.5–72.5 qua 3 seed | 47.98 ± 0.32 / 71.49 ± 0.90 | **Pass** |
| Frozen-Point sanity | mean trùng Point; prob. không làm clean giảm quá 0.5 | Mean trùng; probabilistic Dress `+0.33/+0.30` | **Pass** cho ablation, chưa pass reliability |
| Uncertainty calibration | AUROC ổn định >0.60, Spearman(U,error) dương, AURC giảm và U tăng đúng modality | AUROC ~0.5; Week 2–3 Spearman gần 0 | **Fail / chưa làm C2** |
| Point-matched pilot | Có paired Dress seed42; best checkpoint không collapse, lựa epoch qua validation | Dress diagnostic 47.00/71.10 | **Pass control** |
| HUG e2e pilot | Dress seed42 có `checkpoint_final`; best HUG không thấp hơn Point-matched quá 0.5 R@10 và 1.0 R@50; loss hữu hạn | Dress HC/full chỉ ~29–30/51–53 | **No-go** |
| HUG e2e multi-seed/category | Chỉ sau pilot pass, paired 42/7/123 và CI/delta | đang chạy dở Shirt seed7 | **Không chạy tiếp sweep** |
| Router/U1–U3 | chỉ sau calibration gate pass, clean loss ≤0.3–0.5 R@10 | chưa có reliability signal | **Chưa được phép** |

Mốc tốt nhất có thể kỳ vọng ngắn hạn là **ổn định/diagnose** HUG e2e: một
checkpoint Dress có clean ngang Point-matched trong sai số nhỏ, không phải cố
chạy để vượt Point. Nếu đạt, mới kỳ vọng multi-seed xác nhận gain sạch nhỏ
(khoảng 0–1 R@10) và metric calibration rõ ràng hơn; không nên dự đoán gain lớn
chỉ từ Frozen-Point seed42.

## 5. Protocol chạy chuẩn

### Pha 0 — preflight

```bash
source ref/LAVIS/.venv/bin/activate
./scripts/run_supervisor_experiments.sh preflight
```

Preflight phải pass syntax, data 3 category và PDF ở
`word&md&pdf/2601.11393v2 (1).pdf`.

### Pha 1 — baseline và Frozen-Point (đã hoàn tất)

Không cần train lại nếu không có lý do ablation mới. Có thể tổng hợp:

```bash
./scripts/run_supervisor_person1.sh modality
./scripts/run_supervisor_person1.sh robustness
./scripts/run_supervisor_experiments.sh summarize
```

Không dùng các lệnh trên với `FORCE=1` trừ khi chủ động ghi đè artifact và đã
ghi lý do. `checkpoint_final.pth` là completion marker; `checkpoint_best.pth`
chỉ dùng evaluation.

### Pha 2 — chẩn đoán Person2 trước khi chạy thêm

Job `hug_e2e/shirt/seed7` đang chạy nên để nó tự kết thúc/resume bình thường;
không xoá hoặc overwrite checkpoint. Khi có kết quả, ghi lại epoch tốt nhất và
quyết định theo gate, không lấy final epoch thay best.

Sau đó **không dùng** `run_supervisor_person2.sh clean` mặc định (vì nó sweep
category/seed khi `SEEDS`, `CATEGORIES` không bị giới hạn). Lần diagnostic mới
phải dùng output root mới để không ghi đè artifact dở, và chỉ Dress seed42:

```bash
# Ví dụ cấu hình; chỉ chạy sau khi đồng ý bắt đầu diagnostic mới.
RESULT_ROOT=results/supervisor_diagnostics_v1 \
CHECKPOINT_ROOT=checkpoints/supervisor_diagnostics_v1 \
SEEDS=42 CATEGORIES=dress MODELS=point_matched,hug_e2e \
./scripts/run_supervisor_experiments.sh pilot
```

`train.py` giờ có log dễ đọc: model nào được học, gradient có đi vào Q-Former
và query tokens không, khoảng cách đúng/sai, mức uncertainty và learning rate.
Nó cũng tự dừng khi validation không tốt hơn trong vài epoch, nhưng vẫn giữ
checkpoint tốt nhất để đánh giá. Query tokens được log riêng, không tính lẫn
vào Q-Former; ablation mean-distance cũng chọn checkpoint bằng mean distance.

Hai diagnostic đã chạy ở output riêng; lệnh dưới chỉ để tái lập, không chạy lại:

```bash
./scripts/run_hc_diagnostics.sh
```

- `hug_hc_mean_distance`: bỏ uncertainty khỏi khoảng cách HC.
- `hug_hc_average_negatives`: giữ uncertainty, nhưng lấy trung bình negative
  để mỗi batch không tạo gradient quá lớn. Script tự resume run dở an toàn.

Kết quả: mean-distance vẫn chỉ `29.60/51.56`; average-negatives giảm xuống
`6.25/14.53`. Gradient Q-Former/query tokens có và vision encoder vẫn frozen.
Hai ablation đều fail gate, nên dừng chạy thêm và audit lại công thức HC.

Point B8 là `47.00/71.10`, nên gate là `46.50/70.10`. Cả hai ablation đều thấp
hơn xa gate; không chạy Dress 3 seed, không tuning tiếp và phải đối chiếu công
thức HC với paper trước khi train thêm.

Không giảm batch xuống 4 hoặc dùng gradient accumulation như một “fix” trước:
chúng đổi negative pool và làm mất paired control. Không khởi tạo từ Point rồi
gọi nó e2e reproduction; nếu thử warm-start, phải đặt model ID mới (ví dụ
`hug_warmstart_e2e`) và báo là ablation.

### Pilot tăng Recall từ Point — `point_robust`

Đây là **ablation mới**, không phải full HUG. Mục tiêu là giữ Recall sạch của
Point và giúp mô hình ít phụ thuộc vào từng từ hoàn hảo trong câu mô tả.

Mỗi batch được học hai lần: (1) câu sạch với Point InfoNCE; (2) cùng câu nhưng
rơi ngẫu nhiên 10% token với InfoNCE và một loss giữ thứ hạng gần câu sạch.
Hai lượt backward chạy tuần tự để vẫn dùng batch vật lý 32 mà không tăng mạnh
VRAM. Point ban đầu được đo/lưu ở epoch 0; checkpoint này chỉ bị thay khi điểm
validation sạch trung bình R@10/R@50 tăng.

Chạy pilot Dress seed42 (output root hoàn toàn mới):

```bash
./scripts/run_point_robust_recall.sh
```

Có thể chỉ train/eval sạch trước bằng `RUN_ROBUSTNESS=0`; script tự resume từ
`checkpoint_last.pth` và không ghi đè Point baseline. Cấu hình mặc định:
Point seed42 → 8 epoch, LR `3e-6`, dropout `0.10`, trọng số robust/consistency
`0.5/0.5`, early stopping patience 2.

Đọc kết quả theo delta với đúng Point Dress seed42:

- Thành công chính: clean R@10 tăng ít nhất `+0.5`, R@50 không giảm quá `0.5`.
- Hữu ích về robustness: clean R@10 không giảm quá `0.5`, còn typo/token
  dropout severity 3–4 tăng trung bình ít nhất `+2.0` R@10.
- Fail: clean R@10 giảm quá `0.5`, loss không hữu hạn, hoặc best vẫn là trọng
  số khởi tạo. Khi fail, dừng; chưa đổi batch/lambda đồng thời.
- Chỉ khi qua một gate trên mới chạy thêm Dress seed7/123. Không mở
  Shirt/Toptee trước khi delta lặp lại qua seed.

Kết quả Dress 3 seed đã hoàn tất:

| Seed | Point R@10/R@50 | Point Robust R@10/R@50 | Delta |
|---:|---:|---:|---:|
| 42 | 47.99/70.90 | 49.08/71.44 | +1.09/+0.55 |
| 7 | 47.65/71.05 | 49.18/72.43 | +1.54/+1.39 |
| 123 | 48.29/72.53 | 49.23/72.43 | +0.94/−0.10 |
| Mean | 47.98/71.49 | 49.17/72.10 | **+1.19/+0.61** |

Clean gate pass ở cả 3 seed. Nhiễu text cải thiện cùng chiều, nhưng severity
3–4 chỉ tăng trung bình khoảng `+0.61` R@10, chưa đạt robustness gate `+2.0`.

Control `point_continued` dùng cùng checkpoint Point, batch 32, LR `3e-6`, 8
epoch và early stopping patience 2, nhưng chỉ train clean InfoNCE. Ba seed đã
hoàn tất:

| Model, Dress 3 seed | R@10/R@50 | Delta so với Point |
|---|---:|---:|
| Point | 47.98±0.32/71.49±0.90 | — |
| Point Continued | 49.03±0.39/72.05±0.31 | +1.06/+0.56 |
| Point Robust | 49.17±0.08/72.10±0.57 | +1.19/+0.61 |

Point Robust chỉ hơn Point Continued `+0.13/+0.05` clean. Dưới mọi mức nhiễu
text, delta Robust − Continued là `+0.19/+0.43`; riêng severity 3–4 là
`+0.23/+0.67`, và R@10 không tăng ở mọi seed. Vì vậy phần lớn clean gain đến
từ train thêm; robust objective có đóng góp nhỏ nhưng chưa đủ ổn định để claim
robustness gain.

### Pha 3 — chỉ sau e2e pilot pass

```bash
SEEDS=42,7,123 CATEGORIES=dress ./scripts/run_supervisor_person2.sh clean
./scripts/run_supervisor_person2.sh modality
./scripts/run_supervisor_person2.sh robustness
./scripts/run_supervisor_experiments.sh summarize
```

Chỉ mở Shirt/Toptee sau Dress multi-seed. Robustness luôn chạy trên best
checkpoint, same manifest và báo paired delta/CI; không chạy router nếu failure
AUROC/AUPRC, AURC và severity monotonicity chưa qua Mục 4.

## 6. Cách gọi tên trong báo cáo

Dùng “Corrected Point baseline”, “Frozen-Point HUG controlled ablation”, và
“Corrected/stabilized end-to-end HUG reproduction candidate”. Có thể nói
Frozen-Point Dress *numerically near* paper ở R@10, nhưng không gọi exact
reproduction hay “HUG đã cải thiện chắc chắn”.

Chi tiết audit lịch sử seed42 nằm ở `WEEK2_WEEK3_FINDINGS.md`; protocol này ưu
tiên artifact v2 multi-seed phía trên khi hai nguồn mâu thuẫn.
