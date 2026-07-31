# Đề xuất hướng nghiên cứu mở rộng HUG

## Reliability-Calibrated Adaptive Composition for Robust Composed Image Retrieval

> Trạng thái tài liệu: đề xuất để thảo luận và chốt hướng.  
> Tài liệu này **không** đồng nghĩa với việc đã chốt kiến trúc hay được phép sửa code/chạy thực nghiệm.

## 1. Tóm tắt khuyến nghị

File `Contributions.docx` đề xuất ba hướng:

1. Tương tác chéo đa phương thức bằng cross-attention.
2. Gộp thích ứng giữa visual và semantic.
3. Đánh giá độ bền khi ảnh mờ hoặc text thiếu thông tin.

Ba ý tưởng đều hợp lý, nhưng nếu áp dụng nguyên văn vào HUG thì C1 và một phần
C2 bị trùng với baseline:

- HUG đã dùng BLIP-2 Q-Former để thực hiện early cross-modal interaction giữa
  modification text và visual tokens.
- HUG đã có dynamic weighting theo uncertainty của reference image, text và
  multimodal coordination.

Vì vậy, hướng được khuyến nghị là:

> **Không thêm cross-attention chung chung. Thay vào đó, nghiên cứu xem
> uncertainty có được hiệu chỉnh để phản ánh đúng reliability của từng modality
> hay không, rồi dùng reliability đó để điều khiển adaptive composition dưới
> các mức corruption.**

Tên tạm:

**Reliability-Calibrated Heterogeneous Uncertainty for Robust Composed Image
Retrieval (RC-HUG)**.

Hướng này kết hợp đúng tinh thần C2 và C3, đồng thời biến C1 thành một cơ chế
interaction có mục tiêu rõ ràng: xử lý **disagreement/conflict** thay vì chỉ
thêm capacity.

Lý do chọn:

- Point baseline hiện đạt `R@10=47.94`, `R@50=70.95` trên Fashion-IQ Dress,
  chỉ thấp hơn HUG paper lần lượt 0.43 và 0.61 điểm.
- Một deterministic baseline mạnh như vậy đặt ra câu hỏi nghiên cứu quan trọng:
  uncertainty mang lại lợi ích gì ngoài Recall sạch?
- HUG tuyên bố uncertainty phản ánh quality/ambiguity nhưng chủ yếu minh họa
  định tính; chưa có stress-test có kiểm soát, calibration, failure detection
  hoặc selective retrieval.
- Robustness là hướng có câu chuyện khoa học hoàn chỉnh hơn việc cố tăng thêm
  một lượng rất nhỏ R@50.

## 2. Trạng thái thực nghiệm hiện tại

### 2.1. Các mốc đã có

| Mô hình | Dress R@10 | Dress R@50 | Vai trò |
|---|---:|---:|---|
| Legacy được reproduce | 37.68 | 62.07 | Xác nhận pipeline cũ |
| Point baseline mới | 47.94 | 70.95 | Deterministic strong baseline |
| HUG paper báo cáo | 48.37 | 71.56 | Mục tiêu reproduce |

Điểm đáng chú ý là point baseline mới gần bằng full HUG. Tuy nhiên không nên
kết luận ngay rằng uncertainty vô ích, vì point recipe hiện tại không hoàn toàn
giống “Point Matching” trong ablation của paper. Cần so sánh có kiểm soát trên:

- cùng backbone initialization;
- cùng batch size, optimizer, epochs và early stopping;
- cùng mean representation;
- cùng seed;
- chỉ thay đổi uncertainty head/distance/loss.

### 2.2. HUG hiện làm gì?

HUG mô hình hóa 32 Gaussian components:

- mean query:
  \(\mu_q=h([LQ],x_t,x_r)\);
- mean target:
  \(\mu_c=h([LQ],\emptyset,x_c)\);
- visual uncertainty:
  \(v_r=g_V(\mu_r)\), \(v_c=g_V(\mu_c)\);
- text uncertainty:
  \(v_t=g_T(\mu_t)\);
- coordination uncertainty:
  \(v_m=g_M(\mu_q)\);
- query uncertainty:
  \(v_q=\sum_x w_xv_x\), với \(w_x\propto\exp(-v_x)\).

Do đó:

- C1 “early cross-modal interaction” đã tồn tại trong \(\mu_q\).
- C2 “adaptive fusion” đã tồn tại ở **variance fusion**, nhưng chưa thực sự
  hiệu chỉnh reliability và chưa điều khiển mean composition.
- C3 “robustness” là khoảng trống rõ nhất và có thể dùng để kiểm chứng C2.

## 3. Đánh giá từng gợi ý trong Contributions.docx

### 3.1. C1 — Cross-modal Interaction

**Đánh giá:** hợp lý về động cơ, nhưng novelty thấp nếu chỉ thêm cross-attention.

Q-Former đã để learnable query/text tokens attend vào visual encoder outputs.
Thêm một block cross-attention tương tự có thể:

- tăng số tham số nhưng không tạo giả thuyết mới;
- khó chứng minh gain đến từ interaction thay vì capacity;
- tăng VRAM, trong khi full paper batch 32 đã từng OOM trên RTX 4090;
- dễ overfit Fashion-IQ vì dữ liệu train mỗi category nhỏ.

**Cách tái định nghĩa có giá trị hơn:**

1. **Disagreement-aware interaction:** chỉ kích hoạt adapter khi visual và text
   có xung đột hoặc reliability khác nhau.
2. **Bidirectional interaction có kiểm soát:** text attend image để tìm thuộc
   tính cần sửa; image attend text để tìm vùng/khái niệm cần giữ hoặc loại bỏ.
3. **Residual interaction:** giữ \(\mu_q\) của point checkpoint và chỉ học một
   residual nhỏ, tránh phá representation 70.95.
4. **Conflict token:** thêm một số token chuyên ước lượng “keep/change/conflict”
   thay vì thêm toàn bộ cross-attention stack.

**Khuyến nghị:** không chọn C1 độc lập làm contribution chính. Chỉ dùng nó như
module phụ cho hướng reliability/conflict-aware.

### 3.2. C2 — Adaptive Fusion

**Đánh giá:** khả thi và phù hợp nhất, nhưng phải vượt ra ngoài dynamic weighting
hiện tại.

Dynamic weighting của HUG dùng công thức cố định:

\[
w_x=\operatorname{softmax}(-v_x), \qquad x\in\{r,t,m\}.
\]

Nó có ba hạn chế cần kiểm chứng:

1. Nếu variance chưa calibrated, inverse-variance weighting cũng không đáng tin.
2. Gate chỉ nhìn variance từng phần, chưa nhìn cross-modal disagreement,
   availability hoặc query intent.
3. Weight chỉ gộp uncertainty; mean query vẫn hoàn toàn là \(\mu_m\).

**Đề xuất:** Reliability-Calibrated Adaptive Router.

Đầu vào router có thể gồm:

- pooled hoặc component-level \(\mu_r,\mu_t,\mu_m\);
- \(\log v_r,\log v_t,\log v_m\);
- disagreement như cosine distance giữa projected visual/text components;
- attention entropy hoặc token informativeness;
- availability mask;
- nếu có corruption training: predicted severity, không dùng ground-truth
  severity ở inference.

Router sinh \(\alpha_r,\alpha_t,\alpha_m\) theo từng sample hoặc từng Q-Former
component. Hai mức thiết kế:

#### Thiết kế an toàn: uncertainty routing

\[
v_q=\sum_x \alpha_xv_x+
\sum_x\alpha_x(\mu_x-\bar{\mu})^2.
\]

Term thứ hai là between-expert disagreement. Nó giúp uncertainty tăng khi các
modality/expert bất đồng, thay vì chỉ trung bình variance nội tại.

Ưu điểm:

- giữ nguyên mean point baseline;
- ít nguy cơ giảm clean Recall;
- phù hợp stage paper đang freeze backbone;
- dễ ablate với fixed weight và HUG inverse-variance weight.

#### Thiết kế mở rộng: residual mean routing

\[
\mu'_q=\operatorname{norm}\left(
\mu_m+\gamma_r\alpha_rP_r(\mu_r-\mu_m)
+\gamma_t\alpha_tP_t(\mu_t-\mu_m)
\right).
\]

Không nên dùng \(\alpha_r\mu_r+\alpha_t\mu_t\) trực tiếp vì reference image mô
tả trạng thái **trước modification**, có thể kéo query quay về candidate image.
Residual formulation an toàn hơn và cho phép khởi tạo \(\gamma_r,\gamma_t=0\)
để bắt đầu đúng tại point checkpoint.

### 3.3. C3 — Robustness Evaluation

**Đánh giá:** rất khả thi, cần thiết và có thể trở thành contribution thực
nghiệm độc lập.

Chỉ báo Recall sau blur/word-drop là chưa đủ. Vì HUG là probabilistic model,
robustness protocol nên trả lời ba lớp câu hỏi:

1. Retrieval giảm bao nhiêu khi chất lượng modality giảm?
2. Uncertainty có tăng đúng theo severity và dự đoán được retrieval failure?
3. Adaptive gate có giảm trọng số đúng modality bị hỏng hay không?

Điểm cần cẩn thận:

- Blur/JPEG/brightness thường giữ semantics, nên target cũ vẫn hợp lệ.
- Word dropout có thể xóa thuộc tính quyết định và làm query đổi nghĩa/không còn
  xác định. Recall giảm trong trường hợp này không nhất thiết là lỗi mô hình.
- Caption shuffling tạo mismatch, nên thích hợp đo coordination uncertainty hoặc
  OOD detection, không thích hợp dùng target cũ để diễn giải Recall chuẩn.

Do đó cần chia corruption thành ba nhóm.

#### Nhóm A — Semantics-preserving

Được dùng cho cả Recall và calibration:

- Gaussian blur;
- JPEG compression;
- brightness/contrast nhẹ;
- partial occlusion không che toàn bộ garment;
- typo/character noise;
- word-order perturbation có kiểm soát;
- paraphrase hoặc synonym giữ nguyên modification intent.

#### Nhóm B — Information-removing

Được dùng cho stress-test, risk-coverage và uncertainty:

- xóa adjective/attribute;
- token dropout;
- caption truncation;
- chỉ giữ một trong hai caption Fashion-IQ;
- che vùng ảnh liên quan đến thuộc tính.

Recall vẫn có thể báo cáo nhưng phải ghi rõ target có thể trở nên ambiguous.

#### Nhóm C — Conflict/OOD

Được dùng cho coordination uncertainty/failure detection:

- ghép reference với modification text của triplet khác;
- thay đổi color/shape attribute thành mâu thuẫn;
- text không liên quan ảnh;
- ảnh ngoài domain hoặc text rỗng.

Không nên coi Recall tới target cũ là metric chính.

## 4. Hướng nghiên cứu chính: RC-HUG

### 4.1. Câu hỏi nghiên cứu

**RQ1.** Uncertainty của HUG có calibrated với retrieval error và mức corruption
hay chỉ là một latent score giúp tối ưu loss?

**RQ2.** Learned reliability routing có tốt hơn fixed fusion và công thức
\(\operatorname{softmax}(-v)\) của HUG trên cả clean và corrupted queries?

**RQ3.** Corruption-aware uncertainty supervision có cải thiện robustness mà
không làm giảm deterministic clean performance?

**RQ4.** Gain trên Fashion-IQ đến từ true multimodal composition hay unimodal
shortcuts?

**RQ5.** Coordination uncertainty có phát hiện được text–image conflict và
ambiguous queries không?

### 4.2. Giả thuyết

**H1.** Với model hiện tại, mean uncertainty có thể chưa tăng đơn điệu theo
corruption severity và correlation với rank error còn yếu.

**H2.** Corruption-aware monotonic calibration sẽ cải thiện failure detection và
risk-coverage, kể cả khi clean Recall thay đổi ít.

**H3.** Reliability router nhìn cả variance và disagreement sẽ ổn định hơn
inverse-variance rule khi một modality bị corruption.

**H4.** Freezing strong point backbone và học uncertainty/router trước sẽ giữ
clean mean Recall tốt hơn end-to-end paper fine-tuning.

**H5.** Một phần query Fashion-IQ giải được bằng text-only hoặc image-only; gain
thực sự của cross-modal method sẽ rõ hơn trên compositional-only subset.

### 4.3. Các loss ở mức đề xuất

Chưa chốt hệ số và chưa triển khai. Công thức dưới đây là candidate design.

#### Retrieval loss

\[
\mathcal L_{\text{ret}}=
\mathcal L_{\text{point/paper}}(q_{\text{clean}},c)
+\lambda_{\text{rob}}\mathcal L_{\text{point/paper}}(q_{\text{corr}},c).
\]

Chỉ áp dụng term corrupted retrieval cho corruption giữ semantics.

#### Monotonic uncertainty calibration

\[
\mathcal L_{\text{mono}}=
\max(0,m+U(q^{(s_1)})-U(q^{(s_2)})),\quad s_1<s_2.
\]

Uncertainty phải tăng khi severity tăng. Có thể áp dụng riêng:

- visual corruption giám sát \(U_r\);
- text corruption giám sát \(U_t\);
- mismatch giám sát \(U_m\).

#### Gate supervision yếu

Với image corruption:

\[
\alpha_r^{\text{corr}} < \alpha_r^{\text{clean}},
\]

và tương tự với text. Nên dùng ranking/monotonic loss thay vì ép gate về nhãn
one-hot để tránh heuristic quá cứng.

#### Clean-teacher consistency

\[
\mathcal L_{\text{KD}}=
D_{\mathrm{KL}}\left(
p(c|q_{\text{clean}})\,\|\,p(c|q_{\text{corr}})
\right).
\]

Chỉ dùng cho semantics-preserving corruption. Teacher là frozen point model
hoặc EMA model.

#### Gate regularization

- entropy floor để tránh gate collapse;
- load-balancing nhẹ giữa experts;
- residual magnitude penalty để không phá mean space;
- không dùng regularization buộc trọng số trung bình bằng nhau.

Tổng quát:

\[
\mathcal L=
\mathcal L_{\text{ret}}
+\lambda_u\mathcal L_{\text{mono}}
+\lambda_g\mathcal L_{\text{gate}}
+\lambda_d\mathcal L_{\text{KD}}
+\lambda_e\mathcal L_{\text{entropy}}.
\]

## 5. Protocol đánh giá đề xuất

### 5.1. Datasets

Pha đầu:

- Fashion-IQ Dress để iterate nhanh và nối tiếp checkpoint hiện có.

Pha xác nhận:

- Fashion-IQ: Dress, Shirt, Toptee;
- CIRR: Recall@K và subset Recall@K.

Nếu chỉ có Dress thì khó kết luận robustness tổng quát. Ít nhất final table cần
ba Fashion-IQ categories. CIRR giúp kiểm tra ngoài fashion domain.

### 5.2. Corruption levels

Mỗi corruption cần 4–5 severity levels, gồm level 0 sạch. Ví dụ ban đầu:

- blur sigma: `0, 0.5, 1, 2, 4`;
- JPEG quality: `100, 75, 50, 25, 10`;
- occlusion ratio: `0, 0.1, 0.2, 0.35, 0.5`;
- text dropout: `0, 0.1, 0.2, 0.35, 0.5`;
- typo rate: `0, 0.05, 0.1, 0.2, 0.3`;
- mismatch rate cho tập detection: balanced positive/negative pairs.

Các mức trên chỉ là khởi tạo. Cần xem qualitative samples trước khi chốt để
tránh corruption quá nhẹ hoặc làm mất hoàn toàn semantics.

Corruption phải:

- deterministic theo seed;
- tạo on-the-fly hoặc lưu metadata, không sửa dữ liệu gốc;
- giống nhau giữa các model;
- có unit test xác nhận level 0 không đổi input.

### 5.3. Metrics retrieval

- Recall@10, Recall@50 trên Fashion-IQ;
- Recall@1/5/10/50 và subset Recall trên CIRR;
- relative retention:

\[
\mathrm{RRR}(s)=R(s)/R(0);
\]

- corruption curve AUC;
- worst-severity performance;
- average corruption performance;
- clean–corrupt trade-off.

### 5.4. Metrics uncertainty/calibration

Không nên chỉ báo cáo mean variance.

1. **Spearman correlation** giữa uncertainty và target rank/log-rank.
2. **Failure AUROC/AUPRC**, với failure được định nghĩa target không nằm trong
   top-K.
3. **Risk–coverage curve và AURC:** bỏ qua các query uncertainty cao, đo lỗi
   trên phần còn lại.
4. **Calibration theo bin:** chia query theo uncertainty quantile và báo cáo
   empirical Recall/failure rate.
5. **Severity monotonicity:** tỷ lệ query có uncertainty tăng đúng khi severity
   tăng.
6. **Mismatch AUROC:** dùng coordination uncertainty phân biệt matched và
   mismatched image–text.
7. **Gate–quality alignment:** correlation giữa severity và thay đổi trọng số
   modality tương ứng.

Nếu tạo xác suất từ retrieval logits, có thể thêm ECE/Brier score cho sự kiện
top-K success, nhưng cần temperature calibration trên validation split riêng.

### 5.5. Thống kê

- Ít nhất 3 training seeds cho model cuối và ablation chính.
- Query-level bootstrap 1,000 lần để báo 95% confidence interval.
- Khi so hai model, dùng paired bootstrap trên cùng query.
- Không chọn checkpoint theo test corruption. Chọn theo clean validation hoặc
  một validation robustness score được định nghĩa trước.

## 6. Ma trận thực nghiệm

### Pha 0 — Đóng băng baseline

| ID | Model | Mục tiêu |
|---|---|---|
| B0 | Legacy | Kiểm tra tương thích lịch sử |
| B1 | Point strong baseline | Mốc deterministic 70.95 |
| B2 | Paper mean-only | Tách effect do mean training |
| B3 | Paper probabilistic | Đo effect thực của uncertainty distance |

Điều kiện go/no-go:

- Nếu B3 không hơn B2 trên clean và uncertainty không dự đoán failure tốt hơn
  random, chưa nên xây router; phải sửa calibration trước.

### Pha 1 — Evaluation-only robustness

Không sửa model. Chỉ tạo corruption/evaluation protocol.

| ID | Model | Clean | Image corruption | Text corruption | Mismatch |
|---|---|---:|---:|---:|---:|
| E0 | Point | ✓ | ✓ | ✓ | score-based |
| E1 | HUG mean | ✓ | ✓ | ✓ | score-based |
| E2 | HUG probabilistic | ✓ | ✓ | ✓ | uncertainty |

Đây là pha ưu tiên cao nhất vì rẻ, giúp biết vấn đề thực sự nằm ở đâu trước khi
thiết kế kiến trúc.

### Pha 2 — Calibrated uncertainty

| ID | Monotonic | Modality dropout | KD | Mục tiêu |
|---|---:|---:|---:|---|
| U0 |  |  |  | HUG hiện tại |
| U1 | ✓ |  |  | Calibration theo severity |
| U2 | ✓ | ✓ |  | Robust encoder/head |
| U3 | ✓ | ✓ | ✓ | Giữ clean retrieval |

### Pha 3 — Adaptive router

| ID | Fusion | Input gate | Corruption supervision |
|---|---|---|---:|
| F0 | static equal | none |  |
| F1 | HUG inverse variance | variance |  |
| F2 | learned uncertainty routing | variance |  |
| F3 | reliability routing | variance + disagreement | ✓ |
| F4 | residual mean routing | variance + disagreement | ✓ |

F4 chỉ thực hiện nếu F3 chứng minh được gate calibrated và không collapse.

### Pha 4 — Interaction/conflict extension

Chỉ triển khai nếu error analysis cho thấy coordination conflict là failure mode
chính:

- conflict tokens;
- keep/change decomposition;
- lightweight bidirectional cross-attention adapter;
- phrase/attribute-guided interaction.

Không nên triển khai full cross-attention stack trước F0–F3.

## 7. True-composition audit

Một hướng mở rộng rất đáng làm là tách query theo mức phụ thuộc modality:

1. image-only giải được;
2. text-only giải được;
3. cả hai unimodal đều sai nhưng composed query đúng;
4. ambiguous/noisy query.

Quy trình khả thi:

- chạy image-only, text-only và composed point model;
- dựa trên target rank để tạo các subset tự động;
- kiểm tra thủ công một sample nhỏ;
- báo cáo performance riêng trên compositional-only subset;
- đo uncertainty/gate trên từng subset.

Điều này quan trọng vì nghiên cứu năm 2026 chỉ ra 32.2%–83.6% query trong bốn
CIR benchmarks có thể được giải bằng một modality, và nhiều query còn noisy
hoặc ambiguous. Một model tăng Recall tổng có thể chỉ tận dụng shortcut chứ
không composition tốt hơn.

True-composition audit có thể trở thành contribution thứ ba:

> RC-HUG không chỉ robust hơn dưới corruption mà còn cải thiện rõ trên các query
> thực sự cần multimodal composition.

## 8. Các hướng thay thế đáng cân nhắc

### Hướng A — Uncertainty-calibrated selective CIR

Mục tiêu không chỉ retrieve mà còn biết khi nào không nên tin kết quả.

Output:

- target ranking;
- confidence/uncertainty;
- cơ chế abstain hoặc yêu cầu người dùng làm rõ query.

Ưu điểm:

- tận dụng trực tiếp probabilistic representation;
- metric risk–coverage rõ;
- phù hợp query thiếu thông tin hoặc conflict;
- ít cạnh tranh trực tiếp với các model chỉ tối ưu Recall.

Có thể mở rộng thành multi-round retrieval: khi uncertainty cao, hệ thống hỏi
thêm về color/style/attribute có entropy cao.

### Hướng B — Conflict-aware composed retrieval

Thay vì coi mọi khác biệt image–text là noise, tách:

- **intended modification**: khác biệt cần thực hiện;
- **harmful conflict**: text mơ hồ, không liên quan hoặc mâu thuẫn ngoài ý định;
- **preserved content**: thuộc tính phải giữ.

Coordination uncertainty \(v_m\) hiện chỉ cho một score/vector, chưa phân biệt ba
loại trên. Có thể thêm keep/change/conflict heads hoặc tokens.

Hướng này có novelty cao nhưng cần annotation/pseudo-label tốt. Có thể dùng rule
trên Fashion-IQ attributes hoặc LLM offline, nhưng phải kiểm soát chất lượng
pseudo-label và chi phí.

### Hướng C — Attribute/component-level uncertainty

HUG nói 32 Gaussian components tương ứng fine-grained concepts nhưng không có
supervision đảm bảo component nào là color, sleeve, shape, texture.

Có thể:

- align Q-Former components với noun/adjective phrases;
- dùng phrase-guided contrast;
- đo component uncertainty khi corruption đúng thuộc tính;
- yêu cầu locality: che vùng sleeve chỉ làm tăng một subset components.

Novelty và interpretability cao, nhưng cần phrase extraction/region grounding.
Đây là hướng dài hạn, không phù hợp làm bước đầu.

### Hướng D — Robustness/evaluation paper

Nếu model improvements nhỏ nhưng phát hiện uncertainty không calibrated, có thể
chuyển trọng tâm thành empirical study:

> Do probabilistic CIR models know when multimodal queries are unreliable?

Contribution:

- corruption taxonomy dành riêng cho CIR;
- calibration/selective retrieval metrics;
- đánh giá HUG, point và các probabilistic baselines;
- phân tích shortcut/ambiguity.

Hướng này vẫn có giá trị nếu kết quả “negative”, miễn protocol chặt và phân tích
sâu.

## 9. Xếp hạng hướng theo tính khả thi

| Hướng | Novelty | Rủi ro | Chi phí | Khuyến nghị |
|---|---:|---:|---:|---|
| Robustness + uncertainty calibration | Cao | Thấp–TB | Thấp | Làm đầu tiên |
| Reliability-gated uncertainty fusion | Cao | Trung bình | Thấp–TB | Hướng model chính |
| Residual adaptive mean fusion | TB–Cao | Trung bình | Trung bình | Làm sau gate |
| True-composition audit | Cao | Trung bình | Thấp–TB | Rất nên có |
| Selective/interactive CIR | Cao | Trung bình | Trung bình | Hướng paper độc lập |
| Generic extra cross-attention | Thấp | Trung bình | Cao | Không ưu tiên |
| Attribute-grounded components | Rất cao | Cao | Cao | Hướng dài hạn |
| LLM conflict neutralization | TB | Cao | Cao | Chỉ khi có tài nguyên |

## 10. Tiêu chí thành công và stop rules

### Calibration module được giữ khi

- uncertainty–failure AUROC tăng có ý nghĩa;
- AURC giảm;
- severity monotonicity tăng;
- clean Recall giảm không quá 0.3–0.5 điểm;
- kết quả lặp lại trên ít nhất hai categories hoặc Dress + CIRR.

### Adaptive router được giữ khi

- tốt hơn cả fixed equal và HUG inverse-variance;
- weight của corrupted modality giảm đơn điệu theo severity;
- gate không collapse về một expert;
- robust AUC/average corruption Recall tăng;
- clean Recall không bị đánh đổi đáng kể.

### Residual mean fusion chỉ được làm khi

- uncertainty router đã calibrated;
- error analysis cho thấy mean representation còn fail vì modality imbalance;
- khởi tạo residual zero giữ đúng point checkpoint ở step 0.

### Dừng một hướng khi

- gain chỉ xuất hiện ở một seed;
- gain clean đến từ image-only/text-only shortcut;
- uncertainty thay đổi nhưng không correlate với failure;
- word-drop Recall được dùng để tuyên bố robustness dù query đã đổi nghĩa;
- thêm cross-attention tăng tham số nhưng không cải thiện compositional-only
  subset.

## 11. Kế hoạch thực hiện sau khi chốt

### Tuần 1 — Audit và protocol

- đóng băng baseline/checkpoint;
- xây corruption pipeline không sửa dữ liệu gốc;
- tạo metric robustness và calibration;
- chạy point/HUG evaluation-only.

### Tuần 2 — Error analysis

- robustness curves;
- uncertainty–rank correlation;
- mismatch detection;
- unimodal shortcut/compositional subsets;
- chốt failure mode ưu tiên.

### Tuần 3 — Calibration

- monotonic uncertainty loss;
- modality dropout/corruption training;
- clean-teacher consistency;
- ablation U0–U3.

### Tuần 4 — Adaptive routing

- static vs inverse-variance vs learned reliability gate;
- gate behavior visualization;
- ablation F0–F3.

### Tuần 5 — Generalization

- ba Fashion-IQ categories;
- CIRR nếu dữ liệu sẵn sàng;
- 3 seeds và bootstrap CI.

### Tuần 6 — Tổng hợp

- main tables;
- robustness curves;
- risk–coverage;
- qualitative failure cases;
- viết method/experiment/limitations.

## 12. Bảng kết quả nên chuẩn bị

1. Clean retrieval comparison.
2. Average corruption và worst-severity retrieval.
3. AUC của performance–severity curves.
4. Failure AUROC/AUPRC và AURC.
5. Mismatch detection bằng coordination uncertainty.
6. Static/inverse-variance/learned gate ablation.
7. Calibration-loss ablation.
8. Clean vs compositional-only subset.
9. Gate weight theo corruption severity.
10. Runtime, tham số và VRAM.

## 13. Đề xuất chốt

### Phương án khuyến nghị

Chọn một câu chuyện thống nhất gồm ba contribution:

1. **CIR robustness protocol** tách semantics-preserving,
   information-removing và conflict corruption.
2. **Corruption-calibrated heterogeneous uncertainty** để uncertainty dự đoán
   quality/failure thực sự.
3. **Reliability-gated adaptive composition** dùng calibrated uncertainty và
   cross-modal disagreement để routing, kèm true-composition audit.

Cross-attention chỉ là module tùy chọn để xử lý conflict sau khi error analysis
chứng minh cần thiết.

### Phương án tối giản, rủi ro thấp

Nếu thời gian/tài nguyên hạn chế:

1. Chỉ xây robustness/calibration protocol.
2. Thêm monotonic uncertainty calibration.
3. So sánh point, HUG và calibrated HUG.
4. Báo cáo clean Recall, corruption AUC, mismatch AUROC và AURC.

Phương án này vẫn tạo contribution rõ hơn generic adaptive fusion.

### Phương án tham vọng

Kết hợp calibrated routing với keep/change/conflict tokens và selective
retrieval. Chỉ nên chọn nếu có thêm thời gian, annotation/pseudo-label và GPU.

## 14. Tài liệu liên quan

- [HUG: Heterogeneous Uncertainty-Guided Composed Image Retrieval](https://arxiv.org/html/2601.11393)
  — baseline chính; đã có Q-Former cross-attention, heterogeneous uncertainty
  và inverse-variance dynamic weighting.
- [Probabilistic Embeddings for Cross-Modal Retrieval (PCME), CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Chun_Probabilistic_Embeddings_for_Cross-Modal_Retrieval_CVPR_2021_paper.html)
  — probabilistic image–text embeddings và uncertainty interpretation.
- [Improved Probabilistic Image-Text Representations (PCME++), ICLR 2024](https://openreview.net/forum?id=ft1mr3WlGM)
  — closed-form probabilistic distance, false-negative handling và noisy
  correspondence robustness.
- [Bayesian Triplet Loss: Uncertainty Quantification in Image Retrieval, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Warburg_Bayesian_Triplet_Loss_Uncertainty_Quantification_in_Image_Retrieval_ICCV_2021_paper.html)
  — retrieval uncertainty và probabilistic triplet formulation.
- [Embracing Unimodal Aleatoric Uncertainty for Robust Multimodal Fusion, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Gao_Embracing_Unimodal_Aleatoric_Uncertainty_for_Robust_Multimodal_Fusion_CVPR_2024_paper.html)
  — uncertainty-aware robust multimodal fusion.
- [Cross-modal Feature Alignment and Fusion for CIR, CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/CVFAD/html/Wan_Cross-modal_Feature_Alignment_and_Fusion_for_Composed_Image_Retrieval_CVPRW_2024_paper.html)
  — image/text-guided fusion và adaptive combiner; cho thấy generic C1/C2 đã có
  prior art gần.
- [CCIN: Compositional Conflict Identification and Neutralization, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Tian_CCIN_Compositional_Conflict_Identification_and_Neutralization_for_Composed_Image_Retrieval_CVPR_2025_paper.html)
  — conflict identification/neutralization trong CIR.
- [Self-guided Semantic Inspection for Zero-Shot CIR, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Self-guided_Semantic_Inspection_for_Zero-Shot_Composed_Image_Retrieval_CVPR_2026_paper.html)
  — difference-aware composition và phrase-guided masking.
- [Do CIR Benchmarks Require Multimodal Composition?, 2026](https://arxiv.org/abs/2605.14787)
  — unimodal shortcuts, ambiguity và compositional-only evaluation.
- [MedProbCLIP, WACVW 2026](https://openaccess.thecvf.com/content/WACV2026W/LFMBio/html/Elallaf_MedProbCLIP_Probabilistic_Adaptation_of_Vision-Language_Foundation_Model_for_Reliable_Radiograph-Report_WACVW_2026_paper.html)
  — calibration, risk–coverage, selective retrieval và corruption robustness
  cho probabilistic vision-language retrieval.
- [WISER: Adaptive Fusion for Training-Free ZS-CIR, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_WISER_Wider_Search_Deeper_Thinking_and_Adaptive_Fusion_for_Training-Free_CVPR_2026_paper.html)
  — intent/uncertainty-aware adaptive fusion, useful để định vị novelty.

## 15. Những câu cần chốt trước khi sửa code

1. Mục tiêu chính là tăng clean Recall, robustness, hay uncertainty calibration?
2. Contribution muốn nghiêng về method hay evaluation/analysis?
3. Có cần mở rộng CIRR ngay hay sau khi chứng minh trên Fashion-IQ?
4. Có chấp nhận freeze strong point backbone trong pha calibration/router?
5. Corruption nào phản ánh use case thực tế của nhóm?
6. Có tài nguyên để chạy ba categories × ba seeds không?
7. Có muốn theo selective/interactive retrieval như hướng thứ hai không?
8. Tiêu chí thành công tối thiểu trước khi đầu tư vào cross-attention mới là gì?

