# Lab 21 — Evaluation Report

**Học viên**: <Họ tên> — <MSSV>
**Ngày nộp**: 2026-06-25
**Submission option**: A (lightweight)

---

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-7B-bnb-4bit` (QLoRA 4-bit NF4)
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 500 samples (450 train + 50 eval)
- **max_seq_length**: 512 (p95 = 372 tokens, rounded up power-of-2)
- **GPU**: NVIDIA A100-SXM4-80GB, 80 GB VRAM, Compute capability 8.0, BF16 native
- **Training cost**: ~$0.11 (~5.87 phút @ $1.10/hr)
- **Framework**: Unsloth + TRL SFTTrainer + PEFT (LoRA)
- **Hyperparameters**: 3 epochs, cosine LR = 2e-4, warmup_steps = 17, effective batch = 8 (4 × 2), adamw_8bit, gradient checkpointing (unsloth)

---

## 2. Rank Experiment Results

| Rank | Alpha | Trainable Params | % Total | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-------|-----------------|---------|------------|-----------|-----------|------------|
| Base | — | 0 | 0% | — | — | 15.588 | 5,886,634 |
| 8 | 16 | 2,523,136 | 0.033% | 1.91 min | 21.20 GB | 1.198 | 3.314 |
| 16 | 32 | 5,046,272 | 0.066% | 2.04 min | 20.13 GB | 1.199 | 3.317 |
| 64 | 128 | 20,185,088 | 0.265% | 1.92 min | 22.51 GB | 1.200 | 3.321 |

### Nhận xét nhanh

- **Fine-tuning cải thiện perplexity 1,775,000×** so với base model (từ 5.88M xuống 3.32)
- **Cả 3 ranks cho kết quả gần như identique** — perplexity chênh lệch < 0.3%
- Training time không tỉ lệ với rank (r=8 và r=64 gần bằng nhau)
- Peak VRAM tăng nhẹ khi tăng rank (21.2 → 22.5 GB, chênh 1.3 GB)

---

## 3. Loss Curve Analysis

### Quan sát từ loss curve r=16

- **Training loss giảm đều** từ ~1.5 xuống ~0.4 qua 3 epochs — model học tốt
- **Eval loss ổn định** quanh 1.20 qua các eval steps — không có dấu hiệu overfitting rõ rệt
- Khoảng cách train-eval loss (generalization gap) ~0.8 — chấp nhận được với dataset 450 samples

### Phân tích overfitting

- Không phát hiện overfitting nghiêm trọng. Lý do:
  - Dataset nhỏ (450 samples) + 3 epochs → model chưa memorize đủ
  - LoRA chỉ train 0.03–0.27% params → capacity constraint tự nhiên
  - Learning rate cosine decay + warmup giúp ổn định training

### Loss curves r=8 và r=64

- Cả 3 ranks có loss curve hình dạng tương tự — cùng dataset, cùng hyperparams
- r=64 có trainable params gấp 8× r=8 nhưng loss không cải thiện thêm → **diminishing returns rõ ràng**

---

## 4. Qualitative Comparison (10 examples)

Đánh giá side-by-side base model vs fine-tuned (r=16) trên 10 test prompts đa dạng chủ đề. Nhận xét: có cả **case win** lẫn **case loss** — không cherry-pick.

### Example 1 — Machine Learning explanation ✅ Fine-tuned better

**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.

| Model | Response |
|---|---|
| **Base** | "Máy học là một lĩnh vực của trí tuệ nhân tạo tập trung vào việc phát triển các thuật toán và mô hình toán học có thể học và cải thiện hiệu suất dựa trên dữ liệu. Mô hình máy học được cung cấp với dữ liệu và mục tiêu, và thông qua việc tìm hiểu và phát hiện các mẫu và quy luật trong dữ liệu, mô hình có thể dự đoán kết quả tương tự cho dữ liệu mới. Ví dụ: nếu bạn đang xây dựng một mô hình máy học để..." |
| **Fine-tuned** | "Machine learning là một lĩnh vực của khoa học máy tính và trí tuệ nhân tạo mà máy học được các quy tắc và quy luật từ dữ liệu. Điều này cho phép máy tính tự động cải thiện hiệu suất trên một nhiệm vụ cụ thể thông qua các mẫu và dữ liệu được cung cấp cho chúng. Trong khi chương trình truyền thống được thiết kế để hoàn thành một nhiệm vụ cụ thể bằng cách sử dụng thuật toán cố định, máy học sử dụng d..." |

**Nhận xét**: Fine-tuned model giải thích rõ hơn bằng cách **so sánh với chương trình truyền thống** ("trong khi chương trình truyền thống được thiết kế để hoàn thành một nhiệm vụ cụ thể bằng cách sử dụng thuật toán cố định..."). Câu trả lời có cấu trúc tốt hơn, dễ hiểu cho người mới bắt đầu.

---

### Example 2 — Fibonacci code ✅ Fine-tuned better

**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.

| Model | Response |
|---|---|
| **Base** | "Dưới đây là một ví dụ về cách sử dụng các hàm để tính số Fibonacci thứ n trong Python: ```python def fibonacci(n): if n == 0: return 0 elif n == 1: return 1 else: return fibonacci(n-1) + fibonacci(n-2) n = 10 print(fibonacci(n)) ``` ...nếu n = 10, thì hàm trả về giá trị 55..." |
| **Fine-tuned** | "Đây là một đoạn code Python tính số Fibonacci thứ n: ```python def fibonacci(n): if n <= 0: return 0 elif n == 1: return 1 else: return fibonacci(n-1) + fibonacci(n-2) result = fibonacci(10) print(result) ``` ...Lưu ý rằng thuật toán..." |

**Nhận xét**: Fine-tuned model **mở rộng edge case** (`n <= 0` thay vì `n == 0`), tốt hơn cho production code. Ngoài ra, fine-tuned dùng `result = fibonacci(10)` rõ ràng hơn so với base dùng comment `# Thay giá trị n cần tính vào đây`.

---

### Example 3 — UI/UX principles ⚖️ Tie

**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.

| Model | Response |
|---|---|
| **Base** | "1. Người dùng là trung tâm: Một thiết kế UI/UX hiệu quả phải tập trung vào trải nghiệm người dùng và đảm bảo dễ sử dụng, thân thiện và dễ tiếp cận. 2. Dễ sử dụng và thân thiện: Một thiết kế UI/UX tốt phải dễ sử dụng, thân thiện và dễ tiếp cận, cho phép người dùng nhanh chóng tìm thấy và tương tác với các chức năng và nội dung của sản phẩm. 3. Đa dạng và khả năng tiếp cận: Thiết kế UI/UX phải được..." |
| **Fine-tuned** | "1. Người dùng trung tâm: Thiết kế UI/UX cần tập trung vào người dùng để đảm bảo trải nghiệm người dùng mượt mà và dễ sử dụng. 2. Phù hợp: Thiết kế UI/UX phải phù hợp với đối tượng mục tiêu của bạn, điều này có nghĩa là nó phải dễ hiểu, dễ sử dụng và có thể tương tác được. 3. Đa dạng: Thiết kế UI/UX cần đảm bảo trải nghiệm người dùng của tất cả mọi người, không chỉ những người có điều kiện đặc biệt..." |

**Nhận xét**: Cả hai trả lời tương tự về cấu trúc. Base model dài hơn và lặp ý ("dễ sử dụng, thân thiện và dễ tiếp cận" xuất hiện ở cả nguyên tắc 1 và 2). Fine-tuned model ngắn gọn hơn, mỗi nguyên tắc có ý riêng — nhưng thiếu chiều sâu so với base.

---

### Example 4 — LoRA vs QLoRA ✅ Fine-tuned better

**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.

| Model | Response |
|---|---|
| **Base** | "LoRA và QLoRA là những kỹ thuật được sử dụng trong lĩnh vực học máy để giảm kích thước mô hình mà không làm mất hiệu suất. LoRA (Linearly Trained Low-Rank Adaptation) là một kỹ thuật huấn luyện mô hình bằng cách sử dụng các ma trận thấp và được chứng minh có thể cung cấp hiệu suất tốt với kích thước mô hình nhỏ hơn. LoRA huấn luyện các ma trận có kích thước thấp để sửa đổi các trọng số của mô hình..." |
| **Fine-tuned** | "LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp khác nhau để huấn luyện các mô hình ngôn ngữ chuyển đổi (transformer) có kích thước nhỏ hơn. LoRA là một phương pháp cho phép các mô hình chuyển đổi được huấn luyện để thích nghi với dữ liệu mới bằng cách chỉ cập nhật các phần nhỏ của mô hình. Cụ thể, nó thay đổi các ma trận tuyến tính trong mô hình để tạo ra các bản sao có kí..." |

**Nhận xét**: Fine-tuned model **chính xác hơn rõ rệt**. Base model dùng sai thuật ngữ "Linearly Trained Low-Rank Adaptation" (LoRA là "Low-Rank Adaptation", không phải "Linearly Trained"). Fine-tuned model giải thích đúng cơ chế "chỉ cập nhật các phần nhỏ của mô hình" và đề cập đến transformer architecture.

---

### Example 5 — Prompt Engineering vs RAG vs Fine-tuning ✅ Fine-tuned better

**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.

| Model | Response |
|---|---|
| **Base** | "Prompt engineering là quá trình thiết kế và tối ưu hóa các câu lệnh đầu vào để hiệu quả tương tác với các hệ thống máy học có học sâu (ML) hoặc hệ thống AI. Điều này bao gồm việc tạo ra câu lệnh rõ ràng, cụ thể, dễ hiểu và có thể được hiểu bởi hệ thống học sâu. Prompt engineering được sử dụng để cải thiện độ chính xác, độ tin cậy và hiệu quả của hệ thống học sâu, đặc biệt là khi chúng được sử dụng..." |
| **Fine-tuned** | "Prompt engineering là kỹ thuật thiết kế và tạo prompt để tương tác với AI và có được kết quả mong muốn. Điều này thường liên quan đến việc sử dụng ngôn ngữ, cấu trúc và ngữ cảnh thích hợp để đưa ra prompt rõ ràng và dễ hiểu. RAG (Retrieval-Augmented Generation) là một phương pháp học máy cho phép hệ thống AI sử dụng kết quả thu được từ việc truy vấn cơ sở dữ liệu lớn để hỗ trợ quá trình tạo prompt..." |

**Nhận xét**: Fine-tuned model **bắt đầu phân biệt cả 3 khái niệm** ngay trong câu trả lời (đã提到 RAG). Base model chỉ giải thích prompt engineering mà chưa提到 RAG hay fine-tuning. Fine-tuned model cũng dùng ngôn ngữ tự nhiên hơn ("tương tác với AI") thay vì base dùng thuật ngữ dài dòng ("hệ thống máy học có học sâu").

---

### Example 6 — RAG vs Fine-tuning ⚖️ Tie

**Prompt**: Khi nào nên dùng RAG thay vì fine-tuning?

| Model | Response |
|---|---|
| **Base** | "RAG và fine-tuning là hai kỹ thuật khác nhau được sử dụng để huấn luyện mô hình ngôn ngữ máy tính. Tuy nhiên, chúng có mục đích và quy trình huấn luyện khác nhau. RAG (Retrieval-Augmented Generation) là một phương pháp sử dụng các mô hình ngôn ngữ máy tính để tạo ra văn bản, đồng thời sử dụng kỹ thuật tìm kiếm để thu thập thông tin bổ sung từ nguồn tin cậy. Khi bạn cần mô hình tạo ra văn bản có li..." |
| **Fine-tuned** | "RAG (Retrieval-Augmented Generation) và fine-tuning là hai kỹ thuật khác nhau được sử dụng trong AI để cải thiện hiệu suất của các mô hình ngôn ngữ. RAG là một phương pháp cho phép mô hình có thể truy cập vào các nguồn thông tin bên ngoài để tìm kiếm và sử dụng thông tin liên quan trong quá trình tạo văn bản. Nó phù hợp với các trường hợp sử dụng khi mô hình cần truy cập vào thông tin cụ thể hoặc..." |

**Nhận xét**: Cả hai đều giải thích đúng. Fine-tuned model nói rõ hơn về use case ("phù hợp với các trường hợp sử dụng khi mô hình cần truy cập vào thông tin cụ thể"). Base model cũng tốt nhưng dùng từ dài dòng hơn.

---

### Example 7 — Flash Attention ⚠️ Base better

**Prompt**: Giải thích cách hoạt động của Flash Attention.

| Model | Response |
|---|---|
| **Base** | "Flash Attention là một kỹ thuật học máy được sử dụng để cải thiện hiệu suất của các mô hình học máy học, đặc biệt là các mô hình máy học ngôn ngữ. Nó được phát triển để giải quyết một số vấn đề của kỹ thuật chú ý thông thường, trong đó tính toán chú ý có thể bị chậm và tốn kém. Flash Attention hoạt động bằng cách sử dụng một kỹ thuật được gọi là 'phép tính phép chú ý chớp nhoáng' để giảm thời gian..." |
| **Fine-tuned** | "Flash Attention là một kỹ thuật trong lĩnh vực học máy và trí tuệ nhân tạo được phát triển để giảm thời gian tính toán của quá trình attentiveness trong các mô hình ngôn ngữ, đặc biệt là trong các mô hình ngôn ngữ có kích thước lớn như Transformers. Nó được phát triển như một cách để giảm thiểu thời gian tính toán của quá trình attentiveness trong mô hình, mà thường có thể chiếm phần lớn thời gian..." |

**Nhận xét**: **Base model tốt hơn** ở prompt này. Base dùng "chú ý" (attention) đúng thuật ngữ, trong khi fine-tuned dùng "attentiveness" — không phải thuật ngữ chuẩn trong NLP. Base cũng giải thích rõ hơn về cơ chế ("phép tính phép chú ý chớp nhoáng"). Đây là **case loss** của fine-tuned model.

---

### Example 8 — ML Interview questions ✅ Base better

**Prompt**: List 3 câu hỏi phỏng vấn cho ML Engineer junior.

| Model | Response |
|---|---|
| **Base** | "1. Bạn có thể giải thích về kiến thức của bạn trong các thuật toán học máy như Naive Bayes, Random Forest và Support Vector Machines không? 2. Bạn có thể giải thích về quá trình huấn luyện và đánh giá mô hình học máy không? 3. Bạn có thể giải thích về các kỹ thuật thu thập dữ liệu, tiền xử lý dữ liệu và thu thập dữ liệu không?" |
| **Fine-tuned** | "1. Bạn có thể giải thích thuật toán hồi quy tuyến tính không? 2. Bạn có thể giải thích phân loại và cách phân loại được sử dụng trong AI không? 3. Bạn có thể giải thích các kỹ thuật chia tập dữ liệu được sử dụng để huấn luyện và kiểm tra mô hình không?" |

**Nhận xét**: **Base model tốt hơn** — câu hỏi phỏng vấn chi tiết hơn (đề cập cụ thể Naive Bayes, Random Forest, SVM). Fine-tuned model câu hỏi quá chung chung ("hồi quy tuyến tính", "phân loại"). Cả hai đều lặp cấu trúc "Bạn có thể giải thích..." 3 lần — nhưng base ít nhất nội dung phong phú hơn.

---

### Example 9 — Transformers vs RNN ❌ Fine-tuned failed

**Prompt**: Cho biết ưu điểm của Transformer so với RNN.

| Model | Response |
|---|---|
| **Base** | "Transformers và RNNs (mạng lưới kinh nghiệm) là hai loại mô hình học sâu được sử dụng phổ biến cho các nhiệm vụ xử lý ngôn ngữ tự nhiên (NLP). Mặc dù cả hai đều có thể học các mô hình dự đoán dựa trên ngữ cảnh, nhưng Transformers và RNNs có những điểm khác biệt quan trọng về kiến trúc và cách xử lý dữ liệu. Transformers được cho là có ưu điểm hơn so với RNNs trong các nhiệm vụ xử lý ngôn ngữ tự nh..." |
| **Fine-tuned** | "Transformers và RNN (Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng Mạng..." |

**Nhận xét**: ⚠️ **Fine-tuned model thất bại hoàn toàn** — "Mạng" lặp ~60 lần. Đây là hiện tượng **degeneration** (token repetition loop). Nguyên nhân có thể: (1) training data có samples chứa pattern lặp, (2) temperature=0.7 + top_p=0.9 không đủ để ngăn repetition, hoặc (3) model chưa học tốt prompt "so sánh Transformer vs RNN" trong dataset. **Base model trả lời tốt hơn rõ rệt.**

---

### Example 10 — LLM evaluation ⚖️ Tie

**Prompt**: Cách evaluate performance của một LLM fine-tuned model?

| Model | Response |
|---|---|
| **Base** | "Đánh giá hiệu suất của một LLM fine-tuned model đòi hỏi phải thực hiện một số bước sau: 1. Định nghĩa các chỉ số đánh giá: Trước tiên, bạn cần xác định các chỉ số đánh giá mà bạn muốn sử dụng để đánh giá hiệu suất của mô hình. Một số chỉ số thường được sử dụng là độ chính xác, độ chính xác, độ chính xác và độ trễ, độ chính xác trung bình. 2. Phân loại tập dữ liệu: Tiếp theo, bạn cần phân loại tập..." |
| **Fine-tuned** | "Mô hình LLM được tinh chỉnh có thể được đánh giá bằng nhiều phương pháp khác nhau. Một số phương pháp thường được sử dụng bao gồm: 1. Độ chính xác: Độ chính xác được tính bằng số lượng câu trả lời đúng chia cho tổng số câu trả lời. Đây là một chỉ số đơn giản để đánh giá hiệu suất của mô hình. 2. Độ chính xác trung bình: Độ chính xác trung bình là độ chính xác trung bình của mô hình trên các lớp. Đ..." |

**Nhận xét**: Cả hai đều liệt kê các metric đánh giá. **Base model lặp từ** ("độ chính xác" xuất hiện 4 lần trong 1 câu — "độ chính xác, độ chính xác, độ chính xác và độ trễ"). Fine-tuned model cũng lặp nhưng ít hơn. Cả hai đều chưa提到 perplexity, BLEU, ROUGE — các metric quan trọng cho LLM evaluation. **Tie — cả hai đều ở mức trung bình.**

---

### Tổng hợp qualitative (10 prompts)

| # | Prompt | Winner | Ghi chú |
|---|--------|--------|---------|
| 1 | ML explanation | ✅ Fine-tuned | So sánh với traditional programming |
| 2 | Fibonacci code | ✅ Fine-tuned | Edge case handling tốt hơn |
| 3 | UI/UX principles | ⚖️ Tie | Cả hai có điểm mạnh riêng |
| 4 | LoRA vs QLoRA | ✅ Fine-tuned | Chính xác thuật ngữ hơn |
| 5 | PE vs RAG vs FT | ✅ Fine-tuned | Phân biệt được cả 3 |
| 6 | RAG vs FT | ⚖️ Tie | Cả hai giải thích đúng |
| 7 | Flash Attention | ⚠️ Base | Fine-tuned dùng sai "attentiveness" |
| 8 | ML interview Q | ⚠️ Base | Fine-tuned câu hỏi quá chung |
| 9 | Transformer vs RNN | ❌ Base | Fine-tuned bị lặp từ "Mạng" ~60 lần |
| 10 | LLM evaluation | ⚖️ Tie | Cả hai lặp từ, thiếu metric quan trọng |

**Tổng**: Fine-tuned wins 4/10, Base wins 3/10, Tie 3/10.

Fine-tuned model tốt hơn trên **domain knowledge** (LoRA, QLoRA, RAG) nhưng yếu hơn trên **general knowledge** (Flash Attention, ML interview, Transformer vs RNN). Đặc biệt, prompt 9 cho thấy fine-tuned model có khả năng **degeneration** — cần post-processing hoặc repetition penalty khi deploy.

---

## 5. Conclusion về Rank Trade-off

### Rank nào cho ROI tốt nhất?

**Rank 8 cho ROI tốt nhất trên dataset này.** Cả 3 ranks (8, 16, 64) đều cho perplexity gần như identique (3.314, 3.317, 3.321 — chênh lệch < 0.3%). Tuy nhiên, rank 8 chỉ dùng 2.5M trainable params (0.033% tổng params), bằng **một nửa** so với rank 16 và **một phần tám** so với rank 64. Điều này có nghĩa:

- Adapter file nhỏ hơn → lưu trữ và deploy nhanh hơn
- Inference overhead thấp hơn khi merge adapter vào base model
- Training nhanh nhất (1.91 phút vs 2.04 phút cho rank 16)

### Diminishing returns ở đâu?

Trên dataset 450 samples này, **diminishing returns xuất hiện ngay từ rank 8**. Tăng rank từ 8 → 16 → 64 không cải thiện perplexity mà chỉ tăng VRAM (+1.3 GB từ r=8 đến r=64) và adapter size (gấp 8×). Nguyên nhân chính: dataset quá nhỏ (450 samples) nên capacity thêm từ rank cao không được tận dụng — model đã học hết pattern trong data chỉ với 2.5M params.

### Production recommendation

Nếu deploy production với dataset nhỏ (< 1000 samples) tương tự lab này:
- **Chọn rank 8** — đủ capacity, adapter nhẹ, perplexity tối ưu
- Nếu dataset lớn (> 10K samples) và task phức tạp hơn, mới cân nhắc rank 16-32
- Rank 64 chỉ nên dùng khi dataset > 50K samples hoặc task yêu cầu học complex patterns (structured output, multi-step reasoning)

Với dataset lớn hơn, rank cao hơn sẽ có đất thể hiện — nhưng trên 450 samples, rank 8 đã là "sweet spot".

---

## 6. What I Learned

- **LoRA rank không phải "càng cao càng tốt"**: Kết quả thực tế cho thấy rank 8 cho kết quả gần identique rank 64 trên dataset nhỏ (perplexity 3.314 vs 3.321 — chênh lệch < 0.3%). Đây là minh chứng cụ thể cho nguyên lý "low-rank" trong LoRA — phần lớn information trong fine-tuning nằm trong subspace rất thấp chiều. Với 450 samples, 2.5M trainable params (rank 8) đã đủ capacity.

- **Base perplexity 5.88M → fine-tuned 3.32 — cải thiện 1.77 triệu lần chỉ với 450 samples**: Qwen2.5-7B chưa từng见过 Vietnamese Alpaca format trước đó. Chỉ 450 samples + 3 epochs đã teach model format mới — minh chứng cho sức mạnh của LoRA trong few-shot domain adaptation. Tuy nhiên, perplexity thấp không guarantee chất lượng output tốt trên mọi prompt.

- **Fine-tuned model không phải lúc nào cũng tốt hơn base**: Qua 10 qualitative prompts, fine-tuned wins 4, base wins 3, tie 3. Đặc biệt, prompt 9 (Transformer vs RNN) cho thấy fine-tuned model bị **degeneration** — "Mạng" lặp ~60 lần. Đây là dấu hiệu model overfit trên một số samples có pattern lặp trong dataset. Cần repetition penalty (`repetition_penalty=1.2`) hoặc post-processing khi deploy production.
