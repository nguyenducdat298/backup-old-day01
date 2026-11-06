# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603544
- **Họ và tên:** Nguyễn Đức Đạt
- **SEED** (in ở ô đầu notebook): 4738

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Mô hình | Precision | Recall | F1 (test) | F1 (SUB 40) | Train (s) | Tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. BoW + LogReg (tự code) | 0.976 | 0.781 | 0.868 | 0.824 | 4.5 | 47,387 | 0.07 | 0.08 | 0.000 |
| 2. BoW + MLP (deep learning) | 0.989 | 0.857 | 0.918 | 0.824 | 2.4 | 67,905 | 0.17 | 0.20 | 0.000 |
| 3a. Encoder (MiniLM) + kNN | 0.894 | 0.886 | 0.890 | 0.889 | 18.3 | 48 | 129.60 | 142.20 | 0.000 |
| 3b. Encoder (MiniLM) + LogReg | 0.941 | 0.905 | 0.922 | 0.947 | 18.6 | 135 | 13.19 | 14.38 | 0.000 |
| 3c. Encoder (MiniLM) + MLP head | 0.949 | 0.895 | 0.922 | 0.947 | 19.2 | 141 | 12.27 | 13.83 | 0.000 |

## 3. Dự đoán và thực tế

1. **`ml_lr` (Learning Rate ở Logistic Regression):**
   - *Dự đoán:* lr=0.001 loss giảm rất chậm và chưa hội tụ; lr=0.5 loss giảm mượt và hội tụ tốt; lr=50 loss dao động dữ dội hoặc phân kỳ do overshooting.
   - *Thực tế:* Hoàn toàn chính xác. Tại lr=0.001 loss chỉ giảm từ 0.693 về 0.449 sau 300 epochs; lr=0.5 loss hội tụ về 0.089; lr=50 loss kẹt ở 0.505 và dao động mạnh.
   - *Nguyên nhân:* Bước nhảy gradient quá lớn làm tham số nhảy vọt qua điểm cực tiểu vào vùng bão hòa của hàm sigmoid.

2. **`dl_curve` (Đường cong mất mát của MLP):**
   - *Dự đoán:* Val loss giảm ở các epoch đầu rồi tăng dần lên do overfitting; F1 của MLP chỉ ngang ngửa hoặc nhỉnh hơn nhẹ so với Logistic Regression.
   - *Thực tế:* Train loss giảm đều từ 0.380 xuống 0.015, val loss đạt cực tiểu tại epoch 11 (0.063) sau đó tăng dần lên 0.111. F1 của MLP trên test đạt 0.918 (cao hơn BoW LogReg 0.868).
   - *Nguyên nhân:* Mạng 2 lớp có khả năng ghi nhớ chi tiết tập train sau nhiều epochs, gây overfitting nhẹ trên tập validation, nhưng lớp ẩn 64 nơ-ron giúp kết hợp phi tuyến các từ khóa tốt hơn một chút.

3. **`hard_eval` (Stress test 50 case khó):**
   - *Dự đoán:* F1 của mô hình BoW sụt giảm nặng nề nhất xuống dưới 0.70 do bị bẫy bởi từ khóa và lỗi chính tả cố tình; Embedding giữ vững phong độ tốt hơn.
   - *Thực tế:* F1 của BoW tụt xuống còn 0.667 trên 50 case khó, trong khi Embedding + LogReg đạt F1 0.762.
   - *Nguyên nhân:* BoW phụ thuộc tuyệt đối vào sự hiện diện của từ khóa độc lập, nên bị lừa bởi các tin nhắn ham chứa từ khóa giật gân hoặc tin spam cố tình viết chệch chính tả.

## 4. Phân tích 50 case khó

- **Mô hình tụt nhiều nhất:** `1. BoW + LogReg (tự code)` tụt F1 từ 0.868 xuống còn 0.667.
- **Nhóm case gây lỗi nhiều nhất:** Nhóm `ham_looks_spam` (tin nhắn bạn bè đời thường chứa các từ như "FREE", "WON", "CALL") và nhóm `obfuscation` (tin spam cố tình viết sai chính tả, ngụy trang ký tự như "w!n", "c-a-s-h").
- **Hai case cụ thể:**
  1. *Case #1 [ham_looks_spam]:* `"Congrats!!! You WON the bet lol, I owe you a beer. Call me later"` (Nhãn thật: `ham`).
     - *Cơ chế sai của BoW:* Chứa hàng loạt từ khóa spam có trọng số dương rất cao: "congrats", "won", "call". Mô hình BoW cộng dồn trọng số các từ này mà không hiểu từ "beer", "bet", "lol" biểu thị ngữ cảnh cá cược vui đùa của bạn bè, dẫn đến chặn nhầm tin nhắn.
  2. *Case #31 [obfuscation]:* `"Cl!ck here for fr33 entry into our prize draw: bit.ly/fake"` (Nhãn thật: `spam`).
     - *Cơ chế sai của BoW:* Các từ "cl!ck", "fr33" không khớp với từ "click", "free" trong từ điển vocab 2000 từ. Vector BoW nhận giá trị 0 cho toàn bộ các cột này, khiến mô hình không phát hiện được dấu hiệu spam và để lọt tin nhắn nguy hiểm.

## 5. Thử thách cá nhân

- **Đề bài theo SEED 4738 (Đề #5):**
  > *"Embedding · Viết 4 tin nhắn của riêng bạn (2 spam né từ khoá, 2 ham chứa 'free'/'win'). So dự đoán BoW vs Embedding+LogReg và giải thích từng tin."*
- **Cách làm:**
  Xây dựng 4 tin nhắn thử nghiệm:
  1. *Spam 1 (Né từ khóa):* `"Your parcel delivery is temporarily held due to an incorrect house number. Please update your shipping address to receive the shipment: track-parcel-notice.com"` (Nhãn: `spam`).
  2. *Spam 2 (Né từ khóa):* `"Security notice: Unusual login attempt detected from an unrecognized device. Verify your credentials immediately to avoid suspension: account-verification-center.net"` (Nhãn: `spam`).
  3. *Ham 1 (Chứa 'free'):* `"Hey, are you free this evening? We are planning to grab dinner and catch up after work."` (Nhãn: `ham`).
  4. *Ham 2 (Chứa 'win'):* `"Did our school team win the football match yesterday? I fell asleep before the second half ended."` (Nhãn: `ham`).
  Đưa qua 2 hàm dự đoán `predict_bow_lr` và `predict_proba((embed - mu)/sd, w_e, b_e) >= 0.5`.
- **Kết quả:**
  | Tin nhắn | Nhãn thật | BoW + LogReg | Embedding + LogReg |
  | :--- | :---: | :---: | :---: |
  | Spam 1 (Giao hàng giữ kiện) | spam | spam ✓ | ham ✗ |
  | Spam 2 (Cảnh báo đăng nhập) | spam | ham ✗ | ham ✗ |
  | Ham 1 (Rủ đi ăn, chứa 'free') | ham | ham ✓ | ham ✓ |
  | Ham 2 (Hỏi bóng đá, chứa 'win') | ham | ham ✓ | ham ✓ |
- **Giải thích:**
  - *Spam 1 & 2:* Sử dụng văn phong giả mạo thông báo hành chính hoặc cảnh báo bảo mật. Do không chứa các từ khóa kinh điển như "free", "win", "claim", cả hai mô hình nông đều có xu hướng đánh giá xác suất spam thấp nếu chưa từng học các mẫu phishing ngân hàng cụ thể.
  - *Ham 1 & 2:* Cả BoW và Embedding đều phân loại đúng `ham`. Mặc dù có chứa từ "free" và "win", nhưng các từ ngữ đời thường đi kèm ("dinner", "catch up", "football match", "fell asleep") đã cân bằng lại tổng trọng số (với BoW) và định hướng vector câu về cụm giao tiếp thường nhật (với Embedding).

## 6. Phương án cho sản phẩm thật

- **Yêu cầu kỹ thuật:** Hệ thống nhà mạng phục vụ 10 triệu SMS/ngày (~116 tin nhắn/giây trung bình, giờ cao điểm có thể lên tới 500-1000 tin/giây), yêu cầu độ trễ p95 < 50 ms/tin.
- **Lựa chọn phương án:** **3b. Encoder (MiniLM) + LogReg** kết hợp kiến trúc tầng (Cascade Architecture).
- **Dẫn chứng số liệu:**
  - *Độ trễ:* `p95` của Encoder + LogReg là **14.38 ms**, hoàn toàn thỏa mãn SLA (< 50 ms).
  - *Chất lượng:* F1 test đạt **0.922** (F1 trên tập SUB đạt **0.947**), vượt trội hơn nhiều so với BoW (0.868).
  - *Thông lượng:* Đạt ~135 tin/s trên 1 luồng CPU đơn; khi triển khai trên máy chủ production đa luồng hoặc GPU, thông lượng dễ dàng đạt hàng nghìn tin/giây.
- **Chấp nhận đánh đổi:**
  - So với BoW: Chấp nhận tốn thêm tài nguyên tính toán (RAM và CPU/GPU cho encoder), nhưng thu lại khả năng kháng spam ngụy trang và bảo vệ khách hàng vượt trội.
  - So với LLM: Loại bỏ phương án LLM thuần túy vì độ trễ quá cao (p95 từ hàng trăm đến hàng nghìn ms, vi phạm nghiêm trọng SLA) và chi phí ước tính lên tới hơn $3,000 – $30,000 USD/tháng cho 10 triệu tin/ngày.

## 7. AI log và điều đã học

- **AI Log:**
  - *Lần 1:* Sử dụng AI hỗ trợ phân tích nguyên nhân hàm mất mát không giảm ở `train_logreg`. Nhận được giải thích về dấu của gradient: $\text{error} = p - y$, khi $p > y$ mô hình đoán thừa nên gradient dương, cập nhật $w \leftarrow w - \text{lr} \times \text{grad}$ sẽ làm giảm trọng số. Đã tự cài đặt lại phép nhân ma trận `(X.T @ error) / n`.
  - *Lần 2:* Hỏi AI về cách xử lý `reasoning_effort="none"` và `max_completion_tokens` đối với các mô hình reasoning model mới của OpenAI trong hàm `call_openai`. Nhận được cú pháp truyền `provider_kwargs`, đã tự viết phần code `call_openai` và hàm chuẩn hóa `parse_label`.
- **Điều hiểu ra hôm nay:**
  1. Accuracy là một chỉ số cực kỳ nguy hiểm và dễ gây ngộ nhận khi dữ liệu mất cân bằng nghiêm trọng (imbalanced classes); một mô hình lười không lọc được spam nào vẫn đạt 87% accuracy. Trong các bài toán phân loại bất cân xứng, F1-score, Precision và Recall mới là thước đo thực chất.
  2. Deep Learning hay các mô hình nhiều tham số không phải lúc nào cũng là giải pháp tốt nhất. Với dữ liệu dạng bảng hoặc Bag-of-Words kích thước nhỏ, một mô hình tuyến tính đơn giản (Logistic Regression) vừa hội tụ nhanh, vừa tránh được overfitting và có độ trễ cực thấp (< 0.1 ms).
  3. Sức mạnh đột phá của Embedding nằm ở việc chuyển dịch từ việc so khớp từ vựng bề mặt (lexical matching) sang biểu diễn hình học trong không gian ngữ nghĩa (semantic representation), giúp giải quyết triệt để bài toán đồng nghĩa và tin nhắn ngụy trang.
