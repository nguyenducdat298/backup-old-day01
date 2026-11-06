# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603544
- **Họ và tên:** Nguyễn Đức Đạt
- **SEED:** 4738

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | [x] |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | [x] |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | [x] |

## 2. Kết quả chính

Bảng so sánh 4 thế hệ AI trên cùng tập test (từ `lab1_report.json`):

| Mô hình | F1 (toàn bộ test) | F1 (tập SUB 40) | p95 (ms/tin) | USD / 1k tin |
| :--- | :---: | :---: | :---: | :---: |
| **1. BoW + LogReg (tự code)** | 0.868 | 0.824 | 0.08 ms | $0.00000 |
| **2. BoW + MLP (deep learning)** | 0.918 | 0.824 | 0.25 ms | $0.00000 |
| **3a. Encoder (MiniLM) + kNN** | 0.890 | 0.889 | 16.55 ms | $0.00000 |
| **3b. Encoder (MiniLM) + LogReg** | 0.922 | 0.947 | 7.10 ms | $0.00000 |
| **3c. Encoder (MiniLM) + MLP head**| 0.922 | 0.947 | 7.30 ms | $0.00000 |
| **4a. LLM nhỏ zero-shot (Gemini)** | 0.923 | 0.923 | 887.80 ms | $0.00425 |
| **4b. LLM lớn zero-shot (Gemini)** | 0.923 | 0.923 | 887.80 ms | $0.00425 |
| **4c. LLM nhỏ few-shot (Gemini)** | 0.950 | 0.950 | 1081.52 ms | $0.01225 |

## 3. Dự đoán và thực tế

3 dự đoán lệch nhiều nhất so với kết quả:

1. **`hard_eval` (Bộ 50 case khó):**
   - *Dự đoán:* BoW tụt F1 xuống dưới 0.70.
   - *Thực tế:* F1 của BoW tụt thảm hại xuống **0.540** (accuracy 54%).
   - *Nguyên nhân lệch:* Bộ test khó cố tình gài bẫy từ khóa và ngụy trang từ ngữ (dùng từ đồng nghĩa, cố tình gõ sai chính tả `fr33`, `w!n`). BoW chỉ nhìn token rời rạc nên bị đánh lừa gần như hoàn toàn.

2. **`dl_curve` (MLP so với LogReg):**
   - *Dự đoán:* F1 của MLP sẽ xấp xỉ tương đương Logistic Regression ở Phần 1 (~0.868).
   - *Thực tế:* F1 test của MLP đạt **0.918** (cao hơn đáng kể so với 0.868 của LogReg).
   - *Nguyên nhân lệch:* Lớp ẩn phi tuyến (ReLU) giúp mạng nơ-ron học được các mối tương tác phức tạp giữa các cặp từ xuất hiện đồng thời, khắc phục được giới hạn tuyến tính độc lập của LogReg trên BoW.

3. **`tradeoff` (Độ trễ p95 của Encoder + LogReg):**
   - *Dự đoán:* Độ trễ < 5 ms.
   - *Thực tế:* Độ trễ p95 thực tế là **7.10 ms**.
   - *Nguyên nhân lệch:* Chi phí tính toán forward pass của mô hình MiniLM 6 lớp trên CPU tốn thời gian hơn dự kiến một chút, tuy nhiên vẫn nằm sâu dưới ngưỡng SLA 50 ms.

## 4. Phân tích 50 case khó

- **Mô hình tụt nhiều nhất:** BoW + LogReg tụt từ F1 0.868 xuống còn 0.540 (tụt 0.328).
- **Nhóm case khó nhất với BoW:** Nhóm `ham_looks_spam` (tin nhắn bạn bè chứa từ khóa nhạy cảm) và nhóm `obfuscation` (cố tình viết sai chính tả né bộ lọc).
- **2 case cụ thể và cơ chế gây sai:**
  1. *Case 1 (ham_looks_spam):* `"Mum says the prize for the school raffle is a cake lol"` (nhãn thật: `ham`).
     - *Cơ chế sai:* Tin nhắn cá nhân giữa người thân nhưng chứa từ `"prize"`. BoW thấy từ `"prize"` có trọng số spam rất lớn nên cộng điểm mù quáng và dự đoán nhầm thành `spam`.
  2. *Case 2 (obfuscation):* `"Fr33 mobile ringtones waiting for you! Click to download"` (nhãn thật: `spam`).
     - *Cơ chế sai:* Kẻ lừa đảo thay chữ "e" bằng số "3" (`Fr33`). Token này không có trong từ điển huấn luyện của BoW, dẫn đến vector BoW toàn số 0 và mô hình đoán nhầm thành `ham`.
- **Nhóm khó nhất với LLM:** Nhóm `prompt_injection` (câu lệnh lừa đảo ghi đè hệ thống). Nhưng khi áp dụng prompt phòng thủ bọc thẻ `<sms>...</sms>`, mô hình LLM đã đạt độ chính xác **100% (50/50)**.

## 5. Thử thách cá nhân

- **Đề bài nhận được:** Thử thách #5 (`SEED % 6 = 4`):
  > *"Embedding · Viết 4 tin nhắn của riêng bạn (2 spam né từ khoá, 2 ham chứa 'free'/'win'). So dự đoán BoW vs Embedding+LogReg và giải thích từng tin."*
- **Cách làm:**
  Tạo 4 tin nhắn thực tế:
  1. *Spam 1 (Né từ khóa):* `"Your parcel delivery is temporarily held due to an incorrect house number. Please update your shipping address to receive the shipment: track-parcel-notice.com"` (Thật: spam).
  2. *Spam 2 (Né từ khóa):* `"Security notice: Unusual login attempt detected from an unrecognized device. Verify your credentials immediately to avoid suspension: account-verification-center.net"` (Thật: spam).
  3. *Ham 1 (Chứa "free"):* `"Hey, are you free this evening? We are planning to grab dinner and catch up after work."` (Thật: ham).
  4. *Ham 2 (Chứa "win"):* `"Did our school team win the football match yesterday? I fell asleep before the second half ended."` (Thật: ham).
- **Kết quả & Giải thích:**
  - *Spam 1:* BoW đoán đúng `spam` (nhờ các từ "delivery", "held"), nhưng Embedding đoán sai `ham` (do văn phong dịch vụ bưu chính khá giống tin thông báo bình thường).
  - *Spam 2:* Cả BoW và Embedding đều đoán sai thành `ham`. Lý do: Văn phong trang trọng, câu chữ hành chính né toàn bộ các từ khóa nhạy cảm.
  - *Ham 1 & Ham 2:* Cả hai mô hình đều đoán đúng `ham`. Mặc dù chứa "free" và "win", các từ ngữ ngữ cảnh đời thường xung quanh ("dinner", "catch up", "school team", "football match") đã kéo xác suất về đúng lớp ham.

## 6. Phương án cho sản phẩm thật

- **Yêu cầu hệ thống:** 10 triệu SMS/ngày (~116 tin/giây), yêu cầu SLA độ trễ p95 < 50 ms.
- **Lựa chọn:** Phương án **3b. Encoder (MiniLM) + LogReg**.
- **Dẫn chứng số liệu:**
  - Độ trễ p95 đạt **7.10 ms**, hoàn toàn thỏa mãn yêu cầu SLA (< 50 ms).
  - Thông lượng trên CPU đạt ~140 tin/giây trên 1 core (dễ dàng mở rộng đa tiến trình xử lý hàng ngàn tin/giây).
  - Chất lượng phân loại vượt trội: F1 đạt **0.922** trên toàn bộ test set và **0.947** trên tập SUB.
- **Đánh đổi:** Tốn tài nguyên CPU hơn so với BoW (cần forward pass qua 6 tầng Transformer), nhưng chi phí gần như bằng 0 so với LLM API (nếu dùng LLM xử lý 10 triệu SMS/ngày sẽ tốn từ $3,000 đến $30,000/tháng).
- **Thiết kế kiến trúc phân tầng (Cascade Architecture):**
  - *Tầng 1 (Lọc nhanh):* Dùng BoW + LogReg siêu nhẹ (< 0.1 ms). Các tin có độ tin cậy cao ($p < 0.1$ chắc chắn ham hoặc $p > 0.9$ chắc chắn spam) được trả kết quả ngay. Tầng này lọc được ~80% lượng tin nhắn với chi phí tính toán cực thấp.
  - *Tầng 2 (Lọc sâu):* 20% tin mập mờ còn lại ($0.1 \le p \le 0.9$) được đưa qua Encoder (MiniLM) + LogReg để phân tích ngữ nghĩa câu.
  - *Tầng 3 (Duyệt chuyên sâu):* Tỷ lệ cực nhỏ (< 0.01%) các tin nghi vấn lừa đảo tinh vi sẽ được chuyển đến LLM với prompt phân tích chuyên sâu hoặc nhân viên kiểm duyệt.

## 7. AI log và điều đã học

- **AI log:**
  - *Hỏi AI:* Giải thích bản chất hình học của gradient descent trong Logistic Regression khi cập nhật trọng số; gợi ý cách bọc thẻ cách ly dữ liệu `<sms>` để phòng thủ Prompt Injection.
  - *Nhận được:* Công thức toán học và nguyên lý suy diễn của LLM.
  - *Chỉnh sửa:* Tự tay code logic, tự đặt pacing 4.1s và cơ chế retry backoff cho Gemini API để tránh lỗi rate limit 15 RPM, kiểm tra kỹ lưỡng các hàm không dùng sklearn.
- **Điều hiểu ra:**
  - Hiểu sâu sắc sự tiến hóa qua 4 thế hệ AI: từ việc con người tự chọn đặc trưng (BoW) $\rightarrow$ máy tự học đặc trưng phi tuyến (MLP) $\rightarrow$ kế thừa biểu diễn ngữ nghĩa đã học sẵn (Embedding) $\rightarrow$ giải quyết bài toán qua chỉ dẫn ngôn ngữ tự nhiên không cần huấn luyện (LLM).
  - Nhận thức rõ ràng bài toán đánh đổi trong kỹ thuật AI thực tế: không phải mô hình lớn nhất lúc nào cũng là giải pháp tốt nhất; sự cân bằng giữa độ chính xác, độ trễ, chi phí hạ tầng và độ ổn định mới là yếu tố quyết định sự thành bại của một hệ thống production.
