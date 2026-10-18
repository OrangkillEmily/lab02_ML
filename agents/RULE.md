# QUY TẮC LẬP TRÌNH & TƯƠNG TÁC (LAB 02 - PREDICTING PRODUCT SALES)
# Quy tắc 1: Quyền quyết định và Tương tác AI - Con người

*Vai trò của AI:* Đóng vai trò trợ lý chuyên môn. Chỉ được đưa ra các gợi ý, đề xuất giải pháp và viết mã dự thảo ở mức độ vừa đủ, phù hợp với phạm vi của một bài Lab trên trường.

*Vai trò của Con người:* Là người kiểm duyệt và ra quyết định cuối cùng.

**Nguyên tắc "Hiểu mới dùng":** Tuyệt đối KHÔNG copy/paste hoặc chạy bất kỳ đoạn code nào nếu chưa thực sự hiểu rõ logic bên trong. Nếu AI viết code khó hiểu, phải yêu cầu AI giải thích lại bằng ngôn ngữ bình dân trước khi sử dụng.

# Quy tắc 2: Vibe Coding (Dos & Don'ts)

*Được làm (Dos):* Ưu tiên tính đơn giản, dễ đọc, đi thẳng vào trọng tâm giải quyết bài toán dự đoán doanh số. Code thuật toán từ đầu (from scratch) bằng Toán học và Ma trận phải tường minh, dễ theo dõi.

*Không được làm (Don'ts):* AI không được tự ý "over-engineer" (phức tạp hóa vấn đề), không dùng các design pattern quá trừu tượng. Không tự ý bịa thêm tính năng nằm ngoài phạm vi phân tích bán lẻ trực tuyến.

# Quy tắc 3: Bắt buộc triển khai CẢ 4 MÔ HÌNH

Dự án này sẽ không chọn 1 trong 4, mà phải xây dựng cả 4 thuật toán từ con số không (from scratch): Linear Regression, Decision Tree, Random Forest, và XGBoost để dự đoán doanh số (Regression).

File train.py phải gọi và huấn luyện cả 4 mô hình. File evaluate.py phải xuất ra bảng so sánh hiệu năng (MSE, MAE, RMSE, R^2) của cả 4 mô hình này.

# Quy tắc 4: Giới hạn thư viện (TUYỆT ĐỐI TUÂN THỦ)

Chỉ được phép sử dụng các thư viện đã thống nhất trong requirements.txt (chủ yếu là pandas, numpy, matplotlib, seaborn, streamlit, PyYAML, jupyter, ipykernel)

Nghiêm cấm sử dụng các thư viện Machine Learning có sẵn như scikit-learn, xgboost, hoặc statsmodels cho việc phân chia dữ liệu, tính toán sai số, hay xây dựng thuật toán. Mọi logic cốt lõi phải tự viết bằng numpy và pandas.

# Quy tắc 5: Phải có giải thích code (Docstring)

Mọi hàm (function) và lớp (class) được viết trong thư mục src/ đều phải có phần ghi chú rõ ràng bên dưới tên, bao gồm 3 ý:

Hàm/Lớp này dùng để làm gì?

Đầu vào (Parameters) cần truyền vào cái gì?

Đầu ra (Return) trả về kết quả gì?

# Quy tắc 6: Tôn trọng kiến trúc thư mục

Tuyệt đối KHÔNG tự ý đổi tên file, xóa thư mục hoặc gộp chung các file lại với nhau.

File nào việc nấy (Ví dụ: src/models/*.py chỉ chứa logic toán học của thuật toán, src/train.py chỉ dùng để huấn luyện, cấm viết code vẽ biểu đồ vào đây).

# Quy tắc 7: Chống rò rỉ dữ liệu (Data Leakage)

Tập dữ liệu dùng để kiểm tra (Test set) phải được chia tỷ lệ 80-20 (tự code hàm chia) và tách ra cất riêng ngay từ khâu Tiền xử lý (src/preprocess.py).

Quá trình huấn luyện tuyệt đối không được "nhìn trộm" hay dính dáng đến tập dữ liệu kiểm tra này. Cố định random seed (ví dụ: np.random.seed(42)) để đảm bảo kết quả có thể tái lập.