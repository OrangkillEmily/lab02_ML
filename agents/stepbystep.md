# HƯỚNG DẪN TỪNG BƯỚC & BÀN GIAO DỰ ÁN (STEP-BY-STEP & HANDOFF - LAB 02)
## 1. Rules From Project (Quy tắc làm việc & AI)
*AI Role:* AI đóng vai trò trợ lý chuyên môn, con người đưa ra quyết định cuối cùng. Không tự ý đoán mò, phải hỏi lại nếu thiếu thông tin.

*Triết lý Code (Vibe Coding):* Code đơn giản, tường minh, dễ hiểu, không "over-engineer". Tuyệt đối không dùng code nếu chưa hiểu rõ logic (nhất là code thuật toán từ con số không).

*Yêu cầu cốt lõi:* Bắt buộc phải tự code (from scratch) và huấn luyện ĐỦ 4 mô hình Hồi quy: Linear Regression, Decision Tree, Random Forest, XGBoost.

*Nguyên tắc Dữ liệu:* Chống rò rỉ dữ liệu (Data Leakage) tuyệt đối. Tập Test (20%) phải được cắt và cất đi riêng biệt ngay từ bước Tiền xử lý.

## 2. Chosen Direction (Định hướng & Kiến trúc)
*Nguồn dữ liệu:* Đọc từ data/raw/online_retail.csv.

*Quy trình ELT/ML:* Đọc raw CSV -> Tiền xử lý (Làm sạch + Feature Engineering) -> Chia Train/Test (80-20) -> Lưu xuống đĩa -> Xây dựng Model (From scratch) -> Huấn luyện -> Đánh giá -> Triển khai (Deployment).

*Tiền xử lý:* Xử lý giá trị thiếu, tính doanh thu (Quantity * UnitPrice), trích xuất đặc trưng thời gian (Ngày, Tháng, Năm) từ InvoiceDate.

*Log & Trace:* Thông báo chạy pipeline phải được in rõ ràng ra terminal. Lịch sử thử nghiệm (Experimentation) lưu vào docs/experiment_logs.csv.

*Môi trường:* Quản lý bằng venv và requirements.txt.

## 3. Cấu trúc và Vai trò File (Current Files And Roles)
*agents/:* Chứa định hướng dự án (description, LAB, RULES, stepbystep).

*data/raw/:* Chứa file gốc online_retail.csv.

*data/processed/:* Chứa các ma trận dữ liệu đã làm sạch (X_train, X_test, y_train, y_test).

*docs/Lab02_descript.md:* File báo cáo, tài liệu phân tích nghiệp vụ của dự án.

*docs/experiment_logs.csv:* Nơi ghi nhận lại lịch sử thay đổi tham số (hyperparameters) để tối ưu mô hình.

*models/:* Chứa các trọng số mô hình đã huấn luyện hoàn chỉnh (định dạng .pkl).

*notebooks/lab02_exploration.ipynb:* Khám phá dữ liệu EDA, vẽ biểu đồ.

*notebooks/lab02_timeseries_eda.ipynb:* Phân tích chuỗi thời gian (Time Series EDA), xem xét xu hướng và tính mùa vụ.

*src/__init__.py:* Khai báo src là một package.

*src/eda.py:* Trực quan hóa dữ liệu thô.

*src/preprocess.py:* Làm sạch, thêm đặc trưng (Feature Engineering), cắt Train/Test và lưu vào data/processed/.

*src/models/:* Thư mục chứa code xây dựng thuật toán từ đầu (Linear Regression, Decision Tree, Random Forest, XGBoost).

*src/train.py:* Tải X_train, y_train, gọi mô hình từ src/models/ để huấn luyện, lưu vào models/.

*src/evaluate.py:* Tải mô hình (.pkl) và tập Test để đánh giá, tính các chỉ số Regression (MSE, MAE, RMSE, R^2).

*src/predict.py:* Kịch bản nạp mô hình để dự đoán doanh số cho dữ liệu mới.

*src/pipeline.py:* Kịch bản điều phối, gọi chạy tuần tự preprocess -> train -> evaluate.

*src/utils.py:* Chứa các hàm tiện ích dùng chung (chia tập train/test tự viết, tính sai số, nạp/lưu file).

*app.py:* Code giao diện Streamlit phục vụ triển khai (Model Deployment).

*config.yaml:* Quản lý đường dẫn file và siêu tham số cho các mô hình.

## 4. Install & Setup (Cài đặt môi trường & Chuẩn bị)
Để bắt đầu làm việc, mỗi thành viên bắt buộc thực hiện 3 bước sau để tránh lỗi:

# Bước 4.1. Kích hoạt môi trường ảo (Virtual Environment)

Tuyệt đối không cài trực tiếp thư viện vào Python của máy tính gốc.

Mở Terminal (Git Bash trên VS Code) và gõ:

Tạo môi trường ảo (Làm 1 lần): python -m venv venv

Bật môi trường (Bật mỗi khi mở VS Code): source venv/Scripts/activate

(Dấu hiệu thành công: Đầu dòng lệnh hiện chữ (venv)).

# Bước 4.2. Cài đặt kho thư viện

Gõ lệnh sau để tải các thư viện theo đúng luật (không có scikit-learn):
pip install -r requirements.txt

# Bước 4.3. Xác nhận Dữ liệu thô (Raw Data)

Kiểm tra file online_retail.csv đã có trong data/raw/ chưa. Nếu có, sẵn sàng code!

# 5. Workflow Timeline (Tiến độ thực thi chi tiết 5 Giai đoạn)
## Giai đoạn 1: Khởi tạo không gian làm việc (✅ Đã xong)
Đã dựng xong khung thư mục tĩnh, thiết lập app.py, config.yaml.

Cấu hình .gitignore chuẩn chỉnh.

Chốt luật làm việc tại agents/.

## Giai đoạn 2: Khám phá và Tiền xử lý (Chống rò rỉ dữ liệu)
Mục tiêu: Biến dữ liệu hóa đơn bán lẻ thành bảng đặc trưng số học và cất riêng tập Test.

**Nhiệm vụ 1 (File src/eda.py):**

Đọc data/raw/online_retail.csv.

Phân tích và vẽ biểu đồ các xu hướng mua hàng, sản phẩm bán chạy nhất.

**Nhiệm vụ 2 (File src/preprocess.py):**

Làm sạch: Loại bỏ các dòng có Quantity âm (hóa đơn hủy), xử lý giá trị thiếu.

Feature Engineering: Tạo cột Doanh thu (Sales = Quantity * UnitPrice). Trích xuất ngày, tháng, thứ trong tuần từ InvoiceDate. Mã hóa (Encode) các biến phân loại (như Quốc gia) thành số (tự code One-hot encoding hoặc Label encoding).

Cắt lớp: Tự code hàm chia dữ liệu 80% (Học) - 20% (Thi) trong utils.py, không dùng thư viện ngoài.

Đầu ra: Lưu 4 tập dữ liệu (X_train, X_test, y_train, y_test) dạng .pkl vào data/processed/.

# Giai đoạn 3: Huấn luyện 4 Mô hình (Model Training from Scratch)
Mục tiêu: Dạy máy học cách dự đoán doanh số dựa trên 80% dữ liệu, hoàn toàn bằng Toán học tự viết.

**Nhiệm vụ 1 (Thư mục src/models/):**

Xây dựng thủ công các class cho 4 thuật toán: LinearRegression, DecisionTree, RandomForest, XGBoost.

Các class cần có cấu trúc:

__init__(self): Khởi tạo siêu tham số.

fit(self, X, y): Viết vòng lặp cập nhật trọng số (Linear Regression) hoặc xây dựng cây đệ quy dựa trên tính toán giảm phương sai (Variance Reduction).

predict(self, X): Trả về con số dự đoán.

**Nhiệm vụ 2 (File src/train.py):**

Đọc cấu hình siêu tham số từ config.yaml.

Đọc X_train, y_train.

Khởi tạo và gọi .fit() cho cả 4 thuật toán.

Ghi nhận lịch sử chạy (Experimentation) vào docs/experiment_logs.csv.

Lưu 4 mô hình đã train thành các file .pkl vào models/.

# Giai đoạn 4: Đánh giá & Ráp nối Pipeline
Mục tiêu: Đưa 20% dữ liệu đã giấu ra để làm bài kiểm tra.

**Nhiệm vụ 1 (File src/evaluate.py):**

Tải 4 mô hình và tập Test lên.

Chạy .predict() cho từng mô hình.

Tự code các hàm toán học (trong utils.py) để tính: MSE (Mean Squared Error), MAE, RMSE, và R-squared.

In ra bảng so sánh hiệu năng.

**Nhiệm vụ 2 (File src/pipeline.py):**

Viết luồng tự động gọi preprocess -> train -> evaluate.

# Giai đoạn 5: Triển khai (Deployment) & Nghiệm thu
Mục tiêu: Tạo sản phẩm thực tế và đóng gói báo cáo.

**Nhiệm vụ 1 (Deployment - app.py):** Dùng thư viện Streamlit tạo giao diện Web. Cho phép người dùng nhập thông tin (Tháng, Mã sản phẩm...) và gọi mô hình tốt nhất (từ thư mục models/) để in ra dự đoán doanh số.

**Nhiệm vụ 2 (Báo cáo Notebook):** Mở notebooks/lab02_exploration.ipynb và notebooks/lab02_timeseries_eda.ipynb, trình bày đẹp mắt các biểu đồ EDA và phân tích chuỗi thời gian bằng Markdown.

**Nhiệm vụ 3 (Mặt tiền dự án):** Hoàn thiện file README.md. Ghi rõ danh sách thành viên và các lệnh để chạy pipeline, chạy giao diện app.py cho giảng viên chấm.