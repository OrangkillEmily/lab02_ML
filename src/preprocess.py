import os
import pandas as pd
import numpy as np
import pickle
from src.utils import train_test_split_custom

FORBIDDEN_COLUMNS = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'UnitPrice', 'Sales']

def clean_data(df):
    print("Đang làm sạch dữ liệu: Xử lý missing values và hóa đơn lỗi...")
    # Bỏ hóa đơn hủy hoặc đơn giá ảo
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()
    # Bỏ các dòng thiếu ID khách hàng
    df = df.dropna(subset=['CustomerID']).copy()
    return df

def add_time_features(df):
    print("Đang trích xuất đặc trưng thời gian...")
    df['Sales'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Month'] = df['InvoiceDate'].dt.month
    df['Day'] = df['InvoiceDate'].dt.day
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    return df

def add_aggregate_features_and_encode(train_df, test_df):
    train = train_df.copy()
    test = test_df.copy()
    mappings = {}
    
    # Mã hóa Country
    unique_countries = train['Country'].unique()
    country_to_int = {country: idx for idx, country in enumerate(unique_countries)}
    mappings['country_encode'] = country_to_int
    
    # ĐẶC TRƯNG MỚI: Lịch sử CHI TIÊU trung bình theo Quốc gia
    country_avg_sales = train.groupby('Country')['Sales'].mean().to_dict()
    global_avg_sales = float(train['Sales'].mean()) 
    mappings['country_avg_sales'] = country_avg_sales
    mappings['global_avg_sales'] = global_avg_sales

    for df in [train, test]:
        df['Country_Encoded'] = df['Country'].map(country_to_int).fillna(-1)
        df['Country_Historical_Avg_Sales'] = df['Country'].map(country_avg_sales).fillna(global_avg_sales)
        
    return train, test, mappings

"""
def add_aggregate_features_and_encode(train_df, test_df):
    
    Học hỏi từ nhóm bạn: Tính toán đặc trưng trung bình và Label Encoding CHỈ từ tập Train,
    sau đó áp dụng sang tập Test để chống Data Leakage.
   
    print("Đang mã hóa và tạo đặc trưng Lịch sử (Aggregate Features)...")
    train = train_df.copy()
    test = test_df.copy()
    
    mappings = {}
    
    # 1. Mã hóa Country (Label Encoding)
    unique_countries = train['Country'].unique()
    country_to_int = {country: idx for idx, country in enumerate(unique_countries)}
    mappings['country_encode'] = country_to_int
    
    # 2. Tạo đặc trưng "Trung bình số lượng mua của từng Quốc gia" (dựa trên Train)
    country_avg_qty = train.groupby('Country')['Quantity'].mean().to_dict()
    global_avg_qty = float(train['Quantity'].mean()) # Dùng nếu Test có Quốc gia lạ
    mappings['country_avg_qty'] = country_avg_qty
    mappings['global_avg_qty'] = global_avg_qty

    # 3. Áp dụng các từ điển vào Train và Test
    for df in [train, test]:
        # Map Country -> Số. Nếu gặp nước lạ ở Test, gom vào nhóm -1
        df['Country_Encoded'] = df['Country'].map(country_to_int).fillna(-1)
        # Map Country -> Số lượng trung bình lịch sử
        df['Country_Historical_Avg_Qty'] = df['Country'].map(country_avg_qty).fillna(global_avg_qty)
        
        # Đặc trưng phái sinh: Doanh thu kỳ vọng = Đơn giá lịch sử (Trung bình) * Số lượng lịch sử
        # (Lưu ý: Thực tế ta có UnitPrice trong hóa đơn mới, nên lấy UnitPrice * Qty Lịch sử)
        df['Expected_Revenue'] = df['UnitPrice'] * df['Country_Historical_Avg_Qty']

    return train, test, mappings
    
"""

def assert_no_leakage(feature_columns):
    """Lập trình phòng thủ (Defensive Programming) - Ngăn chặn rò rỉ dữ liệu"""
    for col in FORBIDDEN_COLUMNS:
        assert col not in feature_columns, f"[!] LỖI BẢO MẬT: Cột '{col}' bị lọt vào tập tính toán!"
    print("    -> [OK] Kiểm tra rò rỉ: Dữ liệu an toàn.")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'online_retail.csv')
    processed_dir = os.path.join(project_root, 'data', 'processed')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # 1. Đọc và Tiền xử lý cơ bản
    df = pd.read_csv(raw_data_path, encoding='unicode_escape')
    df = clean_data(df)
    df = add_time_features(df)
    
    # 2. Chia Train/Test (80-20) TRƯỚC KHI tạo Mappings để tránh Leakage
    print("Đang chia tách Train/Test (80-20)...")
    train_df, test_df, _, _ = train_test_split_custom(df, df, test_size=0.2, random_state=42)
    
    # 3. Tính toán Aggregate & Encode
    train_processed, test_processed, mappings = add_aggregate_features_and_encode(train_df, test_df)
    
    # 4. Tách X (Đặc trưng) và y (Mục tiêu)
    feature_cols = ['Month', 'Day', 'DayOfWeek', 'Country_Encoded', 'Country_Historical_Avg_Sales']
    target_col = 'Sales'
    
    assert_no_leakage(feature_cols)
    
    X_train = train_processed[feature_cols]
    X_test = test_processed[feature_cols]
    
    y_train = np.log1p(train_processed[target_col])
    y_test = np.log1p(test_processed[target_col])
    
    # 5. Lưu ra đĩa
    print(f"[*] Đang lưu các ma trận và từ điển mappings vào {processed_dir}...")
    X_train.to_pickle(os.path.join(processed_dir, 'X_train.pkl'))
    X_test.to_pickle(os.path.join(processed_dir, 'X_test.pkl'))
    y_train.to_pickle(os.path.join(processed_dir, 'y_train.pkl'))
    y_test.to_pickle(os.path.join(processed_dir, 'y_test.pkl'))
    
    # Lưu từ điển để dùng cho Streamlit Web App sau này
    with open(os.path.join(processed_dir, 'mappings.pkl'), 'wb') as f:
        pickle.dump(mappings, f)
    
    print("\n ĐÃ HOÀN TẤT TIỀN XỬ LÝ NÂNG CAO!")
    print(f"    - Kích thước tập X_train: {X_train.shape}")
    print(f"    - Kích thước tập X_test:  {X_test.shape}")

if __name__ == "__main__":
    main()

