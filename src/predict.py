import os
import pickle
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

def load_pickle(filepath):
    """Nạp file .pkl (Mô hình hoặc Mappings)"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def prepare_features(date_str, country, mappings):
    """
    Biến đổi thông tin đầu vào thành vector đặc trưng 8 chiều.
    """
    # Xử lý thời gian
    dt = pd.to_datetime(date_str)
    month = dt.month
    day = dt.day
    dayofweek = dt.dayofweek

    # Xử lý quốc gia
    country_encoded = mappings['country_encode'].get(country, -1)
    
    # Lấy số lượng trung bình lịch sử
    hist_sales = mappings['country_avg_sales'].get(country, mappings['global_avg_sales'])
    

    # Tạo vector đặc trưng (BẮT BUỘC PHẢI KHỚP THỨ TỰ LÚC TRAIN)
    X_input = np.array([[
        month,             
        day,               
        dayofweek,         
        country_encoded,   
        hist_sales,       
    ]])

    return X_input, hist_sales


def main():
    # Phần giao diện Dòng lệnh (CLI)
    parser = argparse.ArgumentParser(description="Công cụ CLI dự đoán Doanh thu Bán lẻ.")
    parser.add_argument("--date", type=str, default=datetime.today().strftime('%Y-%m-%d'), help="Ngày (YYYY-MM-DD)")
    parser.add_argument("--country", type=str, default="United Kingdom", help="Quốc gia")
    parser.add_argument("--model", type=str, default="xgboost.pkl", help="File mô hình (VD: xgboost.pkl)")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    mappings_path = os.path.join(project_root, 'data', 'processed', 'mappings.pkl')
    model_path = os.path.join(project_root, 'models', args.model)

    if not os.path.exists(mappings_path) or not os.path.exists(model_path):
        print("[!] Thiếu file model hoặc mappings. Hãy chạy pipeline trước.")
        return

    mappings = load_pickle(mappings_path)
    model = load_pickle(model_path)

    # GỌI HÀM LÕI
    X_input, hist_sales = prepare_features(args.date, args.country, mappings)
    
    prediction_log = model.predict(X_input)[0]
    predicted_sales = max(0, np.expm1(prediction_log))

    print("\n" + "="*50)
    print(" KẾT QUẢ DỰ ĐOÁN DOANH THU ".center(50))
    print("="*50)
    print(f" 📅 Ngày đặt hàng   : {args.date}")
    print(f" 🌍 Quốc gia        : {args.country}")
    print(f" 📦 Sức mua lịch sử : ${hist_sales:,.2f} / đơn hàng")
    print(f" 🧠 Mô hình sử dụng : {args.model}")
    print("-" * 50)
    print(f" 🚀 DOANH THU DỰ KIẾN: ${predicted_sales:,.2f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()