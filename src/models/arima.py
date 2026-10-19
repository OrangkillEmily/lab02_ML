import pandas as pd
import numpy as np
from src.models.linear_regression import LinearRegression

def forecast_next_7_days(df):
    """
    Dự báo tổng doanh thu 7 ngày tới bằng thuật toán Tự hồi quy (Autoregressive - AR).
    Tái Sử dụng class LinearRegression.
    """
    # 1. Tính tổng doanh thu theo ngày
    df_daily = df.groupby(df['InvoiceDate'].dt.date)['Sales'].sum().reset_index()
    df_daily['InvoiceDate'] = pd.to_datetime(df_daily['InvoiceDate'])
    df_daily = df_daily.sort_values('InvoiceDate')
    
    y = df_daily['Sales'].values
    
    # 2. Tạo tập dữ liệu trượt (Sliding Window) với lag = 7
    window_size = 7
    X_train, y_train = [], []
    for i in range(window_size, len(y)):
        X_train.append(y[i-window_size : i])
        y_train.append(y[i])
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # 3. Huấn luyện bằng mô hình Linear Regression tự code
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    
    # 4. Dự đoán đệ quy 7 ngày tới
    last_window = y[-window_size:]
    forecasts = []
    
    for _ in range(7):
        # Predict yêu cầu mảng 2D
        pred = model.predict(last_window.reshape(1, -1))[0]
        # Đảm bảo doanh thu không âm
        pred = max(0, pred)
        forecasts.append(pred)
        # Trượt window lên 1 bước (bỏ ngày cũ nhất, thêm ngày vừa dự đoán)
        last_window = np.append(last_window[1:], pred)
        
    # 5. Định dạng kết quả
    last_date = df_daily['InvoiceDate'].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
    
    return pd.DataFrame({
        'Ngày': future_dates,
        'Dự báo Doanh thu ($)': forecasts
    })