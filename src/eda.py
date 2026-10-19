import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    print(f"Đang tải dữ liệu từ: {filepath}")
    return pd.read_csv(filepath, encoding='unicode_escape')

def basic_clean_for_eda(df):
    print("Đang làm sạch dữ liệu sơ bộ...")
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Sales'] = df['Quantity'] * df['UnitPrice']
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')
    return df

def ensure_dir(directory):
    """Tạo thư mục nếu chưa tồn tại"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def print_outlier_analysis(df):
    print("\n" + "="*60)
    print(" PHÂN TÍCH OUTLIER (GIÁ TRỊ NGOẠI LAI) ".center(60, "="))
    print("="*60)
    
    # 1. Thống kê mô tả (describe)
    print("\n1. BẢNG THỐNG KÊ MÔ TẢ (describe):")
    # Dùng to_string để căn chỉnh format cho đẹp trên Terminal
    print(df[['Quantity', 'UnitPrice', 'Sales']].describe().to_string(float_format="{:.2f}".format))
    
    # 2. Bách phân vị (Percentiles)
    print("\n2. ĐO LƯỜNG GIỚI HẠN BẰNG BÁCH PHÂN VỊ (99%):")
    q99_qty = df['Quantity'].quantile(0.99)
    q99_sales = df['Sales'].quantile(0.99)
    
    print(f"  - 99% đơn hàng có Số lượng (Quantity) từ {q99_qty:.2f} trở xuống.")
    print(f"  - 99% đơn hàng có Doanh thu (Sales) từ ${q99_sales:.2f} trở xuống.")
    
    qty_outliers = len(df[df['Quantity'] > q99_qty])
    sales_outliers = len(df[df['Sales'] > q99_sales])
    print(f"\n  -> Có {qty_outliers} đơn hàng vượt ngưỡng 99% Số lượng (Cần cẩn thận!)")
    print(f"  -> Có {sales_outliers} đơn hàng vượt ngưỡng 99% Doanh thu (Cần cẩn thận!)")
    print("="*60 + "\n")
    
def plot_outlier_boxplots(df, output_dir):
    print("Đang vẽ Biểu đồ Hộp (Boxplots) để nhận diện Outlier...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Vẽ Boxplot cho Quantity
    sns.boxplot(x=df['Quantity'], ax=axes[0], color='#4c78a8')
    axes[0].set_title('Phân bố Số lượng (Quantity)', fontsize=12)
    
    # Vẽ Boxplot cho Sales
    sns.boxplot(x=df['Sales'], ax=axes[1], color='#e45756')
    axes[1].set_title('Phân bố Doanh thu (Sales)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'outlier_boxplots.png'), dpi=150)
    plt.close()

def plot_correlation_heatmap(df, output_dir):
    print("Đang vẽ Ma trận tương quan (Correlation Heatmap)...")
    numeric_cols = ['Quantity', 'UnitPrice', 'Sales']
    corr = df[numeric_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", linewidths=.5)
    plt.title('Ma trận tương quan các biến số học', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150)
    plt.close()

def plot_top_selling_products(df, output_dir, top_n=10):
    print(f"Đang vẽ biểu đồ Top {top_n} sản phẩm bán chạy nhất...")
    top_products = df.groupby('Description')['Quantity'].sum().nlargest(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_products.values, y=top_products.index, palette='viridis')
    plt.title(f'Top {top_n} Sản phẩm bán chạy nhất', fontsize=14)
    plt.xlabel('Tổng số lượng', fontsize=12)
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_products.png'), dpi=150)
    plt.close()

def plot_timeseries_rolling_mean(df, output_dir):
    print("Đang vẽ phân tích chuỗi thời gian (Rolling Means)...")
    # Tính doanh thu theo ngày
    daily_sales = df.set_index('InvoiceDate').resample('D')['Sales'].sum()
    
    # Tính trung bình trượt 7 ngày và 30 ngày (Học từ nhóm bạn)
    rolling_7d = daily_sales.rolling(window=7, min_periods=1).mean()
    rolling_30d = daily_sales.rolling(window=30, min_periods=1).mean()
    
    plt.figure(figsize=(14, 6))
    plt.plot(daily_sales.index, daily_sales.values, alpha=0.3, label='Doanh thu theo ngày', color='#4c78a8')
    plt.plot(rolling_7d.index, rolling_7d.values, label='Trung bình 7 ngày (Xu hướng tuần)', color='#f58518', linewidth=2)
    plt.plot(rolling_30d.index, rolling_30d.values, label='Trung bình 30 ngày (Xu hướng tháng)', color='#54a24b', linewidth=2)
    
    plt.title('Xu hướng doanh thu với Đường trung bình trượt (Rolling Means)', fontsize=14)
    plt.xlabel('Thời gian', fontsize=12)
    plt.ylabel('Doanh thu ($)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timeseries_trends.png'), dpi=150)
    plt.close()

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'online_retail.csv')
    output_dir = ensure_dir(os.path.join(project_root, 'docs', 'images'))
    
    if not os.path.exists(raw_data_path):
        print(f"[!] Lỗi: Không tìm thấy {raw_data_path}")
        return

    df_raw = load_data(raw_data_path)
    df_clean = basic_clean_for_eda(df_raw)
    
    print_outlier_analysis(df_clean)

    # Chạy các biểu đồ và lưu ảnh
    plot_outlier_boxplots(df_clean, output_dir)
    plot_correlation_heatmap(df_clean, output_dir)
    plot_top_selling_products(df_clean, output_dir)
    plot_timeseries_rolling_mean(df_clean, output_dir)
    
    print(f"Hoàn tất! Tất cả biểu đồ đã được lưu tự động tại: {output_dir}")

if __name__ == "__main__":
    main()

