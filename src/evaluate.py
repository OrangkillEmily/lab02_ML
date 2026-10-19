import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import calculate_mae, calculate_mse, calculate_rmse, calculate_r2

def load_model(filepath):
    """Hàm nạp mô hình từ file .pkl"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def plot_diagnostic(y_true, y_pred, model_name, dataset_name, output_dir):
    """
    Vẽ Diagnostic Plots. 
    """
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    residuals = y_true_np - y_pred_np
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Biểu đồ 1: Thực tế vs Dự đoán
    axes[0].scatter(y_true_np, y_pred_np, alpha=0.3, color='#4c78a8')
    max_val = max(np.max(y_true_np), np.max(y_pred_np))
    axes[0].plot([0, max_val], [0, max_val], color='#e45756', linestyle='--', linewidth=2)
    axes[0].set_title(f'Thực tế vs Dự đoán ({model_name} - {dataset_name})', fontsize=12)
    axes[0].set_xlabel('Doanh thu thực tế')
    axes[0].set_ylabel('Doanh thu dự đoán')
    axes[0].grid(True, alpha=0.3)
    
    # Biểu đồ 2: Residuals (Phần dư)
    axes[1].scatter(y_pred_np, residuals, alpha=0.3, color='#f58518')
    axes[1].axhline(0, color='#333333', linewidth=2)
    axes[1].set_title(f'Phân bố phần dư ({model_name} - {dataset_name})', fontsize=12)
    axes[1].set_xlabel('Doanh thu dự đoán')
    axes[1].set_ylabel('Sai số (Thực tế - Dự đoán)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Lưu file với tên phân biệt Train/Test
    filename = f'diagnostic_{model_name.lower()}_{dataset_name.lower()}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()

def main():
    # 1. Cấu hình đường dẫn
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    processed_dir = os.path.join(project_root, 'data', 'processed')
    models_dir = os.path.join(project_root, 'models')
    plot_dir = ensure_dir(os.path.join(project_root, 'docs', 'images'))
    
    # 2. Đọc cả tập Train và tập Test
    print(f" Đang tải dữ liệu Train và Test từ {processed_dir}...")
    try:
        X_train = pd.read_pickle(os.path.join(processed_dir, 'X_train.pkl'))
        y_train = pd.read_pickle(os.path.join(processed_dir, 'y_train.pkl'))
        X_test = pd.read_pickle(os.path.join(processed_dir, 'X_test.pkl'))
        y_test = pd.read_pickle(os.path.join(processed_dir, 'y_test.pkl'))
    except FileNotFoundError:
        print("[!] Không tìm thấy dữ liệu. Hãy chạy src/preprocess.py trước!")
        return

    # 3. Danh sách các mô hình cần đánh giá
    model_files = {
        'Linear_Regression': 'linear_regression.pkl',
        'Decision_Tree': 'decision_tree.pkl',
        'Random_Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl'
    }
    
    results = []
    
    print("\n ĐANG ĐÁNH GIÁ CÁC MÔ HÌNH (TRAIN vs TEST)...")
    
    # 4. Vòng lặp dự đoán, tính toán sai số và vẽ biểu đồ
    for model_name, filename in model_files.items():
        filepath = os.path.join(models_dir, filename)
        if not os.path.exists(filepath):
            print(f"    [!] Bỏ qua {model_name}: Chưa tìm thấy file {filename}")
            continue
            
        print(f"    -> Đang chạy dự đoán với {model_name}...")
        model = load_model(filepath)
        
        # --- DỰ ĐOÁN TRÊN TẬP TRAIN ---
        y_pred_train_log = model.predict(X_train)
        y_train_real = np.expm1(y_train)
        y_pred_train_real = np.expm1(y_pred_train_log)
        
        mae_train = calculate_mae(y_train_real, y_pred_train_real)
        mse_train = calculate_mse(y_train_real, y_pred_train_real)
        rmse_train = calculate_rmse(y_train_real, y_pred_train_real)
        r2_train = calculate_r2(y_train_real, y_pred_train_real)
        
        # --- DỰ ĐOÁN TRÊN TẬP TEST ---
        y_pred_test_log = model.predict(X_test)
        y_test_real = np.expm1(y_test)
        y_pred_test_real = np.expm1(y_pred_test_log)

        mae_test = calculate_mae(y_test_real, y_pred_test_real)
        mse_test = calculate_mse(y_test_real, y_pred_test_real)
        rmse_test = calculate_rmse(y_test_real, y_pred_test_real)
        r2_test = calculate_r2(y_test_real, y_pred_test_real)
        
        # Lưu kết quả song song để so sánh
        results.append({
            'Mô hình': model_name,
            'Train MAE': mae_train,
            'Test MAE': mae_test,
            'Train MSE': mse_train,
            'Test MSE': mse_test,
            'Train RMSE': rmse_train,
            'Test RMSE': rmse_test,
            'Train R2': r2_train,
            'Test R2': r2_test
        })
        
        # Vẽ và lưu biểu đồ Diagnostic cho cả Train và Test
        plot_diagnostic (y_train_real, y_pred_train_real, model_name, "Train", plot_dir)
        plot_diagnostic(y_test_real, y_pred_test_real, model_name, "Test", plot_dir)

    # 5. In bảng so sánh kết quả ra Terminal
    if results:
        results_df = pd.DataFrame(results)
        # Sắp xếp theo Test RMSE từ thấp đến cao (thấp nhất là tốt nhất)
        results_df = results_df.sort_values(by='Test RMSE').reset_index(drop=True)
        
        print("\n" + "="*85)
        print(" BẢNG XẾP HẠNG HIỆU NĂNG CÁC MÔ HÌNH (TRAIN vs TEST) ".center(85, '='))
        print("="*85)
        print(results_df.to_markdown(index=False, floatfmt=".4f"))
        print("="*85)
        
        best_model = results_df.iloc[0]['Mô hình']
        best_r2 = results_df.iloc[0]['Test R2']
        print(f"\n KẾT LUẬN: {best_model} là mô hình tốt nhất thực tế với Test R2 = {best_r2:.4f}")
        
        # Chẩn đoán nhanh bằng code
        best_train_rmse = results_df.iloc[0]['Train RMSE']
        best_test_rmse = results_df.iloc[0]['Test RMSE']
        if best_test_rmse > best_train_rmse * 1.5:
            print(" CẢNH BÁO: Mô hình tốt nhất đang có dấu hiệu Overfitting (Test RMSE lớn hơn nhiều so với Train RMSE).")
            print("            Hãy xem xét tinh chỉnh lại max_depth hoặc min_samples_split.")
            
        print(f"\n Biểu đồ đánh giá đã được lưu tại: {plot_dir}")

if __name__ == "__main__":
    main()

