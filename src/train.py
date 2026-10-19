import os
import yaml
import pickle
import pandas as pd
import time
from datetime import datetime

# Nạp 4 thuật toán "tự code" từ src/models
from src.models.linear_regression import LinearRegression
from src.models.decision_tree import DecisionTree
from src.models.random_forest import RandomForest
from src.models.xgboost import XGBoost

def load_config(config_path="config.yaml"):
    """Đọc cấu hình siêu tham số từ file YAML"""
    print(" Đang đọc cấu hình siêu tham số từ config.yaml...")
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def save_model(model, filename, models_dir):
    """Lưu mô hình đã huấn luyện thành file .pkl"""
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    filepath = os.path.join(models_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def log_experiment(log_file, hyperparams, execution_times):
    """Ghi nhận lịch sử thử nghiệm (Experimentation) vào docs/experiment_logs.csv[cite: 1]"""
    print(f" Đang ghi log thử nghiệm vào {log_file}...")
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "LR_Params": str(hyperparams.get('linear_regression', {})),
        "DT_Params": str(hyperparams.get('decision_tree', {})),
        "RF_Params": str(hyperparams.get('random_forest', {})),
        "XGB_Params": str(hyperparams.get('xgboost', {})),
        "Total_Train_Time_Sec": round(sum(execution_times.values()), 2)
    }
    
    df_log = pd.DataFrame([log_entry])
    file_exists = os.path.isfile(log_file)
    # Mode 'a' (append) giúp ghi nối tiếp các lần chạy khác nhau mà không xóa dữ liệu cũ
    df_log.to_csv(log_file, mode='a', header=not file_exists, index=False, encoding='utf-8')

def main():
    # 1. Khởi tạo đường dẫn hệ thống
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    config_path = os.path.join(project_root, "config.yaml")
    config = load_config(config_path)
    
    processed_dir = os.path.join(project_root, config['paths']['processed_data'])
    models_dir = os.path.join(project_root, config['paths']['models_dir'])
    log_file = os.path.join(project_root, config['paths']['log_file'])
    hp = config['hyperparameters']

    # 2. Nạp dữ liệu huấn luyện
    print(f" Đang tải X_train và y_train từ {processed_dir}...")
    try:
        X_train = pd.read_pickle(os.path.join(processed_dir, 'X_train.pkl'))
        y_train = pd.read_pickle(os.path.join(processed_dir, 'y_train.pkl'))
    except FileNotFoundError:
        print(" Không tìm thấy dữ liệu. Vui lòng chạy tiền xử lý trước!")
        return

    # 3. Đăng ký cấu hình 4 mô hình
    # Lưu ý: Linear Regression nay đã dùng Normal Equation nên fit_intercept=True là đủ.
    models = {
        "Linear Regression": (
            LinearRegression(fit_intercept=True),
            "linear_regression.pkl"
        ),
        "Decision Tree": (
            DecisionTree(
                max_depth=hp['decision_tree']['max_depth'], 
                min_samples_split=hp['decision_tree']['min_samples_split'],
                max_features='sqrt'
            ),
            "decision_tree.pkl"
        ),
        "Random Forest": (
            RandomForest(
                n_trees=hp['random_forest']['n_trees'], 
                max_depth=hp['random_forest']['max_depth'], 
                min_samples_split=hp['random_forest']['min_samples_split'],
                max_features='sqrt',
                n_jobs=2
            ),
            "random_forest.pkl"
        ),
        "XGBoost": (
            XGBoost(
                n_estimators=hp['xgboost']['n_estimators'], 
                learning_rate=hp['xgboost']['learning_rate'], 
                max_depth=hp['xgboost']['max_depth'], 
                min_samples_split=hp['xgboost']['min_samples_split'],
                max_features='sqrt',
                subsample=0.8
            ),
            "xgboost.pkl"
        )
    }

    print("\n" + "="*60)
    print(" BẮT ĐẦU HUẤN LUYỆN 4 MÔ HÌNH (FROM SCRATCH) ".center(60))
    print("="*60)

    execution_times = {}

    # 4. Vòng lặp huấn luyện, tính thời gian và lưu mô hình (Phong cách MLOps)
    for i, (name, (model, filename)) in enumerate(models.items(), 1):
        print(f"  [{i}/4] Đang huấn luyện {name}...", end=" ", flush=True)
        
        # Bắt đầu đo thời gian
        t0 = time.time()
        
        # Huấn luyện
        model.fit(X_train, y_train)
        
        # Kết thúc đo thời gian
        elapsed = time.time() - t0
        execution_times[name] = elapsed
        
        # Lưu file .pkl
        save_model(model, filename, models_dir)
        
        print(f"XONG! (Mất {elapsed:.2f}s) -> Đã lưu {filename}")

    # 5. Ghi log thí nghiệm
    print("-" * 60)
    log_experiment(log_file, hp, execution_times)
    
    print(f"\n ĐÃ HOÀN TẤT HUẤN LUYỆN!")
    print(f"    - Tổng thời gian huấn luyện: {sum(execution_times.values()):.2f} giây")
    print(f"    - Toàn bộ mô hình nằm tại: {models_dir}")

if __name__ == "__main__":
    main()