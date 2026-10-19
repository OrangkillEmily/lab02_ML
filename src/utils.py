import numpy as np
import pandas as pd

def to_numpy_2d(X):
    """Đảm bảo X luôn là ma trận 2D và định dạng float để tránh lỗi shape"""
    arr = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr.astype(float)

def to_numpy_1d(y):
    """Đảm bảo y luôn là mảng 1D"""
    arr = y.to_numpy() if isinstance(y, (pd.DataFrame, pd.Series)) else np.asarray(y)
    return arr.reshape(-1).astype(float)

def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    """Tự code hàm chia tập Train/Test theo tỉ lệ truyền vào (Không dùng scikit-learn)"""
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = len(X)
    test_samples = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    
    X_train = X.iloc[train_indices].copy()
    X_test = X.iloc[test_indices].copy()
    y_train = y.iloc[train_indices].copy()
    y_test = y.iloc[test_indices].copy()
    
    return X_train, X_test, y_train, y_test


def calculate_mae(y_true, y_pred):
    """Tính Mean Absolute Error (Sai số tuyệt đối trung bình)"""
    y_true_np = to_numpy_1d(y_true)
    y_pred_np = to_numpy_1d(y_pred)
    return float(np.mean(np.abs(y_true_np - y_pred_np)))

def calculate_mse(y_true, y_pred):
    """Tính Mean Squared Error (Sai số bình phương trung bình)"""
    y_true_np = to_numpy_1d(y_true)
    y_pred_np = to_numpy_1d(y_pred)
    return float(np.mean((y_true_np - y_pred_np) ** 2))

def calculate_rmse(y_true, y_pred):
    """Tính Root Mean Squared Error (Căn bậc hai của MSE)"""
    return float(np.sqrt(calculate_mse(y_true, y_pred)))

def calculate_r2(y_true, y_pred):
    """Tính R-squared (Hệ số xác định độ phù hợp của mô hình)"""
    y_true_np = to_numpy_1d(y_true)
    y_pred_np = to_numpy_1d(y_pred)
    
    ss_res = np.sum((y_true_np - y_pred_np) ** 2) # Tổng bình phương phần dư
    ss_tot = np.sum((y_true_np - np.mean(y_true_np)) ** 2) # Tổng bình phương toàn phần
    
    # Tránh chia cho 0 nếu ss_tot = 0
    if ss_tot == 0:
        return 0.0
    return float(1 - (ss_res / ss_tot))