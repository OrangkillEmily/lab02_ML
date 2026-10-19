import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from src.utils import to_numpy_2d, to_numpy_1d

class LinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept=True, learning_rate=None, n_iterations=None):
        # learning_rate và n_iterations được giữ lại trong __init__ để không bị lỗi 
        # khi đọc cấu hình cũ từ config.yaml, dù Normal Equation không cần dùng đến chúng.
        self.fit_intercept = fit_intercept
        self.weights = None
        self.intercept = 0.0

    def fit(self, X, y):
        X_np = to_numpy_2d(X)
        y_np = to_numpy_1d(y)
        
        # Thêm cột giá trị 1 vào ma trận X để tính hệ số tự do (bias/intercept)
        if self.fit_intercept:
            X_design = np.c_[np.ones(X_np.shape[0]), X_np]
        else:
            X_design = X_np
            
        # Normal Equation: beta = (X^T * X)^-1 * X^T * y
        # Dùng np.linalg.pinv (pseudo-inverse) an toàn hơn inverse thông thường
        beta = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y_np
        
        if self.fit_intercept:
            self.intercept = beta[0]
            self.weights = beta[1:]
        else:
            self.weights = beta
            
        return self

    def predict(self, X):
        if self.weights is None:
            raise RuntimeError("Mô hình chưa được huấn luyện (gọi fit trước).")
        X_np = to_numpy_2d(X)
        return X_np @ self.weights + self.intercept
    
"""
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Chuyển DataFrame/Series sang Numpy Array để tính toán ma trận
        X_np = X.to_numpy() if hasattr(X, 'to_numpy') else X
        y_np = y.to_numpy() if hasattr(y, 'to_numpy') else y
        
        n_samples, n_features = X_np.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Vòng lặp cập nhật trọng số (Gradient Descent)
        for _ in range(self.n_iterations):
            y_predicted = np.dot(X_np, self.weights) + self.bias
            
            # Tính đạo hàm (Gradient)
            dw = (1 / n_samples) * np.dot(X_np.T, (y_predicted - y_np))
            db = (1 / n_samples) * np.sum(y_predicted - y_np)
            
            # Cập nhật trọng số và sai số
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        X_np = X.to_numpy() if hasattr(X, 'to_numpy') else X
        return np.dot(X_np, self.weights) + self.bias
"""