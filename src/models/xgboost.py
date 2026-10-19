import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from src.models.decision_tree import DecisionTree
from src.utils import to_numpy_2d, to_numpy_1d

class XGBoost(BaseEstimator, RegressorMixin):
    # Thêm subsample và max_features
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3, 
                 min_samples_split=2, subsample=0.8, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample        # Lấy mẫu hàng (Row subsampling)
        self.max_features = max_features  # Lấy mẫu cột (Column subsampling)
        self.trees = []
        self.base_pred = None

    def fit(self, X, y):
        X_np = to_numpy_2d(X)
        y_np = to_numpy_1d(y)
        n_samples = X_np.shape[0]
        
        self.base_pred = float(np.mean(y_np))
        current_preds = np.full(y_np.shape, self.base_pred)
        
        for _ in range(self.n_estimators):
            residuals = y_np - current_preds
            
            # KỸ THUẬT TĂNG TỐC: Stochastic Gradient Boosting
            # Chỉ lấy ngẫu nhiên một tỷ lệ dữ liệu (ví dụ 80%) để đưa vào cây
            if self.subsample < 1.0:
                idx = np.random.choice(n_samples, int(n_samples * self.subsample), replace=False)
                X_train = X_np[idx]
                res_train = residuals[idx]
            else:
                X_train = X_np
                res_train = residuals
                
            # Khởi tạo cây với max_features (Kế thừa file DecisionTree)
            tree = DecisionTree(
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split,
                max_features=self.max_features 
            )
            
            # Hàm fit này giờ sẽ chạy cực nhanh nhờ dữ liệu X_train đã bị thu nhỏ
            # và max_features chỉ chọn 1 phần cột để xét
            tree.fit(X_train, res_train)
            
            # Cập nhật current_preds cho TẤT CẢ các dòng để tính residuals chuẩn xác cho cây tiếp theo
            update = tree.predict(X_np)
            current_preds += self.lr * update
            self.trees.append(tree)
            
        return self

    def predict(self, X):
        if self.base_pred is None:
            raise RuntimeError("Mô hình chưa được huấn luyện (gọi fit trước).")
        X_np = to_numpy_2d(X)
        preds = np.full(X_np.shape[0], self.base_pred)
        
        for tree in self.trees:
            preds += self.lr * tree.predict(X_np)
            
        return preds