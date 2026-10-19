import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from src.utils import to_numpy_2d, to_numpy_1d

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature       
        self.threshold = threshold   
        self.left = left             
        self.right = right           
        self.value = value           

class DecisionTree(BaseEstimator, RegressorMixin):
    # Thêm tham số max_features (dùng cho Random Forest)
    def __init__(self, max_depth=5, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        X_np = to_numpy_2d(X)
        y_np = to_numpy_1d(y)
        self.root = self._build_tree(X_np, y_np, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        
        # Điều kiện dừng
        if n_samples >= self.min_samples_split and depth < self.max_depth:
            best_split = self._get_best_split(X, y, n_features)
            if best_split:
                left_subtree = self._build_tree(best_split['X_left'], best_split['y_left'], depth + 1)
                right_subtree = self._build_tree(best_split['X_right'], best_split['y_right'], depth + 1)
                return Node(best_split['feature'], best_split['threshold'], left_subtree, right_subtree)
        
        leaf_value = np.mean(y)
        return Node(value=leaf_value)
    
    def _get_best_split(self, X, y, n_features):
        best_split = {}
        best_score = -float("inf")
        
        if self.max_features is not None:
            if isinstance(self.max_features, str) and self.max_features == 'sqrt':
                n_features_to_sample = max(1, int(np.sqrt(n_features)))
            else:
                n_features_to_sample = n_features
            feature_indices = np.random.choice(n_features, n_features_to_sample, replace=False)
        else:
            feature_indices = range(n_features)
            
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            
            # Sắp xếp
            sort_idx = np.argsort(feature_values)
            X_sorted = feature_values[sort_idx]
            y_sorted = y[sort_idx]
            
            if X_sorted[0] == X_sorted[-1]:
                continue
                
        
            sum_left = np.cumsum(y_sorted)[:-1]
            total_sum = sum_left[-1] + y_sorted[-1]
            sum_right = total_sum - sum_left
            
            # Tính số lượng phần tử hai bên
            n_left = np.arange(1, len(y_sorted))
            n_right = len(y_sorted) - n_left
            
            # Tìm các vị trí mà giá trị liền kề khác nhau (để có thể cắt)
            valid_splits = X_sorted[1:] != X_sorted[:-1]
            
            if not np.any(valid_splits):
                continue
                
            # Tính điểm (score) cho TOÀN BỘ các điểm cắt cùng lúc
            scores = (sum_left**2) / n_left + (sum_right**2) / n_right
            
            # Chỉ lấy điểm của những vị trí cắt hợp lệ, còn lại gán bằng -vô cực
            scores = np.where(valid_splits, scores, -float('inf'))
            
            # Tìm điểm cắt tốt nhất
            max_score_idx = np.argmax(scores)
            max_score = scores[max_score_idx]
            
            if max_score > best_score:
                best_score = max_score
                best_threshold = (X_sorted[max_score_idx] + X_sorted[max_score_idx + 1]) / 2.0
                best_feature = feature_idx
                
        if best_score > -float("inf"):
            left_idx = np.where(X[:, best_feature] <= best_threshold)[0]
            right_idx = np.where(X[:, best_feature] > best_threshold)[0]
            best_split = {
                'feature': best_feature,
                'threshold': best_threshold,
                'X_left': X[left_idx], 'y_left': y[left_idx],
                'X_right': X[right_idx], 'y_right': y[right_idx]
            }
            return best_split
            
        return None


    def predict(self, X):
        if self.root is None:
            raise RuntimeError("Mô hình chưa được huấn luyện (gọi fit trước).")
        X_np = to_numpy_2d(X)
        return np.array([self._traverse_tree(x, self.root) for x in X_np])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)