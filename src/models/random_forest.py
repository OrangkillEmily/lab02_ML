import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from joblib import Parallel, delayed
from src.models.decision_tree import DecisionTree
from src.utils import to_numpy_2d, to_numpy_1d

class RandomForest(BaseEstimator, RegressorMixin):
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=2, n_jobs=2, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_jobs = n_jobs 
        self.max_features = max_features
        self.trees = []

    def _train_single_tree(self, X_np, y_np):
        tree = DecisionTree(
            max_depth=self.max_depth, 
            min_samples_split=self.min_samples_split,
            max_features=self.max_features
        )
        X_sample, y_sample = self._bootstrap_sample(X_np, y_np)
        tree.fit(X_sample, y_sample)
        return tree

    def fit(self, X, y):
        X_np = to_numpy_2d(X)
        y_np = to_numpy_1d(y)
        
        print(f"[*] Đang huấn luyện Random Forest bằng Joblib...")
        self.trees = Parallel(n_jobs=self.n_jobs, require='sharedmem')(
            delayed(self._train_single_tree)(X_np, y_np) for _ in range(self.n_trees)
        )
        return self

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        if not self.trees:
            raise RuntimeError("Mô hình chưa được huấn luyện.")
        X_np = to_numpy_2d(X)
        tree_preds = np.array([tree.predict(X_np) for tree in self.trees])
        return np.mean(tree_preds, axis=0)
    
