import numpy as np
import pandas as pd

def get_recommendations(customer_id, df, top_n=5):
    """
    Gợi ý sản phẩm bằng User-Based Collaborative Filtering (Cosine Similarity).
    """
    # 1. Kiểm tra khách hàng có tồn tại không
    if customer_id not in df['CustomerID'].unique():
        return None
        
    # 2. Tạo User-Item Matrix (Khách hàng - Sản phẩm)
    # Dùng size() thay vì count() giúp thao tác Groupby nhanh hơn
    user_item = df.groupby(['CustomerID', 'Description']).size().unstack(fill_value=0)
    user_item = (user_item > 0).astype(int)
    
    # 3. Vector của khách hàng hiện tại
    target_user_vector = user_item.loc[customer_id].values
    
    # Lấy SẴN tập hợp các sản phẩm khách hàng mục tiêu ĐÃ MUA. 
    # Dùng Python Set để tra cứu với tốc độ O(1) thay vì O(N) của Pandas.
    target_bought_items = set(user_item.columns[target_user_vector > 0])
    
    # 4. Tính Cosine Similarity bằng Toán học: (A dot B) / (|A| * |B|)
    matrix_values = user_item.values
    dot_product = matrix_values.dot(target_user_vector)
    
    norm_matrix = np.linalg.norm(matrix_values, axis=1)
    norm_target = np.linalg.norm(target_user_vector)
    
    # Tránh lỗi chia cho 0
    norm_matrix[norm_matrix == 0] = 1e-9
    norm_target = norm_target if norm_target > 0 else 1e-9
    
    similarities = dot_product / (norm_matrix * norm_target)
    
    # 5. Tìm Top khách hàng giống nhất (Bỏ qua chính họ ở vị trí số 0)
    similar_users_idx = similarities.argsort()[::-1][1:11]
    
    # 6. Tìm sản phẩm khách hàng này chưa mua nhưng những người giống họ đã mua
    recommended_items = {}
    
    for idx in similar_users_idx:
        similar_user_id = user_item.index[idx]
        similarity_score = similarities[idx]
        
        # Các sản phẩm người giống mình đã mua
        similar_user_vector = user_item.loc[similar_user_id].values
        items_bought = user_item.columns[similar_user_vector > 0]
        
        for item in items_bought:
            if item not in target_bought_items:
                recommended_items[item] = recommended_items.get(item, 0) + similarity_score
                
    # 7. Trả về Top N sản phẩm có điểm gợi ý cao nhất
    sorted_items = sorted(recommended_items.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_items[:top_n]]