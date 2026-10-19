import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from abc import ABC, abstractmethod
from src.predict import prepare_features
from src.models.arima import forecast_next_7_days
from src.models.recommender import get_recommendations

class Info(ABC):
    @abstractmethod
    def get_title(self):
        raise NotImplementedError

    @abstractmethod
    def get_message(self):
        raise NotImplementedError

    def render(self):
        return f"**{self.get_title()}**\n\n{self.get_message()}"

class SalesPredictionInfo(Info):
    def __init__(self, selected_country, hist_sales, order_date, model_choice, predicted_sales):
        self.selected_country = selected_country
        self.hist_sales = hist_sales
        self.order_date = order_date
        self.model_choice = model_choice
        self.predicted_sales = predicted_sales

    def get_title(self):
        return "Giải thích thuật toán"
    
    def get_message(self):
        weekday_names = ["Chủ Nhật", "Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy"]
        weekday_label = weekday_names[self.order_date.weekday()]
        return (
            f"Mô hình nhận thấy thị trường **{self.selected_country}** trong quá khứ thường chi tiêu trung bình khoảng "
            f"**\${self.hist_sales:,.2f}** mỗi đơn. "
            f"Kết hợp với yếu tố thời gian là **{weekday_label}**, "
            f"thuật toán **{self.model_choice}** dự phóng doanh số mang lại sẽ là **\${self.predicted_sales:,.2f}**."
        )
    


# --- CÁC HÀM CACHE DỮ LIỆU ---
@st.cache_data
def load_mappings(filepath):
    with open(filepath, 'rb') as f: return pickle.load(f)

@st.cache_resource
def load_model(filepath):
    with open(filepath, 'rb') as f: return pickle.load(f)

@st.cache_data
def load_raw_dataset(filepath):
    """Tải và làm sạch nhẹ dữ liệu gốc phục vụ Tab 2 và 3"""
    df = pd.read_csv(filepath, encoding='unicode_escape')
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()
    df['Sales'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df.dropna(subset=['CustomerID'])
    return df

# --- GIAO DIỆN CHÍNH ---
def main():
    st.set_page_config(page_title="Hệ Sinh Thái Bán Lẻ AI", page_icon="🛍️", layout="wide")
    
    # Đường dẫn
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mappings_path = os.path.join(current_dir, 'data', 'processed', 'mappings.pkl')
    raw_data_path = os.path.join(current_dir, 'data', 'raw', 'online_retail.csv')
    models_dir = os.path.join(current_dir, 'models')
    
    if not os.path.exists(mappings_path):
        st.error("⚠️ Hãy chạy Pipeline tiền xử lý trước!")
        return
        
    mappings = load_mappings(mappings_path)
    df_raw = load_raw_dataset(raw_data_path)
    
    st.title("🛍️ Hệ Sinh Thái AI: Dự Đoán & Gợi Ý Bán Lẻ")
    st.markdown("Hệ thống tích hợp 3 tính năng cốt lõi được xây dựng hoàn toàn từ các thuật toán Toán học (From Scratch).")
    
    # TẠO 3 TABS THẦN THÁNH
    tab1, tab2, tab3 = st.tabs(["🛒 Tab 1: Dự đoán Đơn hàng (Regression)", 
                                "📈 Tab 2: Dự báo 7 ngày tới (Time Series)", 
                                "🎯 Tab 3: Gợi ý Sản phẩm (Recommender)"])
    
    with tab1:
        st.sidebar.header("⚙️ Cấu hình Tab 1")
        model_choice = st.sidebar.selectbox("Chọn thuật toán dự đoán:", 
                                            ("XGBoost", "Random_Forest", "Decision_Tree", "Linear_Regression"))
        model_path = os.path.join(models_dir, f"{model_choice.lower()}.pkl")
        
        if os.path.exists(model_path):
            model = load_model(model_path)
            
            st.subheader("🛒 Nhập thông tin đơn hàng mới")
            
            col1, col2 = st.columns(2)
            with col1: order_date = st.date_input("Ngày đặt hàng:", datetime.today())
            with col2:
                clist = list(mappings['country_encode'].keys())
                if 'United Kingdom' in clist: clist.insert(0, clist.pop(clist.index('United Kingdom')))
                selected_country = st.selectbox("Quốc gia:", clist)
            

            if st.button("🚀 DỰ ĐOÁN DOANH THU", type="primary"):
                X_input, hist_sales = prepare_features(order_date, selected_country, mappings)
                prediction = max(0, model.predict(X_input)[0])
                
                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("Doanh thu Dự đoán", f"${prediction:,.2f}")
                rc2.metric("Lịch sử Chi tiêu", f"${hist_sales:,.2f} / đơn hàng")
                rc3.metric("Mô hình", model_choice)
                
                st.info(SalesPredictionInfo(selected_country, hist_sales, order_date, model_choice, prediction).render())
        else:
            st.warning(f"Chưa huấn luyện mô hình {model_choice}.")

    
    with tab2:
        st.subheader("📈 Dự báo Doanh thu Tổng Hệ thống trong 7 Ngày Tới")
        st.markdown("Sử dụng mô hình **Tự hồi quy (Autoregressive)** với thuật toán Linear Regression tự xây dựng để dự đoán tương lai dựa trên độ trễ (lag=7) của quá khứ.")
        
        if st.button("📊 BẮT ĐẦU DỰ BÁO 7 NGÀY"):
            with st.spinner("Đang tính toán chuỗi thời gian..."):
                forecast_df = forecast_next_7_days(df_raw)
                
                col_chart, col_data = st.columns([2, 1])
                with col_chart:
                    # Chuyển đổi để vẽ biểu đồ line chart của Streamlit
                    chart_data = forecast_df.set_index('Ngày')
                    st.line_chart(chart_data, color="#f58518")
                
                with col_data:
                    st.dataframe(forecast_df.style.format({'Dự báo Doanh thu ($)': "${:,.2f}"}), use_container_width=True)

    
    with tab3:
        st.subheader("🎯 Hệ thống Gợi ý Sản phẩm (Collaborative Filtering)")
        st.markdown("Thuật toán **Đại số Tuyến tính (Cosine Similarity)** phân tích hành vi của những khách hàng tương đồng để tìm ra 5 sản phẩm mà khách hàng này có khả năng sẽ mua nhất.")
        
        # Gợi ý một vài Customer ID có thật để người dùng dễ test
        sample_ids = df_raw['CustomerID'].dropna().unique()[:5]
        st.caption(f"💡 Ví dụ một vài ID khách hàng hợp lệ: {', '.join([str(int(i)) for i in sample_ids])}")
        
        cust_id_input = st.number_input("Nhập Mã Khách hàng (Customer ID):", min_value=12000, value=int(sample_ids[0]), step=1)
        
        if st.button("🎁 TÌM SẢN PHẨM GỢI Ý"):
            with st.spinner("Đang quét ma trận ma trận người dùng (User-Item Matrix)..."):
                recs = get_recommendations(cust_id_input, df_raw, top_n=5)
                
                if recs is None:
                    st.error("❌ Không tìm thấy mã khách hàng này trong hệ thống!")
                elif not recs:
                    st.warning("⚠️ Khách hàng này đã mua hầu hết các sản phẩm phổ biến, chưa có gợi ý mới.")
                else:
                    st.success(f"Khách hàng **{cust_id_input}** có thể sẽ thích 5 sản phẩm sau:")
                    for i, item in enumerate(recs, 1):
                        st.markdown(f"**{i}.** {item}")

if __name__ == '__main__':
    main()
