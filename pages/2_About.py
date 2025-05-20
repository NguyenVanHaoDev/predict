import streamlit as st


# Giao diện
st.set_page_config(
    page_title="Giới thiêu",
    page_icon="ℹ️",
    layout="wide"
)
st.title("ℹ️ Giới thiệu")
st.markdown("""
Ứng dụng này sử dụng **Machine Learning** để phân loại loài hoa Iris.

👨‍💻 Tác giả: Nguyễn Văn Hảo  
📅 Năm: 2025  
🧠 Mô hình: SVM (Scikit-learn)
""")
