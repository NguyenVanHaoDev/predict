import streamlit as st
import os
from app.model_utils import load_model, load_scaler
from app.main import main_bp


# Hàm dự đoán
def predict_species(features):
    model = load_model()
    scaler = load_scaler()


    if model is None or scaler is None:
        return "Lỗi khi tải mô hình hoặc scaler."

    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)

    # Giả sử bạn dùng factor hóa kiểu như ['setosa', 'versicolor', 'virginica']
    definitions = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    return definitions[prediction[0]]

# Đường dẫn thư mục chứa ảnh, lấy từ thư mục hiện tại + thư mục images
# IMAGE_DIR = os.path.join(os.path.dirname(__file__), 'images')
IMAGE_DIR = os.path.join(main_bp.root_path, main_bp.static_folder, 'images')


# Dictionary lưu link ảnh tương ứng với loại hoa
image_paths = {
    'Iris-setosa': os.path.join(IMAGE_DIR, 'setosa.jpg'),
    'Iris-versicolor': os.path.join(IMAGE_DIR, 'versicolor.jpg'),
    'Iris-virginica': os.path.join(IMAGE_DIR, 'virginica.jpg'),
}

st.set_page_config(
    page_title="Dự đoán Iris",
    page_icon="🔍",
    layout="wide"
)
st.title("🔍 Trang Dự đoán Iris")

st.markdown(
    """
    <style>
    .main {
        background-color: #F9F9F9;
        font-family: 'Segoe UI', sans-serif;
        padding: 20px;
    }
    .stSlider > div {
        color: #3D3D3D;
    }
    .title {
        color: #4F8BF9;
        font-size: 36px;
        font-weight: bold;
    }
    img {
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# st.markdown('<div class="title">', unsafe_allow_html=True)
# st.title("🌸 Chào mừng bạn đến với ứng dụng Dự đoán")
# st.markdown("</div>", unsafe_allow_html=True)
st.markdown("""
    <h1 style='text-align: center; color: #4F8BF9;'>
        🌸 Chào mừng bạn đến với ứng dụng Dự đoán
    </h1>
""", unsafe_allow_html=True)


with st.container():
    st.subheader("🌿 Nhập thông số đặc trưng của hoa Iris:")
    col1, spacer, col2 = st.columns([1, 0.2, 1])
    with col1:
        sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.9)
        sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
        petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 5.1)
        petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.8)
        if st.button("🌼 Dự đoán ngay"):
            features = [sepal_length, sepal_width, petal_length, petal_width]
            result = predict_species(features)

            if "Lỗi" in result:
                st.error(result)
            else:
                with col2:
                    st.markdown("""
                                    <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100%;">
                                """, unsafe_allow_html=True)
                    st.success(f"🌺 Kết quả dự đoán: **{result}**")
                    # Hiển thị ảnh tương ứng
                    st.image(image_paths[result], caption=result, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

        else:
            with col2:
                st.markdown("""
                        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100%;">
                    """, unsafe_allow_html=True)
                st.write("Ảnh loài hoa sẽ hiển thị ở đây khi bạn dự đoán.")
                default_image = os.path.join(IMAGE_DIR, 'default.jpg')
                if os.path.exists(default_image):
                    st.image(default_image, caption="Ảnh minh họa loài hoa", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)



