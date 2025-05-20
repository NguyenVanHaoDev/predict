import os
import pickle
from app.main import main_bp

# Đường dẫn thư mục chứa mô hình và scaler
MODEL_DIR = os.path.join(main_bp.root_path, main_bp.static_folder,'PKL')
MODEL_FILE = os.path.join(main_bp.root_path, main_bp.static_folder,'PKL', 'randomforestmodel.pkl')
SCALER_FILE = os.path.join(main_bp.root_path, main_bp.static_folder,'PKL', 'scaler.pkl')


def save_model(model, scaler):
    """
    Lưu mô hình và scaler vào thư mục MODEL_DIR

    Tham số:
    - model: classifier đã train (ví dụ RandomForestClassifier)
    - scaler: scaler đã fit (ví dụ StandardScaler)

    Tạo thư mục nếu chưa có.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        with open(MODEL_FILE, 'wb') as f_model:
            pickle.dump(model, f_model)
        print(f"Đã lưu mô hình vào {MODEL_FILE}")
    except Exception as e:
        print(f"Lỗi khi lưu mô hình: {e}")

    try:
        with open(SCALER_FILE, 'wb') as f_scaler:
            pickle.dump(scaler, f_scaler)
        print(f"Đã lưu scaler vào {SCALER_FILE}")
    except Exception as e:
        print(f"Lỗi khi lưu scaler: {e}")


def load_model():
    """
    Load mô hình từ file pickle.

    Trả về:
    - model đã load, hoặc None nếu lỗi
    """
    if not os.path.exists(MODEL_FILE):
        print(f"File mô hình không tồn tại: {MODEL_FILE}")
        return None
    try:
        with open(MODEL_FILE, 'rb') as f_model:
            model = pickle.load(f_model)
        return model
    except Exception as e:
        print(f"Lỗi khi load mô hình: {e}")
        return None


def load_scaler():
    """
    Load scaler từ file pickle.

    Trả về:
    - scaler đã load, hoặc None nếu lỗi
    """
    if not os.path.exists(SCALER_FILE):
        print(f"File scaler không tồn tại: {SCALER_FILE}")
        return None
    try:
        with open(SCALER_FILE, 'rb') as f_scaler:
            scaler = pickle.load(f_scaler)
        return scaler
    except Exception as e:
        print(f"Lỗi khi load scaler: {e}")
        return None


# if __name__ == "__main__":
#     # Ví dụ sử dụng, test khi chạy file này trực tiếp
#     from sklearn.ensemble import RandomForestClassifier
#     from sklearn.preprocessing import StandardScaler
#     import numpy as np
#
#     # Tạo model và scaler giả lập (ví dụ)
#     model = RandomForestClassifier()
#     scaler = StandardScaler()
#
#     # Giả lập train model và scaler để test lưu
#     X_train = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
#     y_train = np.array([0, 1])
#     scaler.fit(X_train)
#     X_scaled = scaler.transform(X_train)
#     model.fit(X_scaled, y_train)
#
#     # Lưu model và scaler
#     save_model(model, scaler)
#
#     # Load lại
#     loaded_model = load_model()
#     loaded_scaler = load_scaler()
#
#     # Kiểm tra dự đoán
#     if loaded_model and loaded_scaler:
#         sample = np.array([[2, 3, 4, 5]])
#         sample_scaled = loaded_scaler.transform(sample)
#         print("Dự đoán mẫu:", loaded_model.predict(sample_scaled))
