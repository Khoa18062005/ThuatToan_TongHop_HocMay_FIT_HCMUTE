import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))

def custom_knn(X_train, y_train, test_point, k=3, metric='euclidean', task='classification'):
    distances = []

    # Bước 1: Tính khoảng cách từ điểm test đến tất cả các điểm trong tập train
    for index in range(len(X_train)):
        train_point = X_train.iloc[index]  # Lấy ra từng hàng dữ liệu

        if metric == 'euclidean':
            dist = euclidean_distance(test_point, train_point)
        elif metric == 'manhattan':
            dist = manhattan_distance(test_point, train_point)
        else:
            raise ValueError("Chỉ hỗ trợ 'euclidean' hoặc 'manhattan'")

        # Lưu lại khoảng cách và nhãn tương ứng của điểm đó
        label = y_train.iloc[index]
        distances.append((dist, label))

    # Sắp xếp danh sách theo khoảng cách từ nhỏ đến lớn (gần nhất lên đầu)
    distances.sort(key=lambda x: x[0])

    # Lấy ra k láng giềng gần nhất
    k_nearest = distances[:k]

    # Lấy ra danh sách các nhãn của k láng giềng
    k_nearest_labels = [label for dist, label in k_nearest]

    # Đưa ra quyết định (Dự đoán)
    if task == 'classification':
        # Phân loại: Lấy giá trị xuất hiện nhiều nhất (Majority Vote)
        prediction = max(set(k_nearest_labels), key=k_nearest_labels.count)
    elif task == 'regression':
        # Hồi quy: Lấy trung bình cộng (Average)
        prediction = np.mean(k_nearest_labels)

    return prediction, distances
if __name__ == '__main__':
    # Nếu bạn có file CSV thật, hãy đổi dòng dưới thành: df = pd.read_csv('ten_file_cua_ban.csv')
    df = pd.read_csv('database/database_KNN_01.csv')

    # Định nghĩa các cột
    features_cols = ['Giờ học', 'Giờ chơi']  # Các cột đặc trưng (đầu vào)
    classification_col = 'Kết quả'  # Cột nhãn cho bài toán Phân loại
    regression_col = 'Điểm thi'  # Cột nhãn cho bài toán Hồi quy

    X_train = df[features_cols]
    y_train_class = df[classification_col]
    y_train_reg = df[regression_col]

    # Dữ liệu mới cần dự đoán (HS06)
    # Giờ học = 5, Giờ chơi = 4
    test_point = pd.Series([12, 12], index=features_cols)
    k_value = 3

    # Huấn luyện mô hình
    pred_class, all_dist_class = custom_knn(
        X_train, y_train_class, test_point,
        k=k_value, metric='euclidean', task='classification'
    )
    pred_reg, all_dist_reg = custom_knn(
        X_train, y_train_reg, test_point,
        k=k_value, metric='euclidean', task='regression'
    )
    print(f"\n=> CẦN DỰ ĐOÁN CHO HS06: Giờ học={test_point['Giờ học']}, Giờ chơi={test_point['Giờ chơi']}")
    print("=> Khoảng cách đến các điểm:\n" + "\n".join([f"   - {df['HS'][i]}: {round(d, 2)}" for i, (d, l) in enumerate(all_dist_class)]))
    print(f"=> Kết quả dự đoán phân loại: {pred_class}")
    print(f"=> Điểm thi dự đoán hồi quy: {round(pred_reg, 2)}")

    # Đánh gía mô hình
