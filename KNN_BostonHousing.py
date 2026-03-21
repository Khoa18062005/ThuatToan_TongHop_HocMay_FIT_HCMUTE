import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))


def custom_knn(X_train, y_train, test_point, k=3, metric='euclidean', task='classification'):
    distances = []

    # Tính khoảng cách từ test_point đến toàn bộ X_train
    for index in range(len(X_train)):
        train_point = X_train.iloc[index]

        if metric == 'euclidean':
            dist = euclidean_distance(test_point, train_point)
        elif metric == 'manhattan':
            dist = manhattan_distance(test_point, train_point)
        else:
            raise ValueError("Chỉ hỗ trợ 'euclidean' hoặc 'manhattan'")

        label = y_train.iloc[index]
        distances.append((dist, label))

    # Sắp xếp để tìm k láng giềng gần nhất
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    k_nearest_labels = [label for dist, label in k_nearest]

    if task == 'classification':
        prediction = max(set(k_nearest_labels), key=k_nearest_labels.count)
    elif task == 'regression':
        prediction = np.mean(k_nearest_labels)

    return prediction, distances


# Hàm vẽ đồ thị minh hoạ kết quả huấn luyện
def plot_knn_results(X_train, y_train, features_to_plot, test_point=None, test_prediction=None):
    plt.figure(figsize=(9, 6))

    unique_classes = y_train.unique()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Thêm màu đề phòng nhiều lớp

    # Chỉ lấy 2 đặc trưng để vẽ trên đồ thị 2D
    f1, f2 = features_to_plot[0], features_to_plot[1]

    # Vẽ các điểm trong tập Train
    for i, cls in enumerate(unique_classes):
        idx = y_train == cls
        plt.scatter(X_train.loc[idx, f1], X_train.loc[idx, f2],
                    color=colors[i % len(colors)], label=f'Lớp: {cls}', alpha=0.7)

    # Vẽ thêm điểm Test lên để dễ hình dung
    if test_point is not None and test_prediction is not None:
        plt.scatter(test_point[f1], test_point[f2],
                    color='red', marker='*', s=300, edgecolors='black',
                    label=f'Điểm mới (Dự đoán: {test_prediction})')

    plt.xlabel(f1)
    plt.ylabel(f2)
    # Sửa lại tiêu đề để tự động nhận tên cột thay vì fix cứng chữ SepalLength của hoa Iris
    plt.title(f'Minh hoạ phân loại KNN (dựa trên 2 đặc trưng: {f1} & {f2})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('database/BostonHousing.csv')

    # Loại bỏ cột ID nếu có
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)

    # -- BƯỚC CHUYỂN ĐỔI: TỪ HỒI QUY SANG PHÂN LOẠI --
    # Phân loại cột giá nhà (medv) thành 2 nhóm 'Cao' và 'Thấp' dựa vào giá trị trung vị
    trung_vi_gia = df['medv'].median()
    df['Phan_Loai_Gia'] = np.where(df['medv'] >= trung_vi_gia, 'Cao', 'Thấp')

    # Chọn 4 đặc trưng đại diện: Số phòng (rm), Tuổi nhà (age), Khoảng cách đến trung tâm (dis), Tỷ lệ thu nhập thấp (lstat)
    features_cols = ['rm', 'age', 'dis', 'lstat']
    classification_col = 'Phan_Loai_Gia'

    X = df[features_cols]
    y = df[classification_col]
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    print(f"Kích thước tập Train     : {len(X_train)} mẫu")
    print(f"Kích thước tập Validation: {len(X_val)} mẫu")
    print(f"Kích thước tập Test      : {len(X_test)} mẫu")
    print("=" * 45)

    k_value = 3

    # 3. DỰ ĐOÁN THỬ 1 ĐIỂM (Lấy dòng đầu tiên của tập Test làm mẫu)
    sample_point = X_test.iloc[0]
    actual_label = y_test.iloc[0]

    pred_class, _ = custom_knn(
        X_train, y_train, sample_point,
        k=k_value, metric='euclidean', task='classification'
    )

    print("=> THỬ NGHIỆM DỰ ĐOÁN 1 MẪU TỪ TẬP TEST:")
    print(sample_point.to_string())
    print(f"=> Nhãn thực tế: {actual_label}")
    print(f"=> Kết quả dự đoán từ KNN: {pred_class}")
    print("=" * 45)

    # Dự đoán trên một mẫu dữ liệu mới (ví dụ thông số của một căn nhà mới thu thập được)
    # Tương ứng các thông số: [rm (số phòng), age (tuổi nhà), dis (khoảng cách), lstat (% thu nhập thấp)]
    new_house_data = [6.5, 45.0, 4.2, 12.5]
    new_point = pd.Series(new_house_data, index=features_cols)
    new_pred_class, nearest_neighbors = custom_knn(
        X_train, y_train, new_point,
        k=k_value, metric='euclidean', task='classification'
    )

    print("Thông số căn nhà mới thu thập được:")
    print(new_point.to_string())
    print(f"-> MÔ HÌNH DỰ ĐOÁN GIÁ NHÀ THUỘC LỚP: {new_pred_class.upper()}")

    # ĐÁNH GIÁ MÔ HÌNH
    print("=" * 45)
    print("ĐÁNH GIÁ ACCURACY (BÀI TOÁN PHÂN LOẠI)")

    # Dự đoán trên toàn bộ tập Train
    y_train_pred = [custom_knn(X_train, y_train, X_train.iloc[i], k=k_value)[0] for i in range(len(X_train))]
    acc_train = accuracy_score(y_train, y_train_pred)
    print(f"=> Độ chính xác trên tập Train: {acc_train * 100:.2f}%")

    # Dự đoán trên toàn bộ tập Test
    y_test_pred = [custom_knn(X_train, y_train, X_test.iloc[i], k=k_value)[0] for i in range(len(X_test))]
    acc_test = accuracy_score(y_test, y_test_pred)
    print(f"=> Độ chính xác trên tập Test:  {acc_test * 100:.2f}%")
    print("=" * 45)

    # VẼ ĐỒ THỊ MINH HOẠ KẾT QUẢ HUẤN LUYỆN
    # Mình chọn số phòng (rm) và % tỷ lệ người thu nhập thấp (lstat) để lên đồ thị vì 2 chỉ số này ảnh hưởng rõ nhất đến giá
    features_to_plot = ['rm', 'lstat']
    plot_knn_results(X_train, y_train, features_to_plot, test_point=sample_point, test_prediction=pred_class)