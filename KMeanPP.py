import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
# 1. Hàm dự đoán cụm (gán nhãn)
def predict_clusters(X, centroids):
    X_arr = np.array(X)
    distances = np.sqrt(((X_arr - centroids[:, np.newaxis]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=0)

def KMeansPP_custom(X_train, k=3, max_iters=100):
    print("=" * 50)
    print(f"Tham số k được truyền vào là: {k}")
    print("=" * 50)

    X = np.array(X_train)
    n_samples = X.shape[0]
    centroids = []

    # Chọn ngẫu nhiên 1 tâm đầu tiên
    first_idx = np.random.choice(n_samples)
    centroids.append(X[first_idx])
    print(f"[Khởi tạo K-Means++] Tâm số 1 (chọn ngẫu nhiên): {centroids[0]}")

    # Vòng lặp tìm các tâm còn lại
    for c_id in range(1, k):
        distances_sq = np.array([min([np.sum((c - x) ** 2) for c in centroids]) for x in X])
        sum_distances_sq = np.sum(distances_sq)
        print(f"\n=> Tổng bình phương khoảng cách đến tâm cụm (trước khi chọn tâm {c_id + 1}): {sum_distances_sq:.2f}")

        probabilities = distances_sq / sum_distances_sq

        top_5_idx = np.argsort(probabilities)[-5:][::-1]
        print("=> Top 5 điểm có xác suất được chọn làm tâm tiếp theo cao nhất:")
        for rank, idx in enumerate(top_5_idx):
            print(f"   Top {rank + 1} | Điểm {X[idx]} | Xác suất = {probabilities[idx]:.6f}")

        next_idx = np.random.choice(n_samples, p=probabilities)
        centroids.append(X[next_idx])
        print(f"=> [Kết luận] Tâm cụm số {c_id + 1} được chọn: {centroids[-1]}")

    centroids = np.array(centroids)
    print("\n--- Đã tìm đủ các tâm. Bắt đầu các bước K-Means phổ thông ---")

    # Các bước K-Means phổ thông
    for i in range(max_iters):
        labels = predict_clusters(X, centroids)
        new_centroids = np.array(
            [X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j] for j in range(k)])

        if np.all(centroids == new_centroids):
            print(f"Thuật toán đã hội tụ thành công tại vòng lặp thứ {i + 1}!\n")
            break

        centroids = new_centroids

    return centroids, labels


# 3. Hàm vẽ biểu đồ minh họa
def plot_kmeans_results(X_data, labels, centroids, k_value, feature_names, dataset_name="Test"):
    plt.figure(figsize=(10, 6))

    # Chuyển đổi dữ liệu sang numpy array để dễ index
    X_arr = np.array(X_data)

    # Vẽ các điểm dữ liệu
    plt.scatter(X_arr[:, 0], X_arr[:, 1], c=labels, cmap='rainbow', s=50, alpha=0.7, label=f'Dữ liệu {dataset_name}')

    # Vẽ các tâm cụm (Centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Tâm cụm (Centroids)')

    plt.title(f'K-Means++ Clustering (k={k_value}) trên tập {dataset_name}', fontsize=14)
    plt.xlabel(feature_names[0], fontsize=12)
    plt.ylabel(feature_names[1], fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('database/Mall_Customers.csv')
    features_cols = ['Annual Income (k$)', 'Spending Score (1-100)']
    X_full = df[features_cols]

    # 2. Chia dữ liệu
    X_train_val, X_test = train_test_split(X_full, test_size=0.2, random_state=42)
    X_train, X_val = train_test_split(X_train_val, test_size=0.1, random_state=42)

    print(f"Tổng số mẫu: {len(X_full)}")
    print(f"- Tập Train: {len(X_train)} mẫu")
    print(f"- Tập Validation: {len(X_val)} mẫu")
    print(f"- Tập Test: {len(X_test)} mẫu\n")

    # 3. Huấn luyện mô hình
    k_value = 5
    trained_centroids, train_labels = KMeansPP_custom(X_train, k=k_value)

    # 4. Đánh giá hiệu suất
    val_labels = predict_clusters(X_val, trained_centroids)
    test_labels = predict_clusters(X_test, trained_centroids)

    score_train = silhouette_score(X_train, train_labels) if len(set(train_labels)) > 1 else 0
    score_val = silhouette_score(X_val, val_labels) if len(set(val_labels)) > 1 else 0
    score_test = silhouette_score(X_test, test_labels) if len(set(test_labels)) > 1 else 0

    print("=" * 50)
    print("ĐÁNH GIÁ HIỆU SUẤT (SILHOUETTE SCORE)")
    print(f"Train Score: {score_train:.4f}")
    print(f"Validation Score: {score_val:.4f}")
    print(f"Test Score: {score_test:.4f}")
    print("=" * 50)

    # 5. Vẽ biểu đồ (Gọi hàm vừa tách)
    # Bạn có thể thử đổi X_test thành X_train và test_labels thành train_labels để xem biểu đồ tập Train nhé
    plot_kmeans_results(X_test, test_labels, trained_centroids, k_value, features_cols, dataset_name="Test")