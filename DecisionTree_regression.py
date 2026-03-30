import pandas as pd
import numpy as np
import json


# Hàm tính Mean Squared Error (MSE) cho toàn bộ dữ liệu
def calculate_mse(target_col):
    if len(target_col) == 0:
        return 0
    mean_val = target_col.mean()
    mse = np.mean((target_col - mean_val) ** 2)
    return mse


# Hàm tìm ngưỡng phân chia tốt nhất cho một đặc trưng liên tục
def find_best_split(data, feature_name, target_name):
    unique_values = data[feature_name].unique()
    unique_values.sort()

    # Tạo các điểm giữa các giá trị duy nhất làm ngưỡng
    thresholds = [(unique_values[i] + unique_values[i + 1]) / 2 for i in range(len(unique_values) - 1)]

    best_mse_reduction = -float('inf')
    best_threshold = None

    if len(thresholds) == 0:
        return None, 0  # Không thể phân chia thêm

    total_mse = calculate_mse(data[target_name])

    for threshold in thresholds:
        left_subset = data[data[feature_name] <= threshold]
        right_subset = data[data[feature_name] > threshold]

        if len(left_subset) == 0 or len(right_subset) == 0:
            continue

        weighted_mse = (len(left_subset) / len(data)) * calculate_mse(left_subset[target_name]) + \
                       (len(right_subset) / len(data)) * calculate_mse(right_subset[target_name])

        mse_reduction = total_mse - weighted_mse

        if mse_reduction > best_mse_reduction:
            best_mse_reduction = mse_reduction
            best_threshold = threshold

    return best_threshold, best_mse_reduction


# Hàm huấn luyện cây hồi quy
def decision_tree_regression_custom(data, features, target_col, current_depth=0, max_depth=2, min_samples_split=2):
    # Điều kiện dừng 1: Không còn đặc trưng hoặc ít dữ liệu
    if len(features) == 0 or len(data) < min_samples_split:
        return data[target_col].mean()

    # Điều kiện dừng 2: Đạt độ sâu tối đa
    if current_depth >= max_depth:
        return data[target_col].mean()

    # In quá trình tính toán ở Node gốc (để bạn dễ đối chiếu với lý thuyết)
    if current_depth == 0:
        total_mse = calculate_mse(data[target_col])
        print("--- Thông số tại Node Gốc ---")
        print(f"MSE(S) = {total_mse:.4f}")
        for f in features:
            thresh, reduc = find_best_split(data, f, target_col)
            if thresh is not None:
                print(f"ΔMSE(S, {f}, ngưỡng={thresh:.1f}) = {reduc:.4f}")
        print("-" * 40)

    # Chọn đặc trưng có mức giảm MSE cao nhất
    best_feature = None
    best_threshold = None
    best_mse_reduction = -float('inf')

    for feature in features:
        threshold, mse_reduction = find_best_split(data, feature, target_col)
        if threshold is not None and mse_reduction > best_mse_reduction:
            best_mse_reduction = mse_reduction
            best_feature = feature
            best_threshold = threshold

    # Điều kiện dừng 3: Không tìm thấy phân chia nào giúp giảm MSE
    if best_feature is None or best_mse_reduction <= 0:
        return data[target_col].mean()

    # Phân nhánh nhị phân
    tree = {f"{best_feature} <= {best_threshold:.1f}": {}}
    left_subset = data[data[best_feature] <= best_threshold]
    right_subset = data[data[best_feature] > best_threshold]

    # Gọi đệ quy cho 2 nhánh (Đúng/Sai với điều kiện ngưỡng)
    tree[f"{best_feature} <= {best_threshold:.1f}"]["True"] = decision_tree_regression_custom(
        left_subset, features, target_col, current_depth + 1, max_depth, min_samples_split
    )
    tree[f"{best_feature} <= {best_threshold:.1f}"]["False"] = decision_tree_regression_custom(
        right_subset, features, target_col, current_depth + 1, max_depth, min_samples_split
    )

    return tree


# Hàm dự đoán
def predict_regression(tree, sample):
    if not isinstance(tree, dict):
        return tree

    node_condition = next(iter(tree))
    # Phân tích điều kiện phân chia (VD: 'dien_tich <= 55.0')
    parts = node_condition.split(' ')
    feature_name = parts[0]
    threshold = float(parts[2])

    feature_value = sample.get(feature_name)

    # Đi xuống nhánh tương ứng
    if feature_value <= threshold:
        return predict_regression(tree[node_condition]["True"], sample)
    else:
        return predict_regression(tree[node_condition]["False"], sample)


if __name__ == "__main__":
    df_train = pd.read_csv('database/nha_dat.csv')
    target_column = 'gia'
    feature_columns = ['dien_tich', 'so_phong']
    my_tree = decision_tree_regression_custom(df_train, feature_columns, target_column, max_depth=2,
                                              min_samples_split=2)

    print("=> Cấu trúc Cây Hồi Quy cuối cùng:")
    print(json.dumps(my_tree, indent=4, ensure_ascii=False))
    print("=" * 40)

    # --- DỰ ĐOÁN ---
    nha_moi = {'dien_tich': 55, 'so_phong': 2}
    gia_du_doan = predict_regression(my_tree, nha_moi)
    print(f"Nhà mới: {nha_moi}")
    print(f"-> Giá dự đoán: {gia_du_doan:.2f} tỷ VNĐ")