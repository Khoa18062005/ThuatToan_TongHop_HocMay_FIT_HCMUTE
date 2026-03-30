import pandas as pd
import numpy as np
import math
# Tính toán giá trị Entropy của một tập dữ liệu
def calculate_entropy(target_col):
    counts = target_col.value_counts()
    entropy = 0
    total = len(target_col)
    for count in counts:
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

# Tính Information Gain (IG) cho một đặc trưng
def calculate_information_gain(data, feature_name, target_name):
    # Entropy gốc của tập S
    total_entropy = calculate_entropy(data[target_name])

    # Tính tổng Entropy có trọng số của các nhánh con
    values, counts = zip(*data[feature_name].value_counts().items())
    weighted_entropy = 0
    for i in range(len(values)):
        subset = data[data[feature_name] == values[i]]
        p_subset = counts[i] / sum(counts)
        weighted_entropy += p_subset * calculate_entropy(subset[target_name])

    # Information Gain
    ig = total_entropy - weighted_entropy
    return round(ig, 3)


def decision_tree_classification_custom(data, features, target_col):
    # 1. Nếu tất cả dữ liệu cùng thuộc 1 nhãn -> Trả về nhãn đó (Leaf Node)
    if len(data[target_col].unique()) == 1:
        return data[target_col].iloc[0]

    # 2. Nếu không còn đặc trưng nào để chia -> Trả về nhãn chiếm đa số
    if len(features) == 0:
        return data[target_col].value_counts().idxmax()

    # 3. Chọn đặc trưng có Information Gain cao nhất để phân nhánh
    ig_values = {}
    for feature in features:
        ig = calculate_information_gain(data, feature, target_col)
        ig_values[feature] = ig

    # In ra giá trị IG để kiểm chứng giống slide (chỉ in ở node gốc có 5 dòng)
    if len(data) == 5:
        print("--- Tính toán Information Gain ở Bước 2 ---")
        for f, v in ig_values.items():
            print(f"IG(S, {f}) = {v}")
        print("-" * 40)

    # Tìm đặc trưng có IG lớn nhất (Node)
    best_feature = max(ig_values, key=ig_values.get)

    # Khởi tạo cây với Node hiện tại
    tree = {best_feature: {}}

    # Xóa đặc trưng đã chọn khỏi danh sách để không phân nhánh lại
    remaining_features = [f for f in features if f != best_feature]

    # Phân nhánh cho từng giá trị của đặc trưng tốt nhất
    # Lưu ý: lấy unique values từ tập dữ liệu đầy đủ ban đầu để tránh thiếu nhánh
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]

        if len(subset) == 0:
            # Nếu nhánh rỗng, trả về nhãn đa số của tập cha
            tree[best_feature][value] = data[target_col].value_counts().idxmax()
        else:
            # Gọi đệ quy để xây nhánh con
            tree[best_feature][value] = decision_tree_classification_custom(
                subset, remaining_features, target_col
            )

    return tree

# Hàm dự đoán cho dữ liệu mới
def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree

    root_node = next(iter(tree))
    feature_value = sample.get(root_node)

    if feature_value in tree[root_node]:
        return predict(tree[root_node][feature_value], sample)
    else:
        return "Không xác định"

if __name__ == "__main__":
    import pandas as pd  # Đảm bảo đã import pandas ở đầu file

    # 1. Đọc dữ liệu huấn luyện từ file CSV
    df_train = pd.read_csv('database/DecisionTree_Sample.csv')
    target_column = 'Duyệt vay'
    feature_columns = ['Thu nhập', 'Lịch sử tín dụng', 'Công việc ổn định']
    my_tree = decision_tree_classification_custom(df_train, feature_columns, target_column)

    # In cây quyết định ra màn hình
    print("=> Cấu trúc Cây Quyết Định cuối cùng:")
    print(my_tree)
    print("=" * 40)

    # Thêm người cần dự đoán
    nguoi_thu_7_list = ['Cao', 'Xấu', 'Không']
    khach_hang_moi = dict(zip(feature_columns, nguoi_thu_7_list))
    # Tiến hành dự đoán
    ket_qua = predict(my_tree, khach_hang_moi)

    print(f"Dữ liệu Khách hàng mới (từ list): {nguoi_thu_7_list}")
    print(f"Dữ liệu đưa vào dự đoán: {khach_hang_moi}")
    print(f"-> Kết quả dự đoán 'Duyệt vay' cho người này là: {ket_qua}")