import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU
df_raw = pd.read_csv("database/insurance.csv")

# Chọn các cột rời rạc để hàm tính MSE Reduction
# sex: Giới tính, smoker: Hút thuốc, children: Số con, charges: Chi phí (Mục tiêu)
df = df_raw[['sex', 'smoker', 'children', 'charges']].dropna()

# Chuyển đổi dữ liệu chữ sang số (Encoding)
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

X = df.drop("charges", axis=1)
y = df["charges"]

# Tính Variance (Phương sai) - Tương đương với MSE của một node
def calculate_node_mse(y):
    if len(y) == 0:
        return 0.0
    return np.var(y)  # np.var tính toán trung bình của bình phương độ lệch

# Tính mức giảm MSE
def mse_reduction(X, y, feature):
    total_mse = calculate_node_mse(y)
    values = X[feature].unique()
    weighted_mse = 0.0

    # Tính tổng MSE có trọng số của các nhánh con
    for v in values:
        sub_y = y[X[feature] == v]
        weighted_mse += (len(sub_y) / len(y)) * calculate_node_mse(sub_y)

    return total_mse - weighted_mse

# 3. PHÂN TÍCH VÀ HUẤN LUYỆN MÔ HÌNH
print("=== DECISION TREE REGRESSION (Scikit-learn) ===")
print(f"Tổng số mẫu: {len(df)}")
root_mse = calculate_node_mse(y)
print(f"MSE gốc của toàn bộ tập dữ liệu: {root_mse:,.2f}\n")

print("Mức giảm MSE (MSE Reduction) của từng feature:")
reductions = {}
for feature in X.columns:
    reduc = mse_reduction(X, y, feature)
    reductions[feature] = reduc
    print(f"  - {feature}: Giảm = {reduc:,.2f}")

best_feat = max(reductions, key=reductions.get)
print(f"\n→ Feature phân nhánh tốt nhất ở node gốc: {best_feat} (Giảm được {reductions[best_feat]:,.2f} MSE)\n")

# Khởi tạo mô hình Hồi quy của Scikit-learn
regressor = DecisionTreeRegressor(
    criterion='squared_error',  # Sử dụng MSE làm tiêu chí tách nhánh
    max_depth=3,
    min_samples_split=20,
    random_state=42
)
regressor.fit(X, y)

print(f"Độ sâu cây: {regressor.get_depth()}")
print(f"Số node lá: {regressor.get_n_leaves()}\n")

# 4. DỰ ĐOÁN THỬ NGHIỆM VỚI DỮ LIỆU MỚI
print("=== DỰ ĐOÁN CHI PHÍ BẢO HIỂM MỚI ===")
test_cases = [
    {'sex': 0, 'smoker': 1, 'children': 0, 'Mô tả': 'Nam, hút thuốc, chưa có con'},
    {'sex': 1, 'smoker': 0, 'children': 2, 'Mô tả': 'Nữ, KHÔNG hút thuốc, 2 con'},
    {'sex': 0, 'smoker': 0, 'children': 0, 'Mô tả': 'Nam, KHÔNG hút thuốc, chưa có con'}
]
test_df = pd.DataFrame(test_cases).drop('Mô tả', axis=1)
predictions = regressor.predict(test_df)

for i, case in enumerate(test_cases):
    print(f"  Khách hàng {i + 1} ({case['Mô tả']}) → Dự đoán phí: {predictions[i]:,.2f} USD")

# 5. ĐÁNH GIÁ MÔ HÌNH
print("\n=== FEATURE IMPORTANCE ===")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': regressor.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance.to_string(index=False))

y_pred = regressor.predict(X)
r2 = r2_score(y, y_pred)
print(f"\nHệ số R-squared trên toàn bộ dataset: {r2:.4f} (Càng gần 1.0 càng tốt)")