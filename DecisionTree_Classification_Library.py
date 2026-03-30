import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score

# 1. Đọc dữ liệu từ file có sẵn của bạn
df_raw = pd.read_csv("database/Titanic.csv")

# Tiền xử lý nhanh: Chọn các cột rời rạc để hàm tính Information Gain tự viết chạy chuẩn xác
# Lọc bỏ các dòng chứa giá trị rỗng (NaN)
df = df_raw[['Pclass', 'Sex', 'SibSp', 'Parch', 'Survived']].dropna()

# Chuyển đổi Giới tính từ chữ sang số (male: 0, female: 1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# 2. Tách X (đặc trưng) và y (nhãn dự đoán)
X = df.drop("Survived", axis=1)
y = df["Survived"]

def entropy(y):
    counter = Counter(y)
    total = len(y)
    ent = 0.0
    for c in counter.values():
        p = c / total
        if p > 0:
            ent -= p * np.log2(p)
    return ent

def information_gain(X, y, feature):
    total_entropy = entropy(y)
    values = X[feature].unique()
    weighted_entropy = 0.0
    for v in values:
        sub_y = y[X[feature] == v]
        weighted_entropy += (len(sub_y) / len(y)) * entropy(sub_y)
    return total_entropy - weighted_entropy

print("=== DECISION TREE (Scikit-learn) ===")
print(f"Tổng số mẫu: {len(df)}")
root_entropy = entropy(y)
print(f"Entropy gốc: {root_entropy:.4f}")
print()

print("Information Gain của từng feature:")
gains = {}
for feature in X.columns:
    g = information_gain(X, y, feature)
    gains[feature] = g
    print(f"  - {feature}: Gain = {g:.4f}")

best_feat = max(gains, key=gains.get)
print(f"\n→ Feature tốt nhất: {best_feat} (Gain = {gains[best_feat]:.4f})\n")

# 3. Huấn luyện mô hình
clf = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3,
    min_samples_split=20,
    random_state=42
)
clf.fit(X, y)

print(f"Độ sâu cây: {clf.get_depth()}")
print(f"Số node lá: {clf.get_n_leaves()}")
print()

# 4. Dự đoán thử nghiệm với dữ liệu mới
print("Dự đoán theo decision tree:")
test_cases = [
    {'Pclass': 3, 'Sex': 0, 'SibSp': 0, 'Parch': 0}, # Nam, hạng vé 3, đi 1 mình
    {'Pclass': 1, 'Sex': 1, 'SibSp': 1, 'Parch': 0}, # Nữ, hạng vé 1, có 1 người đi cùng
    {'Pclass': 2, 'Sex': 0, 'SibSp': 2, 'Parch': 1}  # Nam, hạng vé 2, đi cùng gia đình
]
test_df = pd.DataFrame(test_cases)
predictions = clf.predict(test_df)

for i, case in enumerate(test_cases, 1):
    # 0 = Không sống sót, 1 = Sống sót
    print(f"  Hành khách {i} (Pclass={case['Pclass']}, Sex={case['Sex']}) → Dự đoán: {predictions[i-1]}")

# 5. Đánh giá tính quan trọng của đặc trưng và độ chính xác
print("\n=== FEATURE IMPORTANCE ===")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': clf.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance.to_string(index=False))

y_pred = clf.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"\nĐộ chính xác trên toàn bộ dataset: {accuracy:.4f}")