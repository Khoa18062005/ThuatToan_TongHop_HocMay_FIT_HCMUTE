import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Thư viện để huấn luyện Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

def likelihood(x, mean, std):
    tmp = 1/np.sqrt(2 * np.pi * (std**2)) * np.exp(-(x - mean)**2 / (2 * std**2))
    return tmp


def predict(X_samples, model, labels):
    print("\n" + "=" * 145)
    print(
        f"{'Mẫu':<5} | {'P(x|0) (Likelihoods)':<40} | {'P(0|x)':<12} | {'P(x|1) (Likelihoods)':<40} | {'P(1|x)':<12} | {'argmax':<8} | {'Kết quả'}")
    print("-" * 145)

    final_predictions = []

    # 2. Duyệt qua từng dòng dữ liệu cần dự đoán
    for i, x_input in enumerate(X_samples):
        class_posteriors = {}
        row_details = {}

        for c, params in model.items():
            prior = params['Giá trị tiền nghiệm']
            mean = params['Giá trị trung bình']
            std = params['Độ lệch chuẩn']

            # Tính Likelihood cho từng feature (x1, x2, x3, x4)
            likelihoods = 1 / np.sqrt(2 * np.pi * (std ** 2)) * np.exp(-(x_input - mean) ** 2 / (2 * std ** 2))

            # Tính Hậu nghiệm (Posterior) = Prior * Product(Likelihoods)
            posterior = prior * np.prod(likelihoods)

            class_posteriors[c] = posterior
            row_details[c] = {'l_list': likelihoods, 'post': posterior}

        # 3. Tìm lớp chiến thắng (argmax)
        best_class = max(class_posteriors, key=class_posteriors.get)
        final_predictions.append(best_class)

        # 4. Chuẩn bị chuỗi text để in (định dạng số mũ :.1e cho giống trong vở)
        # Giả sử class 0 là Fail, class 1 là Pass
        l0_str = " ".join([f"{val:.1e}" for val in row_details[0]['l_list']])
        l1_str = " ".join([f"{val:.1e}" for val in row_details[1]['l_list']])

        # 5. In dòng dữ liệu của mẫu hiện tại
        print(
            f"{i + 1:<5} | {l0_str:<40} | {row_details[0]['post']:.2e}   | {l1_str:<40} | {row_details[1]['post']:.2e}   | {best_class:<8} | {labels[best_class]}")

    print("=" * 145 + "\n")
    return np.array(final_predictions)


def Gaussian_Naive_Bayes(X_train, Y_train):
    classes = np.unique(Y_train)
    model_params = {}

    print("=" * 50)
    print("CHI TIẾT THÔNG SỐ MÔ HÌNH")
    print("=" * 50)

    for c in classes:
        X_c = X_train[Y_train == c]
        prior = X_c.shape[0] / X_train.shape[0]

        means = np.mean(X_c, axis=0)
        stds = np.std(X_c, axis=0)
        vars_sq = stds ** 2

        model_params[c] = {
            'Giá trị tiền nghiệm': prior,
            'Giá trị trung bình': means,
            'Độ lệch chuẩn': stds
        }

        print(f"\n[CLASS {c}]")
        print(f" - Giá trị tiền nghiệm: {prior:.4f}")
        print(f" - Chi tiết từng đặc trưng:")
        print(f"   {'Feature':<15} | {'Mean':<10} | {'Variance (σ²)':<10}")
        print(f"   {'-' * 43}")

        # Vòng lặp in dữ liệu vào bảng
        for i in range(len(means)):
            print(f"   Feature {i:<8} | {means[i]:<10.4f} | {vars_sq[i]:<10.4f}")

    print("=" * 50 + "\n")
    return model_params


# Hàm huấn luyện mô hình Gaussian Naive Bayes bằng thư viện
def Gaussian_Naive_Bayes_Library(X_train, Y_train):
    print("=" * 50)
    print("CHI TIẾT THÔNG SỐ MÔ HÌNH (TỪ THƯ VIỆN SKLEARN)")
    print("=" * 50)

    # Dùng thư viện để huấn luyện
    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)

    # Truy xuất các giá trị thông số huấn luyện của mô hình
    model_params = {}
    classes = gnb.classes_

    for idx, c in enumerate(classes):
        # Lấy Xác suất tiền nghiệm (Prior)
        prior = gnb.class_prior_[idx]

        # Lấy Giá trị trung bình (Mean)
        means = gnb.theta_[idx]

        # Lấy Phương sai (Variance) và tính Độ lệch chuẩn (Standard Deviation)
        vars_sq = gnb.var_[idx]
        stds = np.sqrt(vars_sq)

        model_params[c] = {
            'Giá trị tiền nghiệm': prior,
            'Giá trị trung bình': means,
            'Độ lệch chuẩn': stds
        }

        # In ra màn hình
        print(f"\n[CLASS {c}]")
        print(f" - Giá trị tiền nghiệm: {prior:.4f}")
        print(f" - Chi tiết từng đặc trưng:")
        print(f"   {'Feature':<15} | {'Mean':<10} | {'Variance (σ²)':<10}")
        print(f"   {'-' * 43}")

        for i in range(len(means)):
            print(f"   Feature {i:<8} | {means[i]:<10.4f} | {vars_sq[i]:<10.4f}")

    print("=" * 50 + "\n")
    return model_params

if __name__ == "__main__":
    df = pd.read_csv('database/Iris.csv')
    target = 'Species'
    labels, y_encoded = np.unique(df[target].values, return_inverse=True)
    y = y_encoded
    x = df.drop(columns = [target, 'Id']).values
    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size = 0.1, random_state = 0)

    # huân luyện mô hình
    model = Gaussian_Naive_Bayes(x_train, y_train)

    # Đánh giá hiệu suất huấn luyện mô hình
    print("\n" + "*" * 50)
    print("ĐÁNH GIÁ HIỆU SUẤT MÔ HÌNH")
    print("*" * 50)

    # Trên tập Train
    # set show_log=False để màn hình không in ra hàng trăm dòng log
    y_pred_train = predict(x_train, model, labels)
    acc_train = accuracy_score(y_train, y_pred_train)
    print(f"Độ chính xác trên tập Huấn luyện (Train): {acc_train * 100:.2f}%")

    # Trên tập Validation
    y_pred_val = predict(x_val, model, labels)
    acc_val = accuracy_score(y_val, y_pred_val)
    print(f"Độ chính xác trên tập Xác thực (Validation): {acc_val * 100:.2f}%")

    # Trên tập test
    y_pred_test = predict(x_test, model, labels)
    acc_test = accuracy_score(y_test, y_pred_test)
    print(f"Độ chính xác trên tập Kiểm tra (Test): {acc_test * 100:.2f}%")
    print("*" * 50 + "\n")

    # Kiểm tra với giá trị thật được thêm vào mô hình
    X_new = np.array([
        [5.5, 7.8, 78, 82],
        [6.0, 8.2, 82, 85],
        [2.8, 6.2, 57, 63]
    ])
    y_pred = predict(X_new, model, labels)



