import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Thư viện dùng để huấn luyện mô hình Bernoulli Naive Bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

def Bernoulli_Naive_Bayes_Library(X_train, y_train, labels=None):
    # 1. Khởi tạo bộ đếm từ. QUAN TRỌNG: Thêm tham số binary=True
    # binary=True giúp chuyển ma trận tần suất thành ma trận 0/1 (Có/Không xuất hiện)
    vectorizer = CountVectorizer(analyzer=lambda x: str(x).split(), binary=True)

    X_train_binary = vectorizer.fit_transform(X_train)

    # 2. Sử dụng mô hình BernoulliNB của sklearn
    nb_model = BernoulliNB(alpha=1.0)  # alpha=1.0 là Laplace Smoothing
    nb_model.fit(X_train_binary, y_train)

    # 3. CHUYỂN ĐỔI (ADAPTER) đưa về cấu trúc dict
    model = {
        'priors': {},
        'doc_counts': {},  # Đổi tên thành doc_counts cho chuẩn Bernoulli (đếm số văn bản chứa từ)
        'class_totals': {},
        'vocab': set(vectorizer.get_feature_names_out()),
        'vocab_size': len(vectorizer.get_feature_names_out())
    }

    feature_names = vectorizer.get_feature_names_out()
    total_docs = len(y_train)

    for idx, c in enumerate(nb_model.classes_):
        model['priors'][c] = np.exp(nb_model.class_log_prior_[idx])  # Sklearn lưu log, ta cần exp ngược lại
        model['doc_counts'][c] = {}

        # Ở BernoulliNB, class_count_ chính là số lượng văn bản thuộc class c (N)
        model['class_totals'][c] = int(nb_model.class_count_[idx])

        # feature_count_ chứa số văn bản có xuất hiện từ đó
        counts_for_class = nb_model.feature_count_[idx]

        for word_idx, count in enumerate(counts_for_class):
            if count > 0:
                word = feature_names[word_idx]
                model['doc_counts'][c][word] = int(count)

    # 4. IN RA KẾT QUẢ
    print("\n" + "=" * 60)
    print("QUÁ TRÌNH HUẤN LUYỆN (BẰNG THƯ VIỆN SKLEARN - BERNOULLI)")
    print("=" * 60)

    vocal_list = list(model['vocab'])
    print(f"• Vocal = {vocal_list}")
    print(f"  => v = {model['vocab_size']}\n")

    for c in nb_model.classes_:
        class_name = labels[c] if labels is not None else str(c)
        print(f"• Thống kê cho lớp [ {class_name} ]:")

        for word, count in model['doc_counts'][c].items():
            print(f"  - count({word}, c = {class_name}) = {count}")

        n_class = model['class_totals'][c]
        n_label_print = f"N_{class_name.lower().replace(' ', '_')}"
        print(f"  => {n_label_print} = {n_class}\n")
    print("=" * 60 + "\n")

    return model


def Bernoulli_Naive_Bayes(X_train, y_train, labels=None):
    # Khởi tạo cấu trúc lưu trữ mô hình
    model = {
        'priors': {},
        'doc_counts': {},  # Lưu số lượng văn bản chứa từ
        'class_totals': {},  # Lưu tổng số văn bản của class
        'vocab': set(),
        'vocab_size': 0
    }

    total_docs = len(y_train)
    classes = np.unique(y_train)

    # Quét dữ liệu để đếm và tính toán
    for c in classes:
        X_c = X_train[y_train == c]
        model['priors'][c] = len(X_c) / total_docs
        model['doc_counts'][c] = {}

        # N: Tổng số văn bản thuộc class c
        model['class_totals'][c] = len(X_c)

        for text in X_c:
            # QUAN TRỌNG: Dùng set() để loại bỏ các từ trùng lặp trong 1 câu (Đặc trưng Bernoulli)
            words = set(str(text).split())
            for word in words:
                model['vocab'].add(word)
                model['doc_counts'][c][word] = model['doc_counts'][c].get(word, 0) + 1

    model['vocab_size'] = len(model['vocab'])

    print("\n" + "=" * 60)
    print("QUÁ TRÌNH HUẤN LUYỆN (TỰ VIẾT - BERNOULLI)")
    print("=" * 60)

    vocal_list = list(model['vocab'])
    print(f"• Vocal = {vocal_list}")
    print(f"  => v = {model['vocab_size']}\n")

    for c in classes:
        class_name = labels[c] if labels is not None else str(c)
        print(f"• Thống kê cho lớp [ {class_name} ]:")

        for word, count in model['doc_counts'][c].items():
            print(f"  - count({word}, c = {class_name}) = {count}")

        n_class = model['class_totals'][c]
        n_label_print = f"N_{class_name.lower().replace(' ', '_')}"
        print(f"  => {n_label_print} = {n_class}\n")
    print("=" * 60 + "\n")
    return model

def predict_bernoulli(X, model, labels=None):
    predictions = []
    classes = list(model['priors'].keys())

    for text_idx, text in enumerate(X):
        print(f"\n--- Test case {text_idx + 1}: '{text}' ---")

        # Dùng set() vì nếu câu test có từ lặp lại, Bernoulli cũng chỉ tính 1 lần
        words = set(str(text).split())
        class_probs = {}

        print("• Xác suất các thành phần:")
        for c in classes:
            class_name = labels[c] if labels is not None else str(c)
            for word in words:
                count_wi_c = model['doc_counts'][c].get(word, 0)
                N = model['class_totals'][c]

                # Công thức Bernoulli: (Count + 1) / (N + 2)
                p_wi_c = (count_wi_c + 1) / (N + 2)

                print(f"  P({word} | {class_name}) = ({count_wi_c} + 1)/({N} + 2) = {count_wi_c + 1}/{N + 2}")

        print("\n• Tính xác suất tổng hợp:")
        for i, c in enumerate(classes):
            class_name = labels[c] if labels is not None else str(c)
            prob = model['priors'][c]
            calc_str = f"{prob:.2f}"

            for word in words:
                count_wi_c = model['doc_counts'][c].get(word, 0)
                N = model['class_totals'][c]

                p_wi_c = (count_wi_c + 1) / (N + 2)
                prob *= p_wi_c
                calc_str += f" * ({count_wi_c + 1}/{N + 2})"

            class_probs[c] = prob
            print(f"  => P(Y = {class_name} | {', '.join(words)}) = {calc_str} = {prob:.5e}  ({i + 1})")

        best_class = max(class_probs, key=class_probs.get)
        best_class_name = labels[best_class] if labels is not None else str(best_class)
        predictions.append(best_class)

        indices_str = " và ".join([f"({i + 1})" for i in range(len(classes))])
        print(f"\n  => Từ {indices_str} => class là '{best_class_name}'")
        print("-" * 50)

    print("=" * 70 + "\n")
    return np.array(predictions)

if __name__ == "__main__":
    df = pd.read_csv('database/Database_Multinomial.csv')
    target = 'Species'
    # Mã hóa nhãn (Spam/Not Spam -> 1/0)
    labels, y_encoded = np.unique(df[target].values, return_inverse=True)
    y = y_encoded

    # Lấy cột Text chứa nội dung văn bản
    x = df['Text'].values

    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.1, random_state=0)

    model = Bernoulli_Naive_Bayes_Custom(x_train, y_train, labels)

    # Đánh giá hiệu suất huấn luyện mô hình
    print("\n" + "*" * 50)
    print("ĐÁNH GIÁ HIỆU SUẤT MÔ HÌNH")
    print("*" * 50)

    # Trên tập Train
    y_pred_train = predict_bernoulli(x_train, model, labels)
    acc_train = accuracy_score(y_train, y_pred_train)
    print(f"Độ chính xác trên tập Huấn luyện (Train): {acc_train * 100:.2f}%")

    # Trên tập Validation
    y_pred_val = predict_bernoulli(x_val, model, labels)
    acc_val = accuracy_score(y_val, y_pred_val)
    print(f"Độ chính xác trên tập Xác thực (Validation): {acc_val * 100:.2f}%")

    # Trên tập test
    y_pred_test = predict_bernoulli(x_test, model, labels)
    acc_test = accuracy_score(y_test, y_pred_test)
    print(f"Độ chính xác trên tập Kiểm tra (Test): {acc_test * 100:.2f}%")
    print("*" * 50 + "\n")

    # Kiểm tra với giá trị thật
    X_new = np.array([
        "buy now",
        "let's meet up cheap",
        "see you"
    ])

    # Dự đoán dữ liệu mới
    y_pred = predict_bernoulli(X_new, model, labels)

    print("DỰ ĐOÁN DỮ LIỆU MỚI:")
    for text, pred_idx in zip(X_new, y_pred):
        print(f"Câu: '{text}' ---> Dự đoán: {labels[pred_idx]}")