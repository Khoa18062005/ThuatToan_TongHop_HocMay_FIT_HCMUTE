import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Thư viện dùng để huấn luyện mô hình Multinomial Naive Bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def Multinomial_Naive_Bayes_Library(X_train, y_train, labels=None):
    # 1. Khởi tạo bộ đếm từ (Dùng lambda để cắt từ y hệt hàm split() trong code cũ của bạn)
    vectorizer = CountVectorizer(analyzer=lambda x: str(x).split())

    # Chuyển đổi văn bản thành ma trận tần suất xuất hiện của từ
    X_train_counts = vectorizer.fit_transform(X_train)

    # 2. Sử dụng mô hình MultinomialNB của thư viện sklearn
    nb_model = MultinomialNB(alpha=1.0)  # alpha=1.0 chính là Laplace Smoothing (+1)
    nb_model.fit(X_train_counts, y_train)

    # 3. CHUYỂN ĐỔI (ADAPTER): Trích xuất thông số từ sklearn đưa về cấu trúc dict
    # Việc này giúp hàm predict() bên dưới không bị lỗi
    model = {
        'priors': {},
        'word_counts': {},
        'class_totals': {},
        'vocab': set(vectorizer.get_feature_names_out()),
        'vocab_size': len(vectorizer.get_feature_names_out())
    }

    feature_names = vectorizer.get_feature_names_out()
    total_docs = len(y_train)

    # Quét qua từng class mà mô hình đã học (VD: 0, 1)
    for idx, c in enumerate(nb_model.classes_):
        # Xác suất tiên nghiệm (Prior)
        model['priors'][c] = nb_model.class_count_[idx] / total_docs

        # Tạo dict con để lưu đếm từ
        model['word_counts'][c] = {}

        # feature_count_ chứa số lần xuất hiện của các từ trong class c
        counts_for_class = nb_model.feature_count_[idx]

        # Tổng số lượng từ trong class c
        model['class_totals'][c] = int(np.sum(counts_for_class))

        # Map số lần đếm vào từng từ cụ thể
        for word_idx, count in enumerate(counts_for_class):
            if count > 0:  # Chỉ lưu những từ có xuất hiện (giống y hệt code cũ)
                word = feature_names[word_idx]
                model['word_counts'][c][word] = int(count)

    # 4. IN RA KẾT QUẢ GIỐNG Y HỆT HÀM CŨ ĐỂ GIÁO VIÊN KIỂM TRA
    print("\n" + "=" * 60)
    print("QUÁ TRÌNH HUẤN LUYỆN (BẰNG THƯ VIỆN SKLEARN)")
    print("=" * 60)

    vocal_list = list(model['vocab'])
    print(f"• Vocal = {vocal_list}")
    print(f"  => v = {model['vocab_size']}\n")

    for c in nb_model.classes_:
        class_name = labels[c] if labels is not None else str(c)
        print(f"• Thống kê cho lớp [ {class_name} ]:")

        for word, count in model['word_counts'][c].items():
            print(f"  - count({word}, c = {class_name}) = {count}")

        n_class = model['class_totals'][c]
        n_label_print = f"N_{class_name.lower().replace(' ', '_')}"
        print(f"  => {n_label_print} = {n_class}\n")
    print("=" * 60 + "\n")

    return model

def Multinomial_Naive_Bayes(X_train, y_train, labels=None):
    # Khởi tạo cấu trúc lưu trữ mô hình
    model = {
        'priors': {},
        'word_counts': {},
        'class_totals': {},
        'vocab': set(),
        'vocab_size': 0
    }

    total_docs = len(y_train)
    classes = np.unique(y_train)

    # 2. Quét dữ liệu để đếm và tính toán
    for c in classes:
        X_c = X_train[y_train == c]
        model['priors'][c] = len(X_c) / total_docs
        model['word_counts'][c] = {}
        model['class_totals'][c] = 0

        for text in X_c:
            words = str(text).split()
            for word in words:
                model['vocab'].add(word)
                model['word_counts'][c][word] = model['word_counts'][c].get(word, 0) + 1
                model['class_totals'][c] += 1

    model['vocab_size'] = len(model['vocab'])

    print("\n" + "=" * 60)
    print("QUÁ TRÌNH HUẤN LUYỆN (TÍNH TOÁN CÁC THAM SỐ)")
    print("=" * 60)

    # Yêu cầu 1 & 2: Show list vocal và kích thước
    vocal_list = list(model['vocab'])
    print(f"• Vocal = {vocal_list}")
    print(f"  => v = {model['vocab_size']}\n")

    # Yêu cầu 3 & 4: Đếm số lần xuất hiện và tổng số từ của từng class
    for c in classes:
        # Lấy tên class dạng chữ (nếu có truyền labels)
        class_name = labels[c] if labels is not None else str(c)

        print(f"• Thống kê cho lớp [ {class_name} ]:")

        # Đếm lần lượt số lần xuất hiện của từng từ
        for word, count in model['word_counts'][c].items():
            print(f"  - count({word}, c = {class_name}) = {count}")

        # Tổng số lần xuất hiện của các từ thuộc class (N)
        n_class = model['class_totals'][c]
        # Xử lý chuỗi một chút để in ra giống slide: N_spam, N_not_spam
        n_label_print = f"N_{class_name.lower().replace(' ', '_')}"
        print(f"  => {n_label_print} = {n_class}\n")
    print("=" * 60 + "\n")
    return model

def predict(X, model, labels=None):
    predictions = []
    # Lấy danh sách các class (VD: 0, 1)
    classes = list(model['priors'].keys())

    for text_idx, text in enumerate(X):
        print(f"\n--- Test case {text_idx + 1}: '{text}' ---")
        words = str(text).split()
        class_probs = {}

        # BƯỚC 1: In ra xác suất của từng thành phần (Giống nửa trên tờ nháp)
        print("• Xác suất các thành phần:")
        for c in classes:
            class_name = labels[c] if labels is not None else str(c)
            for word in words:
                count_wi_c = model['word_counts'][c].get(word, 0)
                N = model['class_totals'][c]
                v = model['vocab_size']

                # Tính xác suất có Laplace Smoothing
                p_wi_c = (count_wi_c + 1) / (N + v)

                # Format in ra: P(word | class) = phân số
                print(f"  P({word} | {class_name}) = ({count_wi_c} + 1)/({N} + {v}) = {count_wi_c + 1}/{N + v}")

        # BƯỚC 2: In ra công thức nhân tổng hợp (Giống nửa dưới tờ nháp)
        print("\n• Tính xác suất tổng hợp:")
        for i, c in enumerate(classes):
            class_name = labels[c] if labels is not None else str(c)

            # Khởi tạo xác suất bằng Prior P(Y)
            prob = model['priors'][c]

            # Tạo chuỗi công thức để in log (Bắt đầu với Prior)
            calc_str = f"{prob:.2f}"

            for word in words:
                count_wi_c = model['word_counts'][c].get(word, 0)
                N = model['class_totals'][c]
                v = model['vocab_size']
                p_wi_c = (count_wi_c + 1) / (N + v)

                # Nhân dồn xác suất
                prob *= p_wi_c
                # Nối thêm vào chuỗi công thức
                calc_str += f" * ({count_wi_c + 1}/{N + v})"

            class_probs[c] = prob

            print(f"  => P(Y = {class_name} | {', '.join(words)}) = {calc_str} = {prob:.5e}  ({i + 1})")

        # BƯỚC 3: So sánh và kết luận (Giống dòng cuối tờ nháp)
        best_class = max(class_probs, key=class_probs.get)
        best_class_name = labels[best_class] if labels is not None else str(best_class)
        predictions.append(best_class)

        # Tự động tạo chuỗi "Từ (1) và (2)..." dựa trên số lượng class
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

    # Lấy cột Text thay vì các cột số liệu như Iris
    x = df['Text'].values

    # Chia dữ liệu
    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.1, random_state=0)

    # Huấn luyện mô hình
    model = Multinomial_Naive_Bayes_Library(x_train, y_train)

    # Đánh giá hiệu suất huấn luyện mô hình
    print("\n" + "*" * 50)
    print("ĐÁNH GIÁ HIỆU SUẤT MÔ HÌNH")
    print("*" * 50)

    # Trên tập Train
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

    # Kiểm tra với giá trị thật
    X_new = np.array([
        "buy now",
        "let's meet up cheap",
        "see you"
    ])

    y_pred = predict(X_new, model, labels)

    print("DỰ ĐOÁN DỮ LIỆU MỚI:")
    for text, pred_idx in zip(X_new, y_pred):
        print(f"Câu: '{text}' ---> Dự đoán: {labels[pred_idx]}")