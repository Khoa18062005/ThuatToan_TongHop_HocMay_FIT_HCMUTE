import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
def Linear_Regression_Onesample(X_train, y_train, w, b, lr, epochs):
    N = len(y_train)
    for epoch in range(epochs):
        print(f'===== Epoch {epoch+1} =====')
        for i in range(N):
            print(f'- Mẫu thứ {i+1}:')
            x = X_train[i]
            y = y_train[i]
            y_pred = np.dot(x,W) + b
            Loss = (y_pred - y)**2
            dw = 2 * (y_pred - y) * x.reshape(-1,1)
            db = 2 * (y_pred - y)
            w = w - lr * dw
            b = b - lr * db
            print(f' + Loss = {Loss}')
            print(f' + W = {w}')
            print(f' + b = {b}')
            print('\n')

def Linear_Regression_Fullsample(X_train, y_train, w, b, lr, epochs):
    N = len(y_train)
    for epoch in range(epochs):
        y_pred = np.dot(X_train, w) + b
        Loss = (1/N) * (np.sum((y_pred - y_train) ** 2))
        dw = (2/N) * np.dot(X_train.T, (y_pred - y_train))
        db = (2/N) * np.sum((y_pred - y_train))
        w = w - lr * dw
        b = b - lr * db
    return w, b
if __name__ == '__main__':
    # 1. Nhập dữ liệu cho mô hình
    df = pd.read_csv('database/Experience_Salary.csv')
    X = df.drop('salary', axis=1).values
    y = df['salary'].values.reshape(-1,1)

    # 2. Phân chia dữ liệu thành tập train và tập test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Khởi tạo giá trị w và b
    # Tính số lượng feature
    INPUT_DIM = X_train.shape[1]
    # Tính số lượng label (target)
    OUTPUT_DIM = y_train.shape[1]
    W = 0.01 * np.random.randn(INPUT_DIM, OUTPUT_DIM)
    b = np.zeros((1, 1))
    print(b)
    print(X_train)

    # 4. Huấn luyện mô hình
    Linear_Regression_Onesample(X_train, y_train, W, b, 0.01, 2)
