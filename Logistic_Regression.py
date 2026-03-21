import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def Logistic_Regression_Onesample(X_train, y_train, w, b, lr, epochs):
    N = len(y_train)
    for epoch in range(epochs):
        for i in range(N):
            x = X_train[i]
            y = y_train[i]
            z = np.dot(x, w) + b
            y_pred = sigmoid(z)
            Loss = -(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))
            dw = (y_pred - y) * x.reshape(-1,1)
            db = (y_pred - y)
            w = w - lr * dw
            b = b - lr * db
    return w, b
def Logistic_Regression_Fullsample(X_train, y_train, w, b, lr, epochs):
    N = len(y_train)
    for epoch in range(epochs):
        z = np.dot(X_train, w) + b
        y_pred = sigmoid(z)
        Loss = np.mean(-(y_train*np.log(y_pred) + (1-y_train)*np.log(1-y_pred)))
        dw = (1/N) * np.dot(X_train.T, (y_pred - y_train))
        db = (1/N) * np.sum((y_pred - y_train))
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