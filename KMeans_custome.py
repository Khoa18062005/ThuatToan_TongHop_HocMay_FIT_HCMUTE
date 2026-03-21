import numpy as np

def euclid(u1, u2):
    tmp = 0
    for i in range(len(u1)):
        tmp += (u1[i] - u2[i]) ** 2
    return np.sqrt(tmp)

def check(u1, p1):
    for i in range(len(u1)):
        if abs(u1[i] - p1[i]) > 0.1:
            return False
    return True

def KMeans_custome(p, u, k):
    iteration = 1

    while True:
        print(f"\n===== VÒNG LẶP {iteration} ======")
        matrix_show = np.zeros((len(p), len(u)))

        print("===== Bước 03: Gán điểm dữ liệu ==========")
        for i in range(len(p)):
            for j in range(len(u)):
                matrix_show[i][j] = round(euclid(p[i], u[j]), 2)

        # [ĐÃ SỬA Ở ĐÂY]: Ép kiểu từ np.float64 về float thuần của Python để in cho đẹp
        for row in matrix_show:
            clean_row = [float(val) for val in row]
            print(clean_row)

        matrix_index = []
        for i in range(len(p)):
            min_index_row = i
            min_index_col = int(np.argmin(matrix_show[i]))
            matrix_index.append([min_index_row, min_index_col])
        print(f'+ matrix_index = {matrix_index}')

        u1_t = []
        u2_t = []
        u3_t = []
        for i in range(len(matrix_index)):
            index_row = matrix_index[i][0]
            index_col = matrix_index[i][1]
            if index_col == 0:
                u1_t.append(p[index_row])
            elif index_col == 1:
                u2_t.append(p[index_row])
            elif index_col == 2:
                u3_t.append(p[index_row])

        print(f'+ Cụm số 01: {u1_t}')
        print(f'+ Cụm số 02: {u2_t}')
        print(f'+ Cụm số 03: {u3_t}')

        print("===== Bước 04: Cập nhật trọng số ==========")
        new_u1 = np.round(np.mean(u1_t, axis=0), 2).tolist()
        new_u2 = np.round(np.mean(u2_t, axis=0), 2).tolist()
        new_u3 = np.round(np.mean(u3_t, axis=0), 2).tolist()

        print(f'+ Cập nhật u1: {new_u1}')
        print(f'+ Cập nhật u2: {new_u2}')
        print(f'+ Cập nhật u3: {new_u3}')

        print("===== Bước 05: Kiểm tra hội tụ ==========")
        is_converged = check(u[0], new_u1) and check(u[1], new_u2) and check(u[2], new_u3)

        if is_converged:
            print(f"==> THÀNH CÔNG: Thuật toán đã hội tụ ở vòng lặp {iteration}!")
            break
        else:
            print("==> Chưa hội tụ, tiếp tục lặp...")
            u = [new_u1, new_u2, new_u3]
            iteration += 1

if __name__ == "__main__":
    p1 = [2.1, 3.1, 1.6]
    p2 = [3.2, 3.6, 2.1]
    p3 = [3.6, 3.1, 2.6]
    p4 = [7.9, 8.1, 7.6]
    p5 = [8.6, 8.7, 8.2]
    p6 = [9.1, 8.1, 8.6]
    p7 = [1.2, 2.1, 1.7]

    u1 = p1
    u2 = p2
    u3 = p3
    u = [u1, u2, u3]
    p = [p1, p2, p3, p4, p5, p6, p7]

    KMeans_custome(p, u, k=3)