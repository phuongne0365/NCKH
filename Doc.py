import numpy as np

def load_data_from_file(filename='input.txt'):    #doc file du lieu , tao ma tran khoang cach
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        sotp = int(lines[0].strip())
        toado = {}
        for i in range(1, sotp + 1):
            x, y = map(float, lines[i].strip().split())
            toado[i] = (x, y)
        matrankc = np.zeros((sotp, sotp))
        for i in range(sotp):
            for j in range(sotp):
                if i != j:
                    x1, y1 = toado[i + 1]
                    x2, y2 = toado[j + 1]
                    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    matrankc[i][j] = distance

        return matrankc, toado, sotp

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{filename}'")
        exit(1)
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        exit(1)
