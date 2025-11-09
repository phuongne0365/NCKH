
from Doc import *
from GA import *
import time

def measure_execution_time(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"\n Thời gian chạy GA: {end - start:.4f} giây")
    return result

if __name__ == "__main__":
    hat = 42                                                      # Định nghĩa random seed
    random.seed(hat)                                              # Set random seed
    np.random.seed(hat)
    matrankc, toado, sotp = load_data_from_file('data/25.txt')  # Đọc dữ liệu
    ga = GATSP(matrankc, toado, sotp, chitiet=True)               # Khởi tạo GA
    ketqua = measure_execution_time(ga.run)                       # Chạy GA và đo thời gian










    # VE BIEU DO
    print(f"\n{'=' * 80}")
    print("TẠO BIỂU ĐỒ")
    print(f"{'=' * 80}")

    import matplotlib.pyplot as plt
    import numpy as np

    # Lấy dữ liệu lịch sử từ kết quả GA
    best_distance_history = np.array(ketqua.get('best_distance_history', []))
    best_fitness_history = np.array(ketqua.get('best_fitness_history', []))

    # Kiểm tra dữ liệu hợp lệ
    if len(best_distance_history) == 0 and len(best_fitness_history) == 0:
        print("⚠️ Không có dữ liệu để vẽ biểu đồ.")
    else:
        # Đảm bảo hai mảng có cùng độ dài
        min_len = min(len(best_distance_history), len(best_fitness_history))
        best_distance_history = best_distance_history[:min_len]
        best_fitness_history = best_fitness_history[:min_len]
        gens = np.arange(1, min_len + 1)

        # Vẽ biểu đồ hội tụ (Convergence)
        fig, ax1 = plt.subplots(figsize=(8, 5))
        color1 = 'tab:blue'
        ax1.set_xlabel('Thế hệ (Generations)')
        ax1.set_ylabel('Best Distance', color=color1)
        ax1.plot(gens, best_distance_history, color=color1, linewidth=2, label='Best Distance')
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Best Fitness', color=color2)
        ax2.plot(gens, best_fitness_history, color=color2, linewidth=2, label='Best Fitness')
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.title("Convergence Curve of Genetic Algorithm (TSP)")
        fig.tight_layout()
        plt.show()

    # Vẽ lộ trình tốt nhất (Best Tour)
    ga.plot_tour(
        ketqua['best_individual'].tour,
        ketqua['best_individual'].kcach,
        "Best TSP Tour Found by Genetic Algorithm"
    )

