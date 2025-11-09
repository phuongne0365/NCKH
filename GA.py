import random
import matplotlib.pyplot as plt

class Individual:                         # lop ca the
                                          # khoi tao ca the
    def __init__(self, tour):
        self.tour = tour                  # tour:hanh trinh( danhsach cac thanh pho)
        self.fitness = -float('inf')      # vo cung lon
        self.kcach = float('inf')         # vo cung be 

    def calculate_fitness(self, matrankc):
        total_distance = 0.0
        size = len(self.tour)
        for i in range(size - 1):
            a = self.tour[i] - 1
            b = self.tour[i + 1] - 1
            total_distance += matrankc[a][b]
        total_distance += matrankc[self.tour[-1] - 1][self.tour[0] - 1]
        self.kcach = total_distance
        self.fitness = -total_distance
        return self.fitness

    def copy(self):
        new = Individual(self.tour.copy())
        new.fitness = self.fitness
        new.kcach = self.kcach
        return new

    def __repr__(self):
        return f"Tour: {self.tour} | Distance: {self.kcach:.2f} | Fitness: {self.fitness:.2f}"


class GATSP:

    def __init__(self, matrankc, toado, sotp,
                 socathe=25,
                 sothehe=150,
                 soelite=2,
                 ktgiaidau=3,
                 tllaighep=0.8,
                 tldotbien=0.2,
                 dungsom=30,
                 hat=42,
                 chitiet=True):

        self.matrankc = matrankc
        self.toado = toado
        self.sotp = sotp

        # Tham so
        self.socathe = socathe
        self.sothehe = sothehe
        self.soelite = soelite
        self.ktgiaidau = ktgiaidau
        self.tllaighep = tllaighep
        self.tldotbien = tldotbien
        self.dungsom = dungsom
        self.hat = hat
        self.chitiet = chitiet

        # Trang thai
        self.cathe = []
        self.lichsutot = []
        self.lichsutb = []
        self.lichsutour = []

        random.seed(self.hat)
        self.colors = plt.cm.get_cmap('tab20')

    # Khoi tao quan the
    def khoitaoqt(self):
        self.cathe = []
        for _ in range(self.socathe):
            tour = list(range(1, self.sotp + 1))
            random.shuffle(tour)
            ind = Individual(tour)
            ind.calculate_fitness(self.matrankc)
            self.cathe.append(ind)

    def tournament_selection(self):
        tournament = random.sample(self.cathe, self.ktgiaidau)
        winner = max(tournament, key=lambda ind: ind.fitness)
        return winner.copy()

    def laigheptt(self, parent1, parent2):
        size = len(parent1.tour)
        cut1, cut2 = sorted(random.sample(range(size), 2))
        def make_child(a, b):
            child = [-1] * size
            child[cut1:cut2] = a.tour[cut1:cut2]
            pos = cut2
            for city in b.tour[cut2:] + b.tour[:cut2]:
                if city not in child:
                    if pos >= size:
                        pos = 0
                    child[pos] = city
                    pos += 1
            return Individual(child)
        return make_child(parent1, parent2), make_child(parent2, parent1)

    def dotbienhd(self, individual):
        i, j = random.sample(range(len(individual.tour)), 2)
        individual.tour[i], individual.tour[j] = individual.tour[j], individual.tour[i]

    def chonelite(self, old_pop, thehecon):
        combined = old_pop + thehecon
        combined_sorted = sorted(combined, key=lambda ind: ind.fitness, reverse=True)
        return [ind.copy() for ind in combined_sorted[:self.socathe]]

    def run(self):
        if self.chitiet:
            self._print_header()
        self.khoitaoqt()
        if self.chitiet:
            self._print_initial_sample()

        generations_without_improvement = 0
        previous_best_fitness = -float('inf')
        best_individual = None
        best_overall_gen = 0

        for gen in range(self.sothehe):
            best_ind = max(self.cathe, key=lambda ind: ind.fitness)
            avg_fit = sum(ind.fitness for ind in self.cathe) / len(self.cathe)

            self.lichsutot.append(best_ind.fitness)
            self.lichsutb.append(avg_fit)
            self.lichsutour.append(best_ind.tour.copy())

            if best_individual is None or best_ind.kcach < best_individual.kcach:
                best_individual = best_ind.copy()
                best_overall_gen = gen + 1

            tolerance = 1e-6  # dung sai để tránh lỗi float
            if best_ind.fitness > previous_best_fitness + tolerance:
                previous_best_fitness = best_ind.fitness
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            if self.chitiet and (gen == 0 or (gen + 1) % 10 == 0 or gen == self.sothehe - 1):
                print(f"The he {gen + 1:3d} | Best distance: {best_ind.kcach:8.2f} | Tour: {' -> '.join(map(str, best_ind.tour))}")

            if generations_without_improvement >= self.dungsom:
                if self.chitiet:
                    print(f"\nDung som tai the he {gen + 1}: Khong cai thien sau {self.dungsom} the he")
                break

            # tao offspring
            thehecon = []
            while len(thehecon) < self.socathe:
                p1 = self.tournament_selection()
                p2 = self.tournament_selection()
                if random.random() < self.tllaighep:
                    c1, c2 = self.laigheptt(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                if random.random() < self.tldotbien:
                    self.dotbienhd(c1)
                if random.random() < self.tldotbien:
                    self.dotbienhd(c2)
                c1.calculate_fitness(self.matrankc)
                c2.calculate_fitness(self.matrankc)
                thehecon.append(c1)
                if len(thehecon) < self.socathe:
                    thehecon.append(c2)

            self.cathe = self.chonelite(self.cathe, thehecon)

        if self.chitiet:
            print("\nKet qua cuoi cung:")
            print(f"  Best found at generation: {best_overall_gen}")
            print(f"  Distance: {best_individual.kcach:.2f}")
            print(f"  Tour: {' -> '.join(map(str, best_individual.tour))} -> {best_individual.tour[0]}")

        return {
            'best_individual': best_individual,
            'population': self.cathe,
            'best_fitness_history': self.lichsutot,
            'avg_fitness_history': self.lichsutb,
            'best_distance_history': [ind.kcach for ind in self.cathe],
            'best_tour_history': self.lichsutour
        }

    # Vẽ biểu đồ
    def plot_convergence(self, result, fname='tsp_convergence.png'):
        gens = range(1, len(result['best_distance_history']) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(gens, result['best_distance_history'], linewidth=2, label='Best Distance')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Distance')
        ax1.set_title('Best Distance Over Generations')
        ax1.grid(True)
        ax1.legend()

        ax2.plot(gens, result['best_fitness_history'], linewidth=2, label='Best Fitness')
        ax2.plot(gens, result['avg_fitness_history'], linestyle='--', linewidth=1.5, label='Avg Fitness')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Fitness')
        ax2.set_title('Fitness Over Generations')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Bieu do da duoc luu: {fname}")
        plt.show()

    def plot_tour(self, tour, kcach, title):
        toado = self.toado  # ✅ Lấy tọa độ từ đối tượng GA
        import matplotlib.pyplot as plt

        x = [toado[i][0] for i in tour]
        y = [toado[i][1] for i in tour]
        x.append(toado[tour[0]][0])
        y.append(toado[tour[0]][1])

        plt.figure(figsize=(8, 6))
        plt.plot(x, y, 'o-', color='blue')
        for i, city in enumerate(tour):
            plt.text(toado[city][0], toado[city][1], str(city), fontsize=9, color='red')
        plt.title(f"{title}\nTổng quãng đường: {kcach:.2f}")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()

    def _print_header(self):
        print("=" * 80)
        print(f"GIAI THUAT DI TRUYEN CHO BAI TOAN TSP - {self.sotp} THANH PHO")
        print("=" * 80)

    def _print_initial_sample(self):
        print("\nToa do cac thanh pho (mot so):")
        for i in range(1, min(self.sotp + 1, 11)):
            x, y = self.toado[i]
            print(f"  {i}: ({x:.1f}, {y:.1f})")
        print(f"\nTham so: Population={self.socathe}, Generations={self.sothehe}, "
              f"Elite={self.soelite}, Crossover={self.tllaighep}, Mutation={self.tldotbien}")
