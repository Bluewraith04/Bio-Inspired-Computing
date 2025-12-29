import numpy as np
import pandas as pd
from ga_operators import compute_pairwise

class GA:
    # # TODO: Define Initializations Parameters
    _modes = ["random-selection", "roulette-wheel", "linear-rank"]
    
    def __init__(self, pop_size, n_strings, n_bits, obj_fn, n_generations, bounds, maximize = True, offset_fitness=False, selection = "roulette-wheel", cross_prob = 0.8, mut_prob = 0.25, random_state=42):
        """
        :args:
            pop_size (int): The number of chromosomes in a single population
            n_vars (int): The number of decision variables and no of strings in a chromosome
            n_bits (int): The number of bits in a single string  
            obj_fn (function): The objective function to be optimized
            n_generations (int): Maximum number of generations evaluated
            bounds (tuple): Upper and Lower bounds of decision variables
            maximize (bool) = True: A boolean flag set to True for maximization and False for minimization
            selection (string) = "roulette-wheel": The method of selection to be used ("roulette-wheel" or "linear-rank")
            cross_prob (float) = 0.8: The probability of a crossover happening
            mut_prob (float) = 0.25: The probabilty of a mutation happening
        """
        self.pop_size = pop_size
        self.n_strings = n_strings
        self.n_bits = n_bits
        self.obj_fn = obj_fn
        self.maximizing = maximize
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob
        if selection not in GA._modes:
            raise ValueError(f"Invalid Selection Scheme '{selection}'")
        self.mode = selection
        self.rng = np.random.default_rng(seed=random_state)
        self.population = self.rng.integers(2, size=(self.pop_size, self.n_strings, self.n_bits))
        self.x = np.zeros(self.pop_size)
        self.values = np.zeros(self.pop_size)
        self.offset_fitness = offset_fitness
        self.fitnesses = np.zeros(self.pop_size)
        self.n_generations = n_generations
        self.lbounds = np.array([b[0] for b in bounds])
        self.ranges = np.array([b[1] - b[0] for b in bounds])
        self.records = {"x": [], "values": [], "bits": []}
        self.iter = 0
        self._init()
        
    def _init(self):
        self._evaluate()
        self._save_best_fit()
        self._select()

    # TODO: Define Evaluation Logic
    def _evaluate(self):
        d = 2 ** np.arange(self.n_bits-1, -1, -1)
        decoded = self.population @ d
        self.x = self.lbounds + (decoded / ((2 ** self.n_bits) - 1)) * self.ranges
        self.values = self.obj_fn(*[self.x[:, i] for i in range(self.n_strings)])
        base_values = self.values if self.maximizing else 1 / self.values
        if self.offset_fitness:
            self.fitnesses = base_values - min(base_values) + 10e-6
        else:
            self.fitnesses = base_values
        
    def _save_best_fit(self):
        idx = np.argmax(self.fitnesses)
        self.records['bits'].append(self.population[idx])
        self.records['x'].append(self.x[idx])
        self.records['values'].append(self.values[idx])

    
    # TODO: Define Selection Logic
    def _select(self):
        if self.mode == "random-selection":
            self.population = self.rng.choice(self.population, size=self.pop_size, replace=True)
        elif self.mode == "roulette-wheel":
            norm = self.fitnesses / sum(self.fitnesses)
            self.population = self.rng.choice(self.population, size=self.pop_size, p=norm, replace=True)
        elif self.mode == "linear-rank":
            ranks = (-self.fitnesses).argsort() + 1
            probs = 2 * (self.pop_size - ranks + 1) / (self.pop_size * (self.pop_size + 1))
            self.population = self.rng.choice(self.population, size=self.pop_size, p=probs, replace=True)
        
    # TODO: Define Best Fit Selection
    def _get_best_fit(self):
        return self.population[np.argmax(self.fitnesses)]
        
    # TODO: Define Mating Selection and Reproduction Logic
    def _generate_mating_pair(self):
        idx = self.rng.integers(0, self.pop_size, size=2)
        while idx[0] == idx[1]:
            idx  = self.rng.integers(0, self.pop_size, size=2)
        return idx
    
    def _compute_pairwise(self, pair, cross=True, verbose=False):
        return compute_pairwise(pair, self.n_strings, self.n_bits, self.cross_prob, self.mut_prob, multi=cross, rng=self.rng, verbose=verbose)
    
    def _crossover(self, verbose=False):
        idx = self._generate_mating_pair()
        pair = [self.population[idx[0]], self.population[idx[1]]]
        # if verbose: print("Parent: ", pair)
        new_pair = self._compute_pairwise(pair, verbose=verbose)
        # if verbose: print("Offspring: ", new_pair, "\n")
        return new_pair
        
    def _mutate(self, verbose=False):
        # if verbose: print("Generating Mating Pair")
        idx = self._generate_mating_pair()
        pair = [self.population[idx[0]], self.population[idx[1]]]
        # if verbose: print("Parent: ", pair)
        new_pair = self._compute_pairwise(pair, cross=False, verbose=verbose)
        # if verbose: print("Offspring: ", new_pair, "\n")
        return new_pair
        
    def _crossover_population(self, verbose=False):
        new_population = np.zeros_like(self.population)
        for i in range(self.pop_size//2):
            new_pair = self._crossover(verbose=verbose)
            new_population[i*2:i*2+2] = new_pair
        if self.pop_size % 2 == 1:
            new_population[-1] = self.records['bits'][-1]
            # if verbose: print("Elite Selection: ", new_population[-1])
        self.population = new_population
    
    def _mutate_population(self, verbose=False):
        new_population = np.zeros_like(self.population)
        for i in range(self.pop_size//2):
            new_pair = self._mutate(verbose=verbose)
            new_population[i*2:i*2+2] = new_pair
        if self.pop_size % 2 == 1:
            new_population[-1] = self.records['bits'][-1]
            # if verbose: print("Elite Selection: ", new_population[-1])
        self.population = new_population
        
    def _step(self):
        self.iter += 1
        self._crossover_population()
        self._mutate_population()
        self._evaluate()
        self._save_best_fit()
        self._select()
        print(f"Generation {self.iter}")
        best = self.generation_best
        print(*[f"x{i+1}: {best['x'][i]}" for i in range(len(best['x']))], sep='\n')
        print(f"Objective Value: {best['value']}\n")
        
        if self.iter == self.n_generations:
            print(f"Optimal Value")
            best = self.best_fit
            print(*[f"x{i+1}: {best['x'][i]}" for i in range(len(best['x']))], sep='\n')
            print(f"Objective Value: {best['value']}\n")
        
        
    def run(self):
        for i in range(self.n_generations):
            self._step()
    
    # TODO: Define Best Fit Property
    @property
    def best_fit(self):
        idx = np.argmax(self.records['values'])
        return {
            'x': self.records['x'][idx],
            'value': self.records['values'][idx]
        }
        
    @property
    def generation_best(self):
        return {
            'x': self.records['x'][-1],
            'value': self.records['values'][-1]
        }
    
# def test_func(x, y):
#     return 2 * x ** 2 + 3 * y ** 2 - x * y

# ga = GA(pop_size=50, n_strings=2, n_bits=10, obj_fn=test_func, n_generations=10, bounds=((0, 4), (1, 3)), offset_fitness=True, random_state=42)
# ga.run()


# # NOTE: Remember to set the random state to the last three digits of your matric number; The XXX in 210407XXX
# # NOTE: Objective function f(x1, x2, x3, x4) = 2(x1)(x2)(x3)(x4) - 4(x1)(x2)(x3) - 2(x2)(x3)(x4) - x1(x2) - x3(x4) + x1^2 + x2^2 
# #                                              + x3^2 + x4^2 - 2x1 - 4x2 + 4x3 - 2x4
# def objective_function(x1, x2, x3, x4):
#     return (2 * x1 * x2 * x3 * x4) - (4 * x1 * x2 * x3) - (2 * x2 * x3 * x4) - (x1 * x2) - (x1 * x4) + \
#         (x1 ** 2) + (x2 ** 2) + (x3 ** 2) + (x4 ** 2)  - (2 * x1) - (4 * x2) + (4 * x3) - (2 * x4)
    
# bounds = (
#     (1, 3),
#     (0, 5),
#     (1, 2),
#     (0, 4)
# )

# b = [np.linspace(b[0], b[1]+1, 51) for b in bounds]
# grids = np.meshgrid(*b)
# print((objective_function(*grids)).min())