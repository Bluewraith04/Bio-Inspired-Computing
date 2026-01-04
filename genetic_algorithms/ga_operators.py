import numpy as np

def compute_pairwise(pair: list, n_strings: int, n_bytes: int, cross_prob=0.8, mut_prob=0.25, multi=True, rng=np.random.default_rng(), verbose=False):
    n = [rng.integers(n_bytes) for i in range(n_strings)]
    # if verbose: print(f"N = {n}")
    cross = rng.choice([crossover, crossover2])
    return cross(pair, n, n_strings, n_bytes, rng, cross_prob, verbose) if multi else mutate(pair, n, n_strings, n_bytes, rng, mut_prob, verbose)

        
def crossover(pair, n: list[int], n_strings: int, n_bytes: int, rng, cross_prob=0.8, verbose=False):
    new_pair = np.zeros((2, n_strings, n_bytes))
    mask = [rng.random() <= cross_prob for i in range(n_strings)]
    # if verbose: print("Mask: ", mask)
    for i in range(n_strings):
        if mask[i]:
            for j in range(n_bytes):
                if j >= n[i]:
                    new_pair[0][i][j] = pair[1][i][j]
                    new_pair[1][i][j] = pair[0][i][j]
                else:
                    new_pair[0][i][j] = pair[0][i][j]
                    new_pair[1][i][j] = pair[1][i][j]
        else:
            new_pair[0][i] = pair[0][i]
            new_pair[1][i] = pair[1][i]
                    
    return new_pair

def crossover2(pair, n: list[int], n_strings: int, n_bytes: int, rng, cross_prob=0.8, verbose=False):
    new_pair = np.zeros((2, n_strings, n_bytes))
    mask = [rng.random() <= cross_prob for i in range(n_strings)]
    
    n2 = [rng.integers(n_bytes) for i in range(n_strings)]
    
    N = [(n[i], n2[i]) if n[i] <= n2[i] else (n2[i], n[i]) for i in range(n_strings)]
    # if verbose: print("Mask: ", mask)
    for i in range(n_strings):
        if mask[i]:
            for j in range(n_bytes):
                if j >= N[i][0] and j <= N[i][1]:
                    new_pair[0][i][j] = pair[1][i][j]
                    new_pair[1][i][j] = pair[0][i][j]
                else:
                    new_pair[0][i][j] = pair[0][i][j]
                    new_pair[1][i][j] = pair[1][i][j]
        else:
            new_pair[0][i] = pair[0][i]
            new_pair[1][i] = pair[1][i]
    
    return new_pair          

def mutate(pair, n: list[int], n_strings: int, n_bytes: int, rng, mut_prob=0.25, verbose=False):
    new_pair = np.zeros((2, n_strings, n_bytes))
    mask = [rng.random() <= mut_prob for i in range(n_strings)]
    # if verbose: print("Mask: ", mask)
    for i in range(n_strings):
        if mask[i]:
            for j in range(n_bytes):
                if j == n[i]:
                    new_pair[0][i][j] = pair[1][i][j]
                    new_pair[1][i][j] = pair[0][i][j]
                else:
                    new_pair[0][i][j] = pair[0][i][j]
                    new_pair[1][i][j] = pair[1][i][j]
        else:
            new_pair[0][i] = pair[0][i]
            new_pair[1][i] = pair[1][i]
                    
    return new_pair

def in_mutate(string, length, rng):
    n_shifts = rng.integers(0, 2)
    for _ in range(n_shifts):
        n = rng.integers(0, length, size=2)
        while np.allclose(*n):
            n = rng.integers(0, length, size=2)
        temp = string[n[0]]
        string[n[0]] = string[n[1]]
        string[n[1]] = temp
    
    