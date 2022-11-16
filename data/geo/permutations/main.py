import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_permutations(in_path, out_path):
    data = pd.read_csv(in_path)
    mrs_gold = data.MR_GOLD
    mrs_mn = data.MR_MN
    mn = data.MN
    permutations = []
    for i in range(len(mn)):
        if mn[i] == 1:
            mr_gold = mrs_gold[i].strip().split()
            perm = [x + 1 for x in range(len(mr_gold))]
            permutations += [perm]
        else:
            mr_gold = mrs_gold[i].strip().split()
            mr_mn = mrs_mn[i].strip().split()
            seen = []
            perm = []
            for token in mr_mn:
                if mr_gold.count(token) == 1:
                    perm += [mr_gold.index(token) + 1]
                else:
                    idx = [x + 1 for x, tok in enumerate(mr_gold) if tok == token]
                    perm += [idx[seen.count(token)]]
                seen += [token]
            permutations += [perm]
    data['PERM'] = permutations
    data.to_csv(out_path)


def perm2cycles(perm):
    cycles = []
    seen = []
    for i in range(1, len(perm) + 1):
        if i not in seen:
            cycle = [i]
            seen += [i]
            next_i = i
            while perm[next_i - 1] != i:
                cycle += [perm[next_i - 1]]
                next_i = perm[next_i - 1]
                seen += [next_i]
            cycles += [cycle]
    return cycles


def compute_cayley(cycles):
    cayley = sum([len(c) for c in cycles]) - len(cycles)
    return cayley


def compute_cycles(in_path, out_path):
    data = pd.read_csv(in_path)
    permutations = data.PERM
    all_cycles = []
    all_cayley = []
    for perm in permutations:
        cycles = perm2cycles(eval(perm))
        cayley = compute_cayley(cycles)
        cycles = ' '.join([str(tuple(c)) if len(c) > 1 else f'({str(c[0])})' for c in cycles])
        all_cycles += [cycles]
        all_cayley += [cayley]

    data['CYCLES'] = all_cycles
    data['CAYLEY'] = all_cayley
    data.to_csv(out_path)


def show_cayley_stats(in_path):
    data = pd.read_csv(in_path)
    data = data[data.MN == 0]
    data.hist(column='CAYLEY')
    plt.show()


def count_permutations(n, k):
    if n < 1:
        return 0
    elif n == 1:
        if k != 1:
            return 0
        else:
            return 1
    else:
        return count_permutations(n - 1, k - 1) + (n - 1) * count_permutations(n - 1, k)


def compute_n_permutations(in_path, out_path):
    data = pd.read_csv(in_path)
    permutations = data.PERM
    cayley = data.CAYLEY
    ns = [len(eval(perm)) for perm in permutations]
    ks = [n - c for n, c in zip(ns, cayley)]
    n_permutations = {}
    i = 0
    for n, k in zip(ns, ks):
        print(i)
        if f'{n},{k}' not in n_permutations:
            print(f'\t{n}, {k}')
            n_permutations[f'{n},{k}'] = count_permutations(n, k)
        i += 1
    result = []
    for n, k in zip(ns, ks):
        result += [n_permutations[f'{n},{k}']]

    data['N_PERM'] = result
    data.to_csv(out_path)


def generate_permutations(n, k):
    if k == 1:
        identity = np.arange(1, n + 1)
        return list(np.random.permutation(identity))
    else:
        p = count_permutations(n - 1, k - 1) / count_permutations(n, k)
        prob = np.random.random()
        if prob <= p:
            other_cycle = generate_permutations(n - 1, k - 1)
            n_cycle = [n]
            permutation = [other_cycle, n_cycle]
        else:
            other_cycle = generate_permutations(n - 1, k)
            ran = np.random.randint(1, n)
            for cycle in other_cycle:
                if ran in cycle:
                    i = cycle.index(ran)
                    cycle.insert(i, n)
            permutation = other_cycle
        return permutation


if __name__ == '__main__':
    # compute_permutations('en_args.csv', 'en_args_perm.csv')
    # perm2cycles([4, 5, 7, 6, 8, 2, 1, 3])
    # compute_cycles('en_args_perm.csv', 'en_args_cycles.csv')
    # show_cayley_stats('en_args_cycles.csv')
    # compute_n_permutations('en_args_cycles.csv', 'en_args_n_perm.csv')
    print(count_permutations(4, 2))
    for i in range(50):
        print(generate_permutations(4, 2))
