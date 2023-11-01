from sympy import ntheory as nt
from random import randint
import tqdm

with open("test.in", 'w') as f:
    N = 10**5
    print(N, file=f)
    for i in tqdm.tqdm(range(N)):
        p = nt.randprime(3, 10**9)
        if i < 2e3:
            p = 998_244_353
        if i < 1e3:
            p = nt.randprime(2, 100)
            b = randint(0, p - 1)
        else:
            b = randint(0, p - 1)
            while pow(b, (p - 1) // 2, p) == p - 1:
                b = randint(0, p - 1)
        print(b, p, file=f)
