from os import system
from random import randint

system(f"g++ sb.cpp -DQUEIT -O2 -std=c++17 -o inc")
system(f"g++ sb2.cpp -DQUEIT -O2 -std=c++17 -o cor")

for i in range(0, 21):
    with open("cum_test.in", 'w') as f:
        print(i, file=f)
        # print(*[randint(0, 1) for i in range(2**i)], file=f)
        # print(*[randint(0, 1) for i in range(2**i)], file=f)
        print(*[randint(0, 998_244_353 - 1) for i in range(2**i)], file=f)
        print(*[randint(0, 998_244_353 - 1) for i in range(2**i)], file=f)
    system(f"./cor < cum_test.in > cor.out 2> cor.err")
    system(f"./inc < cum_test.in > inc.out 2> inc.err")
    a = system("diff cor.out inc.out -q")
    if a:
        print("fuck", i)
        exit(0)
