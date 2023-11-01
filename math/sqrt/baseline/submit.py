import re
from os import system


# a = open("montgomery.hpp").read()
a = ""
b = open("sqrt_mtg.hpp").read()
c = open("IO.hpp").read()
s = open("run.cpp").read()

# s = s[:s.index('\n') + 1] + a + b + c + s[s.index('\n'):]

s = a + b + c + s

s = re.sub("#include \"[\w\.]+\"\n", "", s)

# s = """// thanks https://judge.yosupo.jp/submission/142782

# """ + s

open("submit.cpp", 'w').write(s)
