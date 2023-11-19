import re
from os import system


a = open("montgomery.hpp").read()
b = open("ntt.hpp").read()
c = open("ssa.hpp").read()
d = open("IO.hpp").read()
s = open("run.cpp").read()

# s = s[:s.index('\n') + 1] + a + b + c + s[s.index('\n'):]

s = a + b + c + d + s

s = re.sub("#include \"[\w\.]+\"\n", "", s)


open("submit.cpp", 'w').write(s)
