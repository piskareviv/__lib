import re
from os import system


a = open("barrett.hpp").read()
b = open("fft.hpp").read()
c = open("IO.hpp").read()
s = open("submit0.cpp").read()

s = s[:s.index('\n') + 1] + a + b + c + s[s.index('\n'):]

s = re.sub("#include \"[\w\.]+\"\n", "", s)

open("submit.cpp", 'w').write(s)
