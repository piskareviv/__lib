import re
from os import system


s = ""
s += "#pragma GCC optimize(\"O3\")\n"
s += open("aux.hpp").read() + "\n"
s += open("ssa.hpp").read() + "\n"
s += open("IO.hpp").read() + "\n"
s += open("run.cpp").read() + "\n"

s = re.sub("#include \"[\w\.]+\"\n", "", s)


open("submit.cpp", 'w').write(s)
