import re
from os import system


s = ""
s += "#include <iostream>\n"
s += "#pragma GCC optimize(\"O3\")\n"
s += open("aux.hpp").read() + "\n"
s += open("ntt.hpp").read() + "\n"
s += open("IO.hpp").read() + "\n"
s += open("run.cpp").read() + "\n"

s = re.sub("#include \"[\w\.]+\"\n", "", s)


open("submit.cpp", 'w').write(s)
