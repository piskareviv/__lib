taskset -c 3 perf stat -r 10 -d -e cycles,task-clock,instructions,\
uops_dispatched.port_0,\
uops_dispatched.port_1,\
uops_dispatched.port_5,\
uops_dispatched.port_2_3,\
uops_dispatched.port_4_9,\
uops_dispatched.port_6,\
uops_dispatched.port_7_8,\
uops_executed.core,\
slots taskset -c 3 $1