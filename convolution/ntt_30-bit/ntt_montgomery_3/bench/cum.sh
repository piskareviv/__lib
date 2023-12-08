for i in {0..3}; do
    taskset -c $i ./bench_run $2 $3 &
    if [ "$1" = "wait" ]; then
        wait
        break
    fi
done
wait
