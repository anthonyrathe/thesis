#!/bin/sh
for i in 0 1 2
do
    for j in 0 1 2
    do
        tmux new -s "super_experiment_$1 $i $j" -d
        tmux send-keys -t "super_experiment_$1 $i $j" "python3 super_experiment_$1.py $i $j" C-m
    done
done
