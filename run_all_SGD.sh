#!/bin/bash
Ps=( 1 2 3 4 )
for P in "${Ps[@]}"
do

Is=( 1 2 3 )
for I in "${Is[@]}"
do
	# nohup python3 SGD.py $P $I $1 > results/log_$P\_$I.txt &
	CUDA_VISIBLE_DEVICES=0 python3 SGD.py $P $I $1 > results/log_$P\_$I.txt
done
done
watch nvidia-smi
