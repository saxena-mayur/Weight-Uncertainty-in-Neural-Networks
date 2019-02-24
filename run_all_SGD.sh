#!/bin/bash
Ps=( 1 2 3 )
for P in "${Ps[@]}"
do

Is=( 1 2 3 )
for I in "${Is[@]}"
do
	# nohup python3 SGD.py $P $I $1 > results/log_$P\_$I.txt &
	CUDA_VISIBLE_DEVICES=1 python3 SGD.py $P $I $1 > results/log_$P\_$I.txt
done
done
watch nvidia-smi
