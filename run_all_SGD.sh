#!/bin/bash

MODE=dropout

Ps=( 2 3 ) # for each P in Ps: lr=1e-$P
for P in "${Ps[@]}"
do

Is=( 400 800 1200 ) # hidden units
for I in "${Is[@]}"
do
	# nohup python3 SGD.py $P $I $1 > results/log_$P\_$I.txt &
	CUDA_VISIBLE_DEVICES=1 python3 SGD.py $P $I $1 $MODE > Results/SGD_MNIST_$MODE\_$I\_1e-$P\.txt
done
done