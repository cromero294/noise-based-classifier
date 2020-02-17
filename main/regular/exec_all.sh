#!/bin/bash

# declare an array
arr=( "twonorm" "threenorm" "ringnorm" "waveform" )
 
# for loop that iterates over each element in arr
for i in "${arr[@]}"
do
	python main-error-combination-datasets.py $i
done
