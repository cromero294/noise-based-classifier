#!/bin/bash

# declare an array
arr=( "segment" "magic04" )
 
# for loop that iterates over each element in arr
for i in "${arr[@]}"
do
	python main-error-combination-datasets.py $i
done
