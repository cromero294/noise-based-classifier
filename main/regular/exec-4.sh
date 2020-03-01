#!/bin/bash

# declare an array
arr=( "australian" "diabetes" "german" "magic04" "heart" "ionosphere" "new-thyroid" "ringnorm" "segment" "threenorm" "tic-tac-toe" "twonorm" "waveform" "wdbc" "wine" )

# for loop that iterates over each element in arr
for i in "${arr[@]}"
do
	python prueba2.py $i
done
