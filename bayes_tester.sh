#!/bin/bash

CYAN='\033[1;36m'
NC='\033[0m' # No Color

for w in 1 5 7 10 12 15 20 25 30 35 40 45 50 75 100 125 150 200
do
  for ngram in 1 2 3 4 5
  do
    echo -e ${CYAN}----------- window: $w ngram: $ngram -----------${NC} >> bash_ouput_bayeshashing
    /mnt/c/Users/alexa/Documents/KUL/BigData/Assignment1/Assignment1_BigData/build/src/bdap_assignment1 $w $ngram output >> bash_ouput_bayeshashing
  done
done

