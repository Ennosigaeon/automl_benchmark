#!/bin/sh
for i in 0 1 2 3 4 5 6 7
do
  echo "Starting chunk $i"
  python3 run_hpsklearn.py $i > hpsklearn-$i.log 2>&1 &
done