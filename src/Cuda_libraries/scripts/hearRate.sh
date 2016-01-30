#!/bin/bash

fN="`hostname`-`date +\"%F_%T\"`.log"

#Header
touch $fN
vmstat >> $fN

while [ 0 -eq 0 ] 
do
  sleep .1s
  vmstat | tail -n 1 >> $fN
done  
