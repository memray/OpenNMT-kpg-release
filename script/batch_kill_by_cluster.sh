#!/usr/bin/env bash
cluster="htc"
cluster="smp"

#ids=$(crc-squeue.py | awk  '{printf "%s %s", $1 $2}')
#pars=$(crc-squeue.py | awk  '{printf "%s",  $2}')

#ids=(`squeue -M smp -u rum20 -o '%.7i'`)
#echo ${#ids[@]}
#echo $ids

squeue -M $cluster -u rum20 -o '%.7i' | while IFS= read -r line ; do
    echo "$line"
    echo "scancel -M $cluster $line"
    scancel -M $cluster $line
done

