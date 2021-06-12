#!/usr/bin/env bash
cluster="htc"
for i in {196622..196670}
do
   echo "scancel -M $cluster $i"
   scancel -M $cluster $i
done
