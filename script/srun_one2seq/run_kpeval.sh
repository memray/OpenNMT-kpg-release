#!/usr/bin/env bash

CUR_DIR=$(dirname "$0")
# evaluate with all predictions
#datasets=(duc inspec semeval kp20k_valid500 kp20k_valid2k krapivin nus)
datasets=(kp20k)
#beam_widths=(1 10 25 50)
beam_widths=(50)
for dataset in "${datasets[@]}"
do
    for beam_width in "${beam_widths[@]}"
    do
        # GPU+topbeam_terminate ( -i true means ignore_existing pred or eval)
#        output_dir="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2seq-topbeamends-exhaustive/meng17-one2seq-beam$beam_width-maxlen40"
#        sbatch "$CUR_DIR/"kpeval_gpu.sh -a pred -a eval -a report -c /zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/models/keyphrase/meng17-one2seq/meng17-one2seq-kp20k-topmodels -o $output_dir -g 0 -b 32 -s $beam_width -l 40 -t topbeam -d ${dataset} -p true
        # CPU+evaluate with predictions in top sequences -e (no pred necessity)
#        sbatch "$CUR_DIR/"kpeval_cpu.sh -a eval -a report -c /zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/models/keyphrase/meng17-one2seq/meng17-one2seq-kp20k-topmodels -o $output_dir -g 0 -b 32 -s $beam_width -l 40 -t topbeam -p true -d $dataset -e true
        # GPU+fullbeam
        output_dir="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2seq-fullbeam/meng17-one2seq-beam$beam_width-maxlen40"
        sbatch "$CUR_DIR/"kpeval_gpu.sh -a pred -a eval -a report -c /zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/models/keyphrase/meng17-one2seq/meng17-one2seq-kp20k-topmodels -o $output_dir -g 0 -b 16 -s $beam_width -l 40 -t full -p true -d ${dataset}
    done
done

