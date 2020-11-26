#!/usr/bin/env bash
PROJECT_DIR="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/"

echo $0
echo $PROJECT_DIR
# evaluate with all predictions
datasets=(duc inspec semeval kp20k kp20k_valid2k krapivin nus)
beam_widths=(1 10 25 50)

# it exceeds 64gb memory, so change to 128gbs
datasets=(kp20k)
beam_widths=(1 10 25 50)


for dataset in "${datasets[@]}"
do
    for beam_width in "${beam_widths[@]}"
    do
        # eval self-terminating based on fullbeam
        ckpt_dir="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/models/keyphrase/meng17-one2seq/meng17-one2seq-kp20k-v3/"
        ckpt_dir="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/models/keyphrase/meng17-one2seq/meng17-one2seq-kp20k-v3_orders/"
        output_dir="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2seq/meng17-one2seq-kp20k-v3/meng17-one2seq-fullbeam/meng17-one2seq-beam$beam_width-maxlen40"
        echo "$PROJECT_DIR/script/srun_one2seq/"kpeval_cpu.sh -a eval -c $ckpt_dir -o $output_dir -b 16 -s $beam_width -l 40 -t full -d ${dataset} -e true
        sbatch "$PROJECT_DIR/script/srun_one2seq/"kpeval_cpu.sh -a eval -c $ckpt_dir -o $output_dir -b 16 -s $beam_width -l 40 -t full -d ${dataset} -e true

    done
done
