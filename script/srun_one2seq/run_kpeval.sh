#!/usr/bin/env bash

CUR_DIR=$(dirname "$0")

echo $0
echo $CUR_DIR
# evaluate with all predictions
datasets=(duc inspec semeval kp20k kp20k_valid2k krapivin nus)
#datasets=(kp20k kp20k_valid2k)
beam_widths=(10 25 50)
#beam_widths=(1 10 25 50)
#beam_widths=(50)
for dataset in "${datasets[@]}"
do
    for beam_width in "${beam_widths[@]}"
    do
        # topbeam_terminate ( -i true means ignore_existing pred or eval,  -p true means onepass)
#        ckpt_dir="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/models/keyphrase/meng17-one2seq/meng17-one2seq-kp20k"
#        output_dir="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2seq/meng17-one2seq-kp20k/meng17-one2seq-topbeamends/meng17-one2seq-beam$beam_width-maxlen40"
#        sbatch "$CUR_DIR/"kpeval_cpu.sh -a pred -c $ckpt_dir -o $output_dir -g -1 -b 16 -s $beam_width -l 40 -t topbeam -d ${dataset}
        # CPU+fullbeam
        ckpt_dir="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/models/keyphrase/meng17-one2seq/meng17-one2seq-kp20k-topmodels"
        output_dir="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2seq/meng17-one2seq-kp20k-topmodels/meng17-one2seq-fullbeam/meng17-one2seq-beam$beam_width-maxlen40"
        sbatch "$CUR_DIR/"kpeval_cpu.sh -a pred -c $ckpt_dir -o $output_dir -g -1 -b 16 -s $beam_width -l 40 -t full -d ${dataset}

#         CPU+evaluate with predictions in top sequences -e (no pred necessity)
#        sbatch "$CUR_DIR/"kpeval_cpu.sh -a eval -a report -c /zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/models/keyphrase/meng17-one2seq/meng17-one2seq-kp20k-topmodels -o $output_dir -g 0 -b 32 -s $beam_width -l 40 -t topbeam -p true -d $dataset -e true

    done
done

