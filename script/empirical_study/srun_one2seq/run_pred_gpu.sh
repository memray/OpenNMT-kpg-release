#!/usr/bin/env bash
PROJECT_DIR="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/"

echo $0
echo $PROJECT_DIR
# evaluate with all predictions
#datasets=(duc inspec semeval kp20k kp20k_valid2k krapivin nus)
datasets=(kp20k)
beam_widths=(50)


#beam_widths=(1 10 25 50)
for dataset in "${datasets[@]}"
do
    for beam_width in "${beam_widths[@]}"
    do
        # topbeam_terminate ( -i true means ignore_existing pred or eval,  -p true means onepass)
#        ckpt_dir="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/models/keyphrase/meng17-one2seq/meng17-one2seq-kp20k"
#        output_dir="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2seq/meng17-one2seq-kp20k/meng17-one2seq-topbeamends/meng17-one2seq-beam$beam_width-maxlen40"
#        echo "$PROJECT_DIR/script/srun_one2seq/"kpeval_gpu.sh -a pred -c $ckpt_dir -o $output_dir -b 16 -s $beam_width -l 40 -t topbeam -d ${dataset}
#        sbatch "$PROJECT_DIR/script/srun_one2seq/"kpeval_gpu.sh -a pred -c $ckpt_dir -o $output_dir -b 16 -s $beam_width -l 40 -t topbeam -d ${dataset}
        # CPU+fullbeam
        ckpt_dir="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/models/keyphrase/meng17-one2seq/meng17-one2seq-kp20k-topmodels"
        output_dir="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2seq/meng17-one2seq-kp20k-topmodels/meng17-one2seq-fullbeam/meng17-one2seq-beam$beam_width-maxlen40"
        echo "$PROJECT_DIR/script/srun_one2seq/"kpeval_gpu.sh -a pred -c $ckpt_dir -o $output_dir -b 16 -s $beam_width -l 40 -t full -d ${dataset}
        sbatch "$PROJECT_DIR/script/srun_one2seq/"kpeval_gpu.sh -a pred -c $ckpt_dir -o $output_dir -b 16 -s $beam_width -l 40 -t full -d ${dataset}

    done
done


