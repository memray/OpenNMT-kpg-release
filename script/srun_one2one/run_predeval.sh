#!/usr/bin/env bash
PROJECT_DIR="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/"

CURDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
TEMPLATE_PATH="$CURDIR/eval_cpu_template.sh"

echo $0
echo $PROJECT_DIR
echo $CURDIR

batch_size="8"
max_length="6"
task_args="eval" # pred or eval

# evaluate with all predictions
datasets=(duc inspec semeval kp20k kp20k_valid2k krapivin nus)
#datasets=(duc inspec semeval kp20k_valid2k krapivin nus)
datasets=(kp20k)
beam_size="200" # 8, 16, 32, 64


for dataset in "${datasets[@]}"
do
    # V1 models ( -i true means ignore_existing pred or eval, -p true means onepass)
#    ckpt_dir="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/models/keyphrase/meng17-one2one/meng17-one2one-kp20k-topmodels/"
#    output_dir="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2one/meng17-one2one-kp20k-topmodels/meng17-one2one-fullbeam/meng17-one2one-beam$beam_size-maxlen6/"
#    EXP_NAME="one2one-beamsearch-$task_args-$dataset-beam$beam_size"
#    DUMP_SCRIPT_PATH="$CURDIR/$EXP_NAME.sh"
#    replaces="s/{job_name}/$EXP_NAME/;";
#    replaces="$replaces s|{task_args}|$task_args|g;";
#    replaces="$replaces s|{dataset_args}|$dataset|g;";
#    replaces="$replaces s|{ckpt_dir}|$ckpt_dir|g;";
#    replaces="$replaces s|{output_dir}|$output_dir|g;";
#    replaces="$replaces s|{slurm_output_dir}|$PROJECT_DIR/slurm_output|g;";
#    replaces="$replaces s|{batch_size}|$batch_size|g;";
#    replaces="$replaces s|{max_length}|$max_length|g;";
#    replaces="$replaces s|{beam_size}|$beam_size|g;";
#    cat ${TEMPLATE_PATH} | sed -e "$replaces" > ${DUMP_SCRIPT_PATH}
#    echo $EXP_NAME
#    sbatch ${DUMP_SCRIPT_PATH}

    # selected V1 models
#    ckpt_dir="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/models/keyphrase/meng17-one2one/meng17-one2one-kp20k-topmodels/"
#    output_dir="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2one/meng17-one2one-kp20k-topmodels/meng17-one2one-fullbeam/meng17-one2one-beam$beam_size-maxlen6/"
#    EXP_NAME="one2one-v1top-beamsearch-$task_args-$dataset-beam$beam_size"
#    DUMP_SCRIPT_PATH="$CURDIR/$EXP_NAME.sh"
#    replaces="s/{job_name}/$EXP_NAME/;";
#    replaces="$replaces s|{task_args}|$task_args|g;";
#    replaces="$replaces s|{dataset_args}|$dataset|g;";
#    replaces="$replaces s|{ckpt_dir}|$ckpt_dir|g;";
#    replaces="$replaces s|{output_dir}|$output_dir|g;";
#    replaces="$replaces s|{slurm_output_dir}|$PROJECT_DIR/slurm_output|g;";
#    cat ${TEMPLATE_PATH} | sed -e "$replaces" > ${DUMP_SCRIPT_PATH}
#    sbatch ${DUMP_SCRIPT_PATH}

#    # V2 models
    ckpt_dir="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/models/keyphrase/meng17-one2one/meng17-one2one-kp20k-v2/"
    output_dir="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2one/meng17-one2one-kp20k-v2/meng17-one2one-fullbeam/meng17-one2one-beam$beam_size-maxlen6/"
    EXP_NAME="one2one-v2-beamsearch-$task_args-$dataset-beam$beam_size"
    DUMP_SCRIPT_PATH="$CURDIR/$EXP_NAME.sh"
    replaces="s/{job_name}/$EXP_NAME/;";
    replaces="$replaces s|{task_args}|$task_args|g;";
    replaces="$replaces s|{dataset_args}|$dataset|g;";
    replaces="$replaces s|{ckpt_dir}|$ckpt_dir|g;";
    replaces="$replaces s|{output_dir}|$output_dir|g;";
    replaces="$replaces s|{slurm_output_dir}|$PROJECT_DIR/slurm_output|g;";
    cat ${TEMPLATE_PATH} | sed -e "$replaces" > ${DUMP_SCRIPT_PATH}
    sbatch ${DUMP_SCRIPT_PATH}

#         CPU+evaluate with predictions in top sequences -e (no pred necessity)
#        sbatch "$PROJECT_DIR/"kpeval_cpu.sh -a eval -a report -c /zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/models/keyphrase/meng17-one2one/meng17-one2one-kp20k-topmodels -o $output_dir -g 0 -b 32 -s $beam_width -l 40 -t topbeam -p true -d $dataset -e true

done
