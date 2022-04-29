#!/usr/bin/env bash
PROJECT_DIR="/zfs1/hdaqing/rum20/kp/OpenNMT-kpg-transfer"

CURDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
TEMPLATE_PATH="$CURDIR/kpeval_gpu_template.sh"

slurm_output_dir="$PROJECT_DIR/slurm_output"

partition="titanx" # gtx1080 titanx v100
days="1"
random=$RANDOM

task_args="pred eval" # pred or eval
batch_size=1
beam_size=50
max_length=40
step_base=1

# evaluate with all predictions
datasets=(kp20k openkp kptimes jptimes stackex)
datasets=(kp20k_valid2k openkp_valid2k kptimes_valid2k stackex_valid2k)
dataset_list=""

for dataset in "${datasets[@]}"
do
    dataset_list+=" ${dataset}"
done

#exp_root_dir="/zfs1/pbrusilovsky/rum20/kp/transfer_exps/kp_transformer_DA/"
#exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps/kp_fewshot-v2/"
#exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps/kp_bart_DA/"
#exp_root_dir="/zfs1/pbrusilovsky/rum20/kp/transfer_exps/kp_transformer_fewshot/"
#exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps/kp_fewshot-v3/"
#exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps/kp_mag_fewshot/"
#exp_root_dir="/zfs1/pbrusilovsky/rum20/kp/transfer_exps/kp_transformer_fewshot_v2/"
#exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps/kp_mag/"
#exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps/tf_mag/"
#exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps/bart_mag_fewshot"

exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps_v2/tf_DA/"
exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps_v2/tf_DAFT/"
exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps_v2/tf_DAcompare/"

exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps_v2/bart_PTDA"
exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps_v2/tf_PTDAFT/"
exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps_v2/tf_PTDA_selftrain"
exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps_v2/tf_DAFT/"
exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps_v2/tf_DAcompareFT/"
exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps_v2/tf_PTDAFT_selftrain"
exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps_v2/bart_DAFT"
exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps_v2/bart_PTDAFT"
exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps_v2/bart_FT"
exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps_v2/bart_PTFT"
exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps_v2/bart_DAFT"
exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps_v2/mag_DAFT/"


echo $0
echo $PROJECT_DIR
echo $dataset_list
echo $exp_root_dir

EXP_NAME="predeval-$partition-$random-devset-bs$beam_size"
DUMP_SCRIPT_PATH="$CURDIR/tmp/$EXP_NAME.sh"
rm -f $DUMP_SCRIPT_PATH

replaces="s/{job_name}/$EXP_NAME/;";
replaces="$replaces s|{partition}|$partition|g;";
replaces="$replaces s|{days}|$days|g;";
replaces="$replaces s|{task_args}|$task_args|g;";
replaces="$replaces s|{dataset_args}|$dataset_list|g;";
replaces="$replaces s|{exp_root_dir}|$exp_root_dir|g;";
replaces="$replaces s|{slurm_output_dir}|$slurm_output_dir|g;";
replaces="$replaces s|{batch_size}|$batch_size|g;";
replaces="$replaces s|{max_length}|$max_length|g;";
replaces="$replaces s|{beam_size}|$beam_size|g;";
replaces="$replaces s|{step_base}|$step_base|g;";
cat ${TEMPLATE_PATH} | sed -e "$replaces" > ${DUMP_SCRIPT_PATH}

echo $EXP_NAME
echo $DUMP_SCRIPT_PATH

sbatch $DUMP_SCRIPT_PATH
