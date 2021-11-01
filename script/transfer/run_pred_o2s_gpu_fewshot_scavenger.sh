#!/usr/bin/env bash
PROJECT_DIR="/zfs1/hdaqing/rum20/kp/OpenNMT-kpg-transfer"

CURDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
TEMPLATE_PATH="$CURDIR/kpeval_gpu_scavenger_template.sh"

echo $0
echo $PROJECT_DIR
slurm_output_dir="$PROJECT_DIR/slurm_output"

partition="titanx" # titanx gtx1080 v100
days="1"
random=$RANDOM
task_args="pred" # pred or eval
batch_size=1
beam_size=50
max_length=40
step_base=1

# evaluate with all predictions
datasets=(kp20k openkp kptimes jptimes stackex)
datasets=(kp20k kp20k_valid2k openkp openkp_valid2k kptimes kptimes_valid2k jptimes duc stackex stackex_valid2k)
datasets=(kp20k kp20k_valid2k duc inspec krapivin nus semeval)
datasets=(kp20k_valid2k openkp_valid2k kptimes_valid2k stackex_valid2k)
dataset_list=""

for dataset in "${datasets[@]}"
do
    dataset_list+=" ${dataset}"
done
echo $dataset_list

exp_root_dir="/zfs1/hdaqing/rum20/kp/fairseq-kpg/exps/kp_fewshot10k/"
exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps/kp_mag/"
exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps/kp_mag_fewshot_devbest/"
exp_root_dir="/zfs1/hdaqing/rum20/kp/transfer_exps_v2/tf_DAFT/"

EXP_NAME="sca_$task_args-$partition-$random-bs$beam_size"
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
