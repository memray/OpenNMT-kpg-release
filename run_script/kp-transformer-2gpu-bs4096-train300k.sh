#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:2
#SBATCH --partition=titanx
#SBATCH --partition=gtx1080
#SBATCH --job-name=train-kp
#SBATCH --output=slurm_output/train-magkp-transformer-L4H8-DIM512-DO01-TTTT.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=6-00:00:00 # 6 days walltime in dd-hh:mm format
#SBATCH --qos=long

# Load modules
#module restore
#module load cuda/10.0.130
#module load python/anaconda3.6-5.2.0
#source activate py36
#module unload python/anaconda3.6-5.2.0

# Run the job
export DATA_NAME="magkp"
export TOKEN_NAME="meng17"
export TARGET_TYPE="one2one"

export LAYER=4
export HEADS=8
export EMBED=512
export HIDDEN=512
export Dropout="0.1"
export ContextGate="both"
export Copy=true
export ReuseCopy=true
export Cov=true
export PositionEncoding=true

export EXP_NAME="$DATA_NAME-$TARGET_TYPE-transformer-Layer$LAYER-Heads$HEADS-Dim$HIDDEN-Emb$EMBED-Dropout$Dropout-Copy$Copy-Cov$Cov-Context$ContextGate"

export ROOT_PATH="/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg"
export DATA_PATH="data/keyphrase/$TOKEN_NAME/$DATA_NAME"
export MODEL_PATH="models/keyphrase/$TOKEN_NAME/$EXP_NAME"
export LOG_PATH="output/keyphrase/$TOKEN_NAME/$EXP_NAME.log"
export TENSORBOARD_PATH="runs/keyphrase/$TOKEN_NAME/$EXP_NAME/"

cmd="/ihome/pbrusilovsky/rum20/.conda/envs/py36/bin/python train.py -config config/train/config-transformer-keyphrase.yml -exp $EXP_NAME -data $DATA_PATH -save_model $MODEL_PATH -log_file $LOG_PATH -tensorboard_log_dir $TENSORBOARD_PATH -layers $LAYER -heads $HEADS -word_vec_size $EMBED -rnn_size $HIDDEN -dropout $Dropout -context_gate $ContextGate"

if [ "$Copy" = true ] ; then
    cmd+=" -copy_attn"
fi
if [ "$ReuseCopy" = true ] ; then
    cmd+=" -reuse_copy_attn"
fi
if [ "$Cov" = true ] ; then
    cmd+=" -coverage_attn"
fi
if [ "$PositionEncoding" = true ] ; then
    cmd+=" -position_encoding"
fi

echo $cmd

srun $cmd