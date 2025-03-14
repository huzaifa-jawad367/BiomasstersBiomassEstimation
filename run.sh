#!/usr/bin/env sh


set -eu  # o pipefail

GPU=${GPU:-0, 1}
PORT=${PORT:-29500}
N_GPUS=${N_GPUS:-2} # change to your number of GPUs

OPTIM=adamw
LR=0.001
WD=0.01

SCHEDULER=cosa
MODE=epoch

N_EPOCHS=50
T_MAX=50
loss=nrmse
attn=scse
data_dir=./data
chkps_dir=./models/Model_veg_indices_2

backbone=tf_efficientnetv2_xl_in21k
BS=2
FOLD=0

CHECKPOINT=$chkps_dir/"${backbone}"_f"${FOLD}"_b"${BS}"x"${N_GPUS}"_e"${N_EPOCHS}"_"${loss}"_devscse_attnlin_augs_decplus7_Defualt_config
MASTER_PORT="${PORT}" CUDA_VISIBLE_DEVICES="${GPU}" torchrun --nproc_per_node="${N_GPUS}" \
    ./src/train.py \
        --train-df $data_dir/features_metadata.csv \
        --train-images-dir $data_dir/Volumes/Samsung_T5/BIOMASS/BioMasster_Dataset/v1/DrivenData/train_features \
        --train-labels-dir $data_dir/train_agbm \
        --backbone "${backbone}" \
        --loss "${loss}" \
        --in-channels 19 \
        --optim "${OPTIM}" \
        --learning-rate "${LR}" \
        --weight-decay "${WD}" \
        --scheduler "${SCHEDULER}" \
        --T-max "${T_MAX}" \
        --num-epochs "${N_EPOCHS}" \
        --checkpoint-dir "${CHECKPOINT}" \
        --fold "${FOLD}" \
        --scheduler-mode "${MODE}" \
        --batch-size "${BS}" \
        --augs \
        --dec-attn-type $attn \
        --dec-channels 512 448 384 320 256 \
        --fp16 \
        --grad-accum 2 \


LR=0.0001
N_EPOCHS=50
T_MAX=50
CHECKPOINT_LOAD=$chkps_dir/"${backbone}"_f"${FOLD}"_b"${BS}"x"${N_GPUS}"_e"${N_EPOCHS}"_"${loss}"_devscse_attnlin_augs_decplus7
CHECKPOINT=$chkps_dir/"${backbone}"_f"${FOLD}"_b"${BS}"x"${N_GPUS}"_e"${N_EPOCHS}"_"${loss}"_devscse_attnlin_augs_decplus7_plus800eb

MASTER_PORT="${PORT}" CUDA_VISIBLE_DEVICES="${GPU}" torchrun --nproc_per_node="${N_GPUS}" \
    ./src/train.py \
        --train-df $data_dir/features_metadata.csv \
        --train-images-dir $data_dir/Volumes/Samsung_T5/BIOMASS/BioMasster_Dataset/v1/DrivenData/train_features \
        --train-labels-dir $data_dir/train_agbm \
        --backbone "${backbone}" \
        --loss "${loss}" \
        --in-channels 19 \
        --optim "${OPTIM}" \
        --learning-rate "${LR}" \
        --weight-decay "${WD}" \
        --scheduler "${SCHEDULER}" \
        --T-max "${T_MAX}" \
        --num-epochs "${N_EPOCHS}" \
        --checkpoint-dir "${CHECKPOINT}" \
        --fold "${FOLD}" \
        --scheduler-mode "${MODE}" \
        --batch-size "${BS}" \
        --load $CHECKPOINT_LOAD/model_last.pth \
        --augs \
        --dec-attn-type $attn \
        --dec-channels 512 448 384 320 256 \
        --fp16 \
        --grad-accum 2 \


N_EPOCHS=50
T_MAX=50
CHECKPOINT_LOAD=$chkps_dir/"${backbone}"_f"${FOLD}"_b"${BS}"x"${N_GPUS}"_e"${N_EPOCHS}"_"${loss}"_devscse_attnlin_augs_decplus7_plus800eb
CHECKPOINT=$chkps_dir/"${backbone}"_f"${FOLD}"_b"${BS}"x"${N_GPUS}"_e"${N_EPOCHS}"_"${loss}"_devscse_attnlin_augs_decplus7_plus800eb_100ft
MASTER_PORT="${PORT}" CUDA_VISIBLE_DEVICES="${GPU}" torchrun --nproc_per_node="${N_GPUS}" \
    ./src/train.py \
        --train-df $data_dir/features_metadata.csv \
        --train-images-dir $data_dir/Volumes/Samsung_T5/BIOMASS/BioMasster_Dataset/v1/DrivenData/train_features \
        --train-labels-dir $data_dir/train_agbm \
        --backbone "${backbone}" \
        --loss "${loss}" \
        --in-channels 19 \
        --optim "${OPTIM}" \
        --learning-rate "${LR}" \
        --weight-decay "${WD}" \
        --scheduler "${SCHEDULER}" \
        --T-max "${T_MAX}" \
        --num-epochs "${N_EPOCHS}" \
        --checkpoint-dir "${CHECKPOINT}" \
        --fold "${FOLD}" \
        --scheduler-mode "${MODE}" \
        --batch-size "${BS}" \
        --load $CHECKPOINT_LOAD/model_last.pth \
        --augs \
        --dec-attn-type $attn \
        --dec-channels 512 448 384 320 256 \
        --fp16 \
        --ft \
        --grad-accum 2 \


CHECKPOINT_LOAD=$chkps_dir/"${backbone}"_f"${FOLD}"_b"${BS}"x"${N_GPUS}"_e"${N_EPOCHS}"_"${loss}"_devscse_attnlin_augs_decplus7_plus800eb_100ft
CHECKPOINT=$chkps_dir/"${backbone}"_f"${FOLD}"_b"${BS}"x"${N_GPUS}"_e"${N_EPOCHS}"_"${loss}"_devscse_attnlin_augs_decplus7_plus800eb_200ft
MASTER_PORT="${PORT}" CUDA_VISIBLE_DEVICES="${GPU}" torchrun --nproc_per_node="${N_GPUS}" \
    ./src/train.py \
        --train-df $data_dir/features_metadata.csv \
        --train-images-dir $data_dir/Volumes/Samsung_T5/BIOMASS/BioMasster_Dataset/v1/DrivenData/train_features \
        --train-labels-dir $data_dir/train_agbm \
        --backbone "${backbone}" \
        --loss "${loss}" \
        --in-channels 19 \
        --optim "${OPTIM}" \
        --learning-rate "${LR}" \
        --weight-decay "${WD}" \
        --scheduler "${SCHEDULER}" \
        --T-max "${T_MAX}" \
        --num-epochs "${N_EPOCHS}" \
        --checkpoint-dir "${CHECKPOINT}" \
        --fold "${FOLD}" \
        --scheduler-mode "${MODE}" \
        --batch-size "${BS}" \
	--load $CHECKPOINT_LOAD/model_last.pth \
        --augs \
        --dec-attn-type $attn \
        --dec-channels 512 448 384 320 256 \
        --fp16 \
        --ft \
        --grad-accum 2 \
