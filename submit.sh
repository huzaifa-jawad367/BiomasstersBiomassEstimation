#!/usr/bin/env sh


python \
    ./src/submit.py \
    --test-df ./data/features_metadata.csv \
    --test-images-dir ./data/test_features \
    --model-path ./models/tf_efficientnetv2_l_in21k_f0_b4x1_e20_nrmse_devscse_attnlin_augs_decplus7/modelo_best.pth \
    --tta 4 \
    --batch-size 4 \
    --out-dir ./preds \
