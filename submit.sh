#!/usr/bin/env sh


python \
    ./src/submit.py \
    --test-df ./data/features_metadata.csv \
    --test-images-dir ./data/test_features \
    --model-path ./models/pretrained_by_authors_f0_b8x2_e100_nrmse_devscse_attnlin_augs_decplus7_plus800eb_200ft/modelo_best.pth \
    --tta 4 \
    --batch-size 4 \
    --out-dir ./preds \
