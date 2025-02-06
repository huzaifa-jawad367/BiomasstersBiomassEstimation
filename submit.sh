#!/usr/bin/env sh


python \
    ./src/submit.py \
    --test-df ./data/features_metadata.csv \
    --test-images-dir ./data/test_features \
    --model-path ./models/Model_veg_indices/tf_efficientnetv2_xl_in21k_f0_b2x1_e50_nrmse_devscse_attnlin_augs_decplus7_plus800eb_200ft/modelo_best.pth \
    --tta 4 \
    --batch-size 2 \
    --out-dir ./pred2 \
