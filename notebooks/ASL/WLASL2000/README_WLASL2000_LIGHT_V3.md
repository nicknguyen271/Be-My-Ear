# WLASL2000 Light V3 Notebook

Copy this notebook into:

`E:\Be_My_Ear\notebooks\ASL\WLASL2000`

Notebook:

`09_train_wlasl2000_light_v3_two_stage_finetune.ipynb`

## Purpose

This notebook improves WLASL2000 using:

1. Two-stage fine-tuning
2. Class-balanced focal loss
3. Warmup + cosine learning rate schedule
4. Stronger but safe augmentation
5. Temperature scaling for confidence calibration
6. Light V2 vs Light V3 comparison

## Required files

`E:\Be_My_Ear\data\processed\ASL\WLASL2000\wlasl2000_clean_keypoint_index.csv`

`E:\Be_My_Ear\models\ASL\WLASL1000\bigru_attention_light_v2_wlasl1000.pt`

## Main output

`E:\Be_My_Ear\models\ASL\WLASL2000\light_v3_two_stage_finetuned_from_wlasl1000_wlasl2000.pt`
