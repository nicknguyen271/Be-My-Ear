# WLASL2000 Inference Notebooks

Copy these notebooks into:

`E:\Be_My_Ear\notebooks\ASL\inference`

## Files

1. `10_wlasl2000_video_inference_auto_understanding.ipynb`
2. `11_wlasl2000_webcam_realtime_inference.ipynb`

## Required deployed files

These should already exist:

`E:\Be_My_Ear\app\models\ASL\WLASL2000\selected_wlasl2000_model.pt`

`E:\Be_My_Ear\app\models\ASL\WLASL2000\asl_wlasl2000_labels.json`

`E:\Be_My_Ear\app\models\ASL\WLASL2000\wlasl2000_deployment_config.json`

## Required normalisation file

The notebooks try to find the WLASL2000 normalisation stats from:

`E:\Be_My_Ear\models\ASL\WLASL2000`

Example expected file:

`wlasl2000_light_v2_finetuned_from_wlasl1000_train_norm_stats.npz`

## Recommended order

Run video inference first, then webcam inference.