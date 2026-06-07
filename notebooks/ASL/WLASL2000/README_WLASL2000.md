# WLASL2000 Full Clean Notebooks

Copy these notebooks into:

`E:\Be_My_Ear\notebooks\ASL\WLASL2000`

## Run order

1. `01_inspect_wlasl2000_dataset.ipynb`
2. `02_extract_wlasl2000_keypoints.ipynb`
3. `03_check_wlasl2000_keypoints_strict.ipynb`
4. `04_train_wlasl2000_light_v2_from_scratch.ipynb`
5. `05_train_wlasl2000_light_v2_finetune_from_wlasl1000.ipynb`
6. `06_train_wlasl2000_transformer_optional.ipynb`
7. `07_evaluate_wlasl2000_selected_models.ipynb`
8. `08_prepare_wlasl2000_deployment_package.ipynb`

## Main app model candidate

`05_train_wlasl2000_light_v2_finetune_from_wlasl1000.ipynb`

This fine-tunes WLASL2000 from the best WLASL1000 Light V2 model.

## Deployment logic

- confidence >= 0.70 → speak/show Top-1
- 0.40 <= confidence < 0.70 → show Top-5 suggestions
- confidence < 0.40 → ask user to sign again

## Important

The WLASL2000 model is an isolated-sign recognition model. For full real-life translation, later stages need webcam/video inference, prediction smoothing, sentence buffering, NLP sentence generation, and text-to-speech.