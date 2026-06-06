# WLASL1000 Detailed Notebooks

Copy these notebooks into:

`E:\Be_My_Ear\notebooks\ASL\WLASL1000`

Run order:

1. `01_inspect_wlasl1000_dataset.ipynb`
2. `02_extract_wlasl1000_keypoints.ipynb`
3. `03_check_wlasl1000_keypoints.ipynb`
4. `04_train_wlasl1000_bigru_attention.ipynb`
5. `05_evaluate_wlasl1000_bigru_attention.ipynb`

Selected architecture:

`BiGRU + Temporal Attention V1`

Reason:

The WLASL300 V2 model was weaker than WLASL300 V1, so WLASL1000 continues with the stable V1 model using keypoints + velocity features.