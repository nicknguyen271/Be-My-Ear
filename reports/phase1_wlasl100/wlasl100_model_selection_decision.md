# WLASL100 Model Selection Decision

## Selected model for WLASL300 scaling

**BiGRU + Temporal Attention**

## Reason

The BiGRU + Temporal Attention model outperformed the Small Transformer Encoder on all key WLASL100 metrics.

| Metric | BiGRU + Attention | Transformer |
|---|---:|---:|
| Best Validation F1 | 0.4723 | 0.3740 |
| Best Validation Top-5 | 0.8221 | 0.7592 |
| Test Top-1 Accuracy | 0.4304 | 0.3354 |
| Test Top-3 Accuracy | 0.7089 | 0.6013 |
| Test Top-5 Accuracy | 0.7911 | 0.6709 |
| Test Macro F1 | 0.3855 | 0.2830 |

## Interpretation

The Transformer model was useful as a comparison, but it underperformed on WLASL100. This is likely because WLASL100 has only 1,013 clean samples across 100 classes, making it too small for the Transformer to learn robust temporal patterns.

The BiGRU + Temporal Attention model is more suitable for the current dataset size because it learns movement sequences efficiently while still using attention to focus on important frames.

## Next step

Scale the BiGRU + Temporal Attention architecture to WLASL300.

Planned notebooks:

1. `09_prepare_wlasl300_dataset.ipynb`
2. `10_extract_wlasl300_keypoints.ipynb`
3. `11_train_bigru_attention_wlasl300.ipynb`
4. `12_evaluate_wlasl300_model.ipynb`

## Note

The Transformer should not be discarded permanently. It can be tested again later on WLASL1000 or WLASL2000, where there may be enough data for it to become more effective.
