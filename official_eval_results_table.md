# RealForensics Official Evaluation Results

Metrics below are computed with the official `stage2` inference pipeline and per-video scores exported by `tools/evaluate_official_scores.py`.
Binary predictions use the repository default decision rule: `mean_logit > 0 => fake`.

| Dataset | Real | Fake | Total | AUC | AP | ACC | F1 | Precision | Recall | Specificity | Balanced ACC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FakeAVCeleb | 50 | 100 | 150 | 0.7408 | 0.8601 | 0.6667 | 0.8000 | 0.6667 | 1.0000 | 0.0000 | 0.5000 |
| LAV-DF | 182 | 294 | 476 | 0.3599 | 0.5286 | 0.6176 | 0.7636 | 0.6176 | 1.0000 | 0.0000 | 0.5000 |

## Confusion Matrices

### FakeAVCeleb

| Actual \\ Pred | Real | Fake |
| --- | ---: | ---: |
| Real | 0 | 50 |
| Fake | 0 | 100 |

- TN = 0
- FP = 50
- FN = 0
- TP = 100

### LAV-DF

| Actual \\ Pred | Real | Fake |
| --- | ---: | ---: |
| Real | 0 | 182 |
| Fake | 0 | 294 |

- TN = 0
- FP = 182
- FN = 0
- TP = 294

## Source Files

- FakeAVCeleb metrics: `E:\data\FakeAVCeleb-rf-official-scores\metrics.json`
- FakeAVCeleb scores: `E:\data\FakeAVCeleb-rf-official-scores\video_scores.csv`
- LAV-DF metrics: `E:\data\LAVDF-rf-official-scores\metrics.json`
- LAV-DF scores: `E:\data\LAVDF-rf-official-scores\video_scores.csv`
