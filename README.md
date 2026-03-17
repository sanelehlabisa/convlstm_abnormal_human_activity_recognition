# ConvLSTM Abnormal Human Activity Recognition

Replication and extension of *"A New Approach for Abnormal Human Activities Recognition
Based on ConvLSTM Architecture"*, with multi-dataset support and structured evaluation.

## Model

CNN + LSTM architecture. A per-frame CNN backbone extracts spatial features;
an LSTM models temporal dynamics across the clip. Output: one of 14 activity classes
(13 anomalies + normal), plus a derived binary normal/abnormal verdict.

## Supported Datasets

The model targets **14 canonical classes** drawn from UCF-Crime, the standard benchmark
for surveillance-based abnormal activity recognition. All other datasets are mapped to
this vocabulary via `labels.json` aliases — no retraining required.

| Dataset | Classes | Type | Link |
|---|---|---|---|
| LoDVP Abnormal Activities | 11 | Staged surveillance anomalies | [link](https://kmikt.uniza.sk/ds/abnormal_activities.zip) |
| AirtLab Violence | 2 | Binary violent / non-violent | [link](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos) |
| UCF-Crime *(primary benchmark)* | 14 | Real-world surveillance crime | [link](https://www.crcv.ucf.edu/projects/real-world/) |
| UCF50 / UCF101 | 50 / 101 | General human actions (backbone eval) | [link](https://www.crcv.ucf.edu/data/UCF101.php) |
| Kinetics-400/700 | 400 / 700 | Large-scale action recognition (pretraining) | [link](https://github.com/cvdfoundation/kinetics-dataset) |

## Canonical Classes (14)

Sourced from UCF-Crime. All dataset folder names resolve to one of these via `labels.json`.

| ID | Class | Abnormal |
|---|---|---|
| 0 | normal | ✗ |
| 1 | abuse | ✓ |
| 2 | arrest | ✓ |
| 3 | arson | ✓ |
| 4 | assault | ✓ |
| 5 | burglary | ✓ |
| 6 | explosion | ✓ |
| 7 | fighting | ✓ |
| 8 | road accident | ✓ |
| 9 | robbery | ✓ |
| 10 | shooting | ✓ |
| 11 | shoplifting | ✓ |
| 12 | stealing | ✓ |
| 13 | vandalism | ✓ |

## Usage
```bash
# Train
python -m src.train --dataset_dir dataset_clean --epochs 64 --height 64 --width 64

# Evaluate
python -m src.evaluate --dataset_dir dataset_clean --model_dir models
```

## References

1. LoDVP / ConvLSTM paper — Ullah et al.
2. AirtLab — Castaldi et al.
3. UCF-Crime — Sultani, Chen, Shah (CVPR 2018)