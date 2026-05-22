# BOCS-IDS

### Evaluating Benign-Only One-Class Scoring for Progressive Network Intrusion Detection Under Domain Shift

## Overview

BOCS-IDS is a two-stage machine learning-based intrusion detection framework designed to evaluate benign-only anomaly scoring under progressive cross-dataset conditions.

The project combines supervised intrusion detection with one-class anomaly scoring using:

- Isolation Forest (IF)
- One-Class SVM (OCSVM)
- Random Forest (RF)

The framework is evaluated using:

- CIC-IDS2017 (training)
- CSE-CIC-IDS2018 (testing)

to analyze performance degradation caused by temporal domain shift and attack mimicry.

---

## Features

- Progressive cross-dataset IDS evaluation
- Random Forest feature selection
- Isolation Forest anomaly scoring
- One-Class SVM anomaly scoring
- Two-stage IDS architecture
- Seven-configuration ablation study
- McNemar statistical significance testing
- Confusion matrix visualization
- Feature importance analysis

---

## Models Used

### Stage 1 Classifiers

- Decision Tree (DT)
- Random Forest (RF)
- Support Vector Machine (SVM)
- Naive Bayes (NB)
- Artificial Neural Network (ANN)
- Deep Neural Network (DNN)

### Stage 2 One-Class Models

- Isolation Forest (IF)
- One-Class SVM (OCSVM)

---

## Dataset

### Training Dataset
- CIC-IDS2017

### Testing Dataset
- CSE-CIC-IDS2018

Both datasets were preprocessed using:
- duplicate removal
- NaN/infinity removal
- binary label conversion
- class balancing
- feature alignment

---

## Experimental Workflow

1. Dataset preprocessing
2. Feature selection using Random Forest
3. Stage 1 supervised IDS training
4. Progressive cross-dataset evaluation
5. Benign-only anomaly scorer training
6. Anomaly score augmentation
7. Ablation study evaluation
8. Statistical significance analysis

---

## Results

The experiments demonstrate that:

- Continuous anomaly-score augmentation collapses under domain shift.
- Binary threshold-based anomaly flags provide the only transfer-stable improvement.
- Structural attack mimicry causes malicious traffic to resemble benign traffic across datasets.

Key findings include:
- Cohen’s d progression from 0.10 → 0.39
- Progressive evaluation degradation across all baseline models
- Marginal but statistically significant gains using IF binary flags

---

## Repository Structure

```text
BOCS-IDS/
│
├── screenshots/
├── BOCS_IDS_FINAL.ipynb
├── bocs_ids_experiment_v6.py
├── README.md
├── LICENSE
└── .gitignore
```

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
python bocs_ids_experiment_v6.py
```

---

## Technologies Used

- Python
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn

---

## Research Focus

This work investigates:

- Progressive intrusion detection
- Domain shift in IDS systems
- One-class anomaly scoring
- Attack mimicry effects
- Transfer-stable anomaly augmentation

---

## Paper

The complete research paper is included in this repository.

---

## References

1. Chua and Salam, *Evaluation of Machine Learning Algorithms in Network-Based Intrusion Detection Using Progressive Dataset*, Symmetry, 2023.

2. Sharafaldin et al., *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*, ICISSP, 2018.

3. Liu et al., *Isolation Forest*, IEEE ICDM, 2008.

4. Schölkopf et al., *Estimating the Support of a High-Dimensional Distribution*, Neural Computation, 2001.

---

## Author

Varun
B.Tech CSE | Cybersecurity & Machine Learning
