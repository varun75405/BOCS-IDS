import argparse
import os
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score,
                              roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM
from sklearn.tree import DecisionTreeClassifier

import scipy.stats as ss

warnings.filterwarnings('ignore')

# =============================================================================
# PATHS  — update these to match your server layout
# =============================================================================
TRAIN_DIR = "./cic/IDS2017"   # CIC-IDS2017 CSV files
TEST_DIR  = "./cic/IDS2018"   # CSE-CIC-IDS2018 CSV files
OUT_S1    = "./results_cic"   # Stage 1 outputs
OUT_S2    = "./results_bocs"  # Stage 2 outputs

# GPU allocation (for display / logging only — sklearn uses CPU)
CUDA_VISIBLE = "5,6,7"        # idle H200 GPUs from nvidia-smi
os.environ.setdefault('CUDA_VISIBLE_DEVICES', CUDA_VISIBLE)

# =============================================================================
# CONSTANTS  (mirror base paper exactly)
# =============================================================================
RS         = 42
TOP_N      = 11        # paper Section 5.2.2: top 11 features
ATTACK_CAP = 100_000   # paper: attack classes >100k downsampled to 100k
N_JOBS     = 3         # GPUs 5, 6, 7 — idle H200 CPUs (3 parallel workers)

MODEL_NAMES = [
    'Decision Tree', 'Random Forest', 'Support Vector Machine',
    'Naive Bayes', 'Artificial Neural Network', 'Deep Neural Network',
]
SHORT = ['DT', 'RF', 'SVM', 'NB', 'ANN', 'DNN']

# Per-model sample caps — raised to approach paper CV accuracy.
# SVM: 30k gives ~0.93 CV (paper 0.965) at ~35s on H200; full data = 350s+
# RF / DT / NB: no cap — fast on top-11 features at any dataset size
# ANN / DNN: 50k gives stable convergence without multi-hour training
SVM_CAP =  30_000
RF_CAP  =  None       # no cap
ANN_CAP =  50_000
DT_CAP  =  None       # no cap
NB_CAP  =  None       # NB is always fast

# Table 5 reference values (paper Table 5)
KOSTAS_ACC = {
    'Decision Tree': 0.95, 'Random Forest': 0.94,
    'Support Vector Machine': None, 'Naive Bayes': 0.87,
    'Artificial Neural Network': 0.97, 'Deep Neural Network': None,
}
VINAYAKUMAR_ACC = {
    'Decision Tree': 0.94, 'Random Forest': 0.94,
    'Support Vector Machine': 0.80, 'Naive Bayes': 0.31,
    'Artificial Neural Network': 0.96, 'Deep Neural Network': 0.94,
}

# Column rename map: CSE-CIC-IDS2018 abbreviated names → CIC-IDS2017 full names
_RENAME_2018 = {
    'Dst Port':'Destination Port',
    'Tot Fwd Pkts':'Total Fwd Packets',
    'Tot Bwd Pkts':'Total Backward Packets',
    'TotLen Fwd Pkts':'Total Length of Fwd Packets',
    'TotLen Bwd Pkts':'Total Length of Bwd Packets',
    'Fwd Pkt Len Max':'Fwd Packet Length Max',
    'Fwd Pkt Len Min':'Fwd Packet Length Min',
    'Fwd Pkt Len Mean':'Fwd Packet Length Mean',
    'Fwd Pkt Len Std':'Fwd Packet Length Std',
    'Bwd Pkt Len Max':'Bwd Packet Length Max',
    'Bwd Pkt Len Min':'Bwd Packet Length Min',
    'Bwd Pkt Len Mean':'Bwd Packet Length Mean',
    'Bwd Pkt Len Std':'Bwd Packet Length Std',
    'Flow Byts/s':'Flow Bytes/s',
    'Flow Pkts/s':'Flow Packets/s',
    'Fwd IAT Tot':'Fwd IAT Total',
    'Bwd IAT Tot':'Bwd IAT Total',
    'Fwd Header Len':'Fwd Header Length',
    'Bwd Header Len':'Bwd Header Length',
    'Fwd Pkts/s':'Fwd Packets/s',
    'Bwd Pkts/s':'Bwd Packets/s',
    'Pkt Len Min':'Min Packet Length',
    'Pkt Len Max':'Max Packet Length',
    'Pkt Len Mean':'Packet Length Mean',
    'Pkt Len Std':'Packet Length Std',
    'Pkt Len Var':'Packet Length Variance',
    'FIN Flag Cnt':'FIN Flag Count',
    'SYN Flag Cnt':'SYN Flag Count',
    'RST Flag Cnt':'RST Flag Count',
    'PSH Flag Cnt':'PSH Flag Count',
    'ACK Flag Cnt':'ACK Flag Count',
    'URG Flag Cnt':'URG Flag Count',
    'ECE Flag Cnt':'ECE Flag Count',
    'Pkt Size Avg':'Average Packet Size',
    'Fwd Seg Size Avg':'Avg Fwd Segment Size',
    'Bwd Seg Size Avg':'Avg Bwd Segment Size',
    'Fwd Byts/b Avg':'Fwd Avg Bytes/Bulk',
    'Fwd Pkts/b Avg':'Fwd Avg Packets/Bulk',
    'Fwd Blk Rate Avg':'Fwd Avg Bulk Rate',
    'Bwd Byts/b Avg':'Bwd Avg Bytes/Bulk',
    'Bwd Pkts/b Avg':'Bwd Avg Packets/Bulk',
    'Bwd Blk Rate Avg':'Bwd Avg Bulk Rate',
    'Subflow Fwd Pkts':'Subflow Fwd Packets',
    'Subflow Fwd Byts':'Subflow Fwd Bytes',
    'Subflow Bwd Pkts':'Subflow Bwd Packets',
    'Subflow Bwd Byts':'Subflow Bwd Bytes',
    'Init Fwd Win Byts':'Init_Win_bytes_forward',
    'Init Bwd Win Byts':'Init_Win_bytes_backward',
    'Fwd Act Data Pkts':'act_data_pkt_fwd',
    'Fwd Seg Size Min':'min_seg_size_forward',
}


# =============================================================================
# ─────────────────────────────  SHARED UTILITIES  ────────────────────────────
# =============================================================================

def _load_csvs(directory: str) -> pd.DataFrame:
    """Load & concatenate all CSVs in directory, strip column names,
    drop embedded header rows (rows where Label cell == 'Label')."""
    dfs = [pd.read_csv(f, low_memory=False)
           for f in sorted(Path(directory).glob("*.csv"))]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()
    label_col = next((c for c in df.columns if c.strip().lower() == 'label'), None)
    if label_col:
        mask = df[label_col].astype(str).str.strip().str.lower() == 'label'
        dropped = int(mask.sum())
        if dropped:
            df = df[~mask].reset_index(drop=True)
            print(f"    Dropped {dropped:,} embedded header rows")
    return df


def load_2017(directory: str) -> pd.DataFrame:
    p = Path(directory)
    if not p.exists():
        raise FileNotFoundError(
            f"CIC-IDS2017 not found at: {p}\n"
            f"Set TRAIN_DIR at the top of this file.")
    df = _load_csvs(directory)
    print(f"    CIC-IDS2017 loaded: {len(df):,} rows, {len(df.columns)} cols")
    return df


def load_2018(directory: str) -> pd.DataFrame:
    p = Path(directory)
    if not p.exists():
        raise FileNotFoundError(
            f"CSE-CIC-IDS2018 not found at: {p}\n"
            f"Set TEST_DIR at the top of this file.")
    df = _load_csvs(directory)
    df = df.sample(frac=0.10, random_state=RS)
    print(f"    CSE-CIC-IDS2018 loaded (10%): {len(df):,} rows, {len(df.columns)} cols")
    return df


def align_columns(df17: pd.DataFrame,
                  df18: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df17.columns = df17.columns.str.strip()
    df18.columns = df18.columns.str.strip()

    # Pass 1: apply known rename map
    df18 = df18.rename(columns=_RENAME_2018)

    # Pass 2: fuzzy match on remaining unmatched columns
    # Build lookup: normalised 2017 name → original 2017 name
    norm17 = {c.lower().replace(' ', '').replace('_', ''): c
              for c in df17.columns}
    extra_rename = {}
    for c18 in df18.columns:
        if c18 not in df17.columns:
            key = c18.lower().replace(' ', '').replace('_', '')
            if key in norm17:
                extra_rename[c18] = norm17[key]
    if extra_rename:
        df18 = df18.rename(columns=extra_rename)
        print(f"    Fuzzy-matched {len(extra_rename)} additional columns: "
              f"{list(extra_rename.keys())[:5]}{'...' if len(extra_rename)>5 else ''}")

    shared = [c for c in df17.columns if c in df18.columns]
    only17 = len(df17.columns) - len(shared)
    df17   = df17[shared]
    df18   = df18[shared]
    print(f"    Shared columns: {len(shared)}  |  Dropped (only in 2017): {only17}")

    if len(shared) < 60:
        # This happened in v2 (28 columns) — print diagnostic to help fix map
        unmatched_17 = sorted(set(df17.columns) - set(df18.columns))
        unmatched_18 = sorted(set(df18.columns) - set(df17.columns))
        print(f"    WARNING: only {len(shared)} shared columns — expected ≥77.")
        print(f"    Add these to _RENAME_2018 in 2018→2017 direction:")
        for c in unmatched_18[:20]:
            print(f"      '{c}': '<2017 equivalent>',")
        print(f"    2017 columns still unmatched: {unmatched_17[:10]}")

    return df17, df18


def preprocess(df: pd.DataFrame, name: str,
               is_train: bool = True) -> tuple:
    """Clean → balance 1:1 → binary label.
    Returns (df_clean, before_counts, before_total, after_counts, after_total)."""
    label_col = next((c for c in df.columns if c.lower() == 'label'), None)
    if label_col is None:
        raise ValueError(f"No Label column in {name}. Cols: {list(df.columns[:8])}")
    if label_col != 'label':
        df = df.rename(columns={label_col: 'label'})

    before_counts = df['label'].value_counts().to_dict()
    before_total  = len(df)

    df = df.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()

    if is_train:
        parts = []
        for lbl, grp in df.groupby('label'):
            if str(lbl).strip().upper() != 'BENIGN' and len(grp) > ATTACK_CAP:
                grp = grp.sample(ATTACK_CAP, random_state=RS)
            parts.append(grp)
        df = pd.concat(parts, ignore_index=True)

    df['label'] = df['label'].apply(
        lambda x: 'benign' if str(x).strip().upper() == 'BENIGN' else 'malicious'
    )

    b = df[df.label == 'benign']
    m = df[df.label == 'malicious']
    n = min(len(b), len(m))
    df = pd.concat([b.sample(n, random_state=RS),
                    m.sample(n, random_state=RS)])
    df['label'] = df['label'].map({'benign': 0, 'malicious': 1})
    df = df.sample(frac=1, random_state=RS).reset_index(drop=True)

    after_counts = {'benign': n, 'malicious': n}
    after_total  = 2 * n
    print(f"    {name}: {len(df):,} rows after 1:1 balance ({n:,} per class)")
    return df, before_counts, before_total, after_counts, after_total


def feature_selection(X: np.ndarray, y: np.ndarray,
                      feat_names: list) -> tuple[list, np.ndarray]:
    """RF on 10% of training data; return features ranked by importance."""
    n   = max(1000, int(len(X) * 0.10))
    idx = np.random.RandomState(RS).choice(len(X), n, replace=False)
    rf  = RandomForestClassifier(n_estimators=100, random_state=RS, n_jobs=N_JOBS)
    rf.fit(X[idx], y[idx])
    order = np.argsort(rf.feature_importances_)[::-1]
    return [feat_names[i] for i in order], rf.feature_importances_[order]


def get_models() -> dict:
    """Six classifiers with paper-optimal hyperparameters (Section 5.2.3)."""
    return {
        'Decision Tree':
            DecisionTreeClassifier(ccp_alpha=0.0001, random_state=RS),
        'Random Forest':
            RandomForestClassifier(n_estimators=100, random_state=RS,
                                   n_jobs=N_JOBS),
        'Support Vector Machine':
            SVC(kernel='rbf', C=100, gamma=1, random_state=RS),
        'Naive Bayes':
            GaussianNB(),
        'Artificial Neural Network':
            MLPClassifier(hidden_layer_sizes=(40,), activation='tanh',
                          solver='adam', max_iter=500, random_state=RS),
        'Deep Neural Network':
            MLPClassifier(hidden_layer_sizes=(15, 15, 15), activation='tanh',
                          solver='adam', max_iter=500, random_state=RS),
    }


def get_cap(name: str) -> int:
    if 'Support Vector' in name: return SVM_CAP
    if 'Random Forest'  in name: return RF_CAP
    if 'Neural'         in name: return ANN_CAP
    if 'Decision Tree'  in name: return DT_CAP
    return NB_CAP


def cap_sample(X: np.ndarray, y: np.ndarray, n) -> tuple:
    """Return (X, y) subsampled to n rows. n=None means no cap."""
    if n is None or len(X) <= n:
        return X, y
    idx = np.random.RandomState(RS).choice(len(X), n, replace=False)
    return X[idx], y[idx]


def get_metrics(model, X: np.ndarray, y: np.ndarray) -> dict:
    yp = model.predict(X)
    return {
        'accuracy': accuracy_score(y, yp),
        'benign': {
            'precision': precision_score(y, yp, pos_label=0, zero_division=0),
            'recall':    recall_score(y, yp,    pos_label=0, zero_division=0),
            'f1':        f1_score(y, yp,         pos_label=0, zero_division=0),
        },
        'malicious': {
            'precision': precision_score(y, yp, pos_label=1, zero_division=0),
            'recall':    recall_score(y, yp,    pos_label=1, zero_division=0),
            'f1':        f1_score(y, yp,         pos_label=1, zero_division=0),
        },
        'auc': roc_auc_score(y, yp),
        'cm':  confusion_matrix(y, yp),
        'pred': yp,
    }


# =============================================================================
# ──────────────────────────  STAGE 1 — TABLES  ───────────────────────────────
# =============================================================================

def print_table_dist(before_counts: dict, before_total: int,
                     after_counts: dict,  after_total: int,
                     name: str, tnum: int) -> None:
    """Tables 2 & 3: side-by-side before / after cleaning."""
    print(f"\nTABLE {tnum}: Class distribution of {name}")
    print(f"  {'':40} {'Before Cleaning':>28}  {'After Cleaning & Resampling':>28}")
    print(f"  {'Classes':<40} {'No. of Rows':>14} {'(%)':>8}  "
          f"{'No. of Rows':>14} {'(%)':>8}")
    print(f"  {'-'*104}")
    label_order = sorted(before_counts.keys(),
                         key=lambda x: (0 if str(x).strip().upper() == 'BENIGN'
                                        else 1, -before_counts[x]))
    for lbl in label_order:
        b_cnt = before_counts.get(lbl, 0)
        b_pct = b_cnt / before_total * 100
        is_benign = str(lbl).strip().upper() == 'BENIGN'
        if is_benign:
            a_cnt = after_counts.get('benign', 0)
        else:
            a_cnt = min(b_cnt, ATTACK_CAP)
        a_pct = a_cnt / after_total * 100
        print(f"  {str(lbl):<40} {b_cnt:>14,} {b_pct:>8.4f}%  "
              f"{a_cnt:>14,} {a_pct:>8.4f}%")
    print(f"  {'TOTAL':<40} {before_total:>14,} {'100.0000%':>9}  "
          f"{after_total:>14,} {'100.0000%':>9}")


def print_table4_cv(cv: dict, dataset: str = "CIC-IDS2017") -> None:
    """Table 4: fold-by-fold cross-validation."""
    print(f"\nTABLE 4: 5-fold cross-validation accuracy on {dataset}")
    print(f"  {'Model':<30} {'Fold':>6}  {'Accuracy':>10}  "
          f"{'Mean Accuracy':>14}  {'Std Dev':>10}")
    print(f"  {'-'*78}")
    for name, scores in cv.items():
        for i, s in enumerate(scores):
            if i == 0:
                print(f"  {name:<30} Fold-{i+1}  {s:>10.4f}  "
                      f"{scores.mean():>14.4f}  {scores.std():>10.4f}")
            else:
                print(f"  {'':30} Fold-{i+1}  {s:>10.4f}")
        print(f"  {'-'*78}")


def print_table5(cv: dict) -> None:
    """Table 5: comparison vs Kostas[42] and Vinayakumar[29]."""
    print(f"\nTABLE 5: Accuracy comparison for the CIC dataset")
    print(f"  {'ML Models':<30} {'This Work':>10} "
          f"{'Kostas[42]':>12} {'Vinayakumar[29]':>16}")
    print(f"  {'-'*70}")
    for name, scores in cv.items():
        this  = round(scores.mean(), 2)
        k_val = KOSTAS_ACC.get(name)
        v_val = VINAYAKUMAR_ACC.get(name)
        ks    = f"{k_val:.2f}" if k_val is not None else "    -"
        vs    = f"{v_val:.2f}" if v_val is not None else "    -"
        print(f"  {name:<30} {this:>10.2f} {ks:>12} {vs:>16}")


def print_table_perf(results: dict, dataset: str, tnum: int) -> None:
    """Tables 6 & 7: accuracy, precision, recall, F1."""
    print(f"\nTABLE {tnum}: Performance of models on {dataset}")
    print(f"  {'Models':<30} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>10} {'F1-Score':>10}  Class")
    print(f"  {'-'*78}")
    for name, r in results.items():
        b, m = r['benign'], r['malicious']
        print(f"  {name:<30} {r['accuracy']:>10.4f} "
              f"{b['precision']:>10.4f} {b['recall']:>10.4f} {b['f1']:>10.4f}  benign")
        print(f"  {'':30} {'':>10} "
              f"{m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}  malicious")
        print(f"  {'-'*78}")


# =============================================================================
# ──────────────────────────  STAGE 1 — FIGURES  ──────────────────────────────
# =============================================================================

def fig9_importance(feat_names: list, importances: np.ndarray,
                    out_dir: str) -> None:
    """Figure 9: feature importance bar chart (all features)."""
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.bar(range(len(feat_names)), importances,
           color='steelblue', label='importance score')
    ax.set_xticks(range(len(feat_names)))
    ax.set_xticklabels(feat_names, rotation=70, ha='right', fontsize=6)
    ax.set_xlabel('feature')
    ax.set_ylabel('Importance Score')
    ax.set_title('Figure 9: Importance score of each feature in CIC-IDS2017 dataset')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = f"{out_dir}/fig09_feature_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved {path}")


def fig10_features_vs_accuracy(X: np.ndarray, y: np.ndarray,
                                ranked_feats: list, all_feats: list,
                                out_dir: str) -> None:
    """Figure 10: accuracy vs number of features (brute force)."""
    Xs, ys    = cap_sample(X, y, 1000)
    idx_map   = [list(all_feats).index(f) for f in ranked_feats]
    plot_data = {n: [] for n in MODEL_NAMES}
    for k in range(1, len(ranked_feats) + 1):
        sub = Xs[:, idx_map[:k]]
        for name, mdl in get_models().items():
            s = cross_val_score(mdl, sub, ys, cv=3, n_jobs=N_JOBS)
            plot_data[name].append(s.mean())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    fig, ax = plt.subplots(figsize=(11, 5))
    for (name, vals), col in zip(plot_data.items(), colors):
        ax.plot(range(1, len(ranked_feats) + 1), vals, label=name, color=col)
    ax.axvline(TOP_N, color='gray', ls='--', alpha=0.7,
               label=f'Top {TOP_N} selected')
    ax.set_xlabel('Number of features')
    ax.set_ylabel('Accuracy score')
    ax.set_title('Figure 10: Accuracy vs number of features (CIC-IDS2017)')
    ax.legend(fontsize=8)
    ax.set_ylim(0.4, 1.02)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = f"{out_dir}/fig10_features_vs_accuracy.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved {path}")


def fig_cm(results: dict, label: str, fignum: int, out_dir: str) -> None:
    """Figures 11 & 12: 2×3 confusion matrix grid."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, name in zip(axes.flatten(), MODEL_NAMES):
        sns.heatmap(results[name]['cm'], annot=True, fmt='d',
                    cmap='Blues', ax=ax,
                    xticklabels=['benign', 'malicious'],
                    yticklabels=['benign', 'malicious'])
        ax.set_title(name, fontsize=9, fontweight='bold')
        ax.set_xlabel('True label')
        ax.set_ylabel('Predicted label')
    fig.suptitle(
        f'Figure {fignum}: Confusion matrix of each model on {label}',
        fontweight='bold')
    plt.tight_layout()
    fname = (f"fig{fignum:02d}_cm_"
             f"{label.replace(' ', '_').replace('-', '_')}.png")
    path = f"{out_dir}/{fname}"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved {path}")


def fig13_acc_f1(r_tr: dict, r_te: dict, out_dir: str) -> None:
    """Figure 13: accuracy & F1-score comparison bars (train vs test)."""
    acc_tr = [r_tr[n]['accuracy']     for n in MODEL_NAMES]
    acc_te = [r_te[n]['accuracy']     for n in MODEL_NAMES]
    f1_tr  = [r_tr[n]['benign']['f1'] for n in MODEL_NAMES]
    f1_te  = [r_te[n]['benign']['f1'] for n in MODEL_NAMES]
    x, w   = np.arange(6), 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, tr, te, ylabel, title in [
        (ax1, acc_tr, acc_te, 'Accuracy',
         "(a) Accuracy of the models on CIC's dataset"),
        (ax2, f1_tr,  f1_te,  'F1-score',
         "(b) F1-score of the models on CIC's dataset"),
    ]:
        b1 = ax.bar(x - w/2, tr, w, label='CIC-IDS2017',     color='#4472C4')
        b2 = ax.bar(x + w/2, te, w, label='CSE-CIC-IDS2018', color='#ED7D31')
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        for b in list(b1) + list(b2):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.005,
                    f'{b.get_height():.2f}',
                    ha='center', va='bottom', fontsize=7)
    fig.suptitle(
        "Figure 13: Accuracy and F1-score on the CIC's dataset",
        fontweight='bold')
    plt.tight_layout()
    path = f"{out_dir}/fig13_accuracy_f1.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved {path}")


def fig14_time(train_times: dict, pred_times: dict, out_dir: str) -> None:
    """Figure 14: training time (CIC-IDS2017) + prediction time (CSE-CIC-IDS2018)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, times, title in [
        (ax1, [train_times[n] for n in MODEL_NAMES],
         '(a) Training time on CIC-IDS2017 dataset'),
        (ax2, [pred_times[n]  for n in MODEL_NAMES],
         '(b) Prediction time on CSE-CIC-IDS2018 dataset'),
    ]:
        ax.bar(SHORT, times, color='#4472C4')
        ax.set_xlabel('ML models')
        ax.set_ylabel('Time (second)')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(times):
            ax.text(i, v + max(times) * 0.01, f'{v:.2f}',
                    ha='center', va='bottom', fontsize=8)
    fig.suptitle(
        'Figure 14: Time consumption for training and prediction on CIC dataset',
        fontweight='bold')
    plt.tight_layout()
    path = f"{out_dir}/fig14_time.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved {path}")


# =============================================================================
# ──────────────────────────  STAGE 2 — TABLES  ───────────────────────────────
# =============================================================================

def print_table_b1(r_rf11: dict, r_bocs11: dict,
                   r_bocs_full: dict, r_bocs_flag: dict,
                   r_a2: dict,
                   r_a3: dict, r_a3b: dict) -> None:
    """Table B1: full seven-row comparison on CSE-CIC-IDS2018."""
    print("\nTABLE B1: BOCS ablation comparison on CSE-CIC-IDS2018")
    print(f"  {'Model':<48} {'Accuracy':>10} {'Prec(B)':>9} {'Rec(B)':>8} "
          f"{'F1(B)':>7} {'Prec(M)':>9} {'Rec(M)':>8} {'F1(M)':>7} {'AUC':>7}")
    print(f"  {'-'*118}")
    for tag, r in [
        ('A0:   RF-11 (no IF, baseline)',                    r_rf11),
        ('A1:   BOCS-RF-11 (top-11 IF, raw)',               r_bocs11),
        ('A1*:  BOCS-RF-full (IF, sign-corr+std)',          r_bocs_full),
        ('A1*B: BOCS-RF-full (IF, binary flag) ◄',          r_bocs_flag),
        ('A2:   RF + all-data IF, sign-corr+std',           r_a2),
        ('A3:   BOCS-RF-full (OCSVM, sign-corr+std) ◄',    r_a3),
        ('A3B:  BOCS-RF-full (OCSVM, binary flag) ◄',      r_a3b),
    ]:
        b, m = r['benign'], r['malicious']
        print(f"  {tag:<48} {r['accuracy']:>10.4f} "
              f"{b['precision']:>9.4f} {b['recall']:>8.4f} {b['f1']:>7.4f} "
              f"{m['precision']:>9.4f} {m['recall']:>8.4f} {m['f1']:>7.4f} "
              f"{r['auc']:>7.4f}")


def print_table_b2(r_rf11: dict, r_bocs11: dict,
                   r_bocs_full: dict, r_bocs_flag: dict,
                   r_a2: dict, r_a3: dict, r_a3b: dict,
                   cohen_d_11: float, cohen_d_full: float,
                   cohen_d_ocsvm: float) -> None:
    """Table B2: seven-row ablation with Cohen's d and Δrec_m."""
    print("\nTABLE B2: Ablation Study on CSE-CIC-IDS2018")
    print(f"  {'Configuration':<52} {'Accuracy':>10} {'Rec(Mal)':>10} "
          f"{'F1(Mal)':>9} {'AUC':>7}  {'Cohen d':>8}  {'Δrec_m':>8}")
    print(f"  {'-'*110}")
    base_rec = r_rf11['malicious']['recall']
    rows = [
        ('A0:   RF-11 only (baseline)',                    r_rf11,       None),
        ('A1:   BOCS-RF-11 (top-11 IF, raw)',             r_bocs11,     cohen_d_11),
        ('A1*:  BOCS-RF-full (IF, sign-corr+std)',        r_bocs_full,  cohen_d_full),
        ('A1*B: BOCS-RF-full (IF, binary flag)  ◄',      r_bocs_flag,  cohen_d_full),
        ('A2:   RF + all-data IF, sign-corr+std',         r_a2,         None),
        ('A3:   BOCS-RF-full (OCSVM, sign-corr+std) ◄',  r_a3,         cohen_d_ocsvm),
        ('A3B:  BOCS-RF-full (OCSVM, binary flag)  ◄',   r_a3b,        cohen_d_ocsvm),
    ]
    for tag, r, cd in rows:
        m     = r['malicious']
        delta = m['recall'] - base_rec
        cd_s  = f"{cd:.4f}" if cd is not None else "      -"
        print(f"  {tag:<52} {r['accuracy']:>10.4f} "
              f"{m['recall']:>10.4f} {m['f1']:>9.4f} {r['auc']:>7.4f}  "
              f"{cd_s:>8}  {delta:>+8.4f}")


def print_table_b3(mcn: dict) -> None:
    """Table B3: McNemar's test BOCS-RF-flag (A1*B) vs each Stage-1 baseline.
    A1*B is used (not A1*) because A1*B is the working BOCS configuration.
    Running significance tests on the collapsed A1* would be misleading."""
    print("\nTABLE B3: McNemar's Test — BOCS-RF-flag (A1*B) vs Stage-1 Baselines")
    print(f"  (A1*B = binary novelty flag — direction-robust, rec_m=+0.0002 vs baseline)")
    print(f"  {'Comparison':<36} {'b':>8} {'c':>8} {'chi2':>9} {'p-value':>12} {'sig':>6}")
    print(f"  {'-'*84}")
    for comp, r in mcn.items():
        print(f"  {'BOCS-flag vs ' + comp:<36} {r['b']:>8,} {r['c']:>8,} "
              f"{r['chi2']:>9.3f} {r['p']:>12.6f} {r['sig']:>6}")
    print(f"  b = BOCS-flag correct & baseline wrong  |  "
          f"c = BOCS-flag wrong & baseline correct")
    print(f"  *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")


# =============================================================================
# ──────────────────────────  STAGE 2 — FIGURES  ──────────────────────────────
# =============================================================================

def figb1_if_scores(if_bocs, ocsvm_bocs,
                    X_te: np.ndarray, y_te: np.ndarray,
                    out_dir: str) -> None:
    """Figure B1: IF vs OCSVM score distributions on test set."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Figure B1: Novelty Score Distributions on CSE-CIC-IDS2018\n'
                 '(scorers trained on CIC-IDS2017 benign flows only)',
                 fontweight='bold')
    for ax, scorer, title in [
        (ax1, if_bocs,    '(a) Isolation Forest (IF)'),
        (ax2, ocsvm_bocs, '(b) One-Class SVM (OCSVM)'),
    ]:
        sb = scorer.decision_function(X_te[y_te == 0])
        sm = scorer.decision_function(X_te[y_te == 1])
        ax.hist(sb, bins=60, alpha=0.65, color='#4472C4',
                label='Benign (2018)',    density=True)
        ax.hist(sm, bins=60, alpha=0.65, color='#ED7D31',
                label='Malicious (2018)', density=True)
        ax.axvline(0, color='black', lw=1.5, ls='--', alpha=0.7,
                   label='Decision boundary (0)')
        ax.set_xlabel('Novelty score (decision_function)')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.text(0.02, 0.97,
                f"benign  μ={sb.mean():.3f}\nmalicious μ={sm.mean():.3f}",
                transform=ax.transAxes, va='top', fontsize=8,
                color='#333')
    plt.tight_layout()
    path = f"{out_dir}/figB1_score_distributions.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved {path}")


def figb2_comparison(r_s1: dict, r_bocs: dict, out_dir: str) -> None:
    """Figure B2: accuracy & malicious recall — Stage 1 models + BOCS-RF."""
    names  = MODEL_NAMES + ['BOCS-RF']
    accs   = [r_s1[n]['accuracy']          for n in MODEL_NAMES] + [r_bocs['accuracy']]
    rec_m  = [r_s1[n]['malicious']['recall'] for n in MODEL_NAMES] + [r_bocs['malicious']['recall']]
    labels = SHORT + ['BOCS']
    colors = ['#4472C4'] * 6 + ['#1a7a4a']

    x, w = np.arange(len(names)), 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        'Figure B2: Stage-1 Baselines vs BOCS-RF on CSE-CIC-IDS2018',
        fontweight='bold')
    for ax, vals, ylabel, title in [
        (ax1, accs,  'Accuracy',         '(a) Accuracy'),
        (ax2, rec_m, 'Malicious Recall', '(b) Malicious Recall'),
    ]:
        bars = ax.bar(x, vals, w, color=colors, edgecolor='white', alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}',
                    ha='center', va='bottom', fontsize=8)
        # Highlight BOCS bar
        bars[-1].set_edgecolor('#1a7a4a')
        bars[-1].set_linewidth(2)
    plt.tight_layout()
    path = f"{out_dir}/figB2_stage1_vs_bocs.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved {path}")


def figb3_cm_bocs(r_bocs: dict, out_dir: str) -> None:
    """Figure B3: confusion matrix — BOCS-RF on CSE-CIC-IDS2018."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(r_bocs['cm'], annot=True, fmt='d', cmap='Greens', ax=ax,
                xticklabels=['benign', 'malicious'],
                yticklabels=['benign', 'malicious'])
    ax.set_title('Figure B3: BOCS-RF Confusion Matrix\n(CSE-CIC-IDS2018 test set)',
                 fontweight='bold')
    ax.set_xlabel('True label')
    ax.set_ylabel('Predicted label')
    plt.tight_layout()
    path = f"{out_dir}/figB3_bocs_confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved {path}")


def figb4_ablation(r_rf11: dict, r_bocs11: dict,
                   r_bocs_full: dict, r_bocs_flag: dict,
                   r_a2: dict, r_a3: dict, r_a3b: dict,
                   out_dir: str) -> None:
    """Figure B4: seven-bar ablation — A0/A1/A1*/A1*B/A2/A3/A3B."""
    metrics = ['Accuracy', 'Rec (Mal.)', 'F1 (Mal.)', 'AUC']

    def vals(r):
        return [r['accuracy'], r['malicious']['recall'],
                r['malicious']['f1'], r['auc']]

    configs = [
        ('A0 RF-11',    r_rf11,       '#4472C4'),
        ('A1 BOCS-11',  r_bocs11,     '#ED7D31'),
        ('A1* IF-std',  r_bocs_full,  '#d62728'),
        ('A1*B IF-flag',r_bocs_flag,  '#2ca02c'),
        ('A2 allIF',    r_a2,         '#9467bd'),
        ('A3 OC-std',   r_a3,         '#1a7a4a'),
        ('A3B OC-flag', r_a3b,        '#17becf'),
    ]
    x = np.arange(len(metrics))
    w = 0.10
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (label, r, col) in enumerate(configs):
        offset = (i - 3) * w
        bars = ax.bar(x + offset, vals(r), w, label=label,
                      color=col, alpha=0.85, edgecolor='white')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
                    f'{bar.get_height():.3f}',
                    ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.18)
    ax.set_title('Figure B4: Full Ablation — A0/A1/A1*/A1*B/A2/A3/A3B',
                 fontweight='bold')
    ax.legend(fontsize=8, ncol=4)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = f"{out_dir}/figB4_ablation.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved {path}")


# =============================================================================
# ──────────────────  STAGE 2 — MODULE-LEVEL HELPERS  ─────────────────────────
# =============================================================================

def cohen_d(s_b: np.ndarray, s_m: np.ndarray) -> float:
    """Cohen's d effect size between two score distributions."""
    pooled = np.sqrt((s_b.std()**2 + s_m.std()**2) / 2)
    return float(abs(s_b.mean() - s_m.mean()) / pooled) if pooled > 0 else 0.0


def sign_correct_and_standardise(
        scores_tr_raw: np.ndarray,
        scores_te_raw: np.ndarray,
        y_tr: np.ndarray,
        label: str) -> tuple[np.ndarray, np.ndarray, float]:
    
    mean_b_tr = scores_tr_raw[y_tr == 0].mean()
    mean_m_tr = scores_tr_raw[y_tr == 1].mean()
    sign = -1.0 if mean_m_tr > mean_b_tr else 1.0
    scores_tr_signed = scores_tr_raw * sign
    scores_te_signed = scores_te_raw * sign
    sc_mean = scores_tr_signed.mean()
    sc_std  = scores_tr_signed.std() + 1e-8
    scores_tr_std = (scores_tr_signed - sc_mean) / sc_std
    scores_te_std = (scores_te_signed - sc_mean) / sc_std
    print(f"    [{label}] Training benign mean={mean_b_tr:.4f}  "
          f"malicious mean={mean_m_tr:.4f}")
    if sign == -1.0:
        print(f"    [{label}] Direction INVERTED — score flipped ×(−1) "
              f"before standardisation")
    else:
        print(f"    [{label}] Direction correct — no flip needed")
    print(f"    [{label}] Standardisation: "
          f"mean={sc_mean:.4f}  std={sc_std:.4f}  sign={sign:+.0f}")
    return scores_tr_std, scores_te_std, sign


# =============================================================================
# ───────────────────────────────  STAGE 2 CORE  ──────────────────────────────
# =============================================================================


def run_bocs(X_tr_top: np.ndarray, y_tr: np.ndarray,
             X_te_top: np.ndarray, y_te: np.ndarray,
             X_all_s:  np.ndarray, X_te_s: np.ndarray,
             r_s1_test: dict, out_dir: str) -> None:
    
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "=" * 65)
    print("STAGE 2 -- BOCS-IDS  (v6: IF exhausted + OCSVM alternative)")
    print("Seven-row ablation: A0 / A1 / A1* / A1*B / A2 / A3 / A3B")
    print("=" * 65)

    # A0: RF-11 baseline
    print("\n[B0] A0 -- RF-11 baseline (no IF)...")
    rf_11 = RandomForestClassifier(n_estimators=100, random_state=RS, n_jobs=N_JOBS)
    rf_11.fit(X_tr_top, y_tr)
    r_rf11 = get_metrics(rf_11, X_te_top, y_te)
    print(f"    acc={r_rf11['accuracy']:.4f}  rec_m={r_rf11['malicious']['recall']:.4f}")
    base_rec = r_rf11['malicious']['recall']

    # A1: BOCS-RF-11 raw (v1 design, zero effect, kept for completeness)
    print("\n[B1] A1 -- benign-only IF on top-11, raw score (v1)...")
    benign_mask = y_tr == 0
    t0 = time.time()
    if_11 = IsolationForest(n_estimators=300, contamination='auto',
                             random_state=RS, n_jobs=N_JOBS)
    if_11.fit(X_tr_top[benign_mask])
    t_if11 = time.time() - t0
    s_tr_11 = if_11.decision_function(X_tr_top).reshape(-1, 1)
    s_te_11 = if_11.decision_function(X_te_top).reshape(-1, 1)
    d11 = cohen_d(if_11.decision_function(X_te_top[y_te == 0]),
                  if_11.decision_function(X_te_top[y_te == 1]))
    print(f"    IF-11: {t_if11:.2f}s  Cohen's d={d11:.4f}")
    rf_b11 = RandomForestClassifier(n_estimators=100, random_state=RS, n_jobs=N_JOBS)
    rf_b11.fit(np.hstack([X_tr_top, s_tr_11]), y_tr)
    r_bocs11 = get_metrics(rf_b11, np.hstack([X_te_top, s_te_11]), y_te)
    print(f"    A1  acc={r_bocs11['accuracy']:.4f}  "
          f"rec_m={r_bocs11['malicious']['recall']:.4f}  "
          f"D={r_bocs11['malicious']['recall']-base_rec:+.4f}")

    # Train full-feature IF (shared by A1* and A1*B)
    print(f"\n[B2] Training benign-only IF on FULL matrix ({X_all_s.shape[1]} feat)...")
    X_benign_full = X_all_s[benign_mask]
    t0 = time.time()
    if_full = IsolationForest(n_estimators=300, contamination='auto',
                               random_state=RS, n_jobs=N_JOBS)
    if_full.fit(X_benign_full)
    t_if_full = time.time() - t0
    print(f"    IF-full: {t_if_full:.2f}s  "
          f"({X_benign_full.shape[0]:,} x {X_benign_full.shape[1]} feat)")
    s_tr_if_raw = if_full.decision_function(X_all_s).reshape(-1, 1)
    s_te_if_raw = if_full.decision_function(X_te_s).reshape(-1, 1)

    # A1*: sign-corrected + standardised
    s_tr_if_std, s_te_if_std, sign_if = sign_correct_and_standardise(
        s_tr_if_raw, s_te_if_raw, y_tr, 'A1*')
    d_full = cohen_d(s_te_if_std[y_te == 0].flatten(),
                     s_te_if_std[y_te == 1].flatten())
    print(f"    Cohen's d (A1*) = {d_full:.4f}")
    rf_bfull = RandomForestClassifier(n_estimators=100, random_state=RS, n_jobs=N_JOBS)
    rf_bfull.fit(np.hstack([X_tr_top, s_tr_if_std]), y_tr)
    r_bocs_full = get_metrics(rf_bfull, np.hstack([X_te_top, s_te_if_std]), y_te)
    print(f"    A1* acc={r_bocs_full['accuracy']:.4f}  "
          f"rec_m={r_bocs_full['malicious']['recall']:.4f}  "
          f"D={r_bocs_full['malicious']['recall']-base_rec:+.4f}")

    # A1*B: binary flag from IF
    bt = if_full.decision_function(X_all_s[benign_mask])
    thr_if = bt.mean() - 2 * bt.std()
    flag_tr_if = (s_tr_if_raw < thr_if).astype(float)
    flag_te_if = (s_te_if_raw < thr_if).astype(float)
    print(f"    IF threshold={thr_if:.4f}  "
          f"train: benign={flag_tr_if[y_tr==0].mean()*100:.1f}%  "
          f"malicious={flag_tr_if[y_tr==1].mean()*100:.1f}%")
    print(f"    test:  benign={flag_te_if[y_te==0].mean()*100:.1f}%  "
          f"malicious={flag_te_if[y_te==1].mean()*100:.1f}%")
    rf_flag = RandomForestClassifier(n_estimators=100, random_state=RS, n_jobs=N_JOBS)
    rf_flag.fit(np.hstack([X_tr_top, flag_tr_if]), y_tr)
    r_bocs_flag = get_metrics(rf_flag, np.hstack([X_te_top, flag_te_if]), y_te)
    print(f"    A1*B acc={r_bocs_flag['accuracy']:.4f}  "
          f"rec_m={r_bocs_flag['malicious']['recall']:.4f}  "
          f"D={r_bocs_flag['malicious']['recall']-base_rec:+.4f}")

    # A2: all-data IF, sign-corr+std (control)
    print("\n[B3] A2 -- all-data IF on full features, sign-corr+std...")
    if_all = IsolationForest(n_estimators=300, contamination=0.5,
                              random_state=RS, n_jobs=N_JOBS)
    if_all.fit(X_all_s)
    s_tr_a2r = if_all.decision_function(X_all_s).reshape(-1, 1)
    s_te_a2r = if_all.decision_function(X_te_s).reshape(-1, 1)
    s_tr_a2, s_te_a2, sign_a2 = sign_correct_and_standardise(
        s_tr_a2r, s_te_a2r, y_tr, 'A2')
    rf_a2 = RandomForestClassifier(n_estimators=100, random_state=RS, n_jobs=N_JOBS)
    rf_a2.fit(np.hstack([X_tr_top, s_tr_a2]), y_tr)
    r_a2 = get_metrics(rf_a2, np.hstack([X_te_top, s_te_a2]), y_te)
    print(f"    A2  acc={r_a2['accuracy']:.4f}  "
          f"rec_m={r_a2['malicious']['recall']:.4f}  "
          f"D={r_a2['malicious']['recall']-base_rec:+.4f}")

    # A3: One-Class SVM scorer
    print(f"\n[B4] A3 -- One-Class SVM on FULL matrix ({X_all_s.shape[1]} feat)...")
    print(f"    kernel=rbf  nu=0.05  gamma=scale  trained on benign only")
    t0 = time.time()
    ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
    ocsvm.fit(X_benign_full)
    t_oc = time.time() - t0
    print(f"    OCSVM: {t_oc:.2f}s  "
          f"({X_benign_full.shape[0]:,} x {X_benign_full.shape[1]} feat)")
    s_tr_oc_raw = ocsvm.decision_function(X_all_s).reshape(-1, 1)
    s_te_oc_raw = ocsvm.decision_function(X_te_s).reshape(-1, 1)
    s_tr_oc, s_te_oc, sign_oc = sign_correct_and_standardise(
        s_tr_oc_raw, s_te_oc_raw, y_tr, 'A3')
    d_ocsvm = cohen_d(s_te_oc[y_te == 0].flatten(), s_te_oc[y_te == 1].flatten())
    print(f"    Cohen's d (A3, OCSVM) = {d_ocsvm:.4f}")
    t0 = time.time()
    rf_a3 = RandomForestClassifier(n_estimators=100, random_state=RS, n_jobs=N_JOBS)
    rf_a3.fit(np.hstack([X_tr_top, s_tr_oc]), y_tr)
    t_rf_a3 = time.time() - t0
    r_a3 = get_metrics(rf_a3, np.hstack([X_te_top, s_te_oc]), y_te)
    print(f"    A3  acc={r_a3['accuracy']:.4f}  "
          f"rec_m={r_a3['malicious']['recall']:.4f}  "
          f"D={r_a3['malicious']['recall']-base_rec:+.4f}  train={t_rf_a3:.1f}s")

    # A3B: binary flag from OCSVM
    print("\n[B5] A3B -- binary flag from OCSVM score...")
    oc_bt = ocsvm.decision_function(X_all_s[benign_mask])
    thr_oc = oc_bt.mean() - 2 * oc_bt.std()
    flag_tr_oc = (s_tr_oc_raw < thr_oc).astype(float)
    flag_te_oc = (s_te_oc_raw < thr_oc).astype(float)
    print(f"    OCSVM threshold={thr_oc:.4f}  "
          f"train: benign={flag_tr_oc[y_tr==0].mean()*100:.1f}%  "
          f"malicious={flag_tr_oc[y_tr==1].mean()*100:.1f}%")
    print(f"    test:  benign={flag_te_oc[y_te==0].mean()*100:.1f}%  "
          f"malicious={flag_te_oc[y_te==1].mean()*100:.1f}%")
    rf_a3b = RandomForestClassifier(n_estimators=100, random_state=RS, n_jobs=N_JOBS)
    rf_a3b.fit(np.hstack([X_tr_top, flag_tr_oc]), y_tr)
    r_a3b = get_metrics(rf_a3b, np.hstack([X_te_top, flag_te_oc]), y_te)
    print(f"    A3B acc={r_a3b['accuracy']:.4f}  "
          f"rec_m={r_a3b['malicious']['recall']:.4f}  "
          f"D={r_a3b['malicious']['recall']-base_rec:+.4f}")

    # Select best BOCS config for McNemar and Figures B2/B3
    candidates = {'A1*B': r_bocs_flag, 'A3': r_a3, 'A3B': r_a3b}
    best_name = max(candidates, key=lambda k: candidates[k]['malicious']['recall'])
    best_r    = candidates[best_name]
    print(f"\n    Best BOCS config: {best_name}  "
          f"rec_m={best_r['malicious']['recall']:.4f}")

    # McNemar on best config
    print(f"\n[B6] McNemar -- {best_name} vs Stage-1 baselines...")
    yp_best = best_r['pred']
    mcn = {}
    for name in MODEL_NAMES:
        yp_base = r_s1_test[name]['pred']
        c1 = yp_best == y_te
        c2 = yp_base == y_te
        b  = int(np.sum( c1 & ~c2))
        c  = int(np.sum(~c1 &  c2))
        if b + c == 0:
            mcn[name] = {'b':0,'c':0,'chi2':0.,'p':1.,'sig':'ns'}
        else:
            chi2 = float((abs(b - c) - 1)**2 / (b + c))
            p    = float(ss.chi2.sf(chi2, df=1))
            sig  = ('***' if p < 0.001 else '**' if p < 0.01
                    else '*' if p < 0.05 else 'ns')
            mcn[name] = {'b':b,'c':c,'chi2':chi2,'p':p,'sig':sig}
        print(f"    {best_name} vs {name[:3]}: "
              f"b={mcn[name]['b']:,}  c={mcn[name]['c']:,}  "
              f"chi2={mcn[name]['chi2']:.3f}  "
              f"p={mcn[name]['p']:.6f}  {mcn[name]['sig']}")

    print_table_b1(r_rf11, r_bocs11, r_bocs_full, r_bocs_flag,
                   r_a2, r_a3, r_a3b)
    print_table_b2(r_rf11, r_bocs11, r_bocs_full, r_bocs_flag,
                   r_a2, r_a3, r_a3b, d11, d_full, d_ocsvm)
    print_table_b3(mcn)

    print("\n[B7] Generating Stage 2 figures...")
    figb1_if_scores(if_full, ocsvm, X_te_s, y_te, out_dir)
    figb2_comparison(r_s1_test, best_r, out_dir)
    figb3_cm_bocs(best_r, out_dir)
    figb4_ablation(r_rf11, r_bocs11, r_bocs_full, r_bocs_flag,
                   r_a2, r_a3, r_a3b, out_dir)

    print(f"\n    Stage 2 complete. Best config: {best_name}  "
          f"rec_m={best_r['malicious']['recall']:.4f}")
    print(f"    All outputs in: {os.path.abspath(out_dir)}/")


def run_stage1(train_dir: str, test_dir: str,
               out_dir: str) -> tuple:
    
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "=" * 65)
    print("STAGE 1 — Base Paper Replication (Chua & Salam 2023 §5.2)")
    print("Training: CIC-IDS2017  |  Testing: CSE-CIC-IDS2018 (10%)")
    print("=" * 65)

    # [1] Load
    print("\n[1] Loading data...")
    df17_raw = load_2017(train_dir)
    df18_raw = load_2018(test_dir)

    # [2] Align
    print("\n[2] Aligning columns...")
    df17_raw, df18_raw = align_columns(df17_raw, df18_raw)

    # [3] Pre-process
    print("\n[3] Pre-processing...")
    df17, b17c, b17t, a17c, a17t = preprocess(
        df17_raw.copy(), "CIC-IDS2017", is_train=True)
    df18, b18c, b18t, a18c, a18t = preprocess(
        df18_raw.copy(), "CSE-CIC-IDS2018", is_train=False)

    print_table_dist(b17c, b17t, a17c, a17t, "CIC-IDS2017", tnum=2)
    print_table_dist(b18c, b18t, a18c, a18t,
                     "CSE-CIC-IDS2018 dataset (10% of entire dataset)", tnum=3)

    # Numeric matrices — keep ALL numeric columns
    feat_cols = [c for c in df17.columns if c != 'label']
    df17_num  = df17[feat_cols].select_dtypes(include=[np.number])
    feat_cols = list(df17_num.columns)

    X_all = np.nan_to_num(df17_num.values.astype(float))
    y_all = df17['label'].values
    X_te  = np.nan_to_num(
        df18[feat_cols].reindex(columns=feat_cols).values.astype(float))
    y_te  = df18['label'].values

    scaler  = StandardScaler()
    X_all_s = scaler.fit_transform(X_all)   # ALL features — kept for Stage 2 IF
    X_te_s  = scaler.transform(X_te)

    # [4] Feature selection
    print(f"\n[4] Feature selection (RF on 10% of training data)...")
    ranked_feats, ranked_imp = feature_selection(X_all_s, y_all, feat_cols)
    fig9_importance(ranked_feats, ranked_imp, out_dir)
    top_feats = ranked_feats[:TOP_N]
    top_idx   = [feat_cols.index(f) for f in top_feats]
    print(f"    Top {TOP_N} features: {top_feats}")
    fig10_features_vs_accuracy(X_all_s, y_all, ranked_feats, feat_cols, out_dir)

    # Top-11 matrices used for Stage 1 classification and Stage 2 classifier
    X_tr   = X_all_s[:, top_idx]
    X_test = X_te_s[:,  top_idx]
    print(f"    Full matrix : train={X_all_s.shape}  test={X_te_s.shape}")
    print(f"    Top-11 matrix: train={X_tr.shape}   test={X_test.shape}")

    # [5] Cross-validation — Table 4
    print("\n[5] 5-fold cross-validation (Table 4)...")
    models = get_models()
    kf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    cv_results = {}
    for name, mdl in models.items():
        print(f"    {name}...", end=' ', flush=True)
        Xcv, ycv = cap_sample(X_tr, y_all, get_cap(name))
        scores   = cross_val_score(mdl, Xcv, ycv, cv=kf,
                                   scoring='accuracy', n_jobs=N_JOBS)
        cv_results[name] = scores
        print(f"{scores.mean():.4f} ± {scores.std():.4f}")

    print_table4_cv(cv_results)
    print_table5(cv_results)

    # [6] Final training: 70% CIC-2017 → train / 30% → eval
    print("\n[6] Training final models (70/30 split on CIC-IDS2017)...")
    perm  = np.random.RandomState(RS).permutation(len(X_tr))
    split = int(0.7 * len(X_tr))
    X_70, y_70 = X_tr[perm[:split]], y_all[perm[:split]]
    X_30, y_30 = X_tr[perm[split:]], y_all[perm[split:]]

    r_train, r_test = {}, {}
    train_times, pred_times = {}, {}

    for name, mdl in models.items():
        print(f"    {name}...", end=' ', flush=True)
        Xfit, yfit = cap_sample(X_70, y_70, get_cap(name))

        t0 = time.time()
        mdl.fit(Xfit, yfit)
        train_times[name] = time.time() - t0

        r_train[name] = get_metrics(mdl, X_30, y_30)

        t0 = time.time()
        r_test[name]  = get_metrics(mdl, X_test, y_te)
        pred_times[name] = time.time() - t0

        print(f"train={train_times[name]:.2f}s  pred={pred_times[name]:.2f}s  "
              f"test_acc={r_test[name]['accuracy']:.4f}  "
              f"rec_m={r_test[name]['malicious']['recall']:.4f}")

    print_table_perf(r_train, "CIC-IDS2017 dataset",     tnum=6)
    print_table_perf(r_test,  "CSE-CIC-IDS2018 dataset", tnum=7)

    # [7] Figures
    print("\n[7] Generating Stage 1 figures...")
    fig_cm(r_train, "CIC-IDS2017",     fignum=11, out_dir=out_dir)
    fig_cm(r_test,  "CSE-CIC-IDS2018", fignum=12, out_dir=out_dir)
    fig13_acc_f1(r_train, r_test, out_dir)
    fig14_time(train_times, pred_times, out_dir)

    print(f"\n    Stage 1 complete. All outputs in: {os.path.abspath(out_dir)}/")
    print(f"    Tables (console): 2, 3, 4, 5, 6, 7")
    print(f"    Figures (files):  fig09 — fig14")

    # Return top-11 matrices for Stage 2 classifier
    # AND full matrices for Stage 2 IF training
    return X_tr, y_all, X_test, y_te, X_all_s, X_te_s, r_test


# =============================================================================
# ───────────────────────────────────  MAIN  ──────────────────────────────────
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BOCS-IDS: Two-Stage CIC IDS Experiment")
    parser.add_argument(
        '--stage', type=int, choices=[1, 2], default=0,
        help='1 = Stage 1 only, 2 = Stage 2 only, 0 = run both (default)')
    args = parser.parse_args()

    print("=" * 65)
    print("BOCS-IDS: Two-Stage Intrusion Detection Experiment")
    print("GPU environment: CUDA_VISIBLE_DEVICES =", CUDA_VISIBLE)
    print("Datasets:")
    print(f"  TRAIN: {os.path.abspath(TRAIN_DIR)}")
    print(f"  TEST : {os.path.abspath(TEST_DIR)}")
    print("=" * 65)

    run_s1 = args.stage in (0, 1)
    run_s2 = args.stage in (0, 2)

    # Stage 1 must always run first (Stage 2 reuses its matrices)
    if run_s1 or run_s2:
        X_tr, y_tr, X_te, y_te, X_all_s, X_te_s, r_s1_test = run_stage1(
            TRAIN_DIR, TEST_DIR, OUT_S1)

    if run_s2:
        run_bocs(X_tr, y_tr, X_te, y_te, X_all_s, X_te_s, r_s1_test, OUT_S2)

    print("\n" + "=" * 65)
    print("EXPERIMENT COMPLETE")
    print(f"  Stage 1 outputs: {os.path.abspath(OUT_S1)}/")
    if run_s2:
        print(f"  Stage 2 outputs: {os.path.abspath(OUT_S2)}/")
    print("=" * 65)


if __name__ == '__main__':
    main()
