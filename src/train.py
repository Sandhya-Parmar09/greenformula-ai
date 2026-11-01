"""Train and evaluate the ready-biodegradability classifier.

Data: "All-Public" set (Lunghini et al. 2020, SAR QSAR Environ. Res., doi 10.1080/1062936X.2019.1697360),
redistributed by github.com/mhlee216/Biodegradability_Prediction_QSAR_GCN. Class 1 = readily biodegradable.
"""
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from features import featurize_mol, standardize, feature_names

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_raw/allpublic/All-Public_dataset_Mordred.csv"
OUT = ROOT / "outputs"
SEED = 42


def make_model():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=1, max_features="sqrt",
                                  class_weight="balanced", n_jobs=-1, random_state=SEED)


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, p)}


def scaffold_split(smiles, test_frac=0.2):
    """Whole scaffold groups go to one side, so test molecules are structurally new to the model."""
    groups = {}
    for i, s in enumerate(smiles):
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s))
        except Exception:
            scaf = s
        groups.setdefault(scaf, []).append(i)
    rng = np.random.default_rng(SEED)
    keys = list(groups)
    rng.shuffle(keys)
    test, target = [], test_frac * len(smiles)
    for k in sorted(keys, key=lambda k: len(groups[k])):  # small groups first fills test with diverse scaffolds
        if len(test) >= target:
            break
        test.extend(groups[k])
    test = np.array(sorted(test))
    train = np.setdiff1d(np.arange(len(smiles)), test)
    return train, test


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA, usecols=["SMILES", "Class"])
    mols = [standardize(s) for s in df.SMILES]
    keep = [m is not None for m in mols]
    df = df[keep].reset_index(drop=True)
    mols = [m for m in mols if m is not None]
    X = np.stack([featurize_mol(m) for m in mols])
    y = df.Class.values.astype(int)
    print(f"molecules: {len(y)}  (readily biodegradable: {y.sum()}, not: {len(y) - y.sum()})")

    # 1) 5-fold stratified cross-validation on the whole set
    cv = cross_validate(make_model(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                        scoring=["accuracy", "balanced_accuracy", "f1", "roc_auc"])
    cv_res = {k.replace("test_", ""): (float(v.mean()), float(v.std())) for k, v in cv.items() if k.startswith("test_")}

    # 2) Held-out random 80/20 split
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    m = make_model().fit(X[tr], y[tr])
    p_rand = m.predict_proba(X[te])[:, 1]
    rand_res = metrics(y[te], p_rand)
    cm = confusion_matrix(y[te], (p_rand >= 0.5).astype(int)).tolist()

    # 3) Scaffold split: the harder, more honest test for "new" ingredients
    str_, ste = scaffold_split(df.SMILES.tolist())
    ms = make_model().fit(X[str_], y[str_])
    p_scaf = ms.predict_proba(X[ste])[:, 1]
    scaf_res = metrics(y[ste], p_scaf)

    # Final model on all data, used for ingredient screening
    final = make_model().fit(X, y)
    with open(OUT / "model.pkl", "wb") as f:
        pickle.dump(final, f)

    imp = pd.Series(final.feature_importances_, index=feature_names()).sort_values(ascending=False).head(15)

    report = {"n_molecules": int(len(y)), "n_readily_biodegradable": int(y.sum()),
              "cv_5fold_mean_std": cv_res, "random_holdout_20pct": rand_res,
              "random_holdout_confusion_matrix[[TN,FP],[FN,TP]]": cm,
              "scaffold_holdout_20pct": scaf_res, "scaffold_test_size": int(len(ste)),
              "top_features": imp.round(4).to_dict()}
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ["cv_5fold_mean_std", "random_holdout_20pct", "scaffold_holdout_20pct"]}, indent=2))

    # ROC chart
    plt.figure(figsize=(4.2, 4))
    for name, yt, pp, c in [("Random split", y[te], p_rand, "#1b7a4a"), ("Scaffold split (new chemistry)", y[ste], p_scaf, "#8fbf6a")]:
        fpr, tpr, _ = roc_curve(yt, pp)
        plt.plot(fpr, tpr, color=c, lw=2.2, label=f"{name}  AUC {roc_auc_score(yt, pp):.2f}")
    plt.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("Biodegradability model on unseen molecules", fontsize=10)
    plt.legend(loc="lower right", fontsize=8, frameon=False)
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(OUT / "roc_curve.png", dpi=220); plt.close()


if __name__ == "__main__":
    main()
