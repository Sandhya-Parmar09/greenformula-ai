"""Score each incumbent ingredient and its bio-based alternatives, then rank the swaps.

Swap score = 0.50 * P(readily biodegradable) + 0.25 * bio-based + 0.25 * functional similarity.
Functional similarity is a heuristic: closeness to the incumbent in logP and molecular weight,
the two properties that most drive how an emollient spreads and feels.
"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator

sys.path.insert(0, str(Path(__file__).parent))
from features import featurize_mol, standardize
from ingredients import CASES

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
_bitgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def canon(mol):
    return Chem.MolToSmiles(mol)


def main():
    model = pickle.load(open(OUT / "model.pkl", "rb"))
    train = pd.read_csv(DATA_CSV := ROOT / "data_raw/allpublic/All-Public_dataset_Mordred.csv", usecols=["SMILES", "Class"])
    train_mols = [standardize(s) for s in train.SMILES]
    known = {canon(m): int(c) for m, c in zip(train_mols, train.Class) if m is not None}
    train_fps = [_bitgen.GetFingerprint(m) for m in train_mols if m is not None]

    rows = []
    for case in CASES:
        inc = standardize(case["incumbent"]["smiles"])
        inc_logp, inc_mw = Crippen.MolLogP(inc), Descriptors.MolWt(inc)
        for role, ing in [("incumbent", case["incumbent"])] + [("alternative", a) for a in case["alternatives"]]:
            mol = standardize(ing["smiles"])
            p = float(model.predict_proba(featurize_mol(mol).reshape(1, -1))[0, 1])
            sim_train = max(DataStructs.BulkTanimotoSimilarity(_bitgen.GetFingerprint(mol), train_fps))
            logp, mw = Crippen.MolLogP(mol), Descriptors.MolWt(mol)
            # Scales (2 logP units, 150 g/mol) are rough "noticeably different" steps for cosmetic oils.
            func_sim = float(np.exp(-np.hypot((logp - inc_logp) / 2.0, (mw - inc_mw) / 150.0)))
            rows.append({
                "function": case["function"], "role": role, "ingredient": ing["name"],
                "source": ing["source"], "biobased": ing["biobased"],
                "p_readily_biodegradable": round(p, 3),
                "lab_result_in_dataset": {1: "RB", 0: "NRB"}.get(known.get(canon(mol)), "not in data"),
                "nearest_training_similarity": round(sim_train, 2),
                "confidence": "high" if sim_train >= 0.5 else "medium" if sim_train >= 0.3 else "low (outside training chemistry)",
                "logP": round(logp, 2), "mol_weight": round(mw, 1),
                "functional_similarity": round(func_sim, 2) if role == "alternative" else None,
                "swap_score": round(0.5 * p + 0.25 * ing["biobased"] + 0.25 * func_sim, 3) if role == "alternative" else None,
                "note": ing.get("note", ""),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "swap_screening.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(df.drop(columns=["note", "source"]).to_string(index=False))

    # Summary chart: per function, the incumbent, the top-ranked swap, and any bio-based "trap"
    # (bio-based but predicted not readily biodegradable) — the trap is the point of the tool.
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labels, vals, colors = [], [], []
    for case in CASES:
        sub = df[df.function == case["function"]]
        alts = sub[sub.role == "alternative"].sort_values("swap_score", ascending=False)
        picks = [sub[sub.role == "incumbent"].iloc[0], alts.iloc[0]]
        picks += [r for _, r in alts.iloc[1:].iterrows() if r.p_readily_biodegradable < 0.5]
        for r in picks:
            labels.append(r.ingredient); vals.append(r.p_readily_biodegradable)
            colors.append("#b5473a" if r.role == "incumbent" else "#1b7a4a" if r.p_readily_biodegradable >= 0.5 else "#d99a2b")
        labels.append(""); vals.append(0); colors.append("none")
    labels, vals, colors = labels[:-1], vals[:-1], colors[:-1]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=colors, height=0.7)
    ax.set_yticks(y, labels, fontsize=8)
    for yi, v, c in zip(y, vals, colors):
        if c != "none":
            ax.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=8)
    ax.axvline(0.5, color="#999", ls="--", lw=1)
    ax.set_xlim(0, 1.08); ax.set_xlabel("Predicted probability of ready biodegradability", fontsize=8)
    ax.set_title("Current (red) · AI-picked swap (green) · bio-based but NOT biodegradable (amber)", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(OUT / "swap_chart.png", dpi=220); plt.close()


if __name__ == "__main__":
    main()
