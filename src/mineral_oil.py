"""Screen bio-sourced replacements for mineral oil.

Each candidate gets four checks:
  1. Biodegradable   - ML model from train.py (P(readily biodegradable))
  2. Oxidation-proof - rule from structure: C=C double bonds go rancid; a CH2 between two C=C
                       (bis-allylic, as in linoleic acid) oxidises fastest
  3. pH 2-12 proof   - rule from structure: ester/carbonate bonds hydrolyse in strong acid/alkali
                       (hair colour and treatments span pH 2-12); pure hydrocarbons have nothing to hydrolyse
  4. Texture match   - proxy: closeness in molecular weight to the mineral-oil model molecule
                       (viscosity and spreading rise with chain length)
Real oils are mixtures; each is modelled by one representative molecule (see MODELLED_AS).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

sys.path.insert(0, str(Path(__file__).parent))
from features import featurize_mol, standardize

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
_bitgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

OLEOYL = "CCCCCCCC/C=C\\CCCCCCCC(=O)"
LINOLEOYL = "CCCCC/C=C\\C/C=C\\CCCCCCCC(=O)"
CAPRYLOYL = "CCCCCCCC(=O)"

# (name, type, source, biobased, SMILES, modelled as)
CANDIDATES = [
    ("Mineral oil (today)", "Petroleum hydrocarbon", "Crude oil refining", 0, "C" * 20, "n-eicosane, C20"),
    ("Hemisqualane", "Bio-alkane", "Sugarcane sugar, fermentation", 1, "CC(C)CCCC(C)CCCC(C)CC", "2,6,10-trimethyldodecane, C15"),
    ("C15-19 alkane", "Bio-alkane", "Hydrotreated vegetable oil", 1, "C" * 17, "n-heptadecane, C17"),
    ("Undecane", "Bio-alkane", "Coconut / palm kernel", 1, "C" * 11, "n-undecane, C11"),
    ("Squalane", "Bio-alkane", "Sugarcane / olive", 1, "CC(C)CCCC(C)CCCC(C)CCCCC(C)CCCC(C)CCCC(C)C", "squalane, C30"),
    ("Coco-caprylate", "Plant ester", "Coconut", 1, "CCCCCCCCCCCCOC(=O)CCCCCCC", "lauryl caprylate"),
    ("Isoamyl laurate", "Plant ester", "Plant fatty acid", 1, "CCCCCCCCCCCC(=O)OCCC(C)C", "isoamyl laurate"),
    ("Caprylic/capric triglyceride", "Saturated plant oil", "Coconut / palm kernel", 1,
     f"{CAPRYLOYL}OCC(OC(=O)CCCCCCC)COC(=O)CCCCCCC", "glyceryl tricaprylate"),
    ("Olive oil", "Unsaturated plant oil", "Olive", 1, f"{OLEOYL}OCC(OC(=O)CCCCCCC/C=C\\CCCCCCCC)COC(=O)CCCCCCC/C=C\\CCCCCCCC", "triolein"),
    ("Sunflower oil", "Unsaturated plant oil", "Sunflower", 1, f"{LINOLEOYL}OCC(OC(=O)CCCCCCC/C=C\\C/C=C\\CCCCC)COC(=O)CCCCCCC/C=C\\C/C=C\\CCCCC", "trilinolein"),
]

C_EQ_C = Chem.MolFromSmarts("[C;!a]=[C;!a]")
BIS_ALLYLIC = Chem.MolFromSmarts("[C;!a]=[C;!a][CH2][C;!a]=[C;!a]")
HYDROLYSABLE = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")  # esters and carbonates


def main():
    model = pickle.load(open(OUT / "model.pkl", "rb"))
    train = pd.read_csv(ROOT / "data_raw/allpublic/All-Public_dataset_Mordred.csv", usecols=["SMILES", "Class"])
    train_mols = [standardize(s) for s in train.SMILES]
    known = {Chem.MolToSmiles(m): int(c) for m, c in zip(train_mols, train.Class) if m is not None}
    train_fps = [_bitgen.GetFingerprint(m) for m in train_mols if m is not None]

    ref_mw = None
    rows = []
    for name, kind, source, bio, smi, modelled_as in CANDIDATES:
        mol = standardize(smi)
        assert mol is not None, name
        mw = Descriptors.MolWt(mol)
        ref_mw = ref_mw or mw  # first row is the mineral-oil reference
        p = float(model.predict_proba(featurize_mol(mol).reshape(1, -1))[0, 1])
        n_db = len(mol.GetSubstructMatches(C_EQ_C))
        n_bis = len(mol.GetSubstructMatches(BIS_ALLYLIC))
        n_hyd = len(mol.GetSubstructMatches(HYDROLYSABLE))
        oxid = 1.0 if n_db == 0 else 0.5 if n_bis == 0 else 0.0
        ph = 1.0 if n_hyd == 0 else 0.0
        # 150 g/mol is a rough step at which a cosmetic oil feels noticeably lighter/heavier.
        texture = float(np.exp(-abs(mw - ref_mw) / 150.0))
        sim = max(DataStructs.BulkTanimotoSimilarity(_bitgen.GetFingerprint(mol), train_fps))
        rows.append({
            "candidate": name, "type": kind, "source": source, "biobased": bio, "modelled_as": modelled_as,
            "mol_weight": round(mw, 1),
            "p_readily_biodegradable": round(p, 3),
            "lab_result_in_dataset": {1: "RB", 0: "NRB"}.get(known.get(Chem.MolToSmiles(mol)), "not in data"),
            "model_confidence": "high" if sim >= 0.5 else "medium" if sim >= 0.3 else "low",
            "c_eq_c_bonds": n_db, "bis_allylic_sites": n_bis, "ester_bonds": n_hyd,
            "oxidation_stability": {1.0: "high", 0.5: "medium", 0.0: "low"}[oxid],
            "ph_2_12_stability": "stable" if ph else "hydrolysis risk",
            "texture_match": round(texture, 2),
            # Equal weight on the three "will it survive" checks, less on the texture proxy.
            "overall_score": round(0.3 * p + 0.25 * oxid + 0.25 * ph + 0.2 * texture, 3) if bio else None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mineral_oil_screening.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.drop(columns=["source", "type"]).sort_values("overall_score", ascending=False, na_position="first").to_string(index=False))


if __name__ == "__main__":
    main()
