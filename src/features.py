"""Turn a SMILES string into the feature vector the model uses."""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

FP_SIZE = 2048
_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_SIZE)
_DESC = Descriptors._descList  # ~210 RDKit 2D descriptors
DESC_NAMES = [name for name, _ in _DESC]
_chooser = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()


def standardize(smiles):
    """Parse SMILES and keep the parent molecule, so 'SLS + Na+' is scored as SLS."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _uncharger.uncharge(_chooser.choose(mol))


def featurize_mol(mol):
    fp = np.array(_fpgen.GetCountFingerprintAsNumPy(mol), dtype=np.float32)
    desc = []
    for _, fn in _DESC:
        try:
            desc.append(float(fn(mol)))
        except Exception:
            desc.append(0.0)
    desc = np.nan_to_num(np.array(desc, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    # Some descriptors (e.g. Ipc) explode for big molecules; clip so trees aren't dominated by them.
    desc = np.clip(desc, -1e6, 1e6)
    return np.concatenate([fp, desc])


def feature_names():
    return [f"morgan_{i}" for i in range(FP_SIZE)] + DESC_NAMES
