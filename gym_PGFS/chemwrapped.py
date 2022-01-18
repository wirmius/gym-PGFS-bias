import numpy as np
from functools import reduce

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

# from rdkit.Chem import Mol, MolToSmarts, DetectChemistryProblems, MolFromSmiles, MolToSmiles, SanitizeMol
# from rdkit.Chem.rdChemReactions import SanitizeFlags, SanitizeRxn, ReactionFromSmarts
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from typing import Union, List

# the following lines are imports that cna be redirected to rdchiral, for instance
from rdkit.Chem.rdChemReactions import ChemicalReaction

from rdkit.Chem import Mol

# imports and constants for rendering
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

# for more descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from .constants import DEFAULT_MOL_WIDTH, DEFAULT_MOL_HEIGHT, MOLDSET_SELECT_DESCRIPTORS

# for some guacamole functions
from guacamol.utils.chemistry import initialise_neutralisation_reactions, neutralise_charges, split_charged_mol, canonicalize, remove_duplicates


def SmilesToSVG(smi: str, highlight: List = None, dims = (DEFAULT_MOL_WIDTH, DEFAULT_MOL_HEIGHT)) -> str:
    '''
    No highlights for now (will have to figure those out.
    Parameters
    ----------
    smi
        smiles of the molecule to depict
    highlight
        a list containing the Mol objects of substructures to highlight
    dims
        a tuple (width, height)
    Returns
    -------
        svg as a string
    '''
    mol = ChemMolFromSmilesWrapper(smi)

    # compute the coordinates of atoms
    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DSVG(*dims)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg.replace('svg:','')


def ReactionFromSMARTSAll(sma: str):
    rxn = AllChem.ReactionFromSmarts(sma)
    sanitize_flags = AllChem.SanitizeRxn(rxn)
    if sanitize_flags != AllChem.SanitizeFlags.SANITIZE_NONE:
        print(f"\nReaction had sanitize problems: {sma} ||||  {sanitize_flags}")
        return None
    rxn.Initialize()
    return rxn


def ChemMolToSmilesWrapper(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol)


def ChemMolFromSmilesWrapper(smi: str) -> Chem.Mol:
    try:
        m = Chem.MolFromSmiles(smi, sanitize=False) #AllChem.AddHs(Chem.MolFromSmiles(smi, sanitize=True))
        problems = Chem.DetectChemistryProblems(m)
        if len(problems) != 0:
            m = None
        Chem.SanitizeMol(m)
    except Exception as e:
        m = None
    finally:
        return m


# using the guacamol routines at the beginning to filter out molecule smiles
from guacamol.utils.chemistry import initialise_neutralisation_reactions, neutralise_charges


def ChemGuacamolFilterExtended(smiles: str, include_stereocenters=False) -> str:
    # Drop out if too long
    # if len(smiles) > 200:
    #     return None
    mol = Chem.MolFromSmiles(smiles)
    # Drop out if invalid
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)

    # We only accept molecules consisting of H, B, C, N, O, F, Si, P, S, Cl, aliphatic Se, Br, I.
    metal_smarts = Chem.MolFromSmarts('[!#1!#5!#6!#7!#8!#9!#14!#15!#16!#17!#34!#35!#53]')

    has_metal = mol.HasSubstructMatch(metal_smarts)

    # Exclude molecules containing the forbidden elements.
    if has_metal:
        print(f'metal {smiles}')
        return None

    canon_smi = Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)

    # Drop out if too long canonicalized:
    # if len(canon_smi) > 100:
    #     return None
    # Balance charges if unbalanced
    if canon_smi.count('+') - canon_smi.count('-') != 0:
        new_mol, changed = neutralise_charges(mol, reactions=initialise_neutralisation_reactions())
        # print("neutralized")
        if changed:
            mol = new_mol
            canon_smi = Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)

    return canon_smi


def CheckMol(m: Chem.Mol) -> bool:
    problems = Chem.DetectChemistryProblems(m)
    if len(problems) != 0:
        return False
    sanity = Chem.SanitizeMol(m, catchErrors=False)
    if sanity != AllChem.SanitizeFlags.SANITIZE_NONE:
        return False
    return True

def evaluate_reactivity(mol: Chem.Mol, template_mol: Chem.Mol = None) -> bool:
    if not mol or not template_mol:
        raise Exception("Empty template (or molecule) supplied.")
    else:
        return mol.HasSubstructMatch(template_mol)

def ECFP6_bitvector_numpy(m: Union[str, Chem.Mol], radius: int = 0, size: int = 0) -> np.ndarray:
    if isinstance(m, str):
        m = Chem.MolFromSmiles(m)
    arr = np.zeros((0,), dtype=np.uint8)
    ConvertToNumpyArray(
        AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=size),
        arr
    )
    return arr


# in order to not initialize the calculator all of the time
moldy_calc = MoleculeDescriptors.MolecularDescriptorCalculator(MOLDSET_SELECT_DESCRIPTORS)


def MolDSetDescriptorsNumpy(m: Union[str, Chem.Mol]):
    if isinstance(m, str):
        m = Chem.MolFromSmiles(m)
    d = moldy_calc.CalcDescriptors(m)
    return np.asarray(d)


def ChemMolToSMARTS(mol: Chem.Mol) -> str:
    return Chem.MolToSmarts(mol)


## distance functions
def DiceDistance(fp1, fp2):
    return 2*np.sum(np.logical_xor(fp1, fp2))/(fp1.shape[0]+fp2.shape[0])


def CosDistance(fp1, fp2):
    return 1 - np.dot(fp1, fp2)/(np.linalg.norm(fp1)*np.linalg.norm(fp2))
