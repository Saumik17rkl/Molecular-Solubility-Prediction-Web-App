import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors


######################
# Custom function
######################

# Calculate molecular descriptors
def AromaticProportion(m):
    aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
    aa_count = [1 for i in aromatic_atoms if i]
    AromaticAtom = sum(aa_count)
    HeavyAtom = Descriptors.HeavyAtomCount(m)
    AR = AromaticAtom / HeavyAtom
    return AR


def generate(smiles, verbose=False):
    moldata = [Chem.MolFromSmiles(elem) for elem in smiles]
    baseData = np.arange(1, 1)
    for i, mol in enumerate(moldata):
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_MolWt = Descriptors.MolWt(mol)
        desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
        desc_AromaticProportion = AromaticProportion(mol)

        row = np.array([desc_MolLogP, desc_MolWt, desc_NumRotatableBonds, desc_AromaticProportion])
        baseData = np.vstack([baseData, row]) if i > 0 else row

    columnNames = ["MolLogP", "MolWt", "NumRotatableBonds", "AromaticProportion"]
    descriptors = pd.DataFrame(data=baseData, columns=columnNames)
    return descriptors


def explain_molecule(mol):
    explanation = []
    if mol:
        # Count functional groups
        functional_groups = {"Aromatic": 0, "Aliphatic": 0, "Double bonds": 0, "Single bonds": 0}
        for atom in mol.GetAtoms():
            # Skip atoms that might not have bonds
            if atom.GetDegree() == 0:  # If an atom has no bonds (e.g., isolated hydrogen)
                continue

            if atom.GetIsAromatic():
                functional_groups["Aromatic"] += 1
            elif len(atom.GetBonds()) == 1:  # Single bond atom
                functional_groups["Single bonds"] += 1
            elif len(atom.GetBonds()) == 2:  # Double bond atom
                functional_groups["Double bonds"] += 1
            else:
                functional_groups["Aliphatic"] += 1  # Aliphatic atoms (other than aromatic)

        explanation.append(f"Aromatic atoms count: {functional_groups['Aromatic']}")
        explanation.append(f"Single bonds count: {functional_groups['Single bonds']}")
        explanation.append(f"Double bonds count: {functional_groups['Double bonds']}")
        explanation.append(f"Aliphatic atoms count: {functional_groups['Aliphatic']}")

    return explanation


######################
# Streamlit Web App
######################

# Page Title
image = Image.open('solubility-logo.jpg')
st.image(image, use_container_width=True)

st.write("""
# Molecular Solubility Prediction Web App

This app predicts the **Solubility (LogS)** values of molecules and provides detailed explanations of their structures!

Data obtained from the John S. Delaney. [ESOL:  Estimating Aqueous Solubility Directly from Molecular Structure](https://pubs.acs.org/doi/10.1021/ci034243x). ***J. Chem. Inf. Comput. Sci.*** 2004, 44, 3, 1000-1005.
*** 
""")

######################
# Input molecules (Side Panel)
######################

st.sidebar.header('User Input Features')

# Read SMILES input
SMILES_input = "NCCCC\nCCC\nCN"

SMILES = st.sidebar.text_area("SMILES input", SMILES_input)
SMILES = "C\n" + SMILES  # Adds C as a dummy, first item
SMILES = SMILES.split('\n')

st.header('Input SMILES')
st.write(SMILES[1:])  # Skips the dummy first item

# Calculate molecular descriptors
st.header('Computed molecular descriptors')
X = generate(SMILES)
st.write(X[1:])  # Skips the dummy first item

######################
# Pre-built model
######################

# Reads in saved model
load_model = pickle.load(open('solubility_model.pkl', 'rb'))

# Apply model to make predictions
prediction = load_model.predict(X)

st.header('Predicted LogS values')
st.write(prediction[1:])  # Skips the dummy first item

######################
# Explain the molecular structure
######################

# Generate explanation for the first molecule in the input
mol = Chem.MolFromSmiles(SMILES[1])
explanation = explain_molecule(mol)

st.header('Molecular Structure Explanation')
for item in explanation:
    st.write(item)
