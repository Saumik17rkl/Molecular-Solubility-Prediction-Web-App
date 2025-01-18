# Molecular-Solubility-Prediction-Web-App

Molecular Solubility Prediction Web App
This web application predicts the Solubility (LogS) values of molecules and provides detailed explanations of their structural features. It leverages molecular descriptors and a machine learning model to estimate solubility based on the structure of the molecule represented by its SMILES notation.

Data Source
The data used in this project is from the paper ESOL: Estimating Aqueous Solubility Directly from Molecular Structure by John S. Delaney. J. Chem. Inf. Comput. Sci. 2004, 44, 3, 1000-1005.

Features
SMILES Input: Input the SMILES notation of the molecule you want to predict the solubility for.
Molecular Descriptors Calculation: The app calculates molecular descriptors such as LogP, molecular weight, number of rotatable bonds, and aromatic proportion.
Solubility Prediction: Using a pre-trained model, the app predicts the LogS value (aqueous solubility) for the given molecule.
Molecular Structure Explanation: Provides an explanation of the molecule's functional groups such as aromatic atoms, single bonds, double bonds, and aliphatic atoms.
Requirements
To run this project, you need the following libraries installed:

numpy
pandas
streamlit
pickle
Pillow (PIL)
rdkit
Install them using pip:

bash
Copy
Edit
pip install numpy pandas streamlit pillow rdkit
How to Run the Web App
Clone this repository:
bash
Copy
Edit
git clone https://github.com/your-username/molecular-solubility-prediction.git
cd molecular-solubility-prediction
Install the necessary dependencies:
bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app:
bash
Copy
Edit
streamlit run app.py
The app will open in your web browser.

Code Explanation
Custom Functions
AromaticProportion(m): Calculates the proportion of aromatic atoms in a molecule.
generate(smiles, verbose=False): Converts a list of SMILES strings to molecular descriptors (MolLogP, MolWt, NumRotatableBonds, AromaticProportion).
explain_molecule(mol): Provides a detailed explanation of the molecule's functional groups, such as aromatic atoms, aliphatic atoms, single bonds, and double bonds.
Streamlit Web App
SMILES Input: Users can input SMILES notation for one or more molecules.
Molecular Descriptor Calculation: The app computes the molecular descriptors and displays them.
Model Prediction: A pre-trained model (solubility_model.pkl) is used to predict the solubility (LogS) of the molecules.
Molecular Explanation: The app explains the structural features of the molecule.
File Structure
app.py: Main script to run the Streamlit app.
solubility_model.pkl: Pre-trained machine learning model for solubility prediction.
solubility-logo.jpg: Image for the web app header.
requirements.txt: List of Python packages required to run the app.
Example Input
You can input a SMILES notation like:

Copy
Edit
NCCCC
CCC
CN
This represents a molecule with nitrogen, carbon, and hydrogen atoms.

