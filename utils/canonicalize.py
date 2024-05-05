from rdkit import Chem
from rdkit.Chem import Descriptors

def canonicalize_smiles(df, smiles_col='smiles'):
    def get_canonical(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
        else:
            return smiles

    df[smiles_col] = df[smiles_col].apply(get_canonical)
    return df

def validate_and_filter_smiles(smiles_list):
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:  
                Descriptors.ExactMolWt(mol)  
                valid_indices.append(i)
        except Chem.rdchem.AtomValenceException as e:
            print(f"AtomValenceException at index {i} for SMILES '{smi}': {str(e)}")
        except Exception as e:
            print(f"Other exception at index {i} for SMILES '{smi}': {str(e)}")
    return valid_indices