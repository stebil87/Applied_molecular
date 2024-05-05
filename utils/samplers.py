import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

def prepare_data(df):
    X = df.drop(['BBB', 'smiles'], axis=1)
    y = df['BBB']
    smiles = df['smiles']
    return X, y, smiles

def under_sample(X, y, smiles):
    rus = RandomUnderSampler(random_state=42)
    X_rus, y_rus = rus.fit_resample(X, y)
    df_rus = pd.DataFrame(X_rus, columns=X.columns)
    df_rus['BBB'] = y_rus
    df_rus['smiles'] = smiles.iloc[y_rus.index].values
    return df_rus

def over_sample(X, y, smiles):
    ros = RandomOverSampler(random_state=42)
    X_ros, y_ros = ros.fit_resample(X, y)
    df_ros = pd.DataFrame(X_ros, columns=X.columns)
    df_ros['BBB'] = y_ros
    df_ros['smiles'] = ['no_smiles' if i >= len(smiles) else smiles.iloc[i] for i in range(len(df_ros))]
    return df_ros
