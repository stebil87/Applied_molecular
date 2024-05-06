from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

def apply_RFECV_and_add(df):
    X = df.drop(['BBB', 'smiles'], axis=1)
    y = df['BBB']
    model = RandomForestClassifier()
    cv = StratifiedKFold(5)
    rfe = RFECV(estimator=model, step=100, cv=cv, scoring='accuracy')
    X_rfe = rfe.fit_transform(X, y)
    selected_features = X.columns[rfe.support_]
    sel_df = df[selected_features.tolist() + ['BBB', 'smiles']]
    return sel_df