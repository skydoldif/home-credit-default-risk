import pandas as pd
import numpy as np

def load_data():
    train = pd.read_csv("data/raw/application_train.csv").set_index('SK_ID_CURR')

    X_trainval = preprocess(train.drop('TARGET', axis=1))
    X_test = preprocess(pd.read_csv("data/raw/application_test.csv").set_index('SK_ID_CURR'))

    description = pd.read_csv("data/raw/HomeCredit_columns_description.csv", encoding="latin1", usecols=[1,2,3,4])
    description = description[description.Table=='application_{train|test}.csv'].drop('Table', axis=1)

    description_X_trainval, numerical_vars, cat_vars, binary_vars = compute_description(X_trainval, description)

    return (
        X_trainval, 
        train['TARGET'],
        X_test,
        description,
        description_X_trainval,
        numerical_vars,
        cat_vars,
        binary_vars
    )

def preprocess(X):
    for col in X:
        if X[col].dtype == 'object' and set(X[col].unique()) == {'N', 'Y'}:
            X[col] = X[col].replace({'Y': '1', 'N': '0'}).astype('int64')
        if X[col].dtype == 'object' and ('Unknown' in set(X[col].unique()) or 'XNA' in set(X[col].unique())):
            X.loc[(X[col]=='Unknown') | (X[col]=='XNA'), col] = None
            
    X['NAME_CONTRACT_TYPE'] = X['NAME_CONTRACT_TYPE'].replace({'Revolving loans': '1', 'Cash loans': '0'}).astype('int64')
    X['CODE_GENDER'] = X['CODE_GENDER'].replace({'M': 1, 'F': 0, 'XNA': np.nan})

    return X

def compute_description(X, description):
    description['dtype'] = pd.NA
    description['TYPE'] = pd.NA
    description['nunique'] = pd.NA

    for col in X:
        description.loc[description.Row==col, 'dtype'] = X[col].dtype
        description.loc[description.Row==col, 'nunique'] = X[col].nunique()
        
        if X[col].dtype=='int64' and X[col].nunique()==2:
            if X[col].isna().sum() == 0:
                description.loc[description.Row==col, 'TYPE'] = 'BINARY'
            else:
                description.loc[description.Row==col, 'TYPE'] = 'CATEGORICAL'
        elif X[col].dtype == 'object':
            description.loc[description.Row==col, 'TYPE'] = 'CATEGORICAL'
        elif X[col].dtype == 'float64' or X[col].dtype == 'int64':
            description.loc[description.Row==col, 'TYPE'] = 'NUMERIC'
            
    description = description.merge(
        X.isna().sum().reset_index().rename({0: 'NaN count'}, axis=1), left_on='Row', right_on='index', how='inner'
    ).drop('index', axis=1)
    description['NaN %'] = (description['NaN count']/X.shape[0]*100).round(2)
    description['Special'] = description['Special'].fillna('')

    numerical_vars = description[description.TYPE=='NUMERIC'].Row.tolist()
    binary_vars = description[description.TYPE=='BINARY'].Row.tolist()
    cat_vars = description[description.TYPE=='CATEGORICAL'].Row.tolist()

    return description, numerical_vars, cat_vars, binary_vars
    
