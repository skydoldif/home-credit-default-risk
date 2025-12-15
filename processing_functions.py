import pandas as pd
import numpy as np
import xgboost as xgb

def load_data():
    train = pd.read_csv("data/raw/application_train.csv").set_index('SK_ID_CURR')

    X_trainval = preprocess(train.drop('TARGET', axis=1))
    X_test = preprocess(pd.read_csv("data/raw/application_test.csv").set_index('SK_ID_CURR'))

    description = pd.read_csv("data/raw/HomeCredit_columns_description.csv", encoding="latin1", usecols=[1,2,3,4])
    description = description[description.Table=='application_{train|test}.csv'].drop('Table', axis=1).rename({'Row': 'Feature'}, axis=1)

    description_X_trainval, numerical_vars, cat_vars, binary_vars = compute_description(X_trainval, description)

    X_trainval[cat_vars] = X_trainval[cat_vars].astype('category')
    X_test[cat_vars] = X_test[cat_vars].astype('category')

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

def feature_engineering(X):
    X['NUM_DOCUMENTS_PROVIDED'] = X[[f'FLAG_DOCUMENT_{i}' for i in range(2, 22)]].sum(axis=1)
    X['PCT_DOCUMENTS_PROVIDED'] = X['NUM_DOCUMENTS_PROVIDED']/sum(range(2,22))
    X['AMT_ANNUITY/AMT_INCOME_TOTAL'] = X['AMT_ANNUITY']/X['AMT_INCOME_TOTAL']
    X['AMT_ANNUITY/AMT_CREDIT'] = X['AMT_ANNUITY']/X['AMT_CREDIT']
    X['AMT_ANNUITY/AMT_GOODS_PRICE'] = X['AMT_ANNUITY']/X['AMT_GOODS_PRICE']
    X['AMT_CREDIT/AMT_GOODS_PRICE'] = X['AMT_CREDIT']/X['AMT_GOODS_PRICE']
    X['AMT_CREDIT/AMT_INCOME_TOTAL'] = X['AMT_CREDIT']/X['AMT_INCOME_TOTAL']
    X['CNT_CHILDREN/CNT_FAM_MEMBERS'] = X['CNT_CHILDREN']/X['CNT_FAM_MEMBERS']
    X['EMPLOYEMENT_AGE_RATIO'] = X['DAYS_EMPLOYED']/X['DAYS_BIRTH']
    X['DEF_30_CNT_SOCIAL_CIRCLE / OBS_30_CNT_SOCIAL_CIRCLE'] = X['DEF_30_CNT_SOCIAL_CIRCLE'] / X['OBS_30_CNT_SOCIAL_CIRCLE']
    X['DEF_60_CNT_SOCIAL_CIRCLE / OBS_60_CNT_SOCIAL_CIRCLE'] = X['DEF_60_CNT_SOCIAL_CIRCLE'] / X['OBS_60_CNT_SOCIAL_CIRCLE']

    education_order = {
        "Lower secondary": 1,
        "Secondary / secondary special": 2,
        "Incomplete higher": 3,
        "Higher education": 4,
        "Academic degree": 5
    }
    X['NAME_EDUCATION_TYPE'] = X['NAME_EDUCATION_TYPE'].replace(education_order).astype('int64')

    return X

def compute_description(X, description):
    description['dtype'] = pd.NA
    description['TYPE'] = pd.NA
    description['nunique'] = pd.NA

    for col in X:
        description.loc[description.Feature==col, 'dtype'] = X[col].dtype
        description.loc[description.Feature==col, 'nunique'] = X[col].nunique()
        
        if X[col].dtype=='int64' and X[col].nunique()==2:
            if X[col].isna().sum() == 0:
                description.loc[description.Feature==col, 'TYPE'] = 'BINARY'
            else:
                description.loc[description.Feature==col, 'TYPE'] = 'CATEGORICAL'
        elif X[col].dtype == 'object':
            description.loc[description.Feature==col, 'TYPE'] = 'CATEGORICAL'
        elif X[col].dtype == 'float64' or X[col].dtype == 'int64':
            description.loc[description.Feature==col, 'TYPE'] = 'NUMERIC'
            
    description = description.merge(
        X.isna().sum().reset_index().rename({0: 'NaN count'}, axis=1), left_on='Feature', right_on='index', how='inner'
    ).drop('index', axis=1)
    description['NaN %'] = (description['NaN count']/X.shape[0]*100).round(2)
    description['Special'] = description['Special'].fillna('')

    numerical_vars = description[description.TYPE=='NUMERIC'].Feature.tolist()
    binary_vars = description[description.TYPE=='BINARY'].Feature.tolist()
    cat_vars = description[description.TYPE=='CATEGORICAL'].Feature.tolist()

    return description, numerical_vars, cat_vars, binary_vars
    

def save_submission(model, X, submission_name):
    pd.DataFrame(model.predict(xgb.DMatrix(X, enable_categorical=True)), index=X.index, columns=['TARGET']).reset_index().to_csv(f"data/preprocessed/{submission_name}.csv", index=False)

def get_ordered_importance(importance):
    importance = pd.Series(importance).sort_values(ascending=False).reset_index()
    importance.columns = ['Feature', 'Importance']
    importance['%Importance'] = importance['Importance']/importance['Importance'].sum()*100
    importance['CUMSUM %Importance'] = importance['%Importance'].cumsum()
    return importance

def get_ordered_shap_importance(X, shap_values):
    shap_importance = pd.DataFrame(pd.DataFrame(shap_values, columns=X.columns).abs().mean(axis=0).sort_values(ascending=False)).reset_index()
    shap_importance.columns = ['Feature', 'shap_importance']
    shap_importance['%shap_importance'] = shap_importance['shap_importance']/shap_importance['shap_importance'].sum()*100
    shap_importance['CUMSUM %shap_importance'] = shap_importance['%shap_importance'].cumsum()
    return shap_importance