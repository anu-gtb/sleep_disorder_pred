import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml
import os

def load_data(filepath : str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f'Error loading data from {filepath} : {e}')

def load_params(params_path : str) -> int:
    try:
        with open(params_path,'r') as f:
            params=yaml.safe_load(f)
        return params['model_building']['n_estimators']
    except Exception as e:
        raise Exception(f'Error loading parameters from {params_path} : {e}')
    
def train_model(x : pd.DataFrame,y : pd.Series, n_estimators : int) -> RandomForestClassifier:
    try:
        clf=RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(x,y)
        return clf
    except Exception as e:
        raise Exception(f'Error training model : {e}')

def save_model(model : RandomForestClassifier,filepath : str) -> None:
    try:
        with open(filepath,'wb') as file:
            pickle.dump(model,file)
    except Exception as e:
        raise Exception(f'Error saving model to {filepath}: {e}')

def main():
    try:
        parameters='params.yaml'
        xtrain_data_path='./data/processed/x_train.csv'
        ytrain_data_path='./data/processed/y_train.csv'
        model_name='./models'
        
        n_estimators=load_params(parameters)
        xtrain_data=load_data(xtrain_data_path)
        ytrain_data=load_data(ytrain_data_path)
        
        model=train_model(xtrain_data,ytrain_data,n_estimators)
        save_model(model,os.path.join(model_name,'best_model.pkl'))
    except Exception as e:
        raise Exception(f'An error occurred : {e}')
    
if __name__=='__main__':
    main()