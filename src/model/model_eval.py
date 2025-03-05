import pandas as pd
import pickle
import json
import os
import mlflow
from mlflow.sklearn import log_model
from mlflow.models import infer_signature
from mlflow import log_artifact
import dagshub
from sklearn.metrics import accuracy_score

dagshub.init(repo_owner='anu-gtb',repo_name='sleep_disorder_pred',mlflow=True)
mlflow.set_experiment('DVC-Pipeline')
mlflow.set_tracking_uri('https://dagshub.com/anu-gtb/sleep_disorder_pred.mlflow')

def load_data(filepath : str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f'Error loading data from {filepath}: {e}')
    
def load_model(filepath:str):
    try:
        with open(filepath,'rb') as f:
            model=pickle.load(f)
        return model
    except Exception as e:
        raise Exception(f'Error loading model : {e}')

def evaluation(model,x_test : pd.DataFrame, y_test:pd.Series) -> dict:
    try:  
        y_pred=model.predict(x_test)

        acc=accuracy_score(y_test,y_pred)
    
        metrics_dict={
            'acc':acc
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f'Error evaluating model : {e}')

def save_metrics(metrics_dict:dict,filepath:str)->None:
    try:
        with open(filepath,'w') as file:
            json.dump(metrics_dict,file,indent=1)
    except Exception as e:
        raise Exception(f'ERROR : {e}')
    
def main():
    try:
        xtest_data_path='./data/processed/x_test.csv'
        ytest_data_path='./data/processed/y_test.csv'
        model_path='./models/best_model.pkl'
        metrics_path='./reports'
        
        xtest_data=load_data(xtest_data_path)
        ytest_data=load_data(ytest_data_path)
        
        model=load_model(model_path)
        
        with mlflow.start_run(run_name='RF_FINAL') as run:
            metrics=evaluation(model,xtest_data,ytest_data)
            mlflow.log_metric('accuracy',metrics['acc'])
            save_metrics(metrics,os.path.join(metrics_path,'metrics.json'))
            
            log_artifact(model_path)
            log_artifact(metrics_path)
            log_artifact(__file__)
            sign=infer_signature(xtest_data,model.predict(xtest_data))
            log_model(model,'Best Model',signature=sign)
            
            run_info={'run_id':run.info.run_id,'model_name':'Best Model'}
            reports_path='./reports/run_info.json'
            with open(reports_path,'w') as f:
                json.dump(run_info,f,indent=1)
    
    except Exception as e:
        raise Exception(f'Error :{e}')
    
if __name__=='__main__':
    main()