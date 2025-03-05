import json
from mlflow.tracking import MlflowClient
import mlflow
import dagshub

dagshub.init(repo_owner='anu-gtb',repo_name='sleep_disorder_pred',mlflow=True)

mlflow.set_experiment('Final Model')
mlflow.set_tracking_uri('https://dagshub.com/anu-gtb/sleep_disorder_pred.mlflow')

reports_path='./reports/run_info.json'
with open(reports_path,'r') as f:
    run_info=json.load(f)
    
run_id=run_info['run_id']
model_name=run_info['model_name']

client=MlflowClient()

model_uri=f'runs:/{run_id}/artifacts/{model_name}'

reg=mlflow.register_model(model_uri,model_name)

model_version=reg.version

new_stage='Staging'

client.set_model_version_tag(
    name=model_name,
    version=model_version,
    key='stage',
    value='production'
)
print('DONE')