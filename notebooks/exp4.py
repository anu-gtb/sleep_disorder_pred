import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import accuracy_score
import pickle
import dagshub

dagshub.init(repo_owner='anu-gtb',repo_name='sleep_disorder_pred',mlflow=True)
mlflow.set_experiment("EXP 4")
mlflow.set_tracking_uri('https://dagshub.com/anu-gtb/sleep_disorder_pred.mlflow')

data=pd.read_csv(r"C:\Users\asus\sleep_disorder_pred\sleep_disorder.csv")
train_data,test_data = train_test_split(data,test_size=0.20,stratify=data['Blood Pressure'])

def remove_cols(df):
    for i in df.columns:
        if df[i].nunique()==df.shape[0]:
            df.drop(columns=[i],inplace=True)
    return df

def fill_missing_values(df):
    for column in df.columns:
        if df[column].isnull().any():
            median=df[column].median()
            df[column].fillna(median,inplace=True)
    return df

def label(df1,df2):
    le=LabelEncoder()
    le.fit(df1['Blood Pressure'])
    df1['Blood Pressure']=le.fit_transform(df1['Blood Pressure'])
    df2['Blood Pressure']=le.transform(df2['Blood Pressure'])
    return df1,df2

def onehot(df1,df2,columns_to_encode=None,drop='first',sparse_output=False):
    categorical_columns=[]
    for i in df1.columns:
        if df1[i].dtypes=='object':
            categorical_columns.append(i)## List of all categorical columns
    ohe=OneHotEncoder(drop=drop,sparse_output=sparse_output)
    ohe.fit(df1[categorical_columns])
    
    encoded1=ohe.transform(df1[categorical_columns])
    train_feat1=ohe.get_feature_names_out(categorical_columns)
    train_encoded=pd.DataFrame(encoded1,index=df1.index,columns=train_feat1)
    
    encoded2=ohe.transform(df2[categorical_columns])
    test_encoded=pd.DataFrame(encoded2,index=df2.index,columns=train_feat1)
    
    df1=df1.drop(categorical_columns,axis=1).join(train_encoded)
    df2=df2.drop(categorical_columns,axis=1).join(test_encoded)
    return df1,df2

def standardize(df1,df2):
    sc=StandardScaler()
    df1=pd.DataFrame(sc.fit_transform(df1))
    df2=pd.DataFrame(sc.transform(df2))
    return df1,df2

train_data=remove_cols(train_data)
test_data=remove_cols(test_data)
        
train_data=fill_missing_values(train_data)
test_data=fill_missing_values(test_data)
        
train_data,test_data=label(train_data,test_data)
train_data,test_data=onehot(train_data,test_data)

x_train=train_data.iloc[:,:-3]
y_train=train_data.iloc[:,-3:]
x_test=test_data.iloc[:,:-3]
y_test=test_data.iloc[:,-3:]

x_train,x_test=standardize(x_train,x_test)

rf=RandomForestClassifier(random_state=42)
param_dist={
    'n_estimators':[100,200,300,500,700,1000],
    'max_depth':[None,4,5,7,8,10]
}

random_search=RandomizedSearchCV(estimator=rf,param_distributions=param_dist,n_iter=50,cv=5,n_jobs=-1,verbose=2,random_state=42)

with mlflow.start_run(run_name='Sleep Disorder Prediction') as parent_run:
    random_search.fit(x_train,y_train)
    
    for i in range(len(random_search.cv_results_['params'])):
        with mlflow.start_run(run_name=f'Combination{i+1}',nested=True) as child_run:
                mlflow.log_params(random_search.cv_results_['params'][i])
                mlflow.log_metric('mean test score',random_search.cv_results_['mean_test_score'][i])
    
    print('Best parameters :',random_search.best_params_)
    mlflow.log_params(random_search.best_params_)
    best_rf=random_search.best_estimator_
    best_rf.fit(x_train,y_train)     
    pickle.dump(best_rf,open('model.pkl','wb'))
    y_pred = best_rf.predict(x_test)

    acc = accuracy_score(y_test,y_pred)
                
    mlflow.log_metric("acc",acc)
    
    train_df=mlflow.data.from_pandas(train_data)
    test_df=mlflow.data.from_pandas(test_data)
    
    sign=infer_signature(x_test,y_pred)
    mlflow.log_input(train_df,'train')
    mlflow.log_input(test_df,'test')
    mlflow.sklearn.log_model(best_rf,'Best Model')
    mlflow.log_artifact(__file__)
    mlflow.set_tag('author','anubha')
    mlflow.set_tag('model','Best model')