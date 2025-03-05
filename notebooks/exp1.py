import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler,MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import pickle
import dagshub

dagshub.init(repo_owner='anu-gtb',repo_name='sleep_disorder_pred',mlflow=True)
mlflow.set_experiment("EXP 1")
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

train_data=remove_cols(train_data)
test_data=remove_cols(test_data)
        
train_data=fill_missing_values(train_data)
test_data=fill_missing_values(test_data)
        
train_data,test_data=label(train_data,test_data)
train_data,test_data=onehot(train_data,test_data)
        
#train_processed_data,test_processed_data=standardize(train_data,test_data)

x_train=train_data.iloc[:,:-3]
y_train=train_data.iloc[:,-3:]
x_test=test_data.iloc[:,:-3]
y_test=test_data.iloc[:,-3:]

models={
    'DecisionTree':DecisionTreeClassifier(),
    'KNN':KNeighborsClassifier(),
    'RF':RandomForestClassifier(),
    'XGB':XGBClassifier()
}

with mlflow.start_run(run_name='Sleep Disorder Prediction'):
    for model_name,model in models.items():
        with mlflow.start_run(run_name=model_name,nested=True):
                model.fit(x_train,y_train)
                
                pickle.dump(model,open("model.pkl","wb"))
                
                y_pred = model.predict(x_test)

                acc = accuracy_score(y_test,y_pred)
                #pre = precision_score(y_test,y_pred)
                #rec = recall_score(y_test,y_pred)
                #f1 = f1_score(y_test,y_pred)
                
                mlflow.log_metric("acc",acc)
                #mlflow.log_metric("pre",pre)
                #mlflow.log_metric("rec",rec)
                #mlflow.log_metric("f1",f1)
                mlflow.sklearn.log_model(model,model_name)
                mlflow.log_artifact(__file__)
                mlflow.set_tag('author','anubha')
                mlflow.set_tag('model',model_name)