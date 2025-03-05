import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
import os

def load_data(filepath : str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f'Error loading data from {filepath} : {e}')

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

def scaling(df1,df2):
    sc=StandardScaler()
    df1=pd.DataFrame(sc.fit_transform(df1))
    df2=pd.DataFrame(sc.transform(df2))
    return df1,df2

def save_data(df : pd.DataFrame,filepath : str) -> None:
    try:
        df.to_csv(filepath,index=False)
    except Exception as e:
        raise Exception(f'Error saving data to {filepath} : {e}')

def main():
    try:
        raw_data_path='./data/raw'
        processed_data_path='./data/processed'
    
        train_data=load_data(os.path.join(raw_data_path,'train.csv'))
        test_data=load_data(os.path.join(raw_data_path,'test.csv'))
        
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
        
        x_train,x_test=scaling(x_train,x_test)

        os.makedirs(processed_data_path)

        save_data(x_train,os.path.join(processed_data_path,'x_train.csv'))
        save_data(y_train,os.path.join(processed_data_path,'y_train.csv'))
        save_data(x_test,os.path.join(processed_data_path,'x_test.csv'))
        save_data(y_test,os.path.join(processed_data_path,'y_test.csv'))
        
    except Exception as e:
        raise Exception(f'An error occurred : {e}')
    
if __name__=='__main__':
    main()