import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, auc
from sklearn import metrics
from sklearn.model_selection import cross_val_score

def preprocessing(dataframe):
    df = dataframe.copy()
    df['purchased'] = df['action'].str.count('Purchase')
    df['total_purchased'] = df.groupby('id')['purchased'].transform(sum)
    df['has_purchased'] = np.where((df['total_purchased'] > 0),1, 0)
    df['total_actions'] = df.groupby('id')['action'].transform('count')
    df['email_open'] = df['action'].str.count('EmailOpen')
    df['total_email_open'] = df.groupby('id')['email_open'].transform(sum)
    df['has_email_open'] = np.where((df['total_email_open'] > 0),1, 0)
    df['form_submit'] = df['action'].str.count('FormSubmit')
    df['total_form_submit'] = df.groupby('id')['form_submit'].transform(sum)
    df['has_form_submit'] = np.where((df['total_form_submit'] > 0),1, 0)
    temp1 = df.drop_duplicates(subset = 'id')
    #print(temp1.shape)
    df['email_click_thru'] = df['action'].str.count('EmailClickthrough')
    df['total_email_click_thru'] = df.groupby('id')['email_click_thru'].transform(sum)
    df['has_email_click_thru'] = np.where((df['total_email_click_thru'] > 0),1, 0)
    df['cust_sup'] = df['action'].str.count('CustomerSupport')      
    df['total_cust_sup'] = df.groupby('id')['cust_sup'].transform(sum)
    df['has_cust_sup'] = np.where((df['total_cust_sup'] > 0),1, 0)
    temp2 = df.drop_duplicates(subset = 'id')
    #print(temp2.shape)
    df['page_view'] = df['action'].str.count('PageView')
    df['total_page_view'] = df.groupby('id')['page_view'].transform(sum)
    df['has_page_view'] = np.where((df['total_page_view'] > 0),1, 0)
    df['web_view'] = df['action'].str.count('WebView')
    df['total_web_view'] = df.groupby('id')['web_view'].transform(sum)
    df['has_web_view'] = np.where((df['total_web_view'] > 0),1, 0)
    temp3 = df.drop_duplicates(subset = 'id')
    #print(temp3.shape)
    tempA = dataframe.copy()
    tempA.drop_duplicates(subset = 'id', inplace = True)
    tempA = tempA.rename(columns = {'date':'first_date'})
    tempA['first_date'] = pd.to_datetime(tempA['first_date'])
    tempB =dataframe.copy()
    tempB.drop_duplicates(subset = 'id', keep = 'last', inplace = True)
    tempB = tempB.rename(columns = {'date':'last_date'})
    tempB['last_date'] = pd.to_datetime(tempB['last_date'])
    temp4 = tempA.merge(tempB, on = 'id')
    temp4['days_as_user'] = temp4.last_date - temp4.first_date
    temp4 = temp4.loc[:, ['id', 'days_as_user']]
    temp4['days_as_user'] = (temp4['days_as_user']/ np.timedelta64(1, 'D')).astype(int)
    #print(temp4.shape)
    temp2.drop(['date', 'action', 'purchased', 'total_purchased', 'has_purchased', 'total_actions'],axis = 1, inplace = True)
    temp3.drop(['date', 'action', 'purchased', 'total_purchased', 'has_purchased', 'total_actions'],axis = 1, inplace = True)
    features = pd.merge(temp1, temp2, on = 'id').merge(temp3, on = 'id').merge(temp4, on = 'id')
    del temp1,temp2,temp3,temp4
    features.set_index('id', inplace = True)
    cols= []
    for i in features.columns:
        if '_y' in i:
            cols.append(i)
    features.drop(['date', 'purchased', 'total_purchased', 'action'] + cols, axis = 1, inplace = True)
    f = []
    for i in features.columns:
        if '_x' in i:
            f.append(i.replace('_x',''))
        else:
            f.append(i)
    features.columns = f
    features = features.loc[:,~features.columns.duplicated()]
    return features

def model(X,y,features):
    rf = RandomForestClassifier(n_estimators = 500)
    forest = rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("*"*50)
    print("Feature Importance Table")
    for f in range(X.shape[1]):
        print("%2d) %-*s %f" %(f + 1, 30, features[f], importances[indices[f]]))
       
    scores = cross_val_score(rf, X, y, verbose = 5)
    print("CV scores Mean : {}".format(scores.mean()))
    return rf
    
    

def main():
    """
    First Argument is train data path
    Second Argument is path for test data.
    """
    print("*"*50)
    print("Reading Data File ....")
    train = pd.read_csv(sys.argv[1],sep='\t',header=0)
    test = pd.read_csv(sys.argv[2],sep='\t',header=0)
    print("Reading Data File Completed ")
    print("*"*50)
    print("Extracting Feature ....")
    train_features = preprocessing(train)
    test_features = preprocessing(test)
    print('Feature Extraction Completed')
    print("*"*50)
    train_features['Type'] = 'Train'
    test_features['Type'] = 'Test'
    df = pd.concat([train_features, test_features], axis = 0)
    df.drop(['email_open', 'form_submit', 'email_click_thru', 'cust_sup', 'page_view', 'web_view'], axis = 1, inplace = True)
    identity_col = ['id']
    target_col = ['has_purchased']
    category_cols = ['has_email_open', 'has_form_submit', 'has_email_click_thru', 'has_cust_sup', 'has_page_view', 'has_web_view']
    numeric_cols = ['total_actions', 'total_form_submit', 'total_email_click_thru', 'total_cust_sup', 'total_page_view',
            'total_web_view', 'days_as_user']
    other_col = ['Type']
    numeric_cat_cols = numeric_cols + category_cols
    print("*"*50)
    print("Encoding Variables .....")
    for col in category_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype('str'))
    df['has_purchased'] = encoder.fit_transform(df['has_purchased'].astype('str'))
    print('Encoding Completed')
    print("*"*50)
    train = df[df['Type'] == 'Train']
    test = df[df['Type'] == 'Test']
    features = list(set(list(df.columns)) - set(identity_col) - set(target_col) - set(other_col))
    del df
    print("Spliting Data for Train and validate")
    X_train, X_validate, y_train, y_validate = train_test_split(train[features],train['has_purchased'] , test_size=0.25)
    random.seed(2019)
    print("*"*50)
    rf = model(X_train,y_train,features)
    status = rf.predict_proba(X_validate)
    fpr, tpr, _ = roc_curve(y_validate, status[:, 1])
    roc_auc = auc(fpr, tpr)
    print ("ROC_AUC score on validation".format(roc_auc))
    X_test = test[list(features)]
    print("*"*50)
    print("Predicting the output on test data ....")
    final_status = rf.predict_proba(X_test)
    test['has_purchased'] = final_status[:, 1]
    test.drop(['Type'],axis=1,inplace=True)
    test.sort_values('has_purchased', ascending = False, inplace = True)
    test_results = test.head(1000)
    print("Saving the output to a file ....")
    test_results.to_csv('test_results.csv')
    print("Result Generated")


if __name__ == "__main__":
    main()
    
