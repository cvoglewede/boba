import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from hyperopt import hp, fmin, tpe, Trials, STATUS_OK,space_eval,STATUS_FAIL
from pandas.io.json import json_normalize 
from datetime import date
from datetime import datetime


import pickle
import os


def run_corr_analysis(modeling_df, season, position_group, model_targets, counting_stats):
    modeling_df = modeling_df[modeling_df['Season']<(season - 2)]
    test_dict = {}
    for target in tqdm(model_targets):
        targets = model_targets.copy()
        targets.remove(target)
        trim_df = modeling_df.drop(targets+counting_stats+['Season','Name','playerID','position','Team','PA','IP','G','GS'],axis=1,errors='ignore')
        corr_matrix = trim_df.corr().abs()
        sort_corr = corr_matrix[[target]].sort_values(target,ascending=False)
        ordered_columns = sort_corr[sort_corr[target]>.1].index
        columns_df = corr_matrix[ordered_columns].drop([target]).drop([target],axis=1)
        in_cols = []
        out_cols = []
        for i in columns_df.columns:
            if not i in out_cols:
                in_cols.append(i)
                target_df = columns_df[[i]].sort_values(i,ascending=False)
                bool_check = (target_df[i].between(.9,.999))
                append_cols = list(target_df.index[bool_check])
                for col in append_cols:
                    if col not in out_cols:
                        out_cols.append(col)
                    else:
                        pass
            elif i in out_cols:
                pass
            else:
                print('error dummy')
        test_dict.update({target : in_cols})
    with open(r'boba/recipes/'+str(season)+'/'+position_group+'_relevant_features.yaml', 'w') as file:
        yaml.dump(test_dict, file)
    return test_dict


def isolate_relevant_columns(modeling_df, position_group, target, season):
    with open(r'boba/recipes/'+str(season)+'/'+position_group+'_relevant_features.yaml') as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
    features = [col for col in list(modeling_df.columns) if col in yaml_data[target]]
    # Team and pitch hand removed
    if position_group == 'hitters':
        column_list = ['Season']+ [target] + ['Age'] + ['PA'] + features
        model_df = modeling_df.copy()
        # model_df['Team_2yrago'] = model_df['Team_2yrago'].fillna('- - -')
        # model_df['Team_1yrago'] = model_df['Team_1yrago'].fillna('- - -')
        # model_df['Team_3yrago'] = model_df['Team_3yrago'].fillna('- - -')
        model_df = model_df[column_list]
    elif position_group == 'SP':
        column_list = ['Season']+ [target] + ['Age'] + ['IP'] + features
        model_df = modeling_df.copy()
        # model_df['Team_2yrago'] = model_df['Team_2yrago'].fillna('- - -')
        # model_df['Team_1yrago'] = model_df['Team_1yrago'].fillna('- - -')
        # model_df['Team_3yrago'] = model_df['Team_3yrago'].fillna('- - -')
        model_df['ShO_3yrago'] = model_df['ShO_3yrago'].fillna(0)
        # model_df['pitch_hand'] = model_df['pitch_hand'].fillna('R')
        model_df = model_df[column_list]
    elif position_group == 'RP':
        column_list = ['Season']+ [target] + ['Age'] + ['IP'] + features
        model_df = modeling_df.copy()
        # model_df['Team_2yrago'] = model_df['Team_2yrago'].fillna('- - -')
        # model_df['Team_1yrago'] = model_df['Team_1yrago'].fillna('- - -')
        model_df['ShO_3yrago'] = model_df['ShO_3yrago'].fillna(0)
        # model_df['pitch_hand'] = model_df['pitch_hand'].fillna('R')
        model_df = model_df[column_list]
    return model_df


def evaluation_split(model_df,target, season):
    print("Split model Dataframe into pre-{} data".format((season-1)))
    X = model_df.drop([target],axis=1)
    y = model_df[[target]]
    year_split = (model_df.Season==(season-1))
    test_split = (model_df.Season==(season-2))
    X = X[-year_split]
    y = y[-year_split]
    X_train = X[-test_split]
    X_test = X[test_split]
    y_train = y[-test_split]
    y_test = y[test_split]
    return X_train, X_test, y_train, y_test


def preprocessing_pipeline(X_train, X_test, target, knn):
    print("Fit Preprocessing Steps to X_train and transform X_test")
    numeric_features = X_train.columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=knn)),
        ('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)])
    preprocessing_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    
    model_features = numeric_features.copy()

    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    X_test_processed = preprocessing_pipeline.transform(X_test)
    X_train_processed = pd.DataFrame(X_train_processed, columns = model_features,index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_processed, columns = model_features,index=X_test.index)

    return X_train_processed, X_test_processed