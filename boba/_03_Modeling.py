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
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK,space_eval,STATUS_FAIL
from pandas.io.json import json_normalize 
from datetime import date
from datetime import datetime


import pickle
import os

from .utils import Boba_Utils as u
from ._04_ModelDiagnostics import Boba_Model_Diagnostics as d
from ._05_SystemDiagnostics import Boba_Sys_Diagnostics as sd

class Boba_Modeling(u,d,sd):

    def __init__(self):
        pass

    def isolate_relevant_columns(self, modeling_df, target):
        with open(r'boba/recipes/'+self.position_group+'_relevant_features.yaml') as file:
            yaml_data = yaml.load(file, Loader=yaml.FullLoader)
        features = [col for col in list(modeling_df.columns) if col in yaml_data[target]]
        if self.position_group == 'hitters':
            column_list = ['Season','Team']+ [target] + features
            model_df = modeling_df.copy()
            model_df['Team_2yrago'] = model_df['Team_2yrago'].fillna('- - -')
            model_df['Team_1yrago'] = model_df['Team_1yrago'].fillna('- - -')
            model_df['Team_3yrago'] = model_df['Team_3yrago'].fillna('- - -')
            model_df = model_df[column_list]
        elif self.position_group == 'SP':
            column_list = ['Season','Team','pitch_hand']+ [target] + features
            model_df = modeling_df.copy()
            model_df['Team_2yrago'] = model_df['Team_2yrago'].fillna('- - -')
            model_df['Team_1yrago'] = model_df['Team_1yrago'].fillna('- - -')
            model_df['Team_3yrago'] = model_df['Team_3yrago'].fillna('- - -')
            model_df['ShO_3yrago'] = model_df['ShO_3yrago'].fillna(0)
            model_df['pitch_hand'] = model_df['pitch_hand'].fillna('R')
            model_df = model_df[column_list]
        elif self.position_group == 'RP':
            column_list = ['Season','Team','pitch_hand']+ [target] + features
            model_df = modeling_df.copy()
            model_df['Team_2yrago'] = model_df['Team_2yrago'].fillna('- - -')
            model_df['Team_1yrago'] = model_df['Team_1yrago'].fillna('- - -')
            model_df['ShO_3yrago'] = model_df['ShO_3yrago'].fillna(0)
            model_df['pitch_hand'] = model_df['pitch_hand'].fillna('R')
            model_df = model_df[column_list]
        return model_df

    def evaluation_split(self,model_df,target,test_size, method = 'timeseries'):
        print("Split model Dataframe into pre-{} data".format((self.year-1)))
        X = model_df.drop([target],axis=1)
        y = model_df[[target]]
        year_split = (model_df.Season==(self.year-1))
        test_split = (model_df.Season==(self.year-2))
        X = X[-year_split]
        y = y[-year_split]
        if method == 'timeseries':
            self.eval_split_method = 'timeseries'
            X_train = X[-test_split]
            X_test = X[test_split]
            y_train = y[-test_split]
            y_test = y[test_split]
        else:
            self.eval_split_method = 'random'
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, random_state=self.seed)
        return X_train, X_test, y_train, y_test

    def production_split(self,model_df, target, test_size, method = 'timeseries'):
        print("Split data into true train/test")
        X = model_df.drop([target],axis=1)
        y = model_df[[target]]
        test_split = (model_df.Season==(self.year-1))
        if method == 'timeseries':
            self.prod_split_method = 'timeseries'
            X_train = X[-test_split]
            X_test = X[test_split]
            y_train = y[-test_split]
            y_test = y[test_split]
        else:
            self.prod_split_method = 'random'
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, random_state=self.seed)
        return X_train, X_test, y_train, y_test

    def preprocessing_pipeline(self, X_train, X_test, target, prod):
        print("Fit Preprocessing Steps to X_train and transform X_test")
        cat_cols = list(set(X_train.nunique()[X_train.nunique()<3].keys().tolist() 
                    + X_train.select_dtypes(include='object').columns.tolist()))
        categorical_features = [x for x in cat_cols]
        numeric_features = [x for x in X_train.columns if x not in cat_cols]
        numeric_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=self.knn)),
            ('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore',sparse=False))])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])
        preprocessing_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
        
        # temp_imputer = SimpleImputer(strategy='constant', fill_value='missing')
        # temp_ohe = temp_imputer.fit_transform(X_train[categorical_features])
        ohe = OneHotEncoder(handle_unknown='ignore',sparse=False)
        ohe.fit(X_train[categorical_features])
        model_features = numeric_features + list(ohe.get_feature_names())

        X_train_processed = preprocessing_pipeline.fit_transform(X_train)
        X_test_processed = preprocessing_pipeline.transform(X_test)
        X_train_processed = pd.DataFrame(X_train_processed, columns = model_features,index=X_train.index)
        X_test_processed = pd.DataFrame(X_test_processed, columns = model_features,index=X_test.index)

        if prod:
            if os.path.exists('data/modeling/'+self.position_group+'/'+target):
                model_features_dict = {target:model_features}
                with open(r'data/modeling/'+self.position_group+'/'+target+'/model_features_prod.yaml', 'w') as file:
                    yaml.dump(model_features_dict, file)
                filename = 'data/modeling/'+self.position_group+'/'+target+'/preprocessing_pipeline_prod.sav'
                pickle.dump(preprocessing_pipeline, open(filename, 'wb'))
            else:
                os.mkdir('data/modeling/'+self.position_group+'/'+target)
                model_features_dict = {target:model_features}
                with open(r'data/modeling/'+self.position_group+'/'+target+'/model_features_prod.yaml', 'w') as file:
                    yaml.dump(model_features_dict, file)
                filename = 'data/modeling/'+self.position_group+'/'+target+'/preprocessing_pipeline_prod.sav'
                pickle.dump(preprocessing_pipeline, open(filename, 'wb'))            
        else:
            if os.path.exists('data/modeling/'+self.position_group+'/'+target):
                model_features_dict = {target:model_features}
                with open(r'data/modeling/'+self.position_group+'/'+target+'/model_features_eval.yaml', 'w') as file:
                    yaml.dump(model_features_dict, file)
                filename = 'data/modeling/'+self.position_group+'/'+target+'/preprocessing_pipeline_eval.sav'
                pickle.dump(preprocessing_pipeline, open(filename, 'wb'))
            else:
                os.mkdir('data/modeling/'+self.position_group+'/'+target)
                model_features_dict = {target:model_features}
                with open(r'data/modeling/'+self.position_group+'/'+target+'/model_features_eval.yaml', 'w') as file:
                    yaml.dump(model_features_dict, file)
                filename = 'data/modeling/'+self.position_group+'/'+target+'/preprocessing_pipeline_eval.sav'
                pickle.dump(preprocessing_pipeline, open(filename, 'wb')) 
        return X_train_processed, X_test_processed




    def build_model(self,X_train, X_test,y_train, y_test,target, prod, max_evals,verbose):
        verbose_setting = 1 if verbose==True else 0
        print("Building model for {}".format(target))

        xgb_reg_params = {
            'learning_rate':    hp.uniform('learning_rate', 0, .8),
            'max_depth':        hp.choice('max_depth',        np.arange(1, 15, 1, dtype=int)),
            'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
            'colsample_bytree': hp.uniform('colsample_bytree', .6, 1),
            'subsample':        hp.uniform('subsample', .7, 1),
            'n_estimators':     hp.choice('n_estimators', np.arange(50, 1000, 50, dtype=int))
        }
        xgb_fit_params = {
            'eval_metric': 'rmse',
            'early_stopping_rounds': 50,
            'verbose': False
        }

        xgb_params = dict()
        xgb_params['reg_params'] = xgb_reg_params
        xgb_params['fit_params'] = xgb_fit_params
        xgb_params['loss_func' ] = lambda y, pred: np.sqrt(mean_squared_error(y, pred))
        max_evals = max_evals

        trials = Trials()
        opt_params ,trials = self.Bayes_param_optimization(X_train, X_test, y_train, y_test,  algo=tpe.suggest, trials=trials,space=xgb_params,max_evals=max_evals)
        grid_columns = ['Date','DateTime','environment','position_group','target','algo','rmse'
                        ,'n_trees','learning_rate','max_depth','colsample_bytree','min_child_weight','subsample']
        search_df = self.create_param_search_results(target = target, trials=trials,max_evals= max_evals,xgb_reg_params=xgb_reg_params, prod=prod,grid_columns=grid_columns)
        self.write_param_search_results(search_df=search_df,grid_columns=grid_columns)
        model = xgb.XGBRegressor(**opt_params)
        eval_set = [(X_train, y_train),(X_test, y_test)]
        model.fit(X_train,y_train, early_stopping_rounds=xgb_fit_params['early_stopping_rounds'], eval_metric= xgb_fit_params['eval_metric'], eval_set=eval_set,verbose=verbose_setting)
        
        self.run_model_diagnostics(model=model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, target=target)


        evaluation_df = self.run_sys_scoring(model=model, target=target,prod=prod)
        evaluation_results_df = self.run_sys_diagnostics(evaluation_df=evaluation_df, target=target,prod=prod)

        results_df, results_cols = self.create_model_results(model=model,X_train=X_train, X_test=X_test,y_train=y_train, y_test=y_test,target=target, prod=prod, evaluation_results_df = evaluation_results_df)
        self.write_model_results(results_df=results_df,results_cols=results_cols)
        
        if prod == True:
            env = 'prod'
        else:
            env = 'eval'
        filename_prod = ('models/'+self.position_group+'/'+target+'_'+env+'.sav')
        pickle.dump(model, open(filename_prod, 'wb'))

        return model



    def create_model_results(self,model,X_train, X_test,y_train, y_test,target,prod, evaluation_results_df):
        results_columns = ['Date','DateTime','environment','position_group','target','algo','Evaluation_WinShare','Evaluation_RMSE','Evaluation_R2','test_RMSE','test_R2','test_MAE','train_RMSE','train_R2','train_MAE'
                        ,'n_trees','learning_rate','max_depth','colsample_bytree','min_child_weight','subsample'
                        ,'seed','knn','pt_min','pt_keep_thres','pt_drop','start_year','eval_split_method','prod_split_method']

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        results_df = pd.DataFrame(columns=results_columns)
        d = {
            'learning_rate': model.get_params()['learning_rate'],
            'colsample_bytree': model.get_params()['colsample_bytree'],
            'min_child_weight': model.get_params()['min_child_weight'],
            'max_depth': model.get_params()['max_depth'],
            'n_trees': model.get_params()['n_estimators'],
            'subsample': model.get_params()['subsample'],
            
            'test_RMSE': np.sqrt(mean_squared_error(y_test, test_pred)),
            'test_R2': model.score(X_test,y_test),
            'test_MAE': mean_absolute_error(y_test, test_pred),
            
            'train_RMSE': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_R2': model.score(X_train,y_train),
            'train_MAE': mean_absolute_error(y_train, train_pred),
                        }
        df = pd.DataFrame(data=d, index=[0])
        results_df = pd.concat([results_df,df])
        results_df['Date'] = date.today()
        results_df['DateTime'] = datetime.now()
        results_df['target'] = target
        results_df['environment'] = 'prod' if prod == True else 'eval'
        results_df['position_group'] = self.position_group
        results_df['algo'] = 'xgb'
        results_df['seed'] = self.seed
        results_df['knn'] = self.knn
        results_df['pt_min'] = self.pt_min
        results_df['pt_keep_thres'] = self.pt_keep_thres
        results_df['pt_drop'] = self.pt_drop
        results_df['start_year'] = self.start_year
        results_df['eval_split_method'] = self.eval_split_method
        results_df['prod_split_method'] = self.prod_split_method
        

        if prod == True:
            results_df['Evaluation_WinShare'] = 0
        else:
            results_df['Evaluation_WinShare'] = evaluation_results_df[evaluation_results_df['system']=='BOBA']['WinShare'][0]
        
        rate_stats = [c+'_per_'+self.per_metric for c in self.counting_stats]
        if target in rate_stats:
            colname = target.replace('_per_'+self.per_metric, '')
        else:
            colname = target

        print(colname)
        if prod == True:
            results_df['Evaluation_R2'] = 0
        else:
            results_df['Evaluation_R2'] = evaluation_results_df[evaluation_results_df['system']=='BOBA']['R2'][0]

        if prod == True:
            results_df['Evaluation_RMSE'] = 0
        else:
            results_df['Evaluation_RMSE'] = evaluation_results_df[evaluation_results_df['system']=='BOBA']['RMSE'][0]

        results_df = results_df[results_columns]
        return results_df, results_columns


    def write_model_results(self,results_df,results_cols):
        if os.path.exists('models/model_results.csv'):
            df_old = pd.read_csv('models/model_results.csv')
            df_new = pd.concat([df_old,results_df])
            df_new = df_new[results_cols]
            df_new.to_csv('models/model_results.csv',index=False)
        else: 
            template_df = pd.DataFrame(columns=results_cols)
            df_new = pd.concat([template_df,results_df])
            df_new = df_new[results_cols]
            df_new.to_csv('models/model_results.csv',index=False)

    def create_param_search_results(self,target, trials,max_evals,xgb_reg_params,prod,grid_columns):
        search_df = pd.DataFrame(columns=grid_columns)
        for i in np.arange(0,max_evals,1):
            loss = trials.results[i]['loss']
            raw_params = trials.trials[i]['misc']['vals'].copy()
            for key,val in raw_params.items():
                raw_params[key] = val[0]
            params = space_eval(xgb_reg_params,raw_params)
            d = {'rmse': [loss], 
                'learning_rate': [params['learning_rate']],
                'colsample_bytree': [params['colsample_bytree']],
                'min_child_weight': [params['min_child_weight']],
                'max_depth': [params['max_depth']],
                'n_trees': [params['n_estimators']],
                'subsample': [params['subsample']],
                }
            df = pd.DataFrame(data=d)
            search_df = pd.concat([search_df,df])
        search_df['Date'] = date.today()
        search_df['DateTime'] = datetime.now()
        search_df['target'] = target
        search_df['environment'] = 'prod' if prod == True else 'eval'
        search_df['position_group'] = self.position_group
        search_df['algo'] = 'xgb'
        search_df = search_df[grid_columns]
        return search_df

    def write_param_search_results(self,search_df,grid_columns):
        if os.path.exists('models/grid_search_results.csv'):
            df_old = pd.read_csv('models/grid_search_results.csv')
            df_new = pd.concat([df_old,search_df])
            df_new = df_new[grid_columns]
            df_new.to_csv('models/grid_search_results.csv',index=False)
        else: 
            template_df = pd.DataFrame(columns=grid_columns)
            df_new = pd.concat([template_df,search_df])
            df_new = df_new[grid_columns]
            df_new.to_csv('models/grid_search_results.csv',index=False)


    def generate_prod_predictions(self, target):
        scoring_raw_df = pd.read_csv('data/scoring/scoring_raw_'+self.position_group+'.csv',index_col=0) 
        path = 'data/projections/'+self.position_group+'_'+str(self.year)+'_raw.csv'
        if os.path.exists(path):
                print('does exist')
                scoring_df = pd.read_csv(path,index_col=0)
        else:
            data_group = 'hitters' if self.position_group == 'hitters' else 'pitchers'
            zips_df = pd.read_csv('data/raw/'+data_group+'/projection_systems/zips/'+str(self.year)+'.csv')
            atc_df = pd.read_csv('data/raw/'+data_group+'/projection_systems/atc/'+str(self.year)+'.csv')
            bat_df = pd.read_csv('data/raw/'+data_group+'/projection_systems/thebat/'+str(self.year)+'.csv')
            stmr_df = pd.read_csv('data/raw/'+data_group+'/projection_systems/steamer/'+str(self.year)+'.csv')
            zips_df = zips_df.rename(columns={"K/9": "K_per_9", "BB/9": "BB_per_9"})
            atc_df = atc_df.rename(columns={"K/9": "K_per_9", "BB/9": "BB_per_9"})
            bat_df = bat_df.rename(columns={"K/9": "K_per_9", "BB/9": "BB_per_9"})
            stmr_df = stmr_df.rename(columns={"K/9": "K_per_9", "BB/9": "BB_per_9"})

            if self.position_group == 'hitters':
                scoring_df = scoring_raw_df[self.information_cols]
                for c in self.rate_stats:
                    scoring_df[c] = 0
                for c in self.counting_stats:
                    scoring_df[c] = 0
                zips_df = zips_df[['playerid']+[self.pt_metric]+[x for x in zips_df.columns if x in self.model_targets]+[x for x in zips_df.columns if x in self.counting_stats]]
                atc_df = atc_df[['playerid']+[self.pt_metric]+[x for x in atc_df.columns if x in self.model_targets]+[x for x in atc_df.columns if x in self.counting_stats]]
                bat_df = bat_df[['playerid']+[self.pt_metric]+[x for x in bat_df.columns if x in self.model_targets]+[x for x in bat_df.columns if x in self.counting_stats]]
                stmr_df = stmr_df[['playerid']+[self.pt_metric]+[x for x in stmr_df.columns if x in self.model_targets]+[x for x in stmr_df.columns if x in self.counting_stats]]
            else:
                scoring_df = scoring_raw_df[self.information_cols]
                for c in self.rate_stats:
                    scoring_df[c] = 0
                for c in self.counting_stats:
                    scoring_df[c] = 0
                zips_df = zips_df[['playerid']+[self.pt_metric]+[self.per_metric]+[x for x in zips_df.columns if x in self.model_targets]+[x for x in zips_df.columns if x in self.counting_stats]]
                atc_df = atc_df[['playerid']+[self.pt_metric]+[self.per_metric]+[x for x in atc_df.columns if x in self.model_targets]+[x for x in atc_df.columns if x in self.counting_stats]]
                bat_df = bat_df[['playerid']+[self.pt_metric]+[self.per_metric]+[x for x in bat_df.columns if x in self.model_targets]+[x for x in bat_df.columns if x in self.counting_stats]]
                stmr_df = stmr_df[['playerid']+[self.pt_metric]+[self.per_metric]+[x for x in stmr_df.columns if x in self.model_targets]+[x for x in stmr_df.columns if x in self.counting_stats]]
            

            scoring_df['playerID'] = scoring_df['playerID'].astype('str')
            scoring_df = pd.merge(scoring_df,zips_df,how='left',left_on='playerID', right_on='playerid',suffixes=('','_zips')).drop('playerid',axis=1)
            scoring_df = pd.merge(scoring_df,atc_df,how='left',left_on='playerID', right_on='playerid',suffixes=('','_atc')).drop('playerid',axis=1)
            scoring_df = pd.merge(scoring_df,bat_df,how='left',left_on='playerID', right_on='playerid',suffixes=('','_bat')).drop('playerid',axis=1)
            scoring_df = pd.merge(scoring_df,stmr_df,how='left',left_on='playerID', right_on='playerid',suffixes=('','_stmr')).drop('playerid',axis=1)
            scoring_df = scoring_df.drop(self.rate_stats,axis=1,errors='ignore')
            scoring_df = scoring_df.drop(self.counting_stats,axis=1,errors='ignore')

        temp_df = scoring_raw_df.copy()
        temp_df = temp_df.drop(['Name_zips'],axis=1,errors='ignore')
        temp_df = temp_df.dropna(subset = ['playerID','Name'])
        temp_df['Season'] = (self.year-1)
        temp_df = self.isolate_relevant_columns_scoring(modeling_df = temp_df,target = target)
        pipeline = pickle.load(open('data/modeling/'+self.position_group+'/'+target+'/preprocessing_pipeline_prod.sav', 'rb'))
        temp_df_2 = pipeline.transform(temp_df)
        with open(r'data/modeling/'+self.position_group+'/'+target+'/model_features_prod.yaml') as file:
                        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
        model_features = yaml_data[target]
        temp_df = pd.DataFrame(temp_df_2, columns = model_features,index=temp_df.index)
        model = pickle.load(open('models/'+self.position_group+'/'+target+'_prod.sav', 'rb'))
        temp_df[target+'_Boba'] = model.predict(temp_df)
        temp_df = temp_df[[target+'_Boba']]
        scoring_df = scoring_df.drop([target+'_Boba'],axis=1,errors='ignore')
        new_df = pd.merge(scoring_df,temp_df,left_index=True,right_index=True)
        new_df.to_csv(path)
        # new_df = temp_df.copy()

        return scoring_raw_df, scoring_df, new_df



    def isolate_relevant_columns_scoring(self, modeling_df, target):
        with open(r'boba/recipes/'+self.position_group+'_relevant_features.yaml') as file:
            yaml_data = yaml.load(file, Loader=yaml.FullLoader)
        features = [col for col in list(modeling_df.columns) if col in yaml_data[target]]
        if self.position_group == 'hitters':
            column_list = ['Season','Team'] + features
            model_df = modeling_df.copy()
            model_df['Team_2yrago'] = model_df['Team_2yrago'].fillna('- - -')
            model_df['Team_1yrago'] = model_df['Team_1yrago'].fillna('- - -')
            model_df['Team_3yrago'] = model_df['Team_3yrago'].fillna('- - -')
            model_df = model_df[column_list]
        elif self.position_group == 'SP':
            column_list = ['Season','Team','pitch_hand'] + features
            model_df = modeling_df.copy()
            model_df['Team_2yrago'] = model_df['Team_2yrago'].fillna('- - -')
            model_df['Team_1yrago'] = model_df['Team_1yrago'].fillna('- - -')
            model_df['Team_3yrago'] = model_df['Team_3yrago'].fillna('- - -')
            model_df['ShO_3yrago'] = model_df['ShO_3yrago'].fillna(0)
            model_df['pitch_hand'] = model_df['pitch_hand'].fillna('R')
            model_df = model_df[column_list]
        elif self.position_group == 'RP':
            column_list = ['Season','Team','pitch_hand']+ features
            model_df = modeling_df.copy()
            model_df['Team_2yrago'] = model_df['Team_2yrago'].fillna('- - -')
            model_df['Team_1yrago'] = model_df['Team_1yrago'].fillna('- - -')
            model_df['ShO_3yrago'] = model_df['ShO_3yrago'].fillna(0)
            model_df['pitch_hand'] = model_df['pitch_hand'].fillna('R')
            model_df = model_df[column_list]
        return model_df

    def xgb_reg(self,params):
        reg = xgb.XGBRegressor(**params['reg_params'])
        return self.train_reg(reg)

    def train_reg(self,reg):
        reg.fit(self.X_train, self.y_train,eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],**self.space['fit_params'])
        pred = reg.predict(self.X_test)
        loss = self.space['loss_func'](self.y_test, pred)
        return {'loss': loss, 'status': STATUS_OK}


    def Bayes_param_optimization(self,X_train, X_test, y_train, y_test, trials, algo,space,max_evals):
        self.space = space

        try:
            result = fmin(fn=self.xgb_reg, space=space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        opt_params = space_eval(space['reg_params'], result)
        return opt_params, trials









