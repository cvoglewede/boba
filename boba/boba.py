
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import boto3
from tqdm import tqdm
import yaml

from ._01_ETL import Boba_ETL as etl
from ._02_Preprocessing import Boba_Preprocessing as pp
from ._03_Modeling import Boba_Modeling as m


class BobaModeling(etl,pp,m):

    def __init__(self, year, position_group):
        self.s3_client = boto3.client('s3')
        self.bucket = "boba-voglewede"
        self.year = year
        self.information_cols = ['Season','Name','playerID','position','Team','Age']
        self.position_group = position_group
        if self.position_group=='hitters': 
            self.per_metric = 'PA' 
            self.counting_stats = ['HR','R','RBI','WAR','SB','CS']
            self.rate_stats = ['AVG','OBP','SLG','BABIP','BB%','K%','wOBA']
            self.model_targets = self.rate_stats + [c+'_per_'+self.per_metric for c in self.counting_stats]  
            self.pt_metric = 'PA'      
        elif self.position_group=='SP':
            self.per_metric = 'GS' 
            self.counting_stats = ['ShO','CG','W','WAR'] 
            self.rate_stats = ['ERA','BB_per_9','K_per_9','OBP','SLG']
            self.model_targets = self.rate_stats + [c+'_per_'+self.per_metric for c in self.counting_stats]
            self.pt_metric = 'IP'
        elif self.position_group=='RP':
            self.per_metric = 'G' 
            self.counting_stats = ['SV','HLD','WAR']
            self.rate_stats = ['ERA','BB_per_9','K_per_9','OBP','SLG']
            self.model_targets = self.rate_stats + [c+'_per_'+self.per_metric for c in self.counting_stats]
            self.pt_metric = 'IP'
        else:
            pass

    def __repr__(self):
        return "This is the way"


    def scrape_raw_season_data(self, source, start_year,end_year, writeS3=False):
        seasons = list(np.arange(start_year,end_year+1))
        statcast_seasons = list(np.arange(2015,end_year+1))
        data_group = 'hitters' if self.position_group == 'hitters' else 'pitchers'
        if data_group == 'hitters':
            print("gather data for {} through {} seasonal hitting data".format(start_year,end_year))
            for i in tqdm(seasons):
                etl.FG_hitters_season(self,season=i,writeS3=writeS3)
            print("Fangraphs scrape completed")
            for i in tqdm(statcast_seasons):
                etl.statcast_hitters_season(self,season=i,writeS3=writeS3)
            print("Statcast scrape completed")
        elif data_group == 'pitchers':    
            print("gather data for {} through {} seasonal pitching data".format(start_year,end_year))
            for i in tqdm(seasons):
                etl.FG_pitchers_season(self,season=i,writeS3=writeS3)
            print("Fangraphs scrape completed")
            for i in tqdm(statcast_seasons):
                etl.statcast_pitchers_season(self,season=i,writeS3=writeS3)
            print("Statcast scrape completed")
        else: 
            pass

    def prepare_source_masters(self, writeS3 = False):
        data_group = 'hitters' if self.position_group == 'hitters' else 'pitchers'
        if data_group == 'hitters':
            etl.gather_source_masters(self,position_group=self.position_group, source = 'fangraphs', writeS3 = False)
            etl.gather_source_masters(self,position_group=self.position_group, source = 'statcast', writeS3 = False)
        elif data_group == 'pitchers':  
            etl.gather_source_masters(self,position_group=self.position_group, source = 'fangraphs', writeS3 = False)
            etl.gather_source_masters(self,position_group=self.position_group, source = 'statcast', writeS3 = False)
        else:
            pass  
    

    def create_master_data(self,start_year=2014):
        self.start_year = start_year
        with open(r'boba/recipes/preprocessing_parameters.yaml') as file:
            yaml_data = yaml.load(file, Loader=yaml.FullLoader)

        self.pt_min = yaml_data['parameters'][self.position_group]['pt_min']
        self.pt_keep_thres = yaml_data['parameters'][self.position_group]['pt_keep_thres']
        self.pt_drop = yaml_data['parameters'][self.position_group]['pt_drop']

        fg_df,statcast_df,id_map,fantrax = etl.load_raw_dataframes(self)
        master_df = pp.join_tables(self,fg_df,statcast_df,id_map,fantrax)
        master_df = pp.preliminary_col_reduction(self,master_df)
        master_df = pp.feature_engineering(self,master_df)
        master_df = pp.drop_out_of_position(self,master_df = master_df)
        master_df = pp.limit_years(self,start_year=start_year,master_df=master_df)
        master_df = pp.make_targets(self,master_df=master_df)
        master_df = pp.organize_training_columns(self,master_df=master_df)
        master_df = master_df.reset_index(drop = True)
        master_df.to_csv('data/processed/'+self.position_group+'/master_df.csv',index=True)
        modeling_df = pp.remove_injured(self,master_df=master_df)
        modeling_df = pp.remove_missing(self,modeling_df=modeling_df)
        modeling_df.to_csv('data/processed/'+self.position_group+'/modeling_df.csv',index=True)
        pp.run_corr_analysis(self,modeling_df)

        return master_df, modeling_df


    def modeling_pipeline(self, target, knn = 5,test_size = .3, max_evals = 100, seed = 8,verbose=False):

        modeling_df =  pd.read_csv('data/processed/'+self.position_group+'/modeling_df.csv',index_col=0)   

        self.seed = seed
        self.knn = knn

        model_df = m.isolate_relevant_columns(self,modeling_df = modeling_df,target = target)

        self.X_train, self.X_test, self.y_train, self.y_test = m.evaluation_split(self,model_df=model_df,target=target,test_size=test_size)
        self.X_train_prod, self.X_test_prod, self.y_train_prod, self.y_test_prod = m.production_split(self,model_df=model_df,target=target,test_size=test_size)
        self.X_train, self.X_test = m.preprocessing_pipeline(self, X_train = self.X_train, X_test = self.X_test, target = target, prod=False)
        self.model_eval = m.build_model(self,X_train=self.X_train, X_test=self.X_test,y_train=self.y_train, y_test=self.y_test,target=target,prod=False,max_evals=max_evals,verbose=verbose)
        self.X_train_prod, self.X_test_prod = m.preprocessing_pipeline(self, X_train = self.X_train_prod, X_test = self.X_test_prod, target = target, prod=True)
        self.model_prod = m.build_model(self,X_train=self.X_train, X_test=self.X_test,y_train=self.y_train, y_test=self.y_test,target=target,prod=True,max_evals=max_evals,verbose=verbose)


    def prod_scoring_pipeline(self):
        fg_df,statcast_df,id_map,fantrax = etl.load_raw_dataframes(self)
        scoring_df = pp.create_scoring_data(self,fg_df,statcast_df,id_map,fantrax)
        return scoring_df


class BobaProjections(BobaModeling):

    def __init__(self, year, position_group):
        b_H = BobaModeling(year=year,position_group='hitters')
        b_SP = BobaModeling(year=year,position_group='SP')
        b_RP = BobaModeling(year=year,position_group='RP')

    def create_league(self):
        print("TBD")

    def set_model_weights(self):
        print("TBD")

    def generate_projections(self):
        print("TBD")

    def system_comparison(self):
        print("TBD")



    # def load_training_data():
    #     return data

    # def load_modeling_data():
    #     return data

    # def load_scoring_data():
    #     return data
    
    # def load_projections():
    #     return data

    # def build_projections():
    #     return df

    # def perform_methods():
    #     method_A()
    #     method_B()
    #     return df

    # def compare_methods():
    #     return df, plots






















