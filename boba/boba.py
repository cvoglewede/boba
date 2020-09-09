
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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


    def modeling_pipeline(self, target, knn = 5,test_size = .3, max_evals = 100, seed = 8,verbose=False, split_method='timeseries'):

        modeling_df =  pd.read_csv('data/processed/'+self.position_group+'/modeling_df.csv',index_col=0)   

        self.seed = seed
        self.knn = knn

        model_df = m.isolate_relevant_columns(self,modeling_df = modeling_df,target = target)

        self.X_train, self.X_test, self.y_train, self.y_test = m.evaluation_split(self,model_df=model_df,target=target,test_size=test_size,method=split_method)
        self.X_train_prod, self.X_test_prod, self.y_train_prod, self.y_test_prod = m.production_split(self,model_df=model_df,target=target,test_size=test_size,method=split_method)
        self.X_train, self.X_test = m.preprocessing_pipeline(self, X_train = self.X_train, X_test = self.X_test, target = target, prod=False)
        self.model_eval = m.build_model(self,X_train=self.X_train, X_test=self.X_test,y_train=self.y_train, y_test=self.y_test,target=target,prod=False,max_evals=max_evals,verbose=verbose)
        self.X_train_prod, self.X_test_prod = m.preprocessing_pipeline(self, X_train = self.X_train_prod, X_test = self.X_test_prod, target = target, prod=True)
        self.model_prod = m.build_model(self,X_train=self.X_train, X_test=self.X_test,y_train=self.y_train, y_test=self.y_test,target=target,prod=True,max_evals=max_evals,verbose=verbose)


    def prod_scoring_pipeline(self):
        try:
            os.remove('data/projections/'+self.position_group+'_'+str(self.year)+'_raw.csv')
        except: 
            pass
        fg_df,statcast_df,id_map,fantrax = etl.load_raw_dataframes(self)
        pp.create_scoring_data(self,fg_df,statcast_df,id_map,fantrax)
        for target in tqdm(self.model_targets):
            m.generate_prod_predictions(self, target=target)
        scored_df = pd.read_csv('data/projections/'+self.position_group+'_'+str(self.year)+'_raw.csv',index_col=0)
        # scored_df = scored_df.drop(['Name_zips'],axis=1,errors='ignore')
        scored_df = self.clean_projections(scored_df)
        scored_df.to_csv('data/projections/'+self.position_group+'_'+str(self.year)+'_raw.csv')
        return scored_df

    def clean_projections(self,scored_df):
        systems = ['Boba','zips','stmr','atc','bat']
        if self.position_group == 'hitters':
            for i in tqdm(systems):
                scored_df['NetSB_'+i] = scored_df['SB_'+i] - scored_df['CS_'+i]
        elif self.position_group == 'SP':
            for i in tqdm(systems):
                scored_df['QS_'+i] = ((scored_df['IP_zips']/(12*6.15)-0.11*scored_df['ERA_'+i]))*scored_df['GS_zips']
                # scored_df['QS_'+i] = (scored_df['GS_zips']*(.465-(scored_df['ERA_'+i]*.0872381))+(scored_df['IP_zips']/scored_df['GS_zips'])*.0746775)
                scored_df['QSCGSHO_'+i] = scored_df['QS_'+i]+scored_df['CG_Boba']+scored_df['ShO_Boba']
            scored_df['OPS_Boba'] = scored_df['SLG_Boba']+scored_df['OBP_Boba']
        elif self.position_group == 'RP':
            scored_df['SVHLD_Boba'] = scored_df['SV_Boba']+scored_df['HLD_Boba']
            scored_df['SVHLD_atc'] = scored_df['SV_atc']+scored_df['HLD_Boba']
            scored_df['SVHLD_stmr'] = scored_df['SV_stmr']+scored_df['HLD_Boba']
            scored_df['SVHLD_bat'] = np.nan
            scored_df['SVHLD_zips'] = np.nan
            scored_df['OPS_Boba'] = scored_df['SLG_Boba']+scored_df['OBP_Boba']
        else:
            pass
        return scored_df



class BobaProjections(BobaModeling):

    def __init__(self, year, remove_list):
        self.remove_list = remove_list
        self.b_H = BobaModeling(year=year,position_group='hitters')
        self.b_SP = BobaModeling(year=year,position_group='SP')
        self.b_RP = BobaModeling(year=year,position_group='RP')

        self.proj_H = pd.read_csv('data/projections/hitters_2020_raw.csv',index_col=0)
        self.proj_SP = pd.read_csv('data/projections/SP_2020_raw.csv',index_col=0)
        self.proj_RP = pd.read_csv('data/projections/RP_2020_raw.csv',index_col=0)

        self.stat_cat_H_default = ['OBP','SLG','HR','R','RBI','NetSB']
        self.stat_cat_P_default = ['ERA','OPS','K_per_9','BB_per_9','QSCGSHO','SVHLD']

        self.display_cols_H = self.stat_cat_H_default + ['OPS_H','BB%_Boba','K%_Boba']
        self.display_cols_P = self.stat_cat_P_default.copy()

    def remove_injured_and_optouts(self, df, position_group):
        if position_group == 'hitters':
            for i in self.remove_list:
                if i in list(df['Name']):
                    df.loc[df['Name'] == i, 'PA_zips'] = 0
                    df.loc[df['Name'] == i, 'HR'] = 0
                    df.loc[df['Name'] == i, 'RBI'] = 0
                    df.loc[df['Name'] == i, 'R'] = 0
        elif position_group == 'pitchers':
            for i in self.remove_list:
                if i in list(df['Name']):
                    df.loc[df['Name'] == i, 'IP_zips'] = 0
                    df.loc[df['Name'] == i, 'GS_zips'] = 0
                    df.loc[df['Name'] == i, 'QSCGSHO'] = 0
                    df.loc[df['Name'] == i, 'G_zips'] = 0
                    df.loc[df['Name'] == i, 'SVHLD'] = 0
                    df.loc[df['Name'] == i, 'K_per_9'] = 0
                    df.loc[df['Name'] == i, 'ERA'] = 5
        else: 
            pass
        return df

    def create_league(self, stat_categories_H, stat_categories_P, n_teams=12, catcher=1, first=1, second=1, third=1, ss=1, outfield=3, utility=2, off_bench = 5, startingP = 7, reliefP = 6):

        self.stat_categories_H = stat_categories_H
        self.stat_categories_P = stat_categories_P
        self.n_teams = n_teams
        self.tot_catcher = n_teams*catcher
        self.tot_first = n_teams*first
        self.tot_second = n_teams*second
        self.tot_third = n_teams*third
        self.tot_ss = n_teams*ss
        self.tot_outfield = n_teams*outfield
        self.tot_utility = n_teams*utility
        self.tot_off_bench = n_teams*off_bench

        self.total_offense = self.tot_catcher + self.tot_first + self.tot_second + self.tot_third + self.tot_ss + self.tot_outfield + self.tot_utility
        
        self.tot_startingP = n_teams*startingP
        self.tot_reliefP = n_teams*reliefP

        self.total_roster = self.total_offense + self.tot_startingP + self.tot_reliefP


    def set_model_weights(self, trust= True, weights=None):
        if trust:
            self.w_H_Boba = .8
            self.w_H_zips = .05
            self.w_H_stmr = .05
            self.w_H_atc = .05
            self.w_H_bat = .05

            self.w_SP_Boba = .8
            self.w_SP_zips = .05
            self.w_SP_stmr = .05
            self.w_SP_atc = .05
            self.w_SP_bat = .05

            self.w_RP_Boba = .8
            self.w_RP_zips = .05
            self.w_RP_stmr = .05
            self.w_RP_atc = .05
            self.w_RP_bat = .05
        else:
            self.w_H_Boba = weights['hitters']['Boba']
            self.w_H_zips = weights['hitters']['zips']
            self.w_H_stmr = weights['hitters']['steamer']
            self.w_H_atc = weights['hitters']['atc']
            self.w_H_bat = weights['hitters']['bat']

            self.w_SP_Boba = weights['SP']['Boba']
            self.w_SP_zips = weights['SP']['zips']
            self.w_SP_stmr = weights['SP']['steamer']
            self.w_SP_atc = weights['SP']['atc']
            self.w_SP_bat = weights['SP']['bat']

            self.w_RP_Boba = weights['RP']['Boba']
            self.w_RP_zips = weights['RP']['zips']
            self.w_RP_stmr = weights['RP']['steamer']
            self.w_RP_atc = weights['RP']['atc']
            self.w_RP_bat = weights['RP']['bat']

    def set_custom_parameters(self, value_NetSB = True, versatility_premium = 0, utility_discount = 0, adjust_PA = False, adjust_PA_exceptC = False):
        self.value_NetSB = value_NetSB
        self.versatility_premium = versatility_premium
        self.utility_discount = utility_discount
        
        if adjust_PA:
            self.PA_adj = 630  
        else:
            self.PA_adj = 0

        if adjust_PA_exceptC:
            self.PA_adj_nonC = 630  
        else:
            self.PA_adj_nonC = 0


    def generate_projections(self):
        print("TBD")



    def system_comparison(self):
        print("TBD")


    def compile_hitters(self):
        fantrax = pd.read_csv('data/utils/fantrax.csv')
        fantrax = fantrax[['Player','Position','ADP']]
        id_map = self.load_ID_map()
        join_df = pd.merge(id_map,fantrax, how='left',left_on='FANTRAXNAME', right_on='Player' )
        join_df = join_df[['IDFANGRAPHS','Position','ADP']]
        temp_df = pd.merge(self.proj_H,join_df, how='left',left_on='playerID',right_on='IDFANGRAPHS').drop(['IDFANGRAPHS'],axis=1).rename(columns={'Position':'Fantrax_position'})




    def compile_pitchers(self):
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






















