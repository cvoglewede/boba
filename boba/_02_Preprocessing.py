import pandas as pd
from tqdm import tqdm
import yaml

from .utils import Boba_Utils as u

# warnings.filterwarnings("ignore")

class Boba_Preprocessing(u):

    def __init__(self):
        pass

    def join_tables(self,fg_df, statcast_df, id_map, fantrax):
        statcast = pd.merge(statcast_df,id_map,how='left',left_on='player_id',right_on='MLBID')
        master_df = pd.merge(fg_df,statcast,how='left',left_on=['playerID','Season'],right_on=['IDFANGRAPHS','year'])
        columns = ['playerID'] + [c for c in list(master_df.columns) if c not in ['playerID','IDFANGRAPHS','MLBID']]
        master_df = master_df[columns]
        if self.position_group == 'hitters':
            pass
        else:
           master_df = master_df.rename(columns={"K/9": "K_per_9", "BB/9": "BB_per_9","slg": "SLG","obp": "OBP"})
        return master_df 

    def preliminary_col_reduction(self,master_df):
        with open(r'boba/recipes/initial_columns.yaml') as file:
            yaml_data = yaml.load(file, Loader=yaml.FullLoader)
        if self.position_group == 'hitters':
            column_list = yaml_data['hitters']['column_list']
        else: 
            column_list = yaml_data['pitchers']['column_list']
        master_df = master_df[column_list]
        return master_df

    def gen_career_stats(self,master_df):
        career_df = master_df.copy()
        if self.position_group == 'hitters':
            sum_cols = ['PA','G','H','1B','HR','WAR','R','RBI']
            w_avg_cols = ['AVG','OBP','SLG','OPS','BB%','K%','BABIP','LD%','GB%','FB%','HR/FB','IFFB%','wOBA' \
                    ,'O-Swing%','Z-Swing%','Swing%','O-Contact%','Z-Contact%','Contact%','SwStr%' \
                    ,'Pull%','Soft%','Med%','Hard%','TTO%','barrel_batted_rate','launch_angle_avg','exit_velocity_avg'\
                    , 'sweet_spot_percent','hard_hit_percent','xba','xwoba','xbacon','xobp','xslg','xwobacon','wobacon']
        else: 
            sum_cols = ['IP','G','W','ShO','SV','WAR','BS','HR','SO','HLD']
            w_avg_cols = ['ERA','K_per_9','BB_per_9','K/BB','AVG','WHIP','BABIP','LOB%','FIP','GB/FB','LD%','GB%','FB%','IFFB%','HR/FB','tERA','xFIP'
              ,'O-Swing%','Z-Swing%','O-Contact%','Z-Contact%','Contact%','SwStr%','SIERA', 'OBP','SLG'
              ,'whiff_percent','meatball_percent','popups_percent','solidcontact_percent','barrel_batted_rate','launch_angle_avg'
              , 'exit_velocity_avg', 'sweet_spot_percent', 'xba', 'xbacon', 'xobp', 'xiso', 'xwoba', 'woba','xwobacon','xslg','on_base_plus_slg']

        for i in tqdm(sum_cols):
            career_working = master_df.copy()
            career_working = career_working.sort_values('Season')
            g = career_working.groupby('playerID')
            result = pd.DataFrame(g[i].cumsum()).rename(columns={i: 'career_'+i})
            career_df = pd.merge(career_df,result,left_index=True, right_index=True)
        for i in tqdm(w_avg_cols):
            career_working = master_df.copy()
            career_working = career_working.sort_values('Season')
            career_working['_data_times_weight'] = career_working[i]*career_working[self.pt_metric]
            career_working['_weight_where_notnull'] = career_working[self.pt_metric]*pd.notnull(career_working[i])
            g = career_working.groupby('playerID')
            result = pd.DataFrame(g['_data_times_weight'].cumsum() / g['_weight_where_notnull'].cumsum(),columns=['career_'+i])
            career_df = pd.merge(career_df,result,left_index=True, right_index=True)
        career_df = career_df[['playerID','Season','Name']+['career_'+c for c in sum_cols]+['career_'+c for c in w_avg_cols]]
        career_df['career_asof_season'] = career_df['Season']+1
        return career_df


    def feature_engineering(self,master_df):
        mlb_tenure_df = pd.DataFrame(master_df.groupby(['playerID','Season']).count().groupby('playerID').cumcount(),columns=['MLB_tenure']).reset_index()
        career_stats_df = self.gen_career_stats(master_df)

        master_df['Year-1'] = master_df['Season']-1
        df_1yr = pd.merge(master_df,master_df,how='left',left_on=['playerID','Year-1'],right_on=['playerID','Season'], suffixes=['','_1yrago'])
        df_1yr = df_1yr.drop(['Year-1','Year-1_1yrago','Season_1yrago','Name_1yrago','Age_1yrago','position_1yrago'],axis=1, errors='ignore')
        df_1yr['Year-2'] = df_1yr['Season']-2
        df_2yr = pd.merge(df_1yr,master_df,how='left',left_on=['playerID','Year-2'],right_on=['playerID','Season'], suffixes=['','_2yrago'])
        df_2yr = df_2yr.drop(['Year-2','Year-1','Season_2yrago','Name_2yrago','Age_2yrago','position_2yrago'],axis=1, errors='ignore')
        df_2yr['Year-3'] = df_2yr['Season']-3
        df_3yr = pd.merge(df_2yr,master_df,how='left',left_on=['playerID','Year-3'],right_on=['playerID','Season'], suffixes=['','_3yrago'])
        df_3yr = df_3yr.drop(['Year-3','Year-1','Season_3yrago','Name_3yrago','Age_3yrago','position_3yrago'],axis=1, errors='ignore')
        df = pd.merge(df_3yr,mlb_tenure_df,how='left', left_on=['playerID','Season'],right_on=['playerID','Season'],suffixes=['','_tenure'])
        df = pd.merge(df,career_stats_df,how='left', left_on=['playerID','Season'],right_on=['playerID','career_asof_season'],suffixes=['','_career'])
        df = df.drop(['Season_career','Name_career','pitch_hand_2yrago','pitch_hand_3yrago','pitch_hand_1yrago'],axis=1, errors='ignore')

        # df['Age_sqr'] = df['Age']**2

        with open(r'boba/recipes/initial_columns.yaml') as file:
            yaml_data = yaml.load(file, Loader=yaml.FullLoader)
        if self.position_group == 'hitters':
            cols = yaml_data['hitters']['column_list']
        else: 
            cols = yaml_data['pitchers']['column_list']
        col_list = [e for e in cols if e not in ['Season', 'playerID','Name','position','Team','Age','pitch_hand']]

        for col in tqdm(col_list):
            df[col+'_w3yr'] = (df[col+'_1yrago'].fillna(0)*df[self.pt_metric+'_1yrago'].fillna(0)+df[col+'_2yrago'].fillna(0)*df[self.pt_metric+'_2yrago'].fillna(0)+df[col+'_3yrago'].fillna(0)*df[self.pt_metric+'_3yrago'].fillna(0))/(df[self.pt_metric+'_1yrago'].fillna(0)+df[self.pt_metric+'_2yrago'].fillna(0)+df[self.pt_metric+'_3yrago'].fillna(0))
            df[col+'_1yrDiff'] = df[col+'_1yrago'] - df[col+'_2yrago']
            # df[col+'_1yrDelta'] = (df[col+'_1yrago']/df[col+'_2yrago'])-1

        return df

    
    def drop_out_of_position(self,master_df):
        if self.position_group == 'hitters':
            filter_cond = ~master_df['position'].str.contains('P')
            master_df = master_df[filter_cond]
        else:
            filter_cond = master_df['position']==self.position_group
            master_df = master_df[filter_cond]
        return master_df

    def limit_years(self,master_df,start_year=2015):
        master_df = master_df[(master_df['Season'] >= start_year)]
        return master_df


    def make_targets(self,master_df):
        for stat in self.counting_stats:
            stat_per = master_df[stat]/master_df[self.per_metric]
            master_df.insert(master_df.columns.get_loc(stat),str(stat)+'_per_'+self.per_metric,stat_per)
        return master_df

    def organize_training_columns(self,master_df):
        model_data_cols = list(master_df.columns[master_df.columns.get_loc('Team_1yrago'):])
        if self.position_group == 'hitters':
            model_cols = self.information_cols + self.model_targets + [self.pt_metric] + self.counting_stats + model_data_cols
        else: 
            model_cols = self.information_cols + ['pitch_hand'] + self.model_targets + [self.pt_metric] + [self.per_metric] + self.counting_stats + model_data_cols
        df = master_df[model_cols]
        return df

    def remove_injured(self,master_df):
        high_PT_general = (master_df[self.pt_metric] > self.pt_min)
        low_PT_for_delta = (master_df[self.pt_metric]<self.pt_keep_thres)
        Big_drop_YoY = ((master_df[self.pt_metric+'_1yrago']-master_df[self.pt_metric])>self.pt_drop)
        modeling_df = master_df[~((low_PT_for_delta) & (Big_drop_YoY))]
        modeling_df = modeling_df[high_PT_general]
        return modeling_df

    def remove_missing(self,modeling_df):
        if self.position_group == 'RP':
            modeling_df = modeling_df[~modeling_df['SLG'].isna()]
            modeling_df = modeling_df[~modeling_df['OBP'].isna()]
            return modeling_df
        elif self.position_group == 'SP':
            modeling_df = modeling_df[~modeling_df['SLG'].isna()]
            modeling_df = modeling_df[~modeling_df['OBP'].isna()]
            return modeling_df
        else:
            return modeling_df

    def run_corr_analysis(self,modeling_df):
        modeling_df = modeling_df[modeling_df['Season']<(self.year - 2)]
        test_dict = {}
        for target in self.model_targets:
            print(target)
            targets = self.model_targets.copy()
            targets.remove(target)
            trim_df = modeling_df.drop(targets+self.counting_stats+['Season','Name','playerID','position','Team','PA','IP','G','GS'],axis=1,errors='ignore')
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
        with open(r'boba/recipes/'+self.position_group+'_relevant_features.yaml', 'w') as file:
            yaml.dump(test_dict, file)


    def create_scoring_data(self,fg_df,statcast_df,id_map,fantrax):
        master_df = self.join_tables(fg_df,statcast_df,id_map,fantrax)
        master_df = self.preliminary_col_reduction(master_df)
        master_df = self.feature_engineering_scoring(master_df)
        master_df = self.drop_out_of_position(master_df = master_df)
        features_df =  master_df[(master_df['Season'] == self.year-1)]
        
        
        data_group = 'hitters' if self.position_group == 'hitters' else 'pitchers'
        zips = pd.read_csv('data/raw/'+data_group+'/projection_systems/zips/'+str(self.year)+'.csv')
        zips = zips.rename(columns={'playerid':'playerID'})
        if self.position_group == 'hitters':
            base_df = zips[['playerID','Name']]
        else: 
            base_df = zips[['playerID','Name','GS']]


        features_df['Season'] = self.year
        features_df['MLB_tenure'] = features_df['MLB_tenure']+1
        features_df['Age'] = features_df['Age']+1
        # features_df['Age_sqr'] = features_df['Age']**2
        features_df = features_df.reset_index()
        score_data_cols = list(features_df.columns[features_df.columns.get_loc('Team_1yrago'):])

        if self.position_group == 'hitters':
            score_cols = self.information_cols + score_data_cols
        else: 
            score_cols = self.information_cols + ['pitch_hand'] + score_data_cols

        # score_cols = self.information_cols + score_data_cols
        features_df = features_df[score_cols]

        scoring_df = pd.merge(base_df,features_df,how='left',left_on='playerID',right_on='playerID', suffixes=['_zips',''])
                
        if self.position_group=='hitters':
            pass
        else: 
            scoring_df['position'] = scoring_df.apply(u.agg_position_p_scoring, axis=1)
            scoring_df = self.drop_out_of_position(master_df = scoring_df)
        scoring_df = scoring_df.reset_index(drop=True)
        scoring_df = scoring_df.drop(['GS'],axis=1,errors='ignore')
        scoring_df.to_csv('data/scoring/scoring_raw_'+self.position_group+'.csv')


    def feature_engineering_scoring(self,master_df):
        mlb_tenure_df = pd.DataFrame(master_df.groupby(['playerID','Season']).count().groupby('playerID').cumcount(),columns=['MLB_tenure']).reset_index()
        career_stats_df = self.gen_career_stats(master_df)

        master_df['Year-1'] = master_df['Season']-1
        df_1yr = pd.merge(master_df,master_df,how='left',left_on=['playerID','Season'],right_on=['playerID','Season'], suffixes=['','_1yrago'])
        df_1yr = df_1yr.drop(['Year-1_1yrago','Season_1yrago','Name_1yrago','Age_1yrago','position_1yrago'],axis=1, errors='ignore')
        df_1yr['Year-2'] = df_1yr['Season']-2

        df_2yr = pd.merge(df_1yr,master_df,how='left',left_on=['playerID','Year-1'],right_on=['playerID','Season'], suffixes=['','_2yrago'])
        df_2yr = df_2yr.drop(['Year-1','Season_2yrago','Name_2yrago','Age_2yrago','position_2yrago'],axis=1, errors='ignore')
        df_2yr['Year-3'] = df_2yr['Season']-3

        df_3yr = pd.merge(df_2yr,master_df,how='left',left_on=['playerID','Year-2'],right_on=['playerID','Season'], suffixes=['','_3yrago'])
        df_3yr = df_3yr.drop(['Year-2','Year-1','Season_3yrago','Name_3yrago','Age_3yrago','position_3yrago'],axis=1, errors='ignore')
        
        df = pd.merge(df_3yr,mlb_tenure_df,how='left', left_on=['playerID','Season'],right_on=['playerID','Season'],suffixes=['','_tenure'])
        df = pd.merge(df,career_stats_df,how='left', left_on=['playerID','Season'],right_on=['playerID','career_asof_season'],suffixes=['','_career'])
        df = df.drop(['Season_career','Name_career','pitch_hand_2yrago','pitch_hand_3yrago','pitch_hand_1yrago'],axis=1, errors='ignore')

        with open(r'boba/recipes/initial_columns.yaml') as file:
            yaml_data = yaml.load(file, Loader=yaml.FullLoader)
        if self.position_group == 'hitters':
            cols = yaml_data['hitters']['column_list']
        else: 
            cols = yaml_data['pitchers']['column_list']
        col_list = [e for e in cols if e not in ['Season', 'playerID','Name','position','Team','Age','pitch_hand']]

        for col in tqdm(col_list):
            df[col+'_w3yr'] = (df[col+'_1yrago'].fillna(0)*df[self.pt_metric+'_1yrago'].fillna(0)+df[col+'_2yrago'].fillna(0)*df[self.pt_metric+'_2yrago'].fillna(0)+df[col+'_3yrago'].fillna(0)*df[self.pt_metric+'_3yrago'].fillna(0))/(df[self.pt_metric+'_1yrago'].fillna(0)+df[self.pt_metric+'_2yrago'].fillna(0)+df[self.pt_metric+'_3yrago'].fillna(0))
            df[col+'_1yrDiff'] = df[col+'_1yrago'] - df[col+'_2yrago']
            # df[col+'_1yrDelta'] = (df[col+'_1yrago']/df[col+'_2yrago'])-1

        return df





