import pandas as pd
import yaml
from tqdm import tqdm


def load_ID_map(season):
    id_map_1 = pd.read_csv('data/utils/'+str(season)+'/base_id_map.csv')
    id_map_2 = pd.read_csv('data/utils/id_map_update.csv')
    col_1 = ['MLBID','PLAYERNAME','BIRTHDATE','BATS','THROWS','IDFANGRAPHS','RETROID','BREFID','ESPNID','YAHOOID','FANTRAXNAME']
    col_2 = ['mlb_id','mlb_name','bats','throws','fg_id','bref_id','espn_id','retro_id','yahoo_id']
    id_map_1 = id_map_1[col_1]
    id_map_2 = id_map_2[col_2]
    id_map_2 = id_map_2.rename(columns = {'mlb_id':'MLBID', 
                                        'mlb_name': 'PLAYERNAME',
                                        'bats':'BATS',
                                        'throws':'THROWS',
                                        'fg_id':'IDFANGRAPHS',
                                        'bref_id':'BREFID',
                                        'espn_id':'ESPNID',
                                        'retro_id':'RETROID',
                                        'yahoo_id':'YAHOOID'
                                        })
    id_map = pd.concat([id_map_1,id_map_2])
    id_map = id_map.drop_duplicates(subset=['MLBID'])
    id_map = id_map[['IDFANGRAPHS','MLBID','FANTRAXNAME']]
    return id_map

def load_raw_dataframes(position_group,season):
    data_group = 'hitters' if position_group == 'hitters' else 'pitchers'
    fg_df= pd.read_csv('data/raw/'+data_group+'/fangraphs/season/master.csv')
    statcast_df = pd.read_csv('data/raw/'+data_group+'/statcast/season/master.csv')
    proj_sys_df = pd.read_csv('data/raw/'+data_group+'/projection_systems/master.csv',index_col=0)
    if position_group=='hitters':
        fg_df['position'] = fg_df.apply(agg_position_h, axis=1)
    else: 
        fg_df['position'] = 'P'

    fg_df['playerID'] = fg_df['playerID'].astype('str')
    statcast_df = statcast_df.drop(['qualify_pa','qualify_ip'],axis=1, errors='ignore')
    return fg_df,statcast_df, proj_sys_df


def join_tables(fg_df, statcast_df, id_map, fantrax, position_group):
    statcast = pd.merge(statcast_df,id_map,how='left',left_on='player_id',right_on='MLBID')
    master_df = pd.merge(fg_df,statcast,how='left',left_on=['playerID','Season'],right_on=['IDFANGRAPHS','year'])
    columns = ['playerID'] + [c for c in list(master_df.columns) if c not in ['playerID','IDFANGRAPHS','MLBID']]
    master_df = master_df[columns]
    if position_group == 'hitters':
        pass
    else:
        master_df = master_df.rename(columns={"K/9": "K_per_9", "BB/9": "BB_per_9","slg": "SLG","obp": "OBP"})
    return master_df 

def preliminary_col_reduction(master_df,position_group):
    with open(r'boba/recipes/initial_columns_2021.yaml') as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
    if position_group == 'hitters':
        column_list = yaml_data['hitters']['column_list']
    else: 
        column_list = yaml_data['pitchers']['column_list']
    master_df = master_df[column_list]
    return master_df

def gen_career_stats(master_df, position_group, pt_metric):
    career_df = master_df.copy()
    if position_group == 'hitters':
        sum_cols = ['PA','G','H','HR','WAR','R','RBI']
        w_avg_cols = ['AVG','OBP','SLG','OPS','BB%','K%','BABIP','LD%','GB%','FB%','HR/FB','IFFB%','wOBA' \
                ,'O-Swing%','Z-Swing%','Swing%','O-Contact%','Z-Contact%','Contact%','SwStr%' \
                ,'Pull%','Soft%','Med%','Hard%','barrel_batted_rate','launch_angle_avg','exit_velocity_avg'\
                , 'sweet_spot_percent','hard_hit_percent','xba','xwoba','xbacon','xobp','xslg','xwobacon','wobacon']
    else: 
        sum_cols = ['IP','G','W','ShO','SV','WAR','BS','HLD']
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
        career_working['_data_times_weight'] = career_working[i]*career_working[pt_metric]
        career_working['_weight_where_notnull'] = career_working[pt_metric]*pd.notnull(career_working[i])
        g = career_working.groupby('playerID')
        result = pd.DataFrame(g['_data_times_weight'].cumsum() / g['_weight_where_notnull'].cumsum(),columns=['career_'+i])
        career_df = pd.merge(career_df,result,left_index=True, right_index=True)
    career_df = career_df[['playerID','Season','Name']+['career_'+c for c in sum_cols]+['career_'+c for c in w_avg_cols]]
    career_df['career_asof_season'] = career_df['Season']+1
    return career_df


def feature_engineering(master_df, career_stats_df, position_group, pt_metric):
    mlb_tenure_df = pd.DataFrame(master_df.groupby(['playerID','Season']).count().groupby('playerID').cumcount(),columns=['MLB_tenure']).reset_index()

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

    with open(r'boba/recipes/initial_columns_2021.yaml') as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
    if position_group == 'hitters':
        cols = yaml_data['hitters']['column_list']
    else: 
        cols = yaml_data['pitchers']['column_list']
    col_list = [e for e in cols if e not in ['Season', 'playerID','Name','position','Team','Age','pitch_hand']]

    for col in tqdm(col_list):
        df[col+'_w3yr'] = (df[col+'_1yrago'].fillna(0)*df[pt_metric+'_1yrago'].fillna(0) \
                            +df[col+'_2yrago'].fillna(0)*df[pt_metric+'_2yrago'].fillna(0) \
                            +df[col+'_3yrago'].fillna(0)*df[pt_metric+'_3yrago'].fillna(0)) \
                            /(df[pt_metric+'_1yrago'].fillna(0) \
                            +df[pt_metric+'_2yrago'].fillna(0) \
                            +df[pt_metric+'_3yrago'].fillna(0)) 
        df[col+'_1yrDiff'] = df[col+'_1yrago'] - df[col+'_2yrago']
        # df[col+'_1yrDelta'] = (df[col+'_1yrago']/df[col+'_2yrago'])-1
    return df

def drop_out_of_position(master_df,position_group):
    if position_group == 'hitters':
        filter_cond = ~master_df['position'].str.contains('P')
        master_df = master_df[filter_cond]
    else:
        pass
    return master_df

def limit_years(master_df,start_year=2015):
    master_df = master_df[(master_df['Season'] >= start_year)]
    return master_df

# def remove(self,master_df):
#         high_PT_general = (master_df[self.pt_metric] > self.pt_min)
#         low_PT_for_delta = (master_df[self.pt_metric]<self.pt_keep_thres)
#         Big_drop_YoY = ((master_df[self.pt_metric+'_1yrago']-master_df[self.pt_metric])>self.pt_drop)
#         modeling_df = master_df[~((low_PT_for_delta) & (Big_drop_YoY))]
#         modeling_df = modeling_df[high_PT_general]
#         return modeling_df

def split_pitchers(master_df):
    season_avg = master_df.groupby(['Season']).agg({'IP':'mean','GS':'mean', 'SV':'mean','HLD':'mean'})
    df = master_df.join(season_avg, on='Season', rsuffix='_seasonAVG')
    df['position'] = df.apply(agg_position_p, axis=1)
    df_sp = df[df['position'] == 'SP']
    df_rp = df[df['position'] == 'RP']
    return df_sp, df_rp

def make_targets(master_df, counting_stats, per_metric):
    for stat in counting_stats:
        stat_per = master_df[stat]/master_df[per_metric]
        master_df.insert(master_df.columns.get_loc(stat),str(stat)+'_per_'+per_metric,stat_per)
    return master_df

def organize_training_columns(master_df,position_group):
    model_data_cols = list(master_df.columns[master_df.columns.get_loc('Team_1yrago'):])
    information_cols = ['Season','Name','playerID','position','Team','Age']
    
    per_metric_h = 'PA' 
    pt_metric_h = 'PA'      
    counting_stats_h = ['HR','R','RBI','WAR','SB','CS']
    rate_stats_h = ['AVG','OBP','SLG','BABIP','BB%','K%','wOBA']
    model_targets_h = rate_stats_h + [c+'_per_'+per_metric_h for c in counting_stats_h]  
   
    per_metric_sp = 'GS' 
    pt_metric_sp = 'IP'      
    counting_stats_sp = ['ShO','CG','W','WAR']
    rate_stats_sp = ['ERA','BB_per_9','K_per_9','OBP','SLG']
    model_targets_sp = rate_stats_sp + [c+'_per_'+per_metric_sp for c in counting_stats_sp]  
   
    per_metric_rp = 'G' 
    pt_metric_rp = 'IP'      
    counting_stats_rp = ['SV','HLD','WAR']
    rate_stats_rp = ['ERA','BB_per_9','K_per_9','OBP','SLG']
    model_targets_rp = rate_stats_rp + [c+'_per_'+per_metric_rp for c in counting_stats_rp]  
   
    if position_group == 'hitters':
        model_cols = information_cols + model_targets_h + [pt_metric_h] + counting_stats_h + model_data_cols
    elif position_group == 'SP':
        model_cols = information_cols + model_targets_sp + [pt_metric_sp] + [per_metric_sp] + counting_stats_sp + model_data_cols
    elif position_group == 'RP':
        model_cols = information_cols  + model_targets_rp + [pt_metric_rp] + [per_metric_rp] + counting_stats_rp + model_data_cols
    else:
        raise
    df = master_df[model_cols]
    return df

def remove_missing(master_df, position_group):
    if position_group == 'RP':
        master_df = master_df[~master_df['SLG'].isna()]
        master_df = master_df[~master_df['OBP'].isna()]
        return master_df
    elif position_group == 'SP':
        master_df = master_df[~master_df['SLG'].isna()]
        master_df = master_df[~master_df['OBP'].isna()]
        return master_df
    else:
        return master_df




def agg_position_p(row):
    if (row['GS']>=round(row['GS_seasonAVG'],0)) & ((row['SV']+row['HLD'])<2*(row['SV_seasonAVG']+row['HLD_seasonAVG'])):
        return 'SP'
    elif ((row['HLD']+row['SV'])<=4) & (row['GS']>0) & ((row['IP']/row['G'])>2):
        return 'SP' 
    else:
        return 'RP'

def agg_position_h(row):
    if 'C' in row['position'] :
        return 'C'
    elif '2B' in row['position']:
        return '2B' 
    elif 'SS' in row['position']:
        return 'SS' 
    elif 'OF' in row['position']:
        return 'OF' 
    elif '1B' in row['position']:
        return '1B' 
    elif '3B' in row['position']:
        return '3B' 
    elif 'RP' in row['position']:
        return 'P' 
    elif 'SP' in row['position']:
        return 'P' 
        
    else:
        return row['position']
