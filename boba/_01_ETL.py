import pandas as pd
import logging
import boto3
from botocore.exceptions import ClientError
import numpy as np
from bs4 import BeautifulSoup
import requests
import io
import re
from lxml import html 
from tqdm import tqdm
import os
import json
import warnings

from .utils import Boba_Utils as u

warnings.filterwarnings("ignore")

class Boba_ETL(u):

    def __init__(self):
        pass
        
    
    def load_raw_dataframes(self):
        data_group = 'hitters' if self.position_group == 'hitters' else 'pitchers'
        fg_df= pd.read_csv('data/aggregated/'+data_group+'/fangraphs/master.csv')
        statcast_df = pd.read_csv('data/aggregated/'+data_group+'/statcast/master.csv')
        id_map = pd.read_csv('data/utils/id_map.csv')
        id_map = id_map[['IDFANGRAPHS','MLBID']]
        fantrax = pd.read_csv('data/utils/fantrax.csv')
        if self.position_group=='hitters':
            fg_df['position'] = fg_df.apply(u.agg_position_h, axis=1)
        else: 
            fg_df['position'] = fg_df.apply(u.agg_position_p, axis=1)

        fg_df['playerID'] = fg_df['playerID'].astype('str')
        statcast_df = statcast_df.drop(['qualify_pa','qualify_ip'],axis=1, errors='ignore')
        return fg_df,statcast_df,id_map,fantrax


    def gather_source_masters(self,position_group, source, writeS3 = False):
        path = 'data/raw/'+position_group+'/'+source+'/season/'
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                if '.csv' in file:
                    files.append(os.path.join(r, file))
        for f in files:
            print(f)    
        data = pd.DataFrame()
        for file in tqdm(files):
            df = pd.read_csv(file)
            headings = df.columns
            data = data.append(df)
        data.columns = headings
        aggregated_path = 'data/aggregated/'+position_group+'/'+source+'/master.csv'
        data.to_csv(aggregated_path, index = False)
        if writeS3 == True:
            u.upload_file(self,file_name = aggregated_path)
        else:
            pass


    def FG_hitters_season(self, season, league = 'all', qual = 0, ind = 1, writeS3 = False):
        raw_path = 'data/raw/hitters/fangraphs/season/'+str(season)+'.csv'
        if os.path.exists(raw_path):
            print("file already exists")
            pass
        else:
            league = league
            qual = qual
            ind = ind
            url =  'http://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg={}&qual={}&type=c,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,-1&season={}&month=0&season1={}&ind={}&team=&rost=&age=&filter=&players=&page=1_100000'
            url = url.format(league, qual, season, season, ind)
            s = requests.get(url).content
            soup = BeautifulSoup(s, "lxml")
            table = soup.find('table', {'class': 'rgMasterTable'})
            data = []
            headings = [row.text.strip() for row in table.find_all('th')[1:]]+['playerID']+['position']
            FBperc_indices = [i for i,j in enumerate(headings) if j=='FB%']
            headings[FBperc_indices[1]]='FB% (Pitch)'
            table_body = table.find('tbody')
            rows = table_body.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                cols = [ele.text.strip() for ele in cols]
                s = row.find('a')['href']
                playerid = re.search('playerid=(.*)&', s)
                cols.append(playerid.group(1))
                position = re.search('position=(.*)', s)
                cols.append(position.group(1))
                data.append([ele for ele in cols[1:]])

            data = pd.DataFrame(data=data, columns=headings)
            data.replace(r'^\s*$', np.nan, regex=True, inplace = True)
            percentages = ['Zone% (pi)','Contact% (pi)','Z-Contact% (pi)','O-Contact% (pi)','Swing% (pi)','Z-Swing% (pi)','O-Swing% (pi)','XX% (pi)','SL% (pi)','SI% (pi)','SB% (pi)','KN% (pi)','FS% (pi)','FC% (pi)','FA% (pi)','CU% (pi)','CS% (pi)','CH% (pi)','TTO%','Hard%','Med%','Soft%','Oppo%','Cent%','Pull%','Zone% (pfx)','Contact% (pfx)','Z-Contact% (pfx)','O-Contact% (pfx)','Swing% (pfx)','Z-Swing% (pfx)','O-Swing% (pfx)','UN% (pfx)','KN% (pfx)','SC% (pfx)','CH% (pfx)','EP% (pfx)','KC% (pfx)','CU% (pfx)','SL% (pfx)','SI% (pfx)','FO% (pfx)','FS% (pfx)','FC% (pfx)','FT% (pfx)','FA% (pfx)','SwStr%','F-Strike%','Zone%','Contact%','Z-Contact%','O-Contact%','Swing%','Z-Swing%','O-Swing%','PO%','XX%','KN%','SF%','CH%','CB%','CT%','SL%','FB%','BUH%','IFH%','HR/FB','IFFB%','FB% (Pitch)','GB%', 'LD%','GB/FB','K%','BB%']
            for col in percentages:
                if not data[col].empty:
                    if pd.api.types.is_string_dtype(data[col]):
                        data[col] = data[col].str.strip(' %')
                        data[col] = data[col].str.strip('%')
                        data[col] = data[col].astype(float)/100.
                else:
                    pass
            cols_to_numeric = [col for col in data.columns if col not in ['Name', 'Team', 'Age Rng', 'Dol','playerID','position']]
            data[cols_to_numeric] = data[cols_to_numeric].astype(float)
            data = data.sort_values(['WAR', 'OPS'], ascending=False)
            data.to_csv(raw_path, index = False)
            if writeS3 == True:
                u.upload_file(self,file_name = raw_path)
            else:
                pass      
            
    def FG_pitchers_season(self, season, league = 'all', qual = 0, ind = 1, writeS3 = False):
        raw_path = 'data/raw/pitchers/fangraphs/season/'+str(season)+'.csv'
        if os.path.exists(raw_path):
            print("file already exists")
            pass
        else:
            league = league
            qual = qual
            ind = ind
            url =  'http://www.fangraphs.com/leaders.aspx?pos=all&stats=pit&lg={}&qual={}&type=c,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,-1&season={}&month=0&season1={}&ind={}&team=&rost=&age=&filter=&players=&page=1_100000'
            url = url.format(league, qual, season, season, ind)
            s = requests.get(url).content
            soup = BeautifulSoup(s, "lxml")
            table = soup.find('table', {'class': 'rgMasterTable'})
            data = []
            headings = [row.text.strip() for row in table.find_all('th')[1:]]+['playerID']+['position']
            FBperc_indices = [i for i,j in enumerate(headings) if j=='FB%']
            headings[FBperc_indices[1]]='FB% (Pitch)'
            table_body = table.find('tbody')
            rows = table_body.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                cols = [ele.text.strip() for ele in cols]
                s = row.find('a')['href']
                playerid = re.search('playerid=(.*)&', s)
                cols.append(playerid.group(1))
                position = re.search('position=(.*)', s)
                cols.append(position.group(1))
                data.append([ele for ele in cols[1:]])
            
            data = pd.DataFrame(data=data, columns=headings)
            data = data.drop(['Dollars'],axis=1)
            data.replace(r'^\s*$', np.nan, regex=True, inplace = True)
            percentages = ['FB% (Pitch)','Contact% (pi)', 'Zone% (pi)','Z-Contact% (pi)','O-Contact% (pi)','Swing% (pi)','Z-Swing% (pi)','O-Swing% (pi)','SL% (pi)','SI% (pi)','SB% (pi)','KN% (pi)','FS% (pi)','FC% (pi)','FA% (pi)','CU% (pi)','CS% (pi)','CH% (pi)','TTO%','Hard%','Med%','Soft%','Oppo%','Cent%','Pull%','K-BB%','Zone% (pfx)','Contact% (pfx)','Z-Contact% (pfx)','O-Contact% (pfx)','Swing% (pfx)','Z-Swing% (pfx)','O-Swing% (pfx)','UN% (pfx)','KN% (pfx)','SC% (pfx)','CH% (pfx)','EP% (pfx)','KC% (pfx)','CU% (pfx)','SL% (pfx)','SI% (pfx)','FO% (pfx)','FS% (pfx)','FC% (pfx)','FT% (pfx)','FA% (pfx)','BB%','K%','SwStr%','F-Strike%','Zone%','Contact%','Z-Contact%','O-Contact%','Swing%','Z-Swing%','O-Swing%','XX%','KN%','SF%','CH%','CB%','CT%','SL%','FB%','BUH%','IFH%','HR/FB','IFFB%','GB%','LD%','LOB%', 'XX% (pi)', 'PO%']
            for col in percentages:
                if not data[col].empty:
                    if pd.api.types.is_string_dtype(data[col]):
                        data[col] = data[col].astype(str).str.strip(' %')
                        data[col] = data[col].astype(str).str.strip('%')
                        data[col] = data[col].astype(float)/100.
                else:
                    pass
            cols_to_numeric = [col for col in data.columns if col not in ['Name', 'Team', 'Age Rng','playerID','position']]
            data[cols_to_numeric] = data[cols_to_numeric].astype(float)
            data = data.sort_values(['WAR', 'W'], ascending=False)
            data.to_csv(raw_path, index = False)
            if writeS3 == True:
                u.upload_file(self,file_name = raw_path)
            else:
                pass

    def statcast_hitters_season(self, season, writeS3 = False):
        raw_path = 'data/raw/hitters/statcast/season/'+str(season)+'.csv'
        if os.path.exists(raw_path):
            print("file already exists")
            pass
        else:
            url = 'https://baseballsavant.mlb.com/leaderboard/custom?year={}&type=batter&filter=&sort=7&sortDir=desc&min=10&selections=xba,xslg,xwoba,xobp,xiso,wobacon,xwobacon,bacon,xbacon,xbadiff,xslgdiff,wobadif,exit_velocity_avg,launch_angle_avg,sweet_spot_percent,barrels,barrel_batted_rate,solidcontact_percent,flareburner_percent,poorlyunder_percent,poorlytopped_percent,poorlyweak_percent,hard_hit_percent,z_swing_percent,z_swing_miss_percent,oz_swing_percent,oz_swing_miss_percent,oz_contact_percent,out_zone_swing_miss,out_zone_swing,out_zone_percent,out_zone,meatball_swing_percent,meatball_percent,pitch_count_offspeed,pitch_count_fastball,pitch_count_breaking,pitch_count,iz_contact_percent,in_zone_swing_miss,in_zone_swing,in_zone_percent,in_zone,edge_percent,edge,whiff_percent,swing_percent,pull_percent,straightaway_percent,opposite_percent,batted_ball,f_strike_percent,groundballs_percent,groundballs,flyballs_percent,flyballs,linedrives_percent,linedrives,popups_percent,popups,hp_to_1b,sprint_speed,&chart=false&x=xba&y=xba&r=no&chartType=beeswarm'
            url = url.format(season)
            s = requests.get(url).content
            soup = BeautifulSoup(s, "lxml")
            data = soup.find('script', type='')
            txt = data.contents[0]
            sm_txt = txt[txt.find("["):txt.find("]")+1]
            raw = json.loads(sm_txt)
            df_all = pd.DataFrame(raw)
            df = df_all[['year', 'player_id', 'player_name', 'pitch_count',
                        'pa',
                        'qualify_pa',
                        'qualify_ip',
                        'in_zone_percent',
                        'out_zone_percent',
                        'edge_percent',
                        'z_swing_percent',
                        'oz_swing_percent',
                        'iz_contact_percent',
                        'oz_contact_percent',
                        'whiff_percent',
                        'f_strike_percent',
                        'swing_percent',
                        'meatball_swing_percent',
                        'meatball_percent',
                        'z_swing_miss_percent',
                        'oz_swing_miss_percent',
                        'in_zone',
                        'out_zone',
                        'edge',
                        'barrels',
                        'popups',
                        'flyballs',
                        'linedrives',
                        'groundballs',
                        'popups_percent',
                        'flyballs_percent',
                        'linedrives_percent',
                        'groundballs_percent',
                        'pull_percent',
                        'straightaway_percent',
                        'opposite_percent',
                        'poorlyweak_percent',
                        'poorlytopped_percent',
                        'poorlyunder_percent',
                        'flareburner_percent',
                        'solidcontact_percent',
                        'hr_flyballs_percent',
                        'in_zone_swing',
                        'out_zone_swing',
                        'in_zone_swing_miss',
                        'out_zone_swing_miss',
                        'pitch_count_fastball',
                        'pitch_count_offspeed',
                        'pitch_count_breaking', 'k_percent',
                        'bb_percent',
                        'batted_ball',
                        'barrel',
                        'barrel_batted_rate',
                        'launch_angle_avg',
                        'exit_velocity_avg',
                        'sweet_spot_percent',
                        'hard_hit_percent',
                        'ba',
                        'xba',
                        'bacon',
                        'xbacon',
                        'babip',
                        'obp',
                        'slg',
                        'xobp',
                        'xslg',
                        'iso',
                        'xiso',
                        'woba',
                        'xwoba',
                        'wobacon',
                        'xwobacon',
                        'xbadiff',
                        'xslgdiff',
                        'wobadif',
                        'hp_to_1b',
                        'sprint_speed']]
            df.to_csv(raw_path, index = False)
            if writeS3 == True:
                u.upload_file(self,file_name = raw_path)
            else:
                pass

    def statcast_pitchers_season(self, season, writeS3 = False):
        raw_path = 'data/raw/pitchers/statcast/season/'+str(season)+'.csv'
        if os.path.exists(raw_path):
            print("file already exists")
            pass
        else:        
            url = 'https://baseballsavant.mlb.com/leaderboard/custom?year={}&type=pitcher&filter=&sort=1&sortDir=desc&min=10&selections=slg_percent,p_quality_start,xba,xslg,woba,xwoba,xobp,xiso,wobacon,xwobacon,bacon,xbacon,xbadiff,xslgdiff,wobadif,exit_velocity_avg,launch_angle_avg,sweet_spot_percent,barrels,barrel_batted_rate,solidcontact_percent,flareburner_percent,poorlyunder_percent,poorlytopped_percent,poorlyweak_percent,hard_hit_percent,z_swing_percent,z_swing_miss_percent,oz_swing_percent,oz_swing_miss_percent,oz_contact_percent,out_zone_swing_miss,out_zone_swing,out_zone_percent,out_zone,meatball_swing_percent,meatball_percent,pitch_count_offspeed,pitch_count_fastball,pitch_count_breaking,pitch_count,iz_contact_percent,in_zone_swing_miss,in_zone_swing,in_zone_percent,in_zone,edge_percent,edge,whiff_percent,swing_percent,pull_percent,straightaway_percent,opposite_percent,f_strike_percent,groundballs_percent,flyballs_percent,linedrives_percent,popups_percent,n_ff_formatted,ff_avg_speed,ff_avg_spin,ff_avg_break_x,ff_avg_break_z,ff_avg_break,ff_range_speed,n_sl_formatted,sl_avg_speed,sl_avg_spin,sl_avg_break_x,sl_avg_break_z,sl_avg_break,sl_range_speed,n_ch_formatted,ch_avg_speed,ch_avg_spin,ch_avg_break_x,ch_avg_break_z,ch_avg_break,ch_range_speed,n_cukc_formatted,cu_avg_speed,cu_avg_spin,cu_avg_break_x,cu_avg_break_z,cu_avg_break,cu_range_speed,n_sift_formatted,si_avg_break_x,n_fc_formatted,n_fs_formatted,n_kn_formatted,n_fastball_formatted,n_breaking_formatted,breaking_avg_speed,breaking_avg_spin,breaking_avg_break_x,breaking_avg_break_z,breaking_avg_break,breaking_range_speed,n_offspeed_formatted,offspeed_avg_speed,offspeed_avg_spin,offspeed_avg_break_x,offspeed_avg_break_z,offspeed_avg_break,offspeed_range_speed,&chart=false&x=xba&y=xba&r=no&chartType=beeswarm'
            url = url.format(season)
            s = requests.get(url).content
            soup = BeautifulSoup(s, "lxml")
            data = soup.find('script', type='')
            txt = data.contents[0]
            sm_txt = txt[txt.find("["):txt.find("]")+1]
            raw = json.loads(sm_txt)
            df_all = pd.DataFrame(raw)
            df = df_all[['year', 'player_id', 'player_name', 'pitch_count',
                        'pitch_count',
                        'in_zone_percent',
                        'out_zone_percent',
                        'edge_percent',
                        'z_swing_percent',
                        'oz_swing_percent',
                        'iz_contact_percent',
                        'oz_contact_percent',
                        'whiff_percent',
                        'f_strike_percent',
                        'swing_percent',
                        'meatball_swing_percent',
                        'meatball_percent',
                        'z_swing_miss_percent',
                        'oz_swing_miss_percent',
                        'popups_percent',
                        'flyballs_percent',
                        'linedrives_percent',
                        'groundballs_percent',
                        'pull_percent',
                        'straightaway_percent',
                        'opposite_percent',
                        'poorlyweak_percent',
                        'poorlytopped_percent',
                        'poorlyunder_percent',
                        'flareburner_percent',
                        'solidcontact_percent',
                        'hr_flyballs_percent',
                        'in_zone_swing',
                        'out_zone_swing',
                        'in_zone_swing_miss',
                        'out_zone_swing_miss',
                        'pitch_count_fastball',
                        'pitch_count_offspeed',
                        'pitch_count_breaking',
                        'k_percent',
                        'bb_percent',
                        'batted_ball',
                        'barrel',
                        'barrel_batted_rate',
                        'launch_angle_avg',
                        'exit_velocity_avg',
                        'sweet_spot_percent',
                        'hard_hit_percent',
                        'ba',
                        'xba',
                        'bacon',
                        'xbacon',
                        'babip',
                        'obp',
                        'slg',
                        'xobp',
                        'xslg',
                        'iso',
                        'xiso',
                        'woba',
                        'xwoba',
                        'wobacon',
                        'xwobacon',
                        'xbadiff',
                        'xslgdiff',
                        'wobadif',
                        'pitch_hand',
                        'n_toofew',
                        'n',
                        'n_ff_formatted',
                        'n_ff',
                        'ff_avg_speed',
                        'ff_avg_spin',
                        'ff_avg_break_x',
                        'ff_avg_break_z',
                        'ff_avg_break',
                        'ff_range_speed',
                        'ff_range_break_x',
                        'ff_range_break_z',
                        'ff_range_break',
                        'ff_calc_axis1',
                        'ff_calc_axis2',
                        'n_sl_formatted',
                        'n_sl',
                        'sl_avg_speed',
                        'sl_avg_spin',
                        'sl_avg_break_x',
                        'sl_avg_break_z',
                        'sl_avg_break',
                        'sl_range_speed',
                        'sl_range_break_x',
                        'sl_range_break_z',
                        'sl_range_break',
                        'sl_calc_axis1',
                        'sl_calc_axis2',
                        'n_ch_formatted',
                        'n_ch',
                        'ch_avg_speed',
                        'ch_avg_spin',
                        'ch_avg_break_x',
                        'ch_avg_break_z',
                        'ch_avg_break',
                        'ch_range_speed',
                        'ch_range_break_x',
                        'ch_range_break_z',
                        'ch_range_break',
                        'ch_calc_axis1',
                        'ch_calc_axis2',
                        'n_cukc_formatted',
                        'n_cu',
                        'cu_avg_speed',
                        'cu_avg_spin',
                        'cu_avg_break_x',
                        'cu_avg_break_z',
                        'cu_avg_break',
                        'cu_range_speed',
                        'cu_range_break_x',
                        'cu_range_break_z',
                        'cu_range_break',
                        'cu_calc_axis1',
                        'cu_calc_axis2',
                        'n_sift_formatted',
                        'n_si',
                        'si_avg_speed',
                        'si_avg_spin',
                        'si_avg_break_x',
                        'si_avg_break_z',
                        'si_avg_break',
                        'si_range_speed',
                        'si_range_break_x',
                        'si_range_break_z',
                        'si_range_break',
                        'si_calc_axis1',
                        'si_calc_axis2',
                        'n_fc_formatted',
                        'n_fc',
                        'fc_avg_speed',
                        'fc_avg_spin',
                        'fc_avg_break_x',
                        'fc_avg_break_z',
                        'fc_avg_break',
                        'fc_range_speed',
                        'fc_range_break_x',
                        'fc_range_break_z',
                        'fc_range_break',
                        'fc_calc_axis1',
                        'fc_calc_axis2',
                        'n_fs_formatted',
                        'n_fs',
                        'fs_avg_speed',
                        'fs_avg_spin',
                        'fs_avg_break_x',
                        'fs_avg_break_z',
                        'fs_avg_break',
                        'fs_range_speed',
                        'fs_range_break_x',
                        'fs_range_break_z',
                        'fs_range_break',
                        'fs_calc_axis1',
                        'fs_calc_axis2',
                        'n_kn_formatted',
                        'n_kn',
                        'kn_avg_speed',
                        'kn_avg_spin',
                        'kn_avg_break_x',
                        'kn_avg_break_z',
                        'kn_avg_break',
                        'kn_range_speed',
                        'kn_range_break_x',
                        'kn_range_break_z',
                        'kn_range_break',
                        'kn_calc_axis1',
                        'kn_calc_axis2',
                        'n_fastball_formatted',
                        'n_fastball',
                        'fastball_avg_speed',
                        'fastball_avg_spin',
                        'fastball_avg_break_x',
                        'fastball_avg_break_z',
                        'fastball_avg_break',
                        'fastball_range_speed',
                        'fastball_range_break_x',
                        'fastball_range_break_z',
                        'fastball_range_break',
                        'fastball_calc_axis1',
                        'fastball_calc_axis2',
                        'n_breaking_formatted',
                        'n_breaking',
                        'breaking_avg_speed',
                        'breaking_avg_spin',
                        'breaking_avg_break_x',
                        'breaking_avg_break_z',
                        'breaking_avg_break',
                        'breaking_range_speed',
                        'breaking_range_break_x',
                        'breaking_range_break_z',
                        'breaking_range_break',
                        'breaking_calc_axis1',
                        'breaking_calc_axis2',
                        'n_offspeed_formatted',
                        'n_offspeed',
                        'offspeed_avg_speed',
                        'offspeed_avg_spin',
                        'offspeed_avg_break_x',
                        'offspeed_avg_break_z',
                        'offspeed_avg_break',
                        'offspeed_range_speed',
                        'offspeed_range_break_x',
                        'offspeed_range_break_z',
                        'offspeed_range_break',
                        'offspeed_calc_axis1',
                        'offspeed_calc_axis2',
                        'percent_rank_fastball_velo',
                        'percent_rank_fastball_spin',
                        'percent_rank_cu_spin',
                        'p_ab',
                        'p_k_percent',
                        'p_bb_percent',
                        'p_formatted_ip',
                        'batting_avg',
                        'slg_percent',
                        'on_base_percent',
                        'on_base_plus_slg'
                        ]]
            df.to_csv(raw_path, index = False)
            if writeS3 == True:
                u.upload_file(self,file_name = raw_path)
            else:
                pass











