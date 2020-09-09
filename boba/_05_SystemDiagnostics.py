import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import yaml
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# from .utils import Boba_Utils as u
# from ._03_Modeling import Boba_Modeling as m


class Boba_Sys_Diagnostics():

    def __init__(self):
        pass

    def run_sys_scoring(self, model, target,prod):
        if prod == True:
            pass
        else:
            master_df = pd.read_csv('data/processed/'+self.position_group+'/master_df.csv',index_col=0)  

            path = 'data/scoring/evaluation_'+self.position_group+'_'+str(self.year-1)+'.csv'
            if os.path.exists(path):
                print('does exist')
                evaluation_df = pd.read_csv(path,index_col=0)
            else:
                print('does not exist')
                data_group = 'hitters' if self.position_group == 'hitters' else 'pitchers'
                zips_df = pd.read_csv('data/raw/'+data_group+'/projection_systems/zips/'+str(self.year-1)+'.csv')
                atc_df = pd.read_csv('data/raw/'+data_group+'/projection_systems/atc/'+str(self.year-1)+'.csv')
                bat_df = pd.read_csv('data/raw/'+data_group+'/projection_systems/thebat/'+str(self.year-1)+'.csv')
                stmr_df = pd.read_csv('data/raw/'+data_group+'/projection_systems/steamer/'+str(self.year-1)+'.csv')
                zips_df = zips_df.rename(columns={"K/9": "K_per_9", "BB/9": "BB_per_9"})
                atc_df = atc_df.rename(columns={"K/9": "K_per_9", "BB/9": "BB_per_9"})
                bat_df = bat_df.rename(columns={"K/9": "K_per_9", "BB/9": "BB_per_9"})
                stmr_df = stmr_df.rename(columns={"K/9": "K_per_9", "BB/9": "BB_per_9"})
                evaluation_df = master_df[master_df['Season']==(self.year-1)]
                evaluation_df['playerID'] = evaluation_df['playerID'].astype('str')
                if self.position_group == 'hitters':
                    evaluation_df = evaluation_df[self.information_cols+[self.pt_metric]+self.model_targets+self.counting_stats]
                    zips_df = zips_df[['playerid']+[self.pt_metric]+[x for x in zips_df.columns if x in self.model_targets]+[x for x in zips_df.columns if x in self.counting_stats]]
                    atc_df = atc_df[['playerid']+[self.pt_metric]+[x for x in atc_df.columns if x in self.model_targets]+[x for x in atc_df.columns if x in self.counting_stats]]
                    bat_df = bat_df[['playerid']+[self.pt_metric]+[x for x in bat_df.columns if x in self.model_targets]+[x for x in bat_df.columns if x in self.counting_stats]]
                    stmr_df = stmr_df[['playerid']+[self.pt_metric]+[x for x in stmr_df.columns if x in self.model_targets]+[x for x in stmr_df.columns if x in self.counting_stats]]
                else:
                    evaluation_df = evaluation_df[self.information_cols+[self.pt_metric]+[self.per_metric]+self.model_targets+self.counting_stats]
                    zips_df = zips_df[['playerid']+[self.pt_metric]+[self.per_metric]+[x for x in zips_df.columns if x in self.model_targets]+[x for x in zips_df.columns if x in self.counting_stats]]
                    atc_df = atc_df[['playerid']+[self.pt_metric]+[self.per_metric]+[x for x in atc_df.columns if x in self.model_targets]+[x for x in atc_df.columns if x in self.counting_stats]]
                    bat_df = bat_df[['playerid']+[self.pt_metric]+[self.per_metric]+[x for x in bat_df.columns if x in self.model_targets]+[x for x in bat_df.columns if x in self.counting_stats]]
                    stmr_df = stmr_df[['playerid']+[self.pt_metric]+[self.per_metric]+[x for x in stmr_df.columns if x in self.model_targets]+[x for x in stmr_df.columns if x in self.counting_stats]]
            
                evaluation_df = pd.merge(evaluation_df,zips_df,how='left',left_on='playerID', right_on='playerid',suffixes=('','_zips')).drop('playerid',axis=1)
                evaluation_df = pd.merge(evaluation_df,atc_df,how='left',left_on='playerID', right_on='playerid',suffixes=('','_atc')).drop('playerid',axis=1)
                evaluation_df = pd.merge(evaluation_df,bat_df,how='left',left_on='playerID', right_on='playerid',suffixes=('','_bat')).drop('playerid',axis=1)
                evaluation_df = pd.merge(evaluation_df,stmr_df,how='left',left_on='playerID', right_on='playerid',suffixes=('','_stmr')).drop('playerid',axis=1)
                evaluation_df.to_csv(path)
        
            temp_df = master_df[master_df['Season']==(self.year-1)]
            temp_df['Season'] = (self.year-2)
            temp_df = self.isolate_relevant_columns(modeling_df = temp_df,target = target)
            temp_df = temp_df.drop([target],axis=1)
            pipeline = pickle.load(open('data/modeling/'+self.position_group+'/'+target+'/preprocessing_pipeline_eval.sav', 'rb'))
            temp_df_2 = pipeline.transform(temp_df)
            with open(r'data/modeling/'+self.position_group+'/'+target+'/model_features_eval.yaml') as file:
                        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
            model_features = yaml_data[target]
            temp_df = pd.DataFrame(temp_df_2, columns = model_features,index=temp_df.index)
            temp_df[target+'_Boba'] = model.predict(temp_df)
            temp_df = temp_df[[target+'_Boba']]
            evaluation_df = evaluation_df.drop([target+'_Boba'],axis=1,errors='ignore')
            new_df = pd.merge(evaluation_df,temp_df,left_index=True,right_index=True)
            rate_stats = [c+'_per_'+self.per_metric for c in self.counting_stats]
            if target in rate_stats:
                colname = target.replace('_per_'+self.per_metric, '')
                new_df[colname+'_Boba'] = new_df[target+'_Boba']*new_df[self.per_metric+'_zips']
            else:
                colname = target
            zips_metric = colname+'_zips'
            atc_metric = colname+'_atc'
            bat_metric = colname+'_bat'
            stmr_metric = colname+'_stmr'
            BOBA_metric = colname+'_Boba'
            systems_list = [c for c in [BOBA_metric,zips_metric,stmr_metric,bat_metric,atc_metric] if c in list(new_df.columns)]
            new_df[colname+'_averaged'] = new_df[systems_list].mean(axis=1)
            new_df.to_csv(path)
            return new_df

    def run_sys_diagnostics(self, evaluation_df, target,prod):
        rate_stats = [c+'_per_'+self.per_metric for c in self.counting_stats]
        if target in rate_stats:
            colname = target.replace('_per_'+self.per_metric, '')
        else:
            colname = target
        if prod == True:
            share_df = pd.DataFrame(columns = ['winning_sys'],index=['Boba','Zips','ATC','STMR','BAT','averaged']).fillna(0)
            return share_df    
        else:
            zips_metric = colname+'_zips'
            atc_metric = colname+'_atc'
            bat_metric = colname+'_bat'
            stmr_metric = colname+'_stmr'
            BOBA_metric = colname+'_Boba'
            averaged_metric = colname+'_averaged'
            systems_list = [c for c in [colname,BOBA_metric,zips_metric,stmr_metric,bat_metric,atc_metric,averaged_metric] if c in list(evaluation_df.columns)]           
            eval_results_df = evaluation_df[['Name','playerID','Age']+systems_list].sort_values(averaged_metric,ascending=False)
            eval_results_df['Boba'] = abs(eval_results_df[colname]-eval_results_df[BOBA_metric])*-1
            if zips_metric in list(eval_results_df.columns):
                eval_results_df['Zips'] = abs(eval_results_df[colname]-eval_results_df[zips_metric])*-1
            if atc_metric in list(eval_results_df.columns):
                eval_results_df['ATC'] = abs(eval_results_df[colname]-eval_results_df[atc_metric])*-1
            if stmr_metric in list(eval_results_df.columns):
                eval_results_df['STMR'] = abs(eval_results_df[colname]-eval_results_df[stmr_metric])*-1
            if bat_metric in list(eval_results_df.columns):
                eval_results_df['BAT'] = abs(eval_results_df[colname]-eval_results_df[bat_metric])*-1
            eval_results_df['averaged'] = abs(eval_results_df[colname]-eval_results_df[colname+'_averaged'])*-1
            systems_list_names = [c for c in ['Boba','Zips','ATC','STMR','BAT','averaged'] if c in list(eval_results_df.columns)]

            eval_results_df['winning_val'] = eval_results_df[systems_list_names].max(axis=1)
            eval_results_df['winning_sys'] = eval_results_df[systems_list_names].idxmax(axis=1)

            remove_na_df = eval_results_df.dropna(subset=systems_list)

            share_df = remove_na_df.groupby('winning_sys')['winning_sys'].count().sort_values(ascending=False)

            sns.set(style="darkgrid")
            pva = eval_results_df.groupby(pd.qcut(eval_results_df[averaged_metric], 10))[systems_list].mean()
            pva.index =  list(np.arange(1,11,1))
            pva = pva.reset_index()
            df = pva.melt('index', var_name='cols',  value_name='vals')
            sns.factorplot(x="index", y= 'vals',hue='cols',data=df,legend_out=False)
            plt.title("System Comparison vs Actuals for {}".format(colname))
            plt.xlabel("Average Prediction Sorted Decile")
            plt.ylabel("{}".format(colname))
            plt.legend(loc='upper left')
            plt.show()

            try:
                n = 3
                ws_plot_df = remove_na_df.groupby([pd.qcut(remove_na_df[averaged_metric], n),'winning_sys'])['winning_sys'].count()
                ws_plot_df = pd.DataFrame(ws_plot_df)
                ws_plot_df = ws_plot_df.rename(columns = {'winning_sys':'count'})
                ws_plot_df = ws_plot_df.reset_index(level=[1])
                index_list = []
                for i in list(np.arange(1,n+1,1)):
                    temp_list = np.ones(len(systems_list_names))*i
                    index_list.append(temp_list)
                ws_plot_df.index = np.concatenate(index_list).ravel().tolist()
                ws_plot_df = ws_plot_df.reset_index()
                sns.barplot(x="index",y='count',data=ws_plot_df, hue= 'winning_sys')
                plt.xticks(np.arange(n), ['First Tercile', 'Second Tercile','Third Tercile'])
                plt.show()
            except:
                pass  

            remove_na_df = evaluation_df.dropna(subset=systems_list)

            data = {'system':  ['BOBA', 'zips','steamer','atc','bat','averaged'],
                'WinShare': [share_df['Boba']/share_df.sum(), 
                    share_df['Zips']/share_df.sum() if 'Zips' in list(share_df.index) else 0,
                    share_df['STMR']/share_df.sum() if 'STMR' in list(share_df.index) else 0,
                    share_df['ATC']/share_df.sum() if 'ATC' in list(share_df.index) else 0,
                    share_df['BAT']/share_df.sum() if 'BAT' in list(share_df.index) else 0,
                    share_df['averaged']/share_df.sum()],
                'R2': [r2_score(remove_na_df[colname],remove_na_df[BOBA_metric]), 
                    r2_score(remove_na_df[colname],remove_na_df[zips_metric]) if zips_metric in list(remove_na_df.columns) else 0,
                    r2_score(remove_na_df[colname],remove_na_df[stmr_metric]) if stmr_metric in list(remove_na_df.columns) else 0,
                    r2_score(remove_na_df[colname],remove_na_df[atc_metric]) if atc_metric in list(remove_na_df.columns) else 0,
                    r2_score(remove_na_df[colname],remove_na_df[bat_metric]) if bat_metric in list(remove_na_df.columns) else 0,
                    r2_score(remove_na_df[colname],remove_na_df[averaged_metric])],
                'Corr': [np.corrcoef(remove_na_df[colname],remove_na_df[BOBA_metric])[0,1], 
                        np.corrcoef(remove_na_df[colname],remove_na_df[zips_metric])[0,1] if zips_metric in list(remove_na_df.columns) else 0,
                        np.corrcoef(remove_na_df[colname],remove_na_df[stmr_metric])[0,1] if stmr_metric in list(remove_na_df.columns) else 0,
                        np.corrcoef(remove_na_df[colname],remove_na_df[atc_metric])[0,1] if atc_metric in list(remove_na_df.columns) else 0,
                        np.corrcoef(remove_na_df[colname],remove_na_df[bat_metric])[0,1] if bat_metric in list(remove_na_df.columns) else 0,
                        np.corrcoef(remove_na_df[colname],remove_na_df[averaged_metric])[0,1]],
                'RMSE': [np.sqrt(mean_squared_error(remove_na_df[colname],remove_na_df[BOBA_metric])), 
                        np.sqrt(mean_squared_error(remove_na_df[colname],remove_na_df[zips_metric])) if zips_metric in list(remove_na_df.columns) else 0,
                        np.sqrt(mean_squared_error(remove_na_df[colname],remove_na_df[stmr_metric])) if stmr_metric in list(remove_na_df.columns) else 0,
                        np.sqrt(mean_squared_error(remove_na_df[colname],remove_na_df[atc_metric])) if atc_metric in list(remove_na_df.columns) else 0,
                        np.sqrt(mean_squared_error(remove_na_df[colname],remove_na_df[bat_metric])) if bat_metric in list(remove_na_df.columns) else 0,
                        np.sqrt(mean_squared_error(remove_na_df[colname],remove_na_df[averaged_metric]))],
                'MAE': [mean_absolute_error(remove_na_df[colname],remove_na_df[BOBA_metric]), 
                        mean_absolute_error(remove_na_df[colname],remove_na_df[zips_metric]) if zips_metric in list(remove_na_df.columns) else 0,
                        mean_absolute_error(remove_na_df[colname],remove_na_df[stmr_metric]) if stmr_metric in list(remove_na_df.columns) else 0,
                        mean_absolute_error(remove_na_df[colname],remove_na_df[atc_metric]) if atc_metric in list(remove_na_df.columns) else 0,
                        mean_absolute_error(remove_na_df[colname],remove_na_df[bat_metric]) if bat_metric in list(remove_na_df.columns) else 0,
                        mean_absolute_error(remove_na_df[colname],remove_na_df[averaged_metric])]
                }

            compare_df = pd.DataFrame(data, columns = ['system','WinShare','R2','Corr','RMSE','MAE'])
            compare_df = compare_df.sort_values('RMSE',ascending=True)
            print(compare_df)
            return compare_df