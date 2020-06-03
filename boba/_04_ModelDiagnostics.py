import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
# from .utils import Boba_Utils as u


class Boba_Model_Diagnostics():

    def __init__(self):
        pass

    def run_model_diagnostics(self, model, X_train, X_test, y_train, y_test, target):
        self.get_model_stats(model, X_train, X_test, y_train, y_test, target)
        self.plot_shap_imp(model,X_train)
        self.plot_shap_bar(model,X_train)
        self.residual_plot(model,X_test,y_test,target)
        self.residual_density_plot(model,X_test,y_test,target)
        self.identify_outliers(model, X_test, y_test,target)
        self.residual_mean_plot(model,X_test,y_test,target)
        self.residual_variance_plot(model,X_test,y_test,target)
        self.PVA_plot(model,X_test,y_test,target)
        self.inverse_PVA_plot(model,X_train,y_train,target)
        self.estimates_by_var(model,X_train,y_train,target,'Age')
        self.error_by_var(model,X_train,y_train,target,'Age')
        self.volatility_by_var(model,X_train,y_train,target,'Age')
        
    def get_model_stats(self, model, X_train, X_test, y_train, y_test, target):
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        test_RMSE = np.sqrt(mean_squared_error(y_test, test_pred)),
        test_R2 = model.score(X_test,y_test),
        test_MAE = mean_absolute_error(y_test, test_pred),
        train_RMSE = np.sqrt(mean_squared_error(y_train, train_pred)),
        train_R2 = model.score(X_train,y_train),
        train_MAE = mean_absolute_error(y_train, train_pred),
        df = pd.DataFrame(data = {'RMSE': np.round(train_RMSE,4),
                                  'R^2': np.round(train_R2,4),
                                  'MAE': np.round(train_MAE,4)}, index = ['train'])
        df2 = pd.DataFrame(data = {'RMSE': np.round(test_RMSE,4),
                                  'R^2': np.round(test_R2,4),
                                  'MAE': np.round(test_MAE,4)}, index = ['test'])
        print("Model Statistics for {}".format(target))
        print('-'*40)
        print(df)
        print('-'*40)
        print(df2)
        print('-'*40)


    def plot_shap_imp(self,model,X_train):
        shap_values = shap.TreeExplainer(model).shap_values(X_train)
        shap.summary_plot(shap_values, X_train)
        plt.show()

    def plot_shap_bar(self,model,X_train):
        shap_values = shap.TreeExplainer(model).shap_values(X_train)
        shap.summary_plot(shap_values, X_train, plot_type='bar')
        plt.show()
    
    def feature_imp(self,model,X_train,target):
        sns.set_style('darkgrid')
        names = X_train.columns
        coef_df = pd.DataFrame({"Feature": names, "Importance": model.feature_importances_}, 
                            columns=["Feature", "Importance"])
        coef_df = coef_df.sort_values('Importance',ascending=False)
        coef_df
        fig, ax = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=coef_df.head(20),
                label="Importance", color="b",orient='h')
        plt.title("XGB Feature Importances for {}".format(target))
        plt.show()

    def residual_plot(self,model, X_test, y_test,target):
        pred = model.predict(X_test)
        residuals = pd.Series(pred,index=X_test.index) - pd.Series(y_test[target])
        fig, ax = plt.subplots()
        ax.scatter(pred, residuals)
        ax.plot([pred.min(), pred.max()], [0, 0], 'k--', lw=4)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residuals')
        plt.title("Residual Plot for {}".format(target))
        plt.show()

    def residual_density_plot(self,model, X_test, y_test,target):
        sns.set_style('darkgrid')
        pred = model.predict(X_test)
        residuals = pd.Series(pred,index=X_test.index) - pd.Series(y_test[target])
        sns.distplot(residuals)
        plt.title("Residual Density Plot for {}".format(target))
        plt.show()
    
    def residual_variance_plot(self, model, X_test, y_test,target):

        try:
            pred = model.predict(X_test)
            residuals = pd.Series(pred,index=X_test.index) - pd.Series(y_test[target])
            y_temp = y_test.copy()
            y_temp['pred'] = pred
            y_temp['residuals'] = residuals
            res_var = y_temp.groupby(pd.qcut(y_temp[target], 10))['residuals'].std()
            res_var.index =  [1,2,3,4,5,6,7,8,9,10]
            res_var = res_var.reset_index()
            ax = sns.lineplot(x="index", y="residuals", data=res_var)
            plt.title("Residual Variance plot for {}".format(target))
            plt.xlabel("Prediction Decile")
            plt.ylabel("Residual Variance")
            plt.show()
        except:
            pass   

    def residual_mean_plot(self, model, X_test, y_test,target):
        sns.set_style('darkgrid')
        try:
            pred = model.predict(X_test)
            residuals = pd.Series(pred,index=X_test.index) - pd.Series(y_test[target])
            y_temp = y_test.copy()
            y_temp['pred'] = pred
            y_temp['residuals'] = residuals
            res_var = y_temp.groupby(pd.qcut(y_temp['pred'], 10))['residuals'].mean()
            res_var.index =  [1,2,3,4,5,6,7,8,9,10]
            res_var = res_var.reset_index()
            ax = sns.lineplot(x="index", y="residuals", data=res_var)
            plt.title("Residual Mean plot for {}".format(target))
            plt.xlabel("Prediction Decile")
            plt.ylabel("Residual Mean")
            plt.show()
        except:
            pass   

    def PVA_plot(self,model, X_test, y_test, target):
        sns.set_style('darkgrid')
        try:
            pred = model.predict(X_test)
            residuals = pd.Series(pred,index=X_test.index) - pd.Series(y_test[target])
            y_temp = y_test.copy()
            y_temp['predicted'] = pred
            y_temp['residuals'] = residuals
            pva = y_temp.groupby(pd.qcut(y_temp['predicted'], 10))[target,'predicted'].mean()
            pva.index =  [1,2,3,4,5,6,7,8,9,10]
            pva = pva.reset_index()
            pva = pva.rename(columns={target: "actual"})
            df = pva.melt('index', var_name='cols',  value_name='vals')
            sns.factorplot(x="index", y="vals", hue='cols', data=df,legend_out=False)
            plt.title("Predicted v Actual Chart by Deciles for {}".format(target))
            plt.xlabel("Prediction Decile")
            plt.ylabel("{}".format(target))
            plt.legend(loc='upper left')
            plt.show()
        except:
            pass

    def inverse_PVA_plot(self, model,X_test, y_test,target):
        sns.set_style('darkgrid')
        try:
            pred = model.predict(X_test)
            residuals = pd.Series(pred,index=X_test.index) - pd.Series(y_test[target])
            y_temp = y_test.copy()
            y_temp['predicted'] = pred
            y_temp['residuals'] = residuals
            pva = y_temp.groupby(pd.qcut(y_temp[target], 10))[target,'predicted'].mean()
            pva.index =  [1,2,3,4,5,6,7,8,9,10]
            pva = pva.reset_index()
            pva = pva.rename(columns={target: "actual"})
            df = pva.melt('index', var_name='cols',  value_name='vals')
            sns.factorplot(x="index", y="vals", hue='cols', data=df,legend_out=False)   
            plt.title("Actual v Predicted Chart by Deciles for {}".format(target))
            plt.xlabel("Actual Decile")
            plt.ylabel("{}".format(target))
            plt.legend(loc='upper left')
            plt.show()
        except:
            pass  

    def identify_outliers(self, model, X_test, y_test,target):
        master_df =  pd.read_csv('data/processed/'+self.position_group+'/master_df.csv',index_col=0) 
        index_list = list(X_test.index)
        master_df = master_df.iloc[index_list,:]
        pred_df = pd.DataFrame(data = {'pred':model.predict(X_test),
                                    'residuals':pd.Series(model.predict(X_test),index=X_test.index) - pd.Series(y_test[target])},index=X_test.index)

        master_df = pd.merge(master_df,pred_df,left_index=True,right_index=True)
        master_df = master_df[['Season','Name',target,'Age','pred','residuals']]
        print('Top 20 UnderEstimates')
        print(master_df.sort_values('residuals',ascending=True).head(20))
        print('-'*80)
        print('Top 20 OverEstimates')
        print(master_df.sort_values('residuals',ascending=True).tail(20))


    def estimates_by_var(self, model, X_test, y_test,target,var):
        sns.set_style('darkgrid')
        master_df =  pd.read_csv('data/processed/'+self.position_group+'/master_df.csv',index_col=0) 
        index_list = list(X_test.index)
        master_df = master_df.iloc[index_list,:]
        pred_df = pd.DataFrame(data = {'pred':model.predict(X_test),
                                    'residuals':pd.Series(model.predict(X_test),index=X_test.index) - pd.Series(y_test[target])},index=X_test.index)
        master_df = pd.merge(master_df,pred_df,left_index=True,right_index=True)

        gb = master_df.groupby(master_df[var])['pred',target].mean()
        gb = gb.reset_index()
        gb = gb.rename(columns={target: "actual",'pred':'predicted'})
        df = gb.melt(var, var_name='type',  value_name='vals')
        ax = sns.lineplot(x=var, y="vals", hue="type",data=df)
        
        plt.title("Average Estimated {} by {}".format(target,var))
        plt.xlabel("{}".format(var))
        plt.ylabel("{}".format(target))
        plt.xticks(np.arange(gb[var].min(), gb[var].max(), step=1),rotation=45)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:])
        plt.show()


    def error_by_var(self, model, X_test, y_test,target,var):
        sns.set_style('darkgrid')
        master_df =  pd.read_csv('data/processed/'+self.position_group+'/master_df.csv',index_col=0) 
        index_list = list(X_test.index)
        master_df = master_df.iloc[index_list,:]
        pred_df = pd.DataFrame(data = {'pred':model.predict(X_test),
                                    'residuals':pd.Series(model.predict(X_test),index=X_test.index) - pd.Series(y_test[target])},index=X_test.index)
        master_df = pd.merge(master_df,pred_df,left_index=True,right_index=True)

        gb = master_df.groupby(master_df[var])['residuals'].mean()
        gb = gb.reset_index()
        ax = sns.lineplot(x=var, y="residuals", data=gb)
        plt.title("Average Error by {}".format(var))
        plt.xlabel("{}".format(var))
        plt.ylabel("Residual mean")
        plt.show()


    def volatility_by_var(self, model, X_test, y_test,target,var):
        sns.set_style('darkgrid')
        master_df =  pd.read_csv('data/processed/'+self.position_group+'/master_df.csv',index_col=0) 
        index_list = list(X_test.index)
        master_df = master_df.iloc[index_list,:]
        pred_df = pd.DataFrame(data = {'pred':model.predict(X_test),
                                    'residuals':pd.Series(model.predict(X_test),index=X_test.index) - pd.Series(y_test[target])},index=X_test.index)
        master_df = pd.merge(master_df,pred_df,left_index=True,right_index=True)

        gb = master_df.groupby(master_df[var])['residuals'].std()
        gb = gb.reset_index()
        ax = sns.lineplot(x=var, y="residuals", data=gb)
        plt.title("Estimate Volatility by {}".format(var))
        plt.xlabel("{}".format(var))
        plt.ylabel("Residual std error")
        plt.show()
 