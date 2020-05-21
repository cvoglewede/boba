import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from .utils import Boba_Utils as u


class Boba_Diagnostics(u):

    def __init__(self):
        pass

    def run_model_diagnostics(self, model, X_train, X_test, y_train, y_test, target):
        self.plot_shap_imp(model,X_train)
        self.feature_imp(model,X_train)
        self.residual_plot(model,X_test,y_test,target)
        self.residual_density_plot(model,X_test,y_test,target)
        self.residual_mean_plot(model,X_test,y_test,target)
        self.residual_variance_plot(model,X_test,y_test,target)
        self.PVA_plot(model,X_test,y_test,target)
        self.inverse_PVA_plot(model,X_train,y_train,target)
        
    def plot_shap_imp(self,model,X_train):
        shap_values = shap.TreeExplainer(model).shap_values(X_train)
        shap.summary_plot(shap_values, X_train)
        plt.show()
    
    def feature_imp(self,model,X_train):
        names = X_train.columns
        coef_df = pd.DataFrame({"Feature": names, "Importance": model.feature_importances_}, 
                            columns=["Feature", "Importance"])
        coef_df = coef_df.sort_values('Importance',ascending=False)
        coef_df
        fig, ax = plt.subplots()
        sns.barplot(x="Importance", y="Feature", data=coef_df.head(20),
                label="Importance", color="b",orient='h')
        plt.show()

    def residual_plot(self,model, X_test, y_test,target):
        pred = model.predict(X_test)
        residuals = pd.Series(pred,index=X_test.index) - pd.Series(y_test[target])
        fig, ax = plt.subplots()
        ax.scatter(pred, residuals)
        ax.plot([pred.min(), pred.max()], [0, 0], 'k--', lw=4)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residuals')
        plt.show()

    def residual_density_plot(self,model, X_test, y_test,target):
        pred = model.predict(X_test)
        residuals = pd.Series(pred,index=X_test.index) - pd.Series(y_test[target])
        sns.distplot(residuals)
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
            sns.set(style="whitegrid")
            ax = sns.lineplot(x="index", y="residuals", data=res_var)
            plt.show()
        except:
            pass   

    def residual_mean_plot(self, model, X_test, y_test,target):
        try:
            pred = model.predict(X_test)
            residuals = pd.Series(pred,index=X_test.index) - pd.Series(y_test[target])
            y_temp = y_test.copy()
            y_temp['pred'] = pred
            y_temp['residuals'] = residuals
            res_var = y_temp.groupby(pd.qcut(y_temp['pred'], 10))['residuals'].mean()
            res_var.index =  [1,2,3,4,5,6,7,8,9,10]
            res_var = res_var.reset_index()
            sns.set(style="whitegrid")
            ax = sns.lineplot(x="index", y="residuals", data=res_var)
            plt.show()
        except:
            pass   

    def PVA_plot(self,model, X_test, y_test, target):
        try:
            pred = model.predict(X_test)
            residuals = pd.Series(pred,index=X_test.index) - pd.Series(y_test[target])
            y_temp = y_test.copy()
            y_temp['pred'] = pred
            y_temp['residuals'] = residuals
            pva = y_temp.groupby(pd.qcut(y_temp['pred'], 10))[target,'pred'].mean()
            pva.index =  [1,2,3,4,5,6,7,8,9,10]
            pva = pva.reset_index()
            sns.set(style="whitegrid")
            df = pva.melt('index', var_name='cols',  value_name='vals')
            sns.factorplot(x="index", y="vals", hue='cols', data=df)   
            plt.show()
        except:
            pass

    def inverse_PVA_plot(self, model,X_test, y_test,target):
        try:
            pred = model.predict(X_test)
            residuals = pd.Series(pred,index=X_test.index) - pd.Series(y_test[target])
            y_temp = y_test.copy()
            y_temp['pred'] = pred
            y_temp['residuals'] = residuals
            pva = y_temp.groupby(pd.qcut(y_temp[target], 10))[target,'pred'].mean()
            pva.index =  [1,2,3,4,5,6,7,8,9,10]
            pva = pva.reset_index()
            # fig, ax = plt.subplots()
            sns.set(style="whitegrid")
            df = pva.melt('index', var_name='cols',  value_name='vals')
            sns.factorplot(x="index", y="vals", hue='cols', data=df)   
            # print("PVA Done")
            plt.show()
        except:
            pass  