# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 23:14:01 2022

@author: SaeidNikbakht
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.style.use("ggplot")
from tensorflow.keras.models import load_model
import pickle

class ModelPredictor:
    
    # loading the model
    def MLConf_extractor(self, MLConf):
        model_path = MLConf["model_data"]["callbacks_dict"]["model_path"]
        model_name = MLConf["model_data"]["initializer_dict"]["model_name"]
        trained_model = load_model(model_path+model_name+".h5")
    
        history_path = MLConf["model_data"]["data_saver_dict"]["history_path"]
        history_file_name = "History_" + model_name + ".pickle"
        with open(history_path + history_file_name, "rb") as pickle_file:
            history = pickle.load(pickle_file)
    
        test_data_path = MLConf["model_data"]["data_saver_dict"]["test_data_path"]
        X_test_file_name = "X_test_np_" + model_name + ".pickle"
        with open(test_data_path + X_test_file_name, "rb") as pickle_file:
            X_test = pickle.load(pickle_file)
    
        y_test_file_name = "y_test_np_" + model_name + ".pickle"
        with open(test_data_path + y_test_file_name, "rb") as pickle_file:
            y_test = pickle.load(pickle_file)
    
        clean_data_path = MLConf["model_data"]["data_saver_dict"]["clean_data_path"]
        clean_data_file_name = "df_" + model_name + ".csv"
        clean_df = pd.read_csv(clean_data_path+clean_data_file_name, index_col = "Time", parse_dates = True)
        
        return trained_model, history, X_test, y_test, clean_df
    
    
    def predict_next_values(self, trained_model, X_test_scaled_3d, y_test):
        y_pred = pd.DataFrame(trained_model.predict(X_test_scaled_3d))
        y_pred.columns = y_test.columns
        y_pred.index = y_test.index
        return y_pred
    
    
    def df_reshaper(self, df, y_test):
        temp_dict = dict()
        for date in y_test.index:
            timestamp = pd.to_datetime(date).strftime("%Y-%m-%d %H:%M:%S")
            temp_dict[timestamp] = df.loc[timestamp:,:].iloc[1:17,:].values.flatten()
        y_reshaped_df = pd.DataFrame(temp_dict)
        y_reshaped_df.index = np.arange(1, y_test.shape[1]+1)
        y_reshaped_df = y_reshaped_df.T
        y_reshaped_df.index = pd.to_datetime(y_reshaped_df.index)
        return y_reshaped_df
    
    
    def get_error_df_individual(self, y_pred, y_test):
        error_pred_dict = dict()
        for date in y_test.index:
            
            date_str = date.strftime("%Y-%m-%d %H:%M:%S")
            y_test_today = y_test.loc[date, :]
            y_pred_today = y_pred.loc[date, :]
            error_pred = (abs(y_test_today - y_pred_today)).mean()
            error_pred_dict[date_str] = error_pred

        error_df = pd.DataFrame(error_pred_dict, index = [0]).T
        error_df.index = pd.to_datetime(error_df.index)
        error_df.index.name = "Time"
        error_df.columns = ["avg_error"]
    
        return error_df
    
    
    def get_error_df_total(self, trained_model, X_test, y_test, clean_df):
        y_lstm = self.predict_next_values(trained_model, X_test, y_test)
        for col in list(y_lstm):
            y_lstm.loc[y_lstm[col]<0,col]=0
        y_sim_reshaped = self.df_reshaper(clean_df.loc[:,["pv_sim_15min"]], y_test)
        error_df_sim = self.get_error_df_individual(y_sim_reshaped, y_test)
        error_df_lstm = self.get_error_df_individual(y_lstm, y_test)
        error_df = pd.concat([error_df_lstm, error_df_sim], axis = 1)
        error_df.columns = ["err_lstm", "err_sim"]
        return error_df


    def plot_metrics(self, history, metric_name, path, trained_model):
        val_metric_name = "val_" + metric_name
        fig, [ax1, ax2] = plt.subplots(nrows = 1, ncols = 2, figsize = [25,6])
        loss =history["loss"]
        val_loss = history["val_loss"]
        
        num_epochs = len(loss)
        x_ticks = np.arange(1,num_epochs+1)
        ax1.plot(x_ticks, loss, label = "loss", c = "orange")
        ax1.plot(x_ticks, val_loss, label = "validation_loss", c = "green")
        ax1.legend()
        ax1.grid(which = "both", lw=2)
        ax1.set_xlabel("epochs")
        ax1.set_ylabel("loss")
        
        metric = history[metric_name]
        val_metric = history[val_metric_name]
        ax2.plot(x_ticks, metric, label = metric_name, c = "orange")
        ax2.plot(x_ticks, val_metric, label = val_metric_name, c = "green")
        ax2.legend()
        ax2.grid(which = "both", lw=2)
        ax2.set_xlabel("epochs")
        ax2.set_ylabel("MAE")
        
        plt.savefig(path + "training_history_" + trained_model.name + ".jpg", dpi = 100)
    
    
    def plot_hist_error_comp(self, error_df, path_hist, trained_model, num_bins, bins_range):
        fig, [ax1, ax2] = plt.subplots(ncols = 2, nrows = 1, figsize = [25,8])
        ax1.hist(error_df.err_lstm, label = "LSTM Error Distributioin", color = "g", bins = 16, range = (0,1.6))
        ax1.legend()
        ax1.set_xlabel("Mean Absolute Error")
        ax1.set_ylabel("Frequency of errors")
        ax1.grid(which = "both", lw=2)
    
        ax2.hist(error_df.err_sim, label = "PvLib Simulation Error Distributioin", bins = 16, range = (0,1.6))
        ax2.legend()
        ax2.set_xlabel("Mean Absolute Error")
        ax2.set_ylabel("Frequency of errors")
        ax2.grid(which = "both", lw=2)
        plt.savefig(path_hist + "error_histogram_"  + trained_model.name + ".jpg", dpi = 100)

    def plot_line_error_comp(self, error_df, trained_model, results_path):
        
        error_df_lstm = error_df.loc[:,["err_lstm"]]
        error_df_sim = error_df.loc[:,["err_sim"]]
    
        plt.figure(figsize = [80,8])
        plt.plot(error_df_lstm.sort_index(), label = "Error LSTM", c = "b")
        plt.plot(error_df_sim.sort_index(), label = "Error Pvlib", c = "r")
        plt.legend(fontsize = 40)
        plt.title("Hummel HQ Comparison between LSTM and Pvlib")
        plt.savefig(results_path + "error_line_" + trained_model.name +".jpg", dpi = 100)



