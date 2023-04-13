# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 15:50:47 2022

@author: SaeidNikbakht
"""

import numpy as np
import pandas as pd
import os
import pickle
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
np.random.seed(1234)
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class DataPreprocessor:
    
    def __init__(self, df, splitter_dict, scaler_dict):
        self.df = df
        self.past_steps = splitter_dict["past_steps"]
        self.future_steps = splitter_dict["future_steps"]
        self.test_size = splitter_dict["test_size"]
        self.random_state = splitter_dict["random_state"]
        self.shuffle_flag = splitter_dict["shuffle_flag"]
        self.scaler_type = scaler_dict["scaler_type"]
        self.scaler_path = scaler_dict["scaler_path"]
        
    def prepare_solution_domain_matrix(self):
        # Creating the solution domain matrix
        sd_matrix = self.df.copy()
        column_name = self.df.columns[0]
        for i in range(1, self.past_steps+self.future_steps):
            sd_matrix[i-self.past_steps+1] = sd_matrix[column_name].shift(periods = -i)
    
        sd_matrix = sd_matrix.dropna(how = "any")
        new_index = sd_matrix.index[self.past_steps-1:-1*self.future_steps-1]
        sd_matrix = sd_matrix.iloc[:-1*(self.past_steps+self.future_steps),:]
        sd_matrix.index = new_index
        
        sd_matrix.rename(columns = {column_name:-self.past_steps+1}, inplace = True)
        return sd_matrix

    def data_splitter(self, sd_matrix):
        features = sd_matrix.loc[:,:0]
        labels = sd_matrix.loc[:,1:]
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = self.test_size, random_state = self.random_state, shuffle = self.shuffle_flag)
        return X_train, X_test, y_train, y_test

    def scale_train_data(self, X_train, scaler_type):
        if scaler_type == "normal":
            scaler = MinMaxScaler()
        elif scaler_type == "standard":
            scaler = StandardScaler()
        else:
            print("the scaler type is not correct! please chose between normal and standard")
        X_train_scaled_step1 = X_train.values.ravel().reshape(-1,1)
        X_train_scaled_step2 = scaler.fit_transform(X_train_scaled_step1)
        X_train_scaled_step3 = X_train_scaled_step2.reshape(X_train.shape)
        X_train_scaled_3d = np.expand_dims(X_train_scaled_step3,2)
        return scaler, X_train_scaled_3d
    
    def scale_test_data(self, X_test, scaler):
        X_test_scaled_step1 = X_test.values.flatten().reshape(-1,1)
        X_test_scaled_step2 = scaler.transform(X_test_scaled_step1)
        X_test_scaled_step3 = X_test_scaled_step2.reshape(X_test.shape)
        X_test_scaled_3d = np.expand_dims(X_test_scaled_step3,2)
        return X_test_scaled_3d

    def save_scaler(self, scaler):
        with open(self.scaler_path + self.df.columns[0] + "_" + self.scaler_type + "_scaler" +'_p{}_f{}.pickle'.format(self.past_steps, self.future_steps), 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)
            
    def create_repo_if_not_exist(self, path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
            
    def run_data_preprocessor(self):
        sd_matrix = self.prepare_solution_domain_matrix()
        X_train, X_test, y_train, y_test = self.data_splitter(sd_matrix)
        scaler, X_train_scaled_3d = self.scale_train_data(X_train, self.scaler_type)
        X_test_scaled_3d = self.scale_test_data(X_test, scaler)
        if self.scaler_path:
            self.create_repo_if_not_exist(self.scaler_path)
            self.save_scaler(scaler)
        return X_train_scaled_3d, X_test_scaled_3d, y_train, y_test
    
# Test the class
if __name__ == "__main__":
    
    real_pv_15min = pd.read_csv("datasets/Prod_real.csv", index_col = "Time", parse_dates = True)
    pred_pv_15min = pd.read_csv("datasets/Prod_sim.csv", index_col = "Time", parse_dates = True).loc[real_pv_15min.index]

    splitter_dict = {"past_steps":192,
                     "future_steps":16,
                     "test_size":0.2,
                     "random_state":1234,
                     "shuffle_flag":True}
    
    scaler_dict = {"scaler_type":"standard",
                   "scaler_path":"scalers"}
    
    data_preprocessor = DataPreprocessor(real_pv_15min, splitter_dict, scaler_dict)
    X_train_scaled_3d, X_test_scaled_3d, y_train, y_test = data_preprocessor.run_data_preprocessor()
