# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 17:26:25 2023

@author: SaeidNikbakht
"""

from DataPreprocessor import DataPreprocessor
from ModelCreator import ModelCreator
from ModelTrainer import ModelTrainer
from DataCleaner import DataCleaner
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from datetime import datetime, timedelta
from TBDatabaseConnector import TBDatabaseConnector
import os
import json

class Executer:
    
    def __init__(self, MLConf, start_dt, end_dt):
        self.input_data = MLConf["input_data"]
        self.model_data = MLConf["model_data"]
        self.start_dt = start_dt
        self.end_dt = end_dt
        
    def get_clean_data_individual(self, data_info):
        if data_info["data_storage"] == "TB":
            tb_url = data_info["data_sources"]["TB_info"]["tb_url"]
            tb_auth = data_info["data_sources"]["TB_info"]["tb_auth"]
            entity_id = data_info["data_sources"]["TB_info"]["entity_id"]
            data_key = data_info["data_sources"]["TB_info"]["data_key"]
            tb_db_connector = TBDatabaseConnector(tb_url, tb_auth)
    
            start_ts = int(self.start_dt.timestamp()*1000)
            end_ts = int(self.end_dt.timestamp()*1000)
            try:
                raw_data = tb_db_connector.getDataframeFromDevice(entity_id,[data_key],start_ts,end_ts)
            except:
                raw_data = tb_db_connector.getDataframeFromAsset(entity_id,[data_key],start_ts,end_ts)
                
        elif data_info["data_storage"] == "CSV":
            local_directory = data_info["data_sources"]["local_directory"]
            raw_data = pd.read_csv(local_directory, index_col = "Time", parse_dates = True).loc[self.start_dt:self.end_dt,:]
        # cleaning the data
        data_cleaner_dict = data_info["cleaner_dict"]
        data_cleaner = DataCleaner(raw_data, data_cleaner_dict)
        data_clean_resampled = data_cleaner.data_prep()
        return data_clean_resampled
    
    
    def get_clean_data_total_with_same_index(self):
        clean_data_total_list = []
        for _, data_info in self.input_data.items():
            data_clean_resampled = self.get_clean_data_individual(data_info)
            clean_data_total_list.append(data_clean_resampled)
        clean_data_total_df = pd.concat(clean_data_total_list, axis = 1)
        clean_data_total_df.dropna(inplace= True)
        clean_data_total_df.index.name = "Time"
        return clean_data_total_df
    
    
    
    def get_preprocessed_data_individual(self, clean_data_resampled, splitter_dict, scaler_dict):
        data_preprocessor = DataPreprocessor(clean_data_resampled, splitter_dict)
        X_train_scaled_3d, X_test_scaled_3d, y_train, y_test = data_preprocessor.train_test_provider(scaler_dict)
        return X_train_scaled_3d, X_test_scaled_3d, y_train, y_test
    
    
    
    def get_preprocessed_data_total(self, clean_data_total_df):
        
        X_train_list = []
        X_test_list = []
        for data_name, data_info in self.input_data.items():
            
            resolution = self.input_data[data_name]["cleaner_dict"]["resolution"]
            col_name = data_name + "_" + resolution
            
            clean_data_resampled = clean_data_total_df.loc[:,[col_name]]
            splitter_dict = self.input_data[data_name]["splitter_dict"]
            scaler_dict = self.input_data[data_name]["scaler_dict"]
            isLabel = self.input_data[data_name]["isLabel"]
            
            if isLabel:
                X_train_scaled_3d, X_test_scaled_3d, y_train, y_test = self.get_preprocessed_data_individual(clean_data_resampled, splitter_dict, scaler_dict)
            else:
                X_train_scaled_3d, X_test_scaled_3d, _, _ = self.get_preprocessed_data_individual(clean_data_resampled, splitter_dict, scaler_dict)
    
            X_train_list.append(X_train_scaled_3d)
            X_test_list.append(X_test_scaled_3d)
    
        X_train_total = np.concatenate(X_train_list, axis = 2)
        X_test_total = np.concatenate(X_test_list, axis = 2)
        
        return X_train_total, X_test_total, y_train, y_test
    
    
    
    def create_and_train_model(self, X_train_total, y_train):
        initializer_dict = self.model_data["initializer_dict"]
        compiler_dict = self.model_data["compiler_dict"]
        model_version = self.model_data["model_version"]
        callbacks_dict = self.model_data["callbacks_dict"]
        trainer_dict = self.model_data["trainer_dict"]
        pretrained_model_directory = self.model_data["pretrained_model_directory"]
        if pretrained_model_directory:
            new_model_name = initializer_dict["model_name"]
            raw_model = load_model(pretrained_model_directory)
            raw_model._name = new_model_name
        else:
            model_creator = ModelCreator(initializer_dict, compiler_dict)
            if model_version == 1:
                raw_model = model_creator.multi_step_model_1()
            elif model_version == 2:
                raw_model = model_creator.multi_step_model_2()
            elif model_version == 3:
                raw_model = model_creator.multi_step_model_3()
            elif model_version == 4:
                raw_model = model_creator.multi_step_model_4()
            elif model_version == 5:
                raw_model = model_creator.multi_step_model_5()
            elif model_version == 6:
                raw_model = model_creator.multi_step_model_6()
            elif model_version == 7:
                raw_model = model_creator.multi_step_model_7()
    

        model_trainer = ModelTrainer(raw_model, callbacks_dict, trainer_dict)
        trained_model, history = model_trainer.train_model(X_train_total, y_train)
        
        # saving the model plot
        model_path = callbacks_dict["model_path"]
        model_name = trained_model.name
        plot_model(
            trained_model,
            to_file = model_path + model_name + '.jpg',
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=True,
            dpi=100,
        )
        return trained_model, history
    
    def create_repo_if_not_exist(self, path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
            
    def data_saver(self, trained_model, history, X_test_total, y_test, clean_data_total_df):
        
        # saving history
        history_path = self.model_data["data_saver_dict"]["history_path"]
        self.create_repo_if_not_exist(history_path)
        with open(history_path+"History_" + trained_model.name + ".pickle", "wb") as pickle_file:
            pickle.dump(history.history, pickle_file)
        
        # saving clean_data
        clean_data_path = self.model_data["data_saver_dict"]["clean_data_path"]
        self.create_repo_if_not_exist(clean_data_path)
        clean_data_total_df.to_csv(clean_data_path+"df_" + trained_model.name + ".csv")
        
        # saving test_data
        test_data_path = self.model_data["data_saver_dict"]["test_data_path"]
        self.create_repo_if_not_exist(test_data_path)
        with open(test_data_path+"X_test_np_" + trained_model.name + ".pickle", "wb") as pickle_file:
            pickle.dump(X_test_total, pickle_file)
        with open(test_data_path+"y_test_np_" + trained_model.name + ".pickle", "wb") as pickle_file:
            pickle.dump(y_test, pickle_file)
        
    
    def execute_ml(self):
        # clean resampled data
        clean_data_total_df = self.get_clean_data_total_with_same_index()
        
        # extra cleaning
        col0 = clean_data_total_df.columns[0]
        col1 = clean_data_total_df.columns[1]
        clean_data_total_df.loc["09-Dec-2022":"20-Dec-2022",col0] = clean_data_total_df.loc["09-Dec-2022":"20-Dec-2022",col1]
        clean_data_total_df.loc["11-Nov-2022":"14-Nov-2022",col0] = clean_data_total_df.loc["11-Nov-2022":"14-Nov-2022",col1]
        
        # preprocessed data
        X_train_total, X_test_total, y_train, y_test = self.get_preprocessed_data_total(clean_data_total_df) 
        trained_model, history = self.create_and_train_model(X_train_total, y_train)
        self.data_saver(trained_model, history, X_test_total, y_test, clean_data_total_df)
        
        return trained_model, history, X_test_total, y_test, clean_data_total_df

        
# testing the executer
if __name__ == "__main__":
    
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days = 600)
    
    with open("Assets\HarrerTest3\MLConf\MLConf_test3.json", "r") as json_file:
        MLConf = json.load(json_file)
    executer = Executer(MLConf, start_dt, end_dt)
    trained_model, history, X_test_total, y_test, clean_data_total_df = executer.execute_ml()
    executer.data_saver(trained_model, history, X_test_total, y_test, clean_data_total_df)
