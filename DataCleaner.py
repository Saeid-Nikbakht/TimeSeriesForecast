# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 17:22:06 2023

@author: SaeidNikbakht
"""

import numpy as np
import pandas as pd
from datetime import timedelta

class DataCleaner:
    
    def __init__(self, raw_data:pd.DataFrame, cleaner_dict:dict) -> None:
        self.raw_data = raw_data
        self.value_type = cleaner_dict["value_type"]
        self.resolution = cleaner_dict["resolution"]
        self.column_name = cleaner_dict["column_name"]
        self.max_anomaly_margin = cleaner_dict["max_anomoly_margin"]
        self.min_anomaly_margin = cleaner_dict["min_anomoly_margin"]
        self.agg_func = cleaner_dict["agg_func"]
        self.hours_modifier = cleaner_dict["hours_modifier"]
        
    def cum2abs(self, df:pd.DataFrame):
        df_temp = df.copy()
        col_name = df_temp.columns[0]
        df_temp[col_name] = df_temp[col_name].diff(periods = 1)
        return df_temp

    def remove_outliers(self, df:pd.DataFrame) -> pd.DataFrame:
        df_temp = df.copy()
        for col in list(df_temp):
            desc_df = df_temp.describe([self.max_anomaly_margin/100, self.min_anomaly_margin/100])
            max_bound = desc_df.loc["{}%".format(self.max_anomaly_margin), col]
            min_bound = desc_df.loc["{}%".format(self.min_anomaly_margin), col]
            df_temp.loc[df_temp[col]>max_bound, col] = max_bound
            df_temp.loc[df_temp[col]<min_bound, col] = min_bound
        return df_temp 
    
    def data_prep(self) -> pd.DataFrame:
        if self.value_type == "cum":
            self.raw_data.sort_index(inplace = True)
            abs_data = self.cum2abs(self.raw_data)
        else:
            abs_data = self.raw_data.copy()
        abs_data_modified_hours = abs_data.copy()
        abs_data_modified_hours.index = abs_data.index - timedelta(hours = self.hours_modifier)
        abs_data_1min = abs_data_modified_hours.resample("1min").agg(self.agg_func).interpolate(method="linear")
        cln_data_1min = self.remove_outliers(abs_data_1min)
        cln_data_resampled = np.round(cln_data_1min.resample(self.resolution).agg(self.agg_func).sort_index(),10)
        cln_data_resampled.columns = [self.column_name + "_" + self.resolution]
        return cln_data_resampled

# Test the class
if __name__ == "__main__":
    
    real_pv_15min = pd.read_csv("datasets/Prod_real.csv", index_col = "Time", parse_dates = True)
    pred_pv_15min = pd.read_csv("datasets/Prod_sim.csv", index_col = "Time", parse_dates = True).loc[real_pv_15min.index]

    cleaner_dict = {"value_type": "abs",
                    "resolution": "15min",
                    "column_name": "pv_sim",
                    "max_anomoly_margin": 99.9,
                    "min_anomoly_margin": 0.1,
                    "agg_func": "sum",
                    "hours_modifier":0}
    
    data_cleaner = DataCleaner(real_pv_15min, cleaner_dict)
    cln_data_resampled = data_cleaner.data_prep()
    print(cln_data_resampled.head())