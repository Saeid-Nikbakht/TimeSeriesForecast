# -*- coding: utf-8 -*-
"""
Created on April 01 2023

@author: SaeidNikbakht
"""

import json
from datetime import datetime, timedelta
from Executer import Executer

def main(MLConf, start_dt, end_dt):
    executer = Executer(MLConf, start_dt, end_dt)
    trained_model, history, X_test_total, y_test, clean_data_total_df = executer.execute_ml()
    executer.data_saver(trained_model, history, X_test_total, y_test, clean_data_total_df)

# test the main function
if __name__ == "__main__":
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days = 600)
    with open("MLConf_test.json", "r") as json_file:
        MLConf = json.load(json_file)
    main(MLConf, start_dt, end_dt)