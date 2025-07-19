import os
import sys
from dataclasses import dataclass

import pandas  as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransfromationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTranformation:
    def __init__(self):
        self.data_transformation_config=DataTransfromationConfig()

    def get_data_transformer_object(self):
        try:
            columns = ['rolling_mean_Index of Industrial Production',
            'rolling_mean_Wholesale Price Index (2011-12=100)',
            'rolling_mean_Net Foreign Direct Investment (US $ Million)',
            'rolling_mean_FCNR(B)-Foreign Currency Non-Resident (Banks) (US $ Million)',
            'rolling_mean_Foreign Trade Exports Total (US $ Million)',
            'rolling_mean_Foreign Trade Imports Total (US $ Million)',
            'rolling_mean_Foreign Trade Balance Total (US $ Million)',
            'rolling_mean_Market Borrowing SG Gross Amount Raised (in Crore)',
            'rolling_mean_Exchange Rate of Indian Rupee to US Dollar (Month End)',
            'rolling_mean_Commercial Paper (Amount Outstanding) (in Crore)',
            'rolling_mean_Direct Investment to India (US $ Million)',
            'rolling_mean_Foreign Direct Investment By India (US $ Million)',
            'rolling_mean_Net Portfolio Investment (US $ Million)',
            'rolling_mean_Total Investment Inflows (US $ Million)',
            'rolling_mean_External Commercial Borrowings Registrations (US $ Million)',
            'Consumer Price Index  (2012=100)_lag_1',
            'Consumer Price Index  (2012=100)_lag_2',
            'Consumer Price Index  (2012=100)_lag_3',
            'Consumer Price Index  (2012=100)_lag_4',
            'Consumer Price Index  (2012=100)_lag_5',
            'Consumer Price Index  (2012=100)_lag_6',
            'Consumer Price Index for Agricultural Labourer_lag_1',
            'Consumer Price Index for Agricultural Labourer_lag_2',
            'Consumer Price Index for Agricultural Labourer_lag_3',
            'Consumer Price Index for Agricultural Labourer_lag_4',
            'Consumer Price Index for Agricultural Labourer_lag_5',
            'Consumer Price Index for Agricultural Labourer_lag_6',
            'Consumer Price Index for Rural Labourer_lag_1',
            'Consumer Price Index for Rural Labourer_lag_2',
            'Consumer Price Index for Rural Labourer_lag_3',
            'Consumer Price Index for Rural Labourer_lag_4',
            'Consumer Price Index for Rural Labourer_lag_5',
            'Consumer Price Index for Rural Labourer_lag_6',
            'Consumer Price Index for Food and Beverages_lag_1',
            'Consumer Price Index for Food and Beverages_lag_2',
            'Consumer Price Index for Food and Beverages_lag_3',
            'Consumer Price Index for Food and Beverages_lag_4',
            'Consumer Price Index for Food and Beverages_lag_5',
            'Consumer Price Index for Food and Beverages_lag_6'
            ]
            trans_pipeline= Pipeline(
                steps=[
                    ('scaler', StandardScaler())
                ]
            )

            logging.info('Standard scaling completed')

            preprocessor = ColumnTransformer(
                [
                    ('trans_pipeline',trans_pipeline,columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj=self.get_data_transformer_object()

            target_columns = [
            'Consumer Price Index  (2012=100)',
            'Consumer Price Index for Agricultural Labourer',
            'Consumer Price Index for Rural Labourer',
            'Consumer Price Index for Food and Beverages'
            ]

            columns = ['rolling_mean_Index of Industrial Production',
            'rolling_mean_Wholesale Price Index (2011-12=100)',
            'rolling_mean_Net Foreign Direct Investment (US $ Million)',
            'rolling_mean_FCNR(B)-Foreign Currency Non-Resident (Banks) (US $ Million)',
            'rolling_mean_Foreign Trade Exports Total (US $ Million)',
            'rolling_mean_Foreign Trade Imports Total (US $ Million)',
            'rolling_mean_Foreign Trade Balance Total (US $ Million)',
            'rolling_mean_Market Borrowing SG Gross Amount Raised (in Crore)',
            'rolling_mean_Exchange Rate of Indian Rupee to US Dollar (Month End)',
            'rolling_mean_Commercial Paper (Amount Outstanding) (in Crore)',
            'rolling_mean_Direct Investment to India (US $ Million)',
            'rolling_mean_Foreign Direct Investment By India (US $ Million)',
            'rolling_mean_Net Portfolio Investment (US $ Million)',
            'rolling_mean_Total Investment Inflows (US $ Million)',
            'rolling_mean_External Commercial Borrowings Registrations (US $ Million)',
            'Consumer Price Index  (2012=100)_lag_1',
            'Consumer Price Index  (2012=100)_lag_2',
            'Consumer Price Index  (2012=100)_lag_3',
            'Consumer Price Index  (2012=100)_lag_4',
            'Consumer Price Index  (2012=100)_lag_5',
            'Consumer Price Index  (2012=100)_lag_6',
            'Consumer Price Index for Agricultural Labourer_lag_1',
            'Consumer Price Index for Agricultural Labourer_lag_2',
            'Consumer Price Index for Agricultural Labourer_lag_3',
            'Consumer Price Index for Agricultural Labourer_lag_4',
            'Consumer Price Index for Agricultural Labourer_lag_5',
            'Consumer Price Index for Agricultural Labourer_lag_6',
            'Consumer Price Index for Rural Labourer_lag_1',
            'Consumer Price Index for Rural Labourer_lag_2',
            'Consumer Price Index for Rural Labourer_lag_3',
            'Consumer Price Index for Rural Labourer_lag_4',
            'Consumer Price Index for Rural Labourer_lag_5',
            'Consumer Price Index for Rural Labourer_lag_6',
            'Consumer Price Index for Food and Beverages_lag_1',
            'Consumer Price Index for Food and Beverages_lag_2',
            'Consumer Price Index for Food and Beverages_lag_3',
            'Consumer Price Index for Food and Beverages_lag_4',
            'Consumer Price Index for Food and Beverages_lag_5',
            'Consumer Price Index for Food and Beverages_lag_6'
            ]    

            input_feature_train_df = train_df.drop(columns=target_columns, axis =1)
            target_feature_train_df = train_df[target_columns]

            input_feature_test_df = test_df.drop(columns=target_columns, axis=1)
            target_feature_test_df = test_df[target_columns]

            logging.info('Applying preprocessing object  on training dataframe and testing dataframe.')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info('Saved preprocessing object')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)