import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sktime.split import temporal_train_test_split
from dataclasses import dataclass
from datasets import load_dataset

from src.components.data_transformation import DataTranformation

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            ds = load_dataset("Sujal08/cip-forecasting-Macroeconomic_Indicators_dataset")
            df = ds['train'].to_pandas()

            target_columns = [
                'Consumer Price Index  (2012=100)',
                'Consumer Price Index for Agricultural Labourer',
                'Consumer Price Index for Rural Labourer',
                'Consumer Price Index for Food and Beverages'
            ]

            logging.info("Generating lag features for targets")

            for col in target_columns:
                for lag in range(1, 7):  # lags 1 to 6
                    df[f"{col}_lag_{lag}"] = df[col].shift(lag)

            lagged_cols = [f"{col}_lag_{lag}" for col in target_columns for lag in range(1, 7)]
            untouched_cols = target_columns + lagged_cols + ['Period']

            logging.info("Applying rolling mean to input features")

            input_cols = [col for col in df.columns if col not in untouched_cols]
            rolling_means = df[input_cols].rolling(window=6).mean()

            rolling_means.columns = [f"rolling_mean_{col}" for col in rolling_means.columns]
            df_transformed = pd.concat([rolling_means, df[untouched_cols]], axis=1)

            df_transformed = df_transformed.dropna().copy()

            df = df_transformed.copy()

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info('Train test split initiated')
            future_context = 12
            train_set,test_set=temporal_train_test_split(df,test_size=future_context)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingesion of the data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)



if __name__=='__main__':
    obj=DataIngestion()
    train_data, test_data=obj.initiate_data_ingestion()

    data_transformation = DataTranformation()
    data_transformation.initiate_data_transformation(train_data, test_data)