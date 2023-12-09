import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
#from sklearn.model_selection import train_test_split
from dataclasses import dataclass
#from src.components.data_transformation import DataTransformation

# Initialize the Data Injestion Configuration

@dataclass
class DatatInjestionconfig:
    train_data_path:str = os.path.join('artifacts','Train.csv')
    test_data_path:str = os.path.join('artifacts','Test.csv')
    #raw_data_path:str = os.path.join('artifacts','raw_data.csv')


#creating class for data injestion
class DataInjestion:
    def __init__(self) :
        self.injestion_config =DatatInjestionconfig()

    def initiate_data_injestion(self):
        logging.info('Data injestion method started')

        try:
            df = pd.read_csv(os.path.join('notebook/data','Train.csv'))
            logging.info("Train Dataset read as pandas Dataframe")
            os.makedirs(os.path.dirname(self.injestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.injestion_config.train_data_path,index= False, header=True)

            df= pd.read_csv(os.path.join('notebook/data','Test.csv'))
            logging.info("Test Dataset read as pandas Dataframe")
            df.to_csv(self.injestion_config.test_data_path,header=True,index= False)


            return (
                self.injestion_config.train_data_path,
                self.injestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Exception occured at Data Injestion Stage")
            raise CustomException(e,sys)    
        


if __name__=='__main__':
    obj = DataInjestion()
    train_data,test_data = obj.initiate_data_injestion()


