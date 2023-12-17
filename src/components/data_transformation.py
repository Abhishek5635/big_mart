import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

#Create prepocessor pickle file to save the data in serilized byte format
@dataclass
class DataTransformationConfig:
    """
    Creating Prepocessor pickle file path
    """
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pickle')


class DataTransformation:
    def __init__(self):
        """
        Create obj of the data transformation config class
        """
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformation_object(self,train_data_path):

        """
        Feature engineering 

        Args: train_data_path
        Returns: Preprocessor object

        """
        try:
            logging.info("Data Transformation intiated")
            train_df = pd.read_csv(train_data_path)
            #train_df.drop(columns=[],index=True)
            categorical_col = train_df.select_dtypes(include='object').columns.to_list()
            numerical_col = train_df.select_dtypes(exclude='object').columns.to_list()
 
            
            Item_Fat_Content_categories = list(train_df['Item_Fat_Content'].unique())
            Item_Type_categories = list(train_df['Item_Type'].unique())
            Outlet_Size_categories = list(train_df['Outlet_Size'].unique())
            Outlet_Location_Type_categories = list(train_df['Outlet_Location_Type'].unique())
            Outlet_Type_categoreis = list(train_df['Outlet_Type'].unique())

            logging.info("Pipeline Intiated")

            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy='mean')),
                    ('scalar',StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="most_frequent")),
                    ('ordinalencoder',OrdinalEncoder(categories=[Item_Fat_Content_categories,Item_Type_categories,
                                                                 Outlet_Size_categories,Outlet_Location_Type_categories,
                                                                 Outlet_Type_categoreis])),
                    ('scalar',StandardScaler())
                                                                 ]
                    )
            
            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_col),
                ('cat_pipeline',cat_pipeline,categorical_col)
                ]
                )
            
            return preprocessor
            logging.info("Pipeline Completed")
        
        
        except Exception as e:
            logging.info('Exception occoured at DataTransformation')
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self,train_path,test_path):
        """
        Args: train_data_path, test_data_path

        Returns: train_arr, test_arr, prepocessor_obj_file_path

        """

        try :
            #reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test is completed")
            logging.info(f"Train Dataframe Head : \n {train_df.head().to_string()}")
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformation_object(train_path)

            target_col = "Item_Outlet_Sales"
            drop_col = [target_col,'Item_Identifier','Outlet_Identifier']

            input_feature_train_df = train_df.drop(columns=drop_col,axis = 1)
            target_feature_train_df = train_df[target_col]

            input_feature_test_df = test_df.drop(columns=drop_col,axis=1)
            target_feature_test_df = test_df[target_col]

            # Transformatting using preprocessor obj
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applied preprocessing object on training and testing dataset")

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]


            # Save the preprocessing object as pickle file
            save_object(
                file_path =self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
                )
            
            logging.info("preprocessor pickel file is saved")
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
                )
        except Exception as e:
            logging.info("Exception Occoured in the initiate_datatranformation")
            raise(e,sys)
        
    