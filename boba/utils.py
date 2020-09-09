import pandas as pd
import boto3
from botocore.exceptions import ClientError
import numpy as np
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore")

class Boba_Utils():

    def __init__(self):
        pass
    

    def upload_file(self, file_name, object_name=None):
        # If S3 object_name was not specified, use file_name
        if object_name is None:
            object_name = file_name
        try:
            self.s3_client.upload_file(file_name, self.bucket, object_name)
        except ClientError as e:
            logging.error(e)
            return False
        print("Success")
 
    def clear_files(self):
        path = 'boba/data/scoring/evaluation_hitters_2019.csv'
        if os.path.exists(path):
            os.remove(path)

        path = 'boba/data/scoring/evaluation_RP_2019.csv'
        if os.path.exists(path):
            os.remove(path)
        
        path = 'boba/models/grid_search_result.csv'
        if os.path.exists(path):
            os.remove(path)

        path = 'boba/models/model_results.csv'
        if os.path.exists(path):
            os.remove(path)

    def agg_position_h(row):
        if 'C' in row['position'] :
            return 'C'
        elif '2B' in row['position']:
            return '2B' 
        elif 'SS' in row['position']:
            return 'SS' 
        elif 'OF' in row['position']:
            return 'OF' 
        elif '1B' in row['position']:
            return '1B' 
        elif '3B' in row['position']:
            return '3B' 
        elif 'RP' in row['position']:
            return 'P' 
        elif 'SP' in row['position']:
            return 'P' 
            
        else:
            return row['position']

    
    def agg_position_p(row):
        if (row['GS']>5) & ((row['SV']+row['HLD']) < 5) :
            return 'SP'
        elif row['IP']>15:
            return 'RP' 
        else:
            return 'NonP'

    def agg_position_p_scoring(row):
        if (row['Name']=='Chad Green'):
            return 'RP'
        elif (row['GS']>3) :
            return 'SP'
        else:
            return 'RP'