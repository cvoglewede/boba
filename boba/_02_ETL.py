import pandas as pd
import boto3
from botocore.exceptions import ClientError
import numpy as np
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore")

class Boba_ETL():

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
 
    def gather_FG_hitters(self):
        path = 'data/hitters/raw/fangraphs/season/'
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                if '.csv' in file:
                    files.append(os.path.join(r, file))
        for f in files:
            print(f)    
        data = pd.DataFrame()
        for file in tqdm(files):
            df = pd.read_csv(file,index_col=0)
            headings = df.columns
            data = data.append(df)
        data.columns = headings
        data.to_csv('data/hitters/interim/fangraphs/master_fg.csv')
        return data