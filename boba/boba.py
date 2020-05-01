
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boto3

from ._01_WebScraping import Boba_WebScraping
from ._02_ETL import Boba_ETL

class BobaProjections(Boba_WebScraping, Boba_ETL):

    def __init__(self, year):
        self.s3_client = boto3.client('s3')
        self.bucket = "boba-voglewede"
        self.year = year

    def __repr__(self):
        return "I am a Baseball Projection System!"


    def gather_data(self, source, position_group, year):
        if souce == 'fangraphs_season':
            if position_group == 'hitters':
                print("gather data for FG seasonal hitting data")
                scrape_FG_hitters_season(year)
            elif position_group == 'pitchers':    
                print("gather data for FG seasonal pitching data")
            else: 
                pass
        elif souce == 'statcast_season':
            if position_group == 'hitters':
                print("gather data for statcast seasonal hitting data")
            elif position_group == 'pitchers':    
                print("gather data for statcast seasonal pitching data")
            else: 
                pass