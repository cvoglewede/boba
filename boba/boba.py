
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

