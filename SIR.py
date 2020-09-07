#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 21:53:29 2020

@author: dung
"""

import numpy as np
import pandas as pd
import urllib.request
import json
import sys
from csv import reader
from csv import writer
import matplotlib.pyplot as plt
import argparse

def Get_InitialValues():  
    parser = argparse.ArgumentParser(description='Type in a country')
    parser.add_argument('--country'  , dest = 'country'   , metavar = 'Country CSV format', type = str, nargs= '+', help = 'Country in CSV format', default = 'Vietnam')
    parser.add_argument('--startDate', dest = 'Start_Date', metavar = 'Date in MM/DD/YY format', type = str, nargs= '+', help = 'Date in MM/DD/YY format, default to be 1/22/20', default = '1/22/20')  
    parser.add_argument('--S_0'      , dest = 'S_0'       , metavar = 'S_0', type = int, help = 'initial number of Susceptibles, default = 40000000', default = 40000000 )
    # parser.add_argument('I_0'      , dest = 'I_0'       , metavar = 'I_0', type = int, help = 'initial number of Infectives,   default = 40000000', default = 40000000 )
    # parser.add_argument('R_0'      , dest = 'S_0'       , metavar = 'S_0', type = int, help = 'initial number of Susceptibles, default = 40000000', default = 40000000 )
    args = parser.parse_args()
    return (args.country, args.Start_Date, args.S_0)


def download_data(URL_dict):
    urllib.request.urlretrieve(URL_dict["Confirmed"], './data/Confirmed.csv')
    urllib.request.urlretrieve(URL_dict["Deaths"], './data/Deaths.csv')
    urllib.request.urlretrieve(URL_dict["Recovered"], './data/Recovered.csv')
    

def load_json(json_string):
    #load json into a dictionary
    try:
        with open(json_string, "r") as json_file:
          json_variable = json.load(json_file)
          return json_variable
    except Exception:
        sys.exit("Cannot open JSON file: " + json_string)
        

def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dS_dt   = -beta * S * I
    dI_dt   = beta  * S * I  - gamma* I
    dR_dt   = gamma * I
    
    return ([dS_dt, dI_dt, dR_dt])

def Load_I_0(country, Start_Date):
    df = pd.read_csv('data/Confirmed.csv')
    return df[df['Country/Region'] == country].iloc[0].loc[Start_Date]


def Load_R_0(country, Start_Date):
    df_Deaths = pd.read_csv('data/Deaths.csv')
    df_Recovered = pd.read_csv('data/Recovered.csv')
    Deaths = df_Deaths[df_Deaths['Country/Region'] == country].iloc[0].loc[Start_Date]
    Recovered = df_Recovered[df_Recovered['Country/Region'] == country].iloc[0].loc[Start_Date]
    return (Deaths + Recovered)
    
    
def main():
    country, Start_Date, S_0 = Get_InitialValues()
    I_0 = Load_I_0(country, Start_Date)
    print(I_0)
    R_0 = Load_R_0(country, Start_Date)
    print(R_0)
    
    
    # URL_dict = load_json("data_URL.json") 
    # download_data(URL_dict)
    # df = pd.read_csv('data/Confirmed.csv')
    # df_country = df[df['Country/Region'] == 'Vietnam']
    # ConfirmedForVietnam = df_country.iloc[0].loc['1/22/20':]
    
    # print(ConfirmedForVietnam)    
    # fig, ax = plt.subplots(figsize=(15, 10))
    # ax.set_title('Vietnam')
    # ConfirmedForVietnam.plot(ax=ax)
    # plt.savefig('VietnamConfirmed.png')
    
    

if __name__ == '__main__':
    main()    



    