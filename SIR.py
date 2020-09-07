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
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from datetime import timedelta, datetime

def Get_InitialValues():  
    parser = argparse.ArgumentParser(description='Type in a country')
    parser.add_argument('--country'  , dest = 'country'   , metavar = 'Country CSV format', type = str, nargs= '+', help = 'Country in CSV format', default = 'Vietnam')
    parser.add_argument('--startDate', dest = 'Start_Date', metavar = 'Date in MM/DD/YY format', type = str, nargs= '+', help = 'Date in MM/DD/YY format, default to be 1/22/20', default = '1/23/20')  
    parser.add_argument('--S_0'      , dest = 'S_0'       , metavar = 'S_0', type = int, help = 'initial number of Susceptibles, default = 100000', default = 100000 )
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
        



def Load_I_0(country, Start_Date):
    df = pd.read_csv('data/Confirmed.csv')
    return df[df['Country/Region'] == country].iloc[0].loc[Start_Date]


def Load_R_0(country, Start_Date):
    df_Deaths = pd.read_csv('data/Deaths.csv')
    df_Recovered = pd.read_csv('data/Recovered.csv')
    Deaths = df_Deaths[df_Deaths['Country/Region'] == country].iloc[0].loc[Start_Date]
    Recovered = df_Recovered[df_Recovered['Country/Region'] == country].iloc[0].loc[Start_Date]
    return (Deaths + Recovered)


def Load_Deaths(country, Start_Date):
    df_Deaths = pd.read_csv('data/Deaths.csv')
    return df_Deaths[df_Deaths['Country/Region'] == country].iloc[0].loc[Start_Date:]


def Load_Confirmed(country, Start_Date):
    df_Confirmed = pd.read_csv('data/Confirmed.csv')
    return df_Confirmed[df_Confirmed['Country/Region'] == country].iloc[0].loc[Start_Date:]

def Load_Recovered(country, Start_Date):
    df_Recovered = pd.read_csv('data/Recovered.csv')
    return df_Recovered[df_Recovered['Country/Region'] == country].iloc[0].loc[Start_Date:]
                                                                               
                                                                               
def error(GuessArray, data_Infected, data_Recovered, S_0, I_0, R_0):
    size = len(data_Infected)
    beta, gamma = GuessArray
    def SIR_model(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        dS_dt   = -beta * S * I
        dI_dt   = beta  * S * I  - gamma* I
        dR_dt   = gamma * I    
        return ([dS_dt, dI_dt, dR_dt])
    
    
    solution = solve_ivp(SIR_model, [0, size], [S_0,I_0,R_0], t_eval=np.arange(0, size, 1), vectorized=True)
    l1 = np.sqrt(np.mean((solution.y[1] - data_Infected)**2))
    l2 = np.sqrt(np.mean((solution.y[2] - data_Recovered)**2))
    alpha = 0.1
    returnvalue = alpha * l1 + (1 - alpha) * l2
    return alpha * l1 + (1 - alpha) * l2
    
    
    
def extend_index(index):
    values = index.values
    current = datetime.strptime(values[-1], '%m/%d/%y')
    predictRange = 300
    while len(values) < predictRange:
        current = current + timedelta(days= 1)
        values  = np.append(values, current.strftime('%m/%d/%y'))
    return values


def predict(beta, gamma, infected, recovered, death, country, S_0, I_0, R_0, new_index):        
        size = len(new_index)
        def SIR_model(t, y):
            S, I, R = y
            dS_dt   = -beta * S * I
            dI_dt   = beta  * S * I  - gamma* I
            dR_dt   = gamma * I    
            return ([dS_dt, dI_dt, dR_dt])
        
        
        extended_infected = np.concatenate((infected.values, [None] * (size - len(infected.values))))
        extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
        extended_death = np.concatenate((death.values, [None] * (size - len(death.values))))
        return extended_infected, extended_recovered, extended_death, solve_ivp(SIR_model, [0, size], [S_0,I_0,R_0], t_eval=np.arange(0, size, 1))
                                       
    
     
    
def main():
    country, Start_Date, S_0 = Get_InitialValues()
    I_0 = Load_I_0(country, Start_Date)
    print(I_0)
    R_0 = Load_R_0(country, Start_Date)
    print(R_0)
    #change to test initials
    I_0 = 2
    R_0 = 10
    data_Deaths = Load_Deaths(country, Start_Date)
    data_Confirmed = Load_Confirmed(country, Start_Date)
    data_Recovered = Load_Recovered(country, Start_Date)
    data_Infected  = data_Confirmed - data_Recovered - data_Deaths
    
    
    optimal = minimize(error, [0.001, 0.001], args=(data_Infected, data_Recovered, S_0, I_0, R_0), method='L-BFGS-B', bounds=[(0.00000001, 0.4), (0.00000001, 0.4)])
    print(optimal)
    beta, gamma = optimal.x
    new_index = extend_index(data_Confirmed.index)
    extended_infected, extended_recovered, extended_death, prediction = predict(beta, gamma, data_Infected, data_Recovered, data_Deaths, country, S_0, I_0, R_0, new_index)
    df = pd.DataFrame({'Infected data': extended_infected, 'Recovered data': extended_recovered, 'Death data': extended_death, 'Susceptible': prediction.y[0], 'Infected': prediction.y[1], 'Recovered': prediction.y[2]}, index=new_index)
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_title(country)
    df.plot(ax=ax)
    print(f"country:{country}, beta:{beta:.8f}, gamma:{gamma:.8f}, r_0:{(beta/gamma):.8f}")
    fig.savefig("{}.png".format(country))
    
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



    