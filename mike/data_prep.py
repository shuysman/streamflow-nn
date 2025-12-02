#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:03:40 2024

@author: mt
"""

import numpy as np
import csv, os, urllib
import urllib.request
#import matplotlib.pyplot as plt
import datetime
from scipy import stats
import pandas as pd

web = False

def tryfloat(v):
    try:
        r = float(v)
    except:
        r = np.nan
    return r

def fixlen(s):
    s = str(s)
    if len(s) == 1: r = '0' + s
    else: r = s
    return r

def parse_timestamp(ts):
    ts = ts.strip()
    pts = ts.split('-')
    year = pts[0]
    month = pts[1]
    day = pts[2]
    dt = datetime.datetime(year = int(year), month = int(month), day = int(day))
    return year, month, day, dt

def read_stream(f):
    global stop, start
    infile = open(f, 'rt')
    reader = csv.reader(infile)
    cfs = []
    years = []
    months = []
    days = []
    dts = []
    firstval = False
    d = {}
    for line in reader:
        try:
            x = float(line[1])
        except:
            continue
        timestamp = line[0]
        year, month,day, dt = parse_timestamp(timestamp)
        if dt < start : continue
        cfs.append(tryfloat(line[1]))
        years.append(year)
        months.append(month)
        days.append(day)
        dts.append(dt)
        if dt >= stop: break
    d['cfs'] = np.array(cfs)
    d['year'] = np.array(years)
    d['month'] = np.array(months)
    d['day'] = np.array(days)
    d['dt'] = np.array(dts)
    return d
        

def moving_variance(x, N):
    y = np.zeros((len(x),))
    left_offset =  N-1
    #right_offset = N
    for ctr in range(len(x)):
         chunk = np.array(x[ctr-left_offset:ctr+1])
         if len(chunk) < N :y[ctr] = np.nan
         else:y[ctr] = stats.tvar(chunk)
         
    return y

def n_lags(d,n, n2,key):
    # n = number of items in rolling sum, n2 = number of items in iterated lag values
    one_day = datetime.timedelta(days = 1)
    today = d['dt'][-1]
    keep_lag = n2-1
    s = d[key]
    for j in range(1,n2):
        today = today + one_day
        month = fixlen(today.month)
        year = str(today.year)
        day = fixlen(today.day)
        d['dt'] = np.append(d['dt'], today)
        d['day'] = np.append(d['day'], day)
        d['month'] = np.append(d['month'], day)
        d['year'] = np.append(d['year'], day)
        if j != keep_lag : continue
        this_key = 'cfslag{j}'.format(j = j)
        front_fill = np.array([np.nan] * j)
        end_fill = [np.nan] * (n2-j-1)
        d[this_key] = np.insert(s,0,front_fill)
        d[this_key] = np.array(list(d[this_key]) + end_fill)
        
    d[key] = np.array(list(d[key]) + [np.nan] * (n2-1))
    deltas = d[this_key][1:] - d[this_key][:-1]
    d['lagged_delta'] = np.insert(deltas, 0, np.nan)    
    second_d = d['lagged_delta'][1:] - d['lagged_delta'][:-1]
    second_d = np.insert(second_d, 0, np.nan)
    d['second_d'] = second_d
    lagged_delta_variance = moving_variance(d['lagged_delta'],n2)
    d['lagged delta_variance'] = lagged_delta_variance
    d['chaos_index'] = second_d * d['lagged delta_variance']
    for key in d:
        d[key] = d[key][n2*2:]
    return d

def get_new_wb_data(wb_link):
    global end_wb
    
    wb_dl_link = wb_link.format(y2 = this_year)
    print(wb_dl_link)
    urllib.request.urlretrieve(wb_dl_link, 'new_wb_data.csv')
    outfile = open('screened_wb.csv','w')
    infile = open('new_wb_data.csv')
    i = 1
    header_found = False
    for line in infile:
        if i <= 5:
            i += 1
            continue
        if header_found == False:
            header = line
            header_found = True
            outfile.write(header)
            print('**',header)
        pl = line.split(',')
       
        try:
            x = float(pl[0].strip())
        except:
            i+=1
            continue
        try:
            this_date = pl[1].strip()
            psd = this_date.split('/')
        except:
            continue
        today = datetime.datetime(year = int(psd[2]), month = int(psd[0]), day = int(psd[1]))
        if today > end_wb: continue
        
        outfile.write(line)
    outfile.close()
    infile.close()
    new_wb = pd.read_csv('screened_wb.csv')
    #new_wb['dt'] = pd.to_datetime(new_wb[' DATE'])
    return new_wb

def get_new_stream(link):
    dl_link = link.format(y2 = this_year)
    urllib.request.urlretrieve(dl_link, 'new_stream_data.csv')
    outfile = open('screened_stream.csv','w')
    infile = open('new_stream_data.csv')
    i = 1
    header_found = False
    for line in infile:
        
        if header_found == False:
            header = line
            if 'Discharge' in line:
                header_found = True
                #print(' Header = ', header)
                outfile.write('Date,cfs\n')
            
            i += 1
            continue
        pl = line.split(',')
       
        if '-' not in line: continue
        try:
            this_date = pl[0].strip()
            psd = this_date.split('-')
        except:
            continue
        today = datetime.datetime(year = int(psd[0]), month = int(psd[1]), day = int(psd[2]))
        if today > end_wb: continue
        outline = pl[0] + ',' + pl[1] + '\n'
        outfile.write(outline)
    outfile.close()
    infile.close()
    new_stream = read_stream('screened_stream.csv')
    return new_stream

def fix_names(df):
    columns = list(df.keys())
    rendict = {}
    for c in columns:
        new_c = c.replace('(','')
        new_c = new_c.replace(')','')
        new_c = new_c.strip()
        new_c = new_c.replace(' ','_')
        rendict[c] = new_c
    df = df.rename(columns = rendict)
    return df
            
        
if __name__ == '__main__':
    if web == True:
        os.chdir('/var/www/html/ca_backend/python/lamar_ai')
    cols = ['cfslag7','lagged_delta','second_d','lagged_delta_variance','P_MM_ne','TMAX_C_ne','TMIN_C_ne','ACCUM_SNOWPACK_MM_ne','MELT_MM_ne','AET_MM_ne','Accum_Growing_Degree_Days_C_ne','p_lagged_ne','P_MM_pkr','TMAX_C_pkr','TMIN_C_pkr','ACCUM_SNOWPACK_MM_pkr','MELT_MM_pkr','AET_MM_pkr','Accum_Growing_Degree_Days_C_pkr','p_lagged_pkr']
    trim_data = True
    wb_link1 = 'http://climateanalyzer.science/python/wb_ai_train.py?station=ynpne_from_grid&year1=1981&year2={y2}&title=ynpne%20(XXX1)&station_type=GHCN&pet_type=oudin&max_soil_water=150.0&snow_density=0.2&forgiving=very&table_type=daily&sim_snow=True&use_heatload=False&csv_output=true&graph_table=table'
    wb_link2 = 'http://climateanalyzer.science/python/wb_ai_train.py?station=pkrpeak_from_grid&year1=1981&year2={y2}&title=ynpne%20(XXX1)&station_type=GHCN&pet_type=oudin&max_soil_water=150.0&snow_density=0.2&forgiving=very&table_type=daily&sim_snow=True&use_heatload=False&csv_output=true&graph_table=table'
    stream_link = 'http://climateanalyzer.science/python/make_tables.py?station=lamar_tower&year1=1988&year2={y2}&month=all&title=Lamar%20River%20nr%20Tower%20Ranger%20Station&table_type=stream&norm_per=null&csv=true'
    look_ahead = 8
    now = datetime.datetime.now()
    this_year = now.year
    one_day = datetime.timedelta(days = 1)
    end_wb = now + one_day * look_ahead
    print(end_wb)

    start = datetime.datetime(year = 1988, month = 9, day = 1)
    #stop = datetime.datetime(year = 2024, month = 7, day = 30)
    stop = end_wb
    one_day = datetime.timedelta(days = 1)
    d = get_new_stream(stream_link)
    
    d = n_lags(d, look_ahead, look_ahead, 'cfs')
    
    cfs_df = pd.DataFrame(d)
    #cfs_df['doy'] = cfs_df['dt'].dt.day_of_year
    cfs_df['month'] = cfs_df['month'].astype('int')
    #cfs_df['dowy'] = np.where(cfs_df['month'] >= 10, cfs_df['doy'] - 273, cfs_df['doy'] + 92)
    cfs_df.set_index('dt')
    
    ne_wb = get_new_wb_data(wb_link1)
    pkr_wb = get_new_wb_data(wb_link2)
    for wb in [ne_wb, pkr_wb]:
        p_lagged = np.array(wb[' P (MM)'])[:-1]
        p_lagged = np.insert(p_lagged,0,np.nan)
        wb['p_lagged'] = p_lagged
        wb['dt'] = pd.to_datetime(wb[' DATE'], errors = 'ignore')
        wb.set_index('dt')
    ne_wb = ne_wb.add_suffix('_ne')
    ne_wb = ne_wb.rename(columns = {"dt_ne":"dt"})
    
    pkr_wb = pkr_wb.add_suffix('_pkr')
    pkr_wb = pkr_wb.rename(columns = {"dt_pkr":"dt"})
    merged = pd.merge(cfs_df, ne_wb, how = "left", on = 'dt')
    total = pd.merge(merged, pkr_wb, how = 'left', on = 'dt')
    total = fix_names(total)
    #print(total)
    k_list = ['TMEAN_C', 'WATER_INPUT_TO_SOIL_MM','SOIL_WATER_MM','DAYS_SNOW_MM','', 'RAIN_MM', ]
    if trim_data == True:
        k_list = k_list + ['month','day', 'chaos_index','doy','dowy', 'INDEX_ne','INDEX_pkr', 'D_MM_ne','DATE_pkr','DATE_ne','PET_MM_pkr','RUNOFF_MM_pkr','PET_MM_ne','RUNOFF_MM_ne','D_MM_ne','D_MM_pkr','D_MM_ne', 'INDEX_ne','DATE_ne', 'INDEX_pkr',
 'DATE_pkr',]
    for key in k_list:
        for suffix in ['_pkr', '_ne']:
            dk = key + suffix
            try:
                total = total.drop(dk, axis = 1)
                
            except:
                pass
            try:
                total = total.drop(key, axis = 1)
            except:
                
                pass
    
    training_data = total[total['dt'] <= now]
    
    training_data.reset_index()
    training_data = training_data.drop('year', axis = 1)
    training_data.drop(training_data.tail(1).index,inplace=True) # drop last n rows
    training_dates = training_data['dt']
    training_dates.to_csv('training_dates.csv')
    training_data = training_data.drop('dt', axis = 1)
    training_data.to_csv('Lamar_training_new.csv', index = False, columns = cols)
    future = total[total['dt'] > now]
    future = future[future['year'] == str(now.year)]
    
    future = future.drop('cfs', axis = 1)
    future = future.drop('year', axis = 1)
    future = future[future['cfslag7'] > 0]
    dates = future['dt']
    dates.to_csv('forecast_dates.csv')
    future = future.drop('dt', axis = 1)
    future.reset_index()
    future.to_csv('new_future_data.csv', index = False, columns = cols)
    
    
    
    
    