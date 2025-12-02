#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MUST RUN data_prep.py BEFORE RUNNING THIS SCRIPT. OTHERWISE FORECASTS WILL NOT BE UPDATED.
Created on Mon Jul 22 14:35:05 2024

@author: mt
"""
import boto3, io, csv, datetime, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory



web = False

def tryfloat(v):
    try:
        r = float(v)
    except:
        r = np.nan
    return r

def rd_test_data(f):
    pred_df = pd.read_csv(f)
    payload = pred_df.to_csv(header=False, index=False).encode("utf-8")
    return payload

def get_forecast_dates(f = 'forecast_dates.csv'):
    out = []
    infile = open(f)
    reader = csv.reader(infile)
    header = next(reader)
    for line in reader:
        out.append(parse_timestamp(line[1], s= True))
    infile.close()
    #print('Forecast Dates = ', out)
    return out

def parse_timestamp(ts, s = False):
    ts = ts.strip()
    pts = ts.split('-')
    year = pts[0]
    month = pts[1]
    day = pts[2]
    dt = datetime.datetime(year = int(year), month = int(month), day = int(day), hour = 0, minute = 0)
    if s == False:
        return year, month, day, dt
    else:
        return dt

def get_predictions():
    endpoint_name = 'canvas-new-deployment-04-03-2025-1-20-PM'
    client = boto3.client("runtime.sagemaker", region_name='us-west-2')
    body = rd_test_data('./new_future_data.csv')
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="text/csv",
        Body=body,
        Accept="application/json"
    )

    response_body = response['Body']
    response_str = response_body.read().decode('utf-8')
    response_dict = eval(response_str)
    predictions = response_dict['predictions']
    prediction_list = [x['score'] for x in predictions]
    return prediction_list

def merge(old_predictions_dict,stream_data_dict, predictions, prediction_dts):
    global today, first_dt
    one_day = datetime.timedelta(days = 1)
    # Need to add in actual data as it arrives.
    
    first_new_dt = prediction_dts[0]
    print('first new dt = ', first_new_dt)
    this_day = today
    while this_day < first_new_dt:
        if this_day not in stream_data:
            stream_data_dict[this_day] = np.nan
        if this_day not in old_predictions_dict:
            print('add nan prediction for today if it is missing')
            old_predictions_dict[this_day] = np.nan
        this_day = this_day + one_day
    
    i = 0
    while this_day <= prediction_dts[-1]:
        print(this_day, predictions[i])
        old_predictions_dict[this_day] = predictions[i]
        stream_data_dict[this_day] = np.nan
        this_day = this_day + one_day
        i+=1
    print('make_series of actual...')
    dts,final_actual_flow = make_series_from_dict(stream_data_dict, first_dt,this_day)
    print('make series of predictions....')
    dts,final_predicted_flow= make_series_from_dict(old_predictions_dict, first_dt,this_day)
    return final_actual_flow, final_predicted_flow,dts, 

def make_series_from_dict(thedict,firstdate,lastdate, printvals = False):
    one_day = datetime.timedelta(days = 1)
    this_day = firstdate
    dts = []
    vals = []
    while this_day < lastdate:
        
        dts.append(this_day)
        if printvals:
            print(this_day, thedict[this_day])
        try:
            vals.append(thedict[this_day])
        except:
            vals.append(np.nan)
        
        this_day = this_day + one_day
    dts = np.array(dts)
    vals = np.array(vals)
    return dts, vals
        
        
def graph_it(predicted, actual, dates, sparse_factor):
    global today
    t = dates == today
    break_index = np.where(t == True)[0][0]
    
    dates = [x.strftime("%m/%d/%y") for x in dates]
    fig,ax = plt.subplots()
    cats = np.arange(len(predicted))
    ax.plot(cats,predicted, color = 'red', alpha = 0.5, linewidth = 3, label = 'Predicted')
    ax.plot(cats,actual, color = 'black', alpha = 0.5, linewidth = 3, label = 'Actual')
    plt.axvline(break_index, color = 'black', linewidth = 4, alpha = 0.2)
    tform = blended_transform_factory(ax.transData, fig.transFigure)
    ax.text(break_index+5, 0.095, ">>>>\nFuture\n>>>>", fontsize='small', color='r', transform=tform)
    ax.text(break_index-5, 0.89, "Today:\n{t}\nv".format(t=today.strftime("%m/%d/%y")), fontsize='small', color='black', transform=tform)
    sparse_x = cats[::sparse_factor]
    sparse_dates = dates[::sparse_factor]
    plt.xticks(sparse_x,sparse_dates, rotation = 90)
    plt.legend(loc = 'best')
    plt.ylabel('cfs')  
    plt.title('Lamar River at Tower Junction')
    plt.tight_layout()
    latestfilename = 'latest_lamar_prediction_{tdy}.png'.format(tdy = today.strftime("%m-%d-%y"))
    plt.savefig(latestfilename)
    return latestfilename
    
def edit_html_pages(ltf):
    infile = open('lamar_flow_prediction.html', 'rt')
    outfile = open('new_html_page.html','w')
    for line in infile:
        if '<center><img id = "prediction_graph"' in line:
            outline = '<center><img id = "prediction_graph" src="https://climateanalyzer.science/python/lamar_ai/{nf}" ></center>'.format(nf = ltf)
            print(outline)
            outfile.write(outline)
        else:
            outfile.write(line)
    infile.close()
    outfile.close()
    os.system('mv {of} {nf}'.format(of = 'new_html_page.html', nf = 'lamar_flow_prediction.html'))

if __name__ == '__main__':    
    #MUST RUN data_prep.py BEFORE RUNNING THIS SCRIPT. OTHERWISE FORECASTS WILL NOT BE UPDATED.
    if web == True:
        os.chdir('/var/www/html/ca_backend/python/lamar_ai')
        outpath = '/var/www/html/ca_backend/python/lamar_ai/'
    else:
        outpath = './'
    now = datetime.datetime.now()
    today = datetime.datetime(year = now.year, month = now.month, day = now.day, hour = 0, minute = 0)
    prediction_list = get_predictions()
    print('predictions : ', prediction_list)
    prediction_dates = get_forecast_dates()
    training_dates = get_forecast_dates('training_dates.csv')
    old_predictions = pd.read_csv('full_latest_predictions.csv')
    pdts = [parse_timestamp(x, s = True) for x in old_predictions['dt']]
    old_predictions_dict = {}
    opd = 0
    for pdt in pdts:
        old_predictions_dict[pdt] = float(old_predictions['Predicted Flow'][opd])
        opd +=1
        
    #first_dt = training_dates[0]
    first_dt = datetime.datetime(year= 1988, month = 9, day = 17)
    print('First dt = ', first_dt)
    stream_data = pd.read_csv('screened_stream.csv')
    
    stream_data['dt'] = pd.to_datetime(stream_data['Date'])
    stream_data = stream_data[stream_data['dt'] >= first_dt]
    stream_data_dict = {}
    sdi = 0
    sd = np.array(stream_data['cfs'])
    for sdt in stream_data['dt']:
        stream_data_dict[sdt] = tryfloat(sd[sdi])
        sdi +=1
    merged_actual, merged_predicted, merged_dt = merge(old_predictions_dict,stream_data_dict, prediction_list, prediction_dates)
   
    final_predicted = merged_predicted[-400:]
    final_actual = merged_actual[-400:]
    final_dts = merged_dt[-400:]
    final_actual = np.where(final_actual == 0, np.nan, final_actual)
    latestfilename = graph_it(final_predicted, final_actual, final_dts, 30)
    
    out_df = pd.DataFrame({"dt":merged_dt, 'Actual Flow': merged_actual, 'Predicted Flow': merged_predicted})
    #out_df = out_df[out_df['dt'] < today]
    out_df.to_csv(outpath + 'full_latest_predictions.csv', index = False)
    edit_html_pages(latestfilename)
    


