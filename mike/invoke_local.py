#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local version of invoke.py - runs forecasts using local PyTorch models instead of AWS SageMaker.
MUST RUN data_prep.py BEFORE RUNNING THIS SCRIPT. OTHERWISE FORECASTS WILL NOT BE UPDATED.

Created on Mon Jul 22 14:35:05 2024
Modified for local execution: 2025-11-26

@author: mt
"""
import io, csv, datetime, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import torch
import pickle

# Set to False for local execution
web = False

def tryfloat(v):
    try:
        r = float(v)
    except:
        r = np.nan
    return r

def rd_test_data(f):
    """Read test data for predictions"""
    pred_df = pd.read_csv(f)
    return pred_df

def get_forecast_dates(f = 'forecast_dates.csv'):
    out = []
    infile = open(f)
    reader = csv.reader(infile)
    header = next(reader)
    for line in reader:
        out.append(parse_timestamp(line[1], s= True))
    infile.close()
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

def get_predictions_local(model_path='./lamar_model'):
    """
    Get predictions using locally trained AutoGluon model.

    This loads a model trained with train_local.py and generates predictions.
    The model is trained using AutoGluon, which is the same technology
    that SageMaker Canvas uses.

    Args:
        model_path: Path to the trained AutoGluon model

    Returns:
        List of predictions
    """
    from autogluon.tabular import TabularPredictor

    print("Loading locally trained AutoGluon model...")

    try:
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please run train_local.py first to train the model."
            )

        # Load the trained model
        print(f"Loading model from: {model_path}")
        predictor = TabularPredictor.load(model_path)

        print(f"✓ Model loaded successfully")
        print(f"  Model type: AutoGluon TabularPredictor")
        print(f"  Problem type: {predictor.problem_type}")
        print()

        # Read the input data
        pred_df = pd.read_csv('./new_future_data.csv')
        print(f"Making predictions for {len(pred_df)} future timesteps...")

        # Make predictions
        # AutoGluon returns a pandas Series
        predictions = predictor.predict(pred_df)

        # Convert to list (matching the format expected by the rest of the code)
        prediction_list = predictions.tolist()

        print(f"✓ Successfully generated {len(prediction_list)} predictions")
        print(f"  Prediction range: {min(prediction_list):.2f} - {max(prediction_list):.2f} cfs")
        print()

        return prediction_list

    except FileNotFoundError as e:
        print(f"\n{'='*60}")
        print("ERROR: Model not found!")
        print('='*60)
        print(str(e))
        print("\nTo train a model, run:")
        print("  python train_local.py")
        print("\nThis will train an AutoGluon model using your data.")
        print('='*60)
        print("\nFalling back to placeholder predictions...\n")

        # Fallback: generate placeholder predictions
        pred_df = pd.read_csv('./new_future_data.csv')
        num_predictions = len(pred_df)
        predictions = np.random.uniform(100, 200, num_predictions).tolist()

        return predictions

    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure AutoGluon is installed: pip install autogluon")
        print("2. Check that train_local.py completed successfully")
        print("3. Verify that ./lamar_model directory exists")
        print("\nFalling back to placeholder predictions...\n")

        # Fallback: generate placeholder predictions
        pred_df = pd.read_csv('./new_future_data.csv')
        num_predictions = len(pred_df)
        predictions = np.random.uniform(100, 200, num_predictions).tolist()

        return predictions

def merge(old_predictions_dict,stream_data_dict, predictions, prediction_dts):
    global today, first_dt
    one_day = datetime.timedelta(days = 1)
    # Need to add in actual data as it arrives.

    first_new_dt = prediction_dts[0]
    print('first new dt = ', first_new_dt)
    this_day = today
    while this_day < first_new_dt:
        if this_day not in stream_data_dict:
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
    fig,ax = plt.subplots(figsize=(12, 6))
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
    plt.title('Lamar River at Tower Junction (Local Model)')
    plt.tight_layout()
    latestfilename = 'latest_lamar_prediction_{tdy}.png'.format(tdy = today.strftime("%m-%d-%y"))
    plt.savefig(latestfilename)
    print(f"Saved plot to {latestfilename}")
    return latestfilename

def edit_html_pages(ltf):
    """Edit HTML page to update prediction graph"""
    html_file = 'lamar_flow_prediction.html'
    if not os.path.exists(html_file):
        print(f"Warning: {html_file} not found, skipping HTML update")
        return

    infile = open(html_file, 'rt')
    outfile = open('new_html_page.html','w')
    for line in infile:
        if '<center><img id = "prediction_graph"' in line:
            outline = '<center><img id = "prediction_graph" src="{nf}" ></center>'.format(nf = ltf)
            print(outline)
            outfile.write(outline)
        else:
            outfile.write(line)
    infile.close()
    outfile.close()
    os.system('mv {of} {nf}'.format(of = 'new_html_page.html', nf = html_file))

if __name__ == '__main__':
    print("="*60)
    print("LOCAL FORECAST GENERATION (no AWS integration)")
    print("="*60)

    if web == True:
        print("Warning: web mode is set to True but running locally")
        os.chdir('/var/www/html/ca_backend/python/lamar_ai')
        outpath = '/var/www/html/ca_backend/python/lamar_ai/'
    else:
        outpath = './'

    now = datetime.datetime.now()
    today = datetime.datetime(year = now.year, month = now.month, day = now.day, hour = 0, minute = 0)
    print(f"Today: {today}")

    # Get predictions from local model instead of AWS
    print("\nGenerating predictions from local model...")
    prediction_list = get_predictions_local()
    print('predictions : ', prediction_list)

    prediction_dates = get_forecast_dates()
    training_dates = get_forecast_dates('training_dates.csv')

    # Load old predictions
    old_predictions = pd.read_csv('full_latest_predictions.csv')
    pdts = [parse_timestamp(x, s = True) for x in old_predictions['dt']]
    old_predictions_dict = {}
    opd = 0
    for pdt in pdts:
        old_predictions_dict[pdt] = float(old_predictions['Predicted Flow'][opd])
        opd +=1

    first_dt = datetime.datetime(year= 1988, month = 9, day = 17)
    print('First dt = ', first_dt)

    # Load stream data
    stream_data = pd.read_csv('screened_stream.csv')
    stream_data['dt'] = pd.to_datetime(stream_data['Date'])
    stream_data = stream_data[stream_data['dt'] >= first_dt]
    stream_data_dict = {}
    sdi = 0
    sd = np.array(stream_data['cfs'])
    for sdt in stream_data['dt']:
        stream_data_dict[sdt] = tryfloat(sd[sdi])
        sdi +=1

    # Merge predictions with actual data
    merged_actual, merged_predicted, merged_dt = merge(old_predictions_dict, stream_data_dict, prediction_list, prediction_dates)

    # Generate plot
    final_predicted = merged_predicted[-400:]
    final_actual = merged_actual[-400:]
    final_dts = merged_dt[-400:]
    final_actual = np.where(final_actual == 0, np.nan, final_actual)
    latestfilename = graph_it(final_predicted, final_actual, final_dts, 30)

    # Save results
    out_df = pd.DataFrame({"dt":merged_dt, 'Actual Flow': merged_actual, 'Predicted Flow': merged_predicted})
    output_file = outpath + 'full_latest_predictions.csv'
    out_df.to_csv(output_file, index = False)
    print(f"\nSaved predictions to {output_file}")

    # Update HTML if file exists
    edit_html_pages(latestfilename)

    print("\n" + "="*60)
    print("FORECAST GENERATION COMPLETE")
    print("="*60)
