#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 16:36:20 2025

@author: mt
"""
from autogluon.tabular import TabularDataset, TabularPredictor
data_url = 'https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/'
train_data = TabularDataset(f'{data_url}train.csv')
label = 'signature'
#train_data[label].describe()
predictor = TabularPredictor(label=label).fit(train_data)
test_data = TabularDataset(f'{data_url}test.csv')

y_pred = predictor.predict(test_data.drop(columns=[label]))
evals = predictor.evaluate(test_data, silent=True)
leaderboard = predictor.leaderboard(test_data)