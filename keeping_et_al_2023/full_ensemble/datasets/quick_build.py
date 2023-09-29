import xarray as xr
import statsmodels as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from matplotlib import colors
import os
import glob


def build_inputs(ds, whitelist, N = 10 ** 5, seed = 41506026):
    # 1: Build test and train from source data. For all valid predictors.
    #    Takes 12 minutes however large the test and train sample.
    temp = ds['DTR'].to_dataframe()
    mask = np.logical_not(np.isnan(np.array(temp['DTR'])))
    true_inds = np.argwhere(mask)[:,0]
    np.random.seed(seed)
    sample_inds = np.random.choice(true_inds, size = int(1.25 * N), replace = False)
    df = temp.iloc[sample_inds]
    df = df.drop(columns = ['DTR'])

    print('Adding variables to test-train:')
    for i, var in enumerate(whitelist):
        if (i+1) % np.floor(len(whitelist)/5) == 0:
            print(f'\t{20 * (i+1) / (len(whitelist)/5):.1f}% built')
        df[var] = np.array(ds[var].to_dataframe())[sample_inds]

    df = df.dropna()
    train, test = train_test_split(df, test_size = 0.20)
    return test, train


def apply_thresh(test, train, thresh_row, predictor_list):
    temp_test = test.copy(deep = True)
    temp_train = train.copy(deep = True)
    # Clipping to thresholds:
    for var in predictor_list:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            temp_train[var].loc[temp_train[var] < thresh_row['lo_'+var].value] = thresh_row['lo_'+var].value
            temp_train[var].loc[temp_train[var] > thresh_row['hi_'+var].value] = thresh_row['hi_'+var].value
            temp_test[var].loc[temp_test[var] < thresh_row['lo_'+var].value] = thresh_row['lo_'+var].value
            temp_test[var].loc[temp_test[var] > thresh_row['hi_'+var].value] = thresh_row['hi_'+var].value
    return temp_test, temp_train


def model_fit(test, train, formula, summary = False):
    # 2: Run a model and return: max_vif, aic and auc given a list of predictors
    #    Takes roughly 1.3 N microseconds, where N is the length of test and train.
    #    Therefore N = 10**7 makes a lot of sense at ~13 seconds per ru~, as does
    model = smf.glm(formula,
                    data = train,
                    offset = np.log(train['cell_area']),
                    family = sm.genmod.families.family.Binomial(
                    sm.genmod.families.links.Logit())).fit(disp = 0)
    if summary == True:
        print(model.summary())
    test['p'] = model.predict(exog = test)
    auc = roc_auc_score(test.sort_values(by = 'fire').fire,
                        test.sort_values(by = 'fire').p)
    aic = model.aic
    # Find variance-covariance matrix of the coefficients:
    cov = model.cov_params()
    # Find correlation matrix of the coefficients:
    corr = cov / model.bse / np.array(model.bse)[:,None]
    # Inversion of correlation matrix = partial correlations between coefs
    # Diagonal of that is the correlation of of variable with all others
    # Note that the intercept is excluded by [1:, 1:]
    max_vif = np.max(np.diag(np.linalg.inv(corr.values[1:, 1:])))
    print(f'AIC = {aic}')
    print(f'AUC = {auc}')
    print(f'Max VIF = {max_vif}')
    if summary == True:
        return model.params, aic, auc, max_vif
    return model.params


def build_ds(test, train, formula, variables, thresh_row, ds, N = 10 ** 8):
    # Fitting:
    params = model_fit(test, train, formula, summary = False)
    # Building:
    ds['logit_p'] = 0 * ds['fire']
    for param in params.keys():
        if param == 'Intercept':
            ds['logit_p'] = ds['logit_p'] + params['Intercept']
        else:
            print(param)
            temp = ds[param].clip(min = float(thresh_row['lo_' + param].value),
                                  max = float(thresh_row['hi_' + param].value))
            ds['logit_p'] = ds['logit_p'] + params[param] * temp
        ds['logit_p'].load()
        try:
            ds[param].close()
        except:
            print(param, 'not found')

    ds['logit_p'] = ds['logit_p'] + np.log(ds['cell_area'])
    ds['logit_p'].load()
    ds['cell_area'].close()
    ds['logit_p'].load()

    ds['p'] = np.exp(ds['logit_p'])/(1 + np.exp(ds['logit_p'])).load()
    return ds['p'], params


def runner():
    formula = 'fire ~ ' + ' + '.join(variables)
    print(formula)
    # Building dataset:
    ds_list = []
    for var in variables + ['fire', 'cell_area']:
        ds_list.append(xr.open_mfdataset('/rds/general/user/tk22/'+
                                         'projects/leverhulme_wil'+
                                         'dfires_theo_keeping/liv'+
                                         'e/start_inputs/genesis_'+
                                         f'{var}*20020101_20181231.nc'))
    ds = xr.merge(ds_list, compat = 'override')
    del ds_list
    # Building test-train:
    test, train = build_inputs(ds, variables + ['fire', 'cell_area'],
                               N = N, seed = np.random.randint(1,10**7))
    # Applying thresholds:
    test, train = apply_thresh(test, train, thresh_row,
                               variables)
    # Build dataset:
    da, params = build_ds(test, train, variables, thresh_row, N = 10 ** 8)

