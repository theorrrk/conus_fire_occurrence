import pandas as pd
import xarray as xr
import numpy as np
import warnings
import statsmodels as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
import sys
from sklearn.metrics import roc_auc_score
import os
import json
import time


def gather_dataset(variables):
    ds_list = []
    for var in variables + ['fire', 'cell_area']:
        ds_list.append(xr.open_mfdataset('/rds/general/user/tk22/'+
                                         'projects/leverhulme_wil'+
                                         'dfires_theo_keeping/liv'+
                                         'e/start_inputs/genesis_'+
                                         f'{var}*20020101_20181231.nc'))
    ds = xr.merge(ds_list, compat = 'override')
    del ds_list
    return ds


def define_threshold_options(ds, variables):
    print('Building threshold options dictionary:')
    threshs = {}
    for var in variables:
        print(f'\t{var}')
        sys.stdout.flush()
        min0   = float(ds[var].min().load())
        max0   = float(ds[var].max().load())
        perc1  = float(ds[var].quantile(0.0001).load())
        perc99 = float(ds[var].quantile(0.9999).load())
        std0    = float(ds[var].std().load())
        mean0   = float(ds[var].mean().load())

        arr = list(np.flip(np.arange(mean0, perc1, -std0/4)))[:-1]
        arr = arr + list(np.arange(mean0, perc99, std0/4))
        arr = [min0, perc1] + arr
        arr = arr + [perc99, max0]
        # Removing steps if less than eighth of standard deviation:
        new_arr = []
        for i, element in enumerate(reversed(arr)):
            if i == 0:
                prior_appended_value = element
                new_arr.append(element)
            else:
                if (prior_appended_value - element) < (std0/8):
                    pass # As gap too small to add threshold
                else:
                    prior_appended_value = element
                    new_arr.append(element)
        if min0 not in new_arr:
            new_arr.append(min0) # So full span of variable
        new_arr.reverse()
        threshs[var] = new_arr
    print('\n')
    sys.stdout.flush()
    return threshs


def initial_threshold_indices(variables):
    ind_df = pd.DataFrame(columns = variables)
    ind_df.loc[len(ind_df)] = list(np.zeros(len(variables), dtype = int))
    ind_df.loc[len(ind_df)] = list(-np.ones(len(variables), dtype = int))
    ind_df['index'] = ['min_ind', 'max_ind']
    ind_df = ind_df.set_index('index')
    return ind_df


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
        print(f'\t{var}')
        sys.stdout.flush()
        df[var] = np.array(ds[var].to_dataframe())[sample_inds]
    print('\n')
    sys.stdout.flush()
    df = df.dropna()
    train, test = train_test_split(df, test_size = 0.20)
    return test, train


def build_summary(threshs, ind_df, variables, aic, auc):
    # Definining summary dataframe:
    summary_df = pd.DataFrame(columns = (['lo_' + var for var in variables] + 
                                         ['hi_' + var for var in variables] + 
                                         ['AIC', 'AUC']))
    new_row = ([threshs[var][ind_df.loc['min_ind'][var]] for var in variables] + 
               [threshs[var][ind_df.loc['max_ind'][var]] for var in variables] +
               [aic, auc])
    summary_df.loc[len(summary_df)] = new_row
    return summary_df


def run_model(ttest, ttrain, variables):
    formula = 'fire ~ ' + ' + '.join(variables)
    model = smf.glm(formula,
                    data = ttrain,
                    offset = np.log(ttrain['cell_area']),
                    family = sm.genmod.families.family.Binomial(
                    sm.genmod.families.links.Logit())).fit()
    ttest['p'] = model.predict(exog = ttest)
    auc = roc_auc_score(ttest.sort_values(by = 'fire').fire,
                        ttest.sort_values(by = 'fire').p)
    aic = float(model.aic)
    if np.isnan(aic):
        aic = 10**10 # I.e. do not decrease this one.
        auc = np.nan
    return float(model.aic), float(auc)


def tightening_multistep(var, variables, ind_df, threshs,
                         test, train, direction = 'max', steps = 1):
    # Checking if reduction can be made to thresholds:
    ok_steps = (len(threshs[var]) - ind_df[var].iloc[0] 
                + ind_df[var].iloc[1] - 1)
    if steps > ok_steps:
        return 10**10, np.nan
    
    # Tightening thresholds for that variable by "steps" steps
    temp_ind_df = ind_df.copy(deep = True)
    ttest  = test.copy(deep = True)
    ttrain = train.copy(deep = True)
    if direction == 'max':
        # Tighten thresholds for that variable:
        temp_ind_df.loc['max_ind'][var] += -steps
    if direction == 'min':
        temp_ind_df.loc['min_ind'][var] += steps
    # Defining thresholds:
    lo_threshs = [threshs[v][temp_ind_df.loc['min_ind'][v]] for v in variables]
    hi_threshs = [threshs[v][temp_ind_df.loc['max_ind'][v]] for v in variables]
    # Trimming data:
    for k,v in enumerate(variables):
        ttest[v].clip(lower = lo_threshs[k],
                      upper = hi_threshs[k],
                      inplace = True)
        ttrain[v].clip(lower = lo_threshs[k],
                       upper = hi_threshs[k],
                       inplace = True)
    # Run model:
    return run_model(ttest, ttrain, variables)


def update_thresholds(threshs, ind_df, summary_df, upper_aics, lower_aics, auc,
                      steps = 1, variables = ['VPD_night', 'GPP_1y']):
    # Find best of both clippings:
    best_upper = upper_aics[np.argmin(upper_aics)]
    best_lower = lower_aics[np.argmin(lower_aics)]
    aic = min([best_upper, best_lower])
    # Updating dataframes:
    if best_upper <= best_lower:
        # Upper clipping case:
        ind_df.iloc[1][variables[np.argmin(upper_aics)]] += -steps
        new_row = ([threshs[var][ind_df.loc['min_ind'][var]]
                    for var in variables] + 
                   [threshs[var][ind_df.loc['max_ind'][var]]
                    for var in variables] +
                   [aic, auc])
        summary_df.loc[len(summary_df)] = new_row
    else:
        # Lower clipping case:
        ind_df.iloc[0][variables[np.argmin(lower_aics)]] += steps
        new_row = ([threshs[var][ind_df.loc['min_ind'][var]]
                    for var in variables] + 
                   [threshs[var][ind_df.loc['max_ind'][var]]
                    for var in variables] +
                   [aic, auc])
        summary_df.loc[len(summary_df)] = new_row
    return ind_df, summary_df


def calculate_thresholds(n, variables, index):
    start_time = time.time()
    # 1: Build basic test and train datasets.
    ds = gather_dataset(variables)
    test, train = build_inputs(ds, variables + ['fire', 'cell_area'],
                               N = 10 ** n, seed = np.random.randint(1,10**7))

    # 2: Create a dataframe of high and low thresholds (initially max and min) 
    threshs = define_threshold_options(ds, variables) # Threshold options.

    # 3: Find the AIC of the basic model (no clipping)
    aic, auc = run_model(test, train, variables)
    ind_df = initial_threshold_indices(variables) # Initial threshold values.
    summary_df = build_summary(threshs, ind_df, variables, aic, auc)

    # 4: Create compact tightening functions:

    tighten_upper = lambda var, n: tightening_multistep(var, variables,
                                                       ind_df, threshs,
                                                       test, train,
                                                       direction = 'max',
                                                       steps = n)

    tighten_lower = lambda var, n: tightening_multistep(var, variables,
                                                       ind_df, threshs,
                                                       test, train,
                                                       direction = 'min',
                                                       steps = n)

    # 5: Establish a loop (for loop with break condition)
    for count in range(10**5):
        print('\nCOUNT:', count+1)
        sys.stdout.flush()

        # a: Ratchet in from the upper threshold and find the best AIC.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            upper_aics = [tighten_upper(var, 1)[0] for var in variables]
            upper_aucs = [tighten_upper(var, 1)[1] for var in variables]
            best_upper = upper_aics[np.argmin(upper_aics)]
            best_upper_auc = upper_aucs[np.argmin(upper_aics)]

        # b: Ratchet in from the lower threshold and find the best AIC.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lower_aics = [tighten_lower(var, 1)[0] for var in variables]
            lower_aucs = [tighten_lower(var, 1)[1] for var in variables]
            best_lower = lower_aics[np.argmin(lower_aics)]
            best_lower_auc = lower_aucs[np.argmin(lower_aics)]

        # c: Check if an improvement by step of 2:
        improvement =  ((best_upper < aic - 2) or 
                        (best_lower < aic - 2))

        # d: If either (a) or (b) an improvement on the previous value.
        if improvement == True:
            # i: Update the target AIC to beat.
            print(f'Old AIC: {aic}')
            if best_upper <= best_lower:
                aic = best_upper
                auc = best_upper_auc
            else:
                aic = best_lower
                auc = best_lower_auc
            clip_type = ['Upper','Lower'][np.argmin([best_upper,best_lower])]
            if clip_type == 'Upper':
                clipped_var = variables[np.argmin(upper_aics)]
            elif clip_type == 'Lower':
                clipped_var = variables[np.argmin(lower_aics)]
            print(f'{clipped_var} : {clip_type} threshold tightened by 1 step')
            print(f'New AIC: {aic}. (AUC = {auc})')
            sys.stdout.flush()

            # ii: Find the best performing new thresholds.
            ind_df, summary_df = update_thresholds(threshs, ind_df, summary_df,
                                                   upper_aics, lower_aics, auc,
                                                   steps = 1,
                                                   variables = variables)
            summary_df.to_csv(('/rds/general/user/tk22/home/fire_genesis/'+
                               f'step_2/threshold_summary/threshold_summary_E{n}_{index}.csv'))

        # e: If neither (a) nor (b) an improvement:
        if improvement == False:

            # i: Define max number of steps that could be tightened.
            max_steps = np.max(np.array([len(v) for v in threshs.values()])
                               - np.array(ind_df.iloc[0] - ind_df.iloc[1]))

            # ii: For each variable, ratchet in iteratively from upper thresh 
            for N in range(2, max_steps+1):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    upper_aics = [tighten_upper(var, N)[0] for var in variables]
                    upper_aucs = [tighten_upper(var, N)[1] for var in variables]
                best_upper = upper_aics[np.argmin(upper_aics)]
                best_upper_auc = upper_aucs[np.argmin(upper_aics)]
                if best_upper < aic - 2:
                    upper_steps = N
                    improvement = True
                    break

            # iii: For each variable, ratchet in iteratively from lower thresh 
            for N in range(2, max_steps+1):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    lower_aics = [tighten_lower(var, N)[0] for var in variables]
                    lower_aucs = [tighten_lower(var, N)[1] for var in variables]
                best_lower = lower_aics[np.argmin(lower_aics)]
                best_lower_auc = lower_aucs[np.argmin(lower_aics)]
                if best_lower < aic - 2:
                    lower_steps = N
                    improvement = True
                    break

            # iv: 
            if improvement == True:
                # 1: Was lower or upper clipping better?
                upper_best = (best_upper <= best_lower)
                print(f'Old AIC: {aic}')

                clip_type = ['Upper','Lower'][np.argmin([best_upper,best_lower])]
                if clip_type == 'Upper':
                    clipped_var = variables[np.argmin(upper_aics)]
                elif clip_type == 'Lower':
                    clipped_var = variables[np.argmin(lower_aics)]

                if upper_best:
                    print((f'{clipped_var} : {clip_type} threshold '
                           +f'tightened by {upper_steps} steps'))
                    n_steps = upper_steps
                else:
                    print((f'{clipped_var} : {clip_type} threshold '+
                           f'tightened by {lower_steps} steps'))
                    n_steps = lower_steps


                # 2: Update the target AIC to beat.
                if upper_best:
                    aic = best_upper
                    auc = best_upper_auc
                else:
                    aic = best_lower
                    auc = best_lower_auc

                print(f'New AIC: {aic}. (AUC = {auc})')
                sys.stdout.flush()

                # 3: Find the best performing new thresholds.
                ind_df, summary_df = update_thresholds(threshs, ind_df, summary_df,
                                                       upper_aics, lower_aics, auc, 
                                                       steps = n_steps,
                                                       variables = variables)
                summary_df.to_csv(('/rds/general/user/tk22/home/'+
                                   'fire_genesis/step_2/threshold_summary/'+
                                   f'threshold_summary_E{n}_{index}.csv'))


            # v: Break loop if no improvement.
            else:
                print('Jumping thresholds gives no further improvement.')
                break
    total_seconds = time.time() - start_time
    print(f'\n\n\nTOTAL RUN TIME:\n\n{(total_seconds / (60**2)):.2f} hours')
    sys.stdout.flush()
    return


def main(index, n = 6):
    thresh_ind = index % 100
    sel_ind = int((index - thresh_ind)/100)
    print(f'Threshold Index: {thresh_ind}')
    print(f'Variable Selection Index: {sel_ind}')
    try:
        list_path = (f'/rds/general/user/tk22/home/fire_genesis/step_2/selected_vars/'+
                     f'predictor_list_{sel_ind}.txt')
        with open(list_path, "r") as file:
            predictor_list = json.load(file)
    except:
        return
        
    calculate_thresholds(n, predictor_list, index)
    return



if __name__ == '__main__':
    index = int(os.getenv('PBS_ARRAY_INDEX')) # 0 to 10000
    main(index, n = 6)