import statsmodels.formula.api as smf
import statsmodels.api as sm
import sys
import os
import json
import xarray as xr
import numpy as np
import pandas as pd
import gc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split



def build_inputs(ds, whitelist, N = 10 ** 7):
    # 1: Build test and train from source data. For all valid predictors.
    #    Takes 12 minutes however large the test and train sample.
    print('BUILDING INPUT VARIABLES:')
    sys.stdout.flush()
    temp = ds['MaxT'].to_dataframe()
    mask = np.logical_not(np.isnan(np.array(temp['MaxT'])))
    true_inds = np.argwhere(mask)[:,0]
    sample_inds = np.random.choice(true_inds, size = 5*N, replace = False)
    df = temp.iloc[sample_inds]
    df = df.drop(columns = ['MaxT'])
    for i, var in enumerate(whitelist):
        print(f'\t\t{20 * (i+1) / (len(whitelist)/5):.1f}% built')
        sys.stdout.flush()
        df[var] = np.array(ds[var].to_dataframe())[sample_inds]
    df = df.dropna()
    train, test = train_test_split(df, test_size = 0.5)
    train = train.sample(n = N)
    test = test.sample(n = N)
    return test, train



def model_fit(test, train, predictor_list):
    # 2: Run a model and return: max_vif, aic and auc given a list of predictors
    #    Takes roughly 1.3 N microseconds, where N is the length of test and train.
    #    Therefore N = 10**7 makes a lot of sense at ~13 seconds per run
    if len(predictor_list) == 0:
        formula = 'fire ~ 1'
    else:
        formula = 'fire ~ ' + ' + '.join(predictor_list)
    
    print(formula)
    sys.stdout.flush()
    
    model = smf.glm(formula,
                    data = train,
                    offset = np.log(train['cell_area']),
                    family = sm.genmod.families.family.Binomial(
                    sm.genmod.families.links.Logit())).fit(disp = 0)    
    
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
    return aic, auc, max_vif

def variable_selection_forwards(df):
    # Selection: of those with max_vif < 5, max(AIC) given AUC in top quartile/half
    #            unless max_vif all > 5, then must be < 10.
    temp_1 = df[df['Max VIF'] < 5]
    temp_2 = df[df['Max VIF'] < 10]
    if len(temp_1) > 0:
        # Finding min(AIC):
        var = temp_1.iloc[np.argmin(temp_1['AIC'])]['New Variable']
        aic = np.min(temp_1['AIC'])

    elif len(temp_2) > 0:
        # Finding min(AIC):
        var = temp_2.iloc[np.argmin(temp_2['AIC'])]['New Variable']
        aic = np.min(temp_2['AIC'])

    else:
        print('FAILED: VIFs too high.')
    
    
    return var, aic

def forwards_step(J, test, train, prior_predictors, whitelist, step = 1):
    # 3: A forwards step function; it runs (2) adding one new variable from a
    #    whitelist, and finds the optimal new variable. Saves df of stats from (2).
    
    results = []
    
    for i, new_var in enumerate(whitelist):
        result = model_fit(test, train, prior_predictors + [new_var])
        results.append(result)
        gc.collect()
    
    results = [tuple([whitelist[i]]+list(r)) for i,r in enumerate(results)]
    df = pd.DataFrame(results, columns = ['New Variable', 'AIC', 'AUC', 'Max VIF'])
    df.to_csv(f'/rds/general/user/tk22/home/fire_genesis/step_1/run_{J}/'+
              f'step_{step}_results_forwards_{J}th_run.csv', index = False)
    return df

def variable_selection_switch(df):
    # Selection: of those with max_vif < 5, max(AIC) given AUC in top quartile/half
    #            unless max_vif all > 5, then must be < 10.

    # First VIF condition:
    temp_1 = df[df['Max VIF'] < 5]
    temp_2 = df[df['Max VIF'] < 10]

    if len(temp_1) > 0:
        # Finding min(AIC):
        new_var = temp_1.iloc[np.argmin(temp_1['AIC'])]['New Variable']
        old_var = temp_1.iloc[np.argmin(temp_1['AIC'])]['Old Variable']
        aic = np.min(temp_1['AIC'])

    elif len(temp_2) > 0:
        # Finding min(AIC):
        new_var = temp_2.iloc[np.argmin(temp_2['AIC'])]['New Variable']
        old_var = temp_2.iloc[np.argmin(temp_2['AIC'])]['Old Variable']
        aic = np.min(temp_2['AIC'])

    else:
        print('FAILED: VIFs too high.')
        
    
    return new_var, old_var, aic


def switch_step(J, test, train, prior_predictors, whitelist, step = 1):
    # 4: A backwards-forwards step function; it runs (2) substituting one new 
    #    variable from a whitelist for each old one, and finds the optimal new 
    #    variable. Saves df of stats from (2).

    results = []
    for k,old_var in enumerate(prior_predictors):
        print(f'{k+1} steps of {len(prior_predictors)} complete.')
        sys.stdout.flush()

        for i, new_var in enumerate(whitelist):
            predictors = list((set(prior_predictors) - set([old_var])).union(set([new_var])))
            result = model_fit(test, train, predictors)
            results.append(result)
            gc.collect()
    
    c = 0
    final_results = []
    for old_var in prior_predictors:
        for new_var in whitelist:
            final_results.append(tuple([old_var] + [new_var] + list(results[c])))
            c += 1

    df = pd.DataFrame(final_results, columns = ['Old Variable', 'New Variable', 'AIC', 'AUC', 'Max VIF'])
    df.to_csv(f'/rds/general/user/tk22/home/fire_genesis/step_1/run_{J}/'+
              f'step_{step}_results_switch_{J}th_run.csv', index = False)
    return df


def runner(J, N = 10**7, n_vars = 10):    
    # 5: A runner function that iteratively builds a set of predictors using
    #    alternative forwards and backwards-forwards steps. Printing the stats 
    #    and composition of each new stage of the model to a text-file, until some
    #    condition is met (e.g. AUC = 0.9, or max_vif = 5 or 10)   
    
    ds = xr.open_mfdataset(("/rds/general/user/tk22/projects/" +
                            "leverhulme_wildfires_theo_keeping/" +
                            "live/start_inputs/genesis_*20020101_20181231.nc"))
    
    blacklist = ['lat', 'lon', 'date', 'counts', 'total_fires', 'fire_days']
    whitelist = list(set(list(ds.variables)) - set(blacklist))

    #list_path = (f'/rds/general/user/tk22/home/fire_genesis/step_1/'+
    #             f'run_{J}/predictor_list_{J}.txt')
    #with open(list_path, "r") as file:
    #    predictor_list = json.load(file)
    predictor_list = []
    
    # Generating test and train once:
    test, train = build_inputs(ds, whitelist, N = N)
    
    # Beating old AIC:
    old_aic = 10**10
    
    for step in range(1, n_vars * 2 + 1):

        # Then run fits.
        print(f'Step {step}')
        sys.stdout.flush()
        # Odd steps:
        if step % 2 == 1:
            reduced_whitelist = list(set(whitelist) - set(['fire'] + predictor_list))
            df = forwards_step(J, test, train, predictor_list, reduced_whitelist, step = step)
            var, new_aic = variable_selection_forwards(df)
            predictor_list.append(var)

        # Even steps:
        if step % 2 == 0:
            reduced_whitelist = list(set(whitelist) - set(['fire']))
            df = switch_step(J, test, train, predictor_list, reduced_whitelist, step = step)
            new_var, old_var, new_aic = variable_selection_switch(df)
            predictor_list.remove(old_var)
            predictor_list.append(new_var)
        
        print(f'\tPredictor list: {predictor_list}')    
        print(df.sort_values(by = 'AIC', ascending = True).head())
        sys.stdout.flush()
        
        if step % 2 == 1:   
            if new_aic < old_aic - 2: 
                # As model considered significantly better if 2 AIC units lesser
                # Burnham and Anderson (2004)
                pass
            else:
                print('New AIC greater than prior AIC:')
                print(f'{old_aic:.2f} > {new_aic:.2f} + 2')
                break
        if step % 2 == 0:
            if new_aic < old_aic + 0.01: 
                # As model may be exactly the same, and floating point errors likely.
                pass
            else:
                print('New AIC greater than prior AIC:')
                print(f'{old_aic:.2f} > {new_aic:.2f}')
                break
        old_aic = new_aic
        
        if step == 24:
            list_path = (f'/rds/general/user/tk22/home/fire_genesis/'+
                         f'predictor_sel_run/predictor_list_{J}.txt')
            with open(list_path, "w") as file:
                json.dump(predictor_list, file)
            with open(list_path, "r") as file:
                predictor_list = json.load(file)
                print(predictor_list)
            break
    return


if __name__ == '__main__':
    J = int(os.getenv('PBS_ARRAY_INDEX'))
    runner(J, N = 10**6, n_vars = 30)
