import numpy as np
import xarray as xr
import pandas as pd
import time
import warnings
    

def opt_step(arr1, arr2, b):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Optimising:
        temp = arr1 ** b
        a = float(np.nanmean(arr2))/float(np.nanmean(temp))
        temp = a * arr1 ** b
        # Evaluation metrics:
        mod = np.nanmean(temp, axis = 2)
        obs = np.nanmean(arr2, axis = 2)
        # How well do non-nan quantiles match?
        obs_qs = []
        mod_qs = []
        for q in np.linspace(0,1,10001):
            obs_qs.append(np.nanquantile(np.log10(obs), q))
            mod_qs.append(np.nanquantile(np.log10(mod), q))
        try:
            res = np.array(obs_qs)[~np.isnan(obs_qs)] - np.array(mod_qs)[~np.isnan(obs_qs)]
            rss = np.nanmean(res**2)
        except:
            rss = np.nan
    return a, b, rss


def stretch_dataset(ds, decimals = 11):
    # Target values:
    a_vals = []
    b_vals = []
    rss_vals = []
    # Building arrays:
    arr1 = ds.p.to_numpy()
    arr2 = ds.fire.to_numpy()
    arr2 = arr2 * (arr1 * 0 + 1)
    # Tightening steps:
    t0 = time.time()
    n_steps = 21
    lower_b, upper_b = 0.5, 2.5
    
    for iteration in range(1, decimals):
        print(f'Iteration {iteration}: {(time.time() - t0)/60:.1f} minutes')
        temp_a_vals = []
        temp_b_vals = []
        temp_rss_vals = []
        for b in np.linspace(lower_b, upper_b, n_steps):
            a, b, rss = opt_step(arr1, arr2, b)
            print(f'\ta = {a:.10f}, b = {b:.10f}, rss = {rss:.10f}')
            # a values:
            a_vals.append(a)
            temp_a_vals.append(a)
            # b values:
            b_vals.append(b)
            temp_b_vals.append(b)
            # rss values:
            rss_vals.append(rss)
            temp_rss_vals.append(rss)
        # Getting minimum:
        min_index = np.argmin(temp_rss_vals)
        step_width = (upper_b - lower_b) / (n_steps - 1)
        min_b = temp_b_vals[min_index]
        lower_b = min_b - step_width
        upper_b = min_b + step_width
        
    df = pd.DataFrame({'a'  : a_vals,
                       'b'  : b_vals,
                       'rss': rss_vals})
    return df