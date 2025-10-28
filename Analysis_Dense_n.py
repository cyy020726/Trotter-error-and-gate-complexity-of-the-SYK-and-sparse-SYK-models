import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np 
from tqdm.auto import trange 
from math import comb
import scipy.sparse as sparse
from itertools import combinations 
from scipy.optimize import curve_fit
from decimal import Decimal

# to check if all elements of an array is the same
def is_all_equal(a: np.ndarray) -> int:
    a = np.asarray(a)
    if a.size != 0:
        if a.max() != a.min():
            raise ValueError("Array elements are not all equal.")

def Theory_Upperbound(n,k,t,l,r,p,J_E):
    sigma = np.sqrt( (J_E)**2 * math.factorial(k-1) / (k * n**(k-1)) )
    if l==1:
        mu = max([0, k - (n- k)])
        Qnk = 0
        if k % 2 == 0:
            for s in range(mu,k):
                if s % 2 != 0:
                    Qnk += comb(n - k, k-s) * comb(k,s)
        else:
            for s in range(mu,k):
                if s % 2 == 0:
                    Qnk += comb(n - k, k-s) * comb(k,s)
        if comb(n,k) == 0 or Qnk == 0:
            S = 0
        else:
            # this is the upper bound of the first-order Trotter error
            S =  4 * np.sqrt(2) * p**2 * sigma**2 * np.sqrt( 
                comb(n,k) * Qnk) * t**2 * ((1/ (2*r)) + sigma * np.sqrt(Qnk) * (t / (3 * r**2)) )
        return S
    else:
        Cp = p - 1
        Upsilon = 2 * 5**((l/2) - 1)
        mu = max([0, k - (n- k)])
        Qnk = 0
        if k % 2 == 0:
            for s in range(mu,k):
                if s % 2 != 0:
                    Qnk += comb(n - k, k-s) * comb(k,s)
        else:
            for s in range(mu,k):
                if s % 2 == 0:
                    Qnk += comb(n - k, k-s) * comb(k,s)
        # this is the upper bound of the higher-order Trotter error
        Cl = Upsilon**(l+3) * np.sqrt(l+3) * (l+2)**(3 * (l+2) - 1) / (l+1)
        if Qnk == 0:
            return 0
        else:
            S = Cl * np.sqrt(Cp) * sigma * Qnk**(-1/2) * t * ( 
                comb(n,k) * ( np.sqrt(Cp) * sigma * np.sqrt(Qnk) * t / r)**(l) + 
                comb(n,k)**2 * ( np.sqrt(Cp) * sigma * np.sqrt(Qnk) * t / r )**(l+1) )
        
        return S 
    
def lin_fit(x, a, b):
    return a * x + b

def Extract_Data(database):
    data = pd.read_csv(database, delimiter=",")
    n_vals = np.array(data['n'])
    k_vals = np.array(data['k'])
    l_vals = np.array(data['l'])
    t_vals = np.array(data['t'])
    r_vals = np.array(data['r'])
    p_vals = np.array(data['p'])
    norm_vals = np.array(data['norm'])

    # check if data is consistent, i.e. k,l,t,r,p should be kept at constant
    is_all_equal(k_vals)
    is_all_equal(l_vals)
    is_all_equal(t_vals)
    is_all_equal(r_vals)
    is_all_equal(p_vals)

    # extract the values of simulation params
    k = k_vals[0]
    l = l_vals[0]
    t = t_vals[0]
    r = r_vals[0]
    p = p_vals[0]

    # extract the values of n and norm
    n = []  # stores the n values
    n_dict = {}  # keeps track of how many times each n value appear
    norm_dict = {}  # stores the sum over \norm{\cdots}_p^p for each n
    for i in range(len(n_vals)):
        if n_vals[i] not in n:
            n.append(n_vals[i])
            n_dict.update({n_vals[i]: 1})
            norm_dict.update({n_vals[i]:norm_vals[i]**p})
        else:
            n_dict[n_vals[i]] += 1
            norm_dict[n_vals[i]] += norm_vals[i]**p
    
    # extract the normalized expected Schatten norm
    E = [(norm_dict[each]/n_dict[each])**(1/p) / ((2**(each//2))**(1/p)) for each in n]
    
    # compute the Theoretical upper bounds 
    pred = [Theory_Upperbound(n_i,k,t,l,r,p,J_E) for n_i in n]

    # compute the regularized ratio (remove NaN terms)
    n_reg = []
    est = []  # estimators
    for i in range(len(n)):
        if pred[i] > 0:
            n_reg.append(n[i])
            est.append(E[i] / pred[i])
    
    # compute the log values
    log_n_reg = [np.log(each) for each in n_reg]
    log_est = [np.log(each) for each in est]

    # # fit the log values using log_fit
    # fit_params, _ = curve_fit(lin_fit, log_n_reg, log_est, bounds=[(-np.inf, -np.inf), (np.inf, np.inf)], maxfev=50000)
    # a,b = fit_params
    # fit_data = [lin_fit(each, a,b) for each in log_n_reg]

    return k, np.array(n_reg), np.array(log_est)
        

# ------------------------- Configuration -------------------------
J_E = 1.0
n_set = []  # dummy variable to hold the n values from data extraction
k_set = [2,3,4,5,6]
t = 10.0
l = 2                 
r = 100000               
p = 2    

COLORS = [
    "#1f77b4", "#aec7e8",
    "#ff7f0e", "#ffbb78",
    "#2ca02c", "#98df8a",
    "#d62728", "#ff9896",
    "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94",
    "#e377c2", "#f7b6d2",
    "#7f7f7f", "#c7c7c7",
    "#bcbd22", "#dbdb8d",
    "#17becf", "#9edae5",
]

# define a color map based on locality k
COLOR_dict = {}
for i in range(len(k_set)):
    j = 2 * i
    COLOR_dict.update({k_set[i]: COLORS[j]})

DATABASES = [f"DATABASE/DATA_DENSE_n_sweep(k={k},t={t},l={l},r={r},p={p},J={round(J_E,3)}).txt" for k in k_set]

# ------------------------- Plot the observed error ---------------

plt.figure(figsize=(10,6))
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 20})
# plt.xticks(n)

plt.xlabel(r'$n$')
ylabel = r'$\log\left(\eta^{\mathrm{dense}}_{l,p}\right)$'
ylabel = ylabel.replace('var_l', str(l))
ylabel = ylabel.replace('var_p', str(p))
plt.ylabel(ylabel)

title = r'Error ratio $\eta^{\mathrm{dense}}_{l,p}$ for $l=var_l$ and $p=var_p$'
title = title.replace('var_l', str(l))
title = title.replace('var_p', str(p))
plt.title(title)

plotter = []
for DATABASE in DATABASES:
    k, n_reg, log_est = Extract_Data(DATABASE)
    # load the data points to plotter
    plotter.append((k, n_reg, log_est))

for each in plotter:
    plt.scatter(each[1], each[2], color=COLOR_dict[each[0]])
    plt.plot(each[1], each[2], color=COLOR_dict[each[0]], label=f'k={each[0]}')

plt.legend()
plt.show()

