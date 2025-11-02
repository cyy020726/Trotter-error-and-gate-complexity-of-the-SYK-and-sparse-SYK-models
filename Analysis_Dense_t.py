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

def log_fit(x, a, b):
    return a * x + b

# def normal_fit(x, a, b, c):
#     return b * (x)**(a) + c

def Extract_Data(database):
    data = pd.read_csv(database, delimiter=",")
    n_vals = np.array(data['n'])
    k_vals = np.array(data['k'])
    l_vals = np.array(data['l'])
    t_vals = np.array(data['t'])
    r_vals = np.array(data['r'])
    p_vals = np.array(data['p'])
    norm_vals = np.array(data['norm'])

    # check if data is consistent, i.e. n,k,l,r,p should be kept at constant
    is_all_equal(n_vals)
    is_all_equal(k_vals)
    is_all_equal(l_vals)
    is_all_equal(r_vals)
    is_all_equal(p_vals)

    # extract the values of simulation params
    n = n_vals[0]
    k = k_vals[0]
    l = l_vals[0]
    r = r_vals[0]
    p = p_vals[0]

    # extract the values of t and norm
    t = []  # stores the t values
    t_dict = {}  # keeps track of how many times each t value appear
    norm_dict = {}  # stores the sum over \norm{\cdots}_p^p for each t
    for i in range(len(t_vals)):
        # start from t = epsilon
        if t_vals[i] > 0.001:
            if t_vals[i] not in t:
                t.append(t_vals[i])
                t_dict.update({t_vals[i]: 1})
                norm_dict.update({t_vals[i]:norm_vals[i]**p})
            else:
                t_dict[t_vals[i]] += 1
                norm_dict[t_vals[i]] += norm_vals[i]**p
    
    # extract the normalized expected Schatten norm
    E = [(norm_dict[each]/t_dict[each])**(1/p) / ((2**(n//2))**(1/p)) for each in t]
    
    # compute the log values
    log_t = [np.log(each) for each in t]
    log_E = [np.log(each) for each in E]

    # # fit the log values using log_fit
    # fit_params, _ = curve_fit(log_fit, log_t, log_E, bounds=([0, -np.inf], np.inf), maxfev=50000)
    # a, b = fit_params
    # fit_data = [log_fit(each, a, b) for each in log_t]

    return np.array(t), k, np.array(log_t), np.array(log_E)

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
        
        S = Cl * np.sqrt(Cp) * sigma * Qnk**(-1/2) * t * ( 
            comb(n,k) * ( np.sqrt(Cp) * sigma * np.sqrt(Qnk) * t / r)**(l) + 
            comb(n,k)**2 * ( np.sqrt(Cp) * sigma * np.sqrt(Qnk) * t / r )**(l+1) )
        
        return S 
        

def Prediction(n,k,t_set,l,r,p,J_E):
    logt = np.array([np.log(each) for each in t_set])
    pred = np.array([np.log(Theory_Upperbound(n,k,t,l,r,p,J_E)) for t in t_set])

    # fit the log values using log_fit
    fit_params, _ = curve_fit(log_fit, logt, pred, bounds=([0, -np.inf], np.inf), maxfev=50000)
    a, b = fit_params
    fit_data = [log_fit(each, a, b) for each in logt]

    return logt, pred, fit_data, a, b

# ------------------------- Configuration -------------------------
J_E = 1.0
n = 10
k_set = [2,3,4]
t_set = []  # dummy variable to hold the t values from data extraction
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

DATABASES = [f"DATABASE/DATA_DENSE_t_sweep(n={n},k={k},l={l},r={r},J={round(J_E,3)}).txt" for k in k_set]

# ------------------------- Plot the observed error ---------------

plt.figure(figsize=(10,6))
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 20})
# plt.xticks(n)

label_fontsize = 25
plt.xlabel(r'$t$', fontsize = label_fontsize)
ylabel = r'$\log_e(|||e^{iHt} - S_{var_l}(t/r)^r|||_{\overline{var_p}})$'
ylabel = ylabel.replace('var_l', str(l))
ylabel = ylabel.replace('var_p', str(p))
plt.ylabel(ylabel, fontsize = label_fontsize)

title = r'Observed normalized Trotter Error for $l=var_l$ and $p=var_p$'
title = title.replace('var_l', str(l))
title = title.replace('var_p', str(p))
plt.title(title)

plotter = []
for DATABASE in DATABASES:
    t, k, log_t, log_E = Extract_Data(DATABASE)

    label = r'$k={var_k}$'
    label = label.replace('var_k', str(k))

    # load the t values to t_set
    t_set = t

    # load the data points to plotter
    plotter.append((k, log_t, log_E, label))

def ref_line(x,x_f,y_f,l):
    return (l * x) - (l * x_f) + y_f

def cost(x_vals, y_vals, l):
    x_f = x_vals[-1]
    y_f = y_vals[-1]
    return sum([(y_vals[i] - (ref_line(x_vals[i], x_f, y_f, l)))**2 for i in range(len(x_vals))])

def reference_exponent(data):
    L = np.linspace(l-1, l+2, 1000)
    x_vals = data[1]
    y_vals = data[2]
    costs = [cost(x_vals, y_vals, L_i) for L_i in L]
    L_min = L[costs.index(min(costs))]
    return L_min

for each in plotter:
    label_ref = r': $\sim t^{{var_l}}$'
    l_ref = reference_exponent(each)
    label_ref = label_ref.replace('var_l', str(round(l_ref,3)))
    
    plt.plot(t_set, [ref_line(i, each[1][-1], each[2][-1], l_ref) for i in each[1]], 
                color='red', linestyle=':')
    plt.scatter(t_set, each[2], color=COLOR_dict[each[0]],label=each[3]+label_ref)
plt.legend()
plt.show()


# ------------------------- Plot the theoretical prediction -------

plt.figure(figsize=(10,6))
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 20})
# plt.xticks(n)

plt.xlabel(r'$t$', fontsize = label_fontsize)
ylabel = r'$\log_e\left(\Delta_{var_l}^{\mathrm{dense}}\right)$'
ylabel = ylabel.replace('var_l', str(l))
plt.ylabel(ylabel, fontsize = label_fontsize)

title = r'Theoretical upper bound $\Delta_l^{\mathrm{dense}}$ for $l=var_l$ and $p=var_p$'
title = title.replace('var_l', str(l))
title = title.replace('var_p', str(p))
plt.title(title)

plotter = []
for k in k_set:
    logt, pred, fit_data, a, b = Prediction(n,k,t_set,l,r,p,J_E)
    t = [np.exp(each) for each in logt]

    
    label = r'$k={var_k}$: $\sim t^{{var_a}}$'
    label = label.replace('var_k', str(k))
    label = label.replace('var_a', str(round(a,3)))

    plotter.append((k, t, pred, label))


for each in plotter:
    plt.scatter(each[1], each[2], color=COLOR_dict[each[0]])
    plt.plot(each[1], each[2], color=COLOR_dict[each[0]], linestyle=':', label=each[3])

plt.legend()
plt.show()