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

def Extract_Data(database):
    data = pd.read_csv(database, delimiter=",")
    n_vals = np.array(data['n'])
    k_vals = np.array(data['k'])
    l_vals = np.array(data['l'])
    t_vals = np.array(data['t'])
    r_vals = np.array(data['r'])
    p_vals = np.array(data['p'])
    norm_vals = np.array(data['norm'])
    act_vals = np.array(data['active_count'])

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
    t = []  # store t values
    t_dict1 = {}  # stores {t: [(act, norms**p)]}
    for i in range(len(t_vals)):
        if t_vals[i] not in t:
            t.append(t_vals[i])
            t_dict1.update({t_vals[i]: [(act_vals[i], norm_vals[i]**p)]})
        else:
            t_dict1[t_vals[i]].append((act_vals[i], norm_vals[i]**p))
    
    # average Schatten norm
    avE = [] 
    for each in t:
        # first average over Schatten norms that have the same active counts
        # create a dist that stores {act: [norms**p that has the same act]}
        # the # of act appearing for this t is given by the length of the list
        act_dict = {} 
        for j in range(len(t_dict1[each])):
            if t_dict1[each][j][0] not in act_dict:
                act_dict.update({t_dict1[each][j][0]: [t_dict1[each][j][1]]})
            else:
                act_dict[t_dict1[each][j][0]].append(t_dict1[each][j][1])
        St = []  # stores [(# of act, av. Schatten norm of the same act)]
        for act in act_dict.keys():
            St.append( ( len(act_dict[act]), (sum(act_dict[act]) / len(act_dict[act]))**(1/p) ) ) 

        # then average over the active counts
        N = sum([s[0] for s in St])
        ## sanity check
        if N != len(t_dict1[each]):
            raise ValueError("Inconsequential averages.")
        avE.append( sum([(s[0]/N) * s[1] for s in St]) / ((2**(n//2))**(1/p)) )
            
    
    # compute the log values
    log_t_raw = [np.log(t[i]) for i in range(len(t))]
    log_E_raw = [np.log(avE[i]) for i in range(len(avE))]

    # keep only finite values
    log_t = [log_t_raw[i] for i in range(len(log_t_raw)) if np.isfinite(log_E_raw[i])]
    log_E = [log_E_raw[i] for i in range(len(log_E_raw)) if np.isfinite(log_E_raw[i])]

    # # fit the log values using log_fit
    # fit_params, _ = curve_fit(log_fit, log_t, log_E, bounds=[(0, -np.inf), (np.inf, np.inf)], maxfev=50000)
    # a,b = fit_params
    # fit_data = [log_fit(each, a,b) for each in log_t]

    return np.array(t), k, np.array(log_t), np.array(log_E)

def Theory_Upperbound(n,k,t,l,r,p,J_E):
    Gamma = comb(n,k)
    p_B = n / Gamma
    sigma = np.sqrt( (1/p_B) * (J_E)**2 * math.factorial(k-1) / (k * n**(k-1)) )

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
    # this is the upper bound of the higher-order average Trotter error
    # we have ommited the l-dependent factor beta(l), since we only care about the scaling
    head = (Gamma * np.sqrt(p) * sigma * np.sqrt(p_B) * t / np.sqrt(Qnk))
    body1 = (np.sqrt(p) * sigma * np.sqrt(p_B * Qnk) * t / r)**(l)
    body2 = Gamma * (np.sqrt(p) * sigma * np.sqrt(p_B * Qnk) * t / r)**(l+1)

    S = head * (body1 + body2)
    return S

def Prediction(n,k,t_set,l,r,p,J_E):
    logt_raw = [np.log(t) for t in t_set]
    pred_raw = [np.log(Theory_Upperbound(n,k,t,l,r,p,J_E)) for t in t_set]

    # keep only finite values
    logt = np.array([logt_raw[i] for i in range(len(logt_raw)) if np.isfinite(pred_raw[i])])
    pred = np.array([pred_raw[i] for i in range(len(pred_raw)) if np.isfinite(pred_raw[i])])

    # fit the log values using log_fit
    fit_params, _ = curve_fit(log_fit, logt, pred, bounds=[(0, -np.inf), (np.inf, np.inf)], maxfev=50000)
    a, b = fit_params
    fit_data = [log_fit(each, a, b) for each in logt]

    return logt, pred, fit_data, a, b

# ------------------------- Configuration -------------------------
J_E = 1.0
n = 10
k_set = [2,3,4,5,6]
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

DATABASES = [f"DATABASE/DATA_SPARSE_t_sweep(n={n},k={k},l={l},r={r},J={round(J_E,3)}).txt" for k in k_set]

# ------------------------- Plot the observed error ---------------

plt.figure(figsize=(10,6))
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 20})
# plt.xticks(n)

plt.xlabel(r'$\log_e(t)$')
ylabel = r'$\log_e(\langle|||e^{iHt} - S_{var_l}(t/r)^r|||_{\overline{var_p}}\rangle)$'
ylabel = ylabel.replace('var_l', str(l))
ylabel = ylabel.replace('var_p', str(p))
plt.ylabel(ylabel)

title = r'Observed average normalized Trotter Error for $l=var_l$ and $p=var_p$'
title = title.replace('var_l', str(l))
title = title.replace('var_p', str(p))
plt.title(title)

plotter = []
for DATABASE in DATABASES:
    t, k, log_t, log_E = Extract_Data(DATABASE)

    label = r'$k={var_k}$'
    label = label.replace('var_k', str(k))

    # plt.plot(log_n, fit_data, color=COLOR_dict[k], linestyle=':', label=label)

    # load the t values to t_set
    t_set = t

    # load the data points to plotter
    plotter.append((k, log_t, log_E, label))

LABEL = True
label_ref = r'reference $\sim {var_l}\log_e(t)$'
label_ref = label_ref.replace('var_l', str(l+1))
for each in plotter:
    # plt.plot(each[1], each[3], color=COLOR_dict[each[0]], linestyle=':', label=each[4])
    if LABEL:
        plt.plot(each[1], [(3 * i) - (3 * each[1][-1]) + each[2][-1] for i in each[1]], 
                color='red', linestyle=':', label=label_ref)
        LABEL = False
    else:
        plt.plot(each[1], [(3 * i) - (3 * each[1][-1]) + each[2][-1] for i in each[1]], 
                color='red', linestyle=':')
    plt.scatter(each[1], each[2], color=COLOR_dict[each[0]],label=each[3])
plt.legend()
plt.show()


# ------------------------- Plot the theoretical prediction -------

plt.figure(figsize=(10,6))
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 20})
# plt.xticks(n)

plt.xlabel(r'$\log_e(t)$')
ylabel = r'$\log_e\left(\overline{\Delta}_{var_l}\right)$'
ylabel = ylabel.replace('var_l', str(l))
plt.ylabel(ylabel)

title = r'Theoretical upper bound $\overline{\Delta}_l$ for $l=var_l$ and $p=var_p$'
title = title.replace('var_l', str(l))
title = title.replace('var_p', str(p))
plt.title(title)

plotter = []
for k in k_set:
    logt, pred, fit_data, a, b = Prediction(n,k,t_set,l,r,p,J_E)
    # plt.scatter(log_n, log_E, color=COLOR_dict[k])
    label = r'$k={var_k}$: ${var_a}\log(t) + {var_b}$'
    label = label.replace('var_k', str(k))
    label = label.replace('var_a', str(round(a,3)))
    label = label.replace('var_b', str(round(b,3)))
    # plt.plot(log_n, fit_data, color=COLOR_dict[k], linestyle=':', label=label)

    # load the data points to plotter
    plotter.append((k, logt, pred, fit_data, label))

for each in plotter:
    plt.scatter(each[1], each[2], color=COLOR_dict[each[0]])
    plt.plot(each[1], each[3], color=COLOR_dict[each[0]], linestyle=':', label=each[4])

plt.legend()
plt.show()


plt.figure(figsize=(10,6))
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 20})
# plt.xticks(n)

plt.xlabel(r'$t$')
ylabel = r'$\log_e\left(\overline{\Delta}_{var_l}\right)$'
ylabel = ylabel.replace('var_l', str(l))
plt.ylabel(ylabel)

title = r'Theoretical upper bound $\overline{\Delta}_l$ for $l=var_l$ and $p=var_p$'
title = title.replace('var_l', str(l))
title = title.replace('var_p', str(p))
plt.title(title)

plotter = []
for k in k_set:
    logt, pred, fit_data, a, b = Prediction(n,k,t_set,l,r,p,J_E)
    t = [np.exp(each) for each in logt]

    
    label = r'$k={var_k}$: exponent$\sim{var_a}$'
    label = label.replace('var_k', str(k))
    label = label.replace('var_a', str(round(a,3)))

    plotter.append((k, t, pred, label))


for each in plotter:
    plt.scatter(each[1], each[2], color=COLOR_dict[each[0]])
    plt.plot(each[1], each[2], color=COLOR_dict[each[0]], linestyle=':', label=each[3])

plt.legend()
plt.show()
