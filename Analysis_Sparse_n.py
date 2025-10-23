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
    act_vals = np.array(data['active_count'])
    
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
    n = []  # store n values
    n_dict1 = {}  # stores {n: [(act, norms**p)]}
    for i in range(len(n_vals)):
        if n_vals[i] not in n:
            n.append(n_vals[i])
            n_dict1.update({n_vals[i]: [(act_vals[i], norm_vals[i]**p)]})
        else:
            n_dict1[n_vals[i]].append((act_vals[i], norm_vals[i]**p))
    
    # average Schatten norm
    avE = [] 
    for each in n:
        # first average over Schatten norms that have the same active counts
        # create a dist that stores {act: [norms**p that has the same act]}
        # the # of act appearing for this n is given by the length of the list
        act_dict = {} 
        for j in range(len(n_dict1[each])):
            if n_dict1[each][j][0] not in act_dict:
                act_dict.update({n_dict1[each][j][0]: [n_dict1[each][j][1]]})
            else:
                act_dict[n_dict1[each][j][0]].append(n_dict1[each][j][1])
        Sn = []  # stores [(# of act, av. Schatten norm of the same act)]
        for act in act_dict.keys():
            Sn.append( ( len(act_dict[act]), (sum(act_dict[act]) / len(act_dict[act]))**(1/p) ) ) 

        # then average over the active counts
        N = sum([s[0] for s in Sn])

        # sanity check
        if N != len(n_dict1[each]):
            raise ValueError("Inconsequential averages.")
        avE.append( sum([(s[0]/N) * s[1] for s in Sn]) / ((2**(each//2))**(1/p)) )
    
    # compute the Theoretical upper bounds 
    pred = [Theory_Upperbound(n_i,k,t,l,r,p,J_E) for n_i in n]

    # compute the regularized ratio (remove NaN terms)
    n_reg = []
    est = []  # estimators
    for i in range(len(n)):
        if pred[i] > 0:
            n_reg.append(n[i])
            est.append(avE[i] / pred[i])
    
    # compute the log values
    log_n_reg = [np.log(each) for each in n_reg]
    log_est = [np.log(each) for each in est]

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

DATABASES = [f"DATABASE/DATA_SPARSE_n_sweep(k={k},t={t},l={l},r={r},p={p},J={round(J_E,3)}).txt" for k in k_set]

# ------------------------- Plot the observed error ---------------
plt.figure(figsize=(10,6))
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 20})
# plt.xticks(n)

plt.xlabel(r'$n$')
ylabel = r'$\log\left(\eta^{\mathrm{sparse}}_{l,p}\right)$'
ylabel = ylabel.replace('var_l', str(l))
ylabel = ylabel.replace('var_p', str(p))
plt.ylabel(ylabel)

title = r'Error ratio $\eta^{\mathrm{sparse}}_{l,p}$ for $l=var_l$ and $p=var_p$'
title = title.replace('var_l', str(l))
title = title.replace('var_p', str(p))
plt.title(title)

plotter = []
for DATABASE in DATABASES:
    k, log_n_reg, log_est = Extract_Data(DATABASE)

    # load the data points to plotter
    plotter.append((k, log_n_reg, log_est))

for each in plotter:
    plt.scatter(each[1], each[2], color=COLOR_dict[each[0]])
    plt.plot(each[1], each[2], color=COLOR_dict[each[0]], label=f'k={each[0]}')

plt.legend()
plt.show()

