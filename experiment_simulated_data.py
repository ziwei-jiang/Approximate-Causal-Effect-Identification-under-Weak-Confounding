import math
import numpy as np
import cvxpy as cp
from scipy.special import xlogy
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import StrMethodFormatter
from matplotlib.transforms import Affine2D
from utils import generate_conditional_dist, generate_dirichlet
## change another color option for the plots
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Paired.colors)



        
def get_plot_data(num_dist, z_states, x_states, y_stats, alpha_val):


    ubs = np.load(f'n{num_dist}z{z_states}x{x_states}y{y_states}a0{alpha_val}_ubs.npy')
    lbs = np.load(f'n{num_dist}z{z_states}x{x_states}y{y_states}a0{alpha_val}_lbs.npy')
    tp_lbs = np.load(f'n{num_dist}z{z_states}x{x_states}y{y_states}a0{alpha_val}_tp_lbs.npy')
    tp_ubs = np.load(f'n{num_dist}z{z_states}x{x_states}y{y_states}a0{alpha_val}_tp_ubs.npy')
    Hz = np.load(f'n{num_dist}z{z_states}x{x_states}y{y_states}a0{alpha_val}_entropy_z.npy')
    tighter_bound = np.load(f'n{num_dist}z{z_states}x{x_states}y{y_states}a0{alpha_val}_tighter_bound_index.npy')
    causal_effect = np.load(f'n{num_dist}z{z_states}x{x_states}y{y_states}a0{alpha_val}_causal_effect.npy')

    hz_01 = np.where(Hz<0.1)
    hz_02 = np.where((Hz>=0.1) & (Hz<0.2))
    hz_03 = np.where((Hz>=0.2) & (Hz<0.3))
    hz_04 = np.where((Hz>=0.3) & (Hz<0.4))
    hz_05 = np.where((Hz>=0.4) & (Hz<0.5))
    hz_06 = np.where((Hz>=0.5) & (Hz<0.6))
    hz_07 = np.where((Hz>=0.6) & (Hz<0.7))
    hz_08 = np.where((Hz>=0.7) & (Hz<0.8))
    hz_09 = np.where((Hz>=0.8) & (Hz<0.9))
    hz_10 = np.where((Hz>=0.9) & (Hz<1.0))



    ub_01 = np.array(ubs)[hz_01]
    lb_01 = np.array(lbs)[hz_01]
    tp_lb_01 = np.array(tp_lbs)[hz_01]
    tp_ub_01 = np.array(tp_ubs)[hz_01]
    gap_01 = ub_01 - lb_01
    gap_tp_01 = tp_ub_01 - tp_lb_01
    ave_gap_01 = np.mean(gap_01)
    ave_gap_tp_01 = np.mean(gap_tp_01)
    mid_01 = (ub_01 + lb_01)/2
    tp_mid_01 = (tp_ub_01 + tp_lb_01)/2
    causal_effect_01 = np.array(causal_effect)[hz_01]
    error_01 = gap_01.std() *1.96/ np.sqrt(gap_01.shape[0])
    tp_error_01 = gap_tp_01.std() *1.96/ np.sqrt(gap_01.shape[0])


    ub_02 = np.array(ubs)[hz_02]
    lb_02 = np.array(lbs)[hz_02]
    tp_lb_02 = np.array(tp_lbs)[hz_02]
    tp_ub_02 = np.array(tp_ubs)[hz_02]
    gap_02 = ub_02 - lb_02
    gap_tp_02 = tp_ub_02 - tp_lb_02
    ave_gap_02 = np.mean(gap_02)
    ave_gap_tp_02 = np.mean(gap_tp_02)
    mid_02 = (ub_02 + lb_02)/2
    tp_mid_02 = (tp_ub_02 + tp_lb_02)/2
    causal_effect_02 = np.array(causal_effect)[hz_02]
    error_02 = gap_02.std() *1.96/ np.sqrt(gap_01.shape[0])
    tp_error_02 = gap_tp_02.std() *1.96/ np.sqrt(gap_01.shape[0])

    ub_03 = np.array(ubs)[hz_03]
    lb_03 = np.array(lbs)[hz_03]
    tp_lb_03 = np.array(tp_lbs)[hz_03]
    tp_ub_03 = np.array(tp_ubs)[hz_03]
    gap_03 = ub_03 - lb_03
    gap_tp_03 = tp_ub_03 - tp_lb_03
    ave_gap_03 = np.mean(gap_03)
    ave_gap_tp_03 = np.mean(gap_tp_03)
    mid_03 = (ub_03 + lb_03)/2
    tp_mid_03 = (tp_ub_03 + tp_lb_03)/2
    causal_effect_03 = np.array(causal_effect)[hz_03]
    error_03 = gap_03.std() *1.96/ np.sqrt(gap_01.shape[0])
    tp_error_03 = gap_tp_03.std() *1.96/ np.sqrt(gap_01.shape[0])


    ub_04 = np.array(ubs)[hz_04]
    lb_04 = np.array(lbs)[hz_04]
    tp_lb_04 = np.array(tp_lbs)[hz_04]
    tp_ub_04 = np.array(tp_ubs)[hz_04]
    gap_04 = ub_04 - lb_04
    gap_tp_04 = tp_ub_04 - tp_lb_04
    ave_gap_04 = np.mean(gap_04)
    ave_gap_tp_04 = np.mean(gap_tp_04)
    mid_04 = (ub_04 + lb_04)/2
    tp_mid_04 = (tp_ub_04 + tp_lb_04)/2
    causal_effect_04 = np.array(causal_effect)[hz_04]
    error_04 = gap_04.std() *1.96/ np.sqrt(gap_01.shape[0])
    tp_error_04 = gap_tp_04.std() *1.96/ np.sqrt(gap_01.shape[0])
    ub_05 = np.array(ubs)[hz_05]
    lb_05 = np.array(lbs)[hz_05]
    tp_lb_05 = np.array(tp_lbs)[hz_05]
    tp_ub_05 = np.array(tp_ubs)[hz_05]
    gap_05 = ub_05 - lb_05
    gap_tp_05 = tp_ub_05 - tp_lb_05
    ave_gap_05 = np.mean(gap_05)
    ave_gap_tp_05 = np.mean(gap_tp_05)
    mid_05 = (ub_05 + lb_05)/2
    tp_mid_05 = (tp_ub_05 + tp_lb_05)/2
    causal_effect_05 = np.array(causal_effect)[hz_05]
    error_05 = gap_05.std() *1.96/ np.sqrt(gap_01.shape[0])
    tp_error_05 = gap_tp_05.std() *1.96/ np.sqrt(gap_01.shape[0])

    ub_06 = np.array(ubs)[hz_06]
    lb_06 = np.array(lbs)[hz_06]
    tp_lb_06 = np.array(tp_lbs)[hz_06]
    tp_ub_06 = np.array(tp_ubs)[hz_06]
    gap_06 = ub_06 - lb_06
    gap_tp_06 = tp_ub_06 - tp_lb_06
    ave_gap_06 = np.mean(gap_06)
    ave_gap_tp_06 = np.mean(gap_tp_06)
    mid_06 = (ub_06 + lb_06)/2
    tp_mid_06 = (tp_ub_06 + tp_lb_06)/2
    causal_effect_06 = np.array(causal_effect)[hz_06]
    error_06 = gap_06.std() *1.96/ np.sqrt(gap_01.shape[0])
    tp_error_06 = gap_tp_06.std() *1.96/ np.sqrt(gap_01.shape[0])

    ub_07 = np.array(ubs)[hz_07]
    lb_07 = np.array(lbs)[hz_07]
    tp_lb_07 = np.array(tp_lbs)[hz_07]
    tp_ub_07 = np.array(tp_ubs)[hz_07]
    gap_07 = ub_07 - lb_07
    gap_tp_07 = tp_ub_07 - tp_lb_07
    ave_gap_07 = np.mean(gap_07)
    ave_gap_tp_07 = np.mean(gap_tp_07)
    mid_07 = (ub_07 + lb_07)/2
    tp_mid_07 = (tp_ub_07 + tp_lb_07)/2
    causal_effect_07 = np.array(causal_effect)[hz_07]
    error_07 = gap_07.std() *1.96/ np.sqrt(gap_01.shape[0])
    tp_error_07 = gap_tp_07.std() *1.96/ np.sqrt(gap_01.shape[0])


    ub_08 = np.array(ubs)[hz_08]
    lb_08 = np.array(lbs)[hz_08]
    tp_lb_08 = np.array(tp_lbs)[hz_08]
    tp_ub_08 = np.array(tp_ubs)[hz_08]
    gap_08 = ub_08 - lb_08
    gap_tp_08 = tp_ub_08 - tp_lb_08
    ave_gap_08 = np.mean(gap_08)
    ave_gap_tp_08 = np.mean(gap_tp_08)
    mid_08 = (ub_08 + lb_08)/2
    tp_mid_08 = (tp_ub_08 + tp_lb_08)/2
    causal_effect_08 = np.array(causal_effect)[hz_08]
    error_08 = gap_08.std() *1.96/ np.sqrt(gap_01.shape[0])
    tp_error_08 = gap_tp_08.std() *1.96/ np.sqrt(gap_01.shape[0])


    ub_09 = np.array(ubs)[hz_09]
    lb_09 = np.array(lbs)[hz_09]
    tp_lb_09 = np.array(tp_lbs)[hz_09]
    tp_ub_09 = np.array(tp_ubs)[hz_09]
    gap_09 = ub_09 - lb_09
    gap_tp_09 = tp_ub_09 - tp_lb_09
    ave_gap_09 = np.mean(gap_09)
    ave_gap_tp_09 = np.mean(gap_tp_09)
    mid_09 = (ub_09 + lb_09)/2
    tp_mid_09 = (tp_ub_09 + tp_lb_09)/2
    causal_effect_09 = np.array(causal_effect)[hz_09]
    error_09 = gap_09.std() *1.96/ np.sqrt(gap_01.shape[0])
    tp_error_09 = gap_tp_09.std() *1.96/ np.sqrt(gap_01.shape[0])


    ub_10 = np.array(ubs)[hz_10]
    lb_10 = np.array(lbs)[hz_10]
    tp_lb_10 = np.array(tp_lbs)[hz_10]
    tp_ub_10 = np.array(tp_ubs)[hz_10]
    gap_10 = ub_10 - lb_10
    gap_tp_10 = tp_ub_10 - tp_lb_10
    ave_gap_10 = np.mean(gap_10)
    ave_gap_tp_10 = np.mean(gap_tp_10)
    mid_10 = (ub_10 + lb_10)/2
    tp_mid_10 = (tp_ub_10 + tp_lb_10)/2
    causal_effect_10 = np.array(causal_effect)[hz_10]
    error_10 = gap_10.std() *1.96/ np.sqrt(gap_01.shape[0])
    tp_error_10 = gap_tp_10.std() *1.96/ np.sqrt(gap_01.shape[0])

    ## let x be a numpy array of strings
    x = np.array(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])

    ## plot the ave_gap and ave_gap_tp 
    ave_gap = np.array([ave_gap_01, ave_gap_02, ave_gap_03, ave_gap_04, ave_gap_05, ave_gap_06, ave_gap_07, ave_gap_08, ave_gap_09, ave_gap_10])
    ave_gap_tp = np.array([ave_gap_tp_01, ave_gap_tp_02, ave_gap_tp_03, ave_gap_tp_04, ave_gap_tp_05, ave_gap_tp_06, ave_gap_tp_07, ave_gap_tp_08, ave_gap_tp_09, ave_gap_tp_10])

    errors = np.array([np.abs(error_01).mean(), np.abs(error_02).mean(), np.abs(error_03).mean(), np.abs(error_04).mean(), np.abs(error_05).mean(), np.abs(error_06).mean(), np.abs(error_07).mean(), np.abs(error_08).mean(), np.abs(error_09).mean(), np.abs(error_10).mean()])
    tp_errors = np.array([np.abs(tp_error_01).mean(), np.abs(tp_error_02).mean(), np.abs(tp_error_03).mean(), np.abs(tp_error_04).mean(), np.abs(tp_error_05).mean(), np.abs(tp_error_06).mean(), np.abs(tp_error_07).mean(), np.abs(tp_error_08).mean(), np.abs(tp_error_09).mean(), np.abs(tp_error_10).mean()])

    ## remove the nan value from ave_gap and ave_gap_tp by simply removing those terms
    idx = np.where(np.isnan(ave_gap)==False)
    ave_gap = ave_gap[idx]
    ave_gap_tp = ave_gap_tp[idx]
    x = x[idx]
    return ave_gap, ave_gap_tp, errors, tp_errors, x


fig, ax = plt.subplots()
fig2, ax2 = plt.subplots(figsize=(20, 10))

num_dist = 1000  #n
z_states = 5    #k
x_states = 10    #j
y_states = 2    #i
alpha_val = 1

ave_gap, ave_gap_tp, errors, tp_errors, x = get_plot_data(num_dist, z_states, x_states, y_states, alpha_val)


## plot error bar
ax.errorbar(x, ave_gap_tp, yerr=tp_errors, marker='^', markersize=15, label = 'Tian-Pearl bounds |X|=10, |Y|=2', linewidth=5, linestyle='--', transform=trans2)
ax.errorbar(x, ave_gap, yerr=errors, marker='^', markersize=15, label = 'Our bounds |X|=10, |Y|=2', linewidth=5, transform=trans1)

ax2.bar(x, all_bounds, label='Total number of samples |X|=10, |Y|=2')
ax2.bar(x, tighter_bounds, label='Number of samples with tighter bounds |X|=10, |Y|=2')






num_dist = 1000  #n
z_states = 5    #k
x_states = 2    #j
y_states = 2    #i
alpha_val = 1

ave_gap, ave_gap_tp, errors, tp_errors, x = get_plot_data(num_dist, z_states, x_states, y_states, alpha_val)

## plot error bar
ax.errorbar(x, ave_gap_tp, yerr=tp_errors, marker='d', markersize=15, label = 'Tian-Pearl bounds |X|=2, |Y|=2', linewidth=5, linestyle='--', transform=trans2)
ax.errorbar(x, ave_gap, yerr=errors, marker='d', markersize=15, label = 'Our bounds |X|=2, |Y|=2', linewidth=5, transform=trans1)

ax2.bar(x, all_bounds, label='Total number of samples |X|=2, |Y|=2')
ax2.bar(x, tighter_bounds, label = 'Number of samples with tighter bounds |X|=2, |Y|=2')





num_dist = 1000  #n
z_states = 5    #k
x_states = 2    #j
y_states = 10    #i
alpha_val = 1

ave_gap, ave_gap_tp, errors, tp_errors, x = get_plot_data(num_dist, z_states, x_states, y_states, alpha_val)

ax.errorbar(x, ave_gap_tp, marker='x', markersize=15, yerr=tp_errors, label = 'Tian-Pearl bounds |X|=2, |Y|=10', linewidth=5, linestyle='--', transform=trans2)
ax.errorbar(x, ave_gap, marker='x', markersize=15, yerr=errors, label = 'Our bounds |X|=2, |Y|=10', linewidth=5, transform=trans1)

ax2.bar(x, all_bounds, label='Total number of samples |X|=2, |Y|=10')
ax2.bar(x, tighter_bounds, label = 'Number of samples with tighter bounds |X|=2, |Y|=10')


plt.xticks(fontsize=25)
plt.yticks(fontsize=25)


plt.xlabel('Entropy of confounder', fontsize=25)
plt.ylabel('Average gap between bounds', fontsize=25)
plt.legend(fontsize=25)
## turn on the grid
plt.grid(True)
plt.show()
