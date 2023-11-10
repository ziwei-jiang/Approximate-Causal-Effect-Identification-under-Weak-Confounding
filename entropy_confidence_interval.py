import math
import numpy as np
import cvxpy as cp
from scipy.special import xlogy
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


def bound_y0_dox0(pyx, ub=1):
    
    nx = pyx.shape[1]
    ny = pyx.shape[0]
    px = pyx.sum(axis=0)
    
    # uy_x: 1x(nx*ny) vector
    uy_x = cp.Variable(nx*ny)
    v1 = np.zeros(nx*ny)
    v1[0:nx] = px
    pydox = uy_x @ v1


    v2 = np.zeros((ny, nx*ny))
    for i in range(ny):
        v2[i, i*nx:(i+1)*nx] = px
    # qy: 1xny vector 
    qy = uy_x @ v2.T
    # qyx: 1x(nx*ny) vector
    qyx = qy @ v2

    v3 = np.zeros((nx*ny))
    for i in range(ny):
        v3[i*nx:(i+1)*nx] = px

    uyx = cp.multiply(uy_x, v3)
    dkl = cp.kl_div(uyx, qyx)/math.log(2)
    I = dkl @ np.ones((nx*ny))

    
    v4 = np.zeros((nx*ny, ny))
    for i in range(ny):
        v4[nx*i, i] = px[0]
    
    v5 = np.zeros((nx*ny, nx))
    for i in range(nx):
        v5[i:nx*ny:nx, i] = 1
    constraints  = [uy_x @ v4 == pyx[:,0],
                     uy_x@v5 == np.ones(nx), 
                     uy_x >= 0, uy_x <= 1,
                     I <= ub]

    max_obj = cp.Maximize(pydox)
    min_obj = cp.Minimize(pydox)

    max_prob = cp.Problem(max_obj, constraints)
    max_prob.solve(solver=cp.SCS)
    
    min_prob = cp.Problem(min_obj, constraints)
    min_prob.solve(solver=cp.SCS)

    
    
    return  min_prob.value, max_prob.value



# ######################################## Binary X,Y, sampling x and y_x ########################################

r = 5
c = 5
entr_step = 0.01
entr_ub = 1
yx = np.zeros([2, 2])

x_range = np.linspace(0.01, 1, r, endpoint=False)
y_x_range = np.linspace(0.001, 1, c, endpoint=True)

# figs = []
# for x0 in x_range:
#     print(f"x0: {x0}")
#     columns = []
#     for y0_x0 in y_x_range:
#         print(f"y0_x0 {y0_x0}")
#         yx[0, 0] = y0_x0 * x0
#         yx[1, 0] = x0 - y0_x0 *x0
#         yx[0, 1] = (1-x0)/2
#         yx[1, 1] = (1-x0)/2
        
#         ubs = []
#         lbs = []
#         ## loop from 1 to 0
#         for i in tqdm(np.arange(0, entr_ub+entr_step, entr_step)):
#             hz = entr_ub-i
#             # print(hz)
#             # hz = i
#             lb, ub = bound_y0_dox0(yx, ub=hz)
#             lbs.append(lb)
#             ubs.append(ub)


#         columns.append([lbs, ubs])

#     figs.append(columns)


# ## save figs as pickle
# with open('figs.pickle', 'wb') as f:
#     pickle.dump(figs, f)
entr = np.arange(0, entr_ub+entr_step, entr_step)
entr = entr[::-1]

## read figs from pickle
with open('figs.pickle', 'rb') as f:
    figs = pickle.load(f)


## plot the graphs in rxc grid
fig, axs = plt.subplots(r, c, figsize=(20, 20))
for i in range(r):
    for j in range(c):
        x0 = x_range[i]
        y0_x0 = y_x_range[j]
        tp_lb = [y0_x0*x0]*len(entr)
        tp_ub = [1-(x0-y0_x0*x0)]*len(entr)
        axs[i, j].set_title(f"p(Y, X)=[[{y0_x0*x0:.2f},{x0-y0_x0:.2f}],[{0.5*(1-x0):.2f},{0.5*(1-x0):.2f}]]")
        axs[i, j].set_title(f"p(y|x)= {y0_x0:.2f}, P(x)= {x0:.2f}")

        axs[i, j].plot(entr, figs[i][j][0], color='green')
        axs[i, j].plot(entr, figs[i][j][1], color='green')
        axs[i, j].plot(entr, tp_lb, color='red')
        axs[i, j].plot(entr, tp_ub, color='red')

        axs[i, j].invert_xaxis()
        axs[i, j].tick_params(axis='both', which='major', labelsize=20)
        axs[i, j].xaxis.label.set_fontsize(20)
        axs[i, j].yaxis.label.set_fontsize(20)
        axs[i, j].title.set_fontsize(20)
        ## set the range of y axis
        axs[i, j].set_ylim([-0.2, 1.2])
        
plt.show()


