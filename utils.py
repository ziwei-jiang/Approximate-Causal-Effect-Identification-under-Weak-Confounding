import math
import numpy as np
import cvxpy as cp
from scipy.special import xlogy
import scipy.stats as stats
import matplotlib.pyplot as plt



def entropy_dist(X):
    ## Ignore zero values in distributions
    return -np.sum(X*np.log2(X, where=(X!=0)), axis=1)


def mutual_information_dist(XY):
    X = XY.sum(axis=1)
    Y = XY.sum(axis=2)
    return entropy_dist(X) + entropy_dist(Y) - entropy_dist(XY.reshape(XY.shape[0], -1))



## Generate a distribution of from dirichlet distribution
def generate_dirichlet(alpha, num_states, num_dist):
    return stats.dirichlet.rvs([alpha]*num_states, size=num_dist)


def generate_conditional_dist(num_states, num_parents_states, num_dist):
    v = 1/np.arange(1,num_states+1)
    v = v/np.sum(v)
    conditional_dist = np.zeros((num_dist, num_states, num_parents_states))
    ## Create shifted vectors
    for i in range(num_parents_states):
        vi = np.roll(v, i)
        # vi = np.ones(num_states)
        # vi = np.ones(num_states)*10
        dist = stats.dirichlet.rvs(vi, size=num_dist)
        conditional_dist[:,:,i] = dist
    return conditional_dist
        


def optimization_cf(pyx, ub=1, p =0, q=0):
    ##### for P(Y=p|do(X=q)) #####
    nx = pyx.shape[1]
    ny = pyx.shape[0]
    px = pyx.sum(axis=0)
    # uy_x: 1x(nx*ny) vector
    uy_x = cp.Variable(nx*ny)
    v1 = np.zeros(nx*ny)
    v1[p*nx:p*nx+nx] = px
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
    I = cp.sum(dkl)
    v4 = np.zeros((nx*ny, ny))
    for i in range(ny):
        v4[nx*i+q, i] = px[q]
    v5 = np.zeros((nx*ny, nx))
    for i in range(nx):
        v5[i:nx*ny:nx, i] = 1
    constraints  = [uy_x @ v4 == pyx[:,q],
                     uy_x@v5 == np.ones(nx), 
                     uy_x >= 0, uy_x <= 1,
                     I <= ub]
    max_obj = cp.Maximize(pydox)
    min_obj = cp.Minimize(pydox)
    t = 0
    max_prob = cp.Problem(max_obj, constraints)
    max_prob.solve(solver=cp.SCS)
    t += max_prob.solver_stats.solve_time
    min_prob = cp.Problem(min_obj, constraints)
    min_prob.solve(solver=cp.SCS)
    t += min_prob.solver_stats.solve_time
    return min_prob.value, max_prob.value, t



def optimization_cp(pyx, ub=1, p=0, q=0):
    ##### for P(Y=p|do(X=q)) #####
    nrx = pyx.shape[1]
    y_shape = pyx.shape[0]
    px = pyx.sum(axis=0)
    ry_shape = y_shape**nrx    
    ry_x = cp.Variable((ry_shape,nrx))
    v0 = np.zeros(ry_shape)
    ## assign the first half of the vector to be 1
    for i in range(0, ry_shape//2):
        v0[i] = 1
    v1 = 1 - v0
    pydox = ry_x@px @v0
    vx = np.zeros_like(px)
    vx[q] = px[q]
    ## make a vector of px with the same shape as ry_x with stack
    v4 = np.repeat(px[np.newaxis,:], ry_x.shape[0], axis=0)
    ryx = cp.multiply(ry_x, v4)
    ry = ry_x @ px
    qy = cp.vstack([ry]*nrx).T
    qyx = cp.multiply(qy, v4)
    dkl = cp.kl_div(ryx, qyx)/math.log(2)
    I = cp.sum(dkl)
    t = 0
    constraints = [cp.sum(ry_x@px)== 1, ry_x >= 0, ry_x <= 1, (ry_x@vx)@v0 == pyx[p,q], (ry_x@vx)@v1 == pyx[1:,0].sum(), I <= ub]
    max_prob = cp.Problem(cp.Maximize(pydox),constraints)
    max_prob.solve(solver=cp.SCS, verbose=False)
    ## get solverstats
    t += max_prob.solver_stats.solve_time
    min_prob = cp.Problem(cp.Minimize(pydox),constraints)
    min_prob.solve(solver=cp.SCS)
    t+= min_prob.solver_stats.solve_time
    return min_prob.value, max_prob.value, t