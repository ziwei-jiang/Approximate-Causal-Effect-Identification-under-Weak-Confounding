import math
import numpy as np
import cvxpy as cp
from scipy.special import xlogy
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm





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
        


def optimization(pyx, ub=1, p =0, q=0):
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
    # I = dkl @ np.ones((nx*ny))
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
    

    max_prob = cp.Problem(max_obj, constraints)
    max_prob.solve(solver=cp.SCS)
    # print("The upper bound of p(y0|do(x0)) is\n", max_prob.value)
    min_prob = cp.Problem(min_obj, constraints)
    min_prob.solve(solver=cp.SCS)
    # print("The lower bound of p(y0|do(x0)) is\n", min_prob.value)
    # print(f"The optimization bound is\n [{min_prob.value}, {max_prob.value}]")
    return min_prob.value, max_prob.value


#################### Prepare the insurance dataset ######################
'''
variable Age {
  type discrete [ 3 ] { Adolescent, Adult, Senior };
}
probability ( Age ) {
  table 0.2, 0.6, 0.2;
}


variable SocioEcon {
  type discrete [ 4 ] { Prole, Middle, UpperMiddle, Wealthy };
}
probability ( SocioEcon | Age ) {
  (Adolescent) 0.40, 0.40, 0.19, 0.01;
  (Adult) 0.40, 0.40, 0.19, 0.01;
  (Senior) 0.50, 0.20, 0.29, 0.01;
}



variable RiskAversion {
  type discrete [ 4 ] { Psychopath, Adventurous, Normal, Cautious };
}
probability ( RiskAversion | Age, SocioEcon ) {
  (Adolescent, Prole) 0.02, 0.58, 0.30, 0.10;
  (Adult, Prole) 0.015, 0.285, 0.500, 0.200;
  (Senior, Prole) 0.01, 0.09, 0.40, 0.50;
  (Adolescent, Middle) 0.02, 0.38, 0.50, 0.10;
  (Adult, Middle) 0.015, 0.185, 0.600, 0.200;
  (Senior, Middle) 0.01, 0.04, 0.35, 0.60;
  (Adolescent, UpperMiddle) 0.02, 0.48, 0.40, 0.10;
  (Adult, UpperMiddle) 0.015, 0.285, 0.500, 0.200;
  (Senior, UpperMiddle) 0.01, 0.09, 0.40, 0.50;
  (Adolescent, Wealthy) 0.02, 0.58, 0.30, 0.10;
  (Adult, Wealthy) 0.015, 0.285, 0.400, 0.300;
  (Senior, Wealthy) 0.01, 0.09, 0.40, 0.50;
}



variable SeniorTrain {
  type discrete [ 2 ] { True, False };
}
probability ( SeniorTrain | Age, RiskAversion ) {
  (Adolescent, Psychopath) 0.0, 1.0;
  (Adult, Psychopath) 0.0, 1.0;
  (Senior, Psychopath) 0.000001, 0.999999;
  (Adolescent, Adventurous) 0.0, 1.0;
  (Adult, Adventurous) 0.0, 1.0;
  (Senior, Adventurous) 0.000001, 0.999999;
  (Adolescent, Normal) 0.0, 1.0;
  (Adult, Normal) 0.0, 1.0;
  (Senior, Normal) 0.3, 0.7;
  (Adolescent, Cautious) 0.0, 1.0;
  (Adult, Cautious) 0.0, 1.0;
  (Senior, Cautious) 0.9, 0.1;
}


variable DrivingSkill {
  type discrete [ 3 ] { SubStandard, Normal, Expert };
}
probability ( DrivingSkill | Age, SeniorTrain ) {
  (Adolescent, True) 0.50, 0.45, 0.05;
  (Adult, True) 0.3, 0.6, 0.1;
  (Senior, True) 0.1, 0.6, 0.3;
  (Adolescent, False) 0.50, 0.45, 0.05;
  (Adult, False) 0.3, 0.6, 0.1;
  (Senior, False) 0.4, 0.5, 0.1;
}


variable DrivQuality {
  type discrete [ 3 ] { Poor, Normal, Excellent };
}
probability ( DrivQuality | DrivingSkill, RiskAversion ) {
  (SubStandard, Psychopath) 1.0, 0.0, 0.0;
  (Normal, Psychopath) 0.5, 0.2, 0.3;
  (Expert, Psychopath) 0.3, 0.2, 0.5;
  (SubStandard, Adventurous) 1.0, 0.0, 0.0;
  (Normal, Adventurous) 0.3, 0.4, 0.3;
  (Expert, Adventurous) 0.01, 0.01, 0.98;
  (SubStandard, Normal) 1.0, 0.0, 0.0;
  (Normal, Normal) 0.0, 1.0, 0.0;
  (Expert, Normal) 0.0, 0.0, 1.0;
  (SubStandard, Cautious) 1.0, 0.0, 0.0;
  (Normal, Cautious) 0.0, 0.8, 0.2;
  (Expert, Cautious) 0.0, 0.0, 1.0;
}




variable Antilock {
  type discrete [ 2 ] { True, False };
}
probability ( Antilock | MakeModel, VehicleYear ) {
  (SportsCar, Current) 0.9, 0.1;
  (Economy, Current) 0.001, 0.999;
  (FamilySedan, Current) 0.4, 0.6;
  (Luxury, Current) 0.99, 0.01;
  (SuperLuxury, Current) 0.99, 0.01;
  (SportsCar, Older) 0.1, 0.9;
  (Economy, Older) 0.0, 1.0;
  (FamilySedan, Older) 0.0, 1.0;
  (Luxury, Older) 0.3, 0.7;
  (SuperLuxury, Older) 0.15, 0.85;
}


variable RuggedAuto {
  type discrete [ 3 ] { EggShell, Football, Tank };
}
probability ( RuggedAuto | MakeModel, VehicleYear ) {
  (SportsCar, Current) 0.95, 0.04, 0.01;
  (Economy, Current) 0.5, 0.5, 0.0;
  (FamilySedan, Current) 0.2, 0.6, 0.2;
  (Luxury, Current) 0.1, 0.6, 0.3;
  (SuperLuxury, Current) 0.05, 0.55, 0.40;
  (SportsCar, Older) 0.95, 0.04, 0.01;
  (Economy, Older) 0.9, 0.1, 0.0;
  (FamilySedan, Older) 0.05, 0.55, 0.40;
  (Luxury, Older) 0.1, 0.6, 0.3;
  (SuperLuxury, Older) 0.05, 0.55, 0.40;
}

variable Accident {
  type discrete [ 4 ] { None, Mild, Moderate, Severe };
}
probability ( Accident | Antilock, Mileage, DrivQuality ) {
  (True, FiveThou, Poor) 0.70, 0.20, 0.07, 0.03;
  (False, FiveThou, Poor) 0.6, 0.2, 0.1, 0.1;
  (True, TwentyThou, Poor) 0.4, 0.3, 0.2, 0.1;
  (False, TwentyThou, Poor) 0.3, 0.2, 0.2, 0.3;
  (True, FiftyThou, Poor) 0.3, 0.3, 0.2, 0.2;
  (False, FiftyThou, Poor) 0.2, 0.2, 0.2, 0.4;
  (True, Domino, Poor) 0.2, 0.2, 0.3, 0.3;
  (False, Domino, Poor) 0.1, 0.1, 0.3, 0.5;
  (True, FiveThou, Normal) 0.990, 0.007, 0.002, 0.001;
  (False, FiveThou, Normal) 0.980, 0.010, 0.005, 0.005;
  (True, TwentyThou, Normal) 0.980, 0.010, 0.005, 0.005;
  (False, TwentyThou, Normal) 0.960, 0.020, 0.015, 0.005;
  (True, FiftyThou, Normal) 0.970, 0.020, 0.007, 0.003;
  (False, FiftyThou, Normal) 0.950, 0.030, 0.015, 0.005;
  (True, Domino, Normal) 0.95, 0.03, 0.01, 0.01;
  (False, Domino, Normal) 0.94, 0.03, 0.02, 0.01;
  (True, FiveThou, Excellent) 0.9990, 0.0007, 0.0002, 0.0001;
  (False, FiveThou, Excellent) 0.995, 0.003, 0.001, 0.001;
  (True, TwentyThou, Excellent) 0.995, 0.003, 0.001, 0.001;
  (False, TwentyThou, Excellent) 0.990, 0.007, 0.002, 0.001;
  (True, FiftyThou, Excellent) 0.990, 0.007, 0.002, 0.001;
  (False, FiftyThou, Excellent) 0.980, 0.010, 0.005, 0.005;
  (True, Domino, Excellent) 0.985, 0.010, 0.003, 0.002;
  (False, Domino, Excellent) 0.980, 0.010, 0.007, 0.003;
}



probability ( AntiTheft | RiskAversion, SocioEcon ) {
  (Psychopath, Prole) 0.000001, 0.999999;
  (Adventurous, Prole) 0.000001, 0.999999;
  (Normal, Prole) 0.1, 0.9;
  (Cautious, Prole) 0.95, 0.05;
  (Psychopath, Middle) 0.000001, 0.999999;
  (Adventurous, Middle) 0.000001, 0.999999;
  (Normal, Middle) 0.3, 0.7;
  (Cautious, Middle) 0.999999, 0.000001;
  (Psychopath, UpperMiddle) 0.05, 0.95;
  (Adventurous, UpperMiddle) 0.2, 0.8;
  (Normal, UpperMiddle) 0.9, 0.1;
  (Cautious, UpperMiddle) 0.999999, 0.000001;
  (Psychopath, Wealthy) 0.5, 0.5;
  (Adventurous, Wealthy) 0.5, 0.5;
  (Normal, Wealthy) 0.8, 0.2;
  (Cautious, Wealthy) 0.999999, 0.000001;
}

probability ( HomeBase | RiskAversion, SocioEcon ) {
  (Psychopath, Prole) 0.000001, 0.800000, 0.049999, 0.150000;
  (Adventurous, Prole) 0.000001, 0.800000, 0.050000, 0.149999;
  (Normal, Prole) 0.000001, 0.800000, 0.050000, 0.149999;
  (Cautious, Prole) 0.000001, 0.800000, 0.050000, 0.149999;
  (Psychopath, Middle) 0.15, 0.80, 0.04, 0.01;
  (Adventurous, Middle) 0.01, 0.25, 0.60, 0.14;
  (Normal, Middle) 0.299999, 0.000001, 0.600000, 0.100000;
  (Cautious, Middle) 0.950000, 0.000001, 0.024445, 0.025554;
  (Psychopath, UpperMiddle) 0.35, 0.60, 0.04, 0.01;
  (Adventurous, UpperMiddle) 0.2, 0.4, 0.3, 0.1;
  (Normal, UpperMiddle) 0.500000, 0.000001, 0.400000, 0.099999;
  (Cautious, UpperMiddle) 0.999997, 0.000001, 0.000001, 0.000001;
  (Psychopath, Wealthy) 0.489999, 0.500000, 0.000001, 0.010000;
  (Adventurous, Wealthy) 0.950000, 0.000001, 0.000001, 0.049998;
  (Normal, Wealthy) 0.850000, 0.000001, 0.001000, 0.148999;
  (Cautious, Wealthy) 0.999997, 0.000001, 0.000001, 0.000001;
}




variable MakeModel {
  type discrete [ 5 ] { SportsCar, Economy, FamilySedan, Luxury, SuperLuxury };
}
probability ( MakeModel | SocioEcon, RiskAversion ) {
  (Prole, Psychopath) 0.1, 0.7, 0.2, 0.0, 0.0;
  (Middle, Psychopath) 0.15, 0.20, 0.65, 0.00, 0.00;
  (UpperMiddle, Psychopath) 0.20, 0.05, 0.30, 0.45, 0.00;
  (Wealthy, Psychopath) 0.30, 0.01, 0.09, 0.40, 0.20;
  (Prole, Adventurous) 0.1, 0.7, 0.2, 0.0, 0.0;
  (Middle, Adventurous) 0.15, 0.20, 0.65, 0.00, 0.00;
  (UpperMiddle, Adventurous) 0.20, 0.05, 0.30, 0.45, 0.00;
  (Wealthy, Adventurous) 0.30, 0.01, 0.09, 0.40, 0.20;
  (Prole, Normal) 0.1, 0.7, 0.2, 0.0, 0.0;
  (Middle, Normal) 0.15, 0.20, 0.65, 0.00, 0.00;
  (UpperMiddle, Normal) 0.20, 0.05, 0.30, 0.45, 0.00;
  (Wealthy, Normal) 0.30, 0.01, 0.09, 0.40, 0.20;
  (Prole, Cautious) 0.1, 0.7, 0.2, 0.0, 0.0;
  (Middle, Cautious) 0.15, 0.20, 0.65, 0.00, 0.00;
  (UpperMiddle, Cautious) 0.20, 0.05, 0.30, 0.45, 0.00;
  (Wealthy, Cautious) 0.30, 0.01, 0.09, 0.40, 0.20;
}

variable VehicleYear {
  type discrete [ 2 ] { Current, Older };
}
probability ( VehicleYear | SocioEcon, RiskAversion ) {
  (Prole, Psychopath) 0.15, 0.85;
  (Middle, Psychopath) 0.3, 0.7;
  (UpperMiddle, Psychopath) 0.8, 0.2;
  (Wealthy, Psychopath) 0.9, 0.1;
  (Prole, Adventurous) 0.15, 0.85;
  (Middle, Adventurous) 0.3, 0.7;
  (UpperMiddle, Adventurous) 0.8, 0.2;
  (Wealthy, Adventurous) 0.9, 0.1;
  (Prole, Normal) 0.15, 0.85;
  (Middle, Normal) 0.3, 0.7;
  (UpperMiddle, Normal) 0.8, 0.2;
  (Wealthy, Normal) 0.9, 0.1;
  (Prole, Cautious) 0.15, 0.85;
  (Middle, Cautious) 0.3, 0.7;
  (UpperMiddle, Cautious) 0.8, 0.2;
  (Wealthy, Cautious) 0.9, 0.1;
}

variable Mileage {
  type discrete [ 4 ] { FiveThou, TwentyThou, FiftyThou, Domino };
}
probability ( Mileage ) {
  table 0.1, 0.4, 0.4, 0.1;
}

probability ( CarValue | MakeModel, VehicleYear, Mileage ) {
  (SportsCar, Current, FiveThou) 0.00, 0.10, 0.80, 0.09, 0.01;
  (Economy, Current, FiveThou) 0.1, 0.8, 0.1, 0.0, 0.0;
  (FamilySedan, Current, FiveThou) 0.0, 0.1, 0.9, 0.0, 0.0;
  (Luxury, Current, FiveThou) 0.0, 0.0, 0.0, 1.0, 0.0;
  (SuperLuxury, Current, FiveThou) 0.0, 0.0, 0.0, 0.0, 1.0;
  (SportsCar, Older, FiveThou) 0.03, 0.30, 0.60, 0.06, 0.01;
  (Economy, Older, FiveThou) 0.25, 0.70, 0.05, 0.00, 0.00;
  (FamilySedan, Older, FiveThou) 0.2, 0.3, 0.5, 0.0, 0.0;
  (Luxury, Older, FiveThou) 0.01, 0.09, 0.20, 0.70, 0.00;
  (SuperLuxury, Older, FiveThou) 0.000001, 0.000001, 0.000001, 0.000001, 0.999996;
  (SportsCar, Current, TwentyThou) 0.00, 0.10, 0.80, 0.09, 0.01;
  (Economy, Current, TwentyThou) 0.1, 0.8, 0.1, 0.0, 0.0;
  (FamilySedan, Current, TwentyThou) 0.0, 0.1, 0.9, 0.0, 0.0;
  (Luxury, Current, TwentyThou) 0.0, 0.0, 0.0, 1.0, 0.0;
  (SuperLuxury, Current, TwentyThou) 0.0, 0.0, 0.0, 0.0, 1.0;
  (SportsCar, Older, TwentyThou) 0.16, 0.50, 0.30, 0.03, 0.01;
  (Economy, Older, TwentyThou) 0.7000, 0.2999, 0.0001, 0.0000, 0.0000;
  (FamilySedan, Older, TwentyThou) 0.5, 0.3, 0.2, 0.0, 0.0;
  (Luxury, Older, TwentyThou) 0.05, 0.15, 0.30, 0.50, 0.00;
  (SuperLuxury, Older, TwentyThou) 0.000001, 0.000001, 0.000001, 0.000001, 0.999996;
  (SportsCar, Current, FiftyThou) 0.00, 0.10, 0.80, 0.09, 0.01;
  (Economy, Current, FiftyThou) 0.1, 0.8, 0.1, 0.0, 0.0;
  (FamilySedan, Current, FiftyThou) 0.0, 0.1, 0.9, 0.0, 0.0;
  (Luxury, Current, FiftyThou) 0.0, 0.0, 0.0, 1.0, 0.0;
  (SuperLuxury, Current, FiftyThou) 0.0, 0.0, 0.0, 0.0, 1.0;
  (SportsCar, Older, FiftyThou) 0.40, 0.47, 0.10, 0.02, 0.01;
  (Economy, Older, FiftyThou) 0.990000, 0.009999, 0.000001, 0.000000, 0.000000;
  (FamilySedan, Older, FiftyThou) 0.7, 0.2, 0.1, 0.0, 0.0;
  (Luxury, Older, FiftyThou) 0.1, 0.3, 0.3, 0.3, 0.0;
  (SuperLuxury, Older, FiftyThou) 0.000001, 0.000001, 0.000001, 0.000001, 0.999996;
  (SportsCar, Current, Domino) 0.00, 0.10, 0.80, 0.09, 0.01;
  (Economy, Current, Domino) 0.1, 0.8, 0.1, 0.0, 0.0;
  (FamilySedan, Current, Domino) 0.0, 0.1, 0.9, 0.0, 0.0;
  (Luxury, Current, Domino) 0.0, 0.0, 0.0, 1.0, 0.0;
  (SuperLuxury, Current, Domino) 0.0, 0.0, 0.0, 0.0, 1.0;
  (SportsCar, Older, Domino) 0.90, 0.06, 0.02, 0.01, 0.01;
  (Economy, Older, Domino) 0.999998, 0.000001, 0.000001, 0.000000, 0.000000;
  (FamilySedan, Older, Domino) 0.990000, 0.009999, 0.000001, 0.000000, 0.000000;
  (Luxury, Older, Domino) 0.2, 0.2, 0.3, 0.3, 0.0;
  (SuperLuxury, Older, Domino) 0.000001, 0.000001, 0.000001, 0.000001, 0.999996;
}



probability ( Theft | AntiTheft, HomeBase, CarValue ) {
  (True, Secure, FiveThou) 0.000001, 0.999999;
  (False, Secure, FiveThou) 0.000001, 0.999999;
  (True, City, FiveThou) 0.0005, 0.9995;
  (False, City, FiveThou) 0.001, 0.999;
  (True, Suburb, FiveThou) 0.00001, 0.99999;
  (False, Suburb, FiveThou) 0.00001, 0.99999;
  (True, Rural, FiveThou) 0.00001, 0.99999;
  (False, Rural, FiveThou) 0.00001, 0.99999;
  (True, Secure, TenThou) 0.000002, 0.999998;
  (False, Secure, TenThou) 0.000002, 0.999998;
  (True, City, TenThou) 0.002, 0.998;
  (False, City, TenThou) 0.005, 0.995;
  (True, Suburb, TenThou) 0.0001, 0.9999;
  (False, Suburb, TenThou) 0.0002, 0.9998;
  (True, Rural, TenThou) 0.00002, 0.99998;
  (False, Rural, TenThou) 0.0001, 0.9999;
  (True, Secure, TwentyThou) 0.000003, 0.999997;
  (False, Secure, TwentyThou) 0.000003, 0.999997;
  (True, City, TwentyThou) 0.005, 0.995;
  (False, City, TwentyThou) 0.01, 0.99;
  (True, Suburb, TwentyThou) 0.0003, 0.9997;
  (False, Suburb, TwentyThou) 0.0005, 0.9995;
  (True, Rural, TwentyThou) 0.00005, 0.99995;
  (False, Rural, TwentyThou) 0.0002, 0.9998;
  (True, Secure, FiftyThou) 0.000002, 0.999998;
  (False, Secure, FiftyThou) 0.000002, 0.999998;
  (True, City, FiftyThou) 0.005, 0.995;
  (False, City, FiftyThou) 0.01, 0.99;
  (True, Suburb, FiftyThou) 0.0003, 0.9997;
  (False, Suburb, FiftyThou) 0.0005, 0.9995;
  (True, Rural, FiftyThou) 0.00005, 0.99995;
  (False, Rural, FiftyThou) 0.0002, 0.9998;
  (True, Secure, Million) 0.000001, 0.999999;
  (False, Secure, Million) 0.000001, 0.999999;
  (True, City, Million) 0.000001, 0.999999;
  (False, City, Million) 0.000001, 0.999999;
  (True, Suburb, Million) 0.000001, 0.999999;
  (False, Suburb, Million) 0.000001, 0.999999;
  (True, Rural, Million) 0.000001, 0.999999;
  (False, Rural, Million) 0.000001, 0.999999;
}

probability ( ThisCarCost | ThisCarDam, CarValue, Theft ) {
  (None, FiveThou, True) 0.2, 0.8, 0.0, 0.0;
  (Mild, FiveThou, True) 0.15, 0.85, 0.00, 0.00;
  (Moderate, FiveThou, True) 0.05, 0.95, 0.00, 0.00;
  (Severe, FiveThou, True) 0.03, 0.97, 0.00, 0.00;
  (None, TenThou, True) 0.05, 0.95, 0.00, 0.00;
  (Mild, TenThou, True) 0.03, 0.97, 0.00, 0.00;
  (Moderate, TenThou, True) 0.01, 0.99, 0.00, 0.00;
  (Severe, TenThou, True) 0.000001, 0.999999, 0.000000, 0.000000;
  (None, TwentyThou, True) 0.04, 0.01, 0.95, 0.00;
  (Mild, TwentyThou, True) 0.03, 0.02, 0.95, 0.00;
  (Moderate, TwentyThou, True) 0.001, 0.001, 0.998, 0.000;
  (Severe, TwentyThou, True) 0.000001, 0.000001, 0.999998, 0.000000;
  (None, FiftyThou, True) 0.04, 0.01, 0.95, 0.00;
  (Mild, FiftyThou, True) 0.03, 0.02, 0.95, 0.00;
  (Moderate, FiftyThou, True) 0.001, 0.001, 0.998, 0.000;
  (Severe, FiftyThou, True) 0.000001, 0.000001, 0.999998, 0.000000;
  (None, Million, True) 0.04, 0.01, 0.20, 0.75;
  (Mild, Million, True) 0.02, 0.03, 0.25, 0.70;
  (Moderate, Million, True) 0.001, 0.001, 0.018, 0.980;
  (Severe, Million, True) 0.000001, 0.000001, 0.009998, 0.990000;
  (None, FiveThou, False) 1.0, 0.0, 0.0, 0.0;
  (Mild, FiveThou, False) 0.95, 0.05, 0.00, 0.00;
  (Moderate, FiveThou, False) 0.25, 0.75, 0.00, 0.00;
  (Severe, FiveThou, False) 0.05, 0.95, 0.00, 0.00;
  (None, TenThou, False) 1.0, 0.0, 0.0, 0.0;
  (Mild, TenThou, False) 0.95, 0.05, 0.00, 0.00;
  (Moderate, TenThou, False) 0.15, 0.85, 0.00, 0.00;
  (Severe, TenThou, False) 0.01, 0.99, 0.00, 0.00;
  (None, TwentyThou, False) 1.0, 0.0, 0.0, 0.0;
  (Mild, TwentyThou, False) 0.99, 0.01, 0.00, 0.00;
  (Moderate, TwentyThou, False) 0.01, 0.01, 0.98, 0.00;
  (Severe, TwentyThou, False) 0.005, 0.005, 0.990, 0.000;
  (None, FiftyThou, False) 1.0, 0.0, 0.0, 0.0;
  (Mild, FiftyThou, False) 0.99, 0.01, 0.00, 0.00;
  (Moderate, FiftyThou, False) 0.005, 0.005, 0.990, 0.000;
  (Severe, FiftyThou, False) 0.001, 0.001, 0.998, 0.000;
  (None, Million, False) 1.0, 0.0, 0.0, 0.0;
  (Mild, Million, False) 0.98, 0.01, 0.01, 0.00;
  (Moderate, Million, False) 0.003, 0.003, 0.044, 0.950;
  (Severe, Million, False) 0.000001, 0.000001, 0.029998, 0.970000;
}

'''

pAge = np.array([0.2, 0.6, 0.2])
pSocioecon_Age = np.array([[0.4, 0.4, 0.19, 0.01], [0.4, 0.4, 0.19, 0.01], [0.50, 0.20, 0.29, 0.01]]).T
pSocioeconAge = np.einsum('ij, j -> ij', pSocioecon_Age, pAge)

pSocioecon = np.einsum('ij, j -> i', pSocioecon_Age, pAge)

pRiskaversion_AgeSocioecon = np.array([[[0.02, 0.58, 0.30, 0.10],[0.015, 0.285, 0.500, 0.200],[0.01, 0.09, 0.40, 0.50]],
                                        [[0.02, 0.38, 0.50, 0.10],[0.015, 0.185, 0.600, 0.200],[0.01, 0.04, 0.35, 0.60]],
                                        [[0.02, 0.48, 0.40, 0.10],[0.015, 0.285, 0.500, 0.200],[0.01, 0.09, 0.40, 0.50]],
                                        [[0.02, 0.58, 0.30, 0.10],[0.015, 0.285, 0.400, 0.300],[0.01, 0.09, 0.40, 0.50]]]).transpose(2,1,0)
pAgeSocioecon = pSocioeconAge.T
pRiskaversionAgeSocioecon = np.einsum('ijk, jk -> ijk', pRiskaversion_AgeSocioecon, pAgeSocioecon)

pRiskaversionAge = pRiskaversionAgeSocioecon.sum(axis=2)

pSeniortrain_AgeRiskaversion = np.array([[[0.0, 1.0],[0.0, 1.0],[0.000001, 0.999999]],
                         [[0.0, 1.0],[0.0, 1.0],[0.000001, 0.999999]],
                         [[0.0, 1.0],[0.0, 1.0],[0.3, 0.7]],
                         [[0.0, 1.0],[0.0, 1.0],[0.9, 0.1]]]).transpose(2,1,0)
pAgeRiskaversion = pRiskaversionAge.T
pSeniortrainAgeRiskaversion = np.einsum('ijk, jk -> ijk', pSeniortrain_AgeRiskaversion, pAgeRiskaversion)
pRiskaversionAgeSeniortrain = pSeniortrainAgeRiskaversion.transpose(2,1,0)
pAgeSeniortrain = pRiskaversionAgeSeniortrain.sum(axis=0)
pRiskaversion_AgeSeniortrain = np.einsum('ijk, jk -> ijk', pRiskaversionAgeSeniortrain, np.divide(1, pAgeSeniortrain, where=pAgeSeniortrain!=0))


pDrivskill_AgeSeniortrain = np.array([[[.50, 0.45, 0.05],[ 0.3, 0.6, 0.1],[0.1, 0.6, 0.3]],
                                         [[0.50, 0.45, 0.05],[0.3, 0.6, 0.1],[0.4, 0.5, 0.1]]]).transpose(2,1,0)

pDirivingskillRiskaversion_AgeSeniortrain = np.einsum('ajk, bjk->abjk', pDrivskill_AgeSeniortrain, pRiskaversion_AgeSeniortrain)
pDirivingskillRiskaversion = np.einsum('abjk, jk->ab', pDirivingskillRiskaversion_AgeSeniortrain, pAgeSeniortrain)

pDrivquality_DrivSkillRiskaversion = np.array([
    [[1.0, 0.0, 0.0],[0.5, 0.2, 0.3],[0.3, 0.2, 0.5]],
    [[1.0, 0.0, 0.0],[0.3, 0.4, 0.3],[0.01, 0.01, 0.98]],
    [[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0]],
    [[1.0, 0.0, 0.0],[0.0, 0.8, 0.2],[0.0, 0.0, 1.0]]
]).transpose(2,1,0)

pDrivquality = np.einsum('ijk, jk-> i', pDrivquality_DrivSkillRiskaversion, pDirivingskillRiskaversion)


#### Get three condition variables we want to use
pMakemodel_SocioeconRiskaversion = np.array([
    [[0.1, 0.7, 0.2, 0.0, 0.0],[0.15, 0.20, 0.65, 0.00, 0.00],[0.20, 0.05, 0.30, 0.45, 0.00],[0.30, 0.01, 0.09, 0.40, 0.20]],
    [[0.1, 0.7, 0.2, 0.0, 0.0],[0.15, 0.20, 0.65, 0.00, 0.00],[0.20, 0.05, 0.30, 0.45, 0.00],[0.30, 0.01, 0.09, 0.40, 0.20]],
    [[0.1, 0.7, 0.2, 0.0, 0.0],[0.15, 0.20, 0.65, 0.00, 0.00],[0.20, 0.05, 0.30, 0.45, 0.00],[0.30, 0.01, 0.09, 0.40, 0.20]],
    [[0.1, 0.7, 0.2, 0.0, 0.0],[0.15, 0.20, 0.65, 0.00, 0.00],[0.20, 0.05, 0.30, 0.45, 0.00],[0.30, 0.01, 0.09, 0.40, 0.20]]
]).transpose(2,1,0)
pSocioeconRiskaversion = np.einsum('ijk-> ki', pRiskaversionAgeSocioecon)
pMakemodel = np.einsum('ijk, jk-> i', pMakemodel_SocioeconRiskaversion, pSocioeconRiskaversion)

pVehicleyear_socioeconriskaversion = np.array([
    [[0.15, 0.85],[0.3, 0.7],[0.8, 0.2],[0.9, 0.1]],
    [[0.15, 0.85],[0.3, 0.7],[0.8, 0.2],[0.9, 0.1]],
    [[0.15, 0.85],[0.3, 0.7],[0.8, 0.2],[0.9, 0.1]],
    [[0.15, 0.85],[0.3, 0.7],[0.8, 0.2],[0.9, 0.1]]
]).transpose(2,1,0)
pVehicleyear = np.einsum('ijk, jk-> i', pVehicleyear_socioeconriskaversion, pSocioeconRiskaversion)

pMileage = np.array([0.1, 0.4, 0.4, 0.1])

### -------------------
pAntitheft_RiskaversionSocioeco = np.array([
    [[0.000001, 0.999999],[0.000001, 0.999999],[0.1,0.9],[0.95,0.05]],
    [[0.000001, 0.999999],[0.000001, 0.999999],[0.3, 0.7],[0.999999, 0.000001]],
    [[0.05, 0.95],[0.2, 0.8],[0.9, 0.1],[0.999999, 0.000001]],
    [[0.5,0.5],[0.5,0.5],[0.8,0.2],[0.999999, 0.000001]]
]).transpose(2,1,0)
pAntitheft = np.einsum('ijk, jk-> i', pAntitheft_RiskaversionSocioeco, pSocioeconRiskaversion.T)

pHomebase_RiskaversionSocioeco = np.array([
    [[0.000001, 0.800000, 0.049999, 0.150000],[0.000001, 0.800000, 0.050000, 0.149999],[0.000001, 0.800000, 0.050000, 0.149999],[0.000001, 0.800000, 0.050000, 0.149999]],
    [[0.15, 0.80, 0.04, 0.01],[0.01, 0.25, 0.60, 0.14],[0.299999, 0.000001, 0.600000, 0.100000],[0.950000, 0.000001, 0.024445, 0.025554]],
    [[0.35, 0.60, 0.04, 0.01],[ 0.2, 0.4, 0.3, 0.1],[0.500000, 0.000001, 0.400000, 0.099999],[0.999997, 0.000001, 0.000001, 0.000001]],
    [[0.489999, 0.500000, 0.000001, 0.010000],[ 0.950000, 0.000001, 0.000001, 0.049998],[0.850000, 0.000001, 0.001000, 0.148999],[0.999997, 0.000001, 0.000001, 0.000001]]
]).transpose(2,1,0)
pHomebase = np.einsum('ijk, jk-> i', pHomebase_RiskaversionSocioeco, pSocioeconRiskaversion.T)



pCarvalue_MakemodelVehicleyearMileage = np.array([
    [[[0.00, 0.10, 0.80, 0.09, 0.01],[0.1, 0.8, 0.1, 0.0, 0.0],[0.0, 0.1, 0.9, 0.0, 0.0],[0.0, 0.0, 0.0, 1.0, 0.0],[0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.03, 0.30, 0.60, 0.06, 0.01],[0.25, 0.70, 0.05, 0.00, 0.00],[0.2, 0.3, 0.5, 0.0, 0.0],[0.01, 0.09, 0.20, 0.70, 0.00],[0.000001, 0.000001, 0.000001, 0.000001, 0.999996]]],
    
    [[[0.00, 0.10, 0.80, 0.09, 0.01],[0.1, 0.8, 0.1, 0.0, 0.0],[0.0, 0.1, 0.9, 0.0, 0.0],[0.0, 0.0, 0.0, 1.0, 0.0],[0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.16, 0.50, 0.30, 0.03, 0.01],[0.7000, 0.2999, 0.0001, 0.0000, 0.0000],[0.5, 0.3, 0.2, 0.0, 0.0],[0.05, 0.15, 0.30, 0.50, 0.00],[0.000001, 0.000001, 0.000001, 0.000001, 0.999996]]],
    
    [[[0.00, 0.10, 0.80, 0.09, 0.01],[0.1, 0.8, 0.1, 0.0, 0.0],[0.0, 0.1, 0.9, 0.0, 0.0],[ 0.0, 0.0, 0.0, 1.0, 0.0],[0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.40, 0.47, 0.10, 0.02, 0.01],[0.990000, 0.009999, 0.000001, 0.000000, 0.000000],[0.7, 0.2, 0.1, 0.0, 0.0],[0.1, 0.3, 0.3, 0.3, 0.0],[0.000001, 0.000001, 0.000001, 0.000001, 0.999996]]],
    
    [[[0.00, 0.10, 0.80, 0.09, 0.01],[0.1, 0.8, 0.1, 0.0, 0.0],[0.0, 0.1, 0.9, 0.0, 0.0],[0.0, 0.0, 0.0, 1.0, 0.0],[0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.90, 0.06, 0.02, 0.01, 0.01],[0.999998, 0.000001, 0.000001, 0.000000, 0.000000],[0.990000, 0.009999, 0.000001, 0.000000, 0.000000],[0.2, 0.2, 0.3, 0.3, 0.0],[0.000001, 0.000001, 0.000001, 0.000001, 0.999996]]]
]).transpose(3,2,1,0)

pAccident_AntilockMileageDrivquality = np.array([
  [[[0.70, 0.20, 0.07, 0.03],[0.6, 0.2, 0.1, 0.1]],
  [[0.4, 0.3, 0.2, 0.1],[0.3, 0.2, 0.2, 0.3]],
  [[0.3, 0.3, 0.2, 0.2],[0.2, 0.2, 0.2, 0.4]],
  [[0.2, 0.2, 0.3, 0.3],[0.1, 0.1, 0.3, 0.5]]],
  
  [[[0.990, 0.007, 0.002, 0.001],[0.980, 0.010, 0.005, 0.005]],
  [[0.980, 0.010, 0.005, 0.005],[0.960, 0.020, 0.015, 0.005]],
  [[0.970, 0.020, 0.007, 0.003],[0.950, 0.030, 0.015, 0.005]],
  [[0.95, 0.03, 0.01, 0.01],[0.94, 0.03, 0.02, 0.01]]],
  
  [[[0.9990, 0.0007, 0.0002, 0.0001],[0.995, 0.003, 0.001, 0.001]],
  [[0.995, 0.003, 0.001, 0.001],[0.990, 0.007, 0.002, 0.001]],
  [[0.990, 0.007, 0.002, 0.001],[0.980, 0.010, 0.005, 0.005]],
  [[0.985, 0.010, 0.003, 0.002],[0.980, 0.010, 0.007, 0.003]]]
  ]).transpose(3,2,1,0)

pThiscardam_AccidentRuggedauto = np.array([
  [[1.0, 0.0, 0.0, 0.0],[0.001, 0.900, 0.098, 0.001],[0.000001, 0.000999, 0.700000, 0.299000],[0.000001, 0.000009, 0.000090, 0.999900]],
  [[1.0, 0.0, 0.0, 0.0],[0.200000, 0.750000, 0.049999, 0.000001],[0.001, 0.099, 0.800, 0.100],[0.000001, 0.000999, 0.009000, 0.990000]],
  [[1.0, 0.0, 0.0, 0.0],[0.700000, 0.290000, 0.009999, 0.000001],[0.05, 0.60, 0.30, 0.05],[0.05, 0.20, 0.20, 0.55]]
]).transpose(2,1,0)

pThiscarcost_ThiscardamCarvalueTheft = np.array([
    [[[0.2, 0.8, 0.0, 0.0],[0.15, 0.85, 0.00, 0.00],[ 0.05, 0.95, 0.00, 0.00],[0.03, 0.97, 0.00, 0.00]],
    [[0.05, 0.95, 0.00, 0.00],[0.03, 0.97, 0.00, 0.00],[0.01, 0.99, 0.00, 0.00],[0.000001, 0.999999, 0.000000, 0.000000]],
    [[0.04, 0.01, 0.95, 0.00],[0.03, 0.02, 0.95, 0.00],[0.001, 0.001, 0.998, 0.000],[0.000001, 0.000001, 0.999998, 0.000000]],
    [[0.04, 0.01, 0.95, 0.00],[0.03, 0.02, 0.95, 0.00],[0.001, 0.001, 0.998, 0.000],[0.000001, 0.000001, 0.999998, 0.000000]],
    [[0.04, 0.01, 0.20, 0.75],[0.02, 0.03, 0.25, 0.70],[0.001, 0.001, 0.018, 0.980],[0.000001, 0.000001, 0.009998, 0.990000]]], 

    [[[1.0, 0.0, 0.0, 0.0],[0.95, 0.05, 0.00, 0.00],[0.25, 0.75, 0.00, 0.00],[0.05, 0.95, 0.00, 0.00]],
    [[1.0, 0.0, 0.0, 0.0],[0.95, 0.05, 0.00, 0.00],[0.15, 0.85, 0.00, 0.00],[0.01, 0.99, 0.00, 0.00]],
    [[1.0, 0.0, 0.0, 0.0],[0.99, 0.01, 0.00, 0.00],[0.01, 0.01, 0.98, 0.00],[0.005, 0.005, 0.990, 0.000]],
    [[1.0, 0.0, 0.0, 0.0],[0.99, 0.01, 0.00, 0.00],[0.005, 0.005, 0.990, 0.000],[0.001, 0.001, 0.998, 0.000]],
    [[1.0, 0.0, 0.0, 0.0],[0.98, 0.01, 0.01, 0.00],[0.003, 0.003, 0.044, 0.950],[0.000001, 0.000001, 0.029998, 0.970000]]]
]).transpose(3,2,1,0)

pOthercarcost_AccidentRuggedauto = np.array([
      [[1.0, 0.0, 0.0, 0.0],[0.99000, 0.00500, 0.00499, 0.00001],[0.60000, 0.20000, 0.19998, 0.00002],[0.20000, 0.40000, 0.39996, 0.00004]],
      [[1.0, 0.0, 0.0, 0.0],[9.799657e-01, 9.999650e-03, 9.984651e-03, 4.999825e-05],[0.50000, 0.20000, 0.29997, 0.00003],[0.10000, 0.50000, 0.39994, 0.00006]],
      [[1.0, 0.0, 0.0, 0.0],[ 0.95000, 0.03000, 0.01998, 0.00002],[0.40000, 0.30000, 0.29996, 0.00004],[0.0050, 0.5500, 0.4449, 0.0001]]
]).transpose(2,1,0)


pPropcost_OthercarcostThiscarcost = np.array([
    [[0.7, 0.3, 0.0, 0.0],[0.00, 0.95, 0.05, 0.00],[0.00, 0.00, 0.98, 0.02],[0.0, 0.0, 0.0, 1.0]],
    [[0.00, 0.95, 0.05, 0.00],[0.0, 0.6, 0.4, 0.0],[0.0, 0.0, 0.8, 0.2],[0.0, 0.0, 0.0, 1.0]],
    [[0.00, 0.00, 0.98, 0.02],[0.00, 0.00, 0.95, 0.05],[0.0, 0.0, 0.6, 0.4],[0.0, 0.0, 0.0, 1.0]],
    [[0.0, 0.0, 0.0, 1.0],[0.0, 0.0, 0.0, 1.0],[0.0, 0.0, 0.0, 1.0],[0.0, 0.0, 0.0, 1.0]]
]).transpose(2,1,0)


########################## Experiment  #############################
### Condition on Antilock=True, FiveThou, NormalDrive
pAccident = pAccident_AntilockMileageDrivquality[:,0,0,1]

### Condition on RuggedAuto = eggshell
pThiscardam_Accident = pThiscardam_AccidentRuggedauto[:,:,0]
pOthercarcost_Accident = pOthercarcost_AccidentRuggedauto[:,:,0]

pThiscardam = np.einsum('ij, j->i', pThiscardam_Accident, pAccident)
pOthercarcost = np.einsum('ij, j->i', pOthercarcost_Accident, pAccident)

## Condition on carvalue = Million, theft=true
pThiscarcost_Thiscardam = pThiscarcost_ThiscardamCarvalueTheft[:,:, 4, 0]
pThiscarcost = np.einsum('ij, j->i', pThiscarcost_Thiscardam, pThiscardam)

pPropcostThiscarcost = np.einsum('ijk, j, k->ik', pPropcost_OthercarcostThiscarcost, pOthercarcost, pThiscarcost)
print(pPropcostThiscarcost)
print(pThiscarcost)

eps = 0.2
for i in range(pPropcostThiscarcost.shape[0]):
    for j in range(pPropcostThiscarcost.shape[1]):

                entropy_accident = entropy(pAccident)
                lb, ub = optimization(pPropcostThiscarcost, entropy_accident, i, j)
                neg = np.arange(pPropcostThiscarcost.shape[0]) != i
                tp_lb, tp_ub = pPropcostThiscarcost[i,j], 1-pPropcostThiscarcost[neg,j].sum()

                if ((lb - tp_lb) > eps) or ((ub - tp_ub) < -eps):
                # if True:
                    print(entropy_accident)
                    print(f"p(PropertyCost={i}, ThiscarCost={j}")
                    print(f"Entropy Constraint bounds \n [{lb, ub}]")
                    print(f"Tian-Pearl bounds\n [{tp_lb, tp_ub}]")
                    print("")