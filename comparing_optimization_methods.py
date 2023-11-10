import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import generate_dirichlet, generate_conditional_dist, entropy_dist, optimization_cp, optimization_cf





## Set seed for reproducibility
np.random.seed(0)

num_dist = 100  #n
z_states = 2    #k
x_states = 2    #j
y_states = 2    #i


## Gernerate distributions according to the graphical model
z = generate_dirichlet(alpha=0.1, num_states=z_states, num_dist=num_dist)
x_z = generate_conditional_dist(num_states=x_states, num_parents_states=z_states, num_dist=num_dist)


y_xz = np.zeros((num_dist, y_states, x_states, z_states))
for i in range(z_states):
    y_xz[:,:,:,i] = generate_conditional_dist(num_states=y_states, num_parents_states=x_states, num_dist=num_dist)


num_same = 0
num_better = 0

xz = np.einsum('njk,nk->njk', x_z, z)
x = np.sum(xz, axis=2)


z_x = np.einsum('njk,nj->nkj', xz, np.divide(1,x, where=(x!=0)))


yx_z = np.einsum('nijk, njk->nijk' ,y_xz, x_z)
yxz = np.einsum('nijk, nk->nijk' ,yx_z, z)
yx = np.sum(yxz, axis=-1)
y = np.sum(np.sum(yxz, axis=2),axis=2)
y_dox = np.einsum('nijk,nk->nij', y_xz, z)
y_x = np.einsum('nijk,nkj->nij', y_xz, z_x)

y_z = np.einsum('nijk,nj->nik', y_xz, x)

yz = np.einsum('nik, nk->nik', y_z, z)
z_y = np.einsum('nik, ni->nki', y_z, np.divide(1,y, where=(y!=0)))

Hz = entropy_dist(z)

tot_time_cp = 0
tot_time_cf = 0
diff = 0
# for n in range(num_dist):
for n in tqdm(range(num_dist)):
    print(f"Tian and Pearl's bound of p(y0|do(x0)) is\n [{yx[n, 0,0]},{1-yx[n,1:,0].sum()}]" )
    cf_min, cf_max, cf_t = optimization_cf(yx[n], ub=Hz, p=0, q=0)
    print(f"The CF bound is\n [{cf_min}, {cf_max}]")
    cp_min, cp_max, cp_t =  optimization_cp(yx[n], ub=Hz, p=0, q=0)
    print(f"The CP bound is\n [{cp_min}, {cp_max}]")

    print(f"CP time is {cp_t}")
    print(f"CF time is {cf_t}")
    tot_time_cp += cp_t
    tot_time_cf += cf_t
    diff += np.abs(cp_min - cf_min)
    diff += np.abs(cp_max - cf_max)
    print(f"The difference between optimal values is {diff}")
tot_time_cp = tot_time_cp/num_dist
tot_time_cf = tot_time_cf/num_dist
print(f"Average CP time is {tot_time_cp}")
print(f"Average CF time is {tot_time_cf}")













