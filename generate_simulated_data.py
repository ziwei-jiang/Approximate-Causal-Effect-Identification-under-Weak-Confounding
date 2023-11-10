
import numpy as np
from utils import generate_conditional_dist, generate_dirichlet, mutual_information_dist, entropy_dist, optimization_cf




## Set seed for reproducibility
np.random.seed(0)

num_dist = 1000  #n
z_states = 5    #k
x_states = 10    #j
y_states = 2    #i
alpha_val = 1
## Gernerate distributions according to the graphical model
z = generate_dirichlet(alpha=alpha_val/10, num_states=z_states, num_dist=num_dist)
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
Ixy = mutual_information_dist(yx)


ubs = []
lbs = []
tp_lbs = []
tp_ubs = []
causal_effect = []
effective_entropy = []
tighter_bound = []



## iterate through all distributions
for n in range(num_dist):
    print(n)
    entropy_z = Hz[n]
    for p in range(y_states):
        for q in range(x_states):
            
            lb, ub = optimization_cf(yx[n], entropy_z, p, q)
            tp_lb = yx[n, p,q]
            tp_ub = yx[n,p,q] + 1 - x[n,q]
            diff = 1e-3
            if (lb -tp_lb> diff) or (ub-tp_ub< -diff):
                effective_entropy.append(entropy_z)
                ubs.append(ub)
                lbs.append(lb)
                causal_effect.append(y_dox[n, p,q])
                tp_lbs.append(tp_lb)
                tp_ubs.append(tp_ub)
                
                print(f"P(y={p}|do(x={q})")
                print(f'Entropy of Z: {entropy_z}')
                print(f'Optimization bound is\n {lb} and {ub}')
                print(f"Tian and Pearl's bound is\n [{tp_lb},{tp_ub}]" )
                print(f"The actual actual effect value is\n {y_dox[n, p,q]}" )

                num_better +=1
                tighter_bound.append(True)
                print("")
            else:
                effective_entropy.append(entropy_z)
                ubs.append(ub)
                lbs.append(lb)
                causal_effect.append(y_dox[n, p,q])
                tp_lbs.append(tp_lb)
                tp_ubs.append(tp_ub)
                tighter_bound.append(False)

                num_same +=1


print(f'Number of distributions that are better than Tian and Pearl is {num_better}')
print(f'Number of distributions that are the same as Tian and Pearl is {num_same}')

# save the results as numpy array
np.save(f'n{num_dist}z{z_states}x{x_states}y{y_states}a0{alpha_val}_ubs.npy', np.array(ubs))
np.save(f'n{num_dist}z{z_states}x{x_states}y{y_states}a0{alpha_val}_lbs.npy', np.array(lbs))
np.save(f'n{num_dist}z{z_states}x{x_states}y{y_states}a0{alpha_val}_tp_lbs.npy', np.array(tp_lbs))
np.save(f'n{num_dist}z{z_states}x{x_states}y{y_states}a0{alpha_val}_tp_ubs.npy', np.array(tp_ubs))
np.save(f'n{num_dist}z{z_states}x{x_states}y{y_states}a0{alpha_val}_entropy_z.npy', np.array(effective_entropy))
np.save(f'n{num_dist}z{z_states}x{x_states}y{y_states}a0{alpha_val}_tighter_bound_index', np.array(tighter_bound))
