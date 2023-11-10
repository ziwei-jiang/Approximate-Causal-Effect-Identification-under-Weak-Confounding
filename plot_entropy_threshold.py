import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable



def mutual_information(XY):
    X = XY.sum(axis=2)
    Y = XY.sum(axis=3)
    
    return entropy(X.reshape(XY.shape[0]*XY.shape[1], -1)).reshape(XY.shape[0], XY.shape[1]) + entropy(Y.reshape(XY.shape[0]* XY.shape[1], -1)).reshape(XY.shape[0], XY.shape[1]) - entropy(XY.reshape(XY.shape[0]* XY.shape[1], -1)).reshape(XY.shape[0], XY.shape[1])

def entropy(px):
    entr = px * np.log2(px, where=px!=0)
    entr[np.isnan(entr)] = 0
    return -np.sum(entr, axis=(1)) 

y_x = np.linspace(0, 1, 100)
x =  np.linspace(0, 1, 100)

## make the mesh
y_x, x = np.meshgrid(y_x, x)

yx_l = np.zeros([100, 100, 2, 2])
yx_u = np.zeros([100, 100, 2, 2])

yx_l[:,:,0,0] = y_x* x
yx_l[:,:,1,0] = (1-y_x)* x
yx_l[:,:,0,1] = 0
yx_l[:,:,1,1] = 1-x

yx_u[:,:,0,0] = y_x* x
yx_u[:,:,1,0] = (1-y_x)* x
yx_u[:,:,0,1] = 1-x
yx_u[:,:,1,1] = 0

mi_lb = mutual_information(yx_l)
mi_ub = mutual_information(yx_u)
tp_lb = np.zeros([100, 100])
tp_ub = np.zeros([100, 100])
yx = y_x*x
tp_lb = yx[:, :]
tp_ub = 1-(1 - y_x[:,:])*x
gap = tp_ub - tp_lb


# ## Get the figures with colormaps that indicate the gap

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf1 = ax.plot_surface(x, y_x, mi_lb, label='lower bound', alpha=0.9, facecolors=cm.Blues(gap))
surf2 = ax.plot_surface(x, y_x, mi_ub, label='upper bound', alpha=0.9, facecolors=cm.Oranges(gap))
ax.set_xlabel('P(x=0)')
ax.set_ylabel('P(y=0|x=0)')
ax.set_zlabel('I(Y; X)')

## set the max and min of the z axis
ax.set_zlim(0, 1)

## set viewing angle
ax.view_init(20, 30)

## show the colorbar of facecolors
m1 = cm.ScalarMappable(cmap=cm.Blues)
m2 = cm.ScalarMappable(cmap=cm.Oranges)

m1.set_array(gap)
m2.set_array(gap)
# ## make a scatter plot with same color as surf1
c1 = ax.scatter([],[],[], alpha=0.9)
c2 = ax.scatter([],[],[], alpha=0.9)
plt.colorbar(m1, shrink=0.5, aspect=30, anchor=(-1.5, 0.5))
plt.colorbar(m2, shrink=0.5, aspect=30, anchor=(-0.5, 0.5))
m2.colorbar.set_ticks([])



## make the font of axis larger
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ax.zaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
    item.set_fontsize(14)
ax.legend([c1,c2], ['Entropy threshold for lower bound', 'Entropy threshold for upper bound'], loc='center right', bbox_to_anchor=(0.95, 0.85), fontsize=20, markerscale=5)
ax.view_init(20,30)
plt.show()
