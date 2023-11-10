import argparse
from utils import optimization_cf
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument('--y_x', type=float, help='The conditional probability P(y|x) that corresponding to p(y|do(x)) ', required=True)
parser.add_argument('--x', type=float, help='The marginal probability P(x) that corresponding to P(y|do(x))', required=True)
parser.add_argument('--entr', type=float, default=1, help='The upper bound of confounder entropy')


opt = parser.parse_args()

x = opt.x
x_bar = 1 - x
y_x = opt.y_x
y_bar_x = 1 - y_x

## Construct the joint distribution. Our experiment shows the bounds only depends on the value of p(y|x) and p(x), we pick arbitrary values for other entries.
pyx = np.array([[y_x*x, y_bar_x*x], [x_bar*0.5 , x_bar*0.5]]).T
lb, ub, _ = optimization_cf(pyx, ub=opt.entr, p =0, q=0)
print(f"The bounds of P(y|do(x)) is [{round(lb,3)}, {round(ub,3)}]")