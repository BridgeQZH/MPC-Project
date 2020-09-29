import casadi as ca
import numpy as np
obj = np.full((4),1)
print(obj[2])

x0_ref   = ca.MX.sym('x0_ref', 12)
x0_ref[0:3] = 0.5
print(x0_ref)