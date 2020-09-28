import casadi as ca
import numpy as np
# y_vec = np.array(x0)
# x = ca.DM(np.size(y_vec,0),1).full()
# x = np.array([y_vec[:,-1]]).T
# print(x)

# u_vec = np.array([0.1,0,0,0]).T
# u = ca.DM(np.size(u_vec,0),1).full()

# print(u)

obj = ca.MX(4)
print(obj)