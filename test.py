import casadi as ca
# x = ca.MX.sym('x', 12)
C_eq = ca.DM.zeros(1,12)
C_eq = C_eq + 1
print(C_eq)