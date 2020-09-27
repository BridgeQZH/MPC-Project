import casadi as ca
# x = ca.MX.sym('x', 12)
C_eq = ca.DM.zeros(4,1)
a, b, c, d = ca.vertsplit(C_eq)
print(a, b, c, d)