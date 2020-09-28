import numpy as np
import scipy

from model import Quadrotor
from controller import Controller
from simulation import EmbeddedSimEnvironment
from mpc import MPC

# Create pendulum and controller objects
quadrotor = Quadrotor()
# ctl = Controller()

# Get the system discrete-time dynamics
A, B, C = quadrotor.get_discrete_system_matrices_at_eq()
Q = np.diag([1,2,3,4,5,1,1,1,1,1,1,1])
R = np.diag([1.0/4, 1.0/4, 1.0/4, 1.0/4])

A_np = np.asarray(A)
B_np = np.asarray(B)

P = np.matrix(scipy.linalg.solve_discrete_are(A_np,B_np,Q,R))

# Instantiate controller
ctl = MPC(model=quadrotor, 
        dynamics=quadrotor.discrete_time_dynamics, 
        Q = Q , R = R, P = P,
        horizon=7,
        ulb=None, uub=None, 
        xlb=None, 
        xub=None,
        terminal_constraint=None)

# Part II - Simple Inverted Pendulum
sim_env = EmbeddedSimEnvironment(model=pendulum, 
                                dynamics=pendulum.discrete_time_dynamics,
                                controller=ctl.mpc_controller,
                                time = 6)

t, y, u = sim_env.run([0,0,0,0,0,0,0,0,0,0,0,0])