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
        ulb=-20, uub=20, 
        xlb=[-np.pi/2, -np.pi/2], 
        xub=[np.pi/2, np.pi/2],
        terminal_constraint=None)

# Part II - Simple Inverted Pendulum
sim_env = EmbeddedSimEnvironment(model=pendulum, 
                                dynamics=pendulum.discrete_time_dynamics,
                                controller=ctl.mpc_controller,
                                time = 6)

t, y, u = sim_env.run([0,0,0,0,0,0,0,0,0,0,0,0])
# # Get control gains
# ctl.set_system(A, B, C)
# K = ctl.get_closed_loop_gain()
# lr = ctl.get_feedforward_gain(K)

# Initialize simulation environment
# sim_env = EmbeddedSimEnvironment(model=quadrotor, 
#                                 dynamics=quadrotor.discrete_time_dynamics,
#                                 # controller=ctl.control_law,
#                                 time = 20)

# Enable model disturbance for second simulation environment
# pendulum.enable_disturbance(w=0.01)
# sim_env_with_disturbance = EmbeddedSimEnvironment(model=pendulum, 
#                                 dynamics=pendulum.continuous_time_nonlinear_dynamics,
#                                 controller=ctl.control_law,
#                                 time = 20)

# Also returns time and state evolution
# t, y, u = sim_env.run([0,0,0,0,0,0,0,0,0,0,0,0])
# t, y, u = sim_env_with_disturbance.run([0,0,0,0])

# pendulum = Pendulum()

    # Get the system discrete-time dynamics
    # A, B, Bw, C = pendulum.get_discrete_system_matrices_at_eq()
    # A, B, Bw, C = pendulum.get_augmented_discrete_system() # used for augmented system

    # Solve the ARE for our system to extract the terminal weight matrix P
    