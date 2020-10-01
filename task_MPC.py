import numpy as np
import scipy

from model import Quadrotor, Quadrotor_Integrator
from controller import Controller
from simulation import EmbeddedSimEnvironment
from mpc import MPC

ENABLE_NONLINEAR = False
ENABLE_LINEAR = False
ENABLE_AUGMENTED = True

# Create pendulum and controller objects
quadrotor = Quadrotor()
quadrotor_disturbance = Quadrotor_Integrator()
# ctl = Controller()

# Get the system discrete-time dynamics
A, B, C = quadrotor.get_discrete_system_matrices_at_eq()

Q = np.diag([100, 100, 100,
             100, 100, 100,
             1000, 1000, 1000,
             1000, 1000, 1000])
# R = np.diag([1.0 / 4, 1.0 / 4, 1.0 / 4, 1.0 / 4])
R = np.diag([1, 1, 1, 1])
# R = np.diag([0,0,0,0])

A_np = np.asarray(A)
B_np = np.asarray(B)

P = np.matrix(scipy.linalg.solve_discrete_are(A_np,B_np,Q,R))

if (ENABLE_AUGMENTED == True):
        A, B, Bw, C = quadrotor_disturbance.get_augmented_discrete_system() # used for augmented system
        Q = np.diag([   15000, 15000, 15000,
                        100, 100, 100,
                        10, 10, 10,
                        10, 10, 10, 0.0001])
        # R = np.diag([1.0 / 4, 1.0 / 4, 1.0 / 4, 1.0 / 4])
        R = np.diag([1, 1, 1, 1])
        # R = np.diag([0,0,0,0])

        A_np = np.asarray(A)
        B_np = np.asarray(B)

        P = np.matrix(scipy.linalg.solve_discrete_are(A_np,B_np,Q,R))
        # Instantiate controller
        ctl = MPC(model=quadrotor_disturbance, 
            dynamics=quadrotor_disturbance.quadrotor_augmented_dynamics,   # augmented
            horizon=5,
            Q = Q , R = R, P = P,
            ulb=None, uub=None, 
            xlb=None,   
            xub=None,       
            terminal_constraint=None)
        w = [0.2,-0.05,0.1,0]
        quadrotor_disturbance.enable_disturbance(w=w)
        sim_env_full_dist = EmbeddedSimEnvironment(model=quadrotor_disturbance, 
                                        dynamics=quadrotor_disturbance.quadrotor_augmented_dynamics,    # augmented
                                        controller=ctl.mpc_controller,
                                        time = 6)
        # sim_env_full_dist.set_window(10)
        x0=[0,0,0,0,0,0,0,0,0,0,0,0,0]
        t, y, u = sim_env_full_dist.run(x0=x0)  

'''
Nonlinear situation
'''
if(ENABLE_NONLINEAR == True):
        ctl = MPC(model = quadrotor, 
        dynamics = quadrotor.discrete_nl_dynamics,
        Q = Q, R = R, P = P,
        horizon=3,
        ulb=None, uub=None, 
        xlb=None, 
        xub=None,
        terminal_constraint=None)

        sim_env = EmbeddedSimEnvironment(model=quadrotor, 
                                        dynamics=quadrotor.discrete_nl_dynamics,
                                        controller=ctl.mpc_controller,
                                        time = 6)

        # t, y, u = sim_env.run([0,0,0,0,0,0,0,0,quadrotor.m * quadrotor.g,0,0,0])
        x0=[0,0,0,0,0,0,0,0,0,0,0,0]
        t, y, u = sim_env.run(x0=x0)  
        
'''
Linear situation
'''
if(ENABLE_LINEAR == True):
        ctl = MPC(model=quadrotor, 
                dynamics=quadrotor.discrete_time_dynamics, 
                Q = Q , R = R, P = P,
                horizon=3,
                ulb=None, uub=None, 
                xlb=None, 
                xub=None,
                terminal_constraint=None)

        sim_env = EmbeddedSimEnvironment(model=quadrotor, 
                                        dynamics=quadrotor.discrete_time_dynamics,
                                        controller=ctl.mpc_controller,
                                        time = 6)

        x0=[0,0,0,0,0,0,0,0,0,0,0,0]
        t, y, u = sim_env.run(x0=x0)


