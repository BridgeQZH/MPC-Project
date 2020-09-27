from model import Quadrotor
from controller import Controller
from simulation import EmbeddedSimEnvironment

# Create pendulum and controller objects
quadrotor = Quadrotor()

# Get the system discrete-time dynamics
A, B, C = quadrotor.get_discrete_system_matrices_at_eq()

# Initialize simulation environment
sim_env = EmbeddedSimEnvironment(model=quadrotor, 
                                dynamics=quadrotor.discrete_time_dynamics,
                                # controller=ctl.control_law,
                                time = 20)

# Enable model disturbance for second simulation environment
# pendulum.enable_disturbance(w=0.01)
# sim_env_with_disturbance = EmbeddedSimEnvironment(model=pendulum, 
#                                 dynamics=pendulum.continuous_time_nonlinear_dynamics,
#                                 controller=ctl.control_law,
#                                 time = 20)

# Also returns time and state evolution
x0=[0,0,0,0,0,0,0,0,0,0,0,0]
t, y, u = sim_env.run(x0=x0)
# t, y, u = sim_env_with_disturbance.run([0,0,0,0])