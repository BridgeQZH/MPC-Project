import casadi as ca
import numpy as np
from filterpy.kalman import KalmanFilter

class Quadrotor(object):
    def __init__(self, h=0.1):
        """
        quadrotor model class. 
        
        Describes the movement of a quadrotor with mass 'm' attached to a cart
        with mass 'M'. All methods should return casadi.MX or casadi.DM variable 
        types.

        :param h: sampling time, defaults to 0.1
        :type h: float, optional
        """

        # Model, gravity and sampling time parameters
        self.model = self.quadrotor_linear_dynamics
        self.model_nl = self.quadrotor_nonlinear_dynamics
        self.g = 9.81
        self.l = 0.175
        self.dt = h

        # System reference (x_d) and disturbance (w)
        self.p_d = ca.DM.zeros(3,1)               # position reference
        # self.p_d = [0.5,0.5,0.5]
        self.v_d = ca.DM.zeros(3,1)               # velocity reference
        self.alpha_d = ca.DM.zeros(3,1)           # orientation reference
        self.omega_d = ca.DM.zeros(3,1)          # angular velocity reference
        self.x_d = ca.vertcat(self.p_d, self.v_d, self.alpha_d, self.omega_d)        # system state reference
        self.x_d[2] = 0.5
        self.x_d[1] = 0.5
        self.x_d[0] = 0.5
        # print(self.x_d)
        self.w = 0.0

        # Quadrotor Parameters
        self.m = 1.4            # quadrotor mass
        self.M_x = 0.001         # Inertia along x-axis
        self.M_y = 0.001         # Inertia along y-axis
        self.M_z = 0.005         # Inertia along z-axis

        # Aggregated terms
        
        # Linearize system around vertical equilibrium with no input
        self.p = ca.DM.zeros(3,1)               # position state
        self.v = ca.DM.zeros(3,1)               # velocity state
        self.alpha = ca.DM.zeros(3,1)           # orientation state
        self.omega = ca.DM.zeros(3,1)           # angular velocity state
        self.x_eq = ca.vertcat(self.p, self.v, self.alpha, self.omega)     # system state vertical stack
        self.u_eq = ca.DM.zeros(4,1)            # control input (f_t, tau_x, tau_y, tau_z)
        self.u_eq[0] = self.m * self.g
        self.Integrator = None

        self.set_integrators()
        self.set_discrete_time_system()

        print("quadrotor class initialized")
        print(self)                         # You can comment this line

    def __str__(self):
        return """                                                                  
                                                 
                     Z                                                 
                     ^                                                      
                     |                                                      
                     |                                                         
                     |                                                   
                     +-------> Y                                                  
                    -                           
                   -
                  -
             X  <-                                               
            -----------------------------------------------------------      """

    def set_integrators(self):
        """
        Generate continuous time high-precision integrators.
        """
        
        # Set CasADi variables
        x = ca.MX.sym('x', 12)
        u = ca.MX.sym('u', 4)

        # Integration method - integrator options an be adjusted
        options = {"abstol" : 1e-5, "reltol" : 1e-9, "max_num_steps": 100, 
                   "tf" : self.dt}

        # Create linear dynamics integrator
        dae = {'x': x, 'ode': self.model(x,u), 'p':ca.vertcat(u)}
        self.Integrator = ca.integrator('integrator', 'cvodes', dae, options)

        # Create nonlinear dynamics integrator
        dae = {'x': x, 'ode': self.model_nl(x,u), 'p':ca.vertcat(u)}
        self.Integrator_nl = ca.integrator('integrator', 'cvodes', dae, options)

    def set_discrete_time_system(self):
        """
        Set discrete-time system matrices from linear continuous dynamics.
        """
        
        # Check for integrator definition
        if self.Integrator is None:
            print("Integrator not defined. Set integrators first.")
            exit()

        # Set CasADi variables
        x = ca.MX.sym('x', 12)
        u = ca.MX.sym('u', 4)
    
        # Jacobian of exact discretization
        self.Ad = ca.Function('jac_x_Ad', [x, u], [ca.jacobian(
                            self.Integrator(x0=x, p=u)['xf'], x)])
        self.Bd = ca.Function('jac_u_Bd', [x, u], [ca.jacobian(
                            self.Integrator(x0=x, p=u)['xf'], u)])


    def quadrotor_linear_dynamics(self, x, u):  
        """ 
        quadrotor continuous-time linearized dynamics.

        :param x: state
        :type x: MX variable, 12x1
        :param u: control input
        :type u: MX variable, 4x1
        :return: dot(x)
        :rtype: MX variable, 12x1
        """

        theta, phi, psi = ca.vertsplit(self.alpha)
        w_x, w_y, w_z = ca.vertsplit(self.omega)

        f_z = self.g * self.m   # input control at equilibrium point
        # m = self.m
        Ac = ca.MX.zeros(12,12)
        Bc = ca.MX.zeros(12,4)

        J_a = ca.MX.zeros(3,3)
        J_b = ca.MX.zeros(3,3)
        J_c = ca.MX.zeros(3,3)
        J_d = ca.MX.zeros(3,3)
        
        J_a[0,0] = ca.cos(theta) * ca.sin(psi) - ca.sin(theta) * ca.sin(phi) * ca.cos(psi)
        J_a[0,1] = ca.cos(theta) * ca.cos(phi) * ca.cos(psi)
        J_a[0,2] = ca.sin(theta) * ca.cos(psi) - ca.cos(theta) * ca.sin(phi) * ca.sin(psi)
        J_a[1,0] = -ca.cos(theta) * ca.cos(psi) - ca.sin(theta) * ca.sin(phi) * ca.sin(psi)
        J_a[1,1] = ca.cos(theta) * ca.cos(phi) * ca.sin(psi)
        J_a[1,2] = ca.sin(theta) * ca.sin(psi) + ca.cos(theta) * ca.sin(phi) * ca.cos(psi)
        J_a[2,0] = -ca.sin(theta) * ca.cos(phi)
        J_a[2,1] = -ca.cos(theta) * ca.sin(phi)
        J_a *= f_z / self.m 

        J_b[0,0] = (1.0/ca.cos(theta))**2 * (w_y * ca.sin(phi) + w_z * ca.cos(phi))
        J_b[0,1] = ca.tan(theta) * (w_y * ca.cos(phi) - w_z * ca.sin(phi))
        J_b[1,1] = - w_y * ca.sin(phi) - w_z * ca.cos(phi)
        J_b[2,0] = (1.0 / ca.cos(theta)) * ca.tan(theta) * (w_y * ca.sin(phi) + w_z * ca.cos(phi))
        J_b[2,1] = (1.0 / ca.cos(theta)) * (w_y * ca.cos(phi) - w_z * ca.sin(phi))
        
        J_c[0,0] = 1.0
        J_c[0,1] = ca.sin(phi) * ca.tan(theta)
        J_c[0,2] = ca.cos(phi) * ca.tan(theta)
        J_c[1,1] = ca.cos(phi)
        J_c[1,2] = -ca.sin(phi)
        J_c[2,1] = 1.0 / ca.cos(theta) * ca.sin(phi)
        J_c[2,2] = 1.0 / ca.cos(theta) * ca.cos(phi)

        J_d[0,1] = w_z * (self.M_y - self.M_z) / self.M_x
        J_d[0,2] = w_y * (self.M_y - self.M_z) / self.M_x
        J_d[1,0] = w_z * (self.M_z - self.M_x) / self.M_y
        J_d[1,2] = w_x * (self.M_z - self.M_x) / self.M_y
        J_d[2,0] = w_y * (self.M_x - self.M_y) / self.M_z
        J_d[2,1] = w_x * (self.M_x - self.M_y) / self.M_z

        ### Build Ac matrix
        Ac[0:3,3:6] = ca.MX.eye(3)
        Ac[3:6,6:9] = J_a 
        Ac[6:9,6:9] = J_b
        Ac[6:9,9:12] = J_c
        Ac[9:12,9:12] = J_d

        ### Build Bc matrix
        J_e = ca.MX.zeros(3,1)
        J_f = ca.MX.zeros(3,3)

        J_e[0,0] = (ca.sin(theta)*ca.sin(psi)+ ca.sin(phi)*ca.cos(psi)*ca.cos(theta))/self.m
        J_e[1,0] = (-ca.sin(theta)*ca.cos(psi)+ ca.sin(phi)*ca.cos(theta)*ca.sin(psi))/self.m
        J_e[2,0] = ca.cos(theta)*ca.cos(phi)/self.m

        J_f[0,0] = 1.0/self.M_x
        J_f[1,1] = 1.0/self.M_y
        J_f[2,2] = 1.0/self.M_z

        Bc[3:6,0] = J_e
        Bc[9:12,1:4] = J_f

        self.Ac = Ac
        self.Bc = Bc

        return Ac @ x + Bc @ u

    def quadrotor_nonlinear_dynamics(self, x, u, *_):
        """
        quadrotor nonlinear dynamics.

        :param x: state
        :type x: casadi.DM or casadi.MX
        :param u: control input
        :type u: casadi.DM or casadi.MX
        :return: state time derivative
        :rtype: casadi.DM or casadi.MX, depending on inputs
        """
        
        p_x, p_y, p_z = ca.vertsplit(x[0:3])
        v_x, v_y, v_z = ca.vertsplit(x[3:6])
        theta, phi, psi = ca.vertsplit(x[6:9])
        w_x, w_y, w_z = ca.vertsplit(x[9:12])

        f_z, tau_x, tau_y, tau_z = ca.vertsplit(u)

        # dot(v_x)
        f1 = (ca.sin(theta) * ca.sin(psi) + ca.cos(theta) * ca.sin(phi) * ca.cos(psi)) * f_z / self.m

        # dot(v_y)
        f2 = (-ca.sin(theta) * ca.cos(psi) + ca.cos(theta) * ca.sin(phi) * ca.sin(psi)) * f_z / self.m

        # dot(v_z)
        f3 = ca.cos(theta) * ca.cos(phi) * f_z / self.m - self.g

        # dot(theta)
        f4 = w_x + ca.tan(theta) * (w_y * ca.sin(phi) + w_z * ca.cos(phi))

        # dot(phi)
        f5 = w_y * ca.cos(phi) - w_z * ca.sin(phi)

        # dot(psi)
        f6 = (w_y * ca.sin(phi) + w_z * ca.cos(phi)) / ca.cos(theta)

        # dot(w_x)
        f7 = (tau_x - w_y * w_z * (self.M_z - self.M_y)) / self.M_x

        # dot(w_y)
        f8 = (tau_y - w_x * w_z * (self.M_x - self.M_z)) / self.M_y

        # dot(w_z)
        f9 = (tau_z - w_x * w_y * (self.M_y - self.M_x)) / self.M_z
        
        dxdt = [ v_x, v_y, v_z, f1, f2, f3, f4, f5, f6, f7, f8, f9 ]

        return ca.vertcat(*dxdt)

    def set_reference(self, ref):
        """
        Simple method to set the new system reference.

        :param ref: desired reference [m]
        :type ref: float or casadi.DM 1x1
        """
        self.x_d = ref
        
    def get_discrete_system_matrices_at_eq(self):
        """
        Evaluate the discretized matrices at the equilibrium point

        :return: A,B,C matrices for equilibrium point
        :rtype: casadi.DM 
        """
        A_eq = self.Ad(self.x_eq, self.u_eq)
        B_eq = self.Bd(self.x_eq, self.u_eq)
        
        # Populate a full observation matrix
        C_eq = ca.DM.zeros(1,12) + 1
                
        return A_eq, B_eq, C_eq
        # Bw_eq = self.Bw(self.x_eq, self.u_eq, self.w)

        # return Ad_eq, Bd_eq, Bw_eq, self.Cd_eq



    def discrete_integration(self, x0, u):
        """
        Perform a time step iteration in continuous dynamics.

        :param x0: initial state
        :type x0: 4x1 ( list [a, b, c, d] , ca.MX )
        :param u: control input
        :type u: scalar, 1x1
        :return: dot(x), time derivative
        :rtype: 4x1, ca.DM
        """
        out = self.Integrator(x0=x0, p=u)
        return out["xf"]

    def discrete_nl_dynamics(self, x0, u):
        """Discrete time nonlinear integrator

        :param x0: starting state
        :type x0: ca.DM
        :param u: control input
        :type u: ca.DM
        :return: state at the final sampling interval time
        :rtype: ca.DM
        """
        out = self.Integrator_nl(x0=x0, p=u)
        return out["xf"]

    def discrete_time_dynamics(self,x0,u):
        """ 
        Performs a discrete time iteration step.

        :param x0: initial state
        :type x0: 4x1 ( list [a, b, c, d] , ca.MX )
        :param u: control input
        :type u: scalar, 1x1
        :return: next discrete time state
        :rtype: 4x1, ca.DM
        """
        # Non-linear
        # return self.Ad(self.x_eq, self.u_eq, self.w) @ x0 + \
        #         self.Bd(self.x_eq, self.u_eq, self.w) @ u + \
        #         self.Bw(self.x_eq, self.u_eq, self.w) @ self.w

        return self.Ad(self.x_eq, self.u_eq) @ x0 + \
                self.Bd(self.x_eq, self.u_eq) @ u

    def enable_disturbance(self, w=0.01):
        """
        Enable system disturbance as a wind force.

        :param w: disturbance magnitude, defaults to 0.1
        :type w: float, optional
        """

        # Activate disturbance
        self.w = w

        # Re-generate integrators for dynamics with disturbance
        self.set_integrators()
        self.set_augmented_discrete_system()

    pass


class Quadrotor_Integrator(object):
    def __init__(self, h=0.1):
        """
        quadrotor model class. 
        
        Describes the movement of a quadrotor with mass 'm' attached to a cart
        with mass 'M'. All methods should return casadi.MX or casadi.DM variable 
        types.

        :param h: sampling time, defaults to 0.1
        :type h: float, optional
        """

        # Model, gravity and sampling time parameters
        self.model = self.quadrotor_linear_dynamics
        self.model_nl = self.quadrotor_nonlinear_dynamics
        self.model_ag = self.quadrotor_augmented_dynamics
        self.g = 9.81
        self.l = 0.175
        self.dt = h

        # System reference (x_d) and disturbance (w)
        self.p_d = ca.DM.zeros(3,1)               # position reference
        self.v_d = ca.DM.zeros(3,1)               # velocity reference
        self.alpha_d = ca.DM.zeros(3,1)           # orientation reference
        self.omega_d = ca.DM.zeros(3,1)           # angular velocity reference
        self.xixi = 1
        self.x_d = ca.vertcat(self.p_d, self.v_d, self.alpha_d, self.omega_d, self.xixi)        # system state reference
        self.x_d[2] = 2
        self.x_d[1] = 3
        self.x_d[0] = 4
        self.w = ca.DM.zeros(4,1)

        # quadrotor Parameters
        self.m = 1.4            # quadrotor mass
        self.M_x = 0.001         # Inertia along x-axis
        self.M_y = 0.001         # Inertia along y-axis
        self.M_z = 0.005         # Inertia along z-axis

        # Linearize system around vertical equilibrium with no input
        self.p = ca.DM.zeros(3,1)               # position state
        self.v = ca.DM.zeros(3,1)               # velocity state
        self.alpha = ca.DM.zeros(3,1)           # orientation state
        self.omega = ca.DM.zeros(3,1)           # angular velocity state
        self.x_eq = ca.vertcat(self.p, self.v, self.alpha, self.omega)     # system state vertical stack
        self.u_eq = ca.DM.zeros(4,1)            # control input (f_t, tau_x, tau_y, tau_z)
        self.u_eq[0] = self.m * self.g
        self.Integrator_lin = None
        self.Integrator = None
        self.Integrator_ag = None
        self.Ad_i = None

        self.set_integrators()
        self.set_discrete_time_system()
        self.set_augmented_discrete_system()

        print("quadrotor class initialized")
        print(self)                         # You can comment this line

    def __str__(self):
        return """
                NON-LINEAR                                
                     Z                                                 
                     ^                                                      
                     |                                                      
                     |                                                         
                     |                                                   
                     +-------> Y                                                  
                    -                           
                   -
                  -
             X  <-                                                                        
                    """
    def quadrotor_nonlinear_dynamics(self, x, u, *_):
        """
        quadrotor nonlinear dynamics.

        :param x: state
        :type x: casadi.DM or casadi.MX
        :param u: control input
        :type u: casadi.DM or casadi.MX
        :return: state time derivative
        :rtype: casadi.DM or casadi.MX, depending on inputs
        """
        
        p_x, p_y, p_z = ca.vertsplit(x[0:3])
        v_x, v_y, v_z = ca.vertsplit(x[3:6])
        theta, phi, psi = ca.vertsplit(x[6:9])
        w_x, w_y, w_z = ca.vertsplit(x[9:12])

        f_z, tau_x, tau_y, tau_z = ca.vertsplit(u)

        # dot(v_x)
        f1 = (ca.sin(theta) * ca.sin(psi) + ca.cos(theta) * ca.sin(phi) * ca.cos(psi)) * f_z / self.m

        # dot(v_y)
        f2 = (-ca.sin(theta) * ca.cos(psi) + ca.cos(theta) * ca.sin(phi) * ca.sin(psi)) * f_z / self.m

        # dot(v_z)
        f3 = ca.cos(theta) * ca.cos(phi) * f_z / self.m - self.g

        # dot(theta)
        f4 = w_x + ca.tan(theta) * (w_y * ca.sin(phi) + w_z * ca.cos(phi))

        # dot(phi)
        f5 = w_y * ca.cos(phi) - w_z * ca.sin(phi)

        # dot(psi)
        f6 = (w_y * ca.sin(phi) + w_z * ca.cos(phi)) / ca.cos(theta)

        # dot(w_x)
        f7 = (tau_x - w_y * w_z * (self.M_z - self.M_y)) / self.M_x

        # dot(w_y)
        f8 = (tau_y - w_x * w_z * (self.M_x - self.M_z)) / self.M_y

        # dot(w_z)
        f9 = (tau_z - w_x * w_y * (self.M_y - self.M_x)) / self.M_z
        
        dxdt = [ v_x, v_y, v_z, f1, f2, f3, f4, f5, f6, f7, f8, f9 ]

        return ca.vertcat(*dxdt)
    def set_integrators(self):
        """
        Generate continuous time high-precision integrators.
        """
        
        # Set CasADi variables
        x = ca.MX.sym('x', 12)
        u = ca.MX.sym('u', 4)
        w = ca.MX.sym('w', 4)
        # Integration method - integrator options an be adjusted
        options = {"abstol" : 1e-5, "reltol" : 1e-9, "max_num_steps": 100, 
                   "tf" : self.dt}

        # Create linear dynamics integrator
        dae = {'x': x, 'ode': self.model(x,u,w), 'p':ca.vertcat(u,w)}
        self.Integrator_lin = ca.integrator('integrator', 'cvodes', dae, options)
        
        # Create nonlinear dynamics integrator
        dae = {'x': x, 'ode': self.model_nl(x,u), 'p':ca.vertcat(u)}
        self.Integrator = ca.integrator('integrator', 'cvodes', dae, options)

        if self.Ad_i is not None:
            # Create augmented system dynamics integrator
            x_ag = ca.MX.sym('x', 13)
            dae = {'x': x_ag, 'ode': self.model_ag(x_ag,u), 'p':ca.vertcat(u)}
            self.Integrator_ag = ca.integrator('integrator', 'cvodes', dae, options)

    def set_discrete_time_system(self):
        """
        Set discrete-time system matrices from linear continuous dynamics.
        """
        
        # Check for integrator definition
        if self.Integrator_lin is None:
            print("Integrator_lin not defined. Set integrators first.")
            exit()

        # Set CasADi variables
        x = ca.MX.sym('x', 12)
        u = ca.MX.sym('u', 4)
        w = ca.MX.sym('w', 4)
    
        # Jacobian of exact discretization
        self.Ad = ca.Function('jac_x_Ad', [x, u, w], [ca.jacobian(
                            self.Integrator_lin(x0=x, p=ca.vertcat(u, w))['xf'], x)])
        self.Bd = ca.Function('jac_u_Bd', [x, u, w], [ca.jacobian(
                            self.Integrator_lin(x0=x, p=ca.vertcat(u, w))['xf'], u)])
        self.Bw = ca.Function('jac_u_Bd', [x, u, w], [ca.jacobian(
                            self.Integrator_lin(x0=x, p=ca.vertcat(u, w))['xf'], w)])
        

        # C matrix does not depend on the state
        # TODO: put this in a better place later!
        Cd_eq = ca.DM.zeros(1,12)+1
        #Cd_eq[0,0] = 1
        #Cd_eq[0,1] = 1
        #Cd_eq[0,2] = 1
        #Cd_eq[0,3] = 1

        self.Cd_eq = Cd_eq

    def quadrotor_linear_dynamics(self, x, u, w):  
        """ 
        quadrotor continuous-time linearized dynamics.

        :param x: state
        :type x: MX variable, 12x1
        :param u: control input
        :type u: MX variable, 4x1
        :return: dot(x)
        :rtype: MX variable, 12x1
        """

        theta, phi, psi = ca.vertsplit(self.alpha)
        w_x, w_y, w_z = ca.vertsplit(self.omega)

        f_z = self.g * self.m   # input control at equilibrium point
        # m = self.m
        Ac = ca.MX.zeros(12,12)
        Bc = ca.MX.zeros(12,4)
        Bwc = ca.MX.zeros(12,4)
        Awc = ca.MX.zeros(12,12)

        J_a = ca.MX.zeros(3,3)
        J_b = ca.MX.zeros(3,3)
        J_c = ca.MX.zeros(3,3)
        J_d = ca.MX.zeros(3,3)
        
        J_a[0,0] = ca.cos(theta) * ca.sin(psi) - ca.sin(theta) * ca.sin(phi) * ca.cos(psi)
        J_a[0,1] = ca.cos(theta) * ca.cos(phi) * ca.cos(psi)
        J_a[0,2] = ca.sin(theta) * ca.cos(psi) - ca.cos(theta) * ca.sin(phi) * ca.sin(psi)
        J_a[1,0] = -ca.cos(theta) * ca.cos(psi) - ca.sin(theta) * ca.sin(phi) * ca.sin(psi)
        J_a[1,1] = ca.cos(theta) * ca.cos(phi) * ca.sin(psi)
        J_a[1,2] = ca.sin(theta) * ca.sin(psi) + ca.cos(theta) * ca.sin(phi) * ca.cos(psi)
        J_a[2,0] = -ca.sin(theta) * ca.cos(phi)
        J_a[2,1] = -ca.cos(theta) * ca.sin(phi)
        J_a *= f_z / self.m 

        J_b[0,0] = (1.0/ca.cos(theta))**2 * (w_y * ca.sin(phi) + w_z * ca.cos(phi))
        J_b[0,1] = ca.tan(theta) * (w_y * ca.cos(phi) - w_z * ca.sin(phi))
        J_b[1,1] = - w_y * ca.sin(phi) - w_z * ca.cos(phi)
        J_b[2,0] = (1.0 / ca.cos(theta)) * ca.tan(theta) * (w_y * ca.sin(phi) + w_z * ca.cos(phi))
        J_b[2,1] = (1.0 / ca.cos(theta)) * (w_y * ca.cos(phi) - w_z * ca.sin(phi))
        
        J_c[0,0] = 1.0
        J_c[0,1] = ca.sin(phi) * ca.tan(theta)
        J_c[0,2] = ca.cos(phi) * ca.tan(theta)
        J_c[1,1] = ca.cos(phi)
        J_c[1,2] = -ca.sin(phi)
        J_c[2,1] = 1.0 / ca.cos(theta) * ca.sin(phi)
        J_c[2,2] = 1.0 / ca.cos(theta) * ca.cos(phi)

        J_d[0,1] = w_z * (self.M_y - self.M_z) / self.M_x
        J_d[0,2] = w_y * (self.M_y - self.M_z) / self.M_x
        J_d[1,0] = w_z * (self.M_z - self.M_x) / self.M_y
        J_d[1,2] = w_x * (self.M_z - self.M_x) / self.M_y
        J_d[2,0] = w_y * (self.M_x - self.M_y) / self.M_z
        J_d[2,1] = w_x * (self.M_x - self.M_y) / self.M_z

        ### Build Ac matrix
        Ac[0:3,3:6] = ca.MX.eye(3)
        Ac[3:6,6:9] = J_a 
        Ac[6:9,6:9] = J_b
        Ac[6:9,9:12] = J_c
        Ac[9:12,9:12] = J_d

        ### Build Bc matrix
        J_e = ca.MX.zeros(3,1)
        J_f = ca.MX.zeros(3,3)

        J_e[0,0] = (ca.sin(theta)*ca.sin(psi)+ ca.sin(phi)*ca.cos(psi)*ca.cos(theta))/self.m
        J_e[1,0] = (-ca.sin(theta)*ca.cos(psi)+ ca.sin(phi)*ca.cos(theta)*ca.sin(psi))/self.m
        J_e[2,0] = ca.cos(theta)*ca.cos(phi)/self.m

        J_f[0,0] = 1.0/self.M_x
        J_f[1,1] = 1.0/self.M_y
        J_f[2,2] = 1.0/self.M_z

        Bc[3:6,0] = J_e
        Bc[9:12,1:4] = J_f

        ### Build Bwc
        l = self.l
        k = 0.01
        Bwc_disturbance = ca.MX.zeros(4,4)
        Bwc_disturbance[0,0:4] = [1,1,1,1]
        Bwc_disturbance[1,0:4] = [l,0,-l,0]
        Bwc_disturbance[2,0:4] = [0,-l,0,l]
        Bwc_disturbance[3,0:4] = [-k, k, -k, k]
        Bwc = Bc @ Bwc_disturbance
        ### Store matrices as class variables
        self.Ac = Ac
        self.Bc = Bc
        self.Bwc = Bwc
        self.Awc = Awc  

        return Ac @ x + Bc @ u + Bwc @ w

    def set_reference(self, ref):	
        """	
        Simple method to set the new system reference.	
        :param ref: desired reference [m]	
        :type ref: float or casadi.DM 1x1	
        """	
        self.x_d = ref
    
    def quadrotor_augmented_dynamics(self, x, u):
        """Augmented pendulum system dynamics

        :param x: state
        :type x: casadi.DM
        :param u: control input
        :type u: casadi.DM
        :return: next state
        :rtype: casadi.DM
        """

        return self.Ad_i @ x + self.Bd_i @ u + self.R_i * self.x_d + self.Bw_i @ self.w

    def get_discrete_system_matrices_at_eq(self):	
        """	
        Evaluate the discretized matrices at the equilibrium point	
        :return: A,B,C matrices for equilibrium point	
        :rtype: casadi.DM 	
        """	
        Ad_eq = self.Ad(self.x_eq, self.u_eq, self.w)	
        Bd_eq = self.Bd(self.x_eq, self.u_eq, self.w)	
        Bw_eq = self.Bw(self.x_eq, self.u_eq, self.w)	

        return Ad_eq, Bd_eq, Bw_eq, self.Cd_eq

    def set_augmented_discrete_system(self):
        """
        quadrotor dynamics with integral action.

        :param x: state
        :type x: casadi.DM
        :param u: control input
        :type u: casadi.DM
        """

        # Grab equilibrium dynamics
        Ad_eq = self.Ad(self.x_eq, self.u_eq, self.w)
        Bd_eq = self.Bd(self.x_eq, self.u_eq, self.w)

        # Instantiate augmented system
        self.Ad_i = ca.DM.zeros(13,13)
        self.Bd_i = ca.DM.zeros(13,4)
        self.Bw_i = ca.DM.zeros(13,4)
        self.Cd_i = ca.DM.zeros(1,13)
        self.R_i = ca.DM.zeros(13,1)

        # Populate matrices
        self.Ad_i[0:12,0:12] = Ad_eq
        self.Ad_i[12,0:12] = self.dt @ self.Cd_eq
        self.Ad_i[12,12] = 1

        self.Bd_i[0:12,0:4] = Bd_eq

        self.Bw_i[0:12,0:4] = self.Bw(self.x_eq, self.u_eq, self.w)

        self.R_i[12,0] = self.dt

        self.Cd_i[0,0:12] = self.Cd_eq

    def discrete_nl_dynamics(self, x0, u):	
        """Discrete time nonlinear integrator	
        :param x0: starting state	
        :type x0: ca.DM	
        :param u: control input	
        :type u: ca.DM	
        :return: state at the final sampling interval time	
        :rtype: ca.DM	
        """	
        out = self.Integrator(x0=x0, p=u)	
        return out["xf"]	

    def discrete_time_dynamics(self,x0,u):	
        """ 	
        Performs a discrete time iteration step.	
        :param x0: initial state	
        :type x0: 4x1 ( list [a, b, c, d] , ca.MX )	
        :param u: control input	
        :type u: scalar, 1x1	
        :return: next discrete time state	
        :rtype: 4x1, ca.DM	
        """	

        return self.Ad(self.x_eq, self.u_eq, self.w) @ x0 + self.Bd(self.x_eq, self.u_eq, self.w) @ u + self.Bw(self.x_eq, self.u_eq, self.w) @ self.w	

    def enable_disturbance(self, w):	
        """	
        Enable system disturbance as a wind force.	
        :param w: disturbance magnitude, defaults to 0.1	
        :type w: float, optional	
        """	


        # Activate disturbance	
        self.w = w	
        # Re-generate dynamics	
        self.set_integrators()	
        self.set_discrete_time_system()	
        self.set_augmented_discrete_system()
    

    def quadrotor_linear_dynamics_with_disturbance(self, x, u):
        Ad_eq = self.Ad(self.x_eq, self.u_eq, self.w)
        Bd_eq = self.Bd(self.x_eq, self.u_eq, self.w)
        Bw_eq = self.Bw(self.x_eq, self.u_eq, self.w)

        return Ad_eq @ x + Bd_eq @ u + Bw_eq @ self.w 

    def quadrotor_augmented_dynamics(self, x, u):
        """Augmented quadrotor system dynamics

        :param x: state
        :type x: casadi.DM
        :param u: control input
        :type u: casadi.DM
        :return: next state
        :rtype: casadi.DM
        """

        return self.Ad_i @ x + self.Bd_i @ u + self.R_i * self.x_d + self.Bw_i @ self.w


    def set_equilibrium_point(self, x_eq, u_eq):
        """
        Set a different equilibrium poin for the system.

        :param x_eq: state equilibrium
        :type x_eq: list with 4 floats, [a,b,c,d]
        :param u_eq: control input for equilibrium point
        :type u_eq: float
        """

        self.x_eq = x_eq
        self.u_eq = u_eq
        
    

    def get_augmented_discrete_system(self):
        """
        Get discrete time augmented system with integral component.

        :return: System A, B, C matrices
        :rtype: casadi.DM
        """
        return self.Ad_i, self.Bd_i, self.Bw_i, self.Cd_i


    def continuous_time_linear_dynamics(self, x0, u):
        """
        Perform a time step iteration in continuous dynamics.

        :param x0: initial state
        :type x0: 4x1 ( list [a, b, c, d] , ca.MX )
        :param u: control input
        :type u: scalar, 1x1
        :return: dot(x), time derivative
        :rtype: 4x1, ca.DM
        """
        out = self.Integrator_lin(x0=x0, p=ca.vertcat(u, self.w))
        return out["xf"]

    def continuous_time_nonlinear_dynamics(self, x0, u):
        out = self.Integrator(x0=x0, p=ca.vertcat(u, self.w))
        return out["xf"]

    #===============================================#
    #            Kalman Filter modules              # 
    #===============================================#

    def set_kf_params(self, C, Q, R):
        """
        Set the Kalman Filter variables.

        :param C: observation matrix
        :type C: numpy.array
        :param Q: process noise
        :type Q: numpy.array
        :param R: measurement noise
        :type R: numpy.array
        """
        self.C_KF = C 
        self.Q_KF = Q
        self.R_KF = R

    def init_kf(self,  ):
        """
        Initialize the Kalman Filter estimator.
        """

        # Initialize filter object
        self.kf_estimator = KalmanFilter(dim_x=4, dim_z=np.size(self.C_KF,0), dim_u=2)

        Ad_eq = self.Ad(self.x_eq, self.u_eq, self.w)
        Bd_eq = self.Bd(self.x_eq, self.u_eq, self.w)
        Bw_eq = self.Bw(self.x_eq, self.u_eq, self.w)

        B = ca.DM.zeros(4,2)
        B[:,0] = Bd_eq
        B[:,1] = Bw_eq

        # Set filter parameters
        self.kf_estimator.F = np.asarray(Ad_eq)
        self.kf_estimator.B = np.asarray(B)

        self.kf_estimator.H = self.C_KF
        self.kf_estimator.Q = self.Q_KF
        self.kf_estimator.R = self.R_KF
        # Set initial estimation
        self.kf_estimator.x = x.T 