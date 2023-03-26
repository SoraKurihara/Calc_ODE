import numpy as np

class Solve_ODE:
    def __init__(self, dt=0.01, Func=None):
        self.dt = dt
        self.F  = Func
        
    def Euler_Method(self, X):
        X = np.array(X)
        dfdt = X + self.dt*self.F(X)
        return dfdt

    def RK_Method(self, X): # X is Matrix
        X = np.array(X)
        alpha_ = 1/6 #1/8
        beta_  = 1/3 #3/8
        gamma_ = 1/3 #3/8
        delta_ = 1/6 #1/8
        q_     = 1/2 #1/3
        r_     = 1/2 #1
        s_     = 0   #-1/3
        u_     = 1   #1
        v_     = 0   #-1
        w_     = 0   #1
        k1 = self.F(X)
        k2 = self.F(X + q_*self.dt*k1)
        k3 = self.F(X + r_*self.dt*k2 + s_*self.dt*k1)
        k4 = self.F(X + u_*self.dt*k3 + v_*self.dt*k2 + w_*self.dt*k1)
        dfdt = X + self.dt*(alpha_*k1 + beta_*k2 + gamma_*k3 + delta_*k4)
        return dfdt

    def Calculation(self, Init, T_End, Method='RK'):
        Time = np.arange(0, T_End, self.dt)
        Calc = np.empty((len(Time),len(Init)))
        Calc[0] = Init
        for t in range(1,len(Time)):
            if Method == 'RK':
                Calc[t] = self.RK_Method(Calc[t-1])
            elif Method == 'Euler':
                Calc[t] = self.Euler_Method(Calc[t-1])
        return Calc