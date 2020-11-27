import numpy as np
import scipy as sci
from scipy.optimize import minimize


class SimulatedMinimumDistance():
    ''' 
    This class performs simulated minimum distance (self) estimation.
    Input requirements are
    - model: Class with solution and simulation capabilities: model.solve() and model.simulate(). 
             Properties of model should be contained in model.par
    - mom_data: np.array (1d) of moments in the data to be used for estimation
    - mom_fun: function used to calculate moments in simulated data. Should return a 1d np.array
    
    '''    

    def __init__(self,model,mom_data,mom_fun,name='baseline',method='nelder-mead',est_par=[],lb=[],ub=[],options={'disp': False},print_iter=False,**kwargs): # called when created
        
        self.model = model
        self.mom_data = mom_data
        self.mom_fun = mom_fun
        self.name = name

        # estimation settings
        self.options = options
        self.print_iter = print_iter
        self.method = method

        self.lb = lb
        self.ub = ub

        self.est_par = est_par


    def obj_fun(self,theta,W,*args):
        
        if self.print_iter:
            for p in range(len(theta)):
                print(f' {self.est_par[p]}={theta[p]:2.3f}', end='')

        # 1. update parameters 
        for i in range(len(self.est_par)):
            setattr(self.model.par,self.est_par[i],theta[i]) # like par.key = val

        # 2. solve model with current parameters
        self.model.solve()

        # 3. simulate data from the model and calculate moments [have this as a complete function, used for standard errors]
        self.model.simulate()
        self.mom_sim = self.mom_fun(self.model.sim,*args)

        # 4. calculate objective function and return it
        self.diff = self.mom_data - self.mom_sim
        self.obj  = (np.transpose(self.diff) @ W) @ self.diff

        if self.print_iter:
            print(f' -> {self.obj:2.4f}')

        return self.obj 

    def estimate(self,theta0,W,*args):
        assert(len(W[0])==len(self.mom_data)) # check dimensions of W and mom_data

        # estimate
        self.est_out = minimize(self.obj_fun, theta0, (W, *args), method=self.method,options=self.options)

        # return output
        self.est = self.est_out.x
        self.W = W

    def std_error(self,theta,W,Omega,Nobs,Nsim,step=1.0e-4,*args):
        ''' Calculate standard errors and sensitivity measures '''

        # 1. numerical gradient of moment function wrt theta. 
        grad = self.num_grad_obj(theta,W,*args)

        # 2. asymptotic standard errors [using Omega: V(mom_data_i). If bootstrapped, remember to multiply by Nobs]
        GW  = np.transpose(grad) @ W
        GWG = GW @ grad

        Avar = np.linalg.inv(GWG) @ ( GW @ Omega @ np.transpose(GW) ) @ np.linalg.inv(GWG)
        fac  = (1.0 + 1.0/Nsim)/Nobs # Nsim: number of simulated observations, Nobs: number of observations in data
        self.std = np.sqrt( fac*np.diag(Avar) )

        # 3. Sensitivity measures
        self.sens1 = - np.linalg.inv(GWG) @ GW  # Andrews I, Gentzkow M, Shapiro JM: "Measuring the Sensitivity of Parameter Estimates to Estimation Moments." Quarterly Journal of Economics. 2017;132 (4) :1553-1592
       

    def sensitivity(self,theta,W,fixed_par_str=None,step=1.0e-4,grad=None,do_robust=False,*args):
        ''' sensitivity measures '''

        # 1. numerical gradient of moment function wrt theta. 
        if grad is None:
            grad = self.num_grad_obj(theta,W,*args)

        # 2. calculate key components
        GW  = np.transpose(grad) @ W
        GWG = GW @ grad
        Lambda = - np.linalg.inv(GWG) @ GW

        # 3. Sensitivity measures
        self.sens1 = Lambda  # Andrews I, Gentzkow M, Shapiro JM: "Measuring the Sensitivity of Parameter Estimates to Estimation Moments." Quarterly Journal of Economics. 2017;132 (4) :1553-1592

        # DO sensitivity
        if fixed_par_str:

            # change the estimation parameters to be the fixed ones
            est_par = self.est_par
            self.est_par = fixed_par_str

            # construct vector of fixed values
            gamma = np.empty(len(self.est_par))
            for p in range(len(self.est_par)):
                gamma[p] = getattr(self.model.par,self.est_par[p])

            # calculate gradient of the moment function with respect to gamma
            grad_g = self.num_grad_obj(gamma,W,*args)

            self.est_par = est_par
            self.sens2 = Lambda @ grad_g

            elasticity = np.empty((len(theta),len(gamma)))
            for t in range(len(theta)):
                for g in range(len(gamma)):
                    elasticity[t,g] = self.sens2[t,g]*gamma[g]/theta[t]    
            
            self.sens2e = elasticity

            # if do_robust:
            #     # calcualate robust sensitivity measure


    def num_grad_obj(self,params,W,step=1.0e-4,*args):
        """ 
        Returns the numerical gradient of the moment vector
        Inputs:
            params (1d array): K-element vector of parameters
            W (2d array): J x J weighting matrix
            step (float): step size in finite difference
            *args: additional objective function arguments
        
        Output:
            grad (2d array): J x K gradient of the moment function wrt to the elements in params
        """
        num_par = len(params)
        num_mom = len(W[0])

        # a. numerical gradient. The objective function is (data - sim)'*W*(data - sim) so take the negative of mom_sim
        grad = np.empty((num_mom,num_par))
        for p in range(num_par):
            params_now = params.copy()

            step_now  = np.zeros(num_par)
            step_now[p] = np.fmax(step,np.abs(step*params_now[p]))

            self.obj_fun(params_now + step_now,W,*args)
            mom_forward = self.diff

            self.obj_fun(params_now - step_now,W,*args)
            mom_backward = self.diff

            grad[:,p] = (mom_forward - mom_backward)/(2.0*step_now[p])

        # b. reset the parameters in the model to params
        for i in range(len(self.est_par)):
            setattr(self.model.par,self.est_par[i],params[i]) 
        
        # c. return gradient
        return grad



        