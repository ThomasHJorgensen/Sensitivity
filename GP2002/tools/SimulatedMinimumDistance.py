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

    def __init__(self,model,mom_data,mom_fun,name='baseline',method='nelder-mead',lb=[],ub=[],options={'disp': False},print_iter=False,**kwargs): # called when created
        
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

    def obj_fun(self,theta,est_par,W,*args):
        
        if self.print_iter:
            for p in range(len(theta)):
                print(f' {est_par[p]}={theta[p]:2.3f}', end='')

        # 1. update parameters 
        for i in range(len(est_par)):
            setattr(self.model.par,est_par[i],theta[i]) # like par.key = val

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

    def estimate(self,theta0,est_par,W,*args):
        assert(len(W[0])==len(self.mom_data)) # check dimensions of W and mom_data

        # estimate
        self.est_out = minimize(self.obj_fun, theta0, (est_par,W, *args), method=self.method,options=self.options)

        # return output
        self.est = self.est_out.x
        self.W = W

    def std_error(self,theta,est_par,W,Omega,Nobs,Nsim,step=1.0e-4,*args):
        ''' Calculate standard errors '''

        # 1. numerical gradient of moment function wrt theta. 
        self.grad = self.num_grad_moms(theta,est_par,W,*args)

        # 2. asymptotic standard errors [using Omega: V(mom_data_i). If bootstrapped, remember to multiply by Nobs]
        GW  = np.transpose(self.grad) @ W
        GWG = GW @ self.grad

        Avar = np.linalg.inv(GWG) @ ( GW @ Omega @ np.transpose(GW) ) @ np.linalg.inv(GWG)
        fac  = (1.0 + 1.0/Nsim)/Nobs # Nsim: number of simulated observations, Nobs: number of observations in data
        self.std = np.sqrt( fac*np.diag(Avar) )

    def sensitivity(self,theta,est_par,W,fixed_par_str=None,step=1.0e-7,grad=None,do_robust=False,*args):
        ''' sensitivity measures '''

        # 1. numerical gradient of moment function wrt theta. 
        if grad is None:
            grad = self.num_grad_moms(theta,est_par,W,step=step,*args)
        self.grad = grad

        # 2. calculate key components
        GW  = np.transpose(grad) @ W
        GWG = GW @ grad
        Lambda = - np.linalg.solve(GWG , GW)

        # 3. do sensitivity
        if fixed_par_str:

            # construct vector of fixed values
            gamma = np.array([ getattr(self.model.par,name) for name in  fixed_par_str])

            # calculate gradient of the moment function with respect to gamma
            grad_g = self.num_grad_moms(gamma,fixed_par_str,W,step=step,*args)

            self.sens = Lambda @ grad_g

            elasticity = np.empty((len(theta),len(gamma)))
            for t in range(len(theta)):
                for g in range(len(gamma)):
                    elasticity[t,g] = self.sens[t,g]*gamma[g]/theta[t]    
            
            self.sens_ela = elasticity

            if do_robust: # calcualate robust sensitivity measure
                
                # a. second order derivative of the moment function wrt to theta
                grad2 = self.num_grad_grad(theta,est_par,theta,est_par,W,step=step,*args)

                # b. cross derivative of the moment function wrt to theta then gamma
                grad_cross = self.num_grad_grad(gamma,fixed_par_str,theta,est_par,W,step=step,*args)

                # c. kroniker product of weighted objective function
                self.obj_fun(theta,est_par,W,*args)
                gWkron = np.kron(np.transpose(self.diff) @ W , np.eye(len(theta)) ) 

                nom = gWkron @ grad_cross + (GW @ grad_g)
                denom = gWkron @ grad2 + GWG
                self.sens_robust = - np.linalg.solve(denom , nom)

                elasticity = np.empty((len(theta),len(gamma)))
                for t in range(len(theta)):
                    for g in range(len(gamma)):
                        elasticity[t,g] = self.sens_robust[t,g]*gamma[g]/theta[t]    

                self.sens_ela_robust = elasticity


    def num_grad_moms(self,params,names,W,step=1.0e-4,*args):
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

            self.obj_fun(params_now + step_now,names,W,*args)
            mom_forward = self.diff.copy()

            self.obj_fun(params_now - step_now,names,W,*args)
            mom_backward = self.diff.copy()

            grad[:,p] = (mom_forward - mom_backward)/(2.0*step_now[p])

        # b. reset the parameters in the model to params
        for i in range(len(names)):
            setattr(self.model.par,names[i],params[i]) 
        
        # c. return gradient
        return grad
    
    def num_grad_obj(self,obj_fun,params,names,step=1.0e-4,*args):
        """ 
        Returns the numerical gradient of the moment vector
        Inputs:
            obj_fun (callable): function that returns K-element vector
            params (1d array): L-element vector of parameters
            W (2d array): J x J weighting matrix
            step (float): step size in finite difference
            *args: additional objective function arguments
        
        Output:
            grad (2d array): K x L gradient of the moment function wrt to the elements in params
        """
        num_par = len(params)
        
        # determine number of elements in vector output via first forward
        params_now = params.copy()

        step_now  = np.zeros(num_par)
        step_now[0] = np.fmax(step,np.abs(step*params_now[0]))

        forward  = obj_fun(params_now + step_now,names,*args)
        num_vec = forward.size

        # a. numerical gradient. The objective function is (data - sim)'*W*(data - sim) so take the negative of mom_sim
        grad = np.empty((num_vec,num_par))
        for p in range(num_par):
            params_now = params.copy()

            step_now  = np.zeros(num_par)
            step_now[p] = np.fmax(step,np.abs(step*params_now[p]))

            if (p>0): forward  = obj_fun(params_now + step_now,names,*args)

            backward = obj_fun(params_now - step_now,names,*args)

            grad[:,p] = (forward - backward)/(2.0*step_now[p])
        
        # b. reset the parameters in the model to params
        for i in range(len(names)):
            setattr(self.model.par,names[i],params[i]) 
        
        # c. return gradient
        return grad

    def num_grad_grad(self,params,names,theta,est_par,W,step=1.0e-4,*args):
        """ 
        Returns the numerical gradient of the gradient wrt to theta
        
        Input: 
            params (1d array): vector of L parameter values with respect to which the second order gradient is to be calculated
            names (list): list with L parameter names with respect to which the second order gradient is to be calculated
            theta (1d array): vector of K parameter values with respect to which the first order gradient is to be calculated
            est_par (list): list with K parameter names with respect to which the first order gradient is to be calculated
            W (2d array): J x J weighting matrix
            step (float): step size in finite difference
            *args: additional objective function arguments

        Output:
            grad2 (2d array): J*K x L second order gradient of the moment function wrt to the elements in params
        
        """

        num_par = len(params)
        num_theta = len(theta)
        num_mom = len(W[0])
        grad2 = np.empty((num_mom*num_theta,num_par))

        for p in range(num_par):
            
            # set new parameters [only relevant when doing this for gamma]
            params_now = params.copy()
            step_now  = np.zeros(num_par)
            step_now[p] = np.fmax(step,np.abs(step*params_now[p]))
            
            # determine evaluation values for theta and set parameters
            params_plus = params_now + step_now
            if names == est_par:
                theta_eval = params_plus

            else:
                theta_eval = theta
                for i,par in enumerate(names):
                    setattr(self.model.par,par,params_plus[i]) 

            grad_plus = self.num_grad_moms(theta_eval,est_par,W,*args) # evaluate at theta because that will then be the basis for the gradient in that dimension
            
            # determine evaluation values for theta and set parameters
            params_minus = params_now - step_now

            if names == est_par:
                theta_eval = params_minus

            else:
                theta_eval = theta
                for i,par in enumerate(names):
                    setattr(self.model.par,par,params_minus[i]) 
            
            grad_minus = self.num_grad_moms(theta_eval,est_par,W,*args) # evaluate at theta because that will then be the basis for the gradient in that dimension

            grad_gradp = (grad_plus - grad_minus)/(2.0*step_now[p])
            grad2[:,p] = grad_gradp.ravel() # this is the same as vec(G.T) because Python is row-major

        # b. reset the parameters in the model to params
        for i,par in enumerate(names):
            setattr(self.model.par,par,params[i])

        return grad2 



        