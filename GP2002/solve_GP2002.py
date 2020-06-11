"""
Solves the consumption-saving model in Gourinchas and Parker (2002, Econometrica)
using EGM 

"""

##############
# 1. imports #
##############

import time
import pickle
import numpy as np
from numba import njit, jitclass, prange, boolean, int32, double
import math

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# tools
from tools import linear_interp # for linear interpolation
from tools import misc # various tools
from tools.ConsumptionSavingModel import ConsumptionSavingModel # baseline model classes

import tools.plot as plot

############
# 2. model #
############

# a. parameter class (numba)
parlist = [
    ('Tr',int32),
    ('age_min',int32),
    ('age_max',int32), 
    ('beta',double), 
    ('rho',double),
    ('gamma0',double),
    ('gamma1',double),
    ('G',double[:]),
    ('v',double[:]),
    ('r',double),
    ('R',double),
    ('credit',double),
    ('sigma_trans',double), 
    ('Ntrans',int32),
    ('sigma_perm',double), 
    ('Nperm',int32),
    ('p',double), 
    ('mu',double), 
    ('Na',int32),
    ('grid_a',double[:]), 
    ('grid_m',double[:]),  
    ('grid_age',double[:]),        
    ('a_max',double),    
    ('Nshocks',int32),        
    ('trans',double[:]),
    ('trans_w',double[:]),     
    ('perm',double[:]),
    ('perm_w',double[:]),     
    ('do_print',boolean), # boolean
    ('simN',int32),
    ('mu_a_init',double),
    ('sigma_a_init',double),
    ('init_P',double),
    ('sol_gp',boolean)
]
@jitclass(parlist) # jit class with variables in parlist
class ParClass():
    def __init__(self):
        pass
    
# b. solution class (numba)
sollist = [
    ('m',double[:,:]),
    ('c',double[:,:]),    
]
@jitclass(sollist) # jit class with variables in sollist
class SolClass():
    def __init__(self):
        pass

# b. solution class (numba)
simlist = [
    ('c',double[:,:]), 
    ('m',double[:,:]), 
    ('Y',double[:,:]),
    ('P',double[:,:]),
    ('a',double[:,:]),
    ('C',double[:,:]), 
    ('C_avg',double[:]), 
    ('S',double[:,:]), 
    ('trans',double[:,:]),
    ('perm',double[:,:]),
    ('uni',double[:,:]),
    ('age',double[:,:]),
    ('init_a',double[:]),
    ('init_P',double[:])
]
@jitclass(simlist) # jit class with variables in parlist
class SimClass():
    def __init__(self):
        pass

# b. model class
class GP2002(ConsumptionSavingModel):
    
    # ConsumptionSavingModel has the following methods:
    # .save(self) # save self.par and self.sol
    # .load(self) # load par and sol
    # .__str__(self) # for printing

    def __init__(self,sol_gp=False,**kwargs): # called when created

        self.parlist = parlist # used when saving/loading
        self.par = ParClass()
        self.sollist = sollist # used when saving/loading
        self.sol = SolClass()
        self.simlist = simlist # used when saving/loading
        self.sim = SimClass()

        self.par.sol_gp = sol_gp # 0 if use my implementation

        self.setup(**kwargs)

    def setup(self,**kwargs): # setup using baseline parameters
        
        # a. baseline parameters

        # horizon
        self.par.age_min = 26
        self.par.age_max = 65
        self.par.Tr = self.par.age_max - self.par.age_min + 1 
        
        # preferences
        self.par.beta = 0.9598 
        self.par.rho = 0.5140 

        self.par.gamma0 = 0.0015
        self.par.gamma1 = 0.0710

        # returns 
        self.par.r = 0.03440
        self.par.credit = 0.0

        # grids
        self.par.Na = 300
        self.par.a_max = 15.0

        # shocks
        self.par.p = 0.00302
        self.par.mu = 0.0
        self.par.sigma_trans = 0.0440 #transitory income shock variance
        self.par.Ntrans = 10
        self.par.sigma_perm = 0.0212 #permanent income shock variance
        self.par.Nperm = 10

        # initial distribution
        self.par.mu_a_init = np.exp(-2.794)
        self.par.sigma_a_init = 1.784

        # misc
        self.par.do_print = False

        # simulation
        self.par.simN = 500000
        self.par.init_P = 18.690960
        np.random.seed(2018)

        # b. update baseline parameters using keywords 
        for key,val in kwargs.items():
            setattr(self.par,key,val) # like par.key = val
        
        # c. setup_grids
        self.setup_grids()
        
    def setup_grids(self): # create grids and shocks 
        
        # a. grids        
        self.par.grid_a = misc.nonlinspace(self.par.credit,self.par.a_max,self.par.Na,1.1)
        self.par.grid_m = misc.nonlinspace(self.par.credit+1.0e-6,self.par.a_max,self.par.Na,1.1)
        
        grid_age = [float(age) for age in range(self.par.age_min,self.par.age_max+1+1)]
        self.par.grid_age = np.array(grid_age)
        agep = np.empty((6,len(self.par.grid_age)))
        for i in range(6):
            agep[i,:] = self.par.grid_age**i

        # family shifter
        polF = np.array([0.0, 0.13964975, -0.0047742190, 8.5155210e-5, -7.9110880e-7, 2.9789550e-9]) # constant first (irrelevant)
        v = polF @ agep 
        self.par.v = np.exp(v[1:(self.par.Tr+1)] - v[0:(self.par.Tr)]) # This is (v[t+1]/v[t])^(1/rho)

        # permanent income growth
        polY = np.array([6.8013936, 0.3264338, -0.0148947, 0.000363424, -4.411685e-6, 2.056916e-8]) # constant first
        Ybar = np.exp(polY @ agep ) # matrix multiplication
        self.par.G = Ybar[1:(self.par.Tr+1)]/Ybar[0:(self.par.Tr)] # growth rate is shiftet forward, so g[t+1] is G[t] in code
        self.par.G[-1] = 1.0/self.par.v[-1]  # self.par.G[-1] = 1.0 
        
        # Set permanent income to the first predicted value:
        self.par.init_P = Ybar[0]/1000.0 # line 1651

        # b. random income shocks 
        self.par.perm, self.par.perm_w, self.par.trans, self.par.trans_w, self.par.Nshocks = misc.create_shocks_gp(self.par.sigma_perm,self.par.Nperm,self.par.sigma_trans,self.par.Ntrans,self.par.p,self.par.mu,mu_psi=0.0,mu_xi=0.0) # zero mean of log-shocks
        

    def solve(self): # solve the model
        # update some parameters and grids
        self.par.R = 1 + self.par.r # risk free gross interetst rate
        self.setup_grids()

        # a. allocate solution
        #shape = (self.par.Tr+1,self.par.Na)
        T = self.par.Tr
        shape = (T,self.par.Na)
        self.sol.c = np.empty(shape)
        self.sol.m = np.empty(shape)
        
        # b. backwards induction
        for t in reversed(range(T)):
            
            tic = time.time()
            
            # i. last working period
            if t == T-1:
                solve_bf_retirement(t,self.sol,self.par)

            # ii. all other periods
            else:
                solve_egm(t,self.sol,self.par) 
                

            # iii. print
            toc = time.time()
            if self.par.do_print:
                print(f' t = {t} solved in {toc-tic:.1f} secs')

    def draw_random(self):
        np.random.seed(2018) 
        shape = (self.par.Tr,self.par.simN)
        self.sim.trans = np.random.normal(size=shape)
        self.sim.perm  = np.random.normal(size=shape)

        self.sim.uni = np.random.uniform(0,1,size=shape)

        # c. initial wealth
        self.sim.init_a = np.random.normal(size=self.par.simN) 


    def simulate(self): # simulate the model
        
        # a. allocate
        sim_shape = (self.par.Tr,self.par.simN)
        self.sim.c = np.empty(sim_shape)
        self.sim.C = np.empty(sim_shape)
        self.sim.C_avg = np.empty(self.par.Tr)
        self.sim.m = np.empty(sim_shape)
        self.sim.a = np.empty(sim_shape)
        self.sim.Y = np.empty(sim_shape)
        self.sim.P = np.empty(sim_shape)
        self.sim.S = np.empty(sim_shape)
        
        self.sim.age = np.empty(sim_shape)

        self.sim.init_P = self.par.init_P*np.ones(self.par.simN) 

        tic = time.time()
        # d. call
        for t in range(self.par.Tr):
            simulate(t,self.par,self.sol,self.sim,self.sim.trans[t,:],self.sim.perm[t,:],self.sim.uni[t,:])
            
            # avg consumption without zero-shocks
            I = self.sim.Y[t]>0
            self.sim.C_avg[t] = np.exp(np.mean(np.log(self.sim.C[t,I])))

        toc = time.time()
        if self.par.do_print:
            print(f' t = {t} simulated in {toc-tic:.1f} secs')

# #######################################
# # 3. utility and transition functions #
# #######################################

@njit
def marg_u_func(c,par):
    return c**(-par.rho) 

@njit
def inv_marg_u_func(u,par):
    return u**(-1.0/par.rho) 

# #########################
# # 4. retirement period  #
# #########################
def solve_bf_retirement(t,sol,par):
    c_plus = (par.gamma0 + par.gamma1*par.R*(par.grid_a-par.credit))/par.G[t] # groth factor is 1, so does not matter
    dU = marg_u_func(par.G[t]*c_plus,par)
    sol.c[t,:] = inv_marg_u_func(par.beta*par.R*dU,par)
    sol.m[t,:] = par.grid_a + sol.c[t,:]    

@njit
def solve_egm(t,sol,par):

    c_next = np.zeros(par.Na+1)
    m_next = np.zeros(par.Na+1) + par.credit
    c_next[1:par.Na+1] = sol.c[t+1]
    m_next[1:par.Na+1] = sol.m[t+1]

    c_plus = np.empty(par.Na)

    # loop over shocks
    Eu = np.zeros(par.Na)
    for i in range(par.Nshocks):

        # next-period resources
        fac = par.G[t]*par.perm[i]*par.v[t]
        m_plus = (par.R/fac)*par.grid_a + par.trans[i]

        # interpolate next-period consumption
        linear_interp.interp_1d_vec(m_next,c_next,m_plus,c_plus)
        c_plus = np.fmax(1.0e-6 , c_plus )

        # expected marginal utility
        w = par.trans_w[i]*par.perm_w[i]
        Eu += w*marg_u_func(fac*c_plus,par) # In the original code they do not include all in fac as I do here

    # invert Euler equation
    sol.c[t] = inv_marg_u_func(par.beta*par.R*Eu,par) 
    sol.m[t] = par.grid_a + sol.c[t]

#################
# 7. simulation #
#################
@njit 
def simulate(t,par,sol,sim,trans,perm,uni):
    
    c_sol = np.zeros(par.Na+1)
    m_sol = np.zeros(par.Na+1) + par.credit
    c_sol[1:par.Na+1] = sol.c[t]
    m_sol[1:par.Na+1] = sol.m[t]

    c = sim.c[t]

    perm_shock = np.exp(par.sigma_perm*perm)
    trans_shock = np.exp(par.sigma_trans*trans)*(uni>par.p) + par.mu*(uni<=par.p)

    if t==0:
        sim.P[t] = sim.init_P*perm_shock
        initW = par.mu_a_init*np.exp(par.sigma_a_init*sim.init_a) 
        sim.m[t] = initW + trans_shock 
   
    else:
        sim.P[t] = par.G[t-1]*sim.P[t-1]*perm_shock
        fac = par.G[t-1]*perm_shock*par.v[t-1]
        sim.m[t] = par.R*sim.a[t-1]/fac + trans_shock 

    # Income 
    sim.Y[t] = sim.P[t]*trans_shock

    # interpolate optimal consumption
    linear_interp.interp_1d_vec(m_sol,c_sol,sim.m[t],c)
    sim.C[t] = sim.c[t]*sim.P[t]

    # end-of-period wealth and saving
    sim.a[t] = sim.m[t] - c
    
    if t>0:
        sim.S[t] = (sim.a[t]*sim.P[t] - sim.a[t-1]*sim.P[t-1]) # do not divide with R because I use A and not W


def load_data():    
    filename = "income.txt" # placed in the current folder at this point
    income = np.loadtxt(filename,delimiter=",")

    filename = "sample_moments.txt" # placed in the current folder at this point
    consumption = np.loadtxt(filename,delimiter=",")

    filename = "weight.txt" # placed in the current folder at this point
    weight = np.loadtxt(filename,delimiter=",")

    return (consumption/1000.0,income/1000.0,weight)

# SHOULD BE DELETED
def saving_decomp(par_vec=[],par_list=(),sol_gp=False,do_higher_r=False):

    model = GP2002(sol_gp=sol_gp,do_print=False)
    model_lc = GP2002(sol_gp=sol_gp,do_print=False)
    
    if do_higher_r:
        model.par.r = model.par.r*1.05
        model_lc.par.r = model_lc.par.r*1.05

    # update potential parameters
    num_par = len(par_vec)
    for p in range(num_par):
        setattr(model.par,par_list[p],par_vec[p]) # like par.key = val
        setattr(model_lc.par,par_list[p],par_vec[p]) # like par.key = val
        print(par_list[p],par_vec[p],end=" ")
    print("")
    # solve baseline model
    model.solve() 

    model.draw_random()
    model.simulate()

    print(model.par.rho)

    # simulate from alternative model
    model_lc.par.sigma_perm = 0.0
    model_lc.par.sigma_trans = 0.0
    model_lc.par.p = 0.0
    model_lc.par.credit = -5.0
    # retirement rule
    NT = 88-65
    beta = 1.0/(1.0344) # estimated in GP
    beta_rho = beta**(1.0/model_lc.par.rho)
    R_rho = (1+model_lc.par.r)**(1.0/model_lc.par.rho - 1.0)
    nom = 1.0 - beta_rho*R_rho
    denom = 1.0 - (beta_rho*R_rho)**NT
    model_lc.gamma1 = nom/denom

    #print(model_lc.gamma1)
    # estimates in fig 1
    # model_lc.par.gamma0 = 0.594
    # model_lc.par.gamma1 = 0.077

    model_lc.solve() 

    model_lc.draw_random()
    model_lc.simulate()
    
    # saving
    S    = np.mean(model.sim.S[1:-1,:],axis=1)
    S_lc = np.mean(model_lc.sim.S[1:-1,:],axis=1)
    S_b  = S - S_lc

    # wealth
    W = np.mean(model.sim.a*model.sim.P,axis=1)
    W_lc = np.mean(model_lc.sim.a*model_lc.sim.P,axis=1)
    W_b = W - W_lc

    return (S,S_lc,S_b,W,W_lc,W_b)

# SHOULD BE DELETED
def saving_decomp_wrap(par_vec=[],par_list=(),sol_gp=False,do_higher_r=False):
    S,S_lc,S_b,W,W_lc,W_b = saving_decomp(par_vec,par_list,sol_gp,do_higher_r=do_higher_r)

    ages = [age for age in range(27,65)]
    diff = np.empty(len(ages))
    for a,age in enumerate(ages):
        diff[a] = S_b[age-27] - S_lc[age-27]

    return diff

def saving_decomposition_wrap(par_vec=[],par_list=(),par_vec_add=[],par_list_add=()):
    par_vec_tot =  np.concatenate( (par_vec , par_vec_add), axis=0)
    par_list_tot = (par_list + par_list_add)
    (S,S_lc,S_b,W,W_lc,W_b,diff) = saving_decomposition(par_vec_tot,par_list_tot)
    return diff

def saving_decomposition(par_vec=[],par_list=[]):

    model = GP2002()
    model_lc = GP2002()
 
    # update potential parameters
    num_par = len(par_vec)
    for p in range(num_par):
        setattr(model.par,par_list[p],par_vec[p]) # like par.key = val
        setattr(model_lc.par,par_list[p],par_vec[p]) # like par.key = val
        print(par_list[p],par_vec[p],end=" ")
    print("")

    # solve baseline model
    model.solve() 
    model.draw_random()
    model.simulate()

    # simulate from alternative model
    model_lc.par.sigma_perm = 0.0
    model_lc.par.sigma_trans = 0.0
    model_lc.par.p = 0.0
    model_lc.par.credit = -5.0

    # retirement rule
    NT = 88-65
    beta = 1.0/(1.0344) # estimated in GP
    beta_rho = beta**(1.0/model_lc.par.rho)
    R_rho = (1+model_lc.par.r)**(1.0/model_lc.par.rho - 1.0)
    nom = 1.0 - beta_rho*R_rho
    denom = 1.0 - (beta_rho*R_rho)**NT
    model_lc.gamma1 = nom/denom

    model_lc.solve() 
    model_lc.draw_random()
    model_lc.simulate()
    
    # saving
    S    = np.mean(model.sim.S[1:-1,:],axis=1)
    S_lc = np.mean(model_lc.sim.S[1:-1,:],axis=1)
    S_b  = S - S_lc

    # wealth
    W = np.mean(model.sim.a*model.sim.P,axis=1)
    W_lc = np.mean(model_lc.sim.a*model_lc.sim.P,axis=1)
    W_b = W - W_lc

    ages = [age for age in range(27,65)]
    diff = np.empty(len(ages))
    for a,age in enumerate(ages):
        diff[a] = S_b[age-27] - S_lc[age-27]

    return (S,S_lc,S_b,W,W_lc,W_b,diff)    

def num_grad(obj_fun,theta,dim_fun,step=1.0e-5,*args):
    num_par = len(theta)
    grad = np.empty((dim_fun,num_par))
    for p in range(num_par):
        theta_now = theta[:] 

        step_now  = np.zeros(num_par)
        step_now[p] = np.fmax(step,step*theta_now[p])

        forward  = obj_fun(theta_now + step_now,*args)
        backward = obj_fun(theta_now - step_now,*args)

        grad[:,p] = (forward - backward)/(2.0*step_now[p])

    return grad


if __name__ == "__main__":

    lw = 3
    fs = 17
        
    # solve model
    model = GP2002(do_print=False)
    model.solve() 

    # simulate data from model
    model.draw_random()
    model.simulate()

    # load consumption and income from data and simulate
    consumption,income,weight = load_data()
    
    # plot model implications
    plot.fig1(model)
    plot.fig5(model,consumption,income)
    plot.fig7(model)
    plot.cali(model)
       