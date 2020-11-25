import math
import numpy as np

def nonlinspace(x_min, x_max, n, phi):
    """ like np.linspace between with unequal spacing
    phi = 1 -> eqaul spacing
    phi up -> more points closer to minimum
    """

    assert x_max > x_min
    assert n >= 2
    assert phi >= 1
 
    # 1. recursion
    y = np.empty(n)
 
    y[0] = x_min
    for i in range(1, n):
        y[i] = y[i-1] + (x_max-y[i-1]) / (n-i)**phi
    
    # 3. assert increaing
    assert np.all(np.diff(y) > 0)
 
    return y
 
def gauss_hermite(n):

    # a. calculations
    i = np.arange(1,n)
    a = np.sqrt(i/2)
    CM = np.diag(a,1) + np.diag(a,-1)
    L,V = np.linalg.eig(CM)
    I = L.argsort()
    V = V[:,I].T

    # b. nodes and weights
    x = L[I]
    w = np.sqrt(math.pi)*V[:,0]**2

    return x,w

def log_normal_gauss_hermite(sigma, n,mu=None):

    mu = mu or -0.5*sigma**2

    # a. GaussHermite
    x,w = gauss_hermite(n)

    # b. log-normality
    x = np.exp(x*np.sqrt(2)*sigma+mu)
    w = w/np.sqrt(math.pi)

    return x,w

def create_shocks(sigma_psi,Npsi,sigma_xi,Nxi,pi,mu,mu_psi=None,mu_xi=None):
    
    # a. gauss hermite
    psi, psi_w = log_normal_gauss_hermite(sigma_psi, Npsi,mu_psi)
    xi, xi_w = log_normal_gauss_hermite(sigma_xi, Nxi,mu_xi)
 
    # b. add low inncome shock
    if pi > 0:
         
        # a. weights
        xi_w *= (1.0-pi)
        xi_w = np.insert(xi_w,0,pi)

        # b. values
        xi = (xi-mu*pi)/(1.0-pi)
        xi = np.insert(xi,0,mu)

    
    # c. tensor product
    psi,xi = np.meshgrid(psi,xi,indexing='ij')
    psi_w,xi_w = np.meshgrid(psi_w,xi_w,indexing='ij')

    return psi.ravel(), psi_w.ravel(), xi.ravel(), xi_w.ravel(), psi.size

def create_shocks_gp(sigma_psi,Npsi,sigma_xi,Nxi,pi,mu,mu_psi=None,mu_xi=None):
    
    # a. gauss hermite
    psi, psi_w = log_normal_gauss_hermite(sigma_psi, Npsi,mu_psi)
    xi, xi_w = log_normal_gauss_hermite(sigma_xi, Nxi,mu_xi)
 
    # b. add low inncome shock
    if pi > 0:
         
        # a. weights
        xi_w *= (1.0-pi)
        xi_w = np.insert(xi_w,0,pi)

        # b. values
        xi = (xi-mu*pi) #/(1.0-pi)
        xi = np.insert(xi,0,mu)

    
    # c. tensor product
    psi,xi = np.meshgrid(psi,xi,indexing='ij')
    psi_w,xi_w = np.meshgrid(psi_w,xi_w,indexing='ij')

    return psi.ravel(), psi_w.ravel(), xi.ravel(), xi_w.ravel(), psi.size
