"""
Class and functions for multi-linear interpolation
"""

import numpy as np
from numba import njit,prange, boolean, int64, double, void
from numba.experimental import jitclass

info = [
    ('dimx',int64),
    ('x',double[:]),
    ('dimy',int64),
    ('y',double[:]),
    ('Nx',int64[:]),    
    ('facs',int64[:]),
    ('xoffset',int64[:]),    
    ('yoffset',int64[:]),    
    ('ncube',int64),
    ('add',int64[:]),
    ('pos_left',int64[:]),
    ('denom_last',double),
    ('nomi_last',double[:]),
    ('index_last',int64[:]), 
    ('save_last',boolean)
]
@jitclass(info)
class InterpolatorClass():
    """
    N-dimensional interpolator class 
    """

    # setup
    def __init__(self,grids,values):
        
        self.save_last = True
        
        # a. grids
        self.x = np.hstack(grids)
        self.dimx = np.int64(len(grids))

        # b. values
        self.y = np.vstack(values).ravel()
        self.dimy = np.int64(len(values))
        
        # c. grid information
        self.Nx = np.zeros(self.dimx,dtype=np.int64)
        for i in range(self.dimx):
            self.Nx[i] = grids[i].size
            
        self.xoffset = np.zeros(self.dimx,dtype=np.int64)
        for i in range(1,self.dimx):
            self.xoffset[i] = self.xoffset[i-1] + self.Nx[i-1]
            
        self.facs = np.zeros(self.dimx,dtype=np.int64)
        for d in range(self.dimx-1,-1,-1):
            self.facs[d] = 1
            for j in range(d,self.dimx-1):
                self.facs[d] *= self.Nx[j+1]
             
        self.yoffset = np.zeros(self.dimy,dtype=np.int64)
        for i in range(1,self.dimy):
            self.yoffset[i] = self.yoffset[i-1] + values[0].size

        # d. misc
        self.add = np.zeros((2**self.dimx*self.dimx),dtype=np.int64)
        self.ncube = 2**self.dimx
        for i in range(1,self.ncube):

            now = self.add[i*self.dimx:]
            prev = self.add[(i-1)*self.dimx:]
            now[:self.dimx] = prev[:self.dimx]

            for j in range(self.dimx):
                if prev[j] == 0:
                    now[j] = 1
                    if j > 0:
                        now[0:j] = 0
                    break
        
        self.pos_left = np.zeros(self.dimx,dtype=np.int64)
        
        # e. last
        self.nomi_last = np.zeros(self.ncube)
        self.index_last = np.zeros(self.ncube,dtype=np.int64)
        
@njit(int64(int64,int64,double[:],double))
def binary_search(imin,Nx,x,xi):
        
    # a. checks
    if xi <= x[0]:
        return 0
    elif xi >= x[Nx-2]:
        return Nx-2
    
    # b. binary search
    half = Nx//2
    while half:
        imid = imin + half
        if x[imid] <= xi:
            imin = imid
        Nx -= half
        half = Nx//2
        
    return imin

@njit
def evaluate(interp,xi,ydim=0):
    """ evaluate interpolator class """

    # a. search in each dimension
    for d in range(interp.dimx):
        x = interp.x[interp.xoffset[d]:]
        interp.pos_left[d] = binary_search(0,interp.Nx[d],x,xi[d]) 
            
    # b. denominator
    denom = 1
    for d in range(interp.dimx):
        x = interp.x[interp.xoffset[d]:]
        denom *= (x[interp.pos_left[d]+1]-x[interp.pos_left[d]])
        if interp.save_last and d == interp.dimx-2:
            interp.denom_last = denom
    
    # c. nominator
    nom = 0
    for i in range(interp.ncube):
    
        nomi = 1
        index = 0
        for d in range(interp.dimx):
            
            x = interp.x[interp.xoffset[d]:]
            if interp.add[i*interp.dimx+d] == 1:
                nomi *= (xi[d]-x[interp.pos_left[d]])
            else:
                nomi *= (x[interp.pos_left[d]+1]-xi[d])
                
            index += (interp.pos_left[d] + interp.add[i*interp.dimx+d])*interp.facs[d]
            
            if interp.save_last and d == interp.dimx-2:
                interp.nomi_last[i] = nomi
                interp.index_last[i] = index
            
        nom += nomi*interp.y[interp.yoffset[ydim] + index]
    
    return nom/denom

@njit
def evaluate_only_last(interp,xi_last,ydim=0):
    """ evaluate interpolator class when only the last element has
    changed since the last evaluation
    """

    # a. search
    d = interp.dimx-1
    x = interp.x[interp.xoffset[d]:]
    pos_left = binary_search(0,interp.Nx[d],x,xi_last)

    # b. denominator
    denom = interp.denom_last*(x[pos_left+1]-x[pos_left])
    
    # c. nominator
    nom = 0
    a = xi_last-x[pos_left]
    b = x[pos_left+1]-xi_last
    facs = interp.facs[d]
    for i in range(interp.ncube):

        add = interp.add[i*interp.dimx+d]
        if add == 1:
            nomi = interp.nomi_last[i]*a
        else:
            nomi = interp.nomi_last[i]*b

        index = interp.index_last[i]+(pos_left+add)*facs
            
        nom += nomi*interp.y[interp.yoffset[ydim] + index]
    
    return nom/denom

@njit
def evaluate_only_last_vec(interp,xi_last,yi):
    """ evaluate interpolator class for a monotone vector of points, when only the last element has
    changed since the last evaluation
    """

    pos_left = 0
    nom = np.empty(interp.dimy)
    for ixi in range(xi_last.size):

        # a. search
        d = interp.dimx-1
        x = interp.x[interp.xoffset[d]:]
        while xi_last[ixi] >= x[pos_left+1] and pos_left < interp.Nx[d]-2:
            pos_left += 1

        # b. denominator
        denom = interp.denom_last*(x[pos_left+1]-x[pos_left])
    
        # c. nominator
        for k in range(interp.dimy):
            nom[k] = 0
        a = xi_last[ixi]-x[pos_left]
        b = x[pos_left+1]-xi_last[ixi]
        facs = interp.facs[d]
        for i in range(interp.ncube):

            add = interp.add[i*interp.dimx+d]
            if add == 1:
                nomi = interp.nomi_last[i]*a
            else:
                nomi = interp.nomi_last[i]*b

            index = interp.index_last[i]+(pos_left+add)*facs
                
            for k in range(interp.dimy):
                nom[k] += nomi*interp.y[interp.yoffset[k] + index]
        
        inv_denom = 1/denom
        for k in range(interp.dimy):
            yi[ixi*interp.dimy + k] = nom[k]*inv_denom

######
# 1D #
######

@njit(double(double[:],double[:],double))
def interp_1d(grid,value,xi):
    """ raw 1D interpolation """

    # a. search
    ix = binary_search(0,grid.size,grid,xi)
    
    # b. relative positive
    rel_x = (xi - grid[ix])/(grid[ix+1]-grid[ix])
    
    # c. interpolate
    return value[ix] + rel_x * (value[ix+1]-value[ix])

@njit(void(double[:],double[:],double[:],double[:],boolean))
def _interp_1d_vec(grid,value,xi,yi,monotone):
    """ raw 1D interpolation for a monotone vector """

    ix = 0
    for ixi in range(xi.size):

        if not monotone:
            ix = 0

        # a. search
        while xi[ixi] >= grid[ix+1] and ix < grid.size-2:
            ix += 1

        # b. relative positive
        rel_x = (xi[ixi] - grid[ix])/(grid[ix+1]-grid[ix])
    
        # c. interpolate
        yi[ixi] = value[ix] + rel_x * (value[ix+1]-value[ix])

@njit(void(double[:],double[:],double[:],double[:]))
def interp_1d_vec(grid,value,xi,yi):
    _interp_1d_vec(grid,value,xi,yi,False)

@njit(void(double[:],double[:],double[:],double[:]))
def interp_1d_mon_vec(grid,value,xi,yi):
    _interp_1d_vec(grid,value,xi,yi,True)

######
# 2D #
######

@njit(double(double[:],double[:],double[:,:],double,double))
def interp_2d(grid1,grid2,value,xi1,xi2):
    """ raw 2D interpolation """

    # a. search in each dimension
    ix1 = binary_search(0,grid1.size,grid1,xi1)
    ix2 = binary_search(0,grid2.size,grid2,xi2)
    
    # b. relative positive
    rel_x1 = (xi1 - grid1[ix1])/(grid1[ix1+1]-grid1[ix1])
    rel_x2 = (xi2 - grid2[ix2])/(grid2[ix2+1]-grid2[ix2])
    
    # c. interpolate over inner dimension 
    left = value[ix1,ix2] + rel_x2 * (value[ix1,ix2+1]-value[ix1,ix2])
    right = value[ix1+1,ix2] + rel_x2 * (value[ix1+1,ix2+1]-value[ix1+1,ix2])

    # d. interpolate over outer dimension
    return left + rel_x1*(right-left)


@njit(double(int64,double,double[:],double[:,:],double))
def interp_2d_only_last(ix1,rel_x1,grid2,value,xi2):
    """ 2D interpolation where the index and relative distance is
    known for the first dimension"""

    # a. search
    ix2 = binary_search(0,grid2.size,grid2,xi2)
    
    # b. relative position
    rel_x2 = (xi2 - grid2[ix2])/(grid2[ix2+1]-grid2[ix2])
    
     # c. interpolate over inner dimension 
    left = value[ix1,ix2] + rel_x2 * (value[ix1,ix2+1]-value[ix1,ix2])
    right = value[ix1+1,ix2] + rel_x2 * (value[ix1+1,ix2+1]-value[ix1+1,ix2])

    # d. interpolate over outer dimension
    return left + rel_x1*(right-left)

@njit(void(int64,double,double[:],double[:,:],double[:],double[:]))
def interp_2d_only_last_vec(ix1,rel_x1,grid2,value,xi2,yi):
    """ 2D interpolation where the index and relative distance is
    known for the first dimension for a monotone vector"""

    ix2 = 0
    for ixi in range(xi2.size):

        # a. search
        while xi2[ixi] >= grid2[ix2+1] and ix2 < grid2.size-2:
            ix2 += 1

        # b. relative position
        rel_x2 = (xi2[ixi] - grid2[ix2])/(grid2[ix2+1]-grid2[ix2])
        
        # c. interpolate over inner dimension 
        left = value[ix1,ix2] + rel_x2 * (value[ix1,ix2+1]-value[ix1,ix2])
        right = value[ix1+1,ix2] + rel_x2 * (value[ix1+1,ix2+1]-value[ix1+1,ix2])

        # d. interpolate over outer dimension
        yi[ixi] = left + rel_x1*(right-left)

######
# 3D #
######

@njit(double(double[:],double[:],double[:],double[:,:,:],double,double,double))
def interp_3d(grid1,grid2,grid3,value,xi1,xi2,xi3):
    """ raw 3D interpolation """

    # a. search in each dimension
    j1 = binary_search(0,grid1.size,grid1,xi1)
    j2 = binary_search(0,grid2.size,grid2,xi2)
    j3 = binary_search(0,grid3.size,grid3,xi3)
    
    # b. interpolation
    denom = (grid1[j1+1]-grid1[j1])*(grid2[j2+1]-grid2[j2])*(grid3[j3+1]-grid3[j3])
    nom = 0
    for k1 in range(2):
        nom_1 = (grid1[j1+1]-xi1) if k1 == 0 else (xi1-grid1[j1])
        for k2 in range(2):
            nom_2 = (grid2[j2+1]-xi2) if k2 == 0 else (xi2-grid2[j2])           
            for k3 in range(2):
                nom_3 = (grid3[j3+1]-xi3) if k3 == 0 else (xi3-grid3[j3])                   
                nom += nom_1*nom_2*nom_3*value[j1+k1,j2+k2,j3+k3]

    return nom/denom    