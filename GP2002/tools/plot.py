import numpy as np
from numba import njit, jitclass, prange, boolean, int32, double
import seaborn as sns 

import matplotlib.pyplot as plt
import solve_GP2002 as gp

def fig1(model,add_str=''):
    line_spec = ('-','--','-','-.','-',':','-.')
    lw = 3
    fs = 17

    sol_gp_str = ''
    if model.par.sol_gp:
        sol_gp_str = '_gp'
        

    ages = [26, 35, 45, 55, 65]
    legend = ('26', '35', '45', '55', '65')

    x = np.empty((len(ages),model.par.Na+1))
    y = np.empty((len(ages),model.par.Na+1))
    for i,age in enumerate(ages):
        t = age - model.par.age_min 
        x[i,0] = 0.0
        y[i,0] = 0.0
        x[i,1:] = model.sol.m[t,:]
        y[i,1:] = model.sol.c[t,:]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    for i,age in enumerate(ages):
        ax.plot(x[i,:],y[i,:],linewidth=lw,linestyle=line_spec[i])

    ax.set_xlim(0,3)
    #ax.set_ylim(0,1.31)
    ax.set_ylim(0,1.4)

    ax.set_xlabel('normalized resources',fontsize=fs)
    ax.set_ylabel('normalized consumption',fontsize=fs)

    ax.legend(legend,fontsize=fs)
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=fs)

    plt.savefig('output/replicated_fig1' +sol_gp_str + add_str + '.pdf',bbox_inches="tight")

def fig5(model,consumption,income,add_str=''):
    line_spec = ('-','--','-','-.','-',':','-.')
    lw = 3
    fs = 17

    sol_gp_str = ''
    if model.par.sol_gp:
        sol_gp_str = '_gp'

    age_grid = [age for age in range(model.par.age_min,model.par.age_max+1)]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    ax.plot(age_grid,model.sim.C_avg)
    ax.plot(age_grid,consumption,marker='o')
    ax.plot(age_grid,income,linewidth=lw)

    ax.set_xlabel('Age',fontsize=fs)
    ax.set_ylabel('Thousands of 1987 dollars',fontsize=fs)

    ax.legend(('Fitted consumption','Raw consumption','Income'),fontsize=fs)
    ax.grid(True)

    ax.tick_params(axis='both', which='major', labelsize=fs)
    plt.xlim(25,66)

    plt.savefig('output/replicated_fig5' +sol_gp_str + add_str + '.pdf',bbox_inches="tight")

def fig7_higher_r(par_vec=[],par_list=(),add_str=''):
    line_spec = ('-','--','-','-.','-',':','-.')
    lw = 3
    fs = 17

    age_grid = [age for age in range(25,65)]
    # Figure 7:
    
    S,S_lc,S_b,W,W_lc,W_b,H = gp.saving_decomposition(par_vec,par_list)
    par_vec_high = par_vec.copy()
    #par_list_high = (par_list + 'r')
    for p in range(len(par_vec)):
        if par_list[p]=="r":
            par_vec_high[p] = par_vec[p]*1.05 
    S,S_lc_high,S_b_high,W,W_lc,W_b,H = gp.saving_decomposition(par_vec_high,par_list)
    
    age = 30
    diff = S_b[age-27] - S_lc[age-27]
    diff_high = S_b_high[age-27] - S_lc_high[age-27]

    change_pct = 100.0*(diff_high - diff)/diff
    #print(diff,diff_high,change_pct)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.plot(age_grid[1:-1],S_lc,linewidth=lw,linestyle=line_spec[0])
    ax.plot(age_grid[1:-1],S_b,linewidth=lw,linestyle=line_spec[1])

    ax.plot(age_grid[1:-1],S_lc_high,linewidth=lw,linestyle=line_spec[0])
    ax.plot(age_grid[1:-1],S_b_high,linewidth=lw,linestyle=line_spec[1])

    ax.set_xlabel('Age',fontsize=fs)
    ax.set_ylabel('Thousands of 1987 dollars',fontsize=fs)

    ax.legend(('Life Cycle Savings','Buffer Savings'),fontsize=fs)
    ax.grid(True)

    ax.tick_params(axis='both', which='major', labelsize=fs)
    props = dict(facecolor='black',arrowstyle="->",connectionstyle="arc3")
    str_plot = f'5% higher r\n{change_pct:2.2f}% change'
    ax.annotate(str_plot,xy=(30,5) , xytext=(26,1), arrowprops=props)
    ax.annotate(str_plot,xy=(30,S_lc_high[4]) , xytext=(26,1), arrowprops=props)

    plt.xlim(25,66)
    
    plt.savefig('output/replicated_fig7' + add_str + '_highR' + '.pdf',bbox_inches="tight")

    


def fig7(par_vec=[],par_list=(),add_str=''):
    line_spec = ('-','--','-','-.','-',':','-.')
    lw = 3
    fs = 17

    # Figure 7:
    age_grid = [age for age in range(25,65)]
    (S,S_lc,S_b,W,W_lc,W_b,diff) = gp.saving_decomposition(par_vec,par_list)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.plot(age_grid[1:-1],S_lc,linewidth=lw,linestyle=line_spec[0])
    ax.plot(age_grid[1:-1],S_b,linewidth=lw,linestyle=line_spec[1])

    ax.set_xlabel('Age',fontsize=fs)
    ax.set_ylabel('Thousands of 1987 dollars',fontsize=fs)

    ax.legend(('Life Cycle Savings','Buffer Savings'),fontsize=fs)
    ax.grid(True)

    ax.tick_params(axis='both', which='major', labelsize=fs)
    plt.xlim(25,66)

    plt.savefig('output/replicated_fig7' + add_str + '.pdf',bbox_inches="tight")


    # bottom panel:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.plot(age_grid,W_lc,linewidth=lw,linestyle=line_spec[0])
    ax.plot(age_grid,W_b,linewidth=lw,linestyle=line_spec[1])
    ax.plot(age_grid,W,linewidth=lw/2,linestyle=line_spec[0],color='black')

    ax.set_xlabel('Age',fontsize=fs)
    ax.set_ylabel('Thousands of 1987 dollars',fontsize=fs)

    ax.legend(('Life Cycle Wealth','Buffer Wealth','Total wealth'),fontsize=fs)
    ax.grid(True)

    ax.tick_params(axis='both', which='major', labelsize=fs)
    plt.xlim(25,66)

    plt.savefig('output/replicated_fig7_bottom' + add_str + '.pdf',bbox_inches="tight")


def fig7_old(model,par_vec=[],par_list=(),add_str='',do_higher_r=False):
    line_spec = ('-','--','-','-.','-',':','-.')
    lw = 3
    fs = 17

    sol_gp_str = ''
    if model.par.sol_gp:
        sol_gp_str = '_gp'

    age_grid = [age for age in range(model.par.age_min,model.par.age_max+1)]

    # Figure 7:
    S,S_lc,S_b,W,W_lc,W_b = gp.saving_decomp(par_vec=par_vec,par_list=par_list,sol_gp=model.par.sol_gp,do_higher_r=do_higher_r)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.plot(age_grid[1:-1],S_lc,linewidth=lw,linestyle=line_spec[0])
    ax.plot(age_grid[1:-1],S_b,linewidth=lw,linestyle=line_spec[1])

    ax.set_xlabel('Age',fontsize=fs)
    ax.set_ylabel('Thousands of 1987 dollars',fontsize=fs)

    ax.legend(('Life Cycle Savings','Buffer Savings'),fontsize=fs)
    ax.grid(True)

    ax.tick_params(axis='both', which='major', labelsize=fs)
    plt.xlim(25,66)

    plt.savefig('output/replicated_fig7' +sol_gp_str + add_str + '.pdf',bbox_inches="tight")


    # bottom panel:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.plot(age_grid,W_lc,linewidth=lw,linestyle=line_spec[0])
    ax.plot(age_grid,W_b,linewidth=lw,linestyle=line_spec[1])
    ax.plot(age_grid,W,linewidth=lw/2,linestyle=line_spec[0],color='black')

    ax.set_xlabel('Age',fontsize=fs)
    ax.set_ylabel('Thousands of 1987 dollars',fontsize=fs)

    ax.legend(('Life Cycle Wealth','Buffer Wealth','Total wealth'),fontsize=fs)
    ax.grid(True)

    ax.tick_params(axis='both', which='major', labelsize=fs)
    plt.xlim(25,66)

    plt.savefig('output/replicated_fig7_bottom' + sol_gp_str + add_str + '.pdf',bbox_inches="tight")

def cali(model,add_str=''):
    line_spec = ('-','--','-','-.','-',':','-.')
    lw = 3
    fs = 17

    sol_gp_str = ''
    if model.par.sol_gp:
        sol_gp_str = '_gp'

    age_grid = [age for age in range(model.par.age_min,model.par.age_max+1)]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.plot(age_grid,model.par.G,linewidth=lw,linestyle=line_spec[0])
    ax.plot(age_grid,model.par.v**model.par.rho,linewidth=lw,linestyle=line_spec[1])

    ax.set_xlabel('Age',fontsize=fs)
    ax.set_ylabel('Income growth and family shifter',fontsize=fs)

    ax.legend(('Income growth, $G_{t+1}$','Familiy shifter, $v_{t+1}/v_{t}$'),fontsize=fs)
    ax.grid(True)

    ax.tick_params(axis='both', which='major', labelsize=fs)
    plt.xlim(25,66)

    plt.savefig('output/replicated_cali' +sol_gp_str + add_str + '.pdf',bbox_inches="tight")


def plot_figure(x,y,path_name,xlabel='',ylabel='',legend='', fs=17,lw=3):

    line_spec = ('-','--','-.',':')

    dim_y = 1
    if y.ndim>1:
        dim_y = len(y[:,0])

    dim_x = 1
    if x.ndim>1:
        dim_x = len(x[:,0])

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    for i in range(dim_y):
        if dim_x==1:
            x_val = x
        else:
            x_val = x[i,:]

        ax.plot(x_val,y[i,:],linewidth=lw,linestyle=line_spec[i])

    ax.set_xlabel(xlabel,fontsize=fs)
    ax.set_ylabel(ylabel,fontsize=fs)

    ax.legend(legend,fontsize=fs)
    ax.grid(True)

    ax.tick_params(axis='both', which='major', labelsize=fs)
    plt.xlim(min(x[:]),max(x[:]))

    plt.savefig(path_name+'.eps',bbox_inches="tight")
    plt.savefig(path_name+'.pdf',bbox_inches="tight")

    plt.show()

# SENSITIVITY
def sens_fig_tab(sens,sense,theta,est_par_tex,fixed_par_tex,sol_gp=False,add_str='',save=True):
    sol_gp_str = ''
    if sol_gp:
        sol_gp_str = '_gp'

    if save:
        num_theta = len(theta)
        num_gamma = len(fixed_par_tex)
        name = 'gp2002' + sol_gp_str + add_str 
        with open('output/' + name + '.tex',"w") as file:
            file.write("\\begin{tabular}{l*{%d}{c}} \n" %num_theta)
            file.write("\\multicolumn{%d}{c}{} \\\\ \\toprule \n" %(1+num_theta))
            for p in range(num_theta):
                file.write("& %s " %est_par_tex[p])
            file.write("\\\\ \\cmidrule(lr){2-%d} \n " %(num_theta+1))

            # sensitivity
            for g in range(num_gamma):
                file.write(" %s " %fixed_par_tex[g])
                for p in range(num_theta):
                    file.write("& %2.3f " %sens[p,g])
                file.write("\\\\ \n ")

            # estimates
            file.write("\\midrule Estimates")
            for p in range(num_theta):
                file.write("& %2.3f " %theta[p])

            file.write("\\\\ \\bottomrule \n \\end{tabular}" )

    # heatmap plot of elasticities
    fs = 15
    sns.set(rc={'text.usetex' : True})
    cmap = sns.diverging_palette(10, 220, sep=10, n=100)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax = sns.heatmap(sense,annot=True,fmt="2.2f",annot_kws={"size": fs},xticklabels=fixed_par_tex,yticklabels=est_par_tex,center=0,linewidth=.5,cmap=cmap)
    plt.yticks(rotation=0) 
    ax.tick_params(axis='both', which='major', labelsize=20)
    if save: plt.savefig('output/sense_' + name + '.pdf',bbox_inches="tight")