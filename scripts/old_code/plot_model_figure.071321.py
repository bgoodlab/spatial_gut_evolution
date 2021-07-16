

from matplotlib import pyplot as plt
import numpy as np

import pylab


import sys

import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from numpy.random import randint, shuffle, poisson, binomial, choice, hypergeometric
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe

from scipy.integrate import quad, trapz
from math import log10


max_x = 80

# Create figure objects

pylab.figure(1,figsize=(7, 2.2))
fig = pylab.gcf()

mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.frameon']  = False
mpl.rcParams['legend.fontsize']  = 'small'

# Create grids for plots
outer_grid = gridspec.GridSpec(1,3,width_ratios=[1.5,1.6,1],wspace=0.6)

lam_grid = gridspec.GridSpecFromSubplotSpec(2,2,height_ratios=[1,1],width_ratios=[0.8,0.2],hspace=0.4, subplot_spec=outer_grid[0])

schem_nut_axis = plt.Subplot(fig, lam_grid[0,0])
fig.add_subplot(schem_nut_axis)
schem_nut_axis.set_ylabel('$\\lambda(c)$')
schem_nut_axis.set_xticks([])
#schem_nut_axis.set_xticklabels(["K"])
schem_nut_axis.set_xlabel('Nutrient conc, c') #, labelpad=0)
schem_nut_axis.set_ylim(-0.02,1.5)
schem_nut_axis.set_xlim([0,1])
schem_nut_axis.set_yticks([])
#schem_nut_axis.set_title('Minimal model',fontsize=8)
schem_nut_axis.set_title('Growth rate',fontsize=7)
ttl = schem_nut_axis.title
ttl.set_position([.5, 0.95])


schem_pos_axis = plt.Subplot(fig, lam_grid[1,0])
fig.add_subplot(schem_pos_axis)
schem_pos_axis.set_ylabel('$\\lambda(x)$')
schem_pos_axis.set_xticks([])
#schem_pos_axis.set_xticklabels("$\\ell$")
schem_pos_axis.set_xlabel('Position, $x$') #,labelpad=2.0)
schem_pos_axis.set_ylim(-0.02,1.5)
schem_pos_axis.set_yticks([])
schem_pos_axis.set_xlim([0,1])

pug_grid = gridspec.GridSpecFromSubplotSpec(2,1,height_ratios=[1,1],hspace=0.4, subplot_spec=outer_grid[1])


pug_qwm_axis = plt.Subplot(fig, pug_grid[0])
fig.add_subplot(pug_qwm_axis)
pug_qwm_axis.set_title('Quasi Well-Mixed Regime',fontsize=7)
ttl = pug_qwm_axis.title
ttl.set_position([.5, 0.95])

pug_spa_axis = plt.Subplot(fig, pug_grid[1])
fig.add_subplot(pug_spa_axis)
pug_spa_axis.set_xlabel('Position, $x$ (a.u.)')
pug_spa_axis.set_title('Spatial Regime',fontsize=7)
ttl = pug_spa_axis.title
ttl.set_position([.5, 0.95])

param_grid = gridspec.GridSpecFromSubplotSpec(2,1,height_ratios=[1,1],hspace=0.2, subplot_spec=outer_grid[2])


lame_axis = plt.Subplot(fig, param_grid[0])
fig.add_subplot(lame_axis)
lame_axis.set_ylabel("$\\lambda_e/\\lambda$")
#lame_axis.set_title('Effective params',fontsize=8)

ne_axis = plt.Subplot(fig, param_grid[1])
fig.add_subplot(ne_axis)
ne_axis.set_xlabel('$\\alpha-1$')
#ne_axis.set_ylabel('$\\frac{N_e}{N} \cdot \\frac{vL}{D}$',fontsize=8)
ne_axis.set_ylabel('Scaled $N_e / N$')
#ne_axis.set_ylabel('$N_e / N \cdot v L / 2 D$')



#we'll parameterize the model as follows: 
#washout= 4*D*lambda/v**2
#beta=sqrt(washout-1)
#gamma = kl
#ellstar=lv/D=2*gamma/beta

#we'll also rescale the x axis 
#let t = v x / D

def calculate_gammas(betas):
    betas = np.array(betas)
    # formula for gamma from notes
    return (np.arctan(2*betas/ (betas*betas-1))*(betas>1)
            +(np.pi+np.arctan(2*betas/(betas*betas-1)))*(betas<=1))

def rho_I(t,wash):
    washout=wash
    beta=np.sqrt(washout-1)
    gamma=calculate_gammas(beta)
    ellstar=2*gamma/beta
    return np.exp( t / 2 )*(np.cos((beta/2)*t)+(1/beta)*np.sin((beta/2)*t))

def rho_II(t,wash):
    washout=wash
    beta=np.sqrt(washout-1)
    gamma=calculate_gammas(beta)
    ellstar=2*gamma/beta
    return (np.exp(ellstar/2)*(np.cos(gamma)+(1/beta)*(np.sin(gamma))))

def rho(x,wash):
    washout=wash
    beta=np.sqrt(washout-1)
    gamma=calculate_gammas(beta)
    ellstar=2*gamma/beta
    return np.piecewise(x,[x<=ellstar,x>ellstar],[lambda x:rho_I(x,wash),lambda x: rho_II(x,wash)])

def u_I(t,wash):
    washout=wash
    beta=np.sqrt(washout-1)
    gamma=calculate_gammas(beta)
    ellstar=2*gamma/beta
    return np.exp( -t / 2 )*(np.cos((beta/2)*t)+(1/beta)*np.sin((beta/2)*t))

def u_II(t,wash):
    washout=wash
    beta=np.sqrt(washout-1)
    gamma=calculate_gammas(beta)
    ellstar=2*gamma/beta
    return np.exp(ellstar / 2)*(np.cos((gamma))+(1/beta)*np.sin((gamma)))*np.exp(-t)
    
def u(x,wash):
    washout=wash
    beta=np.sqrt(washout-1)
    gamma=calculate_gammas(beta)
    ellstar=2*gamma/beta
    return np.piecewise(x,[x<=ellstar,x>ellstar],[lambda x:u_I(x,wash),lambda x: u_II(x,wash)])
    
def g(x,wash):
    return rho(x,wash)*u(x,wash)



#schematic plot of model position function 

def step_lambda(x):
    midpoint=int(x)//5
    y=np.ones(midpoint)
    y=np.append(y,np.zeros(x-midpoint))
    return y


#schematic plot of model nutrient function

def step_lambda_nutrients(x):
    midpoint=int(x)//4
    y=np.zeros(midpoint)
    y=np.append(y,np.ones(x-midpoint))
    return y


#analytical expression for lambda_e/lam

def analytic_lam_e(washout):
    beta=np.sqrt(washout-1)
    gamma=calculate_gammas(beta)
    ellstar=2*gamma/beta
    
    term1=(beta*(beta**2*ellstar+ellstar+2)+(beta**2-1)*np.sin(beta*ellstar)-2*beta*np.cos(beta*ellstar))/(2*beta**3)
    term2=(np.cos(gamma)+(1/beta)*(np.sin(gamma)))**2
    return term1/(term1+term2)



#Calculate Ne/N for model A 


#Ne over N times Ellstar
    
def calculate_NeoverN(washout):
    beta=np.sqrt(washout-1)
    gamma=calculate_gammas(beta)
    ellstar=2*gamma/beta
    
    yellow=((beta*(beta**2*ellstar+ellstar+2)+(beta**2-1)*np.sin(2*gamma)
             - 2*beta*np.cos(2*gamma))/(2*beta**3) + (np.cos(gamma)+np.sin(gamma)/beta)**2)

    orange=(beta*(beta**2*ellstar+ellstar+2)+(beta**2-1)*np.sin(2*gamma) - 2*beta*np.cos(2*gamma))/(2*beta**3)

    red = np.exp(ellstar/2)*(np.cos(gamma)+np.sin(gamma)/beta)

    green1=(  ((np.exp(-ellstar/2)/(2*beta**3*(9*beta**2+1)))*( (np.sin(gamma)*(27*beta**4
            -24*beta**2-3)) + (np.sin(3*gamma)*(3*beta**4-12*beta**2+1)) 
            - (6*(9*beta**3+beta)*np.cos(gamma)) -(2*beta*(5*beta**2-3)*np.cos(3*gamma)) ))
            +(64/(18*beta**2+2))  )

    green2=( (8/washout)*((beta**2+1)/(2*beta))**2 * (((np.exp(-ellstar/2)/(2*(9*beta**5+10*beta**3+beta)))
            *( (np.sin(gamma)*(9*beta**4-26*beta**2-3)) + (np.sin(3*gamma)*(-3*beta**4-2*beta**2+1)) 
            - (4*(9*beta**3+beta)*np.cos(gamma)) +(4*(beta**3+beta)*np.cos(3*gamma)) ))
            +((16*beta**2)/(9*beta**4+10*beta**2+1))  ) )

    green = green1+green2

    pink = (4/washout)*np.exp(-ellstar/2)*(np.cos(gamma)+np.sin(gamma)/beta)**3

    NeoverN=yellow*orange/(red*(green+pink))

    return NeoverN




#Plot results

x1=np.linspace(0,100,100)
x2=np.linspace(0,5,100)
x3=np.linspace(0,40,100)

#Schematic of toy model A

schem_nut_axis.plot([0,0.3,0.3,1],[0,0,1,1],'-',color='orange')
schem_nut_axis.plot([0.175,0.28],[1,1],'k:')


schem_nut_axis.text(0.075,0.9,'$\\lambda$',fontsize=8)
#schem_nut_axis.text(0.3,1.05,'$K$')

schem_pos_axis.plot([0,0.7,0.7,1],[1,1,0,0],'-',color='orange')

# Off to right
#schem_pos_axis.plot([0.7,0.7,0.8],[1.07,1.17,1.17],'k:')
#schem_pos_axis.text(0.82,1.07,'$\\ell$',fontsize=9)

# Above
schem_pos_axis.plot([0.7,0.7],[0,1.07],'k:')
schem_pos_axis.text(0.68,1.09,'$\\ell$',fontsize=9)


#Plot qwm profiles

for washout,xs,pug_axis in zip([10,1.04],[x2,x3],[pug_qwm_axis,pug_spa_axis]):

    beta=np.sqrt(washout-1)
    gamma=calculate_gammas(beta)
    ellstar=2*gamma/beta

    rhos = rho(xs,washout)
    us = u(xs,washout)
    gs = g(xs,washout)

    color = 'tab:blue'
    lns1=pug_axis.plot(xs, rhos/rhos[-1], label='$\\rho(x)$',color=color,zorder=2)

    color = 'tab:red'
    lns2=pug_axis.plot(xs, us/us[0], label='u(x)',color=color,zorder=2)

    color = 'tab:green'
    pug_axis.fill_between(xs,np.zeros_like(xs),gs/gs.max(), color=color,    alpha=0.25,zorder=1,label='g(x)')


    if washout>2: # only plot legend for top panel (repeated in bottom panel)
        pug_axis.legend(loc='center right',frameon=False)

    pug_axis.set_ylim([0,1.2])

    pug_axis.axvline(x=ellstar,color='k',ls=':')

    pug_axis.set_yticks([])
    pug_axis.set_xticks([])



#Plot effective lambda

washout_list = 1+np.logspace(log10(0.01),2,100) 
eff_lam=[]
for ele in washout_list:
    eff_lam+=[analytic_lam_e(ele)]

x_asym=np.logspace(-2,-1.5,100)

x1_asym=np.logspace(1.7,2,100)


lame_axis.loglog(washout_list-1, eff_lam, color='black')
lame_axis.tick_params(axis='y')

lame_axis.semilogx(x1_asym/1.7, 4/(x1_asym+1), color="red")
lame_axis.semilogx(x_asym*1.5, 0.8*np.ones(100),color="red")

xs = np.logspace(-2,2,100)
lame_axis.loglog(xs,4/xs,'r:')
lame_axis.loglog(xs,np.ones_like(xs),'r:')


lame_axis.set_ylim([1e-02,2])
lame_axis.set_xlim([1e-02,1e02])
lame_axis.set_xticklabels([])
Nes=[]
#washout_list=np.linspace(1.04,1000,100000)
washout_list = 1+np.logspace(log10(0.01),2,100) 
for washout in washout_list:
    Nes+=[calculate_NeoverN(washout)]

print washout_list
  
x2_asym=np.logspace(2.4-1,3-1,100)

x3_asym=np.logspace(-2,-1.8,100)


ne_axis.loglog(washout_list-1, Nes, '-', color='black')
ne_axis.tick_params()

ne_axis.semilogx(x3_asym*1.5,(np.pi**2/10)*np.power(x3_asym,-3)*np.exp(-np.pi/(np.sqrt(x3_asym))), color="red")
ne_axis.semilogx(x2_asym, 0.2*np.ones(100),color="red")

xs = np.logspace(-2,2,100)
ne_axis.loglog(xs,np.ones_like(xs)*0.5,'r:')
ne_axis.loglog(xs,np.pi**2/10*np.power(xs,-3)*np.exp(-np.pi/np.sqrt(xs)),'r:')


ne_axis.set_xlim([1e-02,1e02])
ne_axis.set_ylim([1e-08,1e01])


#plt.show()
pylab.savefig('../figures/figure_2.pdf',bbox_inches='tight')




