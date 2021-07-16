

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



max_x = 80

# Create figure objects
pylab.figure(1,figsize=(9, 4))
fig = pylab.gcf()

mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.frameon']  = False
mpl.rcParams['legend.fontsize']  = 'small'

# Create grids for plots
outer_grid = gridspec.GridSpec(2,3,width_ratios=[1,2,1],height_ratios=[1,1],wspace=0.5,hspace=0.3)



schem_nut_axis = plt.Subplot(fig, outer_grid[0,0])
fig.add_subplot(schem_nut_axis)
schem_nut_axis.set_ylabel('$\\lambda(n)$')
schem_nut_axis.set_xticks([25])
schem_nut_axis.set_xticklabels(["K"])
schem_nut_axis.set_xlabel('Nutrient concentration, n', labelpad=0)
schem_nut_axis.set_yticks([1])
schem_nut_axis.set_ylim(-0.1,1.5)
schem_nut_axis.set_yticklabels(['$\\lambda$'])




pug_qwm_axis = plt.Subplot(fig, outer_grid[0,1])
fig.add_subplot(pug_qwm_axis)
pug_qwm_axis.set_title('Quasi Well-Mixed Regime ($\\alpha=10$)')


pug_spa_axis = plt.Subplot(fig, outer_grid[1,1])
fig.add_subplot(pug_spa_axis)
pug_spa_axis.set_xlabel('Position, $x$ (a.u.)')
pug_spa_axis.set_title('Spatial Regime, ($\\alpha=1.04$)')


schem_pos_axis = plt.Subplot(fig, outer_grid[1,0])
fig.add_subplot(schem_pos_axis)
schem_pos_axis.set_ylabel('$\\lambda(x)$',labelpad=2.0)
schem_pos_axis.set_xticks([20])
schem_pos_axis.set_xticklabels(r'$\\ell$')
schem_pos_axis.set_xlabel('Position, x',labelpad=2.0)
schem_pos_axis.set_yticks([1])
schem_pos_axis.set_ylim(-0.1,1.5)
schem_pos_axis.set_yticklabels(['$\\lambda$'])




lame_axis = plt.Subplot(fig, outer_grid[0,2])
fig.add_subplot(lame_axis)
lame_axis.set_ylabel('$\\lambda_e/\\lambda$($\\alpha$)')

ne_axis = plt.Subplot(fig, outer_grid[1,2])
fig.add_subplot(ne_axis)
ne_axis.set_xlabel('$\\alpha-1')
ne_axis.set_ylabel('$\\frac{N_e}{N}\\frac{vL}{D}$($\\alpha$)')



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
schem_nut_axis.plot(x1, step_lambda_nutrients(len(x1)), color='orange')
schem_pos_axis.plot(x1, step_lambda(len(x1)),color='orange')


#Plot qwm profiles

washout=10
beta=np.sqrt(washout-1)
gamma=calculate_gammas(beta)
ellstar=2*gamma/beta

color = 'tab:blue'
lns1=pug_qwm_axis.plot(x2, rho(x2,10), label='$\\rho(x)$',color=color)
pug_qwm_axis.set_yticks([])
pug_qwm_axis.set_xticks([])


ax2 = pug_qwm_axis.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
lns2=ax2.plot(x2, u(x2,10), label='u(x)',color=color)
ax2.set_yticks([])
ax2.set_xticks([])


ax3 = pug_qwm_axis.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:green'
lns3=ax3.plot(x2, g(x2,10), label='g(x)', color=color)
ax3.set_yticks([])
ax3.fill_between(x2,np.zeros_like(x2),g(x2,10), color=color, alpha=0.25)


lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax3.legend(lns, labs, loc=0)
ax3.set_xticks([])
ax3.axvline(x=ellstar,color='k', ls=':')



#Plot spatial profiles
washout=1.04
beta=np.sqrt(washout-1)
gamma=calculate_gammas(beta)
ellstar=2*gamma/beta

color = 'tab:blue'
lns1=pug_spa_axis.plot(x3, rho(x3,1.04), label='$\\rho(x)$', color=color)
pug_spa_axis.tick_params(axis='y', labelcolor=color)
pug_spa_axis.set_yticks([])

ax4 = pug_spa_axis.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
lns2=ax4.plot(x3, u(x3,1.04), label='u(x)', color=color)
ax4.set_yticks([])


ax5 = pug_spa_axis.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:green'
lns3=ax5.plot(x3, g(x3,1.04), label='g(x)', color=color)
ax5.set_yticks([])
ax5.set_xticks([])

ax5.fill_between(x3,np.zeros_like(x3),g(x3,1.04), color=color,alpha=0.25)

lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax5.legend(lns, labs, loc=0)
ax5.axvline(x=ellstar,color='k', ls=':')



#Plot effective lambda

wash_list=np.linspace(1.04,1000,10000)
eff_lam=[]
for ele in wash_list:
    eff_lam+=[analytic_lam_e(ele)]

x_asym=np.logspace(-1.4,-0.8,100)

x1_asym=np.logspace(2.5,3,100)

lame_axis.set_ylabel(r'$\frac{\lambda_e}{\lambda}$')



lame_axis.loglog(wash_list-1, eff_lam, color='black')
lame_axis.tick_params(axis='y')

lame_axis.semilogx(x1_asym, 6/(x1_asym), color="red")
lame_axis.semilogx(x_asym, 1.3*np.ones(100),color="red")


Nes=[]
washout_list=np.linspace(1.04,1000,100000)
for washout in washout_list:
    Nes+=[calculate_NeoverN(washout)]

    
x2_asym=np.logspace(2.4,3,100)

x3_asym=np.logspace(-1.44,-1.3,1000)



ne_axis.set_xlabel(r'$\alpha$-1')
ne_axis.set_ylabel(r'$\frac{N_e}{N}\frac{vL}{D}$')

ne_axis.loglog(washout_list-1, Nes, color='black')
ne_axis.tick_params()

ne_axis.semilogx(x3_asym,70*(3*np.pi**2/4)*(x3_asym)*np.exp(-np.pi/(2*np.sqrt(x3_asym))), color="red")
ne_axis.semilogx(x2_asym, 0.6*np.ones(100),color="red")




#plt.show()
pylab.savefig('../figures/figure_2.pdf',bbox_inches='tight')




