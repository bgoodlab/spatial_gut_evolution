import pylab
from scipy.integrate import solve_ivp # numerical solver for initial value problem
from scipy.optimize import brentq, newton # numerical root finding routine
from scipy.special import exprel,log1p,expm1,logsumexp #helper functions for numerical precision
import numpy
import sys

import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib as mpl
import pylab as plt
import matplotlib.gridspec as gridspec
from numpy.random import randint, shuffle, poisson, binomial, choice, hypergeometric
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe


# We're using the following parameterization of model A
#
# washout = 4*lam*D/v^2
# beta = 2 D k / v = sqrt(4 D^2 / v^2 * lam / D * (1-v^2/4 lam D)) = sqrt(washout-1)
# gamma = kl = (v / 2 D)*beta*l --> formula in notes
# ellstar = scaled l = l v / D = 2*gamma/beta
#


# Helper function that implements exact eigenvalue equation for model A
def calculate_gammas(betas):
    betas = numpy.array(betas)
    # formula for gamma from notes
    return numpy.arctan(2*betas/ (betas*betas-1))*(betas>1)+(numpy.pi+numpy.arctan(2*betas/(betas*betas-1)))*(betas<=1)

# Implements differential equation for w(x). Needs to turn it from 2nd order ODE to system of 1st order ODEs to work with solver
# For numerical stability, better to work with gamma(x) = log w(x)/w(0)
def scaled_w_system(t,y,w0,s,washout,ellstar):
        
        # we start from the ODE
        #
        # 0 = D gamma'' + D (gamma')^2 + v gamma' + (1+s)*lam 
        #     - (1+s)*lam*w(0)*exp(gamma)/2 - D*w0*exp(gamma)*(gamma')^2
        #
        # we parameterize the system by calling
        #
        # y0 = gamma
        # y1 = gamma'
        #
        # change x axis to 
        # t = x v / D 
        #
        # dt = v / D dx 
        # partial_t = D / v partial_x 
        # partial_x = v / D partial_t 
        #
        # then turns into
        #
        # 0 = gamma'' + (gamma')^2 + gamma' + (1+s)*(4*lam*D/v^2)/4*[1-w0*exp(gamma)/2]
        #      - w0*exp(gamma)*(gamma')^2
        #

        dy0 = y[1,:] 
        dy1 = - y[1,:] - numpy.square(y[1,:]) - ((1+s) * (washout/4) * (1-w0*numpy.exp(y[0,:])/2)) * (t<=ellstar) + w0*numpy.exp(y[0,:])*numpy.square(y[1,:])
    
        return numpy.vstack([dy0,dy1])

# Calculates boundary value of w(x) at x=ellstar, starting from given initial conditions at x=0
def boundary_value(w0,s,washout,ellstar):  
   
    t_eval = numpy.linspace(0,ellstar,100) 
    
    y0 = numpy.array([0,0])

    system = (lambda t, y: scaled_w_system(t,y,w0,s,washout,ellstar))
    sol = solve_ivp(system, [0, ellstar], y0,vectorized=True,t_eval=t_eval)

    # from equation in notes
    yend = sol.y[1][-1]+exprel(w0*numpy.exp(sol.y[0][-1]))
    
    return yend

# Uses shooting method to calculate value of w0 that satisfies boundary condition at x=ellstar
def calculate_logw0(s,washout,ellstar):
	# upper bound
    #print "in calculate logw0: ", boundary_value(s/2,s,washout,ellstar), boundary_value(2.1,s,washout,ellstar)

    logw0 = brentq(lambda x: boundary_value(numpy.exp(x),s,washout,ellstar),numpy.log(s/2),numpy.log(2.1))
    #logw0 = newton(lambda x: boundary_value(x,s,washout,ellstar),numpy.log(s))
    
    return logw0

# Uses shooting method to calculate ellstar for wildtype population using numerical solver    
def calculate_ellstar(washout,ellstar0):
    
    logellstar = newton(lambda x: boundary_value(0,0,washout,numpy.exp(x)),numpy.log(ellstar0))
    
    return numpy.exp(logellstar)
    
# Calculate w(x) profile for given value of w0, other params
def calculate_logws(w0,s,washout,ellstar):

    # use solver to obtain w(x) for x<ellstar
    t_eval = numpy.linspace(0,ellstar,200) 
    
    y0 = numpy.array([0,0])

    system = (lambda t, y: scaled_w_system(t,y,w0,s,washout,ellstar))
    sol = solve_ivp(system, [0, ellstar], y0,vectorized=True,t_eval=t_eval)

    ts = sol.t
    gammas = sol.y[0]
    gammaprimes = sol.y[1]
 
    # now patch on to analytical solution for x>ellstar
    new_ts = numpy.linspace(ellstar,ellstar+10,100)
    
    # Need to handle this case separately to avoid 0/0 error
    if w0*numpy.exp(gammas[-1]) < 1e-05:
        new_gammas = gammas[-1]-(new_ts-ts[-1])
        new_gammaprimes = - numpy.ones_like(new_ts)
    # Otherwise, when w0 is not too small
    else:
        new_gammas = numpy.log(-log1p(-(1-numpy.exp(-w0*numpy.exp(gammas[-1])))*numpy.exp(-(new_ts-ts[-1])))/w0)
        new_gammaprimes = numpy.log(exprel(-w0))-(new_ts-ts[-1])-numpy.log(1+expm1(-w0)*numpy.exp(-(new_ts-ts[-1]))) - new_gammas

    ts = numpy.hstack([ts,new_ts])
    gammas = numpy.hstack([gammas,new_gammas])
    gammaprimes = numpy.hstack([gammaprimes,new_gammaprimes])
    return ts,gammas, gammaprimes

#######
#
# Main function: plot stuff
#
#######

washout_conditions = [1.9,1.04]
specific_ss = [1e-03,1e-02,1e-01]

# Create figure object
pylab.figure(1,figsize=(7, 2))
fig = pylab.gcf()

mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.frameon']  = False
mpl.rcParams['legend.fontsize']  = 'small'

# Create grids for plots
outer_grid = gridspec.GridSpec(1,3,width_ratios=[2.5,1.4,1],wspace=0.5)
# Grid to hold w(x) vs x plots
w_vs_x_grid =  gridspec.GridSpecFromSubplotSpec(2,1,height_ratios=[1,1],hspace=0.4, subplot_spec=outer_grid[1])
# Grid to hold pfix(s) vs s plots
pfix_vs_s_grid =  gridspec.GridSpecFromSubplotSpec(2,1,height_ratios=[1,1],hspace=0.3, subplot_spec=outer_grid[2])

# Create plots themselves
w_vs_x_axes = []
pfix_vs_s_axes = []
for washout_idx in xrange(0,len(washout_conditions)):

    w_axis = plt.Subplot(fig, w_vs_x_grid[washout_idx])
    fig.add_subplot(w_axis)
    w_vs_x_axes.append(w_axis)
    
    pfix_axis = plt.Subplot(fig, pfix_vs_s_grid[washout_idx])
    fig.add_subplot(pfix_axis)
    pfix_vs_s_axes.append(pfix_axis)

    if washout_idx==(len(washout_conditions)-1):
        w_axis.set_xlabel('Position, $x$ (a.u.)')
        w_axis.set_ylabel('Fixation profile, $w_s(x)$',horizontalalignment='left')
        pfix_axis.set_xlabel('Fitness effect, $s$')
        pfix_axis.set_ylabel('      Effective $p_\mathrm{fix}$', horizontalalignment='left')

# Dummy axis to take up space where schematic panel goes
dummy_axis = plt.Subplot(fig, outer_grid[0])
fig.add_subplot(dummy_axis)
dummy_axis.set_ylim([0,1])
dummy_axis.set_xlim([0,1])

dummy_axis.spines['top'].set_visible(False)
dummy_axis.spines['right'].set_visible(False)
dummy_axis.spines['left'].set_visible(False)
dummy_axis.spines['bottom'].set_visible(False)

dummy_axis.set_xticks([])
dummy_axis.set_yticks([])


for washout_idx in xrange(0,len(washout_conditions)):

    washout = washout_conditions[washout_idx]
    pfix_axis = pfix_vs_s_axes[washout_idx]
    w_axis = w_vs_x_axes[washout_idx]
    
    beta = numpy.sqrt(washout-1)
    gamma = calculate_gammas(beta)
    ellstar = 2*gamma/beta

    sys.stderr.write("beta = %g, ellstar=%g\n" % (beta,ellstar))

    # Recalculate ellstar using numerical solver (aids numerical stability for small s)
    ellstar0 = ellstar
    ellstar = calculate_ellstar(washout,ellstar0)
    
    # Calculate ugammas = log(u) and rhogammas = log(rho) up to overall constant
    ts,ugammas,ugammaprimes = calculate_logws(0,0,washout,ellstar)
    rhogammas = ugammas+ts
    
    # g function (distribution of common ancestors)
    ggammas = rhogammas+ugammas

    #pylab.figure()
    #pylab.plot(ts,ggammas,'-')


    # Calculate w(x) for a whole bunch of s values
    ss = numpy.logspace(-3,-1,50)
    w0s = []
    pfixs = []
    for s in ss:
        sys.stderr.write("Calculating w(x) for s=%g\n" % s)
        logw0 = calculate_logw0(s,washout,ellstar)
        w0 = numpy.exp(logw0)
        w0s.append(w0)
        ts,gammas,gammaprimes = calculate_logws(w0,s,washout,ellstar)
    
        integrand = (gammas+rhogammas)*(ts<=ellstar)
        max_integrand = integrand.max()
    
        # log pfix up to constant
        logpfix = logsumexp(integrand-max_integrand)+max_integrand+logw0
        
        pfixs.append(numpy.exp(logpfix))
    
    w0s = numpy.array(w0s)
    pfixs = numpy.array(pfixs)

    scale_factor = (2*ss[0]/pfixs[0])
    pfixs = pfixs* scale_factor

    # Now plot the results
    
    pfix_axis.loglog(ss,ss/ss[0]*pfixs[0],'k:')  
    pfix_axis.loglog(ss,pfixs,'r-')
    pfix_axis.set_xlim([1e-03,1e-01])
    
    pfix_axis.set_ylim([1e-03,2])
    #w_axis.set_ylim([-30,5])
    
        
    for s in specific_ss:
        w0 = numpy.exp(calculate_logw0(s,washout,ellstar))
        ts,gammas,gammaprimes = calculate_logws(w0,s,washout,ellstar)
        ws = numpy.exp(gammas+numpy.log(w0))
        print s,ws[0]
        w_axis.semilogy(ts,ws,'r-')
    
    #w_axis.set_xlim([0,ellstar*1.5])
    #w_axis.set_xticks([])

    
    w_axis.set_ylim([1e-09,6])
    w_axis.set_yticks([1,1e-03,1e-06,1e-09])
    
    ymin, ymax = w_axis.get_ylim()
    w_axis.plot([ellstar,ellstar],[ymin,ymax],'k:')

    w_axis.set_xticks([])
    
    if washout_idx==0:
        pfix_axis.set_xticklabels([])
   
w_vs_x_axes[0].set_title('Diffusion-dominated growth',fontsize=7)
ttl = w_vs_x_axes[0].title
ttl.set_position([.5, 0.95])
w_vs_x_axes[1].set_title('Flow-dominated growth',fontsize=7)
ttl = w_vs_x_axes[1].title
ttl.set_position([.5, 0.95])

w_vs_x_axes[0].text(-0.4, 1.1, 'b', horizontalalignment='center', verticalalignment='center', transform=w_vs_x_axes[0].transAxes,fontweight='bold',fontsize=7)
pfix_vs_s_axes[0].text(-0.25, 1.15, 'c', horizontalalignment='center', verticalalignment='center', transform=pfix_vs_s_axes[0].transAxes,fontweight='bold',fontsize=7)

fig.savefig('../figures/figure_3.pdf',bbox_inches='tight')
 
