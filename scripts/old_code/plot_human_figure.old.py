import pylab
import numpy
import sys
import parse_scraped_data
from human_profiles import *

import sys

import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib as mpl
import pylab as plt
import matplotlib.gridspec as gridspec
from numpy.random import randint, shuffle, poisson, binomial, choice, hypergeometric
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe

max_x = 80

# Create figure objects
pylab.figure(1,figsize=(5, 3.5))
fig = pylab.gcf()

mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.frameon']  = False
mpl.rcParams['legend.fontsize']  = 'small'

# Create grids for plots
outer_grid = gridspec.GridSpec(3,2,width_ratios=[1,1],height_ratios=[1,1,1],wspace=0.3,hspace=0.2)

outer_inner_grid =  gridspec.GridSpecFromSubplotSpec(2,1,height_ratios=[2,3], hspace=0, subplot_spec=outer_grid[2,1])
inner_grid =  gridspec.GridSpecFromSubplotSpec(1,2,width_ratios=[2,2], wspace=1, subplot_spec=outer_inner_grid[1])
inner_inner_grid = gridspec.GridSpecFromSubplotSpec(1,2,width_ratios=[1,1], wspace=0.5, subplot_spec=inner_grid[0])

dummy_axis = plt.Subplot(fig, outer_inner_grid[0])
fig.add_subplot(dummy_axis)
dummy_axis.set_ylim([0,1])
dummy_axis.set_xlim([0,1])

dummy_axis.spines['top'].set_visible(False)
dummy_axis.spines['right'].set_visible(False)
dummy_axis.spines['left'].set_visible(False)
dummy_axis.spines['bottom'].set_visible(False)

dummy_axis.set_xticks([])
dummy_axis.set_yticks([])


v_axis = plt.Subplot(fig, outer_grid[0,0])
fig.add_subplot(v_axis)
v_axis.set_ylabel('Flow rate, $v(x)$\n ($\\mu m/s$)')
v_axis.set_xlim([0,max_x])
v_axis.set_ylim([0,35])
v_axis.set_xticklabels([])

lam_axis = plt.Subplot(fig, outer_grid[1,0])
fig.add_subplot(lam_axis)
lam_axis.set_ylabel('Growth rate, $\\lambda(x)$\n ($hr^{-1}$)')
lam_axis.set_xlim([0,max_x])
lam_axis.set_xticklabels([])


rho_axis = plt.Subplot(fig, outer_grid[2,0])
fig.add_subplot(rho_axis)
rho_axis.set_ylabel('Density, \n $\\rho(x)/\\rho(L)$')
rho_axis.set_xlabel('Position, $x$ (cm)')
rho_axis.set_xlim([0,max_x])

u_axis = plt.Subplot(fig, outer_grid[0,1])
fig.add_subplot(u_axis)
u_axis.set_ylabel('u(x)/u(0)')
u_axis.set_xlim([0,max_x])
u_axis.set_xticklabels([])

g_axis = plt.Subplot(fig, outer_grid[1,1])
fig.add_subplot(g_axis)
g_axis.set_ylabel('g(x)')
g_axis.set_xlabel('Position, $x$ (cm)')
g_axis.set_xlim([0,max_x])


lame_axis = plt.Subplot(fig, inner_inner_grid[0])
fig.add_subplot(lame_axis)
lame_axis.set_xlabel('$\\lambda_e/\\lambda$')
lame_axis.set_ylim([0,1])
lame_axis.set_xlim([-2,2])
lame_axis.set_xticks([-1,1])
lame_axis.set_xticklabels([])
lame_axis.spines['top'].set_visible(False)
lame_axis.spines['right'].set_visible(False)

ne_axis = plt.Subplot(fig, inner_inner_grid[1])
fig.add_subplot(ne_axis)
ne_axis.set_xlabel('$N_e/N$')
ne_axis.set_ylim([0,1])
ne_axis.set_yticks([])
ne_axis.set_xlim([-2,2])
ne_axis.set_xticks([-1,1])
ne_axis.set_xticklabels([])
ne_axis.spines['top'].set_visible(False)
ne_axis.spines['right'].set_visible(False)

pfix_axis = plt.Subplot(fig, inner_grid[1])
fig.add_subplot(pfix_axis)
pfix_axis.set_ylabel('$p_\mathrm{fix}$')
pfix_axis.set_xlabel('s')
pfix_axis.set_xlim([1e-03,1e-01])
pfix_axis.set_ylim([1e-03,2e-01])

     

# Diffusion length scale in cm
ell_diffusion = parse_scraped_data.ell_diffusion*1e-04 
# Diffusion timescale in hours
t_diffusion = parse_scraped_data.t_diffusion/3600
# Final diffusion length 
vinf = parse_scraped_data.vinf 
# Critical ellstar (in diffusion lengths)
ellstar = parse_scraped_data.scaled_xstar
    
for species,color,pretty_name,bar_position in zip(parse_scraped_data.speciess, parse_scraped_data.species_colors, parse_scraped_data.pretty_speciess,[-1,1]):
    
    
    scaled_v = parse_scraped_data.calculate_scaled_flow
    scaled_growth = parse_scraped_data.parse_scaled_growth_rate(species)
    
    # Recalculate ellstar using numerical solver (aids numerical stability for small s)
    dell = calculate_dell(ellstar,scaled_v,scaled_growth,0)
    
    sys.stderr.write("Recalculated dell = %g; (ellstar=%g)\n" % (dell, ellstar))
    
    # Calculate ugammas = log(u) and rhogammas = log(rho) up to overall constant
    ts,ugammas,ugammaprimes = calculate_logws(0,0,dell,ellstar,scaled_v,scaled_growth)
    
    integrated_exponents = parse_scraped_data.calculate_scaled_integrated_exponents(ts)
    
    rhogammas = ugammas+2*integrated_exponents
    
    # g function (distribution of common ancestors)
    ggammas = rhogammas+ugammas
    # normalize g function
    ggammas = ggammas - logsumexp(ggammas)

    lams = scaled_growth(ts)
    
    vs = scaled_v(ts)*vinf
    v_axis.plot(ts*ell_diffusion, vs,'k-')
    rho_axis.plot(ts*ell_diffusion, numpy.exp(rhogammas-rhogammas.max()),'-',color=color)
    u_axis.plot(ts*ell_diffusion, numpy.exp(ugammas-ugammas.max()),'-',color=color)
    g_axis.plot(ts*ell_diffusion, numpy.exp(ggammas),'-',color=color)
    lam_axis.plot(ts*ell_diffusion, lams/t_diffusion,'-',color=color,label=pretty_name)
    
    #pylab.show()
    #pylab.plot([ellstar,ellstar],[0,1],'k:')
    #sys.exit(0)
    #pylab.savefig('%s_refit_profiles.pdf' % species)
    
    scaled_lame = 1
    scaled_ne = 1
    
    lame_axis.bar([bar_position],[scaled_lame],width=1.4,color=color)
    ne_axis.bar([bar_position],[scaled_ne],width=1.4,color=color)
    


for species,color in zip(parse_scraped_data.speciess, parse_scraped_data.species_colors):

    scaled_v = parse_scraped_data.calculate_scaled_flow
    scaled_growth = parse_scraped_data.parse_scaled_growth_rate(species)
    
    # Recalculate ellstar using numerical solver (aids numerical stability for small s)
    dell = calculate_dell(ellstar,scaled_v,scaled_growth,0)
    
    sys.stderr.write("Recalculated dell = %g; (ellstar=%g)\n" % (dell, ellstar))
    
    # Calculate ugammas = log(u) and rhogammas = log(rho) up to overall constant
    ts,ugammas,ugammaprimes = calculate_logws(0,0,dell,ellstar,scaled_v,scaled_growth)
    
    integrated_exponents = parse_scraped_data.calculate_scaled_integrated_exponents(ts)
    
    rhogammas = ugammas+2*integrated_exponents
    
    # g function (distribution of common ancestors)
    ggammas = rhogammas+ugammas
    
    lams = scaled_growth(ts)
    
    
    # Calculate w(x) for a whole bunch of s values
    ss = numpy.logspace(-3,-1,20)
    w0s = []
    pfixs = []
    for s in ss:
        sys.stderr.write("Calculating w(x) for s=%g\n" % s)
        logw0 = calculate_logw0(s,dell,ellstar,scaled_v,scaled_growth)
        w0 = numpy.exp(logw0)
        w0s.append(w0)
        ts,gammas,gammaprimes = calculate_logws(w0,s,dell,ellstar,scaled_v,scaled_growth)
    	
    	
        integrand = (gammas+rhogammas)*lams*(ts<ellstar)
        max_integrand = integrand.max()
    
        logpfix = logsumexp(integrand-max_integrand)+max_integrand+logw0
        pfixs.append(numpy.exp(logpfix))
    
    w0s = numpy.array(w0s)
    pfixs = numpy.array(pfixs)

    pfixs = pfixs/pfixs[0]*2*ss[0]
    
    
    # Now plot the results

    # pfix vs s
    pfix_axis.loglog(ss,ss/ss[0],'k:')  
    pfix_axis.loglog(ss,pfixs,'-',color=color)

g_axis.set_yticks([])
u_axis.set_yticks([])
rho_axis.set_yticks([])
v_axis.set_yticks([0,10,20,30])
lam_axis.legend(loc='upper right',frameon=False)
pylab.savefig('../figures/figure_4.pdf',bbox_inches='tight')

