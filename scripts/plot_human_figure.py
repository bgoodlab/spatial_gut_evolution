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

from scipy.integrate import quad, trapz



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
v_axis.set_ylabel('Flow rate,\n $v(x)$ [$\\mu m/s$]')
v_axis.set_xlim([0,max_x])
v_axis.set_ylim([0,35])
v_axis.set_xticklabels([])

v_axis.fill_between([0,parse_scraped_data.cecum_end],[0,0],[35,35],color=parse_scraped_data.cecum_color,alpha=0.25)
v_axis.fill_between([parse_scraped_data.cecum_end,parse_scraped_data.ascending_end],[0,0],[35,35],color=parse_scraped_data.ascending_color,alpha=0.25)
v_axis.fill_between([parse_scraped_data.ascending_end,parse_scraped_data.transverse_end],[0,0],[35,35],color=parse_scraped_data.transverse_color,alpha=0.25)

v_axis.text(-11,37,'Cecum',color=parse_scraped_data.cecum_color,weight='bold')
v_axis.text(9,37,'Ascending',color=parse_scraped_data.ascending_color,weight='bold')
v_axis.text(39,37,'Transverse',color=parse_scraped_data.transverse_color,weight='bold')

v_axis.text(-0.35, 1.05, 'a', horizontalalignment='center', verticalalignment='center', transform=v_axis.transAxes,fontweight='bold',fontsize=7)


lam_axis = plt.Subplot(fig, outer_grid[1,0])
fig.add_subplot(lam_axis)
lam_axis.set_ylabel('Growth rate,\n $\\lambda(x)$ [$hr^{-1}$]')
lam_axis.set_xlim([0,max_x])
lam_axis.set_xticklabels([])

lam_axis.text(-0.35, 1.05, 'b', horizontalalignment='center', verticalalignment='center', transform=lam_axis.transAxes,fontweight='bold',fontsize=7)



rho_axis = plt.Subplot(fig, outer_grid[2,0])
fig.add_subplot(rho_axis)
rho_axis.set_ylabel('Density, \n $\\rho(x)/\\rho(L)$')
rho_axis.set_xlabel('Position, $x$ [cm]')
rho_axis.set_xlim([0,max_x])

rho_axis.text(-0.35, 0.95, 'c', horizontalalignment='center', verticalalignment='center', transform=rho_axis.transAxes,fontweight='bold',fontsize=7)


u_axis = plt.Subplot(fig, outer_grid[0,1])
fig.add_subplot(u_axis)
u_axis.set_ylabel('u(x)/u(0)')
u_axis.set_xlim([0,max_x])
u_axis.set_xticklabels([])

u_axis.text(-0.1, 0.9, 'd', horizontalalignment='center', verticalalignment='center', transform=u_axis.transAxes,fontweight='bold',fontsize=7)


g_axis = plt.Subplot(fig, outer_grid[1,1])
fig.add_subplot(g_axis)
g_axis.set_ylabel('g(x)')
g_axis.set_xlabel('Position, $x$ [cm]')
g_axis.set_xlim([0,max_x])

g_axis.text(-0.1, 0.9, 'e', horizontalalignment='center', verticalalignment='center', transform=g_axis.transAxes,fontweight='bold',fontsize=7)


lame_axis = plt.Subplot(fig, inner_inner_grid[0])
fig.add_subplot(lame_axis)
lame_axis.set_xlabel('$\\lambda_e/\\lambda$')
lame_axis.set_ylim([1e-02,1])
lame_axis.set_xlim([-2,2])
lame_axis.set_xticks([-1,1])
lame_axis.set_xticklabels([])
lame_axis.spines['top'].set_visible(False)
lame_axis.spines['right'].set_visible(False)
lame_axis.semilogy([0],[1e-04])

lame_axis.text(-0.1, 1.15, 'f', horizontalalignment='center', verticalalignment='center', transform=lame_axis.transAxes,fontweight='bold',fontsize=7)


ne_axis = plt.Subplot(fig, inner_inner_grid[1])
fig.add_subplot(ne_axis)
ne_axis.set_xlabel('$N_e/N$')
ne_axis.set_ylim([1e-02,1])
ne_axis.set_xlim([-2,2])
ne_axis.set_xticks([-1,1])
ne_axis.set_xticklabels([])
ne_axis.spines['top'].set_visible(False)
ne_axis.spines['right'].set_visible(False)
ne_axis.semilogy([0],[1e-04])
ne_axis.set_yticklabels([])

ne_axis.text(-0.1, 1.15, 'g', horizontalalignment='center', verticalalignment='center', transform=ne_axis.transAxes,fontweight='bold',fontsize=7)


pfix_axis = plt.Subplot(fig, inner_grid[1])
fig.add_subplot(pfix_axis)
pfix_axis.set_ylabel('$p_\mathrm{fix}$')
pfix_axis.set_xlabel('s')
pfix_axis.set_xlim([1e-03,1e-01])
pfix_axis.set_ylim([1e-03,2e-01])

pfix_axis.text(-0.8, 1.0, 'h', horizontalalignment='center', verticalalignment='center', transform=pfix_axis.transAxes,fontweight='bold',fontsize=7)


#pylab.savefig('../figures/figure_4.pdf',bbox_inches='tight')
#sys.exit(0)
     
# Diffusion length scale in cm
ell_diffusion = parse_scraped_data.ell_diffusion*1e-04 
# Diffusion timescale (in hours)
t_diffusion = parse_scraped_data.t_diffusion/3600
# Final flow velocity (in um/s) 
vinf = parse_scraped_data.vinf 
# Critical ellstar (in diffusion lengths)
ellstar = parse_scraped_data.scaled_xstar
# Length of colon (in diffusion lengths)
scaled_L = parse_scraped_data.scaled_L
    
for species,color,pretty_name,bar_position in zip(parse_scraped_data.speciess, parse_scraped_data.species_colors, parse_scraped_data.pretty_speciess,[-1,1]):
    
    sys.stderr.write("Calculating results for %s...\n" % pretty_name)
    
    scaled_v = parse_scraped_data.calculate_scaled_flow
    scaled_growth = parse_scraped_data.parse_scaled_growth_rate(species)
    
    # Recalculate ellstar using numerical solver (aids numerical stability for small s)
    dell = calculate_dell(ellstar,scaled_v,scaled_growth,0)
      
    # Calculate ugammas = log(u) and rhogammas = log(rho) up to overall constant
    ts,ugammas,ugammaprimes = calculate_logws(0,0,dell,ellstar,scaled_v,scaled_growth)
    
    integrated_exponents = parse_scraped_data.calculate_scaled_integrated_exponents(ts)
    
    rhogammas = ugammas+2*integrated_exponents
    
    # g function (distribution of common ancestors)
    ggammas = rhogammas+ugammas
    # normalize g function
    ggammas = ggammas - logsumexp(ggammas)
    gs = numpy.exp(ggammas)

    lams = scaled_growth(ts)
    loglams = numpy.log(lams)
    
    # Compare dell to place where lam hits half maximum
    ellhalf = ts[numpy.argmin(numpy.fabs(lams-lams[0]*0.5))]
    
    sys.stderr.write("Recalculated dell = %g; (ellstar=%g, ellhalf=%g)\n" % (dell, ellstar,ellhalf))
    
    ## Calculate fraction of ancestral region with growth rate less than lam_min
    ## where lam_min = once per week = 1.0/168 1/hr (but converted to diffusion units)
    lam_min = (1.0/48)*t_diffusion
    #lam_min = lams[0]*0.1
    min_idx = numpy.nonzero(lams<lam_min)[0][0]
    fraction = gs[min_idx:].sum()
    
    #min_idx = numpy.nonzero(ts>20.0/ell_diffusion)[0][0]
    #fraction = gs[min_idx:].sum()
    #lam_min = lams[min_idx]
    
    sys.stderr.write("%g percent of cells with growth rate less than %g per day \n" % (fraction,lam_min/t_diffusion*24) )
    
    
    lam_max = 0.5*t_diffusion
    #lam_max = lams[0]*0.9
    max_idx = numpy.nonzero(lams<lam_max)[0][0]
    fraction = gs[:max_idx].sum()
    
    #min_idx = numpy.nonzero(ts>20.0/ell_diffusion)[0][0]
    #fraction = gs[min_idx:].sum()
    #lam_min = lams[min_idx]
    
    sys.stderr.write("%g percent of cells with growth rate > %g per hour \n" % (fraction,lam_max/t_diffusion) )
    
      
    ## Calculate effective growth rate using formula
    ## 
    ## lam_e = int lam(x) u(x) rho(x) dx
    ##
    ## We will eventually plot lam_e / lam(0) 
    ##
    loglame = logsumexp(ggammas+loglams)
    lame = numpy.exp(loglame)
    lame_over_lam = numpy.exp(loglame-loglams[0])
    sys.stderr.write("lam_e/lam(0)= %g\n" % (lame_over_lam))

    ## Calculate effective population size using formula
    ## 
    ## N_e = lam_e / int [lam(x) * u(x)^2 + 2 D u'(x)^2 ] rho(x) dx
    ## 
    ## Note: in rescaled units this is
    ##
    ## N_e = scaled_lam_e / int [ scaled_lam(z) + 2 partial_z(log u(z))] * u(z)^2 rho(z) dz
    ##
    ## We will eventually plot N_e / N = N_e / int rho(z) dz
    ##
    
    # First calculate factor out front
    factors = lams + 2 * ugammaprimes**2 
    log_factors = numpy.log(factors)
    
    # Used to calculate int rho(z) dz over length of colon
    cutoff = numpy.argmin(numpy.abs(ts-scaled_L))
    # cutoff=numpy.where(numpy.abs((ts-scaled_L))<0.01)[0][0]

    # Put everything together
    log_ne_over_n = logsumexp(loglams+rhogammas+ugammas) - logsumexp(log_factors + 2*ugammas + rhogammas) + logsumexp(rhogammas+ugammas) - logsumexp(rhogammas[:cutoff]) 
    ne_over_n = numpy.exp(log_ne_over_n)

    sys.stderr.write("Ne/N= %g\n" % (ne_over_n))
    
    # Plot results
    
    vs = scaled_v(ts)*vinf
    v_axis.plot(ts*ell_diffusion, vs,'k-')
    rho_axis.plot(ts*ell_diffusion, numpy.exp(rhogammas-rhogammas.max()),'-',color=color)
    u_axis.plot(ts*ell_diffusion, numpy.exp(ugammas-ugammas.max()),'-',color=color)
    g_axis.plot(ts*ell_diffusion, numpy.exp(ggammas),'-',color=color)
    lam_axis.plot(ts*ell_diffusion, lams/t_diffusion,'-',color=color,label=pretty_name)
    
    lame_axis.bar([bar_position],[lame_over_lam],width=1.4,color=color)
    ne_axis.bar([bar_position],[ne_over_n],width=1.4,color=color)
    


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

