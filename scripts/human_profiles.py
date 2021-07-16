import pylab
from scipy.integrate import solve_ivp # numerical solver for initial value problem
from scipy.optimize import brentq, newton # numerical root finding routine
from scipy.special import exprel,log1p,expm1,logsumexp #helper functions for numerical precision
import numpy
import sys
import parse_scraped_data   
 
    
# Implements differential equation for w(x). Needs to turn it from 2nd order ODE to system of 1st order ODEs to work with solver
# For numerical stability, better to work with gamma(x) = log w(x)/w(0)
def scaled_w_system(t,y,w0,s,dell,ellstar,scaled_v,scaled_growth):
        
        # we start from the ODE
        #
        # 0 = D gamma'' + D (gamma')^2 + v(x) gamma' + (1+s)*lam 
        #     - (1+s)*lam*w(0)*exp(gamma)/2 - D*w0*exp(gamma)*(gamma')^2
        #
        # we parameterize the system by calling
        #
        # y0 = gamma
        # y1 = gamma'
        #
        # change x axis to 
        # t = x vinf / D 
        #
        # dt = v / D dx 
        # partial_t = D / v partial_x 
        # partial_x = v / D partial_t 
        #
        # then turns into
        #
        # 0 = gamma'' + (gamma')^2 + (v(x) / vinf) gamma' + (1+s)*(4*lam(x)*D/vinf^2)/4*[1-w0*exp(gamma)/2]
        #      - w0*exp(gamma)*(gamma')^2
        #

        dy0 = y[1,:] 
        dy1 = - scaled_v(t)*y[1,:] - numpy.square(y[1,:]) - ((1+s) * (scaled_growth(t+dell)) * (1-w0*numpy.exp(y[0,:])/2)) * (t<=ellstar) + w0*numpy.exp(y[0,:])*numpy.square(y[1,:])
    
        return numpy.vstack([dy0,dy1])

# Calculates boundary value of w(x) at x=ellstar, starting from given initial conditions at x=0
def boundary_value(w0,s,dell,ellstar,scaled_v,scaled_growth):  
   
    t_eval = numpy.linspace(0,ellstar,100) 
    
    y0 = numpy.array([0,0])

    system = (lambda t, y: scaled_w_system(t,y,w0,s,dell,ellstar,scaled_v,scaled_growth))
    sol = solve_ivp(system, [0, ellstar], y0,vectorized=True,t_eval=t_eval)

    # from equation in notes
    yend = sol.y[1][-1]+exprel(w0*numpy.exp(sol.y[0][-1]))
    
    return yend

# Uses shooting method to calculate value of w0 that satisfies boundary condition at x=

def calculate_logw0(s,dell,ellstar,scaled_v,scaled_growth):
	# upper bound
    #print "in calculate logw0: ", boundary_value(s/2,s,washout,ellstar), boundary_value(2.1,s,washout,ellstar)

    logw0 = brentq(lambda x: boundary_value(numpy.exp(x),s,dell,ellstar,scaled_v,scaled_growth),numpy.log(s/2),numpy.log(2.1))
    
    return logw0

# Uses shooting method to calculate ellstar for wildtype population using numerical solver    
def calculate_dell(ellstar,scaled_v,scaled_growth,dell0):
    
    dell = newton(lambda x: boundary_value(0,0,x,ellstar,scaled_v,scaled_growth),dell0)
    
    return dell
    
# Calculate w(x) profile for given value of w0, other params
def calculate_logws(w0,s,dell,ellstar,scaled_v,scaled_growth):

    # use solver to obtain w(x) for x<ellstar
    t_eval = numpy.linspace(0,ellstar,200) 
    
    y0 = numpy.array([0,0])

    system = (lambda t, y: scaled_w_system(t,y,w0,s,dell,ellstar,scaled_v,scaled_growth))
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
# Main function: do stuff
#
#######
if __name__=='__main__':

    for species in parse_scraped_data.speciess:
    
        ellstar = parse_scraped_data.scaled_xstar
    
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
    
        pylab.figure()
        pylab.plot(ts, numpy.exp(rhogammas-rhogammas.max()),'-',label='rho(x)')
        pylab.plot(ts, numpy.exp(ugammas-ugammas.max()),'-',label='u(x)')
        pylab.plot(ts, numpy.exp(ggammas-ggammas.max()),'-',label='g(x)')
        pylab.plot(ts, lams/lams.max(),'-',label='lam(x)')
        pylab.xlabel('Position (diffuion lengths)')
        pylab.legend(loc='center right',frameon=False)
        #pylab.show()
        pylab.plot([ellstar,ellstar],[0,1],'k:')
        #sys.exit(0)
        #pylab.savefig('%s_refit_profiles.pdf' % species)

    