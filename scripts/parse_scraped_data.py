import numpy
from scipy.interpolate import interp1d,UnivariateSpline,PchipInterpolator
from pygam import LinearGAM, s

# Human params from jonas's paper
v0 = 32 # microns per second
vinf = 5.26 # microns per second
gradient = 0.75 # microns per second per cm
# v(x) = v0-gradient*x or vinf
xstar = (v0-vinf)/gradient # 35.65 # xstar = , in cm

D = 1e06 # microns squared per second
L = 189 # length of colon in cm

cecum_end = 6.7
ascending_end = 29.8
transverse_end = 88.1

# From jonas's paper
cecum_color = '#BBCBD0'
ascending_color = '#BCDBE6'
transverse_color = '#E9C5DF'

# colorbewer
#cecum_color = '#b3e2cd'
#ascending_color = '#fdcdac'
#transverse_color = '#cbd5e8'

ascending_color = '#66c2a5'
transverse_color = '#fc8d62'
cecum_color = '#8da0cb'

ell_diffusion = D/vinf # diffusion length scale in micrometers (~20 cm) 
t_diffusion = D/vinf/vinf # diffusion time scale in seconds (~11 hr)
# scaled versions of parameters (measure distance in diffusion lengthscales)
scaled_xstar = xstar*1e04/ell_diffusion
scaled_v0 = v0/vinf
scaled_gradient = gradient/vinf*1e-04*ell_diffusion # so that you can multiply it with something in diffusion length units 
scaled_L = L*1e04/ell_diffusion

speciess = ['bacteroides','eubacterium']
species_colors = ['r','b']
pretty_speciess = ['B. theta','E. rectale']

# Calculate flow rate in um/s as function of position (cm)
def calculate_flow(xs):
    
    vs = numpy.clip(v0-gradient*xs,vinf,100)
    return vs

# Calculate flow rate in units of v0 as function of position (in units of diffusion lengths)
def calculate_scaled_flow(ts):
    
    scaled_vs = numpy.clip(scaled_v0-scaled_gradient*ts,1,100)
    return scaled_vs
    
def calculate_integrated_exponents(xs): 
    # returns int_0^x v(x')/2D dx' for x measured in cm
    # used for converting between rho and u
    
    int_xstar = v0/2/D*xstar*1e04-gradient*xstar*xstar/4/D*1e04
    
    ints =  (v0/2/D*xs*1e04-gradient*xs*xs/4/D*1e04)*(xs<xstar)+(xs>=xstar)*(int_xstar+vinf*(xs-xstar)/2/D*1e04)

    return ints

def calculate_scaled_integrated_exponents(ts):
    # same as above, but for t measured in diffusion lengths
    
    return calculate_integrated_exponents(ts*ell_diffusion*1e-04)    

def parse_scraped_data():
    
    scraped_data = {}
    
    filename = "scraped_jonas_data/bacteroides_scraped_density.csv"
    xs = []
    ys = []
    file = open(filename,"r")
    for line in file:
        items = line.split(",")
        if len(items[0])==0:
            continue
        xs.append(float(items[0]))
        ys.append(float(items[1]))
    xs,ys = zip(*sorted(zip(xs,ys)))
        
    scraped_data['bacteroides_density'] = (numpy.array(xs),numpy.array(ys))
    
    filename = "scraped_jonas_data/eubacterium_scraped_density.csv"
    xs = []
    ys = []
    file = open(filename,"r")
    for line in file:
        items = line.split(",")
        if len(items[0])==0:
            continue
        
        xs.append(float(items[0]))
        ys.append(float(items[1]))
    xs,ys = zip(*sorted(zip(xs,ys)))
        
    scraped_data['eubacterium_density'] = (numpy.array(xs),numpy.array(ys))
    
    filename = "scraped_jonas_data/bacteroides_scraped_growth.csv"
    xs = []
    ys = []
    file = open(filename,"r")
    for line in file:
        items = line.split(",")
        if len(items[0])==0:
            continue
        
        xs.append(float(items[0]))
        ys.append(float(items[1]))
    xs,ys = zip(*sorted(zip(xs,ys)))
        
    scraped_data['bacteroides_growth'] = (numpy.array(xs),numpy.array(ys))
    
    
    filename = "scraped_jonas_data/eubacterium_scraped_growth.csv"
    xs = []
    ys = []
    file = open(filename,"r")
    for line in file:
        items = line.split(",")
        if len(items[0])==0:
            continue
        
        xs.append(float(items[0]))
        ys.append(float(items[1]))
    xs,ys = zip(*sorted(zip(xs,ys)))
        
    scraped_data['eubacterium_growth'] = (numpy.array(xs),numpy.array(ys))
    
    return scraped_data
    
def calculate_interpolation_function(xs,ys):
    spline = UnivariateSpline(xs, numpy.log(ys),s=0.005)
    return lambda x: numpy.exp(spline(x))

#def calculate_interpolation_function(xs,ys):
#    gam = LinearGAM().fit(xs, numpy.log(ys))
#    return lambda x: numpy.exp(gam.predict(x))


def calculate_monotonic_interpolation_function(xs,ys):
    gam = LinearGAM(s(0, constraints='monotonic_dec')).fit(xs, numpy.log(ys))
    return lambda x: numpy.exp(gam.predict(x))
    

def parse_smoothed_growth_rate(species):
    # return smoothed version of screenscraped growth rate
    # divisions per hour as function of x in cm
    
    scraped_data = parse_scraped_data()
    
    flam = calculate_monotonic_interpolation_function(scraped_data['%s_growth' % species][0],scraped_data['%s_growth' % species][1])
    
    return flam
    
def parse_scaled_growth_rate(species):
    # same as above, but measure in diffusion time as a function of diffusion lengths
    
    raw_growth_rate = parse_smoothed_growth_rate(species)
    return lambda t: D/vinf/vinf*raw_growth_rate(t*ell_diffusion*1e-04)/3600

def parse_smoothed_density(species):
    # return smoothed version of screenscraped density
    # units of 10^10 cells/ml as function of x in cm
    
    scraped_data = parse_scraped_data()
    
    frho = calculate_interpolation_function(scraped_data['%s_density' % species][0],scraped_data['%s_density' % species][1])
    
    return frho

def parse_scaled_density(species):
    # same as above but measure space as function of diffusion lengths
    raw_density = parse_smoothed_density(species)
    
    return lambda t: raw_density(t*ell_diffusion*1e-04)
    
    
if __name__=='__main__':
    
    import pylab
    
    
    for species in speciess:
        
        # first make figure in raw units (cm)
    
        xs = numpy.linspace(0,100,100)
    
        frho = parse_smoothed_density(species)
        rhos = frho(xs)
        us = rhos*numpy.exp(-2*calculate_integrated_exponents(xs))
        gs = us*rhos
        
        pylab.figure()
        
        pylab.plot(xs,rhos/rhos.max(),'-',label='rho(x)')
        pylab.plot(xs,us/us.max(),'-',label='u(x)')
        pylab.plot(xs,gs/gs.max(),'-',label='g(x)')
        pylab.xlabel('Position (cm)')
        pylab.legend(loc='lower right',frameon=False)
        pylab.savefig('%s_raw_profiles.pdf' % species, frameon=False)
        
        pylab.figure()
        # first make figure in raw units (cm)
    
        ts = numpy.linspace(0,scaled_xstar+10,100)
    
        scaled_frho = parse_scaled_density(species)
        
        scaled_rhos = scaled_frho(ts)
        
        scaled_us = scaled_rhos*numpy.exp(-2*calculate_scaled_integrated_exponents(ts))
        scaled_gs = scaled_us*scaled_rhos
        
        pylab.figure()
        
        pylab.plot(ts,scaled_rhos/scaled_rhos.max(),'-',label='rho(x)')
        pylab.plot(ts,scaled_us/scaled_us.max(),'-',label='u(x)')
        pylab.plot(ts,scaled_gs/scaled_gs.max(),'-',label='g(x)')
        pylab.xlabel('Position (diffusion lengths)')
        pylab.legend(loc='lower right',frameon=False)
        pylab.savefig('%s_raw_scaled_profiles.pdf' % species, frameon=False)
        