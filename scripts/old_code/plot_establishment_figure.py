import numpy
import pylab
import sys
import matplotlib as mpl

mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['legend.frameon']  = False
mpl.rcParams['legend.fontsize']  = 'small'


file = open("establishment_trajectory.txt","r")
file.readline()
established_ts = []
established_logns = []

for line in file:
    items = line.split()
    t = float(items[0])
    logn = float(items[1])
    
    established_ts.append(t)
    established_logns.append(logn)
    
established_ts = numpy.array(established_ts)
established_logns = numpy.array(established_logns)

file = open("extinct_trajectory.txt","r")
file.readline()
extinct_ts = []
extinct_logns = []

for line in file:
    items = line.split()
    t = float(items[0])
    logn = float(items[1])
    
    extinct_ts.append(t)
    extinct_logns.append(logn)
    
extinct_ts = numpy.array(extinct_ts)
extinct_logns = numpy.array(extinct_logns)

s = (established_logns[-1]-established_logns[-2])/(established_ts[-1]-established_ts[-2])*0.95

pylab.figure(1,figsize=(1.7,1))
pylab.plot(extinct_ts,extinct_logns,'r-',alpha=0.5)
pylab.plot(established_ts,established_logns,'r-')
pylab.plot([0.65,0.9],[0.62,0.62+s*0.25],'k:')
pylab.xlim([established_ts.max()*(-0.1),established_ts.max()])
pylab.ylim([0,established_logns.max()])
pylab.xticks([])
pylab.yticks([])
pylab.xlabel('Time, $t$')
pylab.ylabel('Log $f(t)$')
pylab.text(0.5,0.3,'$p_\\mathrm{fix} \\propto \\int \\lambda(x) w(x) \\rho(x)$')
#pylab.text(0.85,0.62,'$s_e$')
pylab.text(0.92,0.78,'$s_e$')

pylab.gca().spines['top'].set_visible(False)
pylab.gca().spines['right'].set_visible(False)

    

pylab.savefig('../figures/establishment_trajectory.pdf',bbox_inches='tight')
