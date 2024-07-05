import pandas as pd
import numpy as np
import os

import sys
sys.path.append('../')
#from theoretical.analytical.channel 
import modules.channel.channelFlows as cf

import scipy.optimize as opt
import matplotlib.pyplot as plt

figDir = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figDir):
    os.mkdir(figDir)
    
def obtain_t_scale(s, percentage):
    '''Compute the relaxation time scale for startup flow with slip.
    
    Parameters:
    s : slip length ratio (slip length/ half the channel height)
    percentage : percentage to which the maximum velocity has been reached, e.g.
                percentage = 0.9 means that 90% of the maximum velocity have been 
                reached
    '''
    x_0 = 0.0
    dt = 0.001 # decrease for more accuracy
    N = 10
    t=0
    v_stat_max = cf.poiseuille_scaled(x_0,s) # maximum velocity in the center
    v_t_max = cf.navierSlip_scaled(t, x_0, s, N)
    
    # increase velocity until it is close enough to the maximum velocity
    while(abs(v_t_max/v_stat_max) < percentage):
        v_t_max = cf.navierSlip_scaled(t, x_0, s, N)
        t += dt
    return t

def lin_func(x, a, b):
    return a*x + b

# for this scaling, the pressure gradient G = -grad p 
# has to be scaled by G_scale = mu/(2*R*R)
# using G_scaled := G/G_scale
def charTimeScale(S, G_scaled):
    return S * G_scaled


##### 2nd cell
#percentages = [0.1,0.2,0.3,0.4, 0.5,0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
percentages = [0.9]
S = []
for i in range(-4,3):
    S.append(np.power(10,float(i)))
    #S.append(0.3*np.power(10,float(i)))
    S.append(0.5*np.power(10,float(i)))
S.insert(0,0.0)
S = sorted(S)

print("Percentages: " + str(percentages))
print("slip lenght ratio S: " + str(S))

# go through different scales
t_scales = {}
for p in percentages:
    t_scales[p] = {}
    for s in S:
        t_scales[p][s] = obtain_t_scale(s, p)
        print(str(s))


####3rd cell
fig, ax = plt.subplots(1, 1)
fig.set_dpi(180)
fig.set_tight_layout(True)

x_samp = list(S)
x_lin = np.linspace(0, max(x_samp), 3000)
ps = []
slopes = []
ordinate = []

# plot time scales for different percentages
for p, t_scales_p in t_scales.items():
    if p > 0.1:
        line, = ax.loglog(S, list(t_scales_p.values()), \
               ls = '', marker='o', markersize=4,  label="ANA " + str(int(100*p)) + '%')
        color = line.get_color()
        
        # linear function from least square interpolation
        y_samp = list(t_scales_p.values())
        p0 = [2, 0.15]                                      # guessed params
        w, _ = opt.curve_fit(lin_func, x_samp, y_samp, p0=p0)     
        print("Time: " + str(int(100*p)) + '%' + " a=" + str(round(w[0],2)) + " b=" + str(round(w[1],2)))
        y_model = lin_func(x_lin, *w)
        ax.loglog(x_lin, y_model, ls='--', color=color, label=None)
        ps.append(p)
        slopes.append(w[0])
        ordinate.append(w[1])
        
#ax.semilogx(x_lin, ordinate[-3]+charTimeScale(x_lin, 0.9*2.5), color='black', label='dim. ana.')
###
largeSlip = [element * (np.log(10)/2)  for element in S] #(np.log(10)/2)
# largeSlip = [element * (np.power(np.pi,2)/4)  for element in S] #(np.log(10)/2)
ax.loglog(S, largeSlip,ls = '', marker='o', markersize=4,color='red', label='' )
p0 = [2, 0.15]                                      # guessed params
largew, _ = opt.curve_fit(lin_func, x_samp, largeSlip, p0=p0)     
largeSlip_model = lin_func(x_lin, *largew)
ax.loglog(x_lin, largeSlip_model, ls='--', color='red', label=None)  
###
smallSlip = [(1+element) * ((4*np.log(10))/np.power(np.pi,2))  for element in S]
#smallSlip = [(1+element)  for element in S] #* ((2*np.log(10))/np.power(np.pi,2))
ax.loglog(S, smallSlip,ls = '', marker='o', markersize=4,color='green', label='' )
p0 = [2, 0.15]                                      # guessed params
smallw, _ = opt.curve_fit(lin_func, x_samp, smallSlip, p0=p0)     
smallSlip_model = lin_func(x_lin, *smallw)
ax.loglog(x_lin, smallSlip_model, ls='--', color='green', label=None)  
###
ax.grid()
ax.set_xlabel('$S$')
ax.set_ylabel('dimensionless time scale');
ax.set_xlim([min(S), max(S)])
#ax.set_ylim([0.8, 200])
ax.legend(loc="center left", ncol=1, columnspacing=0.4, handletextpad=0.2,bbox_to_anchor=[1.0,0.5])
plt.show()
#fig.savefig(figDir + "/charTimes.pdf")
