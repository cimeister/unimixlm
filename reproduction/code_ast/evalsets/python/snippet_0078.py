from CoolProp.CoolProp import PropsSI
from scipy.interpolate import interp1d
from math import pi
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Specific Speed Chart Inputs
Ns = 0.6
Ds = 3.0
eff = 0.85

# Inlet Conditions
fluid = 'Air'
T1 = 20.+273.15  # Total temperature (K)
p1 = 101325.  # Total Pressure (Pa)

# Conditions to Vary
PRs = [2.0,4.0,6.0,8.0,10.0,12.0,14.0] # Pressure Ratio
RPMs = [3600,7200,14400] # RPM
#RPMs = [10000.,20000.,40000.,50000.] # RPM
#RPMs = [3600,7200,14400] # RPM

#===============================
# Prep
#===============================
g = 9.81 #m/s^2
g = 9.81

# Gas properties
CP = PropsSI('CPMASS',"T",T1,"P",p1,fluid)*1000.0 # KJ/Kg
CV = PropsSI('CVMASS',"T",T1,"P",p1,fluid)*1000.0 # KJ/Kg
kappa = CP / CV
MW = PropsSI('M',fluid)*1000.0 # kg/kmol
R_bar = PropsSI('GAS_CONSTANT',fluid) # kJ/kmol/K
R = R_bar / MW*1000.0  #  J/kg-K
D1 = PropsSI('H', 'T', T1, 'P', p1, fluid) # Density (kg/m3)


# Results
variables = ['RPM','PR','D','pwr','pwr_real','m_dot','V1','T2','p2']
df = pd.DataFrame(columns=variables)

for RPM in RPMs:
    for PR in PRs:
        # Compressor Calculations
    
        # Balje Calculations (Ideal gas)
        omega = 2*pi/60.0 * RPM # rad/s
        omega = RPM
        p3 = p1*PR # Static outlet pressure (Pa)
        H_ad = kappa/(kappa - 1.0)*R*T1*((p3/p1)**((kappa-1.0)/kappa) - 1.0) # kJ/kg
        V1 = (Ns*(H_ad)**(0.75)/omega)**2.0 # m3/s
        D = Ds*V1**0.5/(g*H_ad)**0.75    
        
        # Calculate Mass flow and power output
        m_dot = V1 * D1 # kg/s
        pwr = H_ad * m_dot / 1000.0  # MW
        
        # Compare with real gas calculations
        p2 = p1 * PR
        h1 = PropsSI('H', 'T', T1, 'P', p1, fluid)
        s1 = PropsSI('S', 'T', T1, 'P', p1, fluid)
        h2s = PropsSI('H', 'P', p2, 'S', s1, fluid)
        h2 = h1 + (h2s - h1) / eff
        T2 = PropsSI('T', 'P', p2, 'H', h2, fluid)
        pwr_real = m_dot * (h2 - h1) / 1E6 # MW
        
        print "Balje pwr:    " + str(pwr)
        print "Real-gas pwr: " + str(pwr_real)
        
        # Save Values    
        s = pd.Series(index=variables)
        
        s['RPM'] = RPM
        s['PR'] = PR
        s['D'] = D
        s['pwr'] = pwr
        s['pwr_real'] = pwr_real
        s['m_dot'] = m_dot
        s['V1'] = V1
        s['p2'] = p2
        s['T2'] = T2
        df = df.append(s,ignore_index=True)

    
# Plot Results
sns.set_style('darkgrid')
f,a = plt.subplots(4,1,sharex=True)
sns.lineplot(x='PR',y='D',hue='RPM',data=df,ax=a[0])
sns.lineplot(x='PR',y='pwr',hue='RPM',data=df,ax=a[1])
sns.lineplot(x='PR',y='pwr_real',hue='RPM',data=df,ax=a[2])
sns.lineplot(x='PR',y='m_dot',hue='RPM',data=df,ax=a[3])