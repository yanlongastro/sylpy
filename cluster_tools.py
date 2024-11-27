# some function useful to analyse globular clusters

import numpy as np
from scipy import interpolate



class cluster:
    def __init__(self, mass, position):
        self.mass = mass
        self.position = position
        self.total_mass = np.sum(mass)
        self.half_mass_radius_3d = half_mass_radius(mass, position, projection='3d')
        self.half_mass_radius_2d = half_mass_radius(mass, position, projection='2d')


def spherical_density_profile(mass, position, dN=100, cdf=False, projection='3d'):
    """
    A function that return the mass distribution
    """
    i = 0
    if position.ndim>1:
        if projection=='3d':
            radius = np.linalg.norm(position, axis=-1)
        if projection=='2d':
            radius = np.linalg.norm(position[:,:2], axis=-1)
    else:
        radius = position
    sort = np.argsort(radius)
    radius = radius[sort]
    mass = mass[sort]
    if cdf:
        return radius, np.cumsum(mass)

    r = []
    rho = []
    while i+dN<len(mass):
        dM = np.sum(mass[i:i+dN])
        r1 = radius[i+dN] 
        r0 = radius[i]
        if projection=='3d':
            dV = (r1**3-r0**3)*np.pi*4/3
        if projection=='2d':
            dV = (r1**2-r0**2)*np.pi
        rho_tmp = dM/dV
        r_tmp = (r0+r1)/2
        r.append(r_tmp)
        rho.append(rho_tmp)
        i += dN
    return np.array(r), np.array(rho)

def half_mass_radius(mass, position, projection='3d'):
    if projection=='3d':
        radius = np.linalg.norm(position, axis=-1)
    if projection=='2d':
        radius = np.linalg.norm(position[:,:2], axis=-1)
    sort = np.argsort(radius)
    radius = radius[sort]
    mass = mass[sort]
    cum_mass = np.cumsum(mass)
    r_m = interpolate.InterpolatedUnivariateSpline(cum_mass, radius)
    return r_m(cum_mass[-1]/2)