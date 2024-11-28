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

    def mass_profile(self, dN=100, dr=None, dlogr=None, cdf=False, projection='3d'):
        return spherical_density_profile(self.mass, self.position, dN=dN, dr=dr, dlogr=dlogr, cdf=cdf, projection=projection)


def spherical_density_profile(mass, position, dN=100, dr=None, dlogr=None, cdf=False, projection='3d'):
    """
    For a given mass dist, return the mass profile

    Parameters
    ----------
    mass : array
    position : array
    dN : int
        Number of particles to include in each bin, if it's a PDF
    dr : float
        The mininmum width of each bin, which avoids noise
    dlogr : float
        The mininmum width of each bin in log scales
    cdf : bool
        Whether to show the cumulative distribution
    projection : string, '2d' or '3d'
        Projection that determines if this is a 3d or 2d distribution function
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
    step = dN
    while i+step<len(mass):
        step = dN
        if dr is not None: # make sure the curve is not too noisy in some regions
            while radius[i+step]-radius[i]<dr:
                step += 1
        if dlogr is not None:
            while np.log10(radius[i+step]/radius[i])<dlogr:
                step += 1
        dM = np.sum(mass[i:i+step])
        r1 = radius[i+step] 
        r0 = radius[i]
        if projection=='3d':
            dV = (r1**3-r0**3)*np.pi*4/3
        if projection=='2d':
            dV = (r1**2-r0**2)*np.pi
        rho_tmp = dM/dV
        r_tmp = (r0+r1)/2
        r.append(r_tmp)
        rho.append(rho_tmp)
        i += step
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