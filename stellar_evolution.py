from scipy import interpolate
import numpy as np
import os
package_dir = os.path.dirname(os.path.realpath(__file__))

def unpack_interpolation_table(table=None, file=None):
    if file is not None:
        table = np.loadtxt(file)
    y = table[0]
    with open(file) as f:
        x = f.readlines()[0].split()[1:]
        x = np.array([float(i) for i in x])
    z = table[1:]
    return x, y, z


class remnant_mass:
    """
    Calculate remnant mass for a given initial mass and metallicity.
    """
    def __init__(self):
        self.x, self.y, self.z = unpack_interpolation_table(file=package_dir+'/data/SperaMapelli2017.txt')
        #self.mrem = interpolate.RectBivariateSpline(self.x, self.y, self.z, kx=1, ky=1)
        self.mrem = lambda x, y: interpolate.interpn((self.x, self.y), self.z, np.array([x, y]), method='linear')[0]\
                                *(interpolate.interpn((self.x, self.y), self.z, np.array([x, y]), method='nearest')[0]>2)
    def __call__(self, mass, metallicity):
        return self.mrem(metallicity, mass)
        