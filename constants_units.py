from . import manage_sims as ms
import numpy as np


G_cgs = 6.6743e-8
yr_cgs = 365.2422*24*3600
c_cgs = 299792458*100
pc_cgs = 3.08568e18
kB_cgs = 1.380649e-16
mp_cgs = 1.6726219e-24
Msun_cgs = 1.989e33
Rsun_cgs = 6.957e10

class units:
    def __init__(self, UnitMass_in_g=1.989e+43, UnitLength_in_cm=3.08568e+21, UnitVelocity_in_cm_per_s=1e5, UnitMagneticField_in_gauss=1,
                 param_file=None, snapshot_file=None):
        """
        param_file : if a gizmo run's parameter file is given, we derive units from it
        snapshot_file : derive the units from the snapshot's header
        """
        if param_file is not None:
            params = ms.read_params(param_file)
            for k in ['UnitMass_in_g', 'UnitLength_in_cm', 'UnitVelocity_in_cm_per_s', 'UnitMagneticField_in_gauss']:
                setattr(self, k, params[k])
        elif snapshot_file is not None:
            2333
        else:
            self.UnitMass_in_g = UnitMass_in_g
            self.UnitLength_in_cm = UnitLength_in_cm
            self.UnitVelocity_in_cm_per_s = UnitVelocity_in_cm_per_s
            self.UnitMagneticField_in_gauss = UnitMagneticField_in_gauss
        
        self.derive_units()
        self.derive_constants()
        
    def derive_units(self):
        self.UnitTime_in_s = self.UnitLength_in_cm/self.UnitVelocity_in_cm_per_s
        self.UnitTime_in_yr = self.UnitTime_in_s/yr_cgs
        self.UnitTime_in_Myr = self.UnitTime_in_yr/1e6
        self.UnitTime_in_Gyr = self.UnitTime_in_yr/1e9
        self.UnitMass_in_solar = self.UnitMass_in_g/Msun_cgs
        self.UnitLength_in_pc = self.UnitLength_in_cm/pc_cgs
        self.UnitVelocity_in_m_per_s = self.UnitVelocity_in_cm_per_s/100
    
    def derive_constants(self):
        self.G = G_cgs*self.UnitMass_in_g**(1) *self.UnitLength_in_cm**(-3) *self.UnitTime_in_s**(2)
        self.c = c_cgs/self.UnitVelocity_in_cm_per_s

    def free_fall_time(self, M, R, output_unit=None):
        tff =  np.pi/2 *np.sqrt(R**3/self.G/M/2)
        if output_unit is not None:
            tff *= getattr(self, 'UnitTime_in_'+output_unit)
        return tff
    
    def circular_period(self, M, R, output_unit=None):
        return np.sqrt(8)*2*self.free_fall_time(M, R, output_unit)

cgs = units(1, 1, 1, 1)
SI = units(1000, 100, 100, 1e4)
FIRE = units()
STARFORGE = units(Msun_cgs, pc_cgs, 1e2, 1e4)
star = units(Msun_cgs, Rsun_cgs, 100, 1)