from scipy import interpolate, integrate
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


def kroupa_imf(m):
    m0 = 0.08
    m1 = 0.5
    m2 = 1
    
    s0 = -0.3
    s1 = -1.3
    s2 = -2.3
    s3 = -2.3
    
    res = (m<=m0)*(m/m0)**s0
    res += (m>m0)*(m<=m1)*(m/m0)**s1
    res += (m>m1)*(m<=m2)*(m/m1)**s2   *(m1/m0)**s1
    res += (m>m2)*(m/m2)**s3   *(m1/m0)**s1   *(m2/m1)**s2
    
    return res

def chabrier_imf(m):
    r1 = np.exp(-np.log10(m/0.08)**2)
    r2 = m**(-2.3) * np.exp(-np.log10(1/0.08)**2)
    return r1*(m<=1) + r2*(m>1)

class imf:
    def __init__(self, imf_function, m_min=0.01, m_max=300):
        """
        imf_function : callable
            IMF, doesn't need to be normalized
        """
        
        x = np.linspace(m_min, m_max, num=5_00_000)
        y = imf_function(x)
        norm = integrate.simpson(y, x)
        y /= norm
        
        self.imf = interpolate.InterpolatedUnivariateSpline(x, y)
        yc = np.cumsum(y)
        yc -= yc[0]
        yc /= yc[-1]
        self.mean_mass = integrate.simpson(y*x, x)/integrate.simpson(y, x)
        self.imf_cdf = interpolate.InterpolatedUnivariateSpline(x, yc)
        self.imf_cdf_inv = interpolate.InterpolatedUnivariateSpline(yc, x)
        self.m_min = m_min
        self.m_max = m_max
    
    def sample_a_star(self):
        x = np.random.rand()
        return self.imf_cdf_inv(x)
    
    def sample_stars_of_total_mass(self, mtot):
        m = 0
        res = []
        while m<mtot:
            dm = self.sample_a_star()
            res.append(dm)
            m += dm
        return np.array(res)
    
    def imf_integral(self, m0, m1, moment=0):
        m0 = max(m0, self.m_min)
        m0 = min(m0, self.m_max)
        m1 = min(m1, self.m_max)
        assert m0<=m1, "m0 must be less than m1"
        func = lambda x: self.imf(x)*x**moment
        return integrate.quad(func, m0, m1)[0]
    
def integral_power_law(x0, x1, c, s):
    return (s==-1)*c*np.log(x1/x0) + (s!=-1)*c*(x1**(s+1)-x0**(s+1))/(s+1+1e-100)


def kroupa_imf_normalized(m, m_min=0.01, m_max=300, slopes=[-0.3, -1.3, -2.3, -2.3], cdf=False, mass_weighted=False):
    m0 = 0.08
    m1 = 0.5
    m2 = 1
    
    s0 = slopes[0]
    s1 = slopes[1]
    s2 = slopes[2]
    s3 = slopes[3]
    
    c0 = 1.
    c1 = c0 *m0**(s0-s1)
    c2 = c1 *m1**(s1-s2)
    c3 = c2 *m2**(s2-s3)
    
    norm = integral_power_law(m_min, m0, c0, s0)
    norm += integral_power_law(m0, m1, c1, s1)
    norm += integral_power_law(m1, m2, c2, s2)
    norm += integral_power_law(m2, m_max, c3, s3)
    
    c0 /= norm
    c1 /= norm
    c2 /= norm
    c3 /= norm
    
    if mass_weighted:
        s0 += 1
        s1 += 1
        s2 += 1
        s3 += 1
    
    if not cdf:
        res = (m<=m0)*c0*m**s0
        res += (m>m0)*(m<=m1)*c1*m**s1
        res += (m>m1)*(m<=m2)*c2*m**s2
        res += (m>m2)*c3*m**s3
    else:
        res = integral_power_law(m_min, min(m0, m), c0, s0)
        res += (m>m0)*integral_power_law(m0, min(m1, m), c1, s1)
        res += (m>m1)*integral_power_law(m1, min(m2, m), c2, s2)
        res += (m>m2)*integral_power_law(m2, min(m_max, m), c3, s3)
    return res


def kroupa_imf_weighted_luminosity(m_cut, m_min=0.01, m_max=300, slopes=[-0.3, -1.3, -2.3, -2.3], debug=False):
    m0 = 0.08
    m1 = 0.5
    m2 = 1
    
    s0 = slopes[0]
    s1 = slopes[1]
    s2 = slopes[2]
    s3 = slopes[3]
    
    c0 = 1.
    c1 = c0 *m0**(s0-s1)
    c2 = c1 *m1**(s1-s2)
    c3 = c2 *m2**(s2-s3)
    
    norm = integral_power_law(m_min, m0, c0, s0)
    norm += integral_power_law(m0, m1, c1, s1)
    norm += integral_power_law(m1, m2, c2, s2)
    norm += integral_power_law(m2, m_max, c3, s3)
    
    c0 /= norm
    c1 /= norm
    c2 /= norm
    c3 /= norm
    
    mx0 = 0.012
    mx1 = 0.43
    mx2 = 2
    mx3 = 53.9
    
    cx0 = 0
    cx1 = 0.185
    cx2 = 1
    cx3 = 1.5
    cx4 = 32000
    
    sx0 = 0
    sx1 = 2
    sx2 = 4
    sx3 = 3.5
    sx4 = 1
    
    m_list = [m_min,     m0,     mx0,      mx1,      m1,      m2,      mx2,     mx3,     m_max]
    c_list =      [c0*cx0,  c1*cx0,  c1*cx1,  c1*cx2,   c2*cx2,  c3*cx2,  c3*cx3,  c3*cx4]
    s_list =      [s0+sx0,  s1+sx0,  s1+sx1,  s1+sx2,   s2+sx2,  s3+sx2,  s3+sx3,  s3+sx4]
    
    for i in range(8):
        if(m_list[i+1]>=m_cut):
            m_list[i+1] = m_cut
            #print(i)
            break
    for j in range(i+1, 8):
        c_list[j] = 0.

    res = 0.
    
    for i in range(8):
        if debug:
            print(i, m_list[i], m_list[i+1], c_list[i], s_list[i])
        res += integral_power_law(m_list[i], m_list[i+1], c_list[i], s_list[i])
    return res

def kroupa_light_to_mass_ratio(m_cut, m_min=0.001, m_max=300, slopes=[-0.3, -1.3, -2.3, -2.3]):
    return kroupa_imf_weighted_luminosity(m_cut, m_min, m_max, slopes)/kroupa_imf_normalized(m_cut, m_min, m_max, slopes, cdf=True, mass_weighted=True)

def fire_light_mass_ratio(stellar_age_in_gyr, z_in_solar=1):
    # fire-3: 2203.00040, but this is fire-2
    t1=0.0012
    t2=0.0037
    f1=800.
    f2=1100.*pow(z_in_solar, -0.1)
    tx=np.log10(stellar_age_in_gyr/t2)
    t_g=np.log10(stellar_age_in_gyr/1.2)/0.05
    if(stellar_age_in_gyr<=t1):
        lum=f1
    elif (stellar_age_in_gyr<=t2):
        lum=f1*pow(stellar_age_in_gyr/t1,np.log(f2/f1)/np.log(t2/t1))
    else:
        lum=f2*pow(10.,-1.82*tx+0.42*tx*tx-0.07*tx*tx*tx)*(1.+1.2*np.exp(-0.5*t_g*t_g))
    return lum

def stellar_luminosity_fire(m):
    res = (m<=0.012)*0.
    res += (m>0.012)*(m<=0.43) *0.185*m**2
    res += (m>0.43)*(m<=2) *m**4
    res += (m>2)*(m<=53.9) *1.5*m**3.5
    res += (m>53.9) *32000*m
    return res

def single_star_light_to_mass_ratio(m):
    return stellar_luminosity_fire(m)/m
    


tout_lum_mat = np.loadtxt(package_dir+"/data/tout_fitting_lum")
tout_rad_mat = np.loadtxt(package_dir+"/data/tout_fitting_rad")

def stellar_luminosity_tout96(m, z):
    cs = np.zeros(7)
    lgz = np.log10(z/0.02)
    for i in range(7):
        r = tout_lum_mat[i]
        for j in range(5):
            cs[i] += r[j]*lgz**j
    return (cs[0]*m**5.5 + cs[1]*m**11)/(cs[2] +m**3 +cs[3]*m**5 +cs[4]*m**7 +cs[5]*m**8 +cs[6]*m**9.5)
    
def stellar_radius_tout96(m, z):
    cs = np.zeros(9)
    lgz = np.log10(z/0.02)
    for i in range(9):
        r = tout_rad_mat[i]
        for j in range(5):
            cs[i] += r[j]*lgz**j
    return (cs[0]*m**2.5 + cs[1]*m**6.5 + cs[2]*m**11 +cs[3]*m**19 +cs[4]*m**19.5) \
            /(cs[5] +cs[6]*m**2 +cs[7]*m**8.5 +m**18.5 +cs[8]*m**19.5)


def measure_mass_function(m, nbins=21, mmin=0.01, mmax=100):
    logm = np.log(m)
    bins = np.log(np.logspace(np.log10(mmin), np.log10(mmax), num=nbins))
    val, bins = np.histogram(logm, bins=bins, density=True)
    logmc = (bins[1:]+bins[:-1])/2
    mc = np.exp(logmc)
    return mc, val/mc