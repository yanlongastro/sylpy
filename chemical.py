import numpy as np


def logno(x):
    return np.poly1d([ 0.11873892,  -2.47889661,  17.21350428, -41.20972519])(x)

def logco(x):
    return np.poly1d([ 0.039096,   -0.60627594,  2.55355594, -2.35671008])(x)

def logco_scatter(x):
    return np.poly1d([0.13408303,  -2.62789306,  16.75565625, -35.46541475])(x)

def get_abundance(fire_abundance_solar, z_in_solar, co_model='normal'):
    HYDROGEN_MASSFRAC = 0.76
    HYDROGEN_MASSFRAC = 1-np.sum(fire_abundance_solar[:2])
    res = np.array(fire_abundance_solar)*z_in_solar
    x = np.log10(res[4]/16/HYDROGEN_MASSFRAC)+12 # assume Z propto O/H
    #print(x)
    if co_model=='normal':
        co = 10**(logco(x))*12/16
    if co_model=='scatter':
        co = 10**(logco_scatter(x))*12/16
    no = 10**(logno(x))*14/16
    res[2] = res[4]*co
    res[3] = res[4]*no
    
    res[0] = np.sum(res[2:])
    res[2:] *= fire_abundance_solar[0]*z_in_solar/res[0]
    res[0] = np.sum(res[2:])
    #print(fire_abundance_solar[0]*z_in_solar/np.sum(res[2:]))
    res[1] = (1-res[0])*fire_abundance_solar[1]/(1-fire_abundance_solar[0])
    return res

abundance_solar_fire = np.array([1.39897e-02, 2.70300e-01, 2.53000e-03, 7.41000e-04, 6.13000e-03,
        1.34000e-03, 7.57000e-04, 7.12000e-04, 3.31000e-04, 6.87000e-05,
        1.38000e-03])
elements_fire = ['H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ca', 'Fe']