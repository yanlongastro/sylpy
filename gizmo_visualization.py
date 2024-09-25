import numpy as np
from meshoid import Meshoid
from scipy.interpolate import LinearNDInterpolator
from scipy import stats
from . import gizmo_analysis as ga
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib as plt
import h5py

def read_snapshot(file):
    F = h5py.File(file,"r")
    rho = F["PartType0"]["Density"][:]
    density_cut = (rho*300 > 0.)
    pdata = {}
    for field in "Masses", "Coordinates", "SmoothingLength", "Velocities", "MagneticField", "Pressure", "ElectronAbundance", "SoundSpeed", "Density", "ParticleIDs":
        pdata[field] = F["PartType0"][field][:][density_cut]
    F.close()
    return pdata


def box_cut(pdata, center, halfbox, field='Masses', ord=np.inf, nroll=0):
    pos = pdata["Coordinates"]
    center = np.array(center)
    radius_cut = np.linalg.norm(pos-center, ord=ord, axis=1) <= halfbox
    pos, mass, hsml, v = pos[radius_cut], pdata[field][radius_cut], \
    pdata["SmoothingLength"][radius_cut], pdata["Velocities"][radius_cut]
    print("Number of gas particles:", len(pos))

    # nroll=0, (x, y, z)
    pos = np.roll(pos, nroll, axis=1)
    v = np.roll(v, nroll, axis=1)
    if mass.ndim==2:
        mass = np.roll(mass, nroll, axis=1)
    pos = np.roll(pos, nroll)

    return pos, mass, hsml, v


def create_meshoid_map(pos, mass, hsml, rmax, res=800, xc=np.array([0,0,0]), method='SurfaceDensity'):
    Me = Meshoid(pos, mass, hsml)
    X = Y = np.linspace(-rmax, rmax, res)
    X, Y = np.meshgrid(X, Y)
    res = getattr(Me, method)(Me.m,center=xc,size=rmax*2,res=res).T
    X += xc[0]
    Y += xc[1]
    return X, Y, res


def get_total_AM(m, pos, v, normalize=False):
    """
    Obtain the rotational axis of the body, like disk
    """
    am = np.sum(m[:,np.newaxis]*np.cross(pos, v), axis=0)
    if normalize:
        return am/np.linalg.norm(am)
    else:
        return am
    

import scipy
def rotate_vec(pos, axis, old_axes=[0, 0, 1]):
    old_axes = np.array(old_axes)
    axis = np.array(axis)
    axis /= np.linalg.norm(axis)
    theta = np.arccos(np.dot(axis, old_axes))
    #print(theta)
    rot_axis = np.cross(old_axes, axis)

    M = scipy.linalg.expm(np.cross(np.eye(3), rot_axis/np.linalg.norm(rot_axis)*theta))
    return np.dot(pos, M)

def pan_and_rotate(pdata, xc, axis):
    pdata['Coordinates'] -= np.array(xc)
    for field in "Coordinates", "Velocities", "MagneticField":
        pdata[field] = rotate_vec(pdata[field], axis)
    return pdata


def interpolate_vector_field(pos, vec, rmax, zcut, xc=np.array([0,0,0]), res=256, axes=[0, 1],):
    i1, i2 = axes[0], axes[1]
    i3 = list(set([0, 1, 2]) - set([i1, i2]))[0]

    X = Y = np.linspace(-rmax, rmax, res)
    X, Y = np.meshgrid(X, Y)
    X += xc[i1]
    Y += xc[i2]

    cut = (np.abs(pos[:,i3]<zcut))
    pos = pos[cut]
    vec = vec[cut]

    interp = LinearNDInterpolator(list(zip(pos[:,i1], pos[:,i2])), vec[:,i1])
    U = interp(X, Y)
    interp = LinearNDInterpolator(list(zip(pos[:,i1], pos[:,i2])), vec[:,i2])
    V = interp(X, Y)

    return X, Y, U, V


def project_cylindar_vectors(pos, v):
    pos_2d = pos[:,:-1]
    hat_r = pos_2d/np.linalg.norm(pos_2d, axis=-1)[:, np.newaxis]
    hat_r = np.insert(hat_r, 2, 0, axis=1)
    hat_theta = np.cross([0,0,1], hat_r)
    vel_r = np.sum(v*hat_r, axis=1)
    vel_t = np.sum(v*hat_theta, axis=1)
    return vel_r, vel_t, v[:,-1]

def project_spherical_vectors(pos, v):
    hat_r = pos/np.linalg.norm(pos, axis=-1)[:, np.newaxis]
    vel_r = np.sum(v*hat_r, axis=1)
    vel_t = np.sqrt(np.sum(v*v, axis=1) - vel_r**2)
    return vel_r, vel_t

# get radial dependence
def get_radial_dependence(prop, pos, zlim, dN=100, two_d=True, method='average'):
    zcut = (np.abs(pos[:,-1]) < zlim)
    prop = prop[zcut]
    pos = pos[zcut]
    if two_d:
        pos = pos[:,:-1]
    r = np.linalg.norm(pos, axis=-1)
    r_sort = np.argsort(r)
    prop = prop[r_sort]
    r = r[r_sort]

    ro = []
    po = []
    #print(len(r)//dN)
    for i in range(len(r)//dN):
        ptmp = prop[i*dN:(i+1)*dN]
        rtmp = r[i*dN:(i+1)*dN]
        if method=='average' or method=='mean':
            po.append(np.mean(ptmp))
        if method=='harmonic_mean':
            po.append(stats.hmean(ptmp))
        elif method=='surface_density':
            po.append(np.sum(ptmp)/(np.max(rtmp)**2-np.min(rtmp)**2)/np.pi)
        elif method=='std':
            po.append(np.std(ptmp))
        elif method=='median':
            po.append(np.median(ptmp))
        ro.append(np.mean(rtmp))

    return np.array(ro), np.array(po)

def get_mass_accretion(mass, vr, pos, zlim, dN=100, two_d=True):
    zcut = (np.abs(pos[:,-1]) < zlim)
    mass = mass[zcut]
    vr = vr[zcut]
    pos = pos[zcut]
    if two_d:
        pos = pos[:,:-1]
    r = np.linalg.norm(pos, axis=-1)
    r_sort = np.argsort(r)
    mass = mass[r_sort]
    vr = vr[r_sort]
    r = r[r_sort]

    ro = []
    po = []
    for i in range(len(r)//dN):
        mtmp = mass[i*dN:(i+1)*dN]
        rtmp = r[i*dN:(i+1)*dN]
        vrtmp = vr[i*dN:(i+1)*dN]
        mdot = -mtmp*vrtmp/(rtmp.max()-rtmp.min())
        po.append(np.sum(mdot))
        ro.append(np.mean(rtmp))

    return np.array(ro), np.array(po)

def add_sizebar(ax, size, label, color='w'):
    '''
    Add a size bar to the plot
    '''
    asb = AnchoredSizeBar(ax.transData,
                          size,
                          label,
                          loc='upper left',
                          pad=0.1, borderpad=0.5, sep=5, color=color,
                          frameon=False)
    ax.add_artist(asb)

def snapshot_visualization(fig, ax, filename, rmax, center=[0,0,0], 
                           cmap='inferno', vmin=None, vmax=None, bhids=[], show_time=True, freefall_time_in_sim_unit=None,
                           maxstars=1e10, force_aspect=True, show_sizebar=True, show_axes=False, message=None, axes_scale=1,
                           star_part_type='PartType4'):
    '''
    Make quick plot including gas, BHs, stars.
    '''
    if force_aspect:
        ax.set_aspect('equal')
    
    pdata = read_snapshot(filename)
    pos, mass, hsml, v = box_cut(pdata, np.array(center), rmax)
    X, Y, sdmap = create_meshoid_map(pos, mass, hsml, rmax, res=800, xc=np.array(center), method='SurfaceDensity')
    if vmin is not None:
        sdmap[sdmap<vmin] = vmin
    ax.pcolormesh(X, Y, sdmap, cmap=cmap, norm=colors.LogNorm(vmin=vmin, vmax=vmax), shading='auto')

    sp = ga.snapshot(filename)
    try:
        pos = sp.star('Coordinates', part_type=star_part_type)
        print('Number of stars:', len(pos))
        pos = pos[np.random.choice(np.arange(len(pos)), int(maxstars*np.tanh(len(pos)/maxstars)), replace=False)] # tweak this
        ax.scatter(pos[:,0], pos[:,1], c='lime', s=1)
    except:
        print('No stars')
        pass
    ax.set_xlim(-rmax+center[0], rmax+center[0])
    ax.set_ylim(-rmax+center[1], rmax+center[1])
    
    if len(bhids)>0:
        xx = sp.single_bh(bhids[0], 'Coordinates')
        ax.scatter(xx[0], xx[1], s=150, marker='*', c='k', edgecolors='none')
    try:
        xx = sp.single_bh(bhids[1], 'Coordinates')
        ax.scatter(xx[0], xx[1], s=90, marker='*', facecolors='none', edgecolors='k')
    except:
        pass

    # annotations
    if show_time:
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        if force_aspect:
            width = height = np.min([width, height])
        if freefall_time_in_sim_unit is None:
            txt = r'\textbf{%.3f\,Myr}'%(sp.time_in_yr/1e6)
        else:
            txt = r'\textbf{%.3f\,Myr (%.2f\, $\mathbf{t_{\rm ff}}$)}'%(sp.time_in_yr/1e6, sp.time/freefall_time_in_sim_unit)
        ax.annotate(txt, (height*axes_scale*72-6, height*axes_scale*72-12), 
                    xycoords='axes points', color='w', va='top', ha='right')
    if show_sizebar:
        add_sizebar(ax, rmax/2, r'\textbf{%d\,pc}'%(rmax*1000/2))

    if not show_axes:
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_facecolor((1.0, 0.0, 0.0, 0.0))
        plt.rcParams.update({
            "figure.facecolor":  (1.0, 0.0, 0.0, 0.0),  # red   with alpha = 30%
        })
    if message is not None:
        ax.annotate(message, (6, 6), 
                xycoords='axes points', color='w', va='bottom', ha='left')
    return X, Y, sdmap
