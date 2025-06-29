"""
Analysis code for gizmo, used in BH accretion project

yanlong@caltech.edu
"""


import numpy as np
import re
import h5py
from scipy import spatial
import pandas as pd
import glob
import os
from . import constants_units as cu
from . import manage_sims as ms


unit_time_in_yr = 206265*1000*1.5e8/(86400*365)
G = 4*np.pi**2/(206265000**3/1e10/unit_time_in_yr**2)
pc_in_cm = 3.086e+18
Msun_in_g = 2e33
Na = 6.02e23
mp_in_g = 1.6726e-24



def t_ff(M, R):
    """
    Free fall time: everything in code unit
    """
    return np.pi/2 *np.sqrt(R**3/G/M/2)

def v_esc(M, R):
    return np.sqrt(2*G*M/R)

def v_circ(M, R):
    return np.sqrt(G*M/R)

def bh_outflow_velocity(M, R, Mbh=1e-6):
    return np.sqrt(M/Mbh)*v_esc(M, R)


def show_info(fin):
    for k in list(fin.keys()):
        print(k)
        for atrs in list(fin[k].attrs.keys()):
            print(' ', atrs, '=', fin[k].attrs[atrs])
        for sk in list(fin[k].keys()):
            print(' ', end=' ')
            print(sk)
            if k == 'PartType5':
                print('  ', end=' ')
                print(fin[k][sk][()])
            

            
def parse_path(path, skip=0):
    """
    get numbers from a string like M1e8_R50
    """
    match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
    num_list = [float(x) for x in re.findall(match_number, path)]
    return num_list[skip:]

# Unit: Msun, pc, yr
def set_star_softening(M, Res):
    dM = M/Res**3
    drho = 2227.359223512722/Na
    drho /= Msun_in_g /pc_in_cm**3
    dr = (dM/drho)**(1/3)
    # print(dM, drho)
    return dr


# Unit: Msun, pc, yr
def set_bh_softening(M, R, Res, debug=False):
    if M<=1e6:
        dr_v = 1e3/M*R
    else:
        dr_v = 1e4/M*R
    if debug:
        print('dr_v =', dr_v)
    dr_star = set_star_softening(M, Res)
    if debug:
        print('dr_star =', dr_star)
    cs2 = 10000*8.31/0.001*5/3
    
    # Cooling: this is arbitrary
    cs2 /= 100
    
    # to km/s
    cs2 /= 1000**2
    dr_cs = G*100/1e10/cs2
    # to pc
    dr_cs *= 1000
    if debug:
        print('dr_cs =', dr_cs)
    dr = min(dr_star, dr_cs)
    if debug:
        if dr == dr_star:
            print('Used star softenning length.')
        if dr == dr_cs:
            print('Used sound speed limit.')
        if dr == dr_v:
            print('Used velocity limit.')
    #print(dr)
    return dr


def check_vel(m_bh, xyz_bh, vxyz_bh, m_gas, pos_gas, vel_gas, den):
    r = np.linalg.norm(xyz_bh-pos_gas)
    v = np.linalg.norm(vxyz_bh-vel_gas)
    m_add = 4*np.pi/3*r**3*den
    return (v**2<2*G*(m_bh+m_gas+m_add)/r)


def check_ang_mom(m_bh, xyz_bh, vxyz_bh, m_gas, pos_gas, vel_gas, sink_radius):
    dr = xyz_bh-pos_gas
    dv = vxyz_bh-vel_gas
    drdv = np.sum(dr*dv)
    r = np.linalg.norm(dr)
    v = np.linalg.norm(dv)
    spec_mom = (r*v)**2 - drdv**2
    return (spec_mom < G*(m_bh+m_gas)*sink_radius)


def check_boundedness(xyz_bh, pos_gas, sink_radius):
    r = np.linalg.norm(xyz_bh-pos_gas)
    return (r<sink_radius)

def set_job_name(ic, skip=0):
    if 'BH' in ic:
        job_name = 'B'
    else:
        job_name = 'M'

    par = parse_path(ic, skip=skip)
    M = par[0]
    R = par[1]
    Res = int(par[5])

    ind = int(np.log10(M))
    coe = int(M/10**ind)
    job_name += str(coe)+'%d'%ind

    ind = int(np.log10(R))
    coe = int(R/10**ind)
    job_name += str(coe)+str(ind)

    job_name += '%d'%(np.log(Res)/np.log(2))
    return job_name
 
def set_folder_name(M, R, Res, pre='BH_', turb_seed=42):
    power = int(np.log10(M))
    coeff = int(M/10**power)
    folder = '%sM%de%d_R%d_S0_T1_B0.01_Res%d_n2_sol0.5_%d/'%(pre, coeff, power, R, Res, turb_seed)
    return folder


def calculate_circular_velocity(center, ms, xs, zlim=np.inf):
    dx = xs-center
    zcut = (np.abs(dx[:,-1]) < zlim)
    ms = ms[zcut]
    dx = dx[zcut]
    dr = np.linalg.norm(dx, axis=-1)
    r_sort = np.argsort(dr)
    ms = ms[r_sort]
    dr = dr[r_sort]
    cum_ms = np.cumsum(ms)
    vc = np.sqrt(G*cum_ms/dr)
    return dr, vc

class snapshot:
    def __init__(self, file, showinfo=False):
        self.f =  h5py.File(file, 'r')
        try:
            self.gas_number = self.f['Header'].attrs['NumPart_ThisFile'][0]
            self.star_number = self.f['Header'].attrs['NumPart_ThisFile'][4]
            self.bh_number = self.f['Header'].attrs['NumPart_ThisFile'][5]
            self.time = self.f['Header'].attrs['Time']
            self.UnitLength_In_CGS = self.f['Header'].attrs['UnitLength_In_CGS']
            self.UnitTime_In_CGS = self.f['Header'].attrs['UnitLength_In_CGS']/self.f['Header'].attrs['UnitVelocity_In_CGS']
            self.UnitTime_In_Yr = self.UnitTime_In_CGS/(86400*365)
            self.UnitMass_In_CGS = self.f['Header'].attrs['UnitMass_In_CGS']
            self.UnitDensity_In_CGS = self.f['Header'].attrs['UnitMass_In_CGS']/self.f['Header'].attrs['UnitLength_In_CGS']**3
            self.UnitVelocity_In_CGS = self.f['Header'].attrs['UnitVelocity_In_CGS']
            self.time_in_yr = self.time*self.UnitTime_In_CGS/(86400*365)
            self.bh_sink_radius = self.f['Header'].attrs['Fixed_ForceSoftening_Keplerian_Kernel_Extent'][5]/2.8
        except:
            pass
        
        if showinfo == True:
            show_info(self.f)
    
    def close(self):
        self.f.close()
        
    def gas(self, attr, partial=None, near_bhid=None):
        res = self.f['PartType0'][attr][()]
        if near_bhid is not None:
            partial = self.find_gas_near_bh(near_bhid)
        if partial is None:
            return res
        else:
            return res[partial]
    
    def star(self, attr, partial=[], part_type='PartType4'):
        if part_type in list(self.f.keys()):
            if partial == []:
                return self.f[part_type][attr][()]
            else:
                return self.f[part_type][attr][()][partial]
        else:
            return np.array([])
        
    def bh(self, attr):
        return self.f['PartType5'][attr][()]
    
    def bh_sorted(self, attr, stellar_stage=None):
        ids = self.f['PartType5']['ParticleIDs'][()]
        target = self.f['PartType5'][attr][()]
        if stellar_stage is not None:
            stage_filter = self.f['PartType5']['ProtoStellarStage'][()]==stellar_stage
            ids = ids[stage_filter]
            target = target[stage_filter]
        a = target[ids.argsort()]
        return a
    
    def single_bh(self, bhid, attr, ):
        bhpid = np.where(self.f['PartType5']['ParticleIDs'][()]==bhid)
        if len(bhpid)>1:
            print("More than one particles have the same ID, should be an issue!")
        bhpid = bhpid[0][0]
        return self.f['PartType5'][attr][()][bhpid]

    def single_particle(self, pid, part_type, attr):
        bhpid = np.where(self.f[part_type]['ParticleIDs'][()]==pid)
        if len(bhpid)>1:
            print("More than one particles have the same ID, should be an issue!")
        bhpid = bhpid[0][0]
        return self.f[part_type][attr][()][bhpid]
            
    def find_gas_near_bh(self, bhid=1, kneighbor=96, drmax=10086, p_norm=2, center=None):
        if center is None:
            pos_bh = self.single_bh(bhid, 'Coordinates')
        else:
            pos_bh = center
        pos_gas = self.gas('Coordinates')
        kdtree = spatial.cKDTree(pos_gas)
        dist, inds = kdtree.query(pos_bh, k=kneighbor, eps=0, distance_upper_bound=drmax, p=p_norm)
        if kneighbor==1:
            dist = np.array([dist])
            inds = np.array([inds])
        return inds[dist!=np.inf]
    
    def find_star_near_bh(self, bhid, kneighbor=96, drmax=10086, p_norm=2):
        pos_bh = self.single_bh(bhid, 'Coordinates')
        pos_gas = self.star('Coordinates')
        kdtree = spatial.cKDTree(pos_gas)
        dist, inds = kdtree.query(pos_bh, k=kneighbor, eps=0, distance_upper_bound=drmax, p=p_norm)
        if kneighbor==1:
            dist = np.array([dist])
            inds = np.array([inds])
        return inds[dist!=np.inf]
    
    def multi_type_particles(self, types, attr):
        """
        types: a list of strings, e.g., ['PartType0', 'PartType4']
        attr: the attribute to be returned
        """
        res = []
        for type in types:
            part_type = "PartType"+str(type)
            if part_type in list(self.f.keys()):
                res.append(self.f[part_type][attr][()])
            # else:
            #     res.append(np.array([]))
        return np.concatenate(res)
    
    
    
def get_num_snaps(path, snap='snapshot_*.hdf5', timed=True):
    fns = glob.glob1(path, snap)
    if len(fns)<=1:
        return len(fns)
    imax = 0
    tmax = 0.
    if not timed:
        return len(fns)
    for i in range(1, len(fns)):
        file = os.path.join(path, 'snapshot_%03d.hdf5'%i)
        if not os.path.exists(file):
            break
        t = os.path.getmtime(file)
        if t>=tmax:
            imax += 1
            tmax = t
        else:
            break
            
    return imax+1
        

def pass_row_header(fname):
    """
    delete 'BH=' in every line
    """
    with open(fname, 'r') as fin:
        if 'BH=' in fin.readline():
            for line in fin:
                try:
                    yield line[3:]
                except IndexError:
                    continue
        else:
            for line in fin:
                try:
                    yield line
                except IndexError:
                    continue

        
        
class blackhole_details:
    def __init__(self, outputdir, filename='blackhole_details', tasks=72, io_reduced_mode=False):
        self.io_reduced_mode = io_reduced_mode
        df = None
        for task in range(tasks):
            bhdetail = outputdir+'blackhole_details/'+filename+'_%d.txt'%task
            if io_reduced_mode:
                data = np.loadtxt(pass_row_header(bhdetail))
                sort_key = 1
            else:
                data = np.loadtxt(bhdetail)
                sort_key = 0
            pdf = pd.DataFrame(data=data)
            df = pd.concat([df, pdf])
            print(task, end=' ')
        df = df.sort_values(by=[sort_key])
        self.df = df
    def get_detail(self, bhpid, column):
        if self.io_reduced_mode:
            mask_key = 0
        else:
            mask_key = 1
        mask = self.df[mask_key] == bhpid
        res = self.df[mask][column]
        return res.values
    
    
    
def read_blackhole_details(folder, filename='blackhole_details', tasks=72, sort_key=0):
    df = None
    for task in range(tasks):
        print(task, end=' ')
        bhdetail = folder+'blackhole_details/'+filename+'_%d.txt'%task
        data = np.loadtxt(bhdetail)
        if data.ndim == 1:
            if len(data) == 0:
                continue
            else:
                data = np.array([data])
        pdf = pd.DataFrame(data=data)
        df = pd.concat([df, pdf])
    df = df.sort_values(by=[sort_key])
    return df

    
    
    
    
class simulation:
    """
    A series of snapshots in a single simulation.
    """
    def __init__(self, folder, output='output', params_file='params.txt', timed=True):
        self.sim_folder = folder
        self.output_folder = folder+'/'+output
        self.timed = timed
        self.snapshot_file = self.output_folder+'/snapshot_%03d.hdf5'
        self.png_folders = []
        self.params_file = folder+'/'+params_file
        self.params = ms.read_params(self.params_file)
        self.units = cu.units(param_file=self.params_file)
        self.refresh()
    
    def refresh(self):
        self.last = get_num_snaps(self.output_folder, timed=self.timed)-1
        print(self.output_folder, self.last)
        
    def snapshot(self, i):
        return snapshot(self.snapshot_file%i)
        
    def find_interesting_BHs(self, num=5, first=0, last=None, sort_by_ratio=False):
        """
        find particle ids of BHs with the most significantly BH accretion
        if the first snapshot number is not given, we simply check the final mass
        """
        if last is None:
            last = self.last
        sp1 = snapshot(self.snapshot_file%last)
        if first is not None:
            sp0 = snapshot(self.snapshot_file%first)
            if sort_by_ratio:
                dm = sp1.bh_sorted('Masses')/sp0.bh_sorted('Masses')
            else:
                dm = sp1.bh_sorted('Masses')-sp0.bh_sorted('Masses')
        else:
            dm = sp1.bh_sorted('Masses')
            dm *= (sp1.bh_sorted('ProtoStellarStage')==7)
        return sp1.bh_sorted('ParticleIDs')[(-dm).argsort()][:num]
            
    
    def find_fastest_growth_snapshot(self, num=5, bhid=None, sort_by_ratio=False):
        ms = []
        for i in range(0, self.last+1):
            sp = snapshot(self.snapshot_file%i)
            if bhid == None:
                ms.append(np.sum(sp.bh('Masses')))
            else:
                ms.append(sp.single_bh(bhid, 'Masses'))
        ms = np.array(ms)
        dm = []
        for i in range(self.last):
            if sort_by_ratio:
                dm.append(ms[i+1]/ms[i])
            else:
                dm.append(ms[i+1]-ms[i])
        dm = np.array(dm)
        return (-dm).argsort()[:num]+1
    
    def find_fastest_growth_bh(self, spid, bhs):
        sp0 = snapshot(self.snapshot_file%(spid-1))
        sp = snapshot(self.snapshot_file%spid)
        dbh = []
        for bh in bhs:
            dbh.append(sp.single_bh(bh, 'Masses') - sp0.single_bh(bh, 'Masses'))
        return bhs[np.argmax(dbh)]
    
    def get_bh_history(self, bhid=None, attr='Masses', method='', diff=None, stellar_stage=None):
        age = []
        history = []
        if diff is not None:
            sp0 = snapshot(self.snapshot_file%diff)
        for i in range(self.last+1):
            sp = snapshot(self.snapshot_file%i)
            try:
                if bhid is not None:
                    temp = sp.single_bh(bhid, attr, )
                    if diff is not None:
                        temp -= sp0.single_bh(bhid, attr, )
                else:
                    temp = sp.bh_sorted(attr, stellar_stage=stellar_stage)
                    if diff is not None:
                        temp -= sp0.bh_sorted(attr, stellar_stage=stellar_stage)
            except:
                continue
            
            if 'norm' in method:
                temp = np.linalg.norm(temp, axis=-1)
            if 'log' in method:
                temp = np.log10(temp)
            
            if 'sum' in method:
                temp = np.sum(temp)
            if 'inverse_sum' in method:
                temp = np.sum(1/temp)
            if 'average' in method:
                temp = np.mean(temp)
            if 'stdev' in method:
                temp = np.std(temp)    
            if 'percentile' in method:
                # e.g., must be percentile_0.95
                pct = float(method[11:])
                temp = np.sort(temp)[int(len(temp)*pct)]
            
            history.append(temp)
            age.append(sp.time)
        return np.array(age), np.array(history)
    
    def get_sf_history(self, attr='Masses'):
        age = []
        history = []
        for i in range(self.last+1):
            sp = snapshot(self.snapshot_file%i)
            age.append(sp.time)
            history.append(np.sum(sp.star(attr)))
        return np.array(age), np.array(history)
    
    def get_bh_evolution_track(self, stellar_stage=None):
        res = {}
        res['IDs'] = []
        for sid in range(self.last+1):
            sp = self.snapshot(sid)
            if 'PartType5' in sp.f.keys():
                for pid in sp.f['PartType5']['ParticleIDs'][()]:
                    if stellar_stage is not None:
                        if sp.f['PartType5']['ProtoStellarStage'][()][sp.f['PartType5']['ParticleIDs'][()]==pid][0] != stellar_stage:
                            continue
                    if pid not in res['IDs']:
                        res[str(pid)] = [sid]
                        res['IDs'].append(pid)
                    else:
                        res[str(pid)].append(sid)
        self.bh_tracks = res
        return res
    
    def get_bh_mass_ratio(self, i=0, j=None, attr='BH_Mass'):
        if j is None:
            j = self.last
        si = self.snapshot(i)
        sj = self.snapshot(j)
        mr = sj.bh_sorted(attr)/si.bh_sorted(attr)
        mr.sort()
        return (mr, 1-np.array(range(0, len(mr)))/len(mr))
    
    def get_bh_mass_diff(self, i=0, j=None, attr='BH_Mass'):
        if j is None:
            j = self.last
        si = self.snapshot(i)
        sj = self.snapshot(j)
        mr = sj.bh_sorted(attr)-si.bh_sorted(attr)
        mr.sort()
        return (mr*1e10, 1-np.array(range(0, len(mr)))/len(mr))
    
    def get_gas_history(self, attr='Masses', method='sum', percentiles=[16, 50, 84], crit_density=None, radius_cut=None, ids=None):
        # radius_cut: only return gas out side the radius, in code unit
        age = []
        history = []
        for i in range(self.last+1):
            sp = snapshot(self.snapshot_file%i)
            age.append(sp.time)

            if 'Volume' in attr:
                temp = sp.gas('Masses')/sp.gas('Density')
            elif 'SoundSpeed' in attr:
                temp = np.sqrt(10./9.*sp.gas('InternalEnergy'))
            else:
                temp = sp.gas(attr)
                
            if crit_density is not None:
                temp = temp[sp.gas('Density')>crit_density]

            if radius_cut is not None:
                cut = np.linalg.norm(sp.gas('Coordinates'), axis=-1)>radius_cut
                temp = temp[cut]

            if ids is not None:
                sorter = np.argsort(sp.gas('ParticleIDs'))
                cut = sorter[np.searchsorted(sp.gas('ParticleIDs'), ids, sorter=sorter)]
                temp = temp[cut]
                
            if 'norm' in method:
                temp = np.linalg.norm(temp, axis=1)
            if 'log' in method:
                temp = np.log10(temp)
            
            if 'sum' == method:
                temp = np.sum(temp, axis=0)
            if 'inverse_sum' == method:
                temp = np.sum(1/temp, axis=0)
            if 'average' in method or 'mean' in method:
                temp = np.mean(temp, axis=0)
            if 'stdev' in method:
                temp = np.std(temp, axis=0)
            if 'percentile' in method:
                temp = np.percentile(temp, percentiles, axis=0)
            history.append(temp)
        return np.array(age), np.array(history)
    
    def prepare_png_folder(self, png_folder='pngs'):
        """
        Prepare a folder for png files
        """
        if png_folder not in self.png_folders:
            self.png_folders.append(png_folder)
        png_folder = self.output_folder+"/%s/"%png_folder
        png_file = png_folder+'/snapshot_%03d.png'
        os.makedirs(png_folder, exist_ok=True)

        try:
            cleanup = os.path.getmtime(self.snapshot_file%0)>os.path.getmtime(png_file%0)
        except:
            cleanup = False
        overwrite = cleanup

        if cleanup:
            existings = glob.glob(png_folder+'/*')
            for f in existings:
                os.remove(f)
        return overwrite, png_file
    
    def make_movie(self, png_folder='pngs'):
        png_folder = self.output_folder+"/%s/"%png_folder
        png_file = png_folder+'/snapshot_%03d.png'
        mp4_output = png_folder+'/movie.mp4'
        os.system("ffmpeg -y -framerate 20 -i %s -c:v libx264 -pix_fmt yuv420p -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' %s"%(png_file, mp4_output))