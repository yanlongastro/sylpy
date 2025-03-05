"""
some functions to prepare and manage multiple simulations
"""

import subprocess
import glob
import os
from . import gizmo_analysis as ga
import re

def replace_in_lines(lines, keyword, new):
    """
    replace keywords with new ones in each line
    """
    lines_ = []
    for line in lines:
        line_ = line.replace(keyword, new)
        if not (keyword in line and len(new)==0): # if the new value is simply '', we delete the line
            lines_.append(line_)
    return lines_

def update_and_save(template, output, replace_dict):
    """
    convert the template into a new file by replacing keywords following replace_dict
    """
    with open(template, 'r') as f:
        lines = f.readlines()
        for k in replace_dict:
            lines = replace_in_lines(lines, k, str(replace_dict[k]))
        with open(output, 'w+') as f_:
            f_.writelines("%s" % l for l in lines)


def get_queue_job_ids_torque(username='yanlong'):
    """
    get running information from squeue of torque
    """
    cmd = "qstat -u %s | awk '{print $1}'"%username
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).stdout.decode('UTF-8').split('\n')
    ids = []
    for s in res:
        if bool(re.search(r'\d+', s)):
            ids.append(s)
    # print(ids)
    return ids

def get_all_my_job_info(show_headers=True, system='slurm', username='yanlong'):
    """
    get running information from squeue of slurm or torque
    """
    if system=='slurm':
        cmd = "squeue -u "+username+" --format='%.18i %.100j %.2t %.10M %.6D'"
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).stdout.decode('UTF-8').split('\n')
        runs = []
        for r in res:
            if len(r.split())>0:
                runs.append(r.split())
        if show_headers:
            print(runs[0])
        return runs[1:]
    if system=='torque':
        ids = get_queue_job_ids_torque(username)
        runs = []
        for i in ids:
            run = []
            run.append(i)
            for k in ['Job_Name', 'job_state', 'resources_used.walltime', 'Resource_List.nodect']:
                cmd = "qstat -f %s | awk -F= '/%s/ {print $2}' | xargs"%(i, k)
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).stdout.decode('UTF-8').split('\n')[0]
                run.append(res)
            runs.append(run)
        #print(runs)
        return runs


def get_latest_jobid_in_folder(folder, digits=8):
    """
    """
    folder += '/'
    slurms = glob.glob1(folder, 'slurm*')
    t_slurm = -1.0
    for slurm in slurms:
        if os.path.getctime(folder+slurm) > t_slurm:
            t_slurm = os.path.getctime(folder+slurm)
            jobid = slurm[-4-digits:-4]
    return jobid

def read_params(file):
    """
    read the paramter file for the simulation
    """
    res = {}
    with open(file) as f:
        lines = f.readlines()
    for line in lines:
        if line[0]=='%':
            continue
        ls = line.split()
        if len(ls)>=2:
            res[ls[0]] = ls[1]
            
    # convert strings to correct forms of values
    for k in res.keys():
        is_number = True
        try:
            float(res[k])
        except:
            is_number = False
        if is_number:
            if '.' in res[k] or 'e' in res[k]:
                res[k] = float(res[k])
            else:
                res[k] = int(res[k])
    return res


def get_job_name_from_batch(file, system='slurm'):
    with open(file) as f:
        strs = f.read().split()
        if system=='slurm':
            return strs[strs.index('-J')+1]
        if system=='torque':
            return strs[strs.index('-N')+1]

def get_job_status(folder='.', batch_name='submit.sh', system='slurm', username='yanlong'):
    """
    get current status of the job, return the status and the job id
    """
    job_name = get_job_name_from_batch(folder+'/'+batch_name, system=system)
    st = ''
    run_info = get_all_my_job_info(show_headers=False, system=system, username=username)
    for ii in run_info:
        if job_name==ii[1] or job_name+'+'==ii[1]:
            st = ii[2]
            break
    if 'R' in st:
        return 1, ii[0] # running
    if 'P' in st or 'Q' in st:
        return 0, ii[0] # pending
    if len(st)==0:
        return -1, None # stopped
    
def estimate_simulation_runtime(folder, diff=False, output_dir='output', snapshot_template='snapshot_%03d.hdf5', human=False):
    sim_folder = folder+'/'+output_dir
    n_snaps = ga.get_num_snaps(sim_folder)
    if n_snaps<=1:
        dt = 0.
    else:
        sp = sim_folder+'/'+snapshot_template
        snap0 = (0 if not diff else n_snaps-2)
        dt = os.path.getctime(sp%(n_snaps-1))-os.path.getctime(sp%(snap0))
    if not human:
        return dt
    
    if human:
        dd = dt//86400
        hh = (dt%86400)//3600
        mm = (dt%3600)//60
        ss = (dt%60)
        res = ''
        if dd>0:
            res += '%dd '%dd
        if hh>0:
            res += '%dh '%hh
        if mm>0:
            res += "%d' "%mm
        res += "%d\""%ss
        return res


def auto_resubmit_sims(sims, resubmit=False, batch_name='submit.sh', system='slurm'):
    """
    Display the status of the simulations and resubmit the stopped ones.
    :param sims: list of simulation directories
    :param resubmit: bool, whether to resubmit the stopped simulations
    :param batch_name: str, name of the batch file
    :param system: str, 'torque' or 'slurm'
    :return: None
    """
    i = 0
    n_active = 0
    cwd = os.getcwd()
    for sim in sims:
        num_snaps = ga.get_num_snaps(sim+'/output')
        st, jid = get_job_status(sim, batch_name=batch_name, system=system)
        params = read_params(sim+'/params.txt')
        num_snaps_max = params['TimeMax']/params['TimeBetSnapshot']
        runtime = estimate_simulation_runtime(sim, human=True)
        d_runtime = estimate_simulation_runtime(sim, human=True, diff=True)
        pp = ga.parse_path(sim)
        i += 1

        print('%2d)'%i, end=' ')
        print('%-60s'%sim, end='\t')
        print(num_snaps-1, end='\t')
        print('%-15s'%runtime, end='\t')
        print('%-15s'%d_runtime, end='\t')
        if st<0 and num_snaps>num_snaps_max:
            print("Done!")
            continue
        n_active += 1
        if st==1:
            print("R  %s"%jid)
        if st==0:
            print("PD %s"%jid)

        os.chdir(sim)
        # remove strange core.* files
        subprocess.run(["rm", "-f", "core.*"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if st==-1:
            print('Stopped ', end='')
            if resubmit: # in case of resubmit
                if system == 'torque':
                    res = subprocess.run(["qsub", "resubmit.cita"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if system == 'slurm':
                    res = subprocess.run(["sbatch", "resubmit.sh"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                jid_ = res.stdout.decode('UTF-8').split()[-1]
                print('> %s'%jid_)
            print('')
        os.chdir(cwd)
    print("** We have %d jobs in progress."%n_active)
    return n_active