"""
some functions to prepare and manage multiple simulations
"""

import subprocess
import glob
import os

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

def get_all_my_job_info(show_headers=True):
    """
    get running information from squeue of slurm
    """
    cmd = "squeue -u yanlong --format='%.18i %.100j %.2t %.10M %.6D'"
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).stdout.decode('UTF-8').split('\n')
    runs = []
    for r in res:
        if len(r.split())>0:
            runs.append(r.split())
    if show_headers:
        print(runs[0])
    return runs[1:]


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


def get_job_name_from_batch(file):
    with open(file) as f:
        strs = f.read().split()
        return strs[strs.index('-J')+1]

def get_job_status(folder='.', batch_name='submit.sh'):
    """
    get current status of the job, return the status and the job id
    """
    job_name = get_job_name_from_batch(folder+'/'+batch_name)
    st = ''
    run_info = get_all_my_job_info(show_headers=False)
    for ii in run_info:
        if job_name==ii[1]:
            st = ii[2]
    if 'R' in st:
        return 1, ii[0] # running
    if 'P' in st:
        return 0, ii[0] # pending
    if len(st)==0:
        return -1, None # stopped