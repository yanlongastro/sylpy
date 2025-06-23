import numpy as np

def get_radial_profile(value, position, component=None, method='density', percentiles=None, dN=100, dr=None, dlogr=0.05, cdf=False,):
    """
    """
    i = 0
    if position.ndim>1:
        radius = np.linalg.norm(position[:,:2], axis=-1)
    else:
        radius = position
    sort = np.argsort(radius)
    radius = radius[sort]
    value = value[sort]
    position = position[sort]
    if cdf:
        return radius, np.cumsum(value)
    
    if component is not None:
        assert value.ndim>0, "Value must be >1 in ndim"
        if component in [0, 1, 2]:
            value = value[:, component]
        if component=='radial':
            vec = position[:,:2]/radius[:,None]
            value = np.sum(vec*value[:,:2], axis=-1)
        if component=='toroidal':
            vec = np.stack((-position[:, 1], position[:, 0])).T/radius[:,None]
            value = np.sum(vec*value[:,:2], axis=-1)
        if component=='poloidal':
            value = value[:, 2]

    r = []
    res = []
    step = dN
    while i+step<len(value):
        step = dN
        if dr is not None: # make sure the curve is not too noisy in some regions
            while radius[i+step]-radius[i]<dr:
                step += 1
                if step>=len(value)-i:
                    break
        if dlogr is not None:
            while np.log10(radius[i+step]/radius[i])<dlogr:
                step += 1
                if step>=len(value)-i:
                    break
        step = min(step, len(value)-1-i)
        
        selected_value = value[i:i+step]
        
        r1 = radius[i+step] 
        r0 = radius[i]
        dV = (r1**2-r0**2)*np.pi
        
        if method=='density':
            dM = np.sum(selected_value)
            res_tmp = dM/dV
        if method=='average':
            res_tmp = np.mean(selected_value)
        r_tmp = (r0+r1)/2
        r.append(r_tmp)
        res.append(res_tmp)
        i += step
    return np.array(r), np.array(res)
