"""Independent record-level source-port observation, never message overwrite."""
import re
import numpy as np
from scipy import sparse


def parse_port(value):
    if value is None:return None,'null'
    value=str(value)
    if not value.strip():return None,'empty'
    if not re.fullmatch(r'[0-9]+',value):return None,'opaque_or_noncanonical'
    n=int(value)
    return (n,'observed') if n<=65535 else (None,'outside_port_domain')


def encode(values,message_ports):
    values=list(values);message_ports=np.asarray(message_ports)
    parsed=[parse_port(v) for v in values]
    port=np.array([65536 if n is None else n for n,_ in parsed],np.int32)
    observed=port<65536
    conflict=observed&(message_ports<65536)&(port!=message_ports)
    dense=np.zeros((len(port),18),np.float32)
    dense[:,:16]=((port[:,None]>>np.arange(16))&1)*observed[:,None]
    dense[:,16]=observed;dense[:,17]=conflict
    # Fixed scaling, no population-fitted statistics or validation dependence.
    dense/=np.sqrt(18)
    key=np.where(observed,1+port*2+conflict,0).astype(np.uint32)
    return sparse.csr_matrix(dense),key,{
        'observed':int(observed.sum()),'message_unobserved_record_observed':int((observed&(message_ports==65536)).sum()),
        'conflicting_observations':int(conflict.sum()),'states':[s for _,s in parsed]}
