"""Observe optimizer entry/exit without changing its arguments or numerics."""
import sys
import v163_one_sided_joint_restoration as solver

def observed_propose(callback,*args):
    code=solver.minimize.__code__;previous=sys.getprofile()
    def observer(frame,event,value):
        if frame.f_code is code and event in ['call','return']:
            callback(event,value)
        if previous is not None:previous(frame,event,value)
    sys.setprofile(observer)
    try:return solver.propose(*args)
    finally:sys.setprofile(previous)
