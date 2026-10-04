"""Use the actual entry's CuBLAS environment before importing torch."""
from experiment_review import ROOT

def main():
    source=ROOT/'training/v169_cuda_host_buffer_qualification.py';target=ROOT/'training/v169_cuda_host_buffer_qualification_v2.py';assert not target.exists()
    s=source.read_text(encoding='utf-8').replace('import ctypes,json',"import os\nos.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'\nimport ctypes,json").replace('v169_cuda_host_buffer_qualification_20261002','v169_cuda_host_buffer_qualification_v2_20261002');compile(s,str(target),'exec');target.write_text(s,encoding='utf-8');print(target.relative_to(ROOT))

if __name__=='__main__':main()
