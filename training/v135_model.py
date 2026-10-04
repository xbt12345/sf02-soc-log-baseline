"""Same registered Classifier parameters/features, deterministic compact dense GEMM."""
import torch
from v131_model import (Classifier as HistoricalClassifier,ActiveFirst,batch,objective,
                       tensor_hash,module_norms,infer)

class DenseActiveFirst(ActiveFirst):
    def forward(self,x):
        columns,inverse=torch.unique(x.col_indices(),sorted=True,return_inverse=True)
        compact=torch.sparse_csr_tensor(x.crow_indices(),inverse,x.values(),
            size=(x.shape[0],len(columns)),device=x.device).to_dense()
        effective=(self.weight[:,columns].T[:,None,:]*
            self.r[:,columns].T[:,:,None]).reshape(len(columns),self.r.shape[0]*self.weight.shape[0])
        y=(compact@effective).reshape(x.shape[0],self.r.shape[0],self.weight.shape[0])
        return y*self.s+self.bias

class Classifier(HistoricalClassifier):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Keep tensors, initialization, optimizer parameter order and serialization identical.
        self.first.__class__=DenseActiveFirst
