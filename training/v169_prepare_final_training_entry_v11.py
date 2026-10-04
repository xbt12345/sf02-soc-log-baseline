"""Final source version: per-operation fixed reserve and full action chain."""
from experiment_review import ROOT

def main():
    source=ROOT/'training/v169_prior_pair_training_entry_v10.py';target=ROOT/'training/v169_prior_pair_training_entry_v11.py';assert not target.exists()
    s=source.read_text(encoding='utf-8').replace('from v169_pair_execution_review_v2 import require_run_seal','from v169_pair_execution_review_v3 import require_run_seal')
    s=s.replace('self.margin_completed=self.margins=self.qps=self.proposals=self.updates=0','self.reserve=0;self.margin_completed=self.margins=self.qps=self.proposals=self.updates=0')
    s=s.replace('    def gradient_before(self,cls,mass):',"    def resource_check(self):\n        if shutil.disk_usage(ROOT).free<self.reserve:raise RuntimeError('Registered fixed disk reserve exhausted')\n    def gradient_before(self,cls,mass):\n        self.resource_check()",1)
    s=s.replace('    def margin_before(self,identity):','    def margin_before(self,identity):\n        self.resource_check()',1)
    s=s.replace('    def enter(self,kind):','    def enter(self,kind):\n        self.resource_check()',1)
    old="self.counter=PairCounter(self.model,plan['role_caps'][str(role)],self.folder/'calls.jsonl');self.last_observation=None"
    new="self.counter=PairCounter(self.model,plan['role_caps'][str(role)],self.folder/'calls.jsonl');self.counter.reserve=plan['resources']['fixed_free_disk_reserve_bytes'];self.last_observation=None"
    assert old in s;s=s.replace(old,new);compile(s,str(target),'exec');target.write_text(s,encoding='utf-8');print(target.relative_to(ROOT))

if __name__=='__main__':main()
