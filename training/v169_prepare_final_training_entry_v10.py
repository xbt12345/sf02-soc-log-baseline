"""New immutable version: final same-point replay and bounded metadata."""
from pathlib import Path
from experiment_review import ROOT

def main():
    source=ROOT/'training/v169_prior_pair_training_entry_v9.py';target=ROOT/'training/v169_prior_pair_training_entry_v10.py'
    assert not target.exists();s=source.read_text(encoding='utf-8')
    s=s.replace("    path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\\n').encode('utf-8'))", "    encoded=(json.dumps(value,ensure_ascii=False,indent=2)+'\\n').encode('utf-8')\n    cap=8*1024**2 if path.name in ['fit.json','partial_inconclusive.json','completed_or_inconclusive.json'] else (16384 if path.name=='identity_repeat.json' else 65536)\n    if len(encoded)>cap:raise RuntimeError('Registered JSON size bound exceeded: '+path.name)\n    path.write_bytes(encoded)")
    s=s.replace('from v169_pair_execution_review import require_run_seal','from v169_pair_execution_review_v2 import require_run_seal')
    s=s.replace("    def gradient_before(self,cls,mass):", "    def write(self,**value):\n        encoded=json.dumps(value)+'\\n'\n        if len(encoded.encode('utf-8'))>512:raise RuntimeError('Registered counter-line size exceeded')\n        self.log.write(encoded)\n    def gradient_before(self,cls,mass):",1)
    s=s.replace('self.margins=self.qps=self.proposals=self.updates=0','self.margin_completed=self.margins=self.qps=self.proposals=self.updates=0')
    s=s.replace("    def margin_after(self,identity):self.write(event='completed',kind='complete_margin_derivative',ordinal=self.margins,input_identity=identity,parameter_sha256=self.point)","    def margin_after(self,identity):\n        self.margin_completed+=1;self.write(event='completed',kind='complete_margin_derivative',ordinal=self.margin_completed,input_identity=identity,parameter_sha256=self.point)")
    old="        if not guard['passed']:raise RuntimeError('Restored complete endpoint protection failed')\n        folder=self.folder/'endpoint';folder.mkdir()"
    new="""        if not guard['passed']:raise RuntimeError('Restored complete endpoint protection failed')
        previous=self.last_observation
        if previous is None or previous['parameter_sha256']!=self.identity():raise RuntimeError('No same-point last committed observation; endpoint inconclusive')
        checks=[]
        for scope in ['OOF','deployment']:
            ids=self.ctx['ids'] if scope=='OOF' else np.arange(22546)
            for name,kind in [('q','probability'),('logq','log_probability')]:checks.append(repeat_values(ob[f'{scope}_{name}'][ids],previous[f'{scope}_{name}'][ids],kind))
        for name in rv:checks.append(repeat_values(rv[name],previous['risk'][name],'risk'))
        if not all(check['passed'] for check in checks):raise RuntimeError('Last committed same-point final q/logq/risk/argmax drift')
        folder=self.folder/'endpoint';folder.mkdir();save(folder/'same_parameter_replay.json',dict(parameter_sha256=self.identity(),checks=checks,no_additional_model_calls=True))"""
    assert old in s;s=s.replace(old,new)
    s=s.replace("for scope in ['OOF','deployment']:store_scope(folder,self.ctx,scope,ob[f'{scope}_q'],ob[f'{scope}_logq'])", """for scope in ['OOF','deployment']:
            store_scope(folder,self.ctx,scope,ob[f'{scope}_q'],ob[f'{scope}_logq'])
            import pyarrow.parquet as pq
            logical=pq.read_table(PRIOR/f'role{self.role}/endpoint/{scope}_original_rows.parquet').nbytes
            actual=(folder/f'{scope}_original_rows.parquet').stat().st_size+(folder/f'{scope}_source_rows.parquet').stat().st_size
            if actual>2*logical+2*1024**2:raise RuntimeError('Final complete table storage bound exceeded')""")
    s=s.replace('margin_derivative_attempts=self.counter.margins,QP_attempts=', 'margin_derivative_attempts=self.counter.margins,margin_derivative_completed=self.counter.margin_completed,QP_attempts=')
    compile(s,str(target),'exec');target.write_text(s,encoding='utf-8');print(target.relative_to(ROOT))

if __name__=='__main__':main()
