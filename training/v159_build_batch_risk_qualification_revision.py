"""Use the actual GPU chunk numerator mechanism; preserve different-sum failure."""
from experiment_review import ROOT
def main():
    old=ROOT/'training/v159_numeric_batch_replay_qualification.py';new=old.with_name('v159_numeric_batch_replay_qualification_v2.py');assert not new.exists()
    text=old.read_text(encoding='utf-8').replace("v159_numeric_batch_replay_qualification_20261002'","v159_numeric_batch_replay_qualification_v2_20261002'")
    text=text.replace("PREV/'qualification.json']","PREV/'qualification.json',Path(__file__).with_name('v159_build_batch_risk_qualification_revision.py'),ROOT/'training/v159_numeric_batch_replay_qualification.py',ROOT/'artifacts/v159_numeric_batch_replay_qualification_20261002/failure.json']")
    text=text.replace('q=np.empty((n,3));lp=np.empty_like(q)','q=np.empty((n,3));lp=np.empty_like(q);numerator=np.zeros(3)')
    text=text.replace("q[ii]=qq.cpu().numpy();lp[ii]=ll.cpu().numpy()","q[ii]=qq.cpu().numpy();lp[ii]=ll.cpu().numpy();numerator+=(-(torch.tensor(counts[ii],dtype=torch.float64,device='cuda')*ll).sum(0)).cpu().numpy()")
    text=text.replace('risks=(-(counts*lp).sum(0))[1:]/mass[1:]','risks=numerator[1:]/mass[1:]')
    new.write_text(text,encoding='utf-8');print('v2 qualification uses actual GPU chunk risk arithmetic, numeric policy unchanged')
if __name__=='__main__':main()
