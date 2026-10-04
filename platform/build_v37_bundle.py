"""Build only after the independently checked full official-input gate passes."""
import ast,base64,hashlib,io,json,sys,zipfile
from pathlib import Path

def build():
    root=Path(__file__).resolve().parents[1];training=root/'training';sys.path.insert(0,str(training))
    from v37_contract import CONTRACT
    import v37_representation as rep
    evidence=root/'evidence/2026-09-13/v37_execution'
    audit=json.loads((evidence/'input_audit.json').read_text(encoding='utf-8'))
    if not audit['all_checks_passed']:raise ValueError('Full local input gate has not passed')
    for name,sha in audit['transitive_parser_hashes'].items():
        if hashlib.sha256((training/name).read_bytes()).hexdigest()!=sha:raise ValueError('Audited source changed: '+name)
    if hashlib.sha256((training/'audit_v37_prepared.py').read_bytes()).hexdigest()!=audit['auditor_sha256']:raise ValueError('Auditor changed after gate')
    frozen=json.loads((evidence/'preregistered_contract.json').read_text(encoding='utf-8'))
    if frozen!=CONTRACT:raise ValueError('Preregistration differs')
    pending=['v37_pipeline','run_v37_train','run_v37_prepare','audit_v37_prepared','review_v37_round','test_v37','test_v37_execution'];files={}
    while pending:
        name=pending.pop()
        if name+'.py' in files:continue
        data=(training/(name+'.py')).read_bytes();tree=ast.parse(data.decode('utf-8'),feature_version=(3,8));files[name+'.py']=data
        for node in ast.walk(tree):
            imports=[a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else ([node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else [])
            pending.extend(n for n in imports if (training/(n+'.py')).exists() and n+'.py' not in files)
    def dump(v):return (json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode('utf-8')
    files['local_reference.json']=dump({k:audit[k] for k in ['canonical_input_views_sha256','canonical_groups_sha256','canonical_roles_sha256']})
    files['local_input_gate.json']=dump(audit);files['v37_batch.json']=dump(CONTRACT)
    files['phase_B_receipt.json']=(evidence/'frozen_B2_decisions.json').read_bytes()
    manifest={n:hashlib.sha256(v).hexdigest() for n,v in sorted(files.items())};files['manifest.json']=dump(manifest)
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted(files.items()):
            i=zipfile.ZipInfo(name,date_time=(2026,9,13,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,data)
    data=stream.getvalue();sha=hashlib.sha256(data).hexdigest()
    template=(root/'platform/v37_bootstrap_template.py').read_text(encoding='utf-8')
    source=template.replace('__ARCHIVE_SHA__',sha).replace('__PAYLOAD__',base64.b64encode(data).decode('ascii'))
    ast.parse(source,feature_version=(3,8));output=root/'platform/sf02_v37_update.py';output.write_text(source,encoding='utf-8',newline='\n')
    receipt={'file':output.name,'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'archive_sha256':sha,'runtime_directory':'sf02_v37_'+sha[:12],'files':manifest,
        'representation':rep.VERSION,'contract':CONTRACT,'cloud_execution_completed':False}
    (root/'platform/v37_bundle_receipt.json').write_bytes(dump(receipt))
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('files','contract')}))
if __name__=='__main__':build()
