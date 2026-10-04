"""Build deterministic single-file execution package from reviewed source files."""
import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile


def build(args):
    root=Path(__file__).resolve().parents[1];training=root/'training'
    pending=['v36_pipeline','run_v36_train','run_v36_prepare','audit_v36_prepared','review_v36_round',
             'patch_v36_native_facts','patch_v36_fallback','v36_infer','test_v36_representation','test_v36_learning','test_v36_execution']
    selected={}
    while pending:
        module=pending.pop()
        if module in selected:continue
        file=training/(module+'.py')
        content=file.read_bytes();tree=ast.parse(content.decode('utf-8'),feature_version=(3,8))
        selected[module]=content
        for node in ast.walk(tree):
            names=[]
            if isinstance(node,ast.Import):names=[a.name.split('.')[0] for a in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:names=[node.module.split('.')[0]]
            pending.extend(n for n in names if (training/(n+'.py')).exists() and n not in selected)
    files={n+'.py':v for n,v in selected.items()}
    audit=root/'evidence/2026-09-12/v36_execution/r13_independent_data_audit.json'
    if audit.exists():
        source=json.loads(audit.read_text(encoding='utf-8'))
        if not source['all_checks_passed']:raise ValueError('Local audit failed')
        files['local_reference.json']=(json.dumps({k:source[k] for k in ['canonical_input_views_sha256','canonical_groups_sha256','canonical_roles_sha256']},indent=2)+'\n').encode('utf-8')
    config={'version':'v36-batch-1.1','representation':'v36-representation-1.3',
       'tasks':['known_dev','source_ad','source_duo','source_waf','asa_hard'],
       'views':args.views.split(','),'repeat_weight_candidate':args.repeat_weight,
       'official_data_only':True,'final_model_promotion':False,'model_quality_must_be_reviewed':True,
       'calendar_product_identity_predictors':False,'evaluation_source_calibration':False,
       'no_unseen_or_external_transfer_guarantee':True}
    files['v36_batch.json']=(json.dumps(config,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    manifest={n:hashlib.sha256(v).hexdigest() for n,v in sorted(files.items())}
    files['manifest.json']=json.dumps(manifest,indent=2,sort_keys=True).encode('utf-8')
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted(files.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,9,12,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data)
    data=buffer.getvalue();sha=hashlib.sha256(data).hexdigest()
    template=(root/'platform/v36_bootstrap_template.py').read_text(encoding='utf-8')
    source=template.replace('__ARCHIVE_SHA__',sha).replace('__PAYLOAD__',base64.b64encode(data).decode('ascii'))
    ast.parse(source,feature_version=(3,8))
    output=root/'platform/sf02_v36_update.py';output.write_text(source,encoding='utf-8',newline='\n')
    receipt={'file':output.name,'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'archive_sha256':sha,'runtime_directory':'sf02_v36_'+sha[:12],'source_files':manifest,'configuration':config,
        'locally_built':True,'cloud_execution_not_yet_performed':True}
    (root/'platform/v36_bundle_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--views',default='B0,B2');p.add_argument('--repeat-weight',action='store_true')
    build(p.parse_args())
