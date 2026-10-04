"""Supersede a remaining prose count typo while preserving the initial receipt."""
import json
from v89_common import ROOT,read,save,sha


def main():
    directory=ROOT/'evidence/2026-09-28/v98_boundary_audit'
    destination=directory/'delivery_r2.json';assert not destination.exists()
    original_path=directory/'delivery.json';original=read(original_path)
    bindings=dict(original['artifact_sha256']);past={}
    for path,digest in bindings.items():assert sha(ROOT/path)==digest
    for receipt,h in original['verification']['receipt_sha256'].items():
        assert sha(ROOT/receipt)==h
        for path,digest in read(ROOT/receipt)['artifact_sha256'].items():
            assert sha(ROOT/path)==digest;past[path]=digest
    assert len(past)==1099
    report=ROOT/original['current_plan'];content=report.read_text(encoding='utf-8')
    assert content.count('空格单独控制翻错 34，')==1
    corrected=report.with_name(report.stem+'_R2.md');assert not corrected.exists()
    content=content.replace('空格单独控制翻错 34，','空格单独控制翻错 28，')
    content+='\n发布勘误：初版表格已为28，但反证段落遗漏一处34。R2仅纠正这一处文字，保存初版和原始绑定凭据；全部实验、训练计划和质量结论不变。\n'
    corrected.write_text(content,encoding='utf-8');doc=corrected.relative_to(ROOT).as_posix()
    artifact=ROOT/'artifacts/v98_boundary_and_gate_audit_20260928'
    contract=read(artifact/'next_experiment_contract.json');contract['current_report']=doc
    save(artifact/'next_experiment_contract_r2.json',contract)
    correction={'status':'prose_count_corrected_without_historical_mutation',
        'original_delivery_sha256':sha(original_path),'original_report_sha256':sha(report),
        'change':'One remaining adversarial-review sentence said34 whitespace negative flips; corrected to28 matching its table and all raw role counts.',
        'experiment_outputs_changed':False,'fits':0,'prior_bound_files_verified':1099,
        'initial_v98_bound_files_verified':len(bindings),'source_sha256':sha(__file__)}
    save(artifact/'publication_correction.json',correction)
    for p in [corrected,artifact/'next_experiment_contract_r2.json',artifact/'publication_correction.json',ROOT/'training/v98_correct_publication.py',original_path]:
        bindings[p.relative_to(ROOT).as_posix()]=sha(p)
    final={**original,'current_plan':doc,'artifact_sha256':bindings,
        'publication_correction':correction,'supersedes_initial_delivery':original_path.relative_to(ROOT).as_posix()}
    save(destination,final)
    catalog_path=ROOT/'mcp_readonly/catalog.json';catalog=read(catalog_path)
    old_entries=[]
    for d in catalog['documents']:
        if d['id']=='v98-direction-review':
            old_entries.append({**d,'id':'v98-initial-review','category':'historical_review',
                'summary':'初版有一处空格控制34的文字错误，实测28；当前以v98-direction-review的R2勘误为准。'})
            d.update(path=doc,sha256=sha(corrected))
        if d['id']=='v98-delivery':
            old_entries.append({**d,'id':'v98-initial-delivery','category':'historical_evidence',
                'summary':'原始未修改凭据；反证段落文字勘误见当前v98-delivery，实验结果未变。'})
            d.update(path=destination.relative_to(ROOT).as_posix(),sha256=sha(destination))
    catalog['documents'].extend(old_entries)
    for relative in ['README.md','docs/TRAINING_PLAN.md']:
        path=ROOT/relative;s=path.read_text(encoding='utf-8')
        s=s.replace('V98_BOUNDARY_ROOT_AUDIT_AND_NEXT_PLAN.md','V98_BOUNDARY_ROOT_AUDIT_AND_NEXT_PLAN_R2.md')
        path.write_text(s,encoding='utf-8')
        for d in catalog['documents']:
            if d['path']==relative:d['sha256']=sha(path)
    save(catalog_path,catalog)
    for path,digest in bindings.items():assert sha(ROOT/path)==digest
    print(json.dumps({'published_r2':True,'bound_files':len(bindings),'catalog_documents':len(catalog['documents']),
        'initial_receipt_unchanged':sha(original_path)==correction['original_delivery_sha256'],
        'delivery_sha256':sha(destination)},ensure_ascii=False))


if __name__=='__main__':main()
