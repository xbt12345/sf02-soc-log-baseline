"""Publish parent result evidence and remove superseded current tense."""
import json,shutil
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
OUT=ROOT/'artifacts/v156_independent_review_publication_20261001'
RUN=ROOT/'artifacts/v155_guarded_full_gradient_sam_20261001'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not OUT.exists()
    check_bindings(read(RUN/'completion_receipt.json')['source_sha256'])
    report=ROOT/'docs/V155_INDEPENDENT_RESULT_REVIEW_AND_NEXT_DECISION.md'
    audit=ROOT/'artifacts/v155_independent_endpoint_audit_v2_20261001/audit.json'
    attribution=audit.with_name('failure_attribution.json')
    check_bindings(read(audit)['source_sha256']);check_bindings(read(attribution)['source_sha256'])
    OUT.mkdir();paths=[ROOT/z for z in ['README.md','HANDOFF.md','mcp_readonly/catalog.json','mcp_readonly/tests/test_readonly_mcp.py']]
    snapshot=OUT/'previous_publication_snapshot';snapshot.mkdir()
    for p in paths:
        target=snapshot/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    save(snapshot/'manifest.json',dict(source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths}))
    cp=ROOT/'mcp_readonly/catalog.json';cat=read(cp);pr=cat['project']
    assert (pr['authoritative_delivery_id'],pr['authoritative_direction_id'])==('v155-delivery','v155-results-review')
    pr['current_summary']='最新实际V155六拟合完成：1800梯度/1806proposal/1200更新；零步9完整梯度/108分块前向另计。完整2056871行验收与父独立重计一致；A ASA错1904M/1631S，B 1920M/1629S；B较V146 A新增16M、S修复0。纯TRAIN错0及旧联合保护、五状态窗口保持；匹配和全任务质量失败，未晋升、不追加本配置确认。第二第三及完整目标仍未通过。'
    pr['known_limits']=[s for s in pr['known_limits'] if 'V155当前仅zero-step' not in s]
    pr['known_limits'].append('V155已实际完成6拟合/1800梯度；9零步诊断梯度另计。完整质量失败，没有新独立同类支持或新环境泛化确认。')
    pr['current_direction'].append('父独立验收/失败归因已发布；下一只读审查类条件组成支持与表示关系，不生成标签或开始第七次拟合。')
    entries=[('v155-independent-result-review','V155父独立完整验收与下一步决定',report),('v155-independent-endpoint-audit','V155父官方全人口与实际终点独立重计',audit),('v155-independent-failure-attribution','V155父原行变化及损失失败归因',attribution)]
    assert not {e['id'] for e in cat['documents']}.intersection(x[0] for x in entries)
    cat['documents']=[dict(id=i,title=t,path=p.relative_to(ROOT).as_posix(),category='review_evidence',summary=t,sha256=sha(p),keywords=['V155','独立','当前','完整','失败','下一步']) for i,t,p in entries]+cat['documents']
    raw=(json.dumps(cat,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(raw)<=256*1024;cp.write_bytes(raw)
    for name in ['README.md','HANDOFF.md']:
        p=ROOT/name;s=p.read_text(encoding='utf-8')
        if name=='README.md':s=s.replace('[六拟合完整验收与决定](docs/V155_FULL_GRADIENT_NEIGHBORHOOD_RESULTS_AND_DECISION.md)', '[六拟合完整验收与决定](docs/V155_FULL_GRADIENT_NEIGHBORHOOD_RESULTS_AND_DECISION.md)；[父独立验收及下一步决定](docs/V155_INDEPENDENT_RESULT_REVIEW_AND_NEXT_DECISION.md)',1)
        else:s+='\n\n## V155父独立复核已同步\n\n父 docs/V155_INDEPENDENT_RESULT_REVIEW_AND_NEXT_DECISION.md / artifacts/v155_independent_endpoint_audit_v2_20261001/audit.json 与failure_attribution.json：完整2056871官方真值/112807 ASA/3579封存/6fit成本重计一致，父0模型/梯度/拟合/更新。B较V146 A新增16M、修复0；困难578仍576错。MCP当前摘要与边界删除了V155尚未拟合/仅零步的旧当前时态，9零步另计；旧发布文件状态留在snapshot供历史哈希核验。下一只读类条件/表示关系诊断，不生成标签或启动第七fit。\n'
        p.write_text(s,encoding='utf-8')
    save(OUT/'publication.json',dict(status='independent_V155_actual_review_published_current_tense_corrected',new_classifier_fits=0,new_gradients=0,new_updates=0,
        catalog_bytes=len(raw),source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),report,audit,attribution,ROOT/'README.md',ROOT/'HANDOFF.md',cp]}))
    print(json.dumps(dict(status='independent_result_published',catalog_bytes=len(raw),latest_actual='V155',new_fits=0)))

if __name__=='__main__':main()
