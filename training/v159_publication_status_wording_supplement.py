"""Preserve failed publication test and repair missing trained-history wording."""
import json
from experiment_review import ROOT,read,sha
def main():
    out=ROOT/'artifacts/v159_publication_status_wording_supplement_20261002';assert not out.exists();out.mkdir()
    p=ROOT/'mcp_readonly/catalog.json';(out/'previous_catalog.json').write_bytes(p.read_bytes())
    cat=read(p);cat['project']['current_summary']=cat['project']['current_summary'].replace('最新实际训练V158完成60 fits及完整2056871原行验收但质量失败。','最新实际训练V158完成60次真实拟合及完整2056871原行验收，融合1200完整梯度/1200proposal/1200更新，质量失败、未晋升。')
    raw=(json.dumps(cat,ensure_ascii=False,separators=(',',':'))+'\n').encode();assert len(raw)<=256*1024;p.write_bytes(raw)
    (out/'receipt.json').write_text(json.dumps(dict(status='repair_missing_V158_history_wording_after_one_MCP_test_failure',prior_test_count=15,prior_failures=1,prior_failure='test_execution_is_distinct_from_quality_acceptance missing unchanged trained-history strings',new_model_calls=0,catalog_bytes=len(raw),source_sha256=sha(__file__),catalog_sha256=sha(p)),indent=2)+'\n',encoding='utf-8')
    print('current factual status wording supplemented; no official calls')
if __name__=='__main__':main()
