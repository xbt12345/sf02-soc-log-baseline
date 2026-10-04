"""Record actual read-only MCP tool postconditions after result publication."""
import asyncio,hashlib,json
from pathlib import Path
from mcp import Client
from mcp_readonly import server as readonly

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v158_complete_result_publication_20261001'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

async def main():
    assert not (OUT/'actual_mcp_postconditions.json').exists()
    cat=readonly._load_catalog();entries=readonly._documents_by_id(cat)
    source_paths={ROOT/e['path'] for e in entries.values()}|{readonly.CATALOG_PATH,Path(__file__).resolve(),ROOT/'mcp_readonly/server.py',ROOT/'mcp_readonly/tests/test_readonly_mcp.py'}
    before={str(p.relative_to(ROOT)):digest(p) for p in source_paths}
    async with Client(readonly.server) as client:
        search=await client.call_tool('search',{'query':'当前模型能力和下一步方向'})
        status=await client.call_tool('get_project_status',{})
        fetch=await client.call_tool('fetch',{'id':'v158-delivery'})
        result=await client.call_tool('fetch',{'id':'v158-result-review'})
        supplement=await client.call_tool('fetch',{'id':'v158-whole-bank-supplement'})
        assert not any(z.is_error for z in [search,status,fetch,result,supplement])
        assert search.structured_content['results'][0]['id']=='v158-result-review'
        assert set(status.structured_content['source_ids'])=={'v158-delivery','v158-result-review'}
        assert not status.structured_content['quality_acceptance']
        assert 'V158完成60次真实拟合' in status.structured_content['current_summary']
        assert '148' in supplement.structured_content['text'] and '979' in supplement.structured_content['text']
        actual=json.loads(fetch.structured_content['text']);assert actual['classifier_fits']==60 and actual['latest_actual']=='V158'
        assert not actual['quality_acceptance'] and not actual['model_promoted'] and actual['TRAIN']['B']['all_roles_mastered']
    after={str(p.relative_to(ROOT)):digest(p) for p in source_paths};assert before==after
    receipt=dict(status='actual_read_only_MCP_current_search_status_and_full_delivery_fetch_verified',
        experiment_status=status.structured_content['experiment_status'],authoritative_delivery='v158-delivery',authoritative_direction='v158-result-review',
        quality_acceptance=False,model_promoted=False,classifier_fits=60,catalog_bytes=readonly.CATALOG_PATH.stat().st_size,
        all_source_bytes_unchanged=True,source_sha256=before,
        validation_scope='Actual in-process MCP Client tool calls, plus separately executed ten-test suite; external running HTTP service was not probed.',
        new_official_classifier_calls=0,new_gradients=0,new_fits=0,new_updates=0)
    (OUT/'actual_mcp_postconditions.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_sha256'},ensure_ascii=False),flush=True)

if __name__=='__main__':asyncio.run(main())
