"""Preserve exact historical metadata and file caps using one static archive."""
import json,shutil,sys
from pathlib import Path
from experiment_review import ROOT,read,sha
sys.path.insert(0,str(ROOT))
from mcp_readonly.catalog_io import load_catalog

OUT=ROOT/'artifacts/v167_readonly_catalog_partition_20261002'

def main():
    catalog=ROOT/'mcp_readonly/catalog.json';original=(OUT/'previous/catalog.json').read_bytes();assert catalog.read_bytes()==original;old=json.loads(original);assert old['project']['authoritative_direction_id']=='v167-plan' and len(old['documents'])==447;archive=ROOT/'mcp_readonly/catalog_archive/v167_prior.json';assert not archive.exists() and not (OUT/'partition.json').exists();assert len(original)<=256*1024
    for entry in old['documents']:assert sha(ROOT/entry['path'])==entry['sha256']
    archive.parent.mkdir(parents=True,exist_ok=True);archive.write_bytes(original);new=dict(old);new['documents']=[];new['historical_catalog']=dict(path=archive.relative_to(ROOT).as_posix(),sha256=sha(archive));encoded=(json.dumps(new,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8');assert len(encoded)<=256*1024;candidate=OUT/'candidate_catalog.json';candidate.write_bytes(encoded);effective=load_catalog(ROOT,candidate,256*1024);assert effective['project']==old['project'] and effective['documents']==old['documents'];catalog.write_bytes(encoded);assert load_catalog(ROOT,catalog,256*1024)['documents']==old['documents'];note='\n\n历史目录现在保存在 `catalog_archive/v167_prior.json`，保留原447条 ID、路径、哈希和元数据。`catalog.json`通过一个固定哈希引用它；读取器合并后提供相同search/fetch/status。每个目录和资料文件仍限制256KiB，不允许递归归档、越界路径或重复ID。API操作、只读边界与资料白名单不变。\n';reader=ROOT/'mcp_readonly/README.md';reader.write_bytes((reader.read_text(encoding='utf-8')+note).encode('utf-8'));paths=[Path(__file__).resolve(),ROOT/'training/v167_partition_readonly_catalog.py',ROOT/'artifacts/v167_partition_readonly_catalog_original_console_20261002.txt',ROOT/'mcp_readonly/catalog_io.py',ROOT/'mcp_readonly/server.py',catalog,archive,reader,OUT/'previous/catalog.json',OUT/'previous/server.py'];(OUT/'partition.json').write_text(json.dumps(dict(status='447_exact_historical_catalog_records_preserved_in_single_hash_bound_archive',old_catalog_bytes=len(original),current_catalog_bytes=len(encoded),unchanged_per_file_limit=256*1024,all_447_original_IDs_paths_hashes_metadata_and_order_preserved=True,authoritative_direction='v167-plan',authoritative_delivery='v159-delivery',official_calls=0,no_new_API_tool_or_arbitrary_path_access=True,local_or_cloud_live_channel_not_claimed=True,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths}),ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status='V167_readonly_catalog_losslessly_partitioned',historical_records=447,current_catalog_bytes=len(encoded),unchanged_limit=256*1024,official_calls=0)))

if __name__=='__main__':main()
