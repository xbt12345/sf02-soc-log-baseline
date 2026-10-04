from __future__ import annotations
import hashlib,json,tempfile,unittest
from pathlib import Path
from mcp_readonly.catalog_io import CatalogReadError,load_catalog
from mcp_readonly import server

class CatalogArchiveTests(unittest.TestCase):
    def make_fixture(self,root:Path,archive:dict,documents:list|None=None):
        raw=json.dumps(archive).encode();(root/'history.json').write_bytes(raw);catalog=dict(schema_version=1,project=dict(name='fixture'),documents=documents or [],historical_catalog=dict(path='history.json',sha256=hashlib.sha256(raw).hexdigest()));path=root/'catalog.json';path.write_text(json.dumps(catalog),encoding='utf-8');return path,catalog

    def test_all_447_real_historical_records_remain_exact_and_fetchable(self):
        root=server.PROJECT_ROOT;old=json.loads((root/'artifacts/v167_readonly_catalog_partition_20261002/previous/catalog.json').read_text(encoding='utf-8'));effective=server._load_catalog();now=server._documents_by_id(effective)
        self.assertEqual(len(old['documents']),447)
        for record in old['documents']:
            self.assertEqual(now[record['id']],record);self.assertEqual(hashlib.sha256(server._safe_document(record)[1]).hexdigest(),record['sha256'])
        self.assertEqual(effective['project']['name'],old['project']['name']);self.assertEqual(server.MAX_DOCUMENT_BYTES,256*1024)

    def test_historical_hash_change_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path,_=self.make_fixture(root,dict(schema_version=1,documents=[dict(id='old-evidence',path='old.md',sha256='a'*64,title='Old')]))
            (root/'history.json').write_bytes(b'{}')
            with self.assertRaisesRegex(CatalogReadError,'integrity'):load_catalog(root,path,server.MAX_DOCUMENT_BYTES)

    def test_duplicate_ids_and_nested_archives_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path,_=self.make_fixture(root,dict(schema_version=1,documents=[dict(id='old-evidence')]),[dict(id='old-evidence')])
            with self.assertRaisesRegex(CatalogReadError,'Duplicate'):load_catalog(root,path,server.MAX_DOCUMENT_BYTES)
            path,_=self.make_fixture(root,dict(schema_version=1,documents=[],historical_catalog=dict(path='again.json',sha256='b'*64)))
            with self.assertRaisesRegex(CatalogReadError,'Nested'):load_catalog(root,path,server.MAX_DOCUMENT_BYTES)

    def test_escape_and_oversized_archive_do_not_bypass_original_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path,catalog=self.make_fixture(root,dict(schema_version=1,documents=[]));catalog['historical_catalog']['path']='../outside.json';path.write_text(json.dumps(catalog),encoding='utf-8')
            with self.assertRaisesRegex(CatalogReadError,'Unsafe'):load_catalog(root,path,server.MAX_DOCUMENT_BYTES)
            path,_=self.make_fixture(root,dict(schema_version=1,documents=[]));(root/'history.json').write_bytes(b'x'*(server.MAX_DOCUMENT_BYTES+1))
            with self.assertRaisesRegex(CatalogReadError,'limit'):load_catalog(root,path,server.MAX_DOCUMENT_BYTES)

if __name__=='__main__':unittest.main()
