"""Freeze verified official-only prepared data, CPU runner and small upload parts."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()


def build():
    project=Path(__file__).resolve().parents[1];prepared=project/'artifacts/v32_ready_20260912'
    dest=project/'platform/v32_bundle_20260912';dest.mkdir(exist_ok=False)
    source_files=['soc_v3_prepare.py','soc_v32_prepare.py','v32_finalize.py','v32_split.py','v32_features.py',
      'run_v32_train.py','review_v32_round.py','v32_model_diagnostics.py','test_soc_v3_prepare.py','test_soc_v32_prepare.py','test_v32_execution.py']
    (dest/'training').mkdir();(dest/'prepared').mkdir();(dest/'evidence/2026-09-12').mkdir(parents=True)
    for name in source_files:
        source=project/'training'/name;ast.parse(source.read_text(encoding='utf-8'),feature_version=(3,8))
        shutil.copyfile(source,dest/'training'/name)
    for source in prepared.iterdir():
        if source.suffix in {'.json','.parquet','.py'}:shutil.copyfile(source,dest/'prepared'/source.name)
    shutil.copyfile(project/'evidence/2026-09-12/v32_actual_fixtures.json',dest/'evidence/2026-09-12/v32_actual_fixtures.json')
    shutil.copyfile(project/'platform/verify_v32_bundle.py',dest/'verify_bundle.py')
    shutil.copyfile(project/'platform/run_v32_on_platform.sh',dest/'run_on_platform.sh')
    (dest/'README.md').write_text('''# v3.2 第一轮平台执行包

本包只使用已核对的官方训练集。准备结果已在本地全量检查；C1-W 尚未训练。详细执行审查见原项目 docs/V32_ROUND1_REVIEW.md。

在现有 JupyterLab 终端执行：

```bash
bash /root/work/sf02_v32_round/run_on_platform.sh
```

默认原训练文件为 /root/work/sf02_data/train.parquet；其他位置可作为脚本第一个参数。复用 cm-model 环境，CPU 至多 16 线程，不使用昇腾卡，不安装包。

固定三外折，每折 C=0.1/1.0 二选一后重新拟合一次，最多九次分类器拟合。保留所有记录、原标签和每行权重 1。外层是已见官方数据的开发评价，不是盲测或迁移验收。

脚本先核对文件与依赖、运行表示反例检查、再次核对原训练数据 SHA-256 和资源，再开始训练。失败会停止，不自动换算法、缩样本或改种子。输出使用新时间目录，含模型、完整逐行外层和校准分数、阈值、切片、压力测试与独立复核。最后生成结果 ZIP，供带回原项目审查。

相同环境中的包元数据存在不代表全局依赖健康；本轮只使用 numpy/scipy/sklearn/pyarrow/joblib CPU 路径。模型分数不是经过校准的正确概率。
''',encoding='utf-8')
    files=[{'path':p.relative_to(dest).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(dest.rglob('*')) if p.is_file()]
    manifest={'version':'v32-cpu-bundle-1.0','files':files,'official_train_sha256':'6b6d5e23caebfd1c4f6b70c9e58c27f437bca7f0cd26497eefa3e4908f2cb742',
      'input_policy':'Official train only. No valid_input, valid answers, external data, package installs or NPU training.'}
    (dest/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    archive=project/'platform/sf02_v32_round.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(dest.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(dest).as_posix())
    parts=project/'platform/v32_upload_parts';parts.mkdir(exist_ok=False);part_info=[]
    with archive.open('rb') as src:
        index=1
        while True:
            b=src.read(80*1024**2)
            if not b:break
            path=parts/('sf02_v32_round.zip.part%02d'%index);path.write_bytes(b)
            part_info.append({'name':path.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()});index+=1
    assembly='''"""Verify upload parts and extract the bound v3.2 bundle; no training here."""
import hashlib,json
from pathlib import Path
import zipfile
PARTS=__PARTS__
ARCHIVE_SHA=__SHA__
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
root=Path(__file__).resolve().parent
for part in PARTS:
 path=root/part['name']
 if not path.is_file() or path.stat().st_size!=part['bytes'] or sha(path)!=part['sha256']:raise SystemExit('Missing or damaged part: '+part['name'])
archive=root/'sf02_v32_round.zip'
if archive.exists():
 if sha(archive)!=ARCHIVE_SHA:raise SystemExit('Existing ZIP differs; refusing overwrite')
else:
 with archive.open('xb') as out:
  for part in PARTS:
   with (root/part['name']).open('rb') as src:
    for b in iter(lambda:src.read(8388608),b''):out.write(b)
 if sha(archive)!=ARCHIVE_SHA:raise SystemExit('ZIP hash mismatch')
dest=root/'sf02_v32_round'
if dest.exists():raise SystemExit('Verified ZIP available; extraction directory already exists. No files overwritten.')
with zipfile.ZipFile(archive) as z:
 for name in z.namelist():
  p=(dest/name).resolve()
  if dest.resolve() not in p.parents:raise SystemExit('Unsafe archive path')
 dest.mkdir()
 z.extractall(dest)
print(json.dumps({'archive_sha256':ARCHIVE_SHA,'extracted_to':str(dest),'model_trained':False}))
'''.replace('__PARTS__',repr(part_info)).replace('__SHA__',repr(sha(archive)))
    (parts/'assemble_v32_bundle.py').write_text(assembly,encoding='utf-8')
    report={'bundle_dir':str(dest),'archive':str(archive),'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),
      'parts':part_info,'assemble_sha256':sha(parts/'assemble_v32_bundle.py'),'files':len(files),'model_trained':False,'platform_executed':False}
    (project/'platform/v32_bundle_receipt.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':build()
