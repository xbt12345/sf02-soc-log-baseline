# SF02 P_IS 模型

当前发布：**V128r3 P_IS · 第 50 轮固定终点**。SOC 日志三分类：正常、恶意、可疑。

完整来源闭合开发评价覆盖 **2,056,871 条**记录，**Macro-F1 98.905972%，错误 2,137 条**。逐类结果和适用范围见 [模型说明](MODEL_CARD.md)；精确数值见 [metrics.json](metrics.json)。

## 下载权重

[下载当前模型](https://github.com/xbt12345/sf02-soc-log-baseline/releases/tag/sf02-pis-20261006)

下载 `SF02_PIS_weights_20261006.zip` 并在仓库根目录解压，得到 `weights/fold0`、`fold1`、`fold2`。每折包含 `base.pt`、`residual.pt` 和 `router.joblib`，共 9 件必要组件。[权重清单](model_manifest.json)列出逐件 SHA-256；加载器会核验文件身份。

## 数值输入推理

```bash
python -m pip install -r requirements.txt
python predict.py --weights weights --fold 0 --route asa \
  --features header_features.npz --body body_bytes.npy \
  --lengths body_lengths.npy --output predictions.npz
```

其他格式使用对应数值输入：

```bash
python predict.py --weights weights --fold 0 --route other \
  --features n1_features.npz --output predictions.npz
```

默认使用 CPU，可通过 `--device cuda` 指定 GPU。输入须符合 [input_contract.json](input_contract.json)：66,287 维 CSR 特征，以及 ASA 路径的有序正文 bytes 和独立长度。原始日志预处理、事实特征顺序和来源折映射须由调用方按原合同提供。

```python
from sf02_model import FoldModel

model = FoldModel("weights", fold=0)
probabilities = model.predict_asa(header_features, body_bytes, body_lengths)
labels = probabilities.argmax(axis=1)
```

已发布分数按原记录所属折使用对应模型。三折平均及未知来源部署尚未验证；这是已看开发评价的研究权重快照。

## 检查

```bash
python -m unittest discover -s tests -v
```

推理代码与原固定权重的样本重放核对范围见 [verification.json](verification.json)。
