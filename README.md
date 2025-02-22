<div align="center">
  <h3><b> The official implementation of ShapeX</b></h3>
</div>

## Introduction 🌟

ShapeX, an innovative framework that segments time series into meaningful shapelet-driven segments and employs Shapley values to assess their saliency. At the core of ShapeX lies the Shapelet Describe-and-Detect (SDD) framework, which effectively learns a diverse set of shapelets essential for classification.

## Installation 🛠️

```bash
# Clone the repository
git clone https://github.com/yourusername/ShapeX.git
cd ShapeX

# Install dependencies
pip install -r requirements.txt
```

## Quick Start 🚀

### Training a Segmentation Model

```bash
python script.py \
    --primary_param num_prototypes \
    --secondary_param prototype_len \
    --group default \
    --is_training 1
```

### Computing Saliency Scores

```bash
python exp_saliency.py \
    --data mitecg \
    --model ProtoPTST \
    --num_prototypes 4 \
    --prototype_len 30
```

