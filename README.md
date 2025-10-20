아래 내용을 그대로 `README.md`에 붙여넣으시면 됩니다.

````markdown
# DynaPool

**Input-adaptive pooling benchmark** on CIFAR-100 with a ResNet-18 backbone.  
DynaPool learns **per-instance mixture weights** over four simple operators: **Average, Max, GeM, Attention** via a lightweight gating MLP.

> **Main finding.** On our controlled benchmark, **GeM** attains the best top-1 accuracy (0.5833); **Max** is close behind (0.5819) with lower training time. **DynaPool** is competitive while offering **instance-wise interpretability** at **modest cost**.

---

## 1. Features
- Unified benchmark for {Avg, Max, GeM, Attention} pooling on CIFAR-100 / ResNet-18
- DynaPool: two-layer MLP gating → mixture coefficients \( \alpha \) per input
- Optional entropy regularization for non-collapsed \(\alpha\)
- Clean training script & YAML configs for ablations

---

## 2. Environment

```bash
# create environment (example with conda)
conda create -n dynapool python=3.10 -y
conda activate dynapool

# install dependencies
pip install -r requirements.txt
````

> **CUDA**: install PyTorch for your CUDA version if needed
> (e.g., `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121`).

`requirements.txt` minimal set:

```
torch>=2.2
torchvision>=0.17
torchaudio>=2.2
tqdm
numpy
pyyaml
pandas
```

---

## 3. Quick Start

Train baselines:

```bash
python experiments/train.py --model resnet18 --pool avg   --epochs 300 --batch-size 128
python experiments/train.py --model resnet18 --pool max
python experiments/train.py --model resnet18 --pool gem   --gem-init 1.0
python experiments/train.py --model resnet18 --pool attn
```

Train **DynaPool**:

```bash
python experiments/train.py --model resnet18 --pool dynapool \
  --gate-hidden 256 --gate-dropout 0.1 --alpha-temp 1.0 \
  --alpha-entropy 1e-3
```

Common options:

* `--pool {avg,max,gem,attn,dynapool}`
* `--alpha-temp` : softmax temperature for (\alpha)
* `--alpha-entropy` : (\lambda) in (L_{\text{total}} = L_{\text{ce}} + \lambda , H(\alpha))
* `--gem-param` / `--gem-init` : GeM exponent (p) (internally constrained with softplus)

---

## 4. Results (Reproducing Tables)

**Table I. Main benchmark (mean over 3 seeds)**

| Pooling Method      | Test Acc.  | Train Time (min) | GeM p    | Adaptivity |
| ------------------- | ---------- | ---------------- | -------- | ---------- |
| Average             | 0.5752     | 71.42            | –        | ✗          |
| Max                 | 0.5819     | 69.86            | –        | ✗          |
| **GeM**             | **0.5833** | 73.99            | −0.0000  | Global     |
| Attention           | 0.5722     | 74.79            | –        | Local      |
| **DynaPool (Ours)** | 0.5787     | 84.66            | Adaptive | ✓          |

* We constrain `p` via `p = softplus(θ) + 1e−6` to avoid near-zero rounding ambiguity.

**Table II. Ablation (illustrative)**

| Variant         | Removed    | Acc.   |     Δ |
| --------------- | ---------- | ------ | ----: |
| DynaPool (Full) | –          | 0.5787 |     – |
| w/o GeM         | GeM branch | 0.5670 | −1.2% |
| w/o Attention   | Attention  | 0.5690 | −0.9% |

---

## 5. Repo Structure (suggested)

```
DynaPool/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ dynapool/
│  ├─ __init__.py
│  ├─ pooling.py        # Avg/Max/GeM/Attention
│  ├─ gating.py         # two-layer MLP for α
│  ├─ models.py         # ResNet-18 + DynaPool integration
│  └─ utils.py          # seed, metrics, profiler
├─ experiments/
│  ├─ train.py          # train/eval loop
│  ├─ config.yaml       # default hparams
│  └─ ablation.yaml     # ablation settings
└─ results/
   ├─ table_main.csv
   └─ table_ablation.csv
```

---

## 6. Citation

If you find this useful, please cite:

```bibtex
@misc{kim2025dynapool,
  title  = {DynaPool: Input-Adaptive Pooling Benchmark on CIFAR-100},
  author = {Kim, Young-eun},
  year   = {2025},
  note   = {GitHub: youngsilver-kim/DynaPool}
}
```
