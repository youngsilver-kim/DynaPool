# DynaPool

**DynaPool: Input-adaptive pooling benchmark** on CIFAR-100 (ResNet-18).  
We learn per-instance mixture weights over {Average, Max, GeM, Attention} via a lightweight gating MLP.

> **Main findings.** GeM achieves the best top-1 (0.5833), Max is close (0.5819) with lower training time, and **DynaPool** remains competitive while providing instance-wise interpretability.

## 1. Environment
```bash
conda create -n dynapool python=3.10 -y
conda activate dynapool
pip install -r requirements.txt
