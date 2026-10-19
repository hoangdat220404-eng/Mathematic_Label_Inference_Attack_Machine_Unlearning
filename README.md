# Vector-Based Label Inference Attack against Machine Unlearning (VLI)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Thesis: HCMUS](https://img.shields.io/badge/BSc%20Thesis-VNU--HCMUS-003366.svg)](https://www.hcmus.edu.vn/)

Official implementation and documentation for the Undergraduate Graduation Thesis:  
**"Vector-Based Label Inference Attack against Machine Unlearning"**  
*Faculty of Information Technology, VNU-HCM University of Science (HCMUS).*

---

## 📌 Overview

Machine Unlearning allows trained deep learning models to revoke the influence of sensitive user data to satisfy privacy mandates such as GDPR and CCPA ("Right to be Forgotten"). However, parameter differences between the original model ($\theta$) and unlearned model ($\theta_u$) can inadvertently leak sensitive information.

This repository provides the reference implementation of **Vector-based Label Inference (VLI)**, a mathematical, closed-form attack that extracts the exact label distribution of removed batches from the bias vector shift $\Delta b^L$ of the final classification layer.
- ⚡ **Zero Retraining / Shadow Models:** Operates directly on parameter deltas without costly auxiliary training.
- 🚀 **Near-Zero Latency:** Computes inference instantly ($0$ GFLOPs, $0$ MB VRAM, $< 1\text{ ms}$).
- 🎯 **High Accuracy:** Recovers multi-sample unlearned labels with high fidelity (e.g., $81.1\%$ on CIFAR-10, $80.1\%$ on CIFAR-100).

---

## 📚 Thesis Documents

This repository includes two official documentation files:

1. **`Thesis_Report_Vietnamese.pdf` (Bản Tiếng Việt Chính Thức)**:  
   Báo cáo khóa luận tốt nghiệp đầy đủ bằng tiếng Việt nộp chính thức cho Hội đồng Đánh giá Khóa luận, Khoa Công nghệ Thông tin, Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM. Bao gồm toàn bộ cơ sở lý thuyết, chứng minh toán học, khảo sát thực nghiệm và phân tích bảo mật.

2. **`Extended_Summary_English.pdf` (English Extended Summary)**:  
   A comprehensive English academic summary detailing the problem formulation, theoretical analysis of bias shifts, peeling decomposition algorithms for multi-sample batches, and cross-dataset benchmark comparisons.

---

## ⚙️ Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/Mathematic_Label_Inference_Attack_Machine_Unlearning.git
cd Mathematic_Label_Inference_Attack_Machine_Unlearning

# Create and activate environment
conda create -n unlearning-vli python=3.10 -y
conda activate unlearning-vli

# Install PyTorch and required packages
pip install torch torchvision numpy scipy scikit-learn matplotlib
```

---

## 🚀 Usage Guide

The evaluation workflow consists of two main steps: **(1) Pretraining the target model** and **(2) Executing the unlearning process alongside the label inference attack**.

### Step 1: Prepare and Train Base Model

Train the baseline model on the desired dataset:

```bash
python prepare_model.py \
    --dataset cifar10 \
    --model ResNet18 \
    --pretrain_epochs 10 \
    --finetune_epochs 5
```

---

### Step 2: Run Unlearning & Attack Evaluation

Benchmark the VLI attack against various machine unlearning algorithms (both approximate and exact schemes):

#### 1. Negative Gradient Descent (NegGrad)
Reverses gradient descent updates on the forgotten set:
```bash
python main_attack.py \
    --dataset cifar100 \
    --model MobileNetV2 \
    --unlearned_algo neggrad \
    --aux_size 100 \
    --total_loops 100 \
    --unlr 0.01 \
    --alpha 1 \
    --batch_size 64 \
    --mini_batch_size 64 \
    --local_loops 1 \
    --seed 0
```

#### 2. Fine-tuning on Retain Set
Adapts model parameters exclusively on the remaining retained dataset:
```bash
python main_attack.py \
    --dataset cifar10 \
    --model ResNet18 \
    --unlearned_algo finetuning \
    --aux_size 200 \
    --total_loops 5 \
    --unlr 0.01 \
    --alpha 1 \
    --batch_size 8 \
    --seed 0
```

#### 3. SCRUB (Student-Teacher Distillation Unlearning)
Leverages distillation loss to forget requested points while retaining utility:
```bash
python main_attack.py \
    --dataset cifar10 \
    --model ResNet18 \
    --unlearned_algo scrub \
    --aux_size 200 \
    --total_loops 5 \
    --unlr 0.01 \
    --alpha 1 \
    --batch_size 8 \
    --seed 0
```

#### 4. NegGrad+ (Accelerated Negative Gradient)
Enhanced gradient descent with momentum and weight decay stabilization:
```bash
python main_attack.py \
    --dataset cifar10 \
    --model ResNet18 \
    --unlearned_algo neggradp \
    --aux_size 200 \
    --total_loops 5 \
    --unlr 0.01 \
    --alpha 1 \
    --batch_size 8 \
    --seed 0
```

#### 5. Exact Retraining from Scratch (Retrain)
Retrains the architecture from scratch on the retain set (gold standard baseline):
```bash
python main_attack.py \
    --dataset cifar10 \
    --model ResNet18 \
    --unlearned_algo retrain \
    --pretrain_epochs 20 \
    --pretrain_lr 0.01 \
    --aux_size 200 \
    --total_loops 5 \
    --alpha 1 \
    --batch_size 8 \
    --seed 0
```

---

## 📋 Arguments Reference

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--dataset` | Target dataset (`cifar10`, `cifar100`, `mnist`, `svhn`) | `cifar10` |
| `--model` | Target architecture (`ResNet18`, `MobileNetV2`, `VGG16`) | `ResNet18` |
| `--unlearned_algo` | Unlearning method (`neggrad`, `neggradp`, `finetuning`, `scrub`, `retrain`) | `neggrad` |
| `--aux_size` | Number of unlearned forget samples ($|D_f|$) | `200` |
| `--total_loops` | Number of unlearning iteration loops | `5` |
| `--unlr` | Unlearning learning rate ($\eta$) | `0.01` |
| `--batch_size` | Batch size of samples removed simultaneously ($k = \|B\|$) | `8` |
| `--alpha` | Regularization scaling factor for retain loss | `1.0` |
| `--seed` | Random seed for reproducibility | `0` |

---

## 📊 Benchmark Summary

| Method | Target Dataset | Model | Top-1 iAcc (%) | Extra Training Cost | Attack Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Optimization / Inversion** | CIFAR-10 | ResNet-18 | ~68.4% | High (L-BFGS / GPU) | > 15 s |
| **Shadow Classifier** | CIFAR-10 | ResNet-18 | ~74.2% | Prohibitive (Shadow DNNs) | ~200 ms |
| **VLI (This Work)** | CIFAR-10 | ResNet-18 | **81.1%** | **0 GFLOPs (Zero)** | **< 1 ms** |
| **VLI (This Work)** | CIFAR-100 | MobileNetV2 | **80.1%** | **0 GFLOPs (Zero)** | **< 1 ms** |

---

## 👤 Author & Academic Citation

- **Author:** Le Hoang Dat (黎黄达)
- **Institution:** Ho Chi Minh City University of Science, Vietnam National University (VNU-HCMUS)
- **Advisors:** Assoc. Prof. Le Kim Hung, MSc. Tran Anh Duy
- **Major:** Computer Networking & Cybersecurity (AI & Privacy Specialization)

If you find this codebase or research useful, please cite our thesis:

```bibtex
@mastersthesis{le2026vectorlabelinference,
  author    = {Le, Hoang Dat},
  title     = {Vector-Based Label Inference Attack against Machine Unlearning},
  school    = {Ho Chi Minh City University of Science (VNU-HCMUS)},
  year      = {2026},
  address   = {Ho Chi Minh City, Vietnam},
  note      = {Graduation Thesis in Computer Networking and Cybersecurity}
}
```
