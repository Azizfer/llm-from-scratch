# GPT from Scratch — Building a Transformer Language Model

**An experimental GPT-style language model implemented from scratch in PyTorch.**

This project explores the architecture, training, fine-tuning, and evaluation of a decoder-only Transformer. Starting with the fundamental building blocks of neural networks and self-attention, I developed a language model and experimented with training it on OpenWebText and instruction fine-tuning it on Databricks Dolly 15k.

The goal was to develop a deeper practical understanding of **how large language models work**, from their mathematical foundations to their behavior during training and text generation.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Transformers-555555?style=flat-square)
![Status](https://img.shields.io/badge/Status-Experimental-orange?style=flat-square)

---

## Project Overview

Rather than using a pretrained language model, I implemented the main components of a GPT-style Transformer to understand how they interact throughout the training and generation process.

**Key components implemented:**

- **Decoder-only Transformer:** Causal self-attention, multi-head attention, positional embeddings, residual connections, and layer normalization.
- **Language modeling:** Autoregressive next-token prediction using cross-entropy loss.
- **Training pipeline:** AdamW optimization, gradient clipping, checkpointing, and learning-rate scheduling.
- **Text generation:** Token sampling with top-k and top-p (nucleus) filtering.
- **Instruction fine-tuning:** Experiments with question-answer data to encourage more structured responses.

The model uses the GPT-2 BPE tokenizer provided by `tiktoken`. The Transformer architecture was implemented manually in PyTorch; the tokenizer itself was not trained from scratch.

## Model Architecture

The following configuration reflects the model instantiated in [`src/model.py`](src/model.py).

| Component | Configuration |
|---|---|
| Model type | Decoder-only Transformer |
| Parameters | Approximately 89.39 million |
| Transformer blocks | 12 |
| Attention heads per block | 4 |
| Embedding dimension | 512 |
| Context window | 128 tokens |
| Vocabulary size | 50,257 tokens |
| Tokenizer | GPT-2 BPE |
| Feed-forward dimension | 2,048 |
| Activation function | ReLU |
| Dropout | 0.1 |
| Output layer | Untied language-model head |

**Implementation note:** Although `n_head = 8` is declared in the source, the model constructor explicitly instantiates four attention heads per block. The parameter count is calculated from the published architecture.

## Training and Fine-Tuning

### 1. Pre-training

The model was trained using the [OpenWebText dataset](https://huggingface.co/datasets/Skylion007/openwebtext).

The objective was to predict the next token in a sequence, allowing the model to learn statistical relationships between tokens.

The training implementation includes:

- Mini-batch optimization with AdamW.
- A learning-rate scheduler with warmup and cosine-decay logic.
- Gradient clipping to stabilize training.
- Periodic loss monitoring and checkpoint saving.

The published training logs include measurements through **12,900 steps**.

### 2. Instruction Fine-Tuning

After pre-training, I experimented with instruction fine-tuning using [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k).

The dataset contains instruction-response examples, sometimes accompanied by additional context.

The objective was to encourage the model to generate responses in a more structured question-answer format.

These experiments helped me explore the difference between learning general text patterns and adapting a model to follow specific response formats.

### 3. Text Generation

The model generates text autoregressively, predicting one token at a time.

To control the sampling process, I implemented:

- **Top-k sampling:** Restricts sampling to the highest-probability tokens.
- **Top-p sampling:** Selects tokens from a probability distribution up to a cumulative probability threshold.
- **Temperature-independent probability sampling:** Samples from the filtered model logits.

Generation experiments highlighted the influence of training quality, context length, and fine-tuning on the model's responses.

---

## Experimental Results

### Pre-training Loss

The final recorded entry in the published pre-training log is:

| Step | Training Loss | Validation Loss |
|---:|---:|---:|
| 12,900 | 4.2384 | 4.2395 |

The training history shows a substantial reduction in loss compared with the beginning of training.

![Pre-training loss curve](results/loss_curve.png)

**Observation:** Training and validation losses remained relatively close in the published measurements, although these results alone are not sufficient to establish generalization performance.

### Fine-Tuning Loss

Two fine-tuning experiment logs are available in the repository.

| Experiment | Last Logged Step | Recorded Loss |
|---|---:|---:|
| Run 1 | 1,900 | 2.4878 |
| Run 2 | 1,900 | 1.9288 |

![Fine-tuning loss curve](results/finetune_loss_curve.png)

These values represent recorded training losses rather than independent benchmark scores.

The original experiments also included a reported WikiText-2 evaluation, but the repository does not currently contain a standalone script to reproduce that benchmark.

### Text Generation Observations

The model initially produced largely incoherent text.

After training and instruction fine-tuning, its outputs became more structured, especially when responding to prompts resembling examples encountered during training.

This progression demonstrated the role of fine-tuning in shaping language-model behavior.

However, the model continued to exhibit important limitations, including:

- Difficulty generalizing to unfamiliar questions.
- Sensitivity to prompt wording and context.
- Computational limitations during training.
- Challenges involving overfitting and response reliability.

**The project is an experimental language model, not a production-ready conversational assistant.**

---

## Repository Structure

**Source code**

- [`src/model.py`](src/model.py) — Transformer architecture, forward propagation, and text sampling.
- [`src/tokenizer.py`](src/tokenizer.py) — GPT-2 tokenizer helper functions.
- [`src/dataset.py`](src/dataset.py) — Dataset loading.
- [`src/train.py`](src/train.py) — Language-model training and checkpoint handling.
- [`src/pretrain.py`](src/pretrain.py) — Instruction fine-tuning experiments (historical filename).
- [`src/generate.py`](src/generate.py) — Text-generation example.

**Experimental results**

- [`results/pretrain_losses.csv`](results/pretrain_losses.csv) — Pre-training loss measurements.
- [`results/finetune_run1_losses.csv`](results/finetune_run1_losses.csv) — First fine-tuning experiment.
- [`results/finetune_run2_losses.csv`](results/finetune_run2_losses.csv) — Second fine-tuning experiment.
- [`results/loss_curve.png`](results/loss_curve.png) — Pre-training visualization.
- [`results/finetune_loss_curve.png`](results/finetune_loss_curve.png) — Fine-tuning visualization.

## Reproducibility

This repository contains the source code and recorded results from my experiments.

The scripts were originally developed in an interactive Kaggle environment and currently depend on some externally initialized variables and environment-specific checkpoint paths.

Consequently, the published files are intended primarily for exploring the implementation and experiments rather than providing a fully automated training pipeline.

A standalone training workflow and independently reproducible benchmark evaluation are potential future improvements.

---

## Key Takeaways

This project allowed me to move beyond using existing language models and explore the engineering challenges involved in building one.

Some of the most valuable lessons included:

- Understanding how causal attention, token embeddings, residual connections, and normalization work together.
- Implementing an autoregressive training objective and learning-rate scheduling.
- Exploring the effects of pre-training and instruction fine-tuning.
- Managing computational constraints and training instability.
- Recognizing the difference between decreasing training loss and achieving reliable text generation.
- Understanding the importance of experimental evaluation, reproducibility, and model limitations.

More importantly, the project reinforced my interest in **deep learning, natural language processing, and scientific research**.

---

## Connect

**Mohamed Aziz Ferchichi**

Computer Science Graduate | Data Analytics & Science Engineering Student

- [GitHub](https://github.com/Azizfer)
- [LinkedIn](https://www.linkedin.com/in/aziz-ferchichi-499742218)

*Always learning, experimenting, and building.*
