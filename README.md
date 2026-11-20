# GPT from Scratch — Decoder-Only Transformer

**A hands-on language-model engineering project built in PyTorch.** I implemented a GPT-style decoder-only Transformer, experimented with pre-training on OpenWebText and instruction fine-tuning with Databricks Dolly 15k, and studied the model's outputs and limitations.

This is an **experimental learning project**, not a production chatbot or a claim of general-purpose language understanding.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![NLP](https://img.shields.io/badge/Focus-NLP%20%26%20Transformers-555555?style=flat-square)

## What I built

- A decoder-only Transformer with **causal multi-head self-attention**, token and position embeddings, residual connections, layer normalization, and feed-forward networks.
- A next-token language-modeling objective using cross-entropy loss.
- Training code with **AdamW**, gradient clipping, and a learning-rate schedule incorporating warmup and cosine decay.
- Generation with **top-k** and **top-p (nucleus) sampling**.
- An instruction fine-tuning experiment using prompt/response examples from Dolly 15k.

I used the **GPT-2 BPE tokenizer** through `tiktoken`; I did not train a tokenizer from scratch.

## Model architecture

The table reflects the **model actually constructed in [`src/model.py`](src/model.py)**, rather than older experiment notes.

| Component | Configuration |
| --- | --- |
| Architecture | Decoder-only Transformer |
| Transformer blocks | 12 |
| Attention heads per block | 4 |
| Embedding dimension | 512 |
| Context length | 128 tokens |
| Vocabulary | 50,257 tokens (GPT-2 BPE) |
| Feed-forward hidden dimension | 2,048 |
| Activation | ReLU |
| Dropout | 0.1 |
| Output projection | Separate, untied language-model head |
| Parameters | **~89.39 million, calculated from the published layer definitions** |

> **Architecture note:** The file also declares `n_head = 8` at module scope, but the actual constructor uses `Block(n_embd, n_head=4)`. The table reports the constructed model. The parameter count has been calculated from the source; it has not been checked against a published model checkpoint.

## Data and experiments

1. **Pre-training:** The code loads [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) through Hugging Face Datasets and trains a next-token prediction model. The published loss log contains measurements through **step 12,900**.
2. **Instruction fine-tuning:** The code formats examples from [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) into question/context/answer sequences. Two fine-tuning loss logs are included; the fine-tuning script defines an additional 3,000-step loop. These artifacts do not, by themselves, establish one complete end-to-end run history.
3. **Generation and evaluation:** The repository includes a text-generation script and recorded examples. An earlier experiment reported a WikiText-2 result, but no independently runnable WikiText-2 evaluation script is currently published.

## Results and observations

### Pre-training loss

The last recorded entry in [`results/pretrain_losses.csv`](results/pretrain_losses.csv) is:

| Logged step | Training loss | Validation loss |
| ---: | ---: | ---: |
| 12,900 | 4.2384 | 4.2395 |

These are **logged loss values**, not independently reproduced benchmark results.

![Recorded pre-training loss curve](results/loss_curve.png)

### Fine-tuning loss

Two experiment logs are available:

- [Fine-tuning run 1](results/finetune_run1_losses.csv)
- [Fine-tuning run 2](results/finetune_run2_losses.csv)

The final recorded loss is **2.4878** at step 1,900 in run 1 and **1.9288** at step 1,900 in run 2. These are training measurements, not held-out evaluation scores.

![Recorded fine-tuning loss curve](results/finetune_loss_curve.png)

### Text generation

In my experiments, generations progressed from largely incoherent text toward more structured responses on familiar prompts after fine-tuning. **This does not establish reliable generalization** to unseen questions. The model remained sensitive to training data, compute limits, and overfitting.

## Repository structure

```text
.
├── src/
│   ├── model.py       # Transformer architecture and sampling
│   ├── tokenizer.py   # GPT-2 tokenizer helper
│   ├── dataset.py     # Dataset loading
│   ├── train.py       # Checkpoint-resuming language-model training
│   ├── pretrain.py    # Dolly instruction fine-tuning (historical filename)
│   └── generate.py    # Checkpoint-based generation example
├── results/
│   ├── pretrain_losses.csv
│   ├── finetune_run1_losses.csv
│   ├── finetune_run2_losses.csv
│   ├── loss_curve.png
│   └── finetune_loss_curve.png
└── README.md
```

## Reproducibility status

**The published source is not yet a standalone, one-command training pipeline.** Some scripts expect variables initialized elsewhere (including the model, tokenizer, or dataset splits), and training/generation use Kaggle-specific checkpoint paths. Model weights, environment specifications, and an independently runnable evaluation script are not currently included.

The source and CSV logs are provided for inspection and learning; they should not be interpreted as a fully reproducible training release.

## What I learned

- How attention, masking, residual connections, normalization, and token embeddings fit together in an autoregressive Transformer.
- The practical difference between decreasing training loss and generating reliable answers.
- How training dynamics, checkpointing, resource constraints, and fine-tuning affect model behavior.
- Why reproducible evaluations and clearly documented limitations matter as much as model architecture.

---

**Author:** [Aziz Ferchichi](https://github.com/Azizfer) · [LinkedIn](https://www.linkedin.com/in/aziz-ferchichi-499742218)
