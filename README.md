Project Overview
This repository contains multiple fine-tuning pipelines and evaluation workflows for building and comparing transformer-based sarcasm detection models. Each notebook serves a specific purpose in training, dataset experimentation, and model validation.

UltimateSarcasmDetector.ipynb
Builds the Ultimate Sarcasm Detector models:

ultimate_sarcasm_detector_v1 is trained using 3 datasets:

2 English sarcasm datasets

1 Arabic sarcasm dataset Focused on cross-language robustness and multilingual sarcasm patterns.

ultimate_sarcasm_detector_v2 is trained using 7 English sarcasm datasets, fully combined and balanced. Designed for maximum generalization across English sarcasm variations.

SarcasmLLM.ipynb
Produces the fine-tuned RoBERTa model:

ftroberta_sarcasm is a RoBERTa-base sarcasm classifier trained on a curated English dataset cluster.

SalsaSarcasm.ipynb
Builds a dataset-specific model:

Finetuned_Salsa is trained using a single English sarcasm dataset. Used to compare pure single-dataset performance versus multi-dataset training.

NEWSSLLM.ipynb
Produces a news-style sarcasm detection model:

Finetuned_Sarcasm2 is trained using one English sarcasm dataset focused on news/headline-style content. Useful for testing domain-specific sarcasm detection.

Model_Testing.ipynb
Evaluates all trained models across various datasets. Includes:

Accuracy, Precision, Recall, F1

Confusion matrix visualization

Cross-dataset generalization testing

Robustness and misclassification investigation

Side-by-side comparison of all models

How to Use This Project:
This project includes Google Colab notebooks for fine-tuning and evaluating RoBERTa-base and RoBERTa-large sarcasm detection models. To reproduce the experiments or run the code, please follow the instructions below.

Models:
All the finetuned model have been uploaded to HuggingFace platform:

https://huggingface.co/AutumnSama27

AutumnSama27/Finetuned_Sarcasm2

AutumnSama27/ftroberta_sarcasm

AutumnSama27/ultimate_sarcasm_detector_v2

AutumnSama27/ultimate_sarcasm_detector_v1

AutumnSama27/Finetuned_Salsa

How to View the files:
Since GitHub previews may not render the .ipynb files correctly. To view or run them:

Download the notebooks from this repository.

Go to https://colab.research.google.com/

Click Upload and select the downloaded notebook.

The Code uses Local Runtime
This project was developed and tested on: Python 3.11

This project uses Local Runtime through jupyter notebook

The required libraries are : transformers datasets torch scikit-learn numpy pandas matplotlib seaborn huggingface_hub tqdm

Findings
The project shows that ultimate_sarcasm_detector_v1 performs better than v2 because training on only three datasets gives the model a clearer, more consistent definition of sarcasm, while v2, trained on seven mixed English datasets, struggles due to conflicting tones and labeling standards that make sarcasm harder to learn. RoBERTa-base proved the most stable overall, while SalsaSarcasm, the only model trained on RoBERTa-large, demonstrated unexpectedly strong generalization and performed well even on external datasets such as jokerdD0727/Sarcasm_Detection, despite being trained on only one dataset. Other single-dataset models like Finetuned_Sarcasm2 performed well in-domain but were less consistent on external benchmarks. Across all models, subtle or context-heavy sarcasm remains the most challenging to detect.
