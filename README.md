# DL-LLM-Scripts
# Building Neural Networks and LLMs 

This repository demonstrates how to build a simple neural network using NumPy and Keras, as well as scripts for fine-tuning Large Language Models (LLMs) with HuggingFace Transformers and PEFT-LoRA.

## Contents

- **Building Simple Neural Network (NumPy):**
  - Shows how to implement a basic feedforward neural network for the XOR problem using only NumPy.
  - Includes forward and backward passes, weight updates, and training loop.

- **Building Simple Neural Network (Keras):**
  - Uses TensorFlow Keras to build and train a neural network for the XOR problem.
  - Demonstrates model creation, compilation, training, and prediction.

- **Fine-Tuning LLM using PEFT-LoRA:**
  - Offers a generalized script for fine-tuning a pretrained language model using PEFT (Parameter-Efficient Fine-Tuning) and LoRA (Low-Rank Adaptation).
  - Shows how to integrate LoRA with HuggingFace Transformers and perform training on custom data.

## Getting Started

To run the notebook, you'll need:

- Python 3.x
- Required Python packages:
  - `numpy`
  - `tensorflow`
  - `transformers`
  - `datasets`
  - `peft` (for LoRA fine-tuning)

Install dependencies using pip:

```bash
pip install numpy tensorflow transformers datasets peft
```

## Usage

1. **Simple Neural Network (NumPy):**
   - The notebook demonstrates building and training a neural network for the XOR problem.
   - You can modify the input data and network architecture as needed.

2. **Simple Neural Network (Keras):**
   - Uses Keras for building and training a similar network.
   - You can adjust layers, activation functions, and training parameters.

3. **Fine-Tuning with PEFT-LoRA:**
   - Replace the placeholders with your specific model and dataset.
   - Adjust LoRA parameters (e.g., rank) as needed for your use case.
