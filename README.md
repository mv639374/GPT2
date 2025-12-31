# GPT-2 from scratch

## A picture is worth of thousands of words

<img src="assets/GPT2.jpeg" alt="GPT-2 Architecture" style="max-width: 15cm; max-height: 30cm; width: auto; height: auto;">

## Overview

This repository contains the code and materials for building a GPT-2 language model from scratch, following the methodology and explanations from the book "Build a Large Language Model (From Scratch)" by Sebastian Raschka.

## Structure

1. Working with Text Data (Tokenization & Embeddings)
2. Attention Mechanisms (Self Attention, Masking & Multi-Head Causal Attention)
3. Transformer & LLM Architecture (Layer Normalization, Feed-Forward Networks, Shortcut Connections)
4. Model Pre-Training and Weight-Loading
5. Finetuning for Text Classification
6. Finetuning to follow instructions

## Files & Folders

- `gpt2.py`: Main GPT2 model implementation with all necessary functions (can run the script directly)
- `gpt_download.py`: Model weights download script
- `0_Basics`: Architecture basics (Tokenization, Embeddings, Attention Mechanism, Transformer)
- `1_PreTraining`: PreTraining the LLM and importing model weights
- `2_FineTuning_Classification`: FineTuning the model for ham-spam classification tasks
- `3_FineTuning_Instructions`: FineTuning the model for Instruction following tasks
- `data`: necessary data for pre-training and finetuning
- `notebooks`: practice notebooks

---

# Detailed Overview of Each Component

* Below are the step by step details for each component.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg" width="650px">

---

## 1. Working with Text Data

This section covers tokenization and embedding implementations.

### Tokenization

* Used Byte Pair Encoding (from tiktoken library) to tokenize text, it breaks words into sub-words or characters.
* Vocabulary consists of 50257 tokens. Text is now represented as token indices. Only one special token included: `<|endoftext|>` for end-of-sequence detection.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/11.webp" width="300px">

### Data Sampling (with sliding window)

* We train LLMs to generate one word at a time, so we want to prepare the training data accordingly where the next word in a sequence represents the target to predict.
* For each text chunk, we want the inputs and targets.
* Since we want the model to predict the next word, the targets are the inputs shifted by one position to the right.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/13.webp?123" width="500px">

### Embeddings

- Convert token ids to token embeddings of dimension 384 or 768 (matching GPT-2 small architecture)
- To encode positional meaning of each token, convert each token to positional embeddings of dimension equal to context size.
- Sum token embeddings and positional embeddings to get final input embeddings

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/19.webp" width="400px">

---

## 2. Attention Mechanisms

This section covers self-attention, masking, and multi-head causal attention implementations.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/02.webp" width="600px">

### History

* The problem with modeling long sequences: Translating a text word by word isn't feasible due to the differences in grammatical structures between the source and target languages
* Prior to the introduction of transformer models, encoder-decoder RNNs were commonly used for machine translation tasks
* In this setup, the encoder processes a sequence of tokens from the source language, using a hidden state—a kind of intermediate layer within the neural network—to generate a condensed representation of the entire input sequence.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/04.webp" width="500px">

* Capturing data dependencies with attention mechanisms: Through an attention mechanism, the text-generating decoder segment of the network is capable of selectively accessing all input tokens, implying that certain input tokens hold more significance than others in the generation of a specific output token

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/05.webp" width="500px">

* Self-attention in transformers is a technique designed to enhance input representations by enabling each position in a sequence to engage with and determine the relevance of every other position within the same sequence

### Self-Attention

- In self-attention, the process starts with the calculation of attention scores, which are subsequently normalized to derive attention weights that total 1
- These attention weights are then utilized to generate the context vectors through a weighted summation of the inputs

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/12.webp" width="400px">

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/17.webp" width="600px">

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/18.webp" width="400px">

### Masking & Dropout (Causal Attention)

- Causal self-attention ensures that the model's prediction for a certain position in a sequence is only dependent on the known outputs at previous positions, not on future positions
- In simpler words, this ensures that each next word prediction should only depend on the preceding words
- In addition, we also apply dropout to reduce overfitting during training
- Dropout can be applied in several places:

  - for example, after computing the attention weights;
  - or after multiplying the attention weights with the value vectors
- Here, we will apply the dropout mask after computing the attention weights because it's more common
- Furthermore, in this specific example, we use a dropout rate of 50%, which means randomly masking out half of the attention weights. (When we train the GPT model later, we will use a lower dropout rate, such as 0.1 or 0.2)

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/22.webp" width="400px">

- Note that dropout is only applied during training, not during inference

### Multi-Head Causal Attention

- Below is a summary of the self-attention implemented previously (causal and dropout masks not shown for simplicity)
- This is also called single-head attention:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/24.webp" width="400px">

- We simply stack multiple single-head attention modules to obtain a multi-head attention module:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/25.webp" width="400px">

- The main idea behind multi-head attention is to run the attention mechanism multiple times (in parallel) with different, learned linear projections. This allows the model to jointly attend to information from different representation subspaces at different positions.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/26.webp" width="400px">

## 3. LLM Architecture (Transformer Blocks)

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}
```

### GELU Activation (Feed Forward Layer)

- In deep learning, ReLU (Rectified Linear Unit) activation functions are commonly used due to their simplicity and effectiveness in various neural network architectures
- In LLMs, various other types of activation functions are used beyond the traditional ReLU; two notable examples are GELU (Gaussian Error Linear Unit) and SwiGLU (Swish-Gated Linear Unit)
- GELU and SwiGLU are more complex, smooth activation functions incorporating Gaussian and sigmoid-gated linear units, respectively, offering better performance for deep learning models, unlike the simpler, piecewise linear function of ReLU
- GELU ([Hendrycks and Gimpel 2016](https://arxiv.org/abs/1606.08415)) can be implemented in several ways; the exact version is defined as GELU(x)=x⋅Φ(x), where Φ(x) is the cumulative distribution function of the standard Gaussian distribution.
- In practice, it's common to implement a computationally cheaper approximation: $\text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}} \cdot \left(x + 0.044715 \cdot x^3\right)\right]\right)$ (the original GPT-2 model was also trained with this approximation)
- As we can see, ReLU is a piecewise linear function that outputs the input directly if it is positive; otherwise, it outputs zero
- GELU is a smooth, non-linear function that approximates ReLU but with a non-zero gradient for negative values (except at approximately -0.75)

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch04_compressed/10.webp" width="400px">

### Shortcut Connection

- Originally, shortcut connections were proposed in deep networks for computer vision (residual networks) to mitigate vanishing gradient problems
- A shortcut connection creates an alternative shorter path for the gradient to flow through the network

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch04_compressed/13.webp?1" width="400px">

* Transformer Block summarized:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch04_compressed/14.webp?1" width="400px">

### Generating Text with the GPT Model

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch04_compressed/17.webp" width="800px">

---

## 4. Pre-Training

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/01.webp" width=800px>

* Topics covered here:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/02.webp" width=600px>

### Evaluating generative text models

- To train the model, we need to know how far it is away from the correct predictions (targets)

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/06.webp" width=800px>

- We want to maximize all these values, bringing them close to a probability of 1
- In mathematical optimization, it is easier to maximize the logarithm of the probability score than the probability score itself
- This is because the logarithm converts products into sums and avoids numerical underflow issues
- The cross-entropy loss is defined as: $L = -\frac{1}{N}\sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$ where $y$ is the one-hot encoded target and $\hat{y}$ is the predicted probability distribution
- This loss function measures the average negative log-likelihood of the correct tokens across all positions and sequences in the batch.
- The goal during pre-training is to minimize this loss, which encourages the model to assign higher probabilities to the correct next tokens in the training data.
- This process helps the model learn to predict the next word in a sequence with high confidence, forming the foundation for its ability to generate coherent text.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/07.webp" width=800px>

- A concept related to the cross-entropy loss is the perplexity of an LLM
- The perplexity is simply the exponential of the cross-entropy loss

```python
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)  # flattened logits and targets (batch x num_tokens)
perplexity = torch.exp(loss)
```

- The perplexity is often considered more interpretable because it can be understood as the effective vocabulary size that the model is uncertain about at each step
- In other words, perplexity provides a measure of how well the probability distribution predicted by the model matches the actual distribution of the words in the dataset
- Similar to the loss, a lower perplexity indicates that the model predictions are closer to the actual distribution

### Training & Validation Set

- We divide the dataset into a training and a validation set and use the data loaders to prepare the batches for LLM training
- For visualization purposes, the figure below assumes a `max_length=6`, but for the training loader, we set the `max_length` equal to the context length that the LLM supports
- The figure below only shows the input tokens for simplicity
- Since we train the LLM to predict the next word in the text, the targets look the same as these inputs, except that the targets are shifted by one position

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/09.webp" width=800px>

### Training an LLM

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/11.webp" width=500px>

### Decoding Strategies to control randomness

- Even if we execute the `generate_text_simple` function above multiple times, the LLM will always generate the same outputs
- We now introduce two concepts, so-called decoding strategies, to modify the `generate_text_simple`: *temperature scaling* and *top-k* sampling
- These will allow the model to control the randomness and diversity of the generated text

`temperature scaling`

- Instead of determining the most likely token via `torch.argmax`, we use `torch.multinomial(probas, num_samples=1)` to determine the most likely token by sampling from the softmax distribution
- We can control the distribution and selection process via a concept called temperature scaling
- "Temperature scaling" is just a fancy word for dividing the logits by a number greater than 0
- Temperatures greater than 1 will result in more uniformly distributed token probabilities after applying the softmax
- Temperatures smaller than 1 will result in more confident (sharper or more peaky) distributions after applying the softmax

`top-k sampling`

- To be able to use higher temperatures to increase output diversity and to reduce the probability of nonsensical sentences, we can restrict the sampled tokens to the top-k most likely tokens:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/15.webp" width=500px>

### Loading and Saving Model Weights

- Training LLMs is computationally expensive, so it's crucial to be able to save and load LLM weights

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/16.webp" width=400px>

- The recommended way in PyTorch is to save the model weights, the so-called `state_dict` via by applying the `torch.save` function to the `.state_dict()` method:

```python
torch.save(model.state_dict(), "model.pth")

model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
```

- It's common to train LLMs with adaptive optimizers like Adam or AdamW instead of regular SGD
- These adaptive optimizers store additional parameters for each model weight, so it makes sense to save them as well in case we plan to continue the pretraining later:

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    }, 
    "model_and_optimizer.pth"
)
```

### Load Pre-Trained Weights

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/17.webp" width=500px>



---



## 5. FineTuning for Classification Tasks

## 6. FineTuning for Instructions following Tasks