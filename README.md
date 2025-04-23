# Handwritten Mathematical Equation Transcription and Correction

CSE 60880: Neural Networks

John Kim (dkim37@nd.edu), Tram Trinh (htrinh@nd.edu)

## Part 1: High-Level Solution (02/09/25)

### Overview

The goal is to develop a model that transcribes handwritten mathematical equations into syntactically and semantically correct LaTeX code. The pipeline involves two stages:

1. **Stage 1: RNN/Transformer Transcription**
   
![Figure 1: An example of handwritten mathematical equation](figure-1.png)

Since we are working with sequential data (InkML format), we will experiment with RNN and Transformers for transcribing handwritten equations into raw LaTex. The model will be built from scratch to allow completer control over architecture and training. We believe the main challenge is to how to take advantages of sequential data to extracts relevant features and maps them to a sequence of tokens corresponding to LaTex commands (e.g., `\frac`, `\sqrt`, `^`, etc.)
- Feature Extraction: Learn high-level features distinguishing different mathematical symbols.
- Sequence Mapping: Output an ordered sequence of tokens that accurately represents the mathematical expression.
- Handling Varying Input Quality.

2. **Stage 2: LLM Correction**

The raw LaTex output from the transcription model is processed by a LLM. The LLM refines the transcription by resolving ambiguities, fixing syntax errors, and ensuring structural consistency.

Example: `\cos(0) = 1` might be incorrectly transcribed but fixed at this stage.

We are planning to experience some open-source LLMs are are free and suitable for this task. Some of our options are:
- [LLaMa](https://www.llama.com/): Released by Meta, a collection of models ranging from 7 billion to 70 billion parameters and is designed to be efficient and effective across multiple tasks, including language understanding and generation.
- [BLOOM](https://huggingface.co/bigscience/bloom): BLOOM is a 176-billion-parameter multilingual model, and it is open-access and has been trained on a diverse dataset.
- [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1): Mistral AI's 7.3-billion-parameter model employs grouped-query attention for optimized performance.

This two-stage approach helps in addressing key challenges in handwritten equation recognition, such as symbol ambiguities (e.g., `∆` vs. `∇`),  structural relationships (e.g., fractions, matrices), and the high variability in handwriting styles.

### Dataset

We plan to use the data set described in the following paper: https://arxiv.org/pdf/2404.10690

The dataset consists of:
- 230,000 human-written samples
- 400,000 synthetic samples
- 244 mathematical symbols + 10 syntactic tokens
- Categories of symbols:
  - Latin letters (a-z, A-Z)
  - Numbers (0-9)
  - Punctuation and symbols (.;:+-/… etc.)
  - Greek characters
  - Mathematical constructs (\frac, \sqrt, etc.)
  - Structural elements (nested expressions, matrices, binomial coefficients, etc.)
 
We are also planning on generating a subset of the test dataset with our own handwritten math equations with a Wacom tablet.

### Discussion 

One of the biggest challenges we face in handwritten mathematical expression recognition is the high variability in handwriting styles. Differences in stroke patterns, writing pressure, and individual character formation make it difficult to generalize across users.
- Cursive and print letters and characters
- Overlapping strokes: Complex expressions like summations (\sum), integrals (\int), or nested fractions often have overlapping strokes.
- Distinguishing similar looking symbols, such as `0` vs. `O`, `1` vs. `l`, and `∆` vs. `∇`.

Unlike traditional OCR tasks, we need to capture the hierarchical structure of symbols to recognize mathematical equations accurately. Our model must identify individual symbols and their relationships, distinguish between different mathematical constructs like matrices, fractions, and binomial coefficients, and ensure the correct placement of subscript and superscript elements. To achieve accurate transcription, we need to effectively detect symbols, strokes, and structural relationships, preserving the intended meaning of handwritten equations.

### Part 1 Contributions

We both brainstormed the project idea and the high-level design. We met with Professor Czajka for questions, and finishing the report for Part 1 together.

## Part 2: Dataset

### Introduction

For this phase, we continue to utilize the dataset that was described in Part 1 as it provides a large variety of handwritten mathematical expressions. We have physically downloaded it from sources below:

- Paper: [MATHWRITING: A Large-Scale Handwritten Math Expression Dataset](https://arxiv.org/pdf/2404.10690)
- GitHub Repository: [The MathWriting Dataset: Online Handwritten Mathematical Expressions](https://github.com/google-research/google-research/tree/master/mathwriting)

The dataset is split into train, validation, test, symbols, and synthetic subsets. A core aspect of this splitting strategy is that each contributor ID (i.e., each individual writer) belongs to only one subset (either training, validation, or testing). This ensures that the model is exposed to truly unseen handwriting styles in the test set, preventing overfitting to any specific set of writer characteristics.

1. Training Set:
- Primary partition used for learning model parameters (weights, biases).
- Comprises the majority of the data to capture wide-ranging handwriting styles and symbols.

2. Validation Set:
- Used to tune hyperparameters and conduct early stopping checks.
- Conatins a distinct set of contributiors to test intermediate model generalization.

3. Test Set:
- "Unknown" subset, kept separate until final evaluation.
- Writers here do not appear in the training or validation sets, ensuring unbiased performance metrics.

The dataset aims for minimal overlap between train and test labels (around 8%), contrasting with a higher overlap (about 55%) between train and validation. This measures the model's capacity to handle truly new symbols that it may not encounter during training.

In addition to human-contributed data, the dataset includes synthetic samples. These augment underrepresented symbols and accommodate longer equations that might not fit well on a physical tablet.

### Consideration of MNIST for Pre-Training

We received a comment from Part 1 recommending MNIST as a pre-training dataset for digit recognition. While our current dataset already includes digits, we see value in using MNIST to:

- Placing MNIST digits randomly on a plain background provides a simplified environment for identifying digit shapes.
- Pre-training on a straightforward task (isolated digits) could yield faster or more stable convergence when transitioning to more complex expressions.

Our plan, if time permits, is to use MNIST-based digit placement as an optional pre-training phase. This would help reinforce digit recognition before tackling the nuanced challenges of full mathematical expressions in MATHWRITING.

### Data Cleansing and Preprocessing

- Handwritten Samples:
   - Filtering: Unreadable equations will be removed.
   - Consistency: Writers and stroke patterns are maintained, ensuring each contributor’s work remains grouped in a single subset.
- LaTeX Equations:
   - Multiple LaTeX notations for the same expression are standardized to a single canonical form. However, the unnormalized versions remain available for reference.
   - Uniform tokens (like `\sqrt`, `\frac`, etc.) help the model map from strokes to semantically consistent LaTeX.

The goal is to ensure a clean, coherent dataset where each equation is valid and easily comparable.

### Part 2 Contributions

We worked together to download and organize the MATHWRITING dataset, ensuring proper subset splits (train/validation/test) while addressing the suggested considerations for unseen handwriting styles. We discussed the benefits of pre-training on MNIST and agreed to keep this option open for improving digit detection. We also prepared this report section by coordinating our individual tasks and reviewing each other’s work for clarity and coherence.

## Part 3: Inital Setup and First Model Architecture

### Inital Setup

John has been working on the dataset and the basic setup for our project. He wrote scripts to clean and prepare the InkML files (which contain the handwritten strokes). He implemented code that reads these files and extract useful features like stroke position, speed, and curvature.

John also built the Data Loader to load the data, and a special collate function was made to pad sequences of varying lengths so that we can process them in batches.

### Model Architecture

We both researched on designing the model together, and Tram implemented the architecture. Our model is based on a simple sequence-to-sequence design with an attention mechanism. Here is a simple explanation of our current design:

#### Encoder

The encoder reads the sequence of handwritten stroke features and creates a summary of the information. We use a bidirectional LSTM network, being directional meaning that it should be able to read the sequence from the start and from the end at the same time for better understanding of the context.

#### Attention Mechanism

The attention part helps the decoder focus on the important parts of the encoded information when creating each LaTeX token. At each step, the decoder will look at all the encoder outputs and decides which parts are most important to generate the next token. Tram believed that this “soft alignment” would make the model flexible when dealing with different handwriting styles.

#### Decoder

The decoder takes the information from the encoder and the attention module to generate LaTeX code, one toke at a time. It is implemeted as an LSTM that works step by step. The decoder uses an embedding layer to turn token numbers into a vector space, which makes it easier to work with. We decided to try `teacher forcing` during training in order to feed the correct token (from our ground truth) into the decoder at each step to help it learn faster.

#### Connecting Encoder and Decoder

The encoder is bidirectional, so its hidden state is made up of two parts (one for reading from the start and one from the end). We combine these two parts (by summing them) so that the hidden state matches what the decoder expects. This is important to ensure that the model uses information from both directions.

## Why This Architecture?

Our design is inspired by several projects that uses similar ideas:
- **CROHME (Competition on Recognition of Online Handwritten Mathematical Expressions):** Many projects in this challenge use encoder-decoder architecture with attention to handle the complex layouts of math expressions, so we decided to give it a try.
- **[Im2LaTeX](https://github.com/d-gurgurov/im2latex)**: This is a project that converts images (.PNG) of math expressions into LaTeX code using neural networks. 
- **Simple OCR and Sequence-to-Sequence Models:** Other simpler projects, like basic OCR systems for handwritten digits (using MNIST), have shown that sequence models can learn and convert images or strokes into text. These projects helped us understand the basics before moving on to more complex math expressions.

## Challenges

As we move forward, we are encountering several challenges and open questions that we hope to discuss with Adam and Rasel for further guidance:

- Handwritten mathematical expressions show significant variability in style, stroke order, and clarity. This variability makes it difficult for the model to generalize well across different writers. How can we further normalize or augment our data to better capture this variability? Are there additional preprocessing steps or features (e.g., temporal dynamics) that might help?
- We ran model on the MPS backend (MacOS) for debugging and encountered memory limitations. Although we have reduced the batch size, memory consumption remains a concern. Although, we are planning to train the model using GPU as the next step, we are still wondering whether we should optimize the model to help reduce memory usage.
- Our model currently uses a set of hyperparameters (e.g., number of layers, hidden dimensions, learning rate) that were chosen based on preliminary experiments. However, fine-tuning these parameters is important for achieving optimal performance. Should we consider automated hyperparameter tuning methods, such as grid search or Bayesian optimization?
- Our project plan includes a second stage where an LLM is used to correct the raw LaTeX output. We are currently exploring which open-source LLM would be best suited for this task. How should we interface the output of our transcription model with the correction module?

## Part 3 Contributions
- Team:
   - Reseached projects to guide our design.
   - Designed the model architecture using a sequence-to-sequence approach with an attention mechanism.
- John Kim:
   - Downloaded, extracted, and organized the MathWriting dataset.
   - Developed preprocessing scripts and implemented the PyTorch Dataset/DataLoader.
   - Made sure that each data split (train/validation/test) has unique handwriting styles.
- Tram Trinh:
   - Implemented the encoder, attention module, and decoder for converting handwritten strokes into LaTeX.
   - Developed a method to combine the bidirectional encoder outputs to fit the decoder.


 
## Part 4 Contributions
- Team:
   - Researched how to implement evaluation metrics 
   - Developed part of evaluation metrics that were used.
- John Kim:
   - Created test model and inference functions to allow for evaluation and working with the model
   - ...
- Tram Trinh:
   - Implemented the encoder, attention module, and decoder for converting handwritten strokes into LaTeX.
   - ...
