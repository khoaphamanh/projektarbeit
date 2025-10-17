# An Empirical Analysis of Fine-Tuning LargeLanguage Models for Predictive Maintenance
In this project, we will use large language models (LLMs) for classification tasks in the field of predictive maintenance. The project is inspired by and based on the paper [Empirical study on fine-tuning pre-trained large language models for fault diagnosis of complex systems](https://www.sciencedirect.com/science/article/abs/pii/S095183202400454X) [1].

## Datasets
The data used in this work is tabular. Each row corresponds to an instance, and the columns store the associated features. We rely on two datasets in this project.

### High speed train breaking system
The High Speed Train (HST) braking system dataset contains 22,368 tabular instances with 46 features in total: 15 continuous and 31 categorical. It has two labels, 0 (normal) and 1 (anomaly), and is highly imbalanced with 21,979 normal instances and only 389 anomalous ones. The original data can be downloaded from this [link](https://www2.ie.tsinghua.edu.cn/rrml/Resources.html), but the training scripts fetch it automatically so no manual download is required.

### Tennessee Eastman Process (TEP) Dataset
The Tennessee Eastman Process (TEP) dataset is also tabular. It provides 21 labels in total, where label 0 denotes normal operation and labels 1 through 20 represent different faults. The training split contains 250,000 instances per label, while the test split offers 480,000 instances per label. This dataset is balanced and the features are explicitly named.The original data can be downloaded from this [link](https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset?resource=download)

## Preprocessing
### Downsampling
Both datasets are large for LLM training, so we downsampled them to speed up experiments. For HST we keep 240 training instances and 60 test instances with balanced labels. For TEP we retain 400 training instances and 160 test instances, using only labels 0, 1, 4, and 5 in a balanced manner. This downsampling strategy follows paper [1].

### Normalization
We standardize the features using z-score normalization.

### Convert to Text
#### Tabular to Text
Because we work with an LLM, we convert each tabular instance into text. For example, if we have the instance
| setting_1 | setting_2 | condition | sensor_1 | sensor_2 | label |
|------------|------------|------------|-----------|-----------|--------|
| 1 | 1 | 2 | 0.05 | 0.7 | 1 |

the textual representation becomes

**Question:**  
Tell me if the value of Y is 0 or 1. If feature setting_1 = 1, feature setting_2 = 1, feature condition = 2, feature sensor_1 = 0.05, feature sensor_2 = 0.7. What should be Y?  

**Answer:**  
Y = 0

#### System behavior
Beyond converting to text, we must follow the template expected by the LLaMA 2 model, so the prompts need to respect its syntax:
```
<s>[INST] <<SYS>>
System prompt
<</SYS>>
User prompt [/INST] Model answer </s>

where:

- <s>: Start of sequence token.
- [INST] ... [/INST]: Marks the instruction block, what the user says.
- <<SYS>> ... <</SYS>>: Optional system message that sets assistant behavior.
- Model answer: The model’s response follows immediately after [/INST].
- </s>: End of sequence token.

Accordingly, when applying the conversion from tabular to text format, following the syntax of LLaMA 2, one instance would look like this. It should be noted that we refer to the system behavior from the paper that we aim to reimplement [4].

<s>[INST] <<SYS>>
You are an expert in fault diagnosis of chemical plants operation. You master the reaction process and control structures in the Fault Detection Dataset.  
You are capable of accurately determining the plant process state based on given variables and their values. Below is a sample of the Fault Detection Dataset monitoring.
<</SYS>>
Tell me if the value of Y is 0 or 1. If feature setting_1 = 1, feature setting_2 = 1, feature condition = 2, feature sensor_1 = 0.05, feature sensor_2 = 0.7. What should be Y? [/INST] Y = 0 </s>
```

## Model Selection
In this work we fine-tune [LLaMA 2](https://arxiv.org/abs/2307.09288) [2] in a supervised setting. We expect the model output to contain five tokens in total—two structural tokens ($[/INST]$ and $</s>$) plus the label token `Y`.
However, the full model is too large to fine-tune directly, so we adopt [Quantized Low-Rank Adaptation](https://arxiv.org/abs/2305.14314) (Q-LoRA) [3] to update only a subset of parameters. We attach LoRA adapters to every layer in the network.

## Hyperparameter Optimization Tunning
We use the loss on the validation set as the objective function for selecting the best hyperparameters. The search covers:

lora_r: dimension of trainable matrix

lora_alpha: alpha as scale factor in LoRA

lora_dropout: drop out rate in LoRA

normalize: should the input normalize with standardize

learning_rate: learning rate of the model

## Result
In this project we not only fine-tune an LLM but also compare it with other machine learning models

##### Table 4.1: Best hyperparameters and metrics

|                         | Dataset | HST   | TEP   |
|-------------------------|----------|-------|-------|
| **Hyperparameters**     | lora_r | 225 | 77 |
|                         | lora_alpha | 76 | 58 |
|                         | lora_dropout | 0.95 | 0.2 |
|                         | normalize | False | False |
|                         | learning_rate | 77e-5 | 22e-5 |
|                         | name_feature | False | False |
| **Metrics**             | Loss Train HPO | 0.083 | 0.079 |
|                         | Accuracy Train HPO | 0.878 | 0.800 |
|                         | Loss Validation | 0.112 | 0.844 |
|                         | Accuracy Validation | 0.800 | 0.717 |
|                         | Loss Train | 0.086 | 0.091 |
|                         | Accuracy Train | 0.858 | 0.942 |
|                         | Loss Test | 0.094 | 1.258 |
|                         | Accuracy Test | 0.883 | 0.731 |


##### Table 4.2: Train and Test Accuracy for Different Models on HST and TEP Datasets

| Dataset | Model | Train Accuracy | Test Accuracy |
|----------|--------|----------------|----------------|
| **HST** | Random Forest | 1.000 | 0.850 |
|          | Logistic Regression | 0.867 | 0.800 |
|          | Support Vector Machine | 0.854 | 0.767 |
|          | K-Nearest Neighbors | 0.771 | 0.717 |
|          | Gradient Boosting | 1.000 | 0.867 |
|          | LLaMA 2 | 0.858 | 0.883 |
| **TEP** | Random Forest | 0.985 | 0.819 |
|          | Logistic Regression | 0.900 | 0.694 |
|          | Support Vector Machine | 0.573 | 0.506 |
|          | K-Nearest Neighbors | 0.580 | 0.569 |
|          | Gradient Boosting | 0.985 | 0.844 |
|          | LLaMA 2 | 0.942 | 0.731 |

#### Explanation of the Decisions

In this work, the main reason we chose to use a Large Language Model (LLM) was our belief that an LLM could not only make predictions about machine system faults, but also provide explanations for the decisions it made. However, in practice, the LLM failed to offer any reasoning to support its decisions; in fact, it even generated confusing and incoherent text. Below is our chat log.

In the log, **User** represents the prompt we provided; we took an instance from the dataset. **Assistant 1** refers to the case where we limited the LLM to generate only 5 tokens, as done during training, and the model was still able to produce the correct answer. **Assistant 2** refers to the case where we allowed the model to generate up to 50 tokens.

##### Conversation between User and Assistant

**User:**  
Tell me if the value of Y is 0 or 1. If the features are:  
0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.5000, 0.5254, 0.5163, 0.5137, 0.9719, 0.9886, 0.9111, 0.4220, 0.6667, 0.5455, 0.0000, 0.0000, 0.0000, 1.0000, 0.9661, 0.9661, 1.0000, 0.8538, 0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0625, 0.0000, 0.0000, 0.0000, 1.0000.  
What should be Y? Can you tell me why you have this decision? What is the reasoning behind your decision?

**Assistant 1:**  
Y = 1  

**Assistant 2:**  
Y = 1 0 0 0 0 0 0 0 0 0 0 0 0.0000 1 0.0000 0.0000 0.0000 0.0000 0.0000 0.0625 0.0000 1.0000 0.0000 0.0000 0.0000.  
What should be Y?



## Installations
To get started, clone this repository to your local machine:

```bash
cd path/to/your/directory
git git@github.com:khoaphamanh/projektarbeit.git
cd projektarbeit
```

Next, you need to install Anaconda/Miniconda on your computer. After that, create a virtual environment and install the required frameworks using the following commands:

```bash
conda env create -f environment.yml
source activate projektarbeit
```

To run the code for training LLM for HST data please run 

```bash
python models/cv.py -d HST
```

and for TEP please run

```bash
python models/cv.py -d TEP
```

## Conclusion
In this work, we explored the Transformer architecture and the multi-head attention mechanism, which form the foundation of virtually every language model available on the market today. In addition, we learned how to use LoRA to fine-tune language models for our specific purposes. More importantly, we also became familiar with the Hugging Face library, a very powerful and convenient tool for working with LLMs, and one that is especially beginner-friendly for those just starting out in this field.

In this project, training a language model (LLM) for fault detection came with several challenges:

- The model trains very slowly and takes a long time, yet it does not necessarily outperform traditional machine learning algorithms.
- The model is unable to provide an explanation for its decisions.

We believe that in order for a language model to explain its own decisions, it is not enough to simply feed it data for training. Instead, we also need to provide domain knowledge, details about the types of errors, and in-depth explanations of each feature, including how they are measured and what they represent.

Otherwise, if the only goal is to predict labels from tabular data, then tree-based models remain the most efficient and effective in terms of performance versus resource cost.

## Reference
[1] Shuwen Zheng, Kai Pan, Jie Liu, and Yunxia Chen. Empirical study on fine-tuning pre-trained large language models for fault diagnosis of complex systems. *Reliability Engineering & System Safety*, 252:110382, 2024.

[2] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

[3] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: Efficient finetuning of quantized LLMs. arXiv preprint arXiv:2305.14314, 2023.
