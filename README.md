# An Empirical Analysis of Fine-Tuning LargeLanguage Models for Predictive Maintenance
In this project, we will use large language models (LLMs) for classification tasks in the field of predictive maintenance. The project is inspired by and based on the paper [Empirical study on fine-tuning pre-trained large language models for fault diagnosis of complex systems](https://www.sciencedirect.com/science/article/abs/pii/S095183202400454X) [1].

## Datasets
Dữ liệu mà chúng tôi sử dụng trong công việc này có dạng bảng. Mỗi dòng tương ứng với một instance và sẽ có số cột tương ứng với features. Chúng tôi sử dụng hai tập dữ liệu trong công việc này

### High speed train breaking system
High speed train (HST) breaking system là dữ liệu dạng bảng bao gồm 22368 instances, có tổng cộng 46 features với 15 continous features và 31 categorical features. Chỉ có 2 labels là 0 (normal) và 1 (anomaly). Dữ liệu là không cân bằng với 21979 instance có label 0 và chỉ 389 instance có label 1. Dữ liệu gốc ở link này https://www2.ie.tsinghua.edu.cn/rrml/Resources.html . Cần lưu ý rằng, người dùng không cần phải download dữ liệu xuống một cách thủ công, khi chạy code, dữ liệu sẽ tự động được download xuống. 

### Tennessee Eastman Process (TEP) Dataset
Tennessee Eastman Process (TEP) Dataset là tập dữ liệu dạng bảng. Trong dữ liệu này, có tổng cộng 21 labels, với label 0 là normal, và 20 lỗi còn lại từ label 1 tới 20. Trong dữ liệu train data, mỗi label có 250000 instance, còn trong dữ liệu test, mỗi label sẽ có 480000. Dữ liệu này cân bằng và features có tên.

## Preprocessing
### Downsampling
Do cả 2 dữ liệu này đều khá lớn trong việc huấn luyện LLM do đó chúng tôi đã giảm dữ liệu xuống để làm cho việc huấn luyện mô hình trở nên nhanh hơn. HST sẽ giảm chỉ còn 240 instances cho train data và 60 instances cho test data. Tỉ lệ label là cân bằng. TEP sẽ giảm còn 400 instances cho train data và 160 instances cho test data. Chúng tôi chỉ dùng label 0,1,4,5 và tỉ lệ cho mỗi label là cân bằng. Chúng tôi downsampling theo paper [1].

### Normalization
Chúng tôi sẽ chuẩn hóa dữ liệu theo kiểu Standardize.

### Convert to Text
#### Tabular to Text
Do sử dụng LLM trong công việc này, do đó chúng tôi sẽ chuyển dữ liệu bảng sang dạng text. ví dụ nếu tôi có một instance
| setting_1 | setting_2 | condition | sensor_1 | sensor_2 | label |
|------------|------------|------------|-----------|-----------|--------|
| 1 | 1 | 2 | 0.05 | 0.7 | 1 |

thì khi chuyển sang text sẽ là

**Question:**  
Tell me if the value of Y is 0 or 1. If feature setting_1 = 1, feature setting_2 = 1, feature condition = 2, feature sensor_1 = 0.05, feature sensor_2 = 0.7. What should be Y?  

**Answer:**  
Y = 0

#### System behavior
không chi chuyển sang text, chúng ta còn phải sử dụng đúng template của LLM mà chúng tôi sẽ sử dụng LLaMA 2, do đó chúng tôi sẽ phải sử dụng cú pháp của mô hình này:
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
Trong công việc này, chúng tôi sử dụng [LLaMA 2](https://arxiv.org/abs/2307.09288) [2] để huấn luyện mô hình theo kiểu supervised learning. Túc là chúng tôi kỳ vọng ouput của mô hình sẽ có độ dài 5 tokens, bao gồm 2 tokens $[/INST]$ và $</s>$ và Y = label. 
Tuy nhiên do mô hình này quá lớn, chúng tôi sẽ sử dụng [Quantized Low rank adapation](https://arxiv.org/abs/2305.14314) (Q-LoRA) [3] để huấn luyện một phần của mô hình. chúng tôi áp dụng lora cho tất cả các layers có trong model

## Hyperparameter Optimization Tunning
Trong công việc này, chúng tôi sử dụng loss trong tập dữ liệu validation để làm objective function để lựa chọn hyperarameters tốt nhất. Những hyperparameters mà chúng tôi tune là:

lora_r: dimension of trainable matrix

lora_alpha: alpha as scale factor in LoRA

lora_dropout: drop out rate in LoRA

normalize: should the input normalize with standardize

learning_rate: learning rate of the model

## Result
Trong dự án này, không chỉ huấn luyện LLM, mà chúng tôi còn so sánh với các mô hình Machine Learning khác

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

