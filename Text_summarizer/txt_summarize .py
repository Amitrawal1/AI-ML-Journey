#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from transformers import T5Tokenizer, Trainer, TrainingArguments, T5ForConditionalGeneration


# In[5]:


train_data = pd.read_csv("./data/samsum-train.csv")
validation_data = pd.read_csv("./data/samsum-validation.csv")


# In[6]:


train_data.head


# In[7]:


train_data.shape


# In[9]:


validation_data.shape


# In[10]:


# random sampling and reduce data for now(not compulsary)
train_data = train_data.sample(n=5000, random_state = 42).reset_index(drop=True)
compulsoryvalidation_data = validation_data.sample(n=500, random_state = 42).reset_index(drop=True)


# In[11]:


# Pre-processing Data 
import re

def clean_data(text):
    text = re.sub(r"\r\n"," ",  text)  # remove next line etc.
    text = re.sub(r"s+", " ", text)  # extra space
    text = re.sub(r"<.*?>", " ", text)  # HTML Tags
    text = text.strip().lower()
    return text


# In[12]:


train_data["dialogue"] = train_data["dialogue"].apply(clean_data)
train_data["summary"] = train_data["summary"].apply(clean_data)

validation_data["dialogue"] = validation_data["dialogue"].apply(clean_data)
validation_data["summary"] = validation_data["summary"].apply(clean_data)


# In[13]:


train_data["dialogue"][0]


# In[14]:


# Tokenizar
tokenizer = T5Tokenizer.from_pretrained("t5-small")


# In[15]:


# row data => tokenized inputs for fine-tuning

def tokenize(data):
    inputs = tokenizer(data["dialogue"], padding="max_length", max_lenght = 512, truncation = True)
    targets = tokenizer(data["summary"], padding="max_length", max_lenght = 150, truncation = True)

    inputs["labels"] = targets["input_ids"]  # token id => add to input as labels
    return inputs


# In[16]:


train_dataset = train_data.apply(tokenize, axis=1).tolist()
validation_dataset = validation_data.apply(tokenize, axis=1).tolist()


# In[17]:


train_dataset[0]


# In[33]:


#  inputs_ids
# 1 => EOS    0 => Padding
#  attention_mask => those have 1 its mean this important and valid values other not imp during the training
#  labels - target => summary token


# In[18]:


#  Working with our Model
#NLP => generation task
model = T5ForConditionalGeneration.from_pretrained("t5-small")


# In[19]:


import torch 
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Device: ", device)
model.to(device)


# In[20]:


# Training Arguments for Transformer

training_args = TrainingArguments(
    output_dir = "./results",

    num_train_epochs = 6,
    weight_decay = 0.01,

    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,

    eval_strategy = "epoch",
    save_strategy = "epoch",

    warmup_steps = 500  # 0 => learning rate default
)


# In[21]:


trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = validation_dataset
)


# In[22]:


model.save_pretrained("./saved_summary_model")
tokenizer.save_pretrained("./saved_summary_model")


# In[23]:


model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")


# In[26]:


def summarize_dialogue(dialogue):
    dialogue = clean_data(dialogue) # clean

    # tokenize
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # generate the summary => token ids
    model.to(device)
    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )
    
    # decoded our output
    summary = tokenizer.decode(targets[0], skip_special_tokens=True) # EOS, SEP
    return summary


# In[27]:


test_dialogue = """ 
Reporter: In today's technology news, artificial intelligence continues to expand rapidly across industries, from healthcare to finance and education. Recent reports suggest that AI adoption has significantly increased over the past few years.

Reporter: Companies are investing heavily in machine learning systems to automate tasks, improve decision-making, and enhance customer experiences. However, this growth has also raised questions about job displacement and ethical concerns.

Expert: AI systems are becoming more capable due to advances in deep learning and access to large datasets. These models can now perform complex tasks such as language understanding, image recognition, and even code generation.

Expert: At the same time, there are valid concerns about bias in AI models, as they often reflect the data they are trained on. Ensuring fairness and transparency is becoming a key area of research.

Reporter: Governments and organizations are beginning to introduce regulations to guide the development and deployment of AI technologies. The goal is to balance innovation with accountability.

Expert: Another challenge is explainability. Many modern AI systems, especially deep neural networks, operate as “black boxes,” making it difficult to understand how decisions are made.

Reporter: Experts also highlight the importance of responsible AI development, including data privacy, security, and long-term societal impact.

Expert: Looking ahead, collaboration between researchers, policymakers, and industry leaders will be crucial to ensure that AI systems are developed and used in a safe and beneficial way.
"""

summary = summarize_dialogue(test_dialogue)

print("Summary: ", summary)

