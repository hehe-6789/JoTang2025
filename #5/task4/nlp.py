import os
import pandas as pd
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import (BertTokenizerFast,BertForSequenceClassification,TrainingArguments,Trainer,DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, f1_score
from transformers import AdamW,get_linear_schedule_with_warmup
#设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
     


#定义超参数
Model_name="bert-base-chinese"
Max_length=128
Batch_size=32
Lr=3e-5
Epoches=3
Savedir="./"
Bestmodeldir="./best"


#
os.makedirs(Savedir,exist_ok=True)
os.makedirs(Bestmodeldir,exist_ok=True)

#加载数据集
dataset = load_dataset("parquet",  # 指定格式为 Parquet
    data_files={
        "train": "./train.parquet",
        "validation": "./validation.parquet",
        "test": "./test.parquet"})

#加载tokenizer
tokenizer=BertTokenizerFast.from_pretrained(Model_name)

#数据预处理函数
def preprocess_function(examples):
    texts=[text if text is not None else "" for text in examples["text"]]
    return tokenizer(texts,truncation=True,max_length=Max_length,padding=False,return_overflowing_tokens=False)

#应用数据预处理函数
tokenize_data=dataset.map(preprocess_function,batched=True,remove_columns=["text"])

#数据整理器
data_collator=DataCollatorWithPadding(tokenizer=tokenizer)

#定义评估函数
def compute_metrics(eval_pred):
    pred,labels=eval_pred
    pred=np.argmax(pred,axis=1)
    acc=accuracy_score(y_true=labels,y_pred=pred)
    f1=f1_score(y_true=labels,y_pred=pred,average='binary')
    return {"accuracy":acc,"f1":f1}


#加载模型
model=BertForSequenceClassification.from_pretrained(Model_name,num_labels=2,return_dict=True)

#设置优化器
optimizer=AdamW(
    model.parameters(),
    lr=Lr,
    weight_decay=0.01    
)
total_step=(len(tokenize_data["train"]))//Batch_size

#设置学习率调度器
scheduler=get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_step
)

#设置训练参数
trainer_args=TrainingArguments(output_dir=Savedir,overwrite_output_dir=True,num_train_epochs=Epoches,per_device_train_batch_size=Batch_size,per_device_eval_batch_size=Batch_size,learning_rate=Lr,weight_decay=0.01,logging_dir="./logs",logging_steps=100,evaluation_strategy="epoch",save_strategy="epoch",
load_best_model_at_end=True,  # 训练结束后，自动加载“最佳模型”的checkpoint
metric_for_best_model="f1",   # 明确用F1分数判断“最佳模型”（"accuracy"）                            
greater_is_better=True, 
report_to="none",seed=42
)
#初始化trainer
trainer=Trainer(
model=model,
args=trainer_args,
train_dataset=tokenize_data["train"],
eval_dataset=tokenize_data["validation"],
tokenizer=tokenizer,
data_collator=data_collator,
compute_metrics=compute_metrics,
optimizers=(optimizer,scheduler)
)
#开始训练
set_seed(42)
print("开始训练...")
trainer.train()

#在测试集上评估
test_results=trainer.evaluate(tokenize_data["test"])
print(f"测试集结果：{test_results}")
Bestmodeldir=os.path.join(Bestmodeldir,'')
trainer.save_model(Bestmodeldir)
record=[32,3e-5,3]
record=pd.DataFrame()
record.to_csv('superweight.csv',mode='a',index=False,header=False)

    
