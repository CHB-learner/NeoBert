from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import json
from tqdm import tqdm
import random
from sklearn.metrics import matthews_corrcoef
import random
from sklearn.metrics import accuracy_score, f1_score

# 设置随机种子
random.seed(42)
model_path =  "/hy-tmp/Chb_Bert/output/checkpoint-2350" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# for model_name in os.listdir(model_path_dir):

ACC_classification = 0
# model_path = os.path.join(model_path_dir, model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path,attentions='True')
model.to(device)

eval_dir = './eval_data_withfake/'
for i in os.listdir(eval_dir):
    if i.endswith('.json'):
        with open(eval_dir+i, 'r') as file:
            test_data = json.load(file)
            test_data = random.sample(test_data, 200)
            
        ACC_classification = 0
        true_labels = []
        predicted_labels = []
        
        for i in tqdm(test_data, desc="Evaluating "):
            true_label = i['label']
            input_text = i['text']
            inputs = tokenizer(
                input_text,
                padding="max_length",
                max_length=len(input_text),
                return_token_type_ids=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            output = model(**inputs)
            logits = output.logits.squeeze()
            logits = torch.softmax(logits, dim=-1)
            label = logits.argmax(dim=-1).tolist()
            true_labels.append(int(true_label))
            predicted_labels.append(int(label))
            
        # MCC
        mcc = matthews_corrcoef(true_labels, predicted_labels)
        # 计算准确率
        accuracy = accuracy_score(true_labels, predicted_labels)
        # 计算F1分数
        f1 = f1_score(true_labels, predicted_labels, average='binary')  # 假设是二分类任务
        # print(i)
        print('Accuracy on test data:', accuracy)
        print('F1-score on test data:', f1)
        print('MCC', mcc)
        print('----------------------------------------------------')


