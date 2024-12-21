from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, matthews_corrcoef
import matplotlib.pyplot as plt
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import json
from tqdm import tqdm
import random
from sklearn.metrics import matthews_corrcoef
import random
from sklearn.metrics import accuracy_score, f1_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/hy-tmp/Chb_Bert/output/checkpoint-2350"  # select checkpoint 
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.to(device)

eval_dir = './eval_data_withfake/'
for i in os.listdir(eval_dir):
    if i.endswith('.json'):
        file_name = i.split('.csv')[0]
        with open(eval_dir+i, 'r') as file:
            test_data = json.load(file)
            # test_data = random.sample(test_data, 200)
            df = pd.DataFrame(test_data)
            
        df['text'] = df['text'].astype(str)
        df['label'] = df['label'].astype(str)
        text_ls = df['text'].to_list()
        label_ls = df['label'].to_list()
        
        prob_ls = []
        binding_ls = []
        for index,row in df.iterrows():
            # hla = row['HLA_generate']
            # pep = row['Peptide_generate']
            text = row['text']
            inputs = tokenizer(
                            text,
                            padding="max_length",
                            max_length=128,
                            return_token_type_ids=True,
                            truncation=True,
                            return_tensors="pt"
                        )
            inputs.to(device)
            
            output = model(**inputs)
            logits = output.logits.squeeze()
            logits = torch.softmax(logits, dim=-1)
            label = logits.argmax(dim=-1).tolist()
            prob = logits[label].cpu().item()
            # print(prob)
            # print(label)
            prob_ls.append(prob)
            binding_ls.append(label)
        
        # 计算F1分数
        # y_true = y_true.astype(y_pred.dtype)
        binding_ls = [int(label) for label in binding_ls]
        true_label_ls =  [int(label) for label in df['label'].tolist()]
        
        # 计算准确率
        accuracy = accuracy_score(true_label_ls, binding_ls)
        
        f1 = f1_score(true_label_ls,binding_ls,pos_label=int(1))
        # 计算AUC
        auc = roc_auc_score(true_label_ls, binding_ls)
        # 计算MCC
        mcc = matthews_corrcoef(true_label_ls, binding_ls)
        
        '''
        # 绘制ROC曲线
        fpr, tpr, _ = roc_curve(true_label_ls, prob_ls)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend(loc="lower right")
        plt.savefig('./eval_output/'+file_name+'_roc_curve.png')
        plt.close
        '''

        # print(file_name,'Accuracy:',len(binding_ls)/len(df), 'F1-score:', f1, 'MCC:', mcc)
        
        print(file_name,' 测试集size: ',len(true_label_ls),' Accuracy:',accuracy, ' F1-score:', f1, ' AUC:', auc, ' MCC:', mcc)
        data = {'text': text_ls, 'label': label_ls, 'prob': prob_ls, 'binding': binding_ls}
        df_result = pd.DataFrame(data)
        df_result.to_csv('./eval_output/'+file_name+'.csv', index=False)


