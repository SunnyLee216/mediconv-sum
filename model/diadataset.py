import pandas as pd
from torch.utils.data import DataLoader, Dataset


class DialogDataset(Dataset):
    def __init__(self, data_file, tokenizer,label2id=None,id2label=None,max_length=512,is_dia=True):
        self.data = pd.read_csv(data_file)
        self.tokenizer = tokenizer
        self.is_dia = is_dia   
        if label2id:
            self.label2id = label2id
            self.id2label = id2label
            for label in self.data['section_header']:
                if label not in self.label2id:
                    print("LABEL ERROR")
        else:
            self.label2id = {}
            for label in self.data['section_header']:
                    
                if label not in self.label2id:
                    self.label2id[label] = len(self.label2id)
                    
            self.id2label = {v: k for k, v in self.label2id.items()}
            print(self.label2id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        if self.is_dia:
            text = row['dialogue']
        else:
            text = row['section_text']
        label = row['section_header']

        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return input_ids, attention_mask, self.label2id[label]