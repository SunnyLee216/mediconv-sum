import os
import torch
import pandas as pd

class DoctorPatientDialogueDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx, field=['dialogue','note']):
        conversation = self.data.iloc[idx][field[0]]
        summary = self.data.iloc[idx][field[1]]
        
        # Tokenize the input
        input_ids = self.tokenizer.encode(
            conversation, 
            add_special_tokens=True, 
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        # Tokenize the target
        target_ids = self.tokenizer.encode(
            summary, 
            add_special_tokens=True, 
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        
        # Convert to tensors
        input_ids = torch.tensor(input_ids).long()
        target_ids = torch.tensor(target_ids).long()
        
        return input_ids, target_ids