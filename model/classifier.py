import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import pandas as pd
from sklearn.metrics import accuracy_score
class DialogDataset(Dataset):
    def __init__(self, data_file, tokenizer,label2id=None,id2label=None):
        self.data = pd.read_csv(data_file)
        self.tokenizer = tokenizer
        if label2id:
            self.label2id = label2id
            self.id2label = id2label
        else:
            self.label2id = {}
            for label in self.data['section_header']:
                    
                if label not in self.label2id:
                    self.label2id[label] = len(self.label2id)
                    
            self.id2label = {v: k for k, v in self.label2id.items()}
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['dialogue']
        label = row['section_header']

        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return input_ids, attention_mask, self.label2id[label]
        
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    preds = []
    targets = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, attention_mask,labels = batch
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)
            attention_mask = attention_mask.to(model.device)
            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    

    accuracy = correct / total
    print('Accuracy: {:.4f}'.format(accuracy))
    
    return accuracy

def train():
    seed = 42
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = DialogDataset('../MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskA/TaskA-TrainingSet.csv', tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataset = DialogDataset('../MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskA/TaskA-ValidationSet.csv', tokenizer,id2label=train_dataset.id2label,label2id=train_dataset.label2id)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
    num_labels = len(train_dataset.label2id)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    optimizer =  torch.optim.AdamW (model.parameters(), lr=2e-5)
    
    weight = torch.tensor([1.0 / len(train_dataset.label2id)] * num_labels)
    criterion = torch.nn.CrossEntropyLoss(weight=weight.to(device))
    model.to(device)

    for epoch in range(30):
        model.train()
        N=4
        total_loss = 0
        for i, batch in enumerate(train_loader):
            input_ids, attention_mask, label = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
            loss = criterion(outputs.logits, label)
            loss.backward()
            if (i + 1 )%N==0:
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch}: loss={total_loss}")
        evaluate(model,val_loader)
    model.save_pretrained("classifier_model")


if __name__ == '__main__':
    train()
