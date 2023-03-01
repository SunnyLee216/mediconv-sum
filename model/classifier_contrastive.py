from model.diadataset import DialogDataset
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score,f1_score,recall_score
import pandas as pd
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torchmetrics.classification import  MulticlassRecall
import torch.nn.functional as F
from torchmetrics.functional.classification import multiclass_f1_score

class ContrastiveFinetuner(pl.LightningModule):
    def __init__(self,  
                train_file="../MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskA/TaskA-TrainingSet.csv",
                val_file = "../MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskA/TaskA-ValidationSet.csv",
                test_file = "../MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskA/TaskA-ValidationSet.csv", 
                num_classes=20, margin=0.5, temperature=0.5,
                dropout_prob=0.5, N = 12,lam = 0.1,params=None):
        super().__init__()
        self.params = params
        self.model_name = self.params.model_name
        self.bert = AutoModel.from_pretrained(self.model_name,output_hidden_states=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.num_classes = num_classes
        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_classes)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.loss_fn = nn.CrossEntropyLoss()
        self.similarity_fn = nn.CosineSimilarity(dim=1)
        self.margin = margin
        self.temperature = torch.tensor(temperature)
        self.train_dataset = DialogDataset(self.params.train_file, self.tokenizer, max_length = self.params.max_input_length)
        self.val_dataset = DialogDataset(self.params.val_file, self.tokenizer, max_length = self.params.max_input_length, id2label=self.train_dataset.id2label,label2id=self.train_dataset.label2id)
        self.test_dataset = DialogDataset(self.params.test_file, self.tokenizer, max_length = self.params.max_input_length, id2label=self.train_dataset.id2label,label2id=self.train_dataset.label2id)
        self.gradient_accumulation_steps = self.params.gc_step
        self.automatic_optimization = False
        self.lam = torch.tensor(lam)
        self.batch_f1 = [ ]
        self.batch_recall = []
        self.batch_acc = []
        self.mlr = MulticlassRecall(num_classes=self.num_classes,  average='macro')
        self.contrastive_loss_set = params.constrast

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state
        pooled_output = hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        
        if labels is not None:
            loss_ce = self.loss_fn(logits,labels)
            loss_cl = self.contrastive_loss(logits, labels,hidden_state)
            
            return loss_ce,loss_cl,logits
        else:
            return logits

    def contrastive_loss(self, logits, labels,embeddings):
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float().fill_diagonal_(0)
        neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()

        # calculate similarity between all pairs of embeddings
        
        # TODO
        sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)

        # compute numerator and denominator for positive pairs
        # pos_sim = torch.exp(sim_matrix[pos_mask].reshape(embeddings.shape[0], -1) / self.temperature)
        pos_sim = torch.exp(sim_matrix[pos_mask.bool()].reshape(embeddings.shape[0], -1) / self.temperature)

        pos_sum = torch.sum(pos_sim, dim=1, keepdim=True)

        # compute numerator and denominator for negative pairs
        neg_sim = torch.exp(sim_matrix[neg_mask.bool()].reshape(embeddings.shape[0], -1) / self.temperature)
        neg_sum = torch.sum(neg_sim, dim=1, keepdim=True)

        # compute loss
        loss_cl = -torch.log(pos_sum / (neg_sum + pos_sum)).mean()
        return loss_cl
        

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        input_ids, attention_mask, labels = batch

        loss_ce,loss_cl,logits = self.forward(input_ids, attention_mask, labels)
        # print(loss_cl)
        if self.contrastive_loss_set:
            loss = loss_ce + self.lam * loss_cl
            self.log('contrastive_loss',loss_ce,on_step=False, on_epoch=True,prog_bar=True)
        else:
            loss =loss_ce
        N = self.gradient_accumulation_steps
        loss = loss / N
        self.manual_backward(loss)
        
        # preds = torch.argmax(logits, axis=1)
        

        if (batch_idx + 1) % N == 0:
            opt.step()
            opt.zero_grad()
            sch.step()

        # self.log('contrastive_loss',loss_ce,on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True,prog_bar=True)
        
        return loss
    
    def training_epoch_end(self, outputs):
        self.optimizers().step() 
        self.optimizers().zero_grad()



    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch

        logits = self.forward(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1)
        f1_score = multiclass_f1_score(preds=preds, target=labels, num_classes=self.num_classes, average='macro')
        recall_score = self.mlr(preds=preds, target=labels)
        acc = accuracy_score(y_true=labels.cpu().numpy(), y_pred=preds.cpu().numpy())

        self.log('val_acc', acc,on_epoch=True,prog_bar=True,on_step=True)
        self.log('val_f1', f1_score, on_step=False, on_epoch=True)
        self.log('val_recall', recall_score, on_step=False, on_epoch=True)
        self.batch_f1.append(f1_score)
        self.batch_recall.append(recall_score)
        self.batch_acc.append(acc)
        
        
    

    def validation_epoch_end(self, outputs):
        # avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_f1 = torch.tensor(self.batch_f1).mean()
        avg_recall = torch.tensor(self.batch_recall).mean()
        avg_acc = torch.tensor(self.batch_acc).mean()
        # avg_rouge = torch.stack([x['rouge_score'] for x in outputs]).mean()
        # self.log("avg_rouge_score", avg_rouge,sync_dist=True,prog_bar=True)
        # avg_rougeLsum_fmeasure = torch.stack([x['rougeLsum_fmeasure'] for x in outputs]).mean()
        # self.log("avg_rougeLsum_fmeasure", avg_rougeLsum_fmeasure,sync_dist=True,prog_bar=True)
        # 打印结果
        # self.log('avg_val_loss', avg_loss, on_epoch=True, prog_bar=True)
        self.log('avg_val_f1', avg_f1, on_epoch=True, prog_bar=True)
        self.log('avg_val_recall', avg_recall, on_epoch=True, prog_bar=True)
        self.log('avg_val_acc',avg_acc, on_epoch=True,prog_bar=True)

        # 清空记录的结果
        self.batch_f1 = [ ]
        self.batch_recall = [ ]
        self.batch_acc = [ ]
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.params.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=5, num_training_steps=self.params.total_steps
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.params.batch_size, shuffle=True, num_workers=4)
        return dataloader

    def val_dataloader(self):
        dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.params.batch_size, num_workers=4)
        return dataloader