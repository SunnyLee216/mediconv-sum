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

class ProtoNet(torch.nn.Module):
    def __init__(self, encoder,input_size, output_size, hidden_size=128):
        super(ProtoNet, self).__init__()
        self.encoder = encoder
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc2(x)
        return x
class ProtoClassifier(pl.LightningModule):
    def __init__(self,  
                train_file="../MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskA/TaskA-TrainingSet.csv",
                val_file = "../MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskA/TaskA-ValidationSet.csv",
                test_file = "../MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskA/TaskA-ValidationSet.csv", 
                num_classes=20, margin=0.5, temperature=0.5,
                dropout_prob=0.5, N = 12,lam = 0.1,params=None):
        super().__init__()
        self.params = params
        
        self.dropout = nn.Dropout(dropout_prob)
        # self.num_classes = num_classes
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
        # self.automatic_optimization = False

        self.lam = torch.tensor(lam)
        self.batch_f1 = [ ]
        self.batch_recall = []
        self.batch_acc = []
        
        self.mlr = MulticlassRecall(num_classes=self.num_classes,  average='macro')
        self.contrastive_loss_set = params.constrast
        self.examples_for_contrastive = []

        ##### MODEL #####
        self.model_name = self.params.model_name
        self.bert = AutoModel.from_pretrained(self.model_name,output_hidden_states=True)
        embedding_size = 128
        self.embedding_size = embedding_size
        self.n_classes = num_classes
        
        
        self.encoder = nn.Sequential(
            nn.Linear(in_features=768, out_features=self.embedding_size),
            
        )
        self.prototypes = nn.Parameter(torch.randn(self.n_classes, self.embedding_size))

    def forward(self, input_ids, attention_mask, labels=None):
        
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        outputs = self.encoder(outputs)
        return outputs
        
    def training_step(self, batch, batch_idx) :
        input_ids, attention_mask, labels = batch
        outputs = self.forward(input_ids, attention_mask)
        loss_ins = self.compute_loss_ins(outputs,labels,self.temperature)
        loss_proto = self.compute_loss_proto(self.prototypes,outputs,labels)
        loss = loss_ins+loss_proto
        
        self.log("train_loss", loss)
        return loss
    

    def sim(x, y):
        norm_x = F.normalize(x, dim=-1)
        norm_y = F.normalize(y, dim=-1)
        return torch.matmul(norm_x, norm_y.transpose(1,0))
    

    def compute_loss_ins(self,v_ins, labels, temperature):
        """
        Compute the instance-instance loss.

        Args:
            embeddings: tensor of shape (batch_size, embedding_dim)
            targets: tensor of shape (batch_size,) containing target labels
            temperature: temperature parameter for the softmax function

        Returns:
            instance_instance_loss: instance-instance loss value
        """    
    
        # Compute the log probabilities for each pair of instances
        # similarities = cosine_similarity(embeddings) / temperature  # (batch_size, batch_size)
        # similarities = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)/ temperature

        # mask = (targets.unsqueeze(0) == targets.unsqueeze(1)).float()  # (batch_size, batch_size)
        # mask -= torch.eye(mask.shape[0]).to(embeddings.device)  # remove diagonal elements
        # numerator = torch.exp(similarities) * mask
        # denominator = torch.sum(torch.exp(similarities) * (1 - mask), dim=1)
        # instance_instance_loss = -torch.mean(torch.log(numerator / denominator))
        num_instances = v_ins.shape[0]
        num_classes = self.num_classes

        loss = 0.
        for n in range(num_classes):
            # select instances of class n
            mask = torch.eq(labels, n)
            v_n = v_ins[mask]

            # calculate similarity matrix between instances of class n
            sim_mat = torch.exp(self.sim(v_n, v_n))

            # calculate loss
            pos_score = torch.diagonal(sim_mat)
            neg_score = sim_mat.sum(dim=1) - pos_score
            loss += -torch.log(pos_score / (pos_score + neg_score)).sum()

        loss /= (num_instances * num_classes * (num_instances - num_classes))

        return loss
        
    
    def compute_loss_proto(self, v_ins, labels):
        '''TO compute the loss between instances and proto which have the same class'''
        
    # instance-prototype loss

        num_instances = v_ins.shape[0]
        num_classes = self.num_classes

        # calculate similarity matrix between instances and prototypes
        sim_mat = torch.exp(self.sim(v_ins, self.proto))

        loss = 0.
        for n in range(num_classes):
            # select instances of class n
            mask = torch.eq(labels, n)
            v_n = v_ins[mask]

            # calculate loss
            pos_score = sim_mat[mask, n].squeeze()
            neg_score = sim_mat.sum(dim=1) - pos_score
            loss += -torch.log(pos_score / (pos_score + neg_score)).sum()

        loss /= (num_instances * num_classes)

        return loss
    
    def predict(self, query):
        sim_scores = self.sim(query, self.proto)
        class_scores = []
        for i in range(self.num_classes):
            class_protos = sim_scores[:, i*self.num_proto_per_class: (i+1)*self.num_proto_per_class]
            class_scores.append(torch.mean(class_protos, dim=1))
        class_scores = torch.softmax(class_scores,dim=1)
        class_index = torch.argmax(class_scores, dim=1)
        return class_index
    


    def validation_step(self, batch, batch_idx) :
        input_ids, attention_mask, labels = batch
        v_ins = self.forward(input_ids, attention_mask)
        preds = self.predict(v_ins)
        ## Query2Proto ##

        misclassified_idx = (preds != labels)
        misclassified_inputs = input_ids[misclassified_idx]
        misclassified_preds = preds[misclassified_idx]
        misclassified_labels = labels[misclassified_idx]
        self.misclassified_examples.append((misclassified_inputs, misclassified_preds, misclassified_labels))


        f1_score = multiclass_f1_score(preds=preds, target=labels, num_classes=self.num_classes, average='macro')
        recall_score = self.mlr(preds=preds, target=labels)
        acc = accuracy_score(y_true=labels.cpu().numpy(), y_pred=preds.cpu().numpy())

        self.log('val_acc', acc,on_epoch=True,prog_bar=True,on_step=True)
        self.log('val_f1', f1_score, on_step=False, on_epoch=True)
        self.log('val_recall', recall_score, on_step=False, on_epoch=True)
        self.batch_f1.append(f1_score)
        self.batch_recall.append(recall_score)
        self.batch_acc.append(acc)
        return {'preds':preds,'labels':labels}
        
    




    def validation_epoch_end(self,outputs): 
        avg_f1 = torch.tensor(self.batch_f1).mean()
        if avg_f1>=0.7:
            if self.misclassified_examples:
                for batch in self.misclassified_examples:
                    inputs, preds, labels = batch
                    for i in range(inputs.shape[0]):
                        print(f"Misclassified example: - True label: {self.id2label[labels[i]]} - Predicted label: {self.id2label[preds[i]]}")
    
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
        self.misclassified_examples=[]
    
    def configure_optimizers(self):
        ## TODO
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.params.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=self.params.total_steps
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.params.batch_size, shuffle=True, num_workers=4)
        return dataloader

    def val_dataloader(self):
        dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.params.batch_size, num_workers=4)
        return dataloader
    