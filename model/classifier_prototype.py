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
                num_classes=20, margin=0.5, temperature=0.5,seed=42,
                dropout_prob=0.5, N = 12,lam = 0.1,params=None,lr_prototypes = 1e-3,lr_orthers=1e-3):
        super().__init__()
        pl.utilities.seed.seed_everything(seed=seed)

        ### PARAMETERS ###
        self.params = params
        self.loss_fn = nn.CrossEntropyLoss()
        self.similarity_fn = nn.CosineSimilarity(dim=1)
        self.margin = margin
        self.temperature = torch.tensor(temperature)
        self.lr_prototypes = lr_prototypes
        self.num_classes = num_classes
        self.lr_features = self.params.learning_rate
        self.lr_orthers = lr_orthers
        self.gradient_accumulation_steps = self.params.gc_step
        # self.automatic_optimization = False

        self.lam = torch.tensor(lam).to(self.device)
        self.batch_f1 = [ ]
        self.batch_recall = []
        self.batch_acc = []
        self.num_proto_per_class = 1
        self.mlr = MulticlassRecall(num_classes=self.num_classes,  average='macro')
        self.contrastive_loss_set = params.constrast
        self.examples_for_contrastive = []

        ##### MODEL #####
        self.model_name = self.params.model_name
        self.bert = AutoModel.from_pretrained(self.model_name,output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        embedding_size = 128
        self.embedding_size = embedding_size
        self.n_classes = num_classes
        
        
        self.encoder = nn.Linear(in_features=768, out_features=self.embedding_size)
        self.prototype_vectors = nn.Parameter(torch.randn(self.n_classes, self.embedding_size),requires_grad=True)

         #### DATA LODADER ####
        self.train_dataset = DialogDataset(self.params.train_file, self.tokenizer, max_length = self.params.max_input_length)
        self.val_dataset = DialogDataset(self.params.val_file, self.tokenizer, max_length = self.params.max_input_length, id2label=self.train_dataset.id2label,label2id=self.train_dataset.label2id)
        self.test_dataset = DialogDataset(self.params.test_file, self.tokenizer, max_length = self.params.max_input_length, id2label=self.train_dataset.id2label,label2id=self.train_dataset.label2id)
        self.batch_f1 = [ ]
        self.batch_recall = [ ]
        self.batch_acc = [ ]
        self.misclassified_examples=[]

    def forward(self, input_ids, attention_mask, labels=None):
        
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state
        pooled_output = hidden_state[:, 0] # CLStoken
        outputs_encoded = self.encoder(pooled_output)
        return outputs_encoded
        
    def training_step(self, batch, batch_idx) :
        input_ids, attention_mask, labels = batch
        outputs = self.forward(input_ids, attention_mask)
        loss_ins = self.compute_loss_ins(outputs,labels)
        loss_proto = self.compute_loss_proto(outputs,labels)
        loss = self.lam*loss_ins+loss_proto
        
        self.log("loss_proto",loss_proto)
        self.log("loss_ins",loss_ins)
        self.log("train_loss", loss)
        # ,'loss_ins':loss_ins
        return {'loss':loss,'loss_proto':loss_proto,'loss_ins':loss_ins}
    

    def sim(self,x, y):
        norm_x = F.normalize(x + 1e-8, dim=-1)
        norm_y = F.normalize(y + 1e-8, dim=-1)
        return torch.matmul(norm_x, norm_y.transpose(1, 0))
    

    def compute_loss_ins(self,v_ins, labels):
        """
        Compute the instance-instance loss.

        Args:
            embeddings: tensor of shape (batch_size, embedding_dim)
            labels: tensor of shape (batch_size,) containing target labels
            temperature: temperature parameter for the softmax function

        Returns:
            instance_instance_loss: instance-instance loss value
        """    
        similarities = self.sim(v_ins,v_ins)
        # print(similarities)
         # (batch_size, batch_size)
        
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # (batch_size, batch_size)
        mask -= torch.eye(mask.shape[0]).to(self.device)  # remove diagonal elements
        numerator = torch.exp(similarities) * mask
        denominator = torch.sum(torch.exp(similarities) * (1 - mask), dim=1)
        numerator = torch.clamp(numerator, min=1e-8)
        denominator = torch.clamp(denominator, min=1e-8)
        # print(numerator)
        # print(denominator)
        #print(torch.log(numerator / denominator))
        instance_instance_loss = -torch.mean(torch.log(numerator / denominator))
        # print("instance_loss",instance_instance_loss)

        return instance_instance_loss
        
    
    def compute_loss_proto(self, v_ins, labels):
        '''TO compute the loss between instances and proto which have the same class'''
        
    # instance-prototype loss
        batch_size = v_ins.size(0)
        num_instances = v_ins.shape[0]
        num_classes = self.num_classes
        
        # calculate similarity matrix between instances and prototypes
        similarities = torch.exp(self.sim(v_ins, self.prototype_vectors))
        #print(similarities)
        
         # create a mask to ignore similarities between feature and its own prototype
        mask = torch.ones(batch_size, num_classes).to(self.device)
        mask[torch.arange(batch_size), labels] = 0
       
        # calculate denominator by summing over all classes except the true class of each instance
        denominator = torch.sum(torch.exp(similarities) * mask, dim=1, keepdim=True)
        
        # calculate numerator by taking the similarity between each feature and its own prototype
        numerator = similarities[torch.arange(batch_size), labels].view(-1, 1)
        
        # calculate the instance-prototype loss using log-softmax
        loss = -torch.mean(torch.log(numerator / denominator))
        # print("proto_loss",loss)
        return loss
    
    def predict(self, query):
        sim_scores = self.sim(query, self.prototype_vectors)
        class_scores = []
        for i in range(self.num_classes):
            class_protos = sim_scores[:, i*self.num_proto_per_class: (i+1)*self.num_proto_per_class]
            class_scores.append(torch.mean(class_protos, dim=1))
        class_scores = torch.stack(class_scores, dim=1)  # Convert list to tensor
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
        print(self.prototype_vectors)
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
        joint_optimizer_specs = [{'params': self.bert.parameters(), 'lr': self.lr_features},
                                 {'params': self.encoder.parameters(), 'lr': self.lr_orthers},
                                {'params': self.prototype_vectors, 'lr': self.lr_prototypes},
                                
                                 ]
        
        optimizer = torch.optim.AdamW(joint_optimizer_specs)
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
    