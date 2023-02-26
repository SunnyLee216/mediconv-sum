import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,get_cosine_schedule_with_warmup
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics import CHRFScore
import torch 
import os
import torch.nn as nn
import pandas as pd
import warnings
import torch.nn.functional as F
import csv
warnings.filterwarnings("ignore", message="Error loading .*: <urlopen error [Errno 111] Connection refused>")


class DoctorPatientDialogueDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, max_input_length, max_output_length,is_train=True):
        self.data = pd.read_csv(csv_file)
        # self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.inputs = self.data['dialogue'].apply(lambda x: x[:self.max_input_length])
        if self.is_train:
            if 'section_text' in self.data.columns:
                print('TaskA data loading....')
                self.outputs = self.data.apply(lambda x: x['section_header'] + ' ' + x['section_text'], axis=1)
                self.outputs = self.outputs.apply(lambda x: x[:self.max_output_length])
                # TODO truncate

            else:
                print('TaskB or TaskC data loading....')
                self.outputs = self.data['note'].apply(lambda x: x[:self.max_output_length]) 
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.is_train:
            input_sequence = self.inputs[index]
            output_sequence = self.outputs[index]
            return input_sequence, output_sequence
        else:
            input_sequence = self.inputs[index]
            return input_sequence

class PegasusSummarizer(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.r_factor = 0.5
        # PREPARE DATA
        self.train_dataset = DoctorPatientDialogueDataset(self.params.train_file, self.params.max_input_length, self.params.max_output_length,is_train=True)
        self.val_dataset = DoctorPatientDialogueDataset(self.params.val_file, self.params.max_input_length, self.params.max_output_length,is_train=True)
        self.test_dataset = DoctorPatientDialogueDataset(self.params.val_file, self.params.max_input_length, self.params.max_output_length,is_train=False)

        # pretrained_model_name = "google/pegasus-summarization"
        pretrained_model_name = self.params.pretrained_model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.rouge = ROUGEScore()
        if self.params.chrf_score:
            self.chrf = CHRFScore()
        
        self.warm_up=params.warm
        self.learning_rate = params.learning_rate

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        outputs = self.model(input_ids, attention_mask=attention_mask,labels=decoder_input_ids,output_hidden_states=True,return_dict=True)
        return outputs.loss,outputs.logits

    def compute_kl_loss(self, p, q, pad_mask=None):
    
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
        
        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels, labels_mask = batch
        loss,logits = self.forward(input_ids, attention_mask,labels)
        # loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        if self.params.use_r_drop:
            loss2,logits2 = self.forward(input_ids, attention_mask,labels)
            ce_loss = 0.5*(loss+loss2)
            kl_loss= self.compute_kl_loss(logits, logits2,pad_mask = labels_mask)
            loss = ce_loss + self.r_factor*kl_loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels, labels_mask = batch
        loss,_ = self.forward(input_ids, attention_mask,labels)
        # loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)

        # Add the ROUGE evaluation
        # 调整参数使得能够进行多样化的生成
        # TODO 后期把生成的参数放到后面argument
        if self.params.rouge_score:
            outputs = self.model.generate(input_ids, attention_mask=attention_mask,num_beams=3,max_new_tokens = self.params.max_output_length,length_penalty = 0.5)
            generated_summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            target_summaries = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            rouge_score = self.rouge(generated_summaries, target_summaries)
            self.log("rouge_score", rouge_score,sync_dist=True, on_step=False, on_epoch=True,prog_bar=True)
            self.log("rougeLsum_f", rouge_score['rougeLsum_fmeasure'],sync_dist=True, on_step=False, on_epoch=True)
            self.log("rouge1_f", rouge_score['rouge1_fmeasure'],sync_dist=True, on_step=False, on_epoch=True)
            self.log("rouge2_f", rouge_score['rouge2_fmeasure'],sync_dist=True, on_step=False, on_epoch=True)
        if self.params.chrf_score:
            chrf_score = self.chrf(generated_summaries, target_summaries)
            self.log("chrf_score", chrf_score,sync_dist=True, on_step=False, on_epoch=True,prog_bar=True)

        
        
        # return {'val_loss': loss, "chrf_score":chrf_score, "rougeLsum_fmeasure":rouge_score['rougeLsum_fmeasure'],"rouge1_f":rouge_score['rouge1_fmeasure'],"rouge2_f":rouge_score['rouge2_fmeasure']}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        
        # avg_rouge = torch.stack([x['rouge_score'] for x in outputs]).mean()
        # self.log("avg_rouge_score", avg_rouge,sync_dist=True,prog_bar=True)
        # avg_rougeLsum_fmeasure = torch.stack([x['rougeLsum_fmeasure'] for x in outputs]).mean()
        # self.log("avg_rougeLsum_fmeasure", avg_rougeLsum_fmeasure,sync_dist=True,prog_bar=True)
        return {'avg_val_loss': avg_loss}

    # def prepare_data(self):
    #     df = pd.read_csv(self.params.data_path)
    #     X_train, X_val, y_train, y_val = train_test_split(df['text'], df['summary'], test_size=0.2, random_state=42)

    #     self.train_dataset = TextDataset(X_train, y_train, self.tokenizer, self.params.max_len)
    #     self.val_dataset = TextDataset(X_val, y_val, self.tokenizer, self.params.max_len)
    def test_step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        
        outputs = self.model.generate(input_ids, attention_mask=attention_mask,num_beams=3,max_new_tokens = self.params.max_output_length,length_penalty = 0.5)


        generated_ids = self.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=self.params.max_length,
        num_beams=self.params.num_beams
        )
        preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return {"preds": preds}

    def test_epoch_end(self, outputs):
        preds = []
        for output in outputs:
            preds += output["preds"]
        return {"preds": preds}
        # generated_summaries = [summary for output in outputs for summary in output['generated_summaries']]
        # with open('test.csv', 'w', newline='', encoding='utf-8') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow(['generated_summary'])
        #     for summary in generated_summaries:
        #         writer.writerow([summary])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.params.batch_size, collate_fn=self.collate_fn, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.params.batch_size, collate_fn=self.collate_fn, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=1, collate_fn=self.collate_fn_test,shuffle=False)
    

    def collate_fn_test(self, batch):
        input_sequences = [x[0] for x in batch]
        input_ids = self.tokenizer.batch_encode_plus(input_sequences, padding='max_length', return_tensors='pt', max_length=self.params.max_input_length)['input_ids']
        attention_mask = input_ids.ne(0).long()
        return input_ids, attention_mask

    def collate_fn(self, batch):

        ## 数据增强
        input_sequences = [x[0] for x in batch]
        output_sequences = [x[1] for x in batch]
        # print(input_sequences)
        input_ids = self.tokenizer.batch_encode_plus(input_sequences, padding='max_length', return_tensors='pt', max_length=self.params.max_input_length)['input_ids']
        decoder_input_ids = self.tokenizer.batch_encode_plus(output_sequences, padding='max_length', return_tensors='pt', max_length=self.params.max_output_length)['input_ids']
        # print(input_ids)

        attention_mask = input_ids.ne(0).long()
        decoder_attention_mask = decoder_input_ids.ne(0).long()
        
        return input_ids, attention_mask, decoder_input_ids, decoder_attention_mask

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        if self.warm_up:

            lr_scheduler = get_cosine_schedule_with_warmup(num_warmup_steps=int(0.1*self.params.epochs),num_training_steps=self.params.epochs,)
            scheduler = {'scheduler': lr_scheduler, 'interval': 'step', 'frequency': 1}
            get_cosine_schedule_with_warmup
            return {'optimizer': optimizer,'lr_scheduler': scheduler}

            
        else:
            return torch.optim.Adam(self.model.parameters(), lr=self.params.learning_rate)
        
        

    
