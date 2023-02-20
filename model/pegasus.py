import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics import CHRFScore
import torch 
import os
import torch.nn as nn
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message="Error loading .*: <urlopen error [Errno 111] Connection refused>")
class DoctorPatientDialogueDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, max_input_length, max_output_length):
        self.data = pd.read_csv(csv_file)
        # self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.inputs = self.data['dialogue'].apply(lambda x: x[:self.max_input_length])
        # TODO 把header_section 的部分拼合早section_text
        if 'section_text' in self.data.columns:
            print('TaskA data loading....')
            self.outputs = self.data.apply(lambda x: x['section_header'] + ' ' + x['section_text'], axis=1)
            self.outputs = self.outputs.apply(lambda x: x[:self.max_output_length])
            

        else:
            print('TaskB or TaskC data loading....')
            self.outputs = self.data['note'].apply(lambda x: x[:self.max_output_length]) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_sequence = self.inputs[index]
        output_sequence = self.outputs[index]
        return input_sequence, output_sequence



class PegasusSummarizer(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.train_dataset = DoctorPatientDialogueDataset(self.params.train_file, self.params.max_input_length, self.params.max_output_length)
        self.val_dataset = DoctorPatientDialogueDataset(self.params.val_file, self.params.max_input_length, self.params.max_output_length)
        # pretrained_model_name = "google/pegasus-summarization"
        pretrained_model_name = self.params.pretrained_model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.rouge = ROUGEScore()
        self.chrf = CHRFScore()
        self.learning_rate = params.learning_rate

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        outputs = self.model(input_ids, attention_mask=attention_mask,labels=decoder_input_ids,output_hidden_states=True,return_dict=True)
        return outputs.loss,outputs.logits

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        input_ids, attention_mask, labels, labels_mask = batch
        loss,logits = self.forward(input_ids, attention_mask,labels)

        N = self.params.accumulation_steps

        loss = loss / N
        self.manual_backward(loss)
        # accumulate gradients of N batches
        if (batch_idx + 1) % N == 0:
            opt.step()
            opt.zero_grad()
            sch.step()

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
        outputs = self.model.generate(input_ids, attention_mask=attention_mask,num_beams=5,max_new_tokens = self.params.max_output_length,length_penalty=-1)
        generated_summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        target_summaries = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        rouge_score = self.rouge(generated_summaries, target_summaries)
        chrf_score = self.chrf(generated_summaries, target_summaries)
        self.log("chrf_score", chrf_score,sync_dist=True, on_step=False, on_epoch=True,prog_bar=True)
        self.log("rouge_score", rouge_score,sync_dist=True, on_step=False, on_epoch=True,prog_bar=True)
        self.log("rougeLsum_f", rouge_score['rougeLsum_fmeasure'],sync_dist=True, on_step=False, on_epoch=True)
        self.log("rouge1_f", rouge_score['rouge1_fmeasure'],sync_dist=True, on_step=False, on_epoch=True)
        self.log("rouge2_f", rouge_score['rouge2_fmeasure'],sync_dist=True, on_step=False, on_epoch=True)

        return {'val_loss': loss, "chrf_score":chrf_score, "rougeLsum_fmeasure":rouge_score['rougeLsum_fmeasure'],"rouge1_f":rouge_score['rouge1_fmeasure'],"rouge2_f":rouge_score['rouge2_fmeasure']}

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

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.params.batch_size, collate_fn=self.collate_fn, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.params.batch_size, collate_fn=self.collate_fn, shuffle=False)


    def collate_fn(self, batch):
        input_sequences = [x[0] for x in batch]
        output_sequences = [x[1] for x in batch]
        # print(input_sequences)
        input_ids = self.tokenizer.batch_encode_plus(input_sequences, padding=True, return_tensors='pt', max_length=self.params.max_input_length)['input_ids']
        decoder_input_ids = self.tokenizer.batch_encode_plus(output_sequences, padding=True, return_tensors='pt', max_length=self.params.max_output_length)['input_ids']

        attention_mask = input_ids.ne(0).long()
        decoder_attention_mask = decoder_input_ids.ne(0).long()
        
        return input_ids, attention_mask, decoder_input_ids, decoder_attention_mask

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.learning_rate*0.1, max_lr=self.learning_rate, 
                                                     step_size_up=int(self.params.epochs*0.1), step_size_down=self.params.epochs-int(self.params.epochs*0.1),cycle_momentum=False)
        scheduler = {'scheduler': lr_scheduler, 'interval': 'step', 'frequency': 1}
        # return torch.optim.Adam(self.model.parameters(), lr=self.params.learning_rate)
        return {'optimizer': optimizer,'lr_scheduler': scheduler}
        

    
