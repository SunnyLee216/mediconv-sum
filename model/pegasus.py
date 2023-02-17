import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torchmetrics.text.rouge import ROUGEScore
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
            self.outputs = self.data['section_text'].apply(lambda x: x[:self.max_output_length]) 
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


    def forward(self, input_ids, attention_mask, decoder_input_ids):
        outputs = self.model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
        return outputs[0]

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels, labels_mask = batch
        logits = self.forward(input_ids, attention_mask,labels)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels, labels_mask = batch
        logits = self.forward(input_ids, attention_mask,labels)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)

        # Add the ROUGE evaluation

        outputs = self.model.generate(input_ids, attention_mask=attention_mask,num_beams=5)
        generated_summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        target_summaries = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        rouge = ROUGEScore()
        rouge_score = rouge(generated_summaries, target_summaries)
        self.log("rouge_score", rouge_score,sync_dist=True)
        return {'val_loss': loss, 'rouge_score': rouge_score}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        
        avg_rouge = torch.stack([x['rouge_score'] for x in outputs]).mean()
        self.log("rouge_score", avg_rouge,sync_dist=True)
        return {'avg_val_loss': avg_loss,'rouge_score': avg_rouge}

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
        return torch.optim.Adam(self.model.parameters(), lr=self.params.learning_rate)

    
