import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
# from dataset import DoctorPatientDialogueDataset
from transformers import AutoModel, AutoTokenizer
from model.pegasus import PegasusSummarizer
import argparse
import warnings
warnings.filterwarnings("ignore", message="Error loading punkt: <urlopen error [Errno 111] Connection refused>")
# Create datasets


def main(hparams):
    # pretrained_model_name = "google/pegasus-summarization"
    # tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    # model = AutoModel.from_pretrained(model_name)

    # load dataset
    # train_dataset = DoctorPatientDialogueDataset(csv_file='./MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskA/TaskA-TrainingSet.csv', tokenizer=tokenizer)

    # val_dataset = DoctorPatientDialogueDataset(csv_file='./MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskA/TaskA-ValidationSet.csv', tokenizer=tokenizer)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=32,shuffle=True)

    # val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=32,shuffle=False)
    tb_logger = TensorBoardLogger(hparams.output_dir, name="lightning_logs")
    checkpoint_callback = ModelCheckpoint(monitor='rouge1_f',
                                          mode='max',
                                          save_last=True,
                                          save_top_k=1,
                                          dirpath=os.path.join(tb_logger.log_dir, 'checkpoints'),
                                          filename='ckpt-{epoch:02d}')
    early_stop_callback = EarlyStopping(monitor='rouge1_f', patience=10,mode="max")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model = PegasusSummarizer(hparams)

    # 监控数值变化 TODO
    trainer = pl.Trainer(max_epochs=hparams.epochs,
                      accelerator='gpu', devices=hparams.gpus,
                      default_root_dir=hparams.output_dir,logger=tb_logger,
                      callbacks=[checkpoint_callback,early_stop_callback,lr_monitor]
                      )

    trainer.fit(model)
    #trainer.test(dataloaders=dataloader["test"], ckpt_path="best")

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file",type=str, default="./MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskA/TaskA-TrainingSet.csv", help="Path to the csv data training file")
    parser.add_argument("--val_file",type=str, default="./MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskA/TaskA-ValidationSet.csv", help="Path to the csv data validation file")
    parser.add_argument("--pretrained_model",type=str, default="google/pegasus-xsum", help="pretrained model and then fine-tune")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the outputs")

    parser.add_argument("--max_input_length", type=int, default=2048, help="Max length of the input sequence")
    parser.add_argument("--max_output_length", type=int, default=1024, help="Max length of the output sequence")
    parser.add_argument("--learning_rate", type=float, default=4e-6, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=3, help="batch_size")
    parser.add_argument("--gpus", type=int, default=3, help="gpus")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="gradient accumulation steps")


    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    hparams = parser.parse_args()
    main(hparams)