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
import pandas as pd


def main(hparams):
    early_stop_callback = EarlyStopping(monitor='rouge1_f', patience=10,mode="max")

    model = PPOTextSummarization(lr=1e-3, gamma=0.99, clip_eps=0.2)
    trainer = pl.Trainer(gpus=1, max_epochs=10)
    trainer.fit(model, datamodule=datamodule)
    result = trainer.test(ckpt_path="best")
    df = pd.DataFrame(result[0])
    result_output_path = os.path.join(hparams.output_dir,"test_results.csv")

    df.to_csv(result_output_path, index=False)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file",type=str, default="./MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskA/TaskA-TrainingSet.csv", help="Path to the csv data training file")
    parser.add_argument("--val_file",type=str, default="./MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskA/TaskA-ValidationSet.csv", help="Path to the csv data validation file")
    parser.add_argument("--test_file",type=str, default="./MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskA/TaskA-ValidationSet.csv", help="Path to the csv data validation file")
    parser.add_argument("--pretrained_model",type=str, default="philschmid/bart-large-cnn-samsum", help="pretrained model and then fine-tune")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the outputs")

    parser.add_argument("--max_input_length", type=int, default=1024, help="Max length of the input sequence")
    parser.add_argument("--max_output_length", type=int, default=1024, help="Max length of the output sequence")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=2, help="batch_size")
    parser.add_argument("--gpus", type=int, default=3, help="gpus")
    parser.add_argument("--beams", type=int, default=3, help="gpus")
    parser.add_argument("--warm", type=bool, default=False, help="warm_up")
    parser.add_argument("--chrf_score", type=bool, default=False, help="chrf_score")
    parser.add_argument("--rouge_score", type=bool, default=True, help="rouge_score")
    parser.add_argument("--use_r_drop", type=bool, default=False, help="use_r_drop")

    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    hparams = parser.parse_args()
    main(hparams)