import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
# from dataset import DoctorPatientDialogueDataset
from model.summarizer import Summarizer
from model.long_summarizer import Long_Summarizer
from model.long_summarize_clo import Long_Summarizer_clo
import argparse
import warnings
from pytorch_lightning.strategies import DeepSpeedStrategy
warnings.filterwarnings("ignore", message="Error loading punkt: <urlopen error [Errno 111] Connection refused>")
# Create datasets

import os


def main(hparams):
    # Setting
    # Stancld/longt5-tglobal-large-16384-pubmed-3k_steps
    tb_logger = TensorBoardLogger(hparams.output_dir, name="lightning_logs")
    checkpoint_callback = ModelCheckpoint(monitor='avg_rouge_1',
                                          mode='max',
                                          save_last=True,
                                          save_top_k=1,
                                          dirpath=os.path.join(tb_logger.log_dir, 'checkpoints'),
                                          filename='best-{epoch:02d}')
    early_stop_callback = EarlyStopping(monitor='avg_rouge_1', patience=20,mode="max")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Summarizer or colossalai
    if hparams.strategy=="deepspeed_stage_3_offload" or hparams.strategy=="deepspeed_stage_3":
        model = Long_Summarizer_clo(hparams)
    #     trainer = pl.Trainer(
    #     accelerator="gpu",
    #     default_root_dir=hparams.output_dir,logger=tb_logger,
    #     callbacks=[checkpoint_callback,early_stop_callback,lr_monitor],
    #     devices=hparams.gpus,
    #     strategy=DeepSpeedStrategy(
    #         stage=3,
    #         offload_optimizer=True,
    #         offload_parameters=True,
    #     ),
    #     precision=hparams.precision,
    #     max_epochs=hparams.epochs
    # )
        hparams.strategy = DeepSpeedStrategy(stage=3,
        offload_optimizer=True,
        offload_parameters=True,)
    else:
        model = Long_Summarizer(hparams)

    

    
    trainer = pl.Trainer(max_epochs=hparams.epochs,
                    accelerator='gpu', devices=hparams.gpus,
                    default_root_dir=hparams.output_dir,logger=tb_logger,
                    callbacks=[checkpoint_callback,early_stop_callback,lr_monitor],
                    strategy=hparams.strategy,
                    accumulate_grad_batches=hparams.accumulation_steps,
                    precision=hparams.precision,
                    
                    )
#     

    trainer.fit(model)
    # result = trainer.test(ckpt_path=os.path.join(tb_logger.log_dir, 'checkpoints','best.ckpt'))
    
    # df = pd.DataFrame(result[0])
    # result_output_path = os.path.join(hparams.output_dir,"test_results.csv")

    # df.to_csv(result_output_path, index=False)
    #trainer.test(dataloaders=dataloader["test"], ckpt_path="best")

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    # google/long-t5-tglobal-base
    # google/long-t5-tglobal-large
    # google/long-t5-tglobal-xl
    # Stancld/longt5-tglobal-large-16384-pubmed-3k_steps
    # google/flan-t5-base
    # TaskA/TaskA-TrainingSet.csv
    # TaskA/TaskA-ValidationSet.csv
    # TaskB/taskB_training_split.csv
    # TaskB/taskB_training_split.csv
    parser.add_argument("--train_file",type=str, default="./MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskB/taskB_training_split.csv", help="Path to the csv data training file")
    parser.add_argument("--val_file",type=str, default="./MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskB/taskB_validation_split.csv", help="Path to the csv data validation file")
    parser.add_argument("--test_file",type=str, default="./MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskB/taskB_validation_split.csv", help="Path to the csv data validation file")
    parser.add_argument("--pretrained_model",type=str, default="google/long-t5-tglobal-base", help="pretrained model and then fine-tune")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the outputs")

    parser.add_argument("--max_input_length", type=int, default=1024, help="Max length of the input sequence")
    parser.add_argument("--max_output_length", type=int, default=1024, help="Max length of the output sequence")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
    parser.add_argument("--gpus", type=int, default=2, help="gpus")
    parser.add_argument("--beams", type=int, default=2, help="beams")
    parser.add_argument("--warm", type=bool, default=True, help="warm_up")
    parser.add_argument("--chrf_score", type=bool, default=False, help="chrf_score")
    parser.add_argument("--rouge_score", type=bool, default=True, help="rouge_score")
    parser.add_argument("--use_r_drop", type=bool, default=False, help="use_r_drop")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="gradient accumulation steps")
    parser.add_argument("--is_sled", type=bool, default=False, help="sled")
    parser.add_argument("--auto_lr", type=bool, default=True, help="have better not use")
    parser.add_argument("--strategy", type=str, default="ddp", help="strategy:ddp/ddp_sharded/fsdp_native/fsdp/colossalai/deepspeed_stage_2/deepspeed_stage_2_off_load/deepspeed_stage_3")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--precision", type=str, default=32, help="presicion")
    parser.add_argument("--is_split", type=str, default=None, help="HISTORY OF PRESENT ILLNESS,PHYSICAL EXAM,RESULTS,ASSESSMENT AND PLAN")
    hparams = parser.parse_args()
    main(hparams)