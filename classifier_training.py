import pytorch_lightning as pl
from model.classifier_contrastive import ContrastiveFinetuner
import argparse


def main(params):
    # Step 1: Initialize lightning module
    model = ContrastiveFinetuner(train_file=params.train_file, val_file = params.val_file, test_file = params.val_file,params=params)

    # Step 2: Initialize PyTorch Lightning trainer
    trainer = pl.Trainer(accelerator='gpu', devices=params.gpus, max_epochs=params.total_steps, default_root_dir=params.output_dir)

    # Step 3: Train the model
    trainer.fit(model)



if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    # distilbert-base-uncased
    parser.add_argument("--train_file",type=str, default="./MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskA/TaskA-TrainingSet.csv", help="Path to the csv data training file")
    parser.add_argument("--val_file",type=str, default="./MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskA/TaskA-ValidationSet.csv", help="Path to the csv data validation file")
    parser.add_argument("--test_file",type=str, default="./MEDIQA-Chat-Training-ValidationSets-Feb-10-2023/TaskA/TaskA-ValidationSet.csv", help="Path to the csv data validation file")
    parser.add_argument("--model_name",type=str, default="distilbert-base-uncased", help="pretrained model and then fine-tune")
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="Directory to save the outputs")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the outputs")
    parser.add_argument("--total_steps",type= int,default=50)
    parser.add_argument("--batch_size",type= int,default=8)
    parser.add_argument("--max_input_length", type=int, default=1024, help="Max length of the input sequence")
    parser.add_argument("--max_output_length", type=int, default=1024, help="Max length of the output sequence")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for the optimizer")
    parser.add_argument("--gpus", type=int, default=1, help="gpus")
    parser.add_argument("--constrast", type=bool, default=False, help="contrastive_loss")
    parser.add_argument("--gc_step", type=int, default=8, help="Max length of the input sequence")


    params = parser.parse_args()
    main(params)


