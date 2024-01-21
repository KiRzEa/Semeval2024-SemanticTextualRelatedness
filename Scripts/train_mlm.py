import argparse
import os
import pandas as pd
import math

from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from load_data import *

import warnings
warnings.filterwarnings("ignore")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a STR end-to-end.")
    parser.add_argument("--model_name", required=True, help="Pretrained model name or path.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--use_fp16", action="store_true", help="Use FP16 for training if supported.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for MLM training.")
    parser.add_argument("--language", default=None, help="Choose a specific language to train.")
    # parser.add_argument("--save_model_path", required=True, help="Save path")
    
    return parser.parse_args()
  
if __name__ == '__main__':
    args = parse_arguments()
    
    # Extract arguments
    model_name = args.model_name
    dataset_name = args.dataset_name
    num_train_epochs = args.num_train_epochs
    use_fp16 = args.use_fp16
    batch_size = args.batch_size
    language = args.language
    
    # Masked Language Modeling
    # Load the model
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model_name = model_name.split('/')[-1]
    output_dir = f"./saved/mlm/{model_name}"
    max_length = 256
    mlm_prob = 0.15
    
    languages = ['amh', 'arq', 'ary', 'eng', 'esp', 'hau', 'kin', 'mar', 'tel']

    train_data, _, _ = get_data(languages)
    
    train_dataset, data_collator_fn = get_mlm_data(train_data, tokenizer=tokenizer, language=language)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        prediction_loss_only=True,
        fp16=use_fp16,
        save_total_limit=None,
        save_strategy='no'
    )
    
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator_fn,
        train_dataset=train_dataset,
    )
    
    trainer.train()
    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
