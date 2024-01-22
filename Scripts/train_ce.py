import argparse
import os
import pandas as pd
import math
import os
import logging
from datetime import datetime

from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from load_data import *
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers import InputExample
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a STR end-to-end.")
    parser.add_argument("--model_name", required=True, help="Pretrained model name or path.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--use_fp16", action="store_true", help="Use FP16 for training if supported.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for MLM training.")
    parser.add_argument("--language", default=None, help="Choose a specific language to train.")
    parser.add_argument("--mlm", default=True, help="Whether the chosen was continual pretrained.")
    # parser.add_argument("--save_model_path", required=True, help="Save path")
    
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_arguments()
    
    # Extract arguments
    model_name = args.model_name
    num_train_epochs = args.num_train_epochs
    use_fp16 = args.use_fp16
    batch_size = args.batch_size
    language = args.language
    # Load data
    if language:
      languages = [language]
    else:
      languages = ['amh', 'arq', 'ary', 'eng', 'esp', 'hau', 'kin', 'mar', 'tel']
      
    train_data, dev_data, test_data = get_data(languages)
    dataset = create_dataset(train_data, dev_data, test_data)
    
    # Train cross-encoder
    train_samples, dev_samples, test_samples = get_str_data(dataset, language=language)
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
    
    evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name='sts-dev')
    
    # Get model   
    if mlm:
        model_name = model_name.split('/')[-1]
        model = CrossEncoder(f'./saved/mlm/{model_name}', max_length=512, num_labels=1)
    else:
        model = CrossEncoder(model_name, max_length=512, num_labels=1)
    # Configure the training
    warmup_steps = math.ceil(len(train_dataloader) * num_train_epochs * 0.1) #10% of train data for warm-up
    logger.info("Warmup-steps: {}".format(warmup_steps))
    
    
    # Train the model
    model.fit(train_dataloader=train_dataloader,
              evaluator=evaluator,
              epochs=num_train_epochs,
              warmup_steps=warmup_steps,
              evaluation_steps=256,
              use_amp=True,
              output_path='./saved/str')
    
    print("Training done")
    
    os.makedirs(f'submission/{model_name}/dev', exist_ok=True)
    os.makedirs(f'submission/{model_name}/test', exist_ok=True)
  
    for lang in languages:
        print("[INFO] Creating submission for: ", lang)
        dev_samples = get_examples(dataset, 'dev', lang)
        dev_samples = [sample.texts for sample in dev_samples]
        test_samples = get_examples(dataset, 'test', lang)
        dev_predictions = model.predict(dev_samples).tolist()
        test_predictions = model.predict(test_samples).tolist()
        pd.DataFrame({'PairID': dataset['dev'].filter(lambda x: x['Language'] == lang)['PairID'], 'Pred_Score': dev_predictions}).to_csv(f'submission/{model_name}/dev/pred_{lang}_a.csv', index=False)
        pd.DataFrame({'PairID': dataset['test'].filter(lambda x: x['Language'] == lang)['PairID'], 'Pred_Score': test_predictions}).to_csv(f'submission/{model_name}/test/pred_{lang}_a.csv', index=False)
