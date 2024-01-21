import pandas as pd
import os
import re
from sentence_transformers import InputExample
from typing import Optional, Union
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from datasets import Dataset
from torch.utils.data import DataLoader

class TokenizedSentencesDataset:
    def __init__(self, sentences, tokenizer, max_length, cache_tokenization=False):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization

    def __getitem__(self, item):
        if not self.cache_tokenization:
            return self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)

        self.sentences[item] = self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)

def get_examples(dataset, split, language=None):
    samples = []

    if language:
      dataset = dataset.filter(lambda x: x['Language'] == language)

    for example in dataset[split]:
        if split == 'train':
            samples.append(InputExample(texts=example['Pair'], label=example['Score']))
            samples.append(InputExample(texts=example['Pair'], label=example['Score']))
        elif split == 'dev':
            samples.append(InputExample(texts=example['Pair'], label=example['Score']))
        else:
            samples.append(example['Pair'])
            
    return samples

def load_data(path: str, mode='train', language=None):
  df = pd.read_csv(path)
  df['Language'] = language
  df['Pair'] = df.Text.apply(lambda x: x.split('\n'))
  if mode=='test':
    df['Score'] = [-1.] * len(df)
  return df.drop('Text', axis='columns')

def get_batches(data, batch_size, shuffle=False):
  return DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle)

def get_mlm_data(dataset, tokenizer=None, max_length=512, do_whole_word_mask=False, mlm_prob=1.0, language=None):
  train_sentences = []
  for pair in dataset['Pair']:
    train_sentences += pair
  train_dataset = TokenizedSentencesDataset(train_sentences, tokenizer, max_length)
  if do_whole_word_mask:
    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)
  else:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)

  return train_dataset, data_collator

def get_str_data(dataset, language=None):
  train_samples = get_examples(dataset, split='train')
  dev_samples = get_examples(dataset, split='dev')
  test_samples = get_examples(dataset, split='test')

  return train_samples, test_samples

def get_data(languages):
  train_data_list = []
  dev_data_list = []
  test_data_list = []

  for lang in languages:
    train_path = f'Track A/{lang}/{lang}_train.csv'
    dev_path = f'Track A/{lang}/{lang}_dev_with_labels.csv'
    test_path = f'Track A/{lang}/{lang}_test.csv'

    train_data = load_data(train_path, language=lang, mode='train')
    dev_data = load_data(train_path, language=lang, mode='validation')
    test_path = load_data(train_path, language=lang, mode='test')

    train_data_list.append(train_data)
    dev_data_list.append(dev_data)
    test_data_list.append(test_data)
  
  whole_train_data = pd.concat(train_data_list).reset_index().drop('index', axis='columns')
  whole_dev_data = pd.concat(dev_data_list).reset_index().drop('index', axis='columns')
  whole_test_data = pd.concat(test_data_list).reset_index().drop('index', axis='columns')

  return whole_train_data, whole_dev_data, whole_test_data

if __name__ == '__main__':
    train_file = '/content/Semantic_Relatedness_SemEval2024/Track A/amh/amh_train.csv'
    test_file = '/content/Semantic_Relatedness_SemEval2024/Track A/amh/amh_dev.csv'

    train_data = load_data(train_file, 'train')
    test_data = load_data(test_file, mode='test')
    print("[INFO] Loading data completed.")
    print(test_data.shape)
