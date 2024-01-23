#!/bin/bash

pip install -r requirements.txt --quiet

python train_ce.py \
        --model_name "FacebookAI/xlm-roberta-large" \
        --batch_size 16 \
        --num_train_epochs 10 \
        --mlm False
# -------------------------------------------------
python train_ce.py \
        --model_name "microsoft/mdeberta-v3-base" \
        --batch_size 32 \
        --num_train_epochs 10 \
        --mlm False
# -------------------------------------------------
python train_ce.py \
        --model_name "bert-base-multilingual-cased" \
        --batch_size 32 \
        --num_train_epochs 10 \
        --mlm False
# -------------------------------------------------
python train_ce.py \
        --model_name "hiiamsid/sentence_similarity_spanish_es" \
        --batch_size 32 \
        --num_train_epochs 10 \
        --language "esp" \
        --mlm False
# -------------------------------------------------
python train_ce.py \
        --model_name "l3cube-pune/marathi-sentence-similarity-sbert" \
        --batch_size 32 \
        --num_train_epochs 10 \
        --language "mar" \
        --mlm False
 # -------------------------------------------------       
python train_ce.py \
        --model_name "l3cube-pune/telugu-sentence-similarity-sbert" \
        --batch_size 32 \
        --num_train_epochs 10 \
        --language "tel" \
        --mlm False
# -------------------------------------------------
python train_ce.py \
        --model_name "mbeukman/xlm-roberta-base-finetuned-kinyarwanda-finetuned-ner-swahili" \
        --batch_size 32 \
        --num_train_epochs 10 \
        --language "kin" \
        --mlm False
# -------------------------------------------------
python train_ce.py \
        --model_name "microsoft/deberta-v3-large" \
        --batch_size 16 \
        --num_train_epochs 10 \
        --language "eng" \
        --mlm False
# -------------------------------------------------
python train_ce.py \
        --model_name "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7" \
        --batch_size 32 \
        --num_train_epochs 10 \
        --mlm False
# -------------------------------------------------
python train_ce.py \
        --model_name "intfloat/multilingual-e5-base" \
        --batch_size 32 \
        --num_train_epochs 10 \
        --mlm False
# -------------------------------------------------
python train_ce.py \
        --model_name "intfloat/multilingual-e5-large" \
        --batch_size 16 \
        --num_train_epochs 10 \
        --mlm False
# --------------------------------------------------
python train_ce.py \
        --model_name "intfloat/e5-large-v2" \
        --batch_size 16 \
        --num_train_epochs 10 \
        --language "eng" \
        --mlm False
        
zip -r submission.zip submission
