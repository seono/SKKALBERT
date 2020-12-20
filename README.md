# SKKALBERT

기계 독해 성능 개선을 위한 데이터 증강 기법 구현

[ALBERT-base](https://github.com/google-research/albert), [ALBERT-large](https://github.com/google-research/albert) 모델 사용

[SQuAD 1.1](https://rajpurkar.github.io/SQuAD-explorer/)와 [KorQuAD 1.0](https://korquad.github.io/category/1.0_KOR.html) 데이터를 이용하여 실험


### [Example Code](https://colab.research.google.com/drive/1hTS7fgwPHWL6ijfF7Kgcyxbh02cN1tLw?usp=sharing)
학습, 데이터 제한 5000, 동의어 교체 편집 기법 사용(--eda_type sr)
```shell
python3 run_qa_small.py \
  --checkpoint bert_small/korquad_small.bin \
  --config_file bert_small/bert_small.json \
  --output_dir bert_small/output/limit_data_5000/ \
  --vocab_file bert_small/ko_vocab_32k.txt \
  --train_file data/korquad/KorQuAD_v1.0_train.json \
  --max_seq_length 384 \
  --doc_stride 128 \
  --max_query_length 128 \
  --max_answer_length 30 \
  --per_gpu_train_batch_size 6 \
  --learning_rate 3e-5 \
  --num_train_epochs 20.0 \
  --adam_epsilon 1e-6 \
  --warmup_proportion 0.1 \
  --train_data_limit 5000 \
  --min_score 0.85 \
  --eda_type sr
```
평가
```shell
python3 eval_qa.py \
  --check_list True \
  --checkpoint bert_small/output/limit_data_5000/ \
  --config_file bert_small/bert_small.json \
  --output_dir bert_small/output/ \
  --vocab_file bert_small/ko_vocab_32k.txt \
  --predict_file data/korquad/KorQuAD_v1.0_dev.json \
  --max_seq_length 384 \
  --doc_stride 128 \
  --max_query_length 128 \
  --max_answer_length 30 \
  --batch_size 4 \
  --n_best_size 20
```
결과
```shell
Evaluating: 100% 1832/1832 [00:28<00:00, 64.14it/s]
36.00623484586075 56.618519894936234
Evaluating: 100% 1832/1832 [00:28<00:00, 64.52it/s]
37.72081745756841 57.37260467328269
...
```
## 데이터 증강 기법

https://user-images.githubusercontent.com/47937302/102715176-e7aabd00-4316-11eb-8cb5-988ff97208f4.png

SR(동의어 교체), RD(무작위 삭제), RI(무작위 삽입), RS(무작위 교체)

### 문장 부분 단어 단위 기법 효과
위의 SR, RD, RI, RS 기법별 성능 차이를 비교

### 문단 부분 문장 단위 기법 효과
문장 단위는 무작위 삭제(RD)와 무작위 교체(RS)를 수행


### 모델 규모에 따른 성능 향상

ALBERT-base와 ALBERT-large모델 비교

### 데이터 크기에 따른 문장 단위 기법 성능 향상

주어진 학습데이터를 제한하여 기법 성능 향상 정도를 비교
