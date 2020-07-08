# SKKALBERT

This is a repository of Korean ALBERT model.

run_korquad dataset 2.1버전도 상관없음 evaluate할 때 dev파일도 같으니 -> 오히려 데이터 줄어서 추천
	

>20.04.26
 run_korquad.py 업데이트

run_korquad 1136 1204 line에 dataset 파일 여러 개일 때 load다 하도록 수정

train_examples = [] 배열에 read_squad_example(input_file)의 배열이 for문 돌면서 모든파일에 대해 정보가 추가되는 방식.

이	후에 읽어온 특징들을 계산하는데 이 때 시간이 가장많이드는듯

>사용 방법 

python2에 tensorflow==1.14버전으로 (python3버전은 안해봐서 모름)

python2 run_korquad.py <parameter 들> --do_train=True --train_file=”파일1 파일2 파일3 …” …. 띄어쓰기로 구분하고 “”로 한번에 parameter전달해야하고 마지막에 parameter v2=True라는 flag도 추가해줘야함

	e.g) python2 ./run_korquad.py\
    --init_checkpoint=gs://korquad/pretrained_files/bert_model.ckpt\
    --bert_config_file=gs://korquad/pretrained_files/config/bert_config.json\
    --vocab_file=gs://korquad/pretrained_files/vocab/vocab.txt\
    --do_train=True\
    --train_file="gs://korquad/og_train_v2/korquad2.1_train_00.json gs://korquad/og_train_v2/korquad2.1_train_01.json gs://korquad/og_train_v2/korquad2.1_train_02.json gs://korquad/og_train_v2/korquad2.1_train_03.json gs://korquad/og_train_v2/korquad2.1_train_04.json gs://korquad/og_train_v2/korquad2.1_train_05.json gs://korquad/og_train_v2/korquad2.1_train_06.json gs://korquad/og_train_v2/korquad2.1_train_07.json gs://korquad/og_train_v2/korquad2.1_train_08.json gs://korquad/og_train_v2/korquad2.1_train_09.json gs://korquad/og_train_v2/korquad2.1_train_10.json gs://korquad/og_train_v2/korquad2.1_train_11.json gs://korquad/og_train_v2/korquad2.1_train_12.json gs://korquad/og_train_v2/korquad2.1_train_13.json gs://korquad/og_train_v2/korquad2.1_train_14.json gs://korquad/og_train_v2/korquad2.1_train_15.json gs://korquad/og_train_v2/korquad2.1_train_16.json gs://korquad/og_train_v2/korquad2.1_train_17.json gs://korquad/og_train_v2/korquad2.1_train_18.json gs://korquad/og_train_v2/korquad2.1_train_19.json gs://korquad/og_train_v2/korquad2.1_train_20.json gs://korquad/og_train_v2/korquad2.1_train_21.json gs://korquad/og_train_v2/korquad2.1_train_22.json gs://korquad/og_train_v2/korquad2.1_train_23.json gs://korquad/og_train_v2/korquad2.1_train_24.json gs://korquad/og_train_v2/korquad2.1_train_25.json gs://korquad/og_train_v2/korquad2.1_train_26.json gs://korquad/og_train_v2/korquad2.1_train_27.json gs://korquad/og_train_v2/korquad2.1_train_28.json gs://korquad/og_train_v2/korquad2.1_train_29.json gs://korquad/og_train_v2/korquad2.1_train_30.json gs://korquad/og_train_v2/korquad2.1_train_31.json gs://korquad/og_train_v2/korquad2.1_train_32.json gs://korquad/og_train_v2/korquad2.1_train_33.json gs://korquad/og_train_v2/korquad2.1_train_34.json gs://korquad/og_train_v2/korquad2.1_train_35.json gs://korquad/og_train_v2/korquad2.1_train_36.json gs://korquad/og_train_v2/korquad2.1_train_37.json gs://korquad/og_train_v2/korquad2.1_train_38.json"\
    --do_predict=True\
    --predict_file="gs://korquad/og_dev_v2/korquad2.1_dev_00.json gs://korquad/og_dev_v2/korquad2.1_dev_01.json gs://korquad/og_dev_v2/korquad2.1_dev_02.json gs://korquad/og_dev_v2/korquad2.1_dev_03.json gs://korquad/og_dev_v2/korquad2.1_dev_04.json"\
    --train_batch_size=16 --learning_rate=1e-4 --num_train_epochs=1.0 --max_seq_length=384 --doc_stride=128 --output_dir=gs://korquad/out_put/ --use_tpu=True --tpu_name=korquad2 --do_lower_case=False --v2=True

>20.04.26~ 개선점

    디렉토리로 묶어서 --train_dir = gs://korquad/og_train_v2/ 전달하는 방법으로 하면 더 깔끔
    
    flag도 추가해서 --is_dir=True 이런 flag도 있으면 좋을 듯

>20.07.08
    기존의 run_korquad.py 및 google bert에서 제공하는 squad와 korquad1.0버전용 question answering task code및 모델은 커서 colab이나 가지고 있는 computational resource만으로 부족하여 더 가벼운 모델을 통하여 baseline을 구축한다.

    baseline 구축에 있어서 여러 문제점이 발견되었다.

1) 제공된 코드또한 korquad 1.0버전이나 다른 데이터가 작은 task를 학습하기에는 수월하지만 korquad2.0버전의 경우에는 데이터가 크기에 colab에서 OOM문제가 발생하거나, code running 도중 페이지가 다운되는 경우가 허다하였다. (8개 이상의 지문을 넘어가는 경우 멈춤)

2) bert모델에서 context를 읽고 질문과 해당 answer에 대한 data를 읽는 도중 현재 context와 answer, answer start 등 많은 정보를 출력하는 코드가 있는데 이 때 colab에서 해당 출력들을 다른 작업하다가 확인하는 경우 출력들이 많이 쌓여서 다운되는 것으로 보인다.

따라서, 출력 부분은 최대한 줄였으며 과정을 나누었다.

3) 과정을 두 부분으로 나누었다 본래 코드에서는 context와 question, answer 세 가지를 통하여 데이터에서 feature를 뽑아낸 뒤 이 feature를 통하여 모델을 학습한다. 이 부분을 feature를 뽑아내어 데이터로 저장하는 part1, 뽑아낸 데이터를 다시 load하여 학습하는 part2로 나누었다. 해당 방법을 통하여 colab이 다운되는 경우를 최대한 피하였다.

4) 파트를 나누거나 뽑아낸 feature를 저장하기위하여 run_qa.py에서 argument를 여러개 추가하였고 그와 더불어 main함수에 있어서 line 291부터 333까지를 변경하였다. argument를 통하여 feature extract를 하는지 feature load를 하는지를 구분하여 진행가능하다. 또한 feature extract하는 데있어서 utils 디렉토리 내에 korquad_utils.py 또한 변경하였다. 해당 부분은 korquad2.0버전의 json 구조에 맞게 데이터를 읽거나 출력을 줄이는 방향으로 수정하였다.

>추가된 argument(parameter)
    
    parser.add_argument("--train_file_name", default='train/korquad2.1_train_', type=str,
                        help="KorQuAD json directory for training. E.g., train")

    korquad2.1_train_xx.json 파일을 불러내기 위하여 해당 번호전까지의 korquad2.1_train_ 파일명을 입력하면된다.

    parser.add_argument("--extract_exam", default=False, type=bool,
                        help="if True only extract examples from train_file")
                        
    feature extract 여부를 묻는다. 입력하지 않는 경우 False로 feature추출을 하지 않게 된다.

    parser.add_argument("--output_exam", default='train/examples/', type=str,
                        help="korquad examples.")

    추출한 feature 데이터를 저장할 경로

    parser.add_argument("--train_file_start", default=0, type=int,
                        help="KorQuAD json file start number. E.g., 0")

    parser.add_argument("--train_file_end", default=38, type=int,
                        help="KorQuAD json file end number. E.g., 38")

    0부터 38까지의 데이터가 있는데 start와 end를 사용하여 부분적으로 train file에서 extract할 수 있다.

    parser.add_argument("--example_dir", default='train/examples/', type=str,
                        help="korquad extracted examples")

    load하려는 example directory의 path이며 이미 코드내에 feature파일은 korquad_example_xx_x.bin파일명으로 저장하게 통일하였으므로 path만 입력

    parser.add_argument("--load_examples", default=False, type=bool,
                        help="if True only load examples and do not train")

    load example 여부

    parser.add_argument("--example_file_start", default=0, type=int,
                        help="KorQuAD example file start number. E.g., 0")

    parser.add_argument("--example_file_end", default=38 , type=int,
                        help="KorQuAD example file end number. E.g., 38")

    parser.add_argument("--extract_start", default= 0, type= int,
                        help="if extract example stop while extracting korquad json file. E.g., 3")

    extract_start는 colab이나 본인 노트북에서 추출하다가 멈출경우 1000개 데이터에서 처음부터 하기에는 부담이 있어서 만들었다.

>run code e.g.

    python run_qa.py \
  --checkpoint bert_small/korquad_00.bin \
  --config_file bert_small/bert_small.json \
  --vocab_file bert_small/ko_vocab_32k.txt \
  --output_dir bert_small/output/ \
  --train_file_name train/korquad2.1_train_ \
  --output_exam train/examples/ \
  --extract_exam True \
  --extract_start 2\
  --example_dir train/examples/ \
  --train_file_start 29 \
  --train_file_end 36\
  --max_seq_length 384 \
  --doc_stride 128 \
  --max_query_length 64 \
  --max_answer_length 300 \
  --parse_size 250 \
  --per_gpu_train_batch_size 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 4.0 \
  --adam_epsilon 1e-6 \
  --warmup_proportion 0.1