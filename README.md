# SKKALBERT

This is a repository of Korean ALBERT model.

run_korquad dataset 2.1버전도 상관없음 evaluate할 때 dev파일도 같으니 -> 오히려 데이터 줄어서 추천
	

>20.04.26
 run_korquad.py 업데이트
 run_korquad 1136 1204 line에 dataset 파일 여러 개일 때 load다 하도록 수정 train_examples = [] 배열에 read_squad_example(input_file)의 배열이 for문 돌면서 모든파일에 대해 정보가 추가되는 방식.
    이	후에 읽어온 특징들을 계산하는데 이 때 시간이 가장많이드는듯
	사용 방법 python2에 tensorflow==1.14버전으로 (python3버전은 안해봐서 모름)
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
개선점 :
    디렉토리로 묶어서 --train_dir = gs://korquad/og_train_v2/ 전달하는 방법으로 하면 더 깔끔
    flag도 추가해서 --is_dir=True 이런 flag도 있으면 좋을 듯