#!/bin/bash

ulimit -S -n 4096

export PYTHONPATH=`pwd`:$PYTHONPATH

conf_dir=conf/large
exp_dir=exp/diarization/correct

# train
train_dir=/path/EEND/egs/mini_librispeech/v1/data/simu/data/train_clean_100_ns2_beta2_10000 # own path
dev_dir=/path/EEND/egs/mini_librispeech/v1/data/simu/data/dev_clean_ns2_beta2_500  # own path
model_dir=$exp_dir/models
train_conf=$conf_dir/train.yaml

affix=simu

# inference
infer_conf=$conf_dir/infer.yaml
test_dir=/path/EEND/egs/mini_librispeech/v1/data/simu/data/test_clean_ns2_beta2_500  # own path
test_model=$model_dir/avg.th
infer_out_dir=$exp_dir/infer/$affix

# scoring
work=$infer_out_dir/.work
scoring_dir=$exp_dir/score/$affix

stage=1
gpu=6

# Prepare initial speaker activity
base_model=exp/diarization/sa-eend/models/avg.th  # baseline model path
sas_out_dir=exp/diarization/sa-eend/infer     # baseline sas output 
if [ $stage -le 0 ]; then
    echo "Prepare initial speaker activity"
	for data in train dev test; do
		sas_out=$sas_out_dir/$data
		eval sas_dir="$"${data}"_dir"
		
        CUDA_VISIBLE_DEVICES=$gpu \
			python eend/bin/infer_eend.py -c $infer_conf $sas_dir $base_model $sas_out
		
		# prepare scp file
		find $sas_out | grep '\.h5' | awk -F '/' '{print $NF}' | cut -d "." -f 1 > $sas_out/sas.list
		find $sas_out | grep '\.h5' > $sas_out/sas.path
		paste $sas_out/sas.list $sas_out/sas.path > $sas_out/sas.scp
		
	done	
fi

train_sas=$sas_out_dir/train/sas.scp
dev_sas=$sas_out_dir/dev/sas.scp
test_sas=$sas_out_dir/test/sas.scp

# Training
if [ $stage -le 1 ]; then
    echo "Start training"
CUDA_VISIBLE_DEVICES=6 \
	  python eend/bin/train_correct.py -c $train_conf $train_dir $dev_dir $train_sas $dev_sas $model_dir
fi

# Model averaging
if [ $stage -le 2 ]; then
    echo "Start model averaging"
    ifiles=`eval echo $model_dir/transformer{1..2}.th`
    python eend/bin/model_averaging.py $test_model $ifiles
fi

# Inferring
if [ $stage -le 3 ]; then
    echo "Start inferring"
    CUDA_VISIBLE_DEVICES=$gpu \
		python eend/bin/infer_correct.py -c $infer_conf $test_dir $test_sas $test_model $infer_out_dir
fi

# Scoring
if [ $stage -le 4 ]; then
    echo "Start scoring"
	mkdir -p temp
    mkdir -p $work
    mkdir -p $scoring_dir
	med=11
	find $infer_out_dir -iname "*.h5" > $work/file_list
	for th in 0.5 0.6 0.7; do
		echo "th=$th"
		python eend/bin/make_rttm.py --median=$med --threshold=$th \
			--frame_shift=80 --subsampling=10 --sampling_rate=8000 \
			$work/file_list $scoring_dir/hyp_${th}_$med.rttm
		utils/analysis_diarization.sh --collar 0.25 \
			$test_dir/rttm $scoring_dir/hyp_${th}_$med.rttm
	done >  $scoring_dir/score
	echo "scoreing done"
	cat $scoring_dir/score
fi
