#!/bin/bash

ulimit -S -n 4096

export PYTHONPATH=`pwd`:$PYTHONPATH

conf_dir=conf/large
exp_dir=exp/diarization/sa-eend

# Training
model_dir=$exp_dir/models
train_conf=$conf_dir/train.yaml
train_dir=/path/EEND/egs/mini_librispeech/v1/data/simu/data/train_clean_100_ns2_beta2_10000
dev_dir=/path/EEND/egs/mini_librispeech/v1/data/simu/data/dev_clean_ns2_beta2_500

affix=simu

# Inference
test_model=$model_dir/avg.th
infer_conf=$conf_dir/infer.yaml
test_dir=/path/EEND/egs/mini_librispeech/v1/data/simu/data/test_clean_ns2_beta2_500
infer_out_dir=$exp_dir/infer/$affix

# Scoring
work=$infer_out_dir/.work
scoring_dir=$exp_dir/score/$affix

stage=4
gpu=6

# Training
if [ $stage -le 1 ]; then
    echo "Start training"
CUDA_VISIBLE_DEVICES=$gpu \
    python eend/bin/train_eend.py -c $train_conf $train_dir $dev_dir $model_dir
fi

# Model averaging
if [ $stage -le 2 ]; then
    echo "Start model averaging"
    ifiles=`eval echo $model_dir/transformer{1..2}.th`
    python eend/bin/model_averaging.py $test_model $ifiles
fi

# Inference
if [ $stage -le 3 ]; then
    echo "Start inferring"
CUDA_VISIBLE_DEVICES=$gpu \
    python eend/bin/infer_eend.py -c $infer_conf $test_dir $test_model $infer_out_dir
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
