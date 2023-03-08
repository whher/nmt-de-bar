#! usr/bin/bash

# System: Base
# Direction: bar-de

export PYTHONPATH=$(pwd)/subword-nmt:$PYTHONPATH
pip install torch==1.11.0
pip install sacrebleu==1.4.14

# wrapper for cross validation

train_it=("1" "2" "3" "4" "5")
train_it2=("4" "5" "1" "2" "3")
test_it=("5" "1" "2" "3" "4")

for ((i=0;i<=4;i++))
do
    #echo ${train_it[i]}
    #echo ${train_it2[i]}
    #echo ${test_it[i]}
    
    cv_fold="bar-de${train_it[i]}-${train_it2[i]}"
    source_train="bar${train_it[i]}-${train_it2[i]}"
    target_train="de${train_it[i]}-${train_it2[i]}"
    source_test="bar${test_it[i]}"
    target_test="de${test_it[i]}"

    echo $cv_fold

# learn bpe

    python -m learn_joint_bpe_and_vocab --input corpus/parallel/bar-de/$source_train corpus/parallel/bar-de/$target_train -s 30000 -o model_baseline_cv/$cv_fold/bpe.codes --write-vocabulary model_baseline_cv/$cv_fold/bpe.vocab.bar model_baseline_cv/$cv_fold/bpe.vocab.de

# apply bpe

    python -m apply_bpe -c model_baseline_cv/$cv_fold/bpe.codes --dropout 0.1 --vocabulary model_baseline_cv/$cv_fold/bpe.vocab.bar --vocabulary-threshold 50 < corpus/parallel/bar-de/$source_train > model_baseline_cv/$cv_fold/$source_train.BPE

    python -m apply_bpe -c model_baseline_cv/$cv_fold/bpe.codes --dropout 0.1 --vocabulary model_baseline_cv/$cv_fold/bpe.vocab.de --vocabulary-threshold 50 < corpus/parallel/bar-de/$target_train > model_baseline_cv/$cv_fold/$target_train.BPE

    python -m apply_bpe -c model_baseline_cv/$cv_fold/bpe.codes --vocabulary model_baseline_cv/$cv_fold/bpe.vocab.bar --vocabulary-threshold 50 < corpus/parallel/bar-de/$source_test > model_baseline_cv/$cv_fold/$source_test.BPE

    python -m apply_bpe -c model_baseline_cv/$cv_fold/bpe.codes --vocabulary model_baseline_cv/$cv_fold/bpe.vocab.de --vocabulary-threshold 50 < corpus/parallel/bar-de/$target_test > model_baseline_cv/$cv_fold/$target_test.BPE

# serialize data

    python -m sockeye.prepare_data -s model_baseline_cv/$cv_fold/$source_train.BPE -t model_baseline_cv/$cv_fold/$target_train.BPE -o model_baseline_cv/$cv_fold/train_data_$cv_fold --shared-vocab

# start training

    python -m sockeye.train -d model_baseline_cv/$cv_fold/train_data_$cv_fold -vs model_baseline_cv/$cv_fold/$source_test.BPE -vt model_baseline_cv/$cv_fold/$target_test.BPE -o model_baseline_cv/$cv_fold/$cv_fold --encoder transformer --decoder transformer --num-layers 6 --num-embed 512 --transformer-model-size 512 --transformer-attention-heads 8 --transformer-feed-forward-num-hidden 2048 --max-seq-len 90 --decode-and-evaluate 500 --max-num-checkpoint-not-improved 3 --shared-vocab

done


