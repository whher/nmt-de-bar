#! usr/bin/bash

# System: Transfer
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

# joint bpe already learned

# apply bpe

    python -m apply_bpe -c model_transfer_cv/bpe.codes --dropout 0.1 --vocabulary model_transfer_cv/bpe.vocab.bar --vocabulary-threshold 50 < corpus/parallel/bar-de/$source_train > model_transfer_cv/$cv_fold/child.train.BPE.bar

    python -m apply_bpe -c model_transfer_cv/bpe.codes --dropout 0.1 --vocabulary model_transfer_cv/bpe.vocab.de --vocabulary-threshold 50 < corpus/parallel/bar-de/$target_train > model_transfer_cv/$cv_fold/child.train.BPE.de

    python -m apply_bpe -c model_transfer_cv/bpe.codes --vocabulary model_transfer_cv/bpe.vocab.bar --vocabulary-threshold 50 < corpus/parallel/bar-de/$source_test > model_transfer_cv/$cv_fold/child.test.BPE.bar

    python -m apply_bpe -c model_transfer_cv/bpe.codes --vocabulary model_transfer_cv/bpe.vocab.de --vocabulary-threshold 50 < corpus/parallel/bar-de/$target_test > model_transfer_cv/$cv_fold/child.test.BPE.de

# prepare data for child

    python -m sockeye.prepare_data -s model_transfer_cv/$cv_fold/child.train.BPE.bar -t model_transfer_cv/$cv_fold/child.train.BPE.de -o model_transfer_cv/$cv_fold/train_data_child_$cv_fold --source-vocab model_transfer_cv/train_data_fr-de_parent/vocab.src.0.json --target-vocab model_transfer_cv/train_data_fr-de_parent/vocab.trg.0.json --shared-vocab

# continue training

    python -m sockeye.train --config model_transfer_cv/fr-de-parent-model/args.yaml -d model_transfer_cv/$cv_fold/train_data_child_$cv_fold -vs model_transfer_cv/$cv_fold/child.test.BPE.bar -vt model_transfer_cv/$cv_fold/child.test.BPE.de --params model_transfer_cv/fr-de-parent-model/params.best -o model_transfer_cv/$cv_fold/bar-de-child-model

done


