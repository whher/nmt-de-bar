#! usr/bin/bash

# System: transfer
# Direction: fr-de and de-fr (parents)

export PYTHONPATH=$(pwd)/subword-nmt:$PYTHONPATH
pip install torch==1.11.0
pip install sacrebleu==1.4.14

# skip cross validation

# learn joint bpe (parent and child)

python -m learn_joint_bpe_and_vocab --input corpus/parallel/bar-de/bar1-5 corpus/parallel/fr-de/train.de corpus/parallel/fr-de/train.fr -s 30000 -o model_transfer_cv/bpe.codes --write-vocabulary model_transfer_cv/bpe.vocab.bar model_transfer_cv/bpe.vocab.de model_transfer_cv/bpe.vocab.fr

# apply bpe

python -m apply_bpe -c model_transfer_cv/bpe.codes --dropout 0.1 --vocabulary model_transfer_cv/bpe.vocab.fr --vocabulary-threshold 50 < corpus/parallel/fr-de/train.fr > model_transfer_cv/parent.train.BPE.fr

python -m apply_bpe -c model_transfer_cv/bpe.codes --dropout 0.1 --vocabulary model_transfer_cv/bpe.vocab.de --vocabulary-threshold 50 < corpus/parallel/fr-de/train.de > model_transfer_cv/parent.train.BPE.de

python -m apply_bpe -c model_transfer_cv/bpe.codes --vocabulary model_transfer_cv/bpe.vocab.fr --vocabulary-threshold 50 < corpus/parallel/fr-de/test.fr > model_transfer_cv/parent.test.BPE.fr

python -m apply_bpe -c model_transfer_cv/bpe.codes --vocabulary model_transfer_cv/bpe.vocab.de --vocabulary-threshold 50 < corpus/parallel/fr-de/test.de > model_transfer_cv/parent.test.BPE.de

# serialize data

python -m sockeye.prepare_data -s model_transfer_cv/parent.train.BPE.fr -t model_transfer_cv/parent.train.BPE.de -o model_transfer_cv/train_data_fr-de_parent --shared-vocab

# start training

python -m sockeye.train -d model_transfer_cv/train_data_fr-de_parent -vs model_transfer_cv/parent.test.BPE.fr -vt model_transfer_cv/parent.test.BPE.de -o model_transfer_cv/fr-de-parent-model --encoder transformer --decoder transformer --num-layers 6 --num-embed 512 --transformer-model-size 512 --transformer-attention-heads 8 --transformer-feed-forward-num-hidden 2048 --max-seq-len 90 --decode-and-evaluate 500 --max-num-checkpoint-not-improved 6 --shared-vocab

# System: Transfer
# Direction: de-fr (parent)

# serialize data

python -m sockeye.prepare_data -s model_transfer_cv/parent.train.BPE.de -t model_transfer_cv/parent.train.BPE.fr -o model_transfer_cv/train_data_de-fr_parent --shared-vocab

# train parent model

python -m sockeye.train -d model_transfer_cv/train_data_de-fr_parent -vs model_transfer_cv/parent.test.BPE.de -vt model_transfer_cv/parent.test.BPE.fr -o model_transfer_cv/de-fr-parent-model --encoder transformer --decoder transformer --num-layers 6 --num-embed 512 --transformer-model-size 512 --transformer-attention-heads 8 --transformer-feed-forward-num-hidden 2048 --max-seq-len 90 --decode-and-evaluate 500 --max-num-checkpoint-not-improved 6 --shared-vocab


