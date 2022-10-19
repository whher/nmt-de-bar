# This is the bash commands for training the baseline model
# Translation direction is German --> Bavarian

# Lines 2 and 3 have to be executed upon every login
export PYTHONPATH=$(pwd)/subword-nmt:$PYTHONPATH
pip install torch==1.11.0

# Create BPE joint vocabulary (Can be skipped if BPE codes are already available)
python -m learn_joint_bpe_and_vocab --input de-bar/parallel/train.src de-bar/parallel/train.tgt \
      -s 30000 \
      -o bpe.codes \
      --write-vocabulary bpe.vocab.de bpe.vocab.bar

# Apply BPE codes to training corpus with dropout rate 0.1
python -m apply_bpe -c bpe.codes --dropout 0.1 --vocabulary bpe.vocab.de --vocabulary-threshold 50 < de-bar/parallel/train.src > de-bar/parallel/train.BPE.src
python -m apply_bpe -c bpe.codes --dropout 0.1 --vocabulary bpe.vocab.bar --vocabulary-threshold 50 < de-bar/parallel/train.tgt > de-bar/parallel/train.BPE.tgt

# Apply BPE codes to validation corpus
python -m apply_bpe -c bpe.codes --vocabulary bpe.vocab.de --vocabulary-threshold 50 < de-bar/parallel/test.src > de-bar/parallel/test.BPE.src
python -m apply_bpe -c bpe.codes --vocabulary bpe.vocab.bar --vocabulary-threshold 50 < de-bar/parallel/test.tgt > de-bar/parallel/test.BPE.tgt

# Serialize training datasets
python -m sockeye.prepare_data \
      -s de-bar/parallel/train.BPE.src \
      -t de-bar/parallel/train.BPE.tgt \
      -o de-bar/parallel/train_data \
			--shared-vocab

# Begin training
python -m sockeye.train -d de-bar/parallel/train_data
      -vs de-bar/parallel/test.BPE.src
      -vt de-bar/parallel/test.BPE.tgt
      -o de-bar-base-model 
      --encoder transformer
      --decoder transformer 
      --num-layers 6 
      --num-embed 256 
      --transformer-model-size 256 
      --transformer-attention-heads 8 
      --transformer-feed-forward-num-hidden 1024 
      --max-seq-len 60 
      --decode-and-evaluate 500 
      --max-num-checkpoint-not-improved 1 
      --shared-vocab
