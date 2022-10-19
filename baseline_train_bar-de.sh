# Lines 2 and 3 have to be executed upon every login
export PYTHONPATH=$(pwd)/subword-nmt:$PYTHONPATH
pip install torch==1.11.0

# Create BPE joint vocabulary
python -m learn_joint_bpe_and_vocab --input bar-de/parallel/train.src bar-de/parallel/train.tgt \
      -s 30000 \
      -o bpe.codes \
      --write-vocabulary bpe.vocab.bar bpe.vocab.de

# Apply BPE codes to training corpus with dropout rate 0.1
python -m apply_bpe -c bpe.codes --dropout 0.1 --vocabulary bpe.vocab.bar --vocabulary-threshold 50 < bar-de/parallel/train.src > bar-de/parallel/train.BPE.src
python -m apply_bpe -c bpe.codes --dropout 0.1 --vocabulary bpe.vocab.de --vocabulary-threshold 50 < bar-de/parallel/train.tgt > bar-de/parallel/train.BPE.tgt

# Apply BPE codes to validation corpus
python -m apply_bpe -c bpe.codes --vocabulary bpe.vocab.bar --vocabulary-threshold 50 < bar-de/parallel/test.src > bar-de/parallel/test.BPE.src
python -m apply_bpe -c bpe.codes --vocabulary bpe.vocab.de --vocabulary-threshold 50 < bar-de/parallel/test.tgt > bar-de/parallel/test.BPE.tgt

# Serialize training datasets
python -m sockeye.prepare_data \
      -s bar-de/parallel/train.BPE.src \
      -t bar-de/parallel/train.BPE.tgt \
      -o bar-de/parallel/train_data \
			--shared-vocab

# Begin training
python -m sockeye.train -d bar-de/parallel/train_data
      -vs bar-de/parallel/test.BPE.src
      -vt bar-de/parallel/test.BPE.tgt
      -o bar-de-base-model 
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
