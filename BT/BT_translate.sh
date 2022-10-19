# Bash commands to back-translate monolingual target data

export PYTHONPATH=$(pwd)/subword-nmt:$PYTHONPATH
pip install torch==1.11.0

# Translate German --> Bavarian
while read -r line; do echo "$line" | python -m apply_bpe -c bpe.codes --vocabulary bpe.vocab.de --vocabulary-threshold 50 | python -m sockeye.translate -m de-bar-base-model 2>/dev/null | sed -r 's/@@( |$)//g' >> bar-de/mono/BT-bar-translated.txt; done < bar-de/mono/BT-de-sampled.txt

# Translate Bavarian --> German
while read -r line; do echo "$line" | python -m apply_bpe -c bpe.codes --vocabulary bpe.vocab.bar --vocabulary-threshold 50 | python -m sockeye.translate -m bar-de-base-model 2>/dev/null | sed -r 's/@@( |$)//g' >> de-bar/mono/BT-de-translated.txt; done < de-bar/mono/BT-bar-sampled.txt
