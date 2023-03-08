#! usr/bin/bash

# Create augmented text

export PYTHONPATH=$(pwd)/subword-nmt:$PYTHONPATH
pip install torch==1.11.0
pip install sacrebleu==1.4.14

# back-translate German target for Bavarian source text

while read -r line; do echo "$line" | python -m apply_bpe -c bpe.codes --vocabulary bpe.vocab.de --vocabulary-threshold 50 | python -m sockeye.translate -m model_baseline_cv/de-bar1-4/de-bar1-4 2>/dev/null | sed -r 's/@@( |$)//g' >> corpus/monolingual/BT-bar-translated.txt; done < corpus/monolingual/raw/BT-de-sampled.txt

# back-translate Bavarian target for German source text

while read -r line; do echo "$line" | python -m apply_bpe -c bpe.codes --vocabulary bpe.vocab.bar --vocabulary-threshold 50 | python -m sockeye.translate -m model_baseline_cv/bar-de1-4/bar-de1-4 2>/dev/null | sed -r 's/@@( |$)//g' >> corpus/monolingual/BT-de-translated.txt; done < corpus/monolingual/raw/BT-bar-sampled.txt
