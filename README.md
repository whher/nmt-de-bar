# German-Bavarian NMT
This is a German-Bavarian translation system trained using the [sockeye toolkit](https://github.com/awslabs/sockeye).

## Data Source
Data source for our experiment comes from OPUS. Below you can find the links.

| Type | Lang. | Name | Sent. Count | Link |
| -----|-------|------|-------------|------|
| Parallel | bar-de | WikiMatrix | 86.4K | http://opus.nlpl.eu/WikiMatrix-v1.php |
| Parallel | bar-de | WikiMedia  | 1.9K  | http://opus.nlpl.eu/wikimedia-v20210402.php |
| Parallel | bar-de | XLEnt      | 11.4K | http://opus.nlpl.eu/XLEnt-v1.1.php |
| Parallel | bar-de | Tatoeba    | 61    | https://opus.nlpl.eu/Tatoeba-v2021-07-22.php |
| Parallel | fr-de  | Tatoeba    | 112K  | http://opus.nlpl.eu/Tatoeba-v2022-03-03.php |
| Parallel | fr-de  | WikiMedia  | 72K   | http://opus.nlpl.eu/wikimedia-v20210402.php |
| Monolingual | bar | Wiki | 295K | https://opus.nlpl.eu/Tatoeba-v2020-05-31.php |
| Monolingual | de  | Wiki | 258K | https://opus.nlpl.eu/Tatoeba-v2020-05-31.php |

## Results

| Lang. | Model | BLEU | chrF | TER |
|-------|-------|------|------|-----|
| bar-de | Baseline | 66.0 | 78.1 | 32.7 |
| bar-de | Back-translated | 73.4 | 82.5 | 25.0 |
| bar-de | Transferred | 53.9 | 70.5 | 41.9 |
| de-bar | Baseline | 61.2 | 74.4 | 36.2 |
| de-bar | Back-translated | 63.4 | 76.3 | 31.9 |
| de-bar | Transferred | 48.2 | 63.9 | 44.4 |
