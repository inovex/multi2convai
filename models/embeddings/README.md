# Embeddings

Some of our models our models require pretrained embeddings. This page explains how to download them. Take a look into the readme of our models or in the use case readmes ([corona](../corona/README.md), [logistics](../logistics/README.md), [quality](../quality/README.md)) to know on which pretrained embeddings they rely.

For details how to download our multi2convai models check out this [README](../README.md/#download-models).

## Download fastText

Check out https://fasttext.cc/ to learn about the background of fastText embeddings. FastText offers a broad range of pretrained word embeddings distributed under [Creative Commons Attribution-Share-Alike License](https://creativecommons.org/licenses/by-sa/3.0/).

In order to download embeddings from fastText (e.g. for English), [do it via their fasttext python package](https://fasttext.cc/docs/en/crawl-vectors.html) or execute the following commands yourself:

````terminal
mkdir fasttext/en
curl https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec --output fasttext/en/wiki.en.vec

ls fasttext/en
>>> wiki.en.vec

````

While downloading the fasttext embeddings you'll notice that they're quite large (some are bigger than 6 GB). To keep our pipelines quick and responsive we will serialize the embeddings and restrict the vocab to the top 200k words. You can use `serialize_fasttext.py` in our `scripts` section to do the same.

````terminal
python serialize_fasttext.py --raw-path fasttext/en/wiki.en.vec --vocab-path fasttext/en/wiki.200k.en.vocab --embeddings-path fasttext/en/wiki.200k.en.embed -n 200000

ls fasttext/en
>>> wiki.200k.en.embed    wiki.200k.en.vocab    wiki.en.vec

````

Please refer to the fastText documentation if you face any issues: https://fasttext.cc/

## Download pretrained LM

We used pretrained language models (e.g. BERT or RoBERTa) for finetuning and as contextual word embeddings for other models. Pretrained language models are available on the [huggingface model hub](https://huggingface.co/models). There are several ways to download models from the model hub. Typically, you would check the `</> Use in Transformers` button on a specific model page to learn how to use it.

Let's say we want to download: https://huggingface.co/bert-base-german-dbmdz-uncased. In that case `</> Use in Transformers` tells us that there are two ways of using the model: via the `transformers` library or `git-lfs`.

### Download and save via transformers-library

Let's say we want to download: https://huggingface.co/bert-base-german-dbmdz-uncased. In that case `</> Use in Transformers` tells us that there are two ways of using the model:

````python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-german-dbmdz-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-german-dbmdz-uncased")

````

In order to have them available locally, you can either pass the parameter `cache_dir = "transformers/bert-base-german-dbmdz-uncased"` to both `from_pretrained` calls. Not that this will result in filenames including hashes that are hard to read. Or you can use the `save_pretrained` method to store the tokenizer and model:

````python
import os

tokenizer.save_pretrained("transformers/bert-base-german-dbmdz-uncased")
model.save_pretrained("transformers/bert-base-german-dbmdz-uncased")

os.listdir("transformers")
>>> ["bert-base-german-dbmdz-uncased", "README.md"]

os.listdir("transformers/bert-base-german-dbmdz-uncased")
>>> ["config.json", "pytorch_model.bin", "special_tokens_map.json", "tokenizer_config.json", "vocab.txt"]
````

### Download with git-lfs

Please refer to the huggingface hub documentation in case you face problems: https://huggingface.co/docs/hub/adding-a-model#uploading-your-files

Install git lfs if not done yet: `git lfs install` and then run:

````bash
cd embeddings/transformers/
git clone https://huggingface.co/bert-base-german-dbmdz-uncased

ls embeddings/transformers
>>> bert-base-german-dbmdz-uncased	README.md

ls embeddings/transformers/bert-base-german-dbmdz-uncased
>>> config.json, pytorch_model.bin, README.md, special_tokens_map.json, tokenizer_config.json, vocab.txt

````
