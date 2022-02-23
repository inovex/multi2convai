# Models

Models developed in the Multi2ConvAI project are available via the huggingface model hub. You can find an overview of available models here: https://huggingface.co/inovex (search for models with the prefix `multi2convai-xxx`).

## Naming Schema

Our models share the following naming schema: `multi2convai-<use_case>-<language>-<model_type>-<embeddings>`

- `use_case`: name of the use case the model belongs to (either "corona", "logistics" or "quality")
- `language`: two char identifier of the supported language (e.g. "de" for German, "en" for English, "ml" for multilingual models)
- `model_type`: abbreviated form of the model type / architecture (e.g. "logreg" for logistic regression or "bert" for a finetuned Bert model)
- `embeddings`: optional parameter indicating the embeddings used (e.g. "ft" for fasttext embeddings or "bert" when using pretrained bert embeddings as input for another model)

Further details about our use cases can be found in this blog on our project website: [use cases EN](https://multi2conv.ai/blog/en/use-cases), [use cases DE](https://multi2conv.ai/blog/de/use-cases).

## Download Models

Models can either be downloaded using `git-lfs` or the `huggingface-hub` package. Note that `git-lfs` is the preferred option as the cached download using `huggingface-hub` will lead to unreadable filenames. Each use case subfolder contains a list of models available in that use case. For each model type we specify the repository structure (e.g. logistic regression models will consist of a `model.pth` and a `label_dict.json`). We'll use `multi2convai-corona-de-logreg-ft` to showcase the download process.


### Download with git-lfs (preferred option)

Please refer to the huggingface hub documentation in case you face problems: https://huggingface.co/docs/hub/adding-a-model#uploading-your-files

Install git lfs if not done yet: `git lfs install` and then run:

````terminal
cd corona/
git clone https://huggingface.co/inovex/multi2convai-corona-de-logreg-ft

ls corona/
>>> multi2convai-corona-de-logreg-ft	README.md

ls corona/multi2convai-corona-de-logreg-ft
>>> README.md    label_dict.json	model.pth

````

### Download with huggingface-hub

Please refer to the huggingface hub documentation in case you face problems: https://huggingface.co/docs/hub/adding-a-library#download-files-from-the-hub. Note that this might require logging in to huggingface and can lead to unreadable filenames.

In order to download `multi2convai-corona-de-logreg-ft` in python, run:
````python
from huggingface_hub import hf_hub_download
import os

hf_hub_download(repo_id="inovex/multi2convai-corona-de-logreg-ft", filename="label_dict.json", cache_dir="corona/multi2convai-corona-de-logreg-ft2")
hf_hub_download(repo_id="inovex/multi2convai-corona-de-logreg-ft", filename="label_dict.json", cache_dir="corona/multi2convai-corona-de-logreg-ft")

os.listdir("corona")
>>> ["multi2convai-corona-de-logreg-ft", "README.md"]

os.listdir("corona/multi2convai-corona-de-logreg-ft")
>>> ['<hash1>.<hash2>.lock', '<hash1>.<hash2>', '<hash1>.<hash2>.json']
````
