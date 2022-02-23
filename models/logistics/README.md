# Logistics use case

## Available models

| model name | files required to run pipeline | model type | embeddings |
|------------|--------------------------------|------------|------------|
| [multi2convai-logistics-de-logreg-ft](https://huggingface.co/inovex/multi2convai-logistics-de-logreg-ft) | `model.pth`, `label_dict.json` | logistic regression | fasttext |
| [multi2convai-logistics-en-logreg-ft](https://huggingface.co/inovex/multi2convai-logistics-en-logreg-ft) | `model.pth`, `label_dict.json` | logistic regression | fasttext |


Our models share the following naming schema: `multi2convai-<use_case>-<language>-<model_type>-<embeddings>`

## Naming Schema

Our models share the following naming schema: `multi2convai-<use_case>-<language>-<model_type>-<embeddings>`

- `use_case`: name of the use case the model belongs to (either "corona", "logistics" or "quality")
- `language`: two char identifier of the supported language (e.g. "de" for German, "en" for English, "ml" for multilingual models)
- `model_type`: abbreviated form of the model type / architecture (e.g. "logreg" for logistic regression or "bert" for a finetuned Bert model)
- `embeddings`: optional parameter indicating the embeddings used (e.g. "ft" for fasttext embeddings or "bert" when using pretrained bert embeddings as input for another model)

Further details about our use cases can be found in this blog on our project website: [use cases EN](https://multi2conv.ai/blog/en/use-cases), [use cases DE](https://multi2conv.ai/blog/de/use-cases).
