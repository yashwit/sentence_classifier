# Sentence Transformers & Multi-Task Learning

This is a simple Python program using PyTorch to classify short sentences into **Informative**, **Question**, or **Statement**, **Exclamation** classifications and **Positive**, **Negative**, or **Neutral** sentiments based on pre-labeled training data.

## Example Training Sentences

| Sentence| Classification | Sentiment |
| -- | -- |-- |
| I love LLMs!|Exclamation|Positive|
| Is this a sentence classifier?|Question|Neutral|
| LLMs are typically built on a type of neural network called a Transformer.|Informative|Neutral|
| The weather outside is dreary|Statement|Negative|


## Requirements

* Python 3.x
* PyTorch
* Hugging Face transformers

Install requirements.txt via pip:

```bash
pip install -r requirements.txt
```

## Usage

Run the script using:

```bash
python sentences.py
```

You can modify the `main()` function to test with your own sentences.
