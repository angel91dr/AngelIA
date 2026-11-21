
import json
import re
from pathlib import Path


class AngelIATokenizer:
    def __init__(self):
        self.vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3
        }

    def tokenize(self, text):
        return re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)

    def build_vocab(self, texts):
        for text in texts:
            for token in self.tokenize(text):
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]

        
    
    def decode(self, token_ids):
        reverse_vocab = {
            value: key for key, value in self.vocab.items()
        }

        tokens = [
            reverse_vocab.get(token_id, "<UNK>")
            for token_id in token_ids
        ]

        text = ""

        no_space_before = {".", ",", ":", ";", "%", ")", "]", "}"}
        opening = {"(", "[", "{"}

        for token in tokens:
            if not text:
                text = token
            elif token in no_space_before or token == "'":
                text += token
            elif text.endswith("'") or text[-1] in opening:
                text += token
            else:
                text += " " + token

        return text



    def save(self, path="data/vocab.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.vocab, file, ensure_ascii=False, indent=4)

    def load(self, path="data/vocab.json"):
        with open(path, "r", encoding="utf-8") as file:
            self.vocab = json.load(file)
