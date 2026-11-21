
from pathlib import Path
from tokenizer import AngelIATokenizer

ROOT = Path(__file__).resolve().parents[1]

tokenizer = AngelIATokenizer()
tokenizer.load(ROOT / "data" / "vocab.json")

phrase = "Bonjour AngelIA !"

tokens = tokenizer.tokenize(phrase)
numbers = tokenizer.encode(phrase)
reconstructed = tokenizer.decode(numbers)

print("🧠 TEST DU TOKENIZER ANGELIA")
print("----------------------------")
print("Phrase originale :", phrase)
print("Mots découpés :", tokens)
print("Nombres obtenus :", numbers)
print("Texte reconstruit :", reconstructed)
