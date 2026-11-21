
import sys
from pathlib import Path

# Ajouter le dossier principal au chemin Python
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tokenizer import AngelIATokenizer


# Premières phrases d'apprentissage d'AngelIA
sentences = [
    "Bonjour !",
    "Salut, comment vas-tu ?",
    "Je vais très bien, merci.",
    "Comment tu t'appelles ?",
    "Je m'appelle AngelIA.",
    "Quel âge as-tu ?",
    "Je suis une intelligence artificielle.",
    "J'aime discuter avec les gens.",
    "Le chocolat est délicieux.",
    "Les fraises sont délicieuses.",
    "Que fais-tu aujourd'hui ?",
    "Je suis heureuse de te parler.",
    "À bientôt !",
    "Merci beaucoup.",
    "Je ne suis pas d'accord.",
]


def train_tokenizer():
    tokenizer = AngelIATokenizer()

    tokenizer.build_vocab(sentences)

    tokenizer.save(ROOT / "data" / "vocab.json")

    print("🧠 AngelIA a construit son vocabulaire !")
    print(f"Nombre de tokens : {len(tokenizer.vocab)}")
    print("Vocabulaire sauvegardé dans data/vocab.json")


if __name__ == "__main__":
    train_tokenizer()
