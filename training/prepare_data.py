
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tokenizer import AngelIATokenizer


def prepare_data():
    conversation_file = ROOT / "data" / "conversations.txt"

    with open(conversation_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    pairs = []
    current_user = None

    for line in lines:
        line = line.strip()

        if line.startswith("Utilisateur:"):
            current_user = line.replace("Utilisateur:", "", 1).strip()

        elif line.startswith("AngelIA:") and current_user is not None:
            assistant = line.replace("AngelIA:", "", 1).strip()

            pairs.append({
                "user": current_user,
                "assistant": assistant
            })

            current_user = None

    # Charger le tokenizer et son vocabulaire
    tokenizer = AngelIATokenizer()
    vocab_path = ROOT / "data" / "vocab.json"

    if vocab_path.exists():
        tokenizer.load(vocab_path)

    # Ajouter les marqueurs des interlocuteurs
    if "<USER>" not in tokenizer.vocab:
        tokenizer.vocab["<USER>"] = len(tokenizer.vocab)

    if "<ANGELIA>" not in tokenizer.vocab:
        tokenizer.vocab["<ANGELIA>"] = len(tokenizer.vocab)

    # Construire le vocabulaire
    for pair in pairs:
        tokenizer.build_vocab([
            pair["user"],
            pair["assistant"]
        ])

    tokenizer.save(vocab_path)

    # Préparer les données
    training_data = []

    for pair in pairs:
        training_data.append({
            "user": tokenizer.encode(pair["user"]),
            "assistant": tokenizer.encode(pair["assistant"])
        })

    output_file = ROOT / "data" / "training_data.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            training_data,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("🧠 Données d'AngelIA préparées !")
    print(f"Conversations trouvées : {len(pairs)}")
    print(f"Vocabulaire : {len(tokenizer.vocab)} tokens")
    print("Fichier créé : data/training_data.json")


if __name__ == "__main__":
    prepare_data()
