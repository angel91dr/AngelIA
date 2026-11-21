
import json
import random
from pathlib import Path
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.brain import AngelIABrain


def train():
    # Reproductibilité des premiers tests
    random.seed(42)
    torch.manual_seed(42)

    # Charger le vocabulaire
    with open(ROOT / "data" / "vocab.json", "r", encoding="utf-8") as file:
        vocab = json.load(file)

    # Charger les conversations préparées
    with open(ROOT / "data" / "training_data.json", "r", encoding="utf-8") as file:
        conversations = json.load(file)

    if not conversations:
        raise ValueError("Aucune conversation à apprendre !")

    vocab_size = len(vocab)
    bos_id = vocab["<BOS>"]
    eos_id = vocab["<EOS>"]

    # Créer le cerveau
    brain = AngelIABrain(vocab_size=vocab_size)

    optimizer = torch.optim.Adam(brain.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    epochs = 200

    print("🧠 Début de l'apprentissage d'AngelIA...")
    print(f"Conversations : {len(conversations)}")
    print(f"Vocabulaire : {vocab_size} tokens")
    print(f"Époques : {epochs}")
    print("------------------------------------")

    brain.train()

    for epoch in range(epochs):
        random.shuffle(conversations)
        total_loss = 0.0

        for pair in conversations:
            user_tokens = pair["user"]
            assistant_tokens = pair["assistant"]

            # Séquence : début + message utilisateur + réponse + fin
            sequence = [bos_id] + user_tokens + assistant_tokens + [eos_id]

            inputs = torch.tensor(sequence[:-1], dtype=torch.long)
            targets = torch.tensor(sequence[1:], dtype=torch.long)

            # Ne calculer la perte que sur la réponse d'AngelIA
            first_answer_position = len(user_tokens)

            logits, _ = brain(inputs.unsqueeze(0))

            answer_logits = logits[0, first_answer_position:, :]
            answer_targets = targets[first_answer_position:]

            loss = criterion(answer_logits, answer_targets)

            optimizer.zero_grad()
            loss.backward()

            # Éviter des gradients trop importants
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(conversations)

        if (epoch + 1) % 20 == 0:
            print(
                f"Époque {epoch + 1}/{epochs} "
                f"| Erreur : {average_loss:.4f}"
            )

    # Sauvegarder le cerveau entraîné
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    save_path = models_dir / "angelia_brain.pth"

    torch.save({
        "model_state_dict": brain.state_dict(),
        "vocab_size": vocab_size,
        "embedding_size": 64,
        "hidden_size": 128
    }, save_path)

    print("------------------------------------")
    print("🎉 Premier entraînement terminé !")
    print(f"Cerveau sauvegardé : {save_path}")


if __name__ == "__main__":
    train()
