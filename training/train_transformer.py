
import json
import random
from pathlib import Path
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformer.model import AngelIATransformer


def train():
    random.seed(42)
    torch.manual_seed(42)

    # Charger le vocabulaire
    with open(ROOT / "data" / "vocab.json", "r", encoding="utf-8") as file:
        vocab = json.load(file)

    # Charger les conversations
    with open(ROOT / "data" / "training_data.json", "r", encoding="utf-8") as file:
        conversations = json.load(file)

    if not conversations:
        raise ValueError("Aucune conversation disponible !")

    vocab_size = len(vocab)

    bos_id = vocab["<BOS>"]
    eos_id = vocab["<EOS>"]
    user_id = vocab["<USER>"]
    angelia_id = vocab["<ANGELIA>"]

    # Créer le Transformer
    brain = AngelIATransformer(
        vocab_size=vocab_size,
        embedding_size=64,
        num_heads=4,
        num_layers=2,
        max_sequence_length=256,
        dropout=0.1
    )

    optimizer = torch.optim.AdamW(
        brain.parameters(),
        lr=0.0003
    )

    criterion = nn.CrossEntropyLoss()

    epochs = 150

    # Préparer les exemples indépendamment
    training_examples = []

    for pair in conversations:

        context = [
            bos_id,
            user_id
        ] + pair["user"]

        answer = [
            angelia_id
        ] + pair["assistant"] + [
            eos_id
        ]

        sequence = context + answer

        answer_position = len(context) - 1

        if len(sequence) > brain.max_sequence_length:
            raise ValueError(
                "Une conversation dépasse la longueur maximale."
            )

        training_examples.append({
            "sequence": sequence,
            "answer_position": answer_position
        })

    print("🧠 Début de l'entraînement d'AngelIA...")
    print(f"Conversations : {len(conversations)}")
    print(f"Exemples : {len(training_examples)}")
    print(f"Vocabulaire : {vocab_size} tokens")
    print(f"Époques : {epochs}")
    print("------------------------------------")

    # Entraînement
    for epoch in range(epochs):

        brain.train()

        shuffled_examples = training_examples.copy()
        random.shuffle(shuffled_examples)

        total_loss = 0.0

        for example in shuffled_examples:

            sequence = example["sequence"]
            answer_position = example["answer_position"]

            inputs = torch.tensor(
                [sequence[:-1]],
                dtype=torch.long
            )

            targets = torch.tensor(
                [sequence[1:]],
                dtype=torch.long
            )

            logits = brain(inputs)

            answer_logits = logits[
                0,
                answer_position:,
                :
            ]

            answer_targets = targets[
                0,
                answer_position:
            ]

            loss = criterion(
                answer_logits,
                answer_targets
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                brain.parameters(),
                1.0
            )

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(training_examples)

        if (epoch + 1) % 10 == 0:
            print(
                f"Époque {epoch + 1}/{epochs} "
                f"| Erreur : {average_loss:.4f}"
            )

    # Sauvegarder le cerveau
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    save_path = models_dir / "angelia_transformer.pth"

    torch.save(
        {
            "model_state_dict": brain.state_dict(),
            "vocab_size": vocab_size,
            "embedding_size": 64,
            "num_heads": 4,
            "num_layers": 2,
            "max_sequence_length": 256,
            "dropout": 0.1
        },
        save_path
    )

    print("------------------------------------")
    print("🎉 Entraînement terminé !")
    print(f"Cerveau sauvegardé : {save_path}")


if __name__ == "__main__":
    train()
