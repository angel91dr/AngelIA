
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.brain import AngelIABrain
from src.tokenizer import AngelIATokenizer


class AngelIAChat:
    def __init__(self):
        self.tokenizer = AngelIATokenizer()
        self.tokenizer.load(ROOT / "data" / "vocab.json")

        with open(ROOT / "data" / "vocab.json", "r", encoding="utf-8") as file:
            self.vocab = json.load(file)

        checkpoint = torch.load(
            ROOT / "models" / "angelia_brain.pth",
            map_location="cpu",
            weights_only=True
        )

        self.brain = AngelIABrain(
            vocab_size=checkpoint["vocab_size"],
            embedding_size=checkpoint["embedding_size"],
            hidden_size=checkpoint["hidden_size"]
        )

        self.brain.load_state_dict(checkpoint["model_state_dict"])
        self.brain.eval()

        self.bos_id = self.vocab["<BOS>"]
        self.eos_id = self.vocab["<EOS>"]
        self.pad_id = self.vocab["<PAD>"]

    def generate(self, message, max_tokens=40):
        tokens = self.tokenizer.encode(message)

        if not tokens:
            return "Je n'ai pas compris ton message."

        sequence = [self.bos_id] + tokens
        input_tensor = torch.tensor([sequence], dtype=torch.long)

        generated = []

        with torch.no_grad():
            for _ in range(max_tokens):
                logits, _ = self.brain(input_tensor)

                next_token_logits = logits[0, -1, :].clone()

                next_token_logits[self.bos_id] = float("-inf")
                next_token_logits[self.pad_id] = float("-inf")

                next_token = int(torch.argmax(next_token_logits).item())

                if next_token == self.eos_id:
                    break

                generated.append(next_token)

                input_tensor = torch.cat(
                    [
                        input_tensor,
                        torch.tensor([[next_token]], dtype=torch.long)
                    ],
                    dim=1
                )

        if not generated:
            return "Je ne sais pas encore quoi répondre."

        return self.tokenizer.decode(generated)


def main():
    chat = AngelIAChat()

    print("================================")
    print("       ANGELIA - PREMIER CHAT")
    print("================================")
    print("Écris 'quitter' pour arrêter.\n")

    while True:
        message = input("Toi : ").strip()

        if message.lower() == "quitter":
            print("AngelIA : À bientôt ! ❤️")
            break

        if not message:
            continue

        response = chat.generate(message)

        print("AngelIA :", response)
        print()


if __name__ == "__main__":
    main()
