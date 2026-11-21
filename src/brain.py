
import torch
import torch.nn as nn


class AngelIABrain(nn.Module):
    def __init__(self, vocab_size, embedding_size=64, hidden_size=128):
        super().__init__()

        # Transformer les identifiants des tokens en vecteurs
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # Réseau récurrent qui traite les tokens dans l'ordre
        self.gru = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Prédire le prochain token du vocabulaire
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)

        output, hidden = self.gru(embedded, hidden)

        logits = self.output(output)

        return logits, hidden


if __name__ == "__main__":
    brain = AngelIABrain(vocab_size=92)

    test = torch.tensor([[4, 5, 6]], dtype=torch.long)

    result, hidden = brain(test)

    print("🧠 Cerveau neuronal d'AngelIA initialisé !")
    print("Tokens en entrée :", test.shape)
    print("Prédictions :", result.shape)
    print("Nombre de paramètres :", sum(p.numel() for p in brain.parameters()))
