
import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_size, num_heads, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x):
        sequence_length = x.size(1)

        # Empêcher le modèle de regarder les tokens futurs
        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        output, _ = self.attention(
            x, x, x,
            attn_mask=mask,
            need_weights=False
        )

        return output


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_size)
        self.attention = CausalSelfAttention(
            embedding_size,
            num_heads,
            dropout
        )

        self.norm2 = nn.LayerNorm(embedding_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 4),
            nn.GELU(),
            nn.Linear(embedding_size * 4, embedding_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))

        return x


class AngelIATransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size=64,
        num_heads=4,
        num_layers=2,
        max_sequence_length=256,
        dropout=0.1
    ):
        super().__init__()

        self.max_sequence_length = max_sequence_length

        # Représentation des tokens
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )

        # Position de chaque token dans la phrase
        self.position_embedding = nn.Embedding(
            max_sequence_length,
            embedding_size
        )

        # Blocs Transformer
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_size,
                    num_heads,
                    dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(embedding_size)

        # Prédiction du prochain token
        self.output = nn.Linear(embedding_size, vocab_size)

    def forward(self, input_ids):
        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.max_sequence_length:
            raise ValueError(
                "La séquence dépasse la longueur maximale autorisée."
            )

        positions = torch.arange(
            sequence_length,
            device=input_ids.device
        )

        tokens = self.token_embedding(input_ids)
        positions = self.position_embedding(positions)

        x = tokens + positions

        x = self.blocks(x)
        x = self.final_norm(x)

        logits = self.output(x)

        return logits


if __name__ == "__main__":
    torch.manual_seed(42)

    brain = AngelIATransformer(vocab_size=92)

    test = torch.tensor([[4, 5, 6, 7]], dtype=torch.long)

    result = brain(test)

    print("🧠 Transformer AngelIA initialisé !")
    print("Entrée :", test.shape)
    print("Sortie :", result.shape)

    parameters = sum(
        parameter.numel()
        for parameter in brain.parameters()
    )

    print("Nombre de paramètres :", parameters)
