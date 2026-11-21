
import re
import torch
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformer.model import AngelIATransformer
from src.tokenizer import AngelIATokenizer
from src.memory import AngelIAMemory
from src.knowledge import AngelIAKnowledge


# ---------------------------------
# CHARGEMENT DU TOKENIZER
# ---------------------------------

tokenizer = AngelIATokenizer()
tokenizer.load(ROOT / "data" / "vocab.json")

vocab = tokenizer.vocab

bos_id = vocab["<BOS>"]
eos_id = vocab["<EOS>"]
user_id = vocab["<USER>"]
angelia_id = vocab["<ANGELIA>"]

special_ids = {
    vocab["<PAD>"],
    vocab["<UNK>"],
    bos_id,
    user_id,
    angelia_id
}


# ---------------------------------
# CHARGEMENT DES MEMOIRES
# ---------------------------------

memory = AngelIAMemory()
knowledge = AngelIAKnowledge()


# ---------------------------------
# CHARGEMENT DU TRANSFORMER
# ---------------------------------

checkpoint = torch.load(
    ROOT / "models" / "angelia_transformer.pth",
    map_location="cpu",
    weights_only=True
)

brain = AngelIATransformer(
    vocab_size=checkpoint["vocab_size"],
    embedding_size=checkpoint["embedding_size"],
    num_heads=checkpoint["num_heads"],
    num_layers=checkpoint["num_layers"],
    max_sequence_length=checkpoint["max_sequence_length"],
    dropout=checkpoint["dropout"]
)

brain.load_state_dict(checkpoint["model_state_dict"])
brain.eval()


# ---------------------------------
# FORMATAGE DES ANIMAUX
# ---------------------------------

def format_animal(animal):
    animal = animal.strip()

    if re.match(
        r"^(un|une|des|le|la|les)\s",
        animal,
        re.IGNORECASE
    ):
        return animal

    feminine_animals = {
        "tortue", "souris", "poule", "vache", "chèvre",
        "chevre", "lapine", "chienne", "chatte"
    }

    if animal.lower() in feminine_animals:
        return f"une {animal}"

    return f"un {animal}"


# ---------------------------------
# FORMATAGE DES SOUVENIRS
# ---------------------------------

def format_memory(category, value):
    if isinstance(value, list):
        if not value:
            return "Je n'ai pas encore retenu cette information !"

        if category == "animal":
            value = [format_animal(item) for item in value]

        if len(value) == 1:
            items = value[0]
        elif len(value) == 2:
            items = f"{value[0]} et {value[1]}"
        else:
            items = ", ".join(value[:-1])
            items += f" et {value[-1]}"
    else:
        items = value

    responses = {
        "prenom": f"Tu m'as dit que tu t'appelles {items} !",
        "age": f"Tu m'as dit que tu as {items} ans !",
        "animal": f"Tu m'as dit que tu as {items} !",
        "preference": f"Tu m'as dit que tu aimes {items} !",
        "projet": f"Tu m'as dit que tu travailles sur {items} !"
    }

    return responses.get(
        category,
        "Je me souviens de cette information !"
    )


# ---------------------------------
# REPONSES AFFECTUEUSES
# ---------------------------------

def affectionate_response(message):
    normalized = memory.normalize(message)

    if normalized in (
        "je t'aime",
        "je t'aime beaucoup",
        "je t'adore"
    ):
        return (
            "C'est vraiment gentil ! ❤️ "
            "Je suis là pour discuter avec toi "
            "et te tenir compagnie !"
        )

    if normalized == "tu me manques":
        return (
            "C'est adorable de me dire ça ! ❤️ "
            "Je suis contente de pouvoir discuter avec toi !"
        )

    return None


# ---------------------------------
# GENERATION DU TRANSFORMER
# ---------------------------------

def generate_transformer_response(message):
    user_tokens = tokenizer.encode(message)

    context = [
        bos_id,
        user_id
    ] + user_tokens

    sequence = context.copy()

    with torch.no_grad():
        for step in range(100):
            if len(sequence) >= brain.max_sequence_length:
                break

            inputs = torch.tensor(
                [sequence],
                dtype=torch.long
            )

            logits = brain(inputs)
            next_token_logits = logits[0, -1, :].clone()

            forbidden_ids = {
                vocab["<PAD>"],
                vocab["<UNK>"],
                bos_id,
                user_id
            }

            for token_id in forbidden_ids:
                next_token_logits[token_id] = float("-inf")

            if step > 0:
                next_token_logits[angelia_id] = float("-inf")

            next_token_id = torch.argmax(
                next_token_logits
            ).item()

            if next_token_id == eos_id:
                break

            sequence.append(next_token_id)

    response_tokens = sequence[len(context):]

    response_tokens = [
        token_id
        for token_id in response_tokens
        if token_id not in special_ids
        and token_id != eos_id
    ]

    response = tokenizer.decode(response_tokens).strip()

    if not response:
        return (
            "Je suis encore en train d'apprendre "
            "à mieux te répondre !"
        )

    if response.lower().startswith("beaucoup"):
        return (
            "Je suis encore en train d'apprendre "
            "à mieux te répondre !"
        )

    if response[-1] not in ".!?":
        response += " !"

    return response


# ---------------------------------
# REPONSE A UNE QUESTION
# ---------------------------------

def generate_single_response(message):

    # REPONSES AFFECTUEUSES

    affectionate = affectionate_response(message)

    if affectionate:
        return affectionate

    # MEMOIRE PERSONNELLE : RETROUVER

    remembered = memory.recall(message)

    if remembered:
        if remembered["value"]:
            return format_memory(
                remembered["category"],
                remembered["value"]
            )

        return (
            "Je ne connais pas encore cette information. "
            "Tu peux me la dire !"
        )

    # MEMOIRE DES CONNAISSANCES : RETROUVER

    remembered_knowledge = knowledge.recall(message)

    if remembered_knowledge:

        subject = remembered_knowledge["subject"]
        relation = remembered_knowledge["relation"]
        answer = remembered_knowledge["answer"]

        if relation == "est_en":
            return f"{subject} se trouve en {answer} !"

        if relation == "capitale_de":
            return f"{subject} est la capitale de {answer} !"

        return f"{subject} est {answer} !"

    # MEMOIRE PERSONNELLE : APPRENDRE

    new_memory = memory.remember(message)

    if new_memory:
        category = new_memory["category"]
        value = new_memory["value"]

        if new_memory.get("already_known"):
            return "Je connais déjà cette information ! ❤️"

        if new_memory.get("updated"):
            return (
                f"D'accord ! Je remplace mon ancien "
                f"souvenir par {value} !"
            )

        if category == "prenom":
            return f"Enchantée {value} ! Je retiens ton prénom !"

        if category == "age":
            return f"D'accord ! Je retiens que tu as {value} ans !"

        if category == "animal":
            return (
                f"Oh ! Je retiens que tu as "
                f"{format_animal(value)} !"
            )

        if category == "preference":
            return f"Je retiens aussi que tu aimes {value} !"

        if category == "projet":
            return (
                f"C'est intéressant ! Je retiens aussi "
                f"ton projet : {value} !"
            )

    # MEMOIRE DES CONNAISSANCES : APPRENDRE

    new_knowledge = knowledge.remember(message)

    if new_knowledge:

        subject = new_knowledge["subject"]
        relation = new_knowledge["relation"]
        answer = new_knowledge["answer"]

        if new_knowledge["updated"]:
            prefix = "D'accord ! J'ai mis à jour mon souvenir : "
        else:
            prefix = "D'accord ! Je retiens que "

        if relation == "est_en":
            return f"{prefix}{subject} se trouve en {answer} !"

        if relation == "capitale_de":
            return f"{prefix}{subject} est la capitale de {answer} !"

        return f"{prefix}{subject} est {answer} !"

    # AUCUN SOUVENIR : UTILISER LE TRANSFORMER

    return generate_transformer_response(message)


# ---------------------------------
# PLUSIEURS QUESTIONS EN MEME TEMPS
# ---------------------------------

def generate_response(message):
    questions = re.split(
        r"(?<=[?!])\s+|\n+",
        message.strip()
    )

    questions = [
        question.strip()
        for question in questions
        if question.strip()
    ]

    responses = []

    for question in questions:
        response = generate_single_response(question)

        if response:
            responses.append(response)

    return " ".join(responses)


# ---------------------------------
# LANCEMENT DE LA CONVERSATION
# ---------------------------------

def chat():
    print("🧠 AngelIA Transformer est prête !")
    print("Écris 'quitter' pour arrêter.")
    print("Tu peux envoyer plusieurs questions.")
    print("Appuie deux fois sur Entrée pour envoyer.")
    print("------------------------------------")

    while True:
        print("\nToi : ", end="", flush=True)
        lines = []

        while True:
            try:
                line = input()
            except EOFError:
                return

            if not line.strip():
                if lines:
                    break
                continue

            if not lines and line.lower().strip() == "quitter":
                print("AngelIA : À bientôt ! ❤️")
                return

            lines.append(line)

        message = "\n".join(lines)
        response = generate_response(message)

        print("AngelIA :", response)


# ---------------------------------
# PROGRAMME PRINCIPAL
# ---------------------------------

if __name__ == "__main__":
    chat()
