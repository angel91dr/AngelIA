
import json
import re
import unicodedata
from pathlib import Path


class AngelIAMemory:

    def __init__(self):

        self.memory_file = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "memory.json"
        )

        self.memories = self.load_memory()

    # ---------------------------------
    # NORMALISATION
    # ---------------------------------

    def normalize(self, text):

        text = text.lower().strip()

        text = unicodedata.normalize("NFD", text)

        text = "".join(
            char for char in text
            if unicodedata.category(char) != "Mn"
        )

        text = re.sub(r"[^\w\s']", " ", text)

        return " ".join(text.split())

    # ---------------------------------
    # NETTOYAGE DES VALEURS
    # ---------------------------------

    def normalize_item(self, value):

        value = value.strip()

        value = re.sub(
            r"^(un|une|des|le|la|les|l')\s*",
            "",
            value,
            flags=re.IGNORECASE
        )

        return self.normalize(value)

    # ---------------------------------
    # CHARGEMENT
    # ---------------------------------

    def load_memory(self):

        if not self.memory_file.exists():
            return {}

        try:

            with open(
                self.memory_file,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

        except (json.JSONDecodeError, OSError):

            print("Attention : impossible de lire la mémoire.")
            return {}

        if not isinstance(data, dict):
            return {}

        # Conversion des anciennes mémoires.

        for category in ("prenom", "age"):

            value = data.get(category)

            if isinstance(value, str):

                data[category] = {
                    "current": value,
                    "history": [value]
                }

            elif isinstance(value, dict):

                current = value.get("current")
                history = value.get("history", [])

                if not isinstance(history, list):
                    history = []

                if current and current not in history:
                    history.append(current)

                data[category] = {
                    "current": current,
                    "history": history
                }

        for category in ("animal", "preference", "projet"):

            value = data.get(category, [])

            if isinstance(value, str):
                value = [value]

            if not isinstance(value, list):
                value = []

            cleaned = []
            seen = set()

            for item in value:

                if not isinstance(item, str):
                    continue

                key = self.normalize_item(item)

                if key and key not in seen:

                    seen.add(key)
                    cleaned.append(item)

            data[category] = cleaned

        return data

    # ---------------------------------
    # SAUVEGARDE
    # ---------------------------------

    def save_memory(self):

        self.memory_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            self.memory_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.memories,
                file,
                ensure_ascii=False,
                indent=4
            )

    # ---------------------------------
    # AJOUTER UN SOUVENIR
    # ---------------------------------

    def add_to_list(self, category, value):

        values = self.memories.setdefault(category, [])

        new_key = self.normalize_item(value)

        for existing in values:

            if self.normalize_item(existing) == new_key:
                return False

        values.append(value)

        return True

    # ---------------------------------
    # APPRENTISSAGE
    # ---------------------------------

    def remember(self, message):

        prefix = (
            r"(?:(?:en fait|non|"
            r"je voulais dire que|"
            r"je me suis trompe(?:e)?)"
            r"\s*,?\s*)?"
        )

        patterns = {

            "prenom": [
                prefix + r"je m'appelle\s+([A-Za-zÀ-ÖØ-öø-ÿ'-]+)",
                prefix + r"mon prénom est\s+([A-Za-zÀ-ÖØ-öø-ÿ'-]+)"
            ],

            "age": [
                prefix + r"j'ai\s+(\d{1,3})\s+ans",
                prefix + r"mon âge est\s+(\d{1,3})"
            ],

            "animal": [
                prefix + r"j'ai\s+(?:un|une)\s+([A-Za-zÀ-ÖØ-öø-ÿ'-]+)",
                prefix + r"j'ai aussi\s+(?:un|une)\s+([A-Za-zÀ-ÖØ-öø-ÿ'-]+)"
            ],

            "preference": [
                prefix + r"j'adore\s+(.+)",
                prefix + r"j'aime beaucoup\s+(.+)",
                prefix + r"j'aime\s+(.+)"
            ],

            "projet": [
                prefix + r"je développe\s+(.+)",
                prefix + r"je travaille sur\s+(.+)"
            ]
        }

        for category, expressions in patterns.items():

            for expression in expressions:

                match = re.search(
                    expression,
                    message,
                    re.IGNORECASE
                )

                if not match:
                    continue

                value = match.group(1).strip()
                value = value.rstrip(" .!?")

                if not value:
                    continue

                if category in ("prenom", "age"):

                    record = self.memories.setdefault(
                        category,
                        {
                            "current": None,
                            "history": []
                        }
                    )

                    old_value = record.get("current")

                    record["current"] = value

                    if value not in record["history"]:
                        record["history"].append(value)

                    updated = (
                        old_value is not None
                        and old_value != value
                    )

                    already_known = (
                        old_value == value
                    )

                else:

                    added = self.add_to_list(
                        category,
                        value
                    )

                    updated = False
                    already_known = not added

                if not already_known:
                    self.save_memory()

                return {
                    "category": category,
                    "value": value,
                    "updated": updated,
                    "already_known": already_known
                }

        return None

    # ---------------------------------
    # RECONNAISSANCE DES QUESTIONS
    # ---------------------------------

    def recall(self, message):

        normalized = self.normalize(message)

        questions = {

            "prenom": [
                "quel est mon prenom",
                "quel est mon nom",
                "comment je m'appelle",
                "tu connais mon prenom",
                "tu te souviens de mon prenom",
                "tu te rappelles de mon prenom",
                "tu sais comment je m'appelle"
            ],

            "age": [
                "quel age ai je",
                "quel age j'ai",
                "tu connais mon age",
                "quel est mon age",
                "tu te souviens de mon age",
                "tu te rappelles de mon age",
                "tu sais quel age j'ai"
            ],

            "animal": [
                "quel animal j'ai",
                "j'ai quel animal",
                "tu connais mon animal",
                "tu te souviens de mon animal",
                "quels animaux j'ai",
                "tu as retenu quels animaux"
            ],

            "preference": [
                "qu'est ce que j'aime",
                "qu'est ce que j'adore",
                "quelle est ma preference",
                "tu te souviens de ce que j'aime",
                "tu sais ce que j'aime",
                "quelles sont mes preferences"
            ],

            "projet": [
                "sur quoi je travaille",
                "quel est mon projet",
                "tu connais mon projet",
                "tu te souviens de mon projet",
                "quels sont mes projets",
                "tu sais sur quoi je travaille"
            ]
        }

        for category, possible_questions in questions.items():

            if normalized not in possible_questions:
                continue

            if category in ("prenom", "age"):

                record = self.memories.get(category, {})

                value = (
                    record.get("current")
                    if isinstance(record, dict)
                    else record
                )

            else:

                value = self.memories.get(category, [])

            return {
                "category": category,
                "value": value
            }

        return None
