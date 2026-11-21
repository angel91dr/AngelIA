
import json
import re
import unicodedata
from pathlib import Path


class AngelIAKnowledge:

    def __init__(self):
        self.file = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "knowledge.json"
        )

        self.knowledge = self.load()

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

        text = re.sub(r"[^\w\s'-]", " ", text)

        text = " ".join(text.split())

        text = re.sub(
            r"^(le|la|les|un|une|des|l')\s*",
            "",
            text
        )

        return text

    # ---------------------------------
    # CHARGEMENT
    # ---------------------------------

    def load(self):
        if not self.file.exists():
            return {}

        try:
            with open(self.file, "r", encoding="utf-8") as file:
                data = json.load(file)

            return data if isinstance(data, dict) else {}

        except (json.JSONDecodeError, OSError):
            return {}

    # ---------------------------------
    # SAUVEGARDE
    # ---------------------------------

    def save(self):
        self.file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.file, "w", encoding="utf-8") as file:
            json.dump(
                self.knowledge,
                file,
                ensure_ascii=False,
                indent=4
            )

    # ---------------------------------
    # APPRENTISSAGE
    # ---------------------------------

    def remember(self, message):

        message = message.strip().rstrip(".!? ")

        patterns = [
            (
                r"(.+?)\s+est\s+la\s+capitale\s+de\s+(.+)",
                "capitale_de"
            ),
            (
                r"(.+?)\s+est\s+en\s+(.+)",
                "est_en"
            ),
            (
                r"(.+?)\s+se trouve en\s+(.+)",
                "est_en"
            ),
            (
                r"(.+?)\s+se situe en\s+(.+)",
                "est_en"
            )
        ]

        for pattern, relation in patterns:

            match = re.fullmatch(
                pattern,
                message,
                re.IGNORECASE
            )

            if not match:
                continue

            subject = match.group(1).strip()
            answer = match.group(2).strip()

            if not subject or not answer:
                continue

            key = self.normalize(subject)

            old_value = self.knowledge.get(key)

            self.knowledge[key] = {
                "subject": subject,
                "relation": relation,
                "answer": answer
            }

            self.save()

            return {
                "subject": subject,
                "relation": relation,
                "answer": answer,
                "updated": old_value is not None
            }

        return None

    # ---------------------------------
    # RECHERCHE DIRECTE
    # ---------------------------------

    def recall(self, message):

        question = self.normalize(message)

        question = re.sub(
            r"^(quelle|quel|quels|quelles) est ",
            "",
            question
        )

        question = question.rstrip(" ?!.")

        if question in self.knowledge:
            return self.knowledge[question]

        return self.recall_indirect(question)

    # ---------------------------------
    # ASSOCIATION DES CONNAISSANCES
    # ---------------------------------

    def recall_indirect(self, question):

        patterns = [
            r"sur quel continent se trouve (.+)",
            r"dans quel continent se trouve (.+)",
            r"ou se trouve (.+)"
        ]

        for pattern in patterns:

            match = re.fullmatch(pattern, question)

            if not match:
                continue

            requested_subject = self.normalize(
                match.group(1)
            )

            first_fact = self.knowledge.get(
                requested_subject
            )

            if not first_fact:
                continue

            if first_fact.get("relation") != "capitale_de":
                continue

            country = self.normalize(
                first_fact["answer"]
            )

            second_fact = self.knowledge.get(country)

            if not second_fact:
                continue

            if second_fact.get("relation") != "est_en":
                continue

            return {
                "subject": first_fact["subject"],
                "relation": "est_en",
                "answer": second_fact["answer"]
            }

        return None
