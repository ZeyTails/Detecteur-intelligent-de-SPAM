"""Surveille Mailpit, classe les e-mails et ajoute des tags.

Ce fichier regroupe l'ancien classifier.py et watcher.py.
"""

import argparse
import json
import os
import re
import time
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any

import joblib
import requests
from bs4 import BeautifulSoup


BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / "runtime"
MODEL_PATH = BASE_DIR.parent / "models" / "spam_tfidf_nb.joblib"
API = os.environ.get("MAILPIT_API", "http://localhost:8025/api/v1").rstrip("/")
POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "2"))
REVIEW_MARGIN = float(os.environ.get("REVIEW_MARGIN", "0.08"))
TOP_TERMS = int(os.environ.get("TOP_TERMS", "3"))
TIMEOUT = 8


def normalize_label(value: object) -> str:
    value = str(value or "").strip().lower()
    if value in {"spam", "1", "true"}:
        return "spam"
    if value in {"ham", "0", "false"}:
        return "ham"
    return ""


class SpamClassifier:
    """Charge le modèle entraîné et retourne une décision exploitable."""

    def __init__(self, model_path: str | None = None) -> None:
        candidates = [
            Path(model_path) if model_path else None,
            Path(os.environ["MODEL_PATH"]) if os.environ.get("MODEL_PATH") else None,
            MODEL_PATH,
        ]
        path = next((item for item in candidates if item and item.exists()), None)
        if path is None:
            raise FileNotFoundError("Modèle introuvable. Lancez d'abord : python Spam.py")

        loaded = joblib.load(path)
        if not isinstance(loaded, dict):
            raise ValueError("Le modèle doit contenir le vectoriseur TF-IDF et le classifieur.")
        self.vectorizer = loaded.get("tfidf") or loaded.get("vectorizer")
        self.classifier = loaded.get("nb") or loaded.get("model") or loaded.get("classifier")
        if self.vectorizer is None or self.classifier is None:
            raise ValueError("Le modèle ne contient pas les éléments attendus.")
        self.threshold = float(loaded.get("spam_threshold", 0.5))
        try:
            self.threshold = float(os.environ.get("SPAM_THRESHOLD", self.threshold))
        except ValueError:
            pass

    @staticmethod
    def preprocess(text: str) -> str:
        text = (text or "").lower()
        text = re.sub(
            r"^(from|to|subject|cc|bcc|date|received|return-path|message-id):.*$",
            " ",
            text,
            flags=re.MULTILINE,
        )
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def predict(self, text: str) -> dict[str, Any]:
        clean = self.preprocess(text)
        features = self.vectorizer.transform([clean])
        probabilities = self.classifier.predict_proba(features)[0]
        classes = [str(item).lower() for item in self.classifier.classes_]
        spam_index = classes.index("spam") if "spam" in classes else (classes.index("1") if "1" in classes else 1)
        probability = float(probabilities[spam_index])
        label = "spam" if probability >= self.threshold else "ham"
        distance = abs(probability - self.threshold)
        confidence = "low" if distance < 0.10 else ("medium" if distance < 0.25 else "high")
        return {
            "label": label,
            "proba_spam": probability,
            "threshold": self.threshold,
            "confidence": confidence,
            "top_terms": self.explain(clean, label, TOP_TERMS),
        }

    def explain(self, clean: str, label: str, top_k: int) -> list[str]:
        """Retourne les mots les plus influents, si le modèle le permet."""
        if not hasattr(self.classifier, "feature_log_prob_"):
            return []
        row = self.vectorizer.transform([clean]).tocoo()
        if row.nnz == 0:
            return []
        classes = [str(item).lower() for item in self.classifier.classes_]
        spam_idx = classes.index("spam") if "spam" in classes else 1
        ham_idx = classes.index("ham") if "ham" in classes else 0
        weights = self.classifier.feature_log_prob_[spam_idx] - self.classifier.feature_log_prob_[ham_idx]
        names = self.vectorizer.get_feature_names_out()
        scored = [(float(weights[i]) * float(value), str(names[i])) for i, value in zip(row.col, row.data)]
        scored.sort(key=lambda item: item[0], reverse=(label == "spam"))
        return [term for _, term in scored[:top_k]]


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except (OSError, json.JSONDecodeError):
        return default


def save_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def message_id(message: dict[str, Any]) -> str:
    for key in ("ID", "Id", "id", "MessageID", "MessageId"):
        if message.get(key):
            return str(message[key])
    return ""


def list_messages(session: requests.Session) -> list[dict[str, Any]]:
    payload = session.get(f"{API}/messages", timeout=TIMEOUT).json()
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return list(payload.get("messages") or payload.get("Messages") or [])
    return []


def get_message(session: requests.Session, mid: str) -> tuple[dict[str, Any], bytes]:
    detail: dict[str, Any] = {}
    raw = b""
    for url in (f"{API}/message/{mid}", f"{API}/messages/{mid}"):
        response = session.get(url, timeout=TIMEOUT)
        if response.ok:
            detail = response.json()
            break
    for url in (f"{API}/message/{mid}/raw", f"{API}/messages/{mid}/raw"):
        response = session.get(url, timeout=TIMEOUT)
        if response.ok and response.content:
            raw = response.content
            break
    return detail, raw


def raw_content(raw: bytes) -> tuple[str, dict[str, str]]:
    message = BytesParser(policy=policy.default).parsebytes(raw)
    subject = str(message.get("Subject", "")).strip()
    parts: list[str] = []
    for part in message.walk() if message.is_multipart() else [message]:
        if part.get_content_type() not in {"text/plain", "text/html"}:
            continue
        try:
            content = str(part.get_content() or "")
        except Exception:
            continue
        if part.get_content_type() == "text/html":
            content = BeautifulSoup(content, "html.parser").get_text(" ", strip=True)
        parts.append(content)
    return " ".join([subject, *parts]).strip(), {
        "subject": subject,
        "expected": normalize_label(message.get("X-Expected-Label")),
        "run_id": str(message.get("X-Run-Id", "")).strip(),
        "from": str(message.get("From", "")).strip(),
    }


def detail_content(detail: dict[str, Any]) -> str:
    message = detail.get("Message", {}) if isinstance(detail.get("Message"), dict) else {}
    subject = str(detail.get("Subject") or detail.get("subject") or message.get("Subject") or "")
    text = detail.get("Text") or detail.get("text") or message.get("Text") or ""
    html = detail.get("HTML") or detail.get("html") or message.get("HTML") or ""
    if isinstance(text, list):
        text = " ".join(str(item) for item in text)
    if not text and html:
        html = " ".join(str(item) for item in html) if isinstance(html, list) else str(html)
        text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    return f"{subject} {text}".strip()


def normal_tags(tags: object) -> list[str]:
    output: list[str] = []
    for tag in tags or []:
        value = str(tag if isinstance(tag, str) else tag.get("Name") or tag.get("name") or "").strip()
        if value and value not in output:
            output.append(value)
    return output


def add_tags(session: requests.Session, mid: str, existing: object, new: list[str]) -> bool:
    old_detection_prefixes = (
        "final:", "model:", "conf:", "truth:", "check:", "run:",
        "resultat:", "confiance:", "verification:",
    )
    tags = [
        tag for tag in normal_tags(existing)
        if not tag.lower().startswith(old_detection_prefixes)
    ]
    tags.extend(tag for tag in new if tag not in tags)
    response = session.put(f"{API}/tags", json={"IDs": [mid], "Tags": tags}, timeout=TIMEOUT)
    return response.ok


def default_metrics() -> dict[str, Any]:
    return {"processed": 0, "tagged": 0, "spam": 0, "ham": 0, "review": 0, "ok": 0, "ko": 0}


def process_one(session: requests.Session, classifier: SpamClassifier, message: dict[str, Any], metrics: dict[str, Any], dry_run: bool) -> bool:
    mid = message_id(message)
    detail, raw = get_message(session, mid)
    raw_text, metadata = raw_content(raw) if raw else ("", {})
    text = detail_content(detail) or raw_text
    prediction = classifier.predict(text)
    label = str(prediction["label"])
    expected = metadata.get("expected", "")
    is_review = abs(prediction["proba_spam"] - prediction["threshold"]) < REVIEW_MARGIN
    decision = "final:review" if is_review else f"final:{label}"
    readable_result = "normal" if label == "ham" else "spam"
    readable_confidence = {"low": "faible", "medium": "moyenne", "high": "forte"}[prediction["confidence"]]
    tags = [f"resultat:{readable_result}", f"confiance:{readable_confidence}"]
    run_id = metadata.get("run_id", "")
    if expected:
        correct = expected == label
        tags.append(f"verification:{'correcte' if correct else 'incorrecte'}")
        metrics["ok" if correct else "ko"] += 1
    tagged = True if dry_run else add_tags(session, mid, detail.get("Tags") or detail.get("tags") or message.get("Tags") or [], tags)
    metrics["processed"] += 1
    metrics["tagged"] += int(tagged)
    metrics[label] += 1
    metrics["review"] += int(decision == "final:review")
    print(f"[{mid}] spam={prediction['proba_spam']:.2f} → {label.upper()} ({prediction['confidence']}) {'ok' if tagged else 'tag error'}", flush=True)
    with (RUNTIME_DIR / "predictions_log.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps({"message_id": mid, "run_id": run_id, "expected_label": expected, **prediction, "decision": decision}, ensure_ascii=False) + "\n")
    return tagged


def main() -> None:
    parser = argparse.ArgumentParser(description="Surveille Mailpit et classe les nouveaux e-mails.")
    parser.add_argument("--once", action="store_true", help="Process messages once, then stop")
    parser.add_argument("--reset-state", action="store_true", help="Forget previously processed messages")
    parser.add_argument("--dry-run", action="store_true", help="Predict without adding Mailpit tags")
    parser.add_argument("--poll-seconds", type=float, default=POLL_SECONDS, metavar="SECONDS", help="Time between checks")
    parser.add_argument("--max-messages", type=int, default=0, metavar="NUMBER", help="Maximum messages per check")
    args = parser.parse_args()

    RUNTIME_DIR.mkdir(exist_ok=True)
    state_path, metrics_path = RUNTIME_DIR / "state.json", RUNTIME_DIR / "metrics.json"
    state = {"seen_ids": []} if args.reset_state else load_json(state_path, {"seen_ids": []})
    seen = set(state.get("seen_ids", []))
    metrics = load_json(metrics_path, default_metrics())
    for key, value in default_metrics().items():
        metrics.setdefault(key, value)
    classifier = SpamClassifier()
    print(f"Détecteur actif sur {API}. Modèle : {MODEL_PATH.name}")

    with requests.Session() as session:
        while True:
            try:
                count = 0
                for message in list_messages(session):
                    mid = message_id(message)
                    if not mid or mid in seen:
                        continue
                    if process_one(session, classifier, message, metrics, args.dry_run):
                        seen.add(mid)
                    count += 1
                    if args.max_messages and count >= args.max_messages:
                        break
                save_json(state_path, {"seen_ids": sorted(seen)})
                save_json(metrics_path, metrics)
                if args.once:
                    print(f"Résumé : {metrics}")
                    return
            except requests.RequestException as error:
                print(f"Erreur Mailpit : {error}", flush=True)
            time.sleep(max(0.1, args.poll_seconds))


if __name__ == "__main__":
    main()
