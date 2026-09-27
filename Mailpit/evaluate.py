"""Calcule les résultats d'un test envoyé dans Mailpit.

Ce fichier remplace evaluate_run.py.
"""

import argparse
import json
import os
import re
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any

import requests


API = os.environ.get("MAILPIT_API", "http://localhost:8025/api/v1").rstrip("/")
TIMEOUT = 8
RUN_RE = re.compile(r"\[RUN:([^\]]+)\]", re.IGNORECASE)
AUDIT_PATH = Path(__file__).resolve().parent / "runtime" / "predictions_log.jsonl"


def normalize_label(value: object) -> str:
    value = str(value or "").strip().lower()
    if value in {"spam", "1", "true"}:
        return "spam"
    if value in {"ham", "0", "false"}:
        return "ham"
    return ""


def normalize_run_id(value: str) -> str:
    value = (value or "").strip()
    return value.split(":", 1)[1].strip() if value.lower().startswith("run:") else value


def message_id(message: dict[str, Any]) -> str:
    for key in ("ID", "Id", "id", "MessageID", "MessageId"):
        if message.get(key):
            return str(message[key])
    return ""


def list_messages(session: requests.Session) -> list[dict[str, Any]]:
    response = session.get(f"{API}/messages", timeout=TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list):
        return payload
    return list(payload.get("messages") or payload.get("Messages") or []) if isinstance(payload, dict) else []


def detail_and_raw(session: requests.Session, mid: str) -> tuple[dict[str, Any], bytes]:
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


def tags_from(value: object) -> list[str]:
    tags: list[str] = []
    for tag in value or []:
        if isinstance(tag, str):
            name = tag
        elif isinstance(tag, dict):
            name = str(tag.get("Name") or tag.get("name") or tag.get("Tag") or "")
        else:
            name = str(tag)
        if name.strip():
            tags.append(name.strip().lower())
    return tags


def headers_from(raw: bytes) -> dict[str, str]:
    if not raw:
        return {"subject": "", "run_id": "", "expected": ""}
    message = BytesParser(policy=policy.default).parsebytes(raw)
    return {
        "subject": str(message.get("Subject", "")).strip(),
        "run_id": str(message.get("X-Run-Id", "")).strip(),
        "expected": normalize_label(message.get("X-Expected-Label")),
    }


def prediction_from_tags(tags: list[str]) -> str:
    if "resultat:spam" in tags:
        return "spam"
    if "resultat:normal" in tags:
        return "ham"
    for label in ("spam", "ham"):
        if f"final:{label}" in tags or f"model:{label}" in tags:
            return label
    return ""


def expected_from_tags(tags: list[str]) -> str:
    for label in ("spam", "ham"):
        if f"truth:{label}" in tags:
            return label
    return ""


def run_from_subject(subject: str) -> str:
    match = RUN_RE.search(subject)
    return match.group(1).strip() if match else ""


def run_from_tags(tags: list[str]) -> str:
    for tag in tags:
        if tag.startswith("run:"):
            return tag[4:]
    return ""


def subject_key(subject: str) -> str:
    return " ".join(str(subject or "").split()).strip().lower()


def load_audit_index() -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """Charge le journal local comme secours si Mailpit ne renvoie pas les tags."""
    by_message: dict[str, dict[str, Any]] = {}
    by_run_subject: dict[tuple[str, str], dict[str, Any]] = {}
    if not AUDIT_PATH.exists():
        return by_message, by_run_subject
    try:
        lines = AUDIT_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return by_message, by_run_subject

    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        predicted = normalize_label(item.get("predicted_label") or item.get("label"))
        record = {
            "expected": normalize_label(item.get("expected_label")),
            "predicted": predicted,
            "proba_spam": item.get("proba_spam"),
            "threshold": item.get("threshold"),
            "confidence": str(item.get("confidence") or ""),
            "top_terms": item.get("top_terms") if isinstance(item.get("top_terms"), list) else [],
            "decision": str(item.get("decision") or ""),
        }
        mid = str(item.get("message_id") or "").strip()
        run_id = normalize_run_id(str(item.get("run_id") or "").strip()).lower()
        subject = subject_key(str(item.get("subject") or ""))
        if mid:
            by_message[mid] = record
        if run_id and subject:
            by_run_subject[(run_id, subject)] = record
    return by_message, by_run_subject


def metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    tn = fp = fn = tp = skipped = 0
    for record in records:
        expected, predicted = record["expected"], record["predicted"]
        if expected not in {"ham", "spam"} or predicted not in {"ham", "spam"}:
            skipped += 1
        elif expected == "spam" and predicted == "spam":
            tp += 1
        elif expected == "ham" and predicted == "spam":
            fp += 1
        elif expected == "spam" and predicted == "ham":
            fn += 1
        else:
            tn += 1
    used = tn + fp + fn + tp

    def divide(top: int, bottom: int) -> float:
        return top / bottom if bottom else 0.0

    spam_precision = divide(tp, tp + fp)
    spam_recall = divide(tp, tp + fn)
    ham_precision = divide(tn, tn + fn)
    ham_recall = divide(tn, tn + fp)
    return {
        "used_records": used,
        "skipped_records": skipped,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": divide(tp + tn, used),
        "spam": {
            "precision": spam_precision,
            "recall": spam_recall,
            "f1": divide(2 * spam_precision * spam_recall, spam_precision + spam_recall),
            "support": tp + fn,
        },
        "ham": {
            "precision": ham_precision,
            "recall": ham_recall,
            "f1": divide(2 * ham_precision * ham_recall, ham_precision + ham_recall),
            "support": tn + fp,
        },
    }


def print_report(result: dict[str, Any]) -> None:
    matrix = result["confusion"]
    print("Matrice de confusion :")
    print(f"[[{matrix['tn']} {matrix['fp']}]\n [{matrix['fn']} {matrix['tp']}]]")
    print(f"Accuracy : {result['accuracy']:.1%}")
    for label in ("ham", "spam"):
        values = result[label]
        print(
            f"{label} — précision={values['precision']:.1%}, "
            f"rappel={values['recall']:.1%}, F1={values['f1']:.1%}, "
            f"support={values['support']}"
        )
    print(f"Messages évalués : {result['used_records']} | ignorés : {result['skipped_records']}")


def readable_label(label: str) -> str:
    return {"ham": "normal", "spam": "spam"}.get(label, label or "inconnu")


def readable_confidence(value: str) -> str:
    return {"low": "faible", "medium": "moyenne", "high": "forte"}.get(value, value or "inconnue")


def print_message_details(record: dict[str, Any]) -> None:
    expected = str(record.get("expected") or "")
    predicted = str(record.get("predicted") or "")
    valid = expected in {"ham", "spam"} and predicted in {"ham", "spam"}
    correct = valid and expected == predicted
    print("Détails du message :")
    print(f"Identifiant : {record.get('message_id', '')}")
    print(f"Sujet       : {record.get('subject', '') or '-'}")
    print(f"Attendu     : {readable_label(expected)}")
    print(f"Prédit      : {readable_label(predicted)}")
    print(f"Vérification: {'correcte' if correct else ('incorrecte' if valid else 'inconnue')}")

    probability = record.get("proba_spam")
    threshold = record.get("threshold")
    if probability is not None:
        print(f"Probabilité spam : {float(probability):.2%}")
    if threshold is not None:
        print(f"Seuil utilisé    : {float(threshold):.2%}")
    confidence = str(record.get("confidence") or "")
    if confidence:
        print(f"Confiance        : {readable_confidence(confidence)}")
    terms = record.get("top_terms") or []
    if terms:
        print(f"Mots influents   : {', '.join(str(term) for term in terms[:5])}")


def print_details(records: list[dict[str, Any]]) -> None:
    print("\nDétails des messages :")
    for record in records:
        expected = readable_label(str(record.get("expected") or ""))
        predicted = readable_label(str(record.get("predicted") or ""))
        probability = record.get("proba_spam")
        probability_text = f"{float(probability):.2%}" if probability is not None else "-"
        status = "correct" if expected == predicted and expected in {"normal", "spam"} else "incorrect"
        print(
            f"- {record.get('message_id', '')}: attendu={expected}, "
            f"prédit={predicted}, probabilité_spam={probability_text}, "
            f"résultat={status}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Évalue les prédictions faites par app.py.")
    parser.add_argument("--run-id", dest="run_id", default="", metavar="ID", help="Test identifier")
    parser.add_argument("--message-id", dest="message_id", default="", metavar="ID", help="Show one message")
    parser.add_argument("--limit", dest="limit", type=int, default=0, metavar="NUMBER", help="Maximum number of messages")
    parser.add_argument("--save-json", dest="save_json", default="", metavar="PATH", help="Path for the exported report")
    parser.add_argument("--details", action="store_true", help="Show technical details for each message")
    args = parser.parse_args()
    wanted_run = normalize_run_id(args.run_id).lower()
    audit_by_message, audit_by_run_subject = load_audit_index()
    records: list[dict[str, Any]] = []

    with requests.Session() as session:
        messages = list_messages(session)
        if args.limit > 0:
            messages = messages[:args.limit]
        for message in messages:
            mid = message_id(message)
            if not mid or (args.message_id and mid != args.message_id):
                continue
            detail, raw = detail_and_raw(session, mid)
            tags = tags_from(detail.get("Tags") or detail.get("tags") or message.get("Tags") or [])
            headers = headers_from(raw)
            subject = headers["subject"] or str(detail.get("Subject") or detail.get("subject") or "")
            current_run = normalize_run_id(headers["run_id"] or run_from_subject(subject) or run_from_tags(tags))
            if wanted_run and current_run.lower() != wanted_run:
                continue
            expected = headers["expected"] or expected_from_tags(tags)
            predicted = prediction_from_tags(tags)
            audit = audit_by_message.get(mid) or audit_by_run_subject.get(
                (current_run.lower(), subject_key(subject)), {}
            )
            expected = expected or audit.get("expected", "")
            predicted = predicted or audit.get("predicted", "")
            records.append({
                "message_id": mid,
                "run_id": current_run,
                "subject": subject,
                "expected": expected,
                "predicted": predicted,
                "proba_spam": audit.get("proba_spam"),
                "threshold": audit.get("threshold"),
                "confidence": audit.get("confidence", ""),
                "top_terms": audit.get("top_terms", []),
                "decision": audit.get("decision", ""),
            })

    if args.message_id:
        if not records:
            print("Message introuvable.")
            return
        print_message_details(records[0])
        output: dict[str, Any] = {"record": records[0]}
    else:
        result = metrics(records)
        print_report(result)
        if args.details:
            print_details(records)
        output = {"run_id": args.run_id, "metrics": result, "records": records}
    if args.save_json:
        Path(args.save_json).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Rapport enregistré dans {args.save_json}")


if __name__ == "__main__":
    main()
