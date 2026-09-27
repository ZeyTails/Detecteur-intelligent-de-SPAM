"""Envoie des e-mails de démonstration vers Mailpit.

Ce fichier remplace mailer.py, send_test.py et replay_csv.py.
"""

import argparse
import csv
import random
import smtplib
import time
import uuid
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path


DEFAULT_HOST = "localhost"
DEFAULT_PORT = 1025
DEFAULT_FROM = "tester@example.com"
DEFAULT_TO = "receiver@example.com"

HAM_TEMPLATES = [
    {
        "subject": "Sprint planning update",
        "body": "Hi team, sprint planning starts at 9am tomorrow. Please bring your backlog updates.",
    },
    {
        "subject": "Project plan update",
        "body": "Hello, please find attached the updated project plan. Let me know if you have questions.",
    },
]
SPAM_TEMPLATES = [
    {
        "subject": "Claim your special reward now",
        "body": "Your account has been selected for a special reward. Click here now to claim it.",
    },
    {
        "subject": "Cash in on your home equity",
        "body": "Mortgage lenders are ready to compete for your business. Take action now for easy terms.",
    },
]


def normalize_label(value: object) -> str:
    value = str(value or "").strip().lower()
    if value in {"spam", "1", "true"}:
        return "spam"
    if value in {"ham", "normal", "0", "false"}:
        return "ham"
    return ""


def run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def make_message(
    subject: str,
    body: str,
    sender: str,
    recipient: str,
    expected: str = "",
    current_run: str = "",
    source: str = "",
    html: str = "",
) -> EmailMessage:
    message = EmailMessage()
    message["Subject"] = subject.strip() or "No subject"
    message["From"] = sender
    message["To"] = recipient
    if current_run:
        message["X-Run-Id"] = current_run
    if source:
        message["X-Demo-Source"] = source
    if expected:
        message["X-Expected-Label"] = expected
    message.set_content(body or "")
    if html:
        message.add_alternative(html, subtype="html")
    return message


def send(messages: list[EmailMessage], host: str, port: int, interval: float) -> int:
    with smtplib.SMTP(host, port, timeout=10) as smtp:
        for index, message in enumerate(messages, start=1):
            smtp.send_message(message)
            if interval > 0 and index < len(messages):
                time.sleep(interval)
    return len(messages)


def add_connection_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--smtp-host", dest="smtp_host", default=DEFAULT_HOST, metavar="HOST", help="SMTP server address")
    parser.add_argument("--smtp-port", dest="smtp_port", type=int, default=DEFAULT_PORT, metavar="PORT")
    parser.add_argument("--from-addr", dest="from_addr", default=DEFAULT_FROM, metavar="ADDRESS")
    parser.add_argument("--to-addr", dest="to_addr", default=DEFAULT_TO, metavar="ADDRESS")
    parser.add_argument("--run-id", dest="run_id", default="", metavar="ID", help="Message group identifier")


def command_templates(args: argparse.Namespace) -> None:
    current_run = args.run_id.strip() or run_id("manual")
    rng = random.Random(args.seed)
    messages: list[EmailMessage] = []
    selected_types = ["ham", "spam"] if args.kind == "both" else [args.kind]
    for kind in selected_types:
        templates = HAM_TEMPLATES if kind == "ham" else SPAM_TEMPLATES
        for _ in range(max(1, args.count)):
            template = rng.choice(templates)
            messages.append(
                make_message(
                    subject=f"[RUN:{current_run}] {template['subject']}",
                    body=template["body"],
                    sender=args.from_addr,
                    recipient=args.to_addr,
                    expected="" if args.no_expected_label else kind,
                    current_run=current_run,
                    source="templates",
                )
            )
    total = send(messages, args.smtp_host, args.smtp_port, max(0, args.interval))
    print(f"{total} message(s) envoyé(s). run_id={current_run}")


def command_message(args: argparse.Namespace) -> None:
    current_run = args.run_id.strip() or run_id("manual")
    expected = "" if args.no_expected_label else normalize_label(args.expected_label)
    message = make_message(
        subject=f"[RUN:{current_run}] {args.subject.strip() or 'Message manuel'}",
        body=args.body,
        sender=args.from_addr,
        recipient=args.to_addr,
        expected=expected,
        current_run=current_run,
        source="manual",
        html=args.html_body,
    )
    send([message], args.smtp_host, args.smtp_port, 0)
    print(f"Message envoyé. run_id={current_run}")


def parse_label_map(value: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for pair in value.split(","):
        if ":" not in pair:
            continue
        key, label = pair.split(":", 1)
        normalized = normalize_label(label)
        if normalized:
            mapping[key.strip()] = normalized
    return mapping


def command_csv(args: argparse.Namespace) -> None:
    path = Path(args.csv_path)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    if args.limit > 0:
        rows = rows[:args.limit]
    mapping = parse_label_map(args.label_map)
    current_run = args.run_id.strip() or run_id("csv")
    total = 0

    for index, row in enumerate(rows, start=1):
        text = str(row.get(args.text_col, "") or "")
        if args.skip_empty and not text.strip():
            continue
        raw_label = str(row.get(args.label_col, "") or "").strip()
        expected = normalize_label(raw_label) or mapping.get(raw_label, "")
        message_key = f"m{index:04d}-{uuid.uuid4().hex[:8]}"
        subject = str(row.get(args.subject_col, "") or "").strip() or f"Dataset email #{index}"
        sender = str(row.get(args.from_col, "") or "").strip() or args.from_addr
        recipient = str(row.get(args.to_col, "") or "").strip() or args.to_addr
        message = make_message(
            subject=f"[RUN:{current_run}] {subject} [MID:{message_key}]",
            body=text,
            sender=sender,
            recipient=recipient,
            expected=expected,
            current_run=current_run,
            source="csv",
        )
        if args.dry_run:
            print(f"[APERÇU {index}] expected={expected or '-'} subject={subject}")
        else:
            send([message], args.smtp_host, args.smtp_port, 0)
            print(f"[ENVOYÉ {index}] expected={expected or '-'} subject={subject}")
            if args.interval > 0 and index < len(rows):
                time.sleep(args.interval)
        total += 1
    print(f"Terminé : {total} message(s). run_id={current_run}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Envoie des e-mails vers Mailpit.")
    commands = parser.add_subparsers(dest="command", required=True)

    templates = commands.add_parser("templates", help="Send built-in ham/spam examples")
    templates.add_argument("--kind", choices=["ham", "spam", "both"], default="both", help="Message type to send")
    templates.add_argument("--count", type=int, default=1, metavar="NUMBER", help="Number of messages per type")
    templates.add_argument("--interval", type=float, default=0, metavar="SECONDS", help="Seconds between sends")
    templates.add_argument("--seed", type=int, default=42, metavar="VALUE", help="Random selection seed")
    templates.add_argument("--no-expected-label", action="store_true", help="Do not add the reference label")
    add_connection_options(templates)
    templates.set_defaults(handler=command_templates)

    message = commands.add_parser("message", help="Envoie un message personnalisé")
    message.add_argument("--subject", default="", metavar="TEXT")
    message.add_argument("--body", default="", metavar="TEXT")
    message.add_argument("--html-body", default="", metavar="TEXT")
    message.add_argument("--expected-label", choices=["ham", "spam"], default="")
    message.add_argument("--no-expected-label", action="store_true", help="Do not add the reference label")
    add_connection_options(message)
    message.set_defaults(handler=command_message)

    csv_command = commands.add_parser("csv", help="Rejoue les lignes d'un fichier CSV")
    csv_command.add_argument("--csv-path", dest="csv_path", required=True, metavar="PATH", help="CSV file path")
    csv_command.add_argument("--text-col", dest="text_col", default="text", metavar="NAME")
    csv_command.add_argument("--subject-col", dest="subject_col", default="subject", metavar="NAME")
    csv_command.add_argument("--label-col", dest="label_col", default="y", metavar="NAME")
    csv_command.add_argument("--from-col", dest="from_col", default="from", metavar="NAME")
    csv_command.add_argument("--to-col", dest="to_col", default="to", metavar="NAME")
    csv_command.add_argument("--label-map", dest="label_map", default="0:ham,1:spam", metavar="MAPPING")
    csv_command.add_argument("--interval", dest="interval", type=float, default=1, metavar="SECONDS", help="Seconds between sends")
    csv_command.add_argument("--limit", dest="limit", type=int, default=0, metavar="NUMBER", help="Maximum number of rows")
    csv_command.add_argument("--shuffle", dest="shuffle", action="store_true", help="Shuffle rows before sending")
    csv_command.add_argument("--seed", dest="seed", type=int, default=42, metavar="VALUE")
    csv_command.add_argument("--skip-empty", dest="skip_empty", action="store_true")
    csv_command.add_argument("--dry-run", dest="dry_run", action="store_true", help="Preview without sending")
    add_connection_options(csv_command)
    csv_command.set_defaults(handler=command_csv)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
