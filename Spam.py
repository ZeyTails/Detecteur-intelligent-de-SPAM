import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB


RANDOM_STATE = 42
TEST_SIZE = 0.25
VAL_SIZE = 0.20
SELECTION_BETA = 2.0  # favorise le rappel spam

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


# ---- 1) Charger tes 2 datasets ----
sa_path = RAW_DATA_DIR / "spam_assassin.csv"
en_path = RAW_DATA_DIR / "spam_ham_dataset.csv"
if not sa_path.exists():
    sa_path = BASE_DIR / "spam_assassin.csv"
if not en_path.exists():
    en_path = BASE_DIR / "spam_ham_dataset.csv"

df_sa = pd.read_csv(sa_path)          # text, target (0/1)
df_en = pd.read_csv(en_path)          # label, text, label_num, Unnamed: 0

# Unifier le format : garder seulement text + y(0/1)
df_sa = df_sa.rename(columns={"target": "y"})[["text", "y"]]
df_en = df_en.rename(columns={"label_num": "y"})[["text", "y"]]

# Fusion
df = pd.concat([df_sa, df_en], ignore_index=True)
df = df.dropna(subset=["text", "y"]).copy()
df["y"] = df["y"].astype(int)


# ---- 2) Nettoyage ----
def clean_email(text: str, keep_urls: bool = False) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()

    t = re.sub(
        r"^(from|to|subject|cc|bcc|date|received|return-path|message-id):.*$",
        " ",
        t,
        flags=re.MULTILINE,
    )

    t = re.sub(r"<[^>]+>", " ", t)

    if not keep_urls:
        t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", t)

    t = re.sub(r"[^a-z\s]", " ", t)

    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---- 3) Tokenization ----
def tokenize(text: str):
    return text.split()


# ---- 4) Stemming / Lemmatization ----
def setup_nltk():
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


def lemmatize_tokens(tokens):
    from nltk.stem import WordNetLemmatizer
    lem = WordNetLemmatizer()
    return [lem.lemmatize(w) for w in tokens]


# ---- 5) Appliquer le preprocessing ----
df["clean"] = df["text"].apply(lambda s: clean_email(s, keep_urls=False))
df["tokens"] = df["clean"].apply(tokenize)

setup_nltk()
df["lemmatized"] = df["tokens"].apply(lemmatize_tokens)

# Convertir la liste de tokens lemmatises en texte
df["norm_text"] = df["lemmatized"].apply(lambda toks: " ".join(toks))
df = df[df["norm_text"].str.len() > 0].copy()

print("Distribution classes:", df["y"].value_counts().to_dict())
print(df[["y", "text", "clean", "tokens", "norm_text"]].head(6))

# Sauvegarder le resultat preprocess
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
processed_path = PROCESSED_DIR / "spam_processed.csv"
df.to_csv(processed_path, index=False)
print(f"\nResultat sauvegarde dans '{processed_path}'")


# ---- 6) Split train/validation/test ----
# Important: split AVANT fit TF-IDF pour eviter la fuite de donnees.
X_text = df["norm_text"]
y = df["y"]

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_train_text,
    y_train,
    test_size=VAL_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_train,
)


# ---- 7) TF-IDF ----
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents="unicode",
)

X_train = tfidf.fit_transform(X_train_text)
X_val = tfidf.transform(X_val_text)
X_test = tfidf.transform(X_test_text)

print("TF-IDF matrix shape (train):", X_train.shape)
print("Exemples de features TF-IDF:", tfidf.get_feature_names_out()[:40])


# ---- 8) Entrainement + tuning leger (NB + seuil) ----
# On garde l'esprit de la pipeline de base:
# - Naive Bayes
# - evaluation par matrice + rapport
# Ici on teste juste quelques reglages pour mieux detecter spam.
candidate_models = {
    "MultinomialNB": MultinomialNB,
    "ComplementNB": ComplementNB,
}
candidate_alphas = [0.1, 0.3, 0.5, 1.0, 2.0]
candidate_thresholds = np.arange(0.20, 0.81, 0.05)

best_score = -1.0
best_model_name = "MultinomialNB"
best_model_class = MultinomialNB
best_alpha = 1.0
best_threshold = 0.50

for model_name, model_class in candidate_models.items():
    for alpha in candidate_alphas:
        model = model_class(alpha=alpha)
        model.fit(X_train, y_train)

        val_proba_spam = model.predict_proba(X_val)[:, 1]
        for threshold in candidate_thresholds:
            y_val_pred = (val_proba_spam >= threshold).astype(int)
            score = fbeta_score(y_val, y_val_pred, beta=SELECTION_BETA, zero_division=0)
            if score > best_score:
                best_score = score
                best_model_name = model_name
                best_model_class = model_class
                best_alpha = alpha
                best_threshold = float(threshold)

print(
    "\nMeilleurs hyperparametres:",
    {
        "model": best_model_name,
        "alpha": best_alpha,
        "spam_threshold": round(best_threshold, 3),
        f"f{SELECTION_BETA:.0f}_val": round(best_score, 4),
    },
)

# Entrainement final sur train+validation
X_train_full = vstack([X_train, X_val])
y_train_full = np.concatenate([y_train.to_numpy(), y_val.to_numpy()])

nb = best_model_class(alpha=best_alpha)
nb.fit(X_train_full, y_train_full)

# Predictions test
proba_test_spam = nb.predict_proba(X_test)[:, 1]
y_pred_default = (proba_test_spam >= 0.50).astype(int)
y_pred_tuned = (proba_test_spam >= best_threshold).astype(int)

print("\nMatrice de confusion (seuil 0.50):\n", confusion_matrix(y_test, y_pred_default))
print(
    "\nRapport de classification (seuil 0.50):\n",
    classification_report(y_test, y_pred_default, target_names=["ham", "spam"]),
)

print(f"\nMatrice de confusion (seuil optimise {best_threshold:.2f}):\n", confusion_matrix(y_test, y_pred_tuned))
print(
    f"\nRapport de classification (seuil optimise {best_threshold:.2f}):\n",
    classification_report(y_test, y_pred_tuned, target_names=["ham", "spam"]),
)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODELS_DIR / "spam_tfidf_nb.joblib"
joblib.dump(
    {
        "tfidf": tfidf,
        "nb": nb,
        "spam_threshold": best_threshold,
        "model_name": best_model_name,
        "alpha": best_alpha,
        "selection_metric": f"fbeta(beta={SELECTION_BETA}) on validation",
    },
    model_path,
)
print(f"\nModele sauvegarde dans : {model_path}")
