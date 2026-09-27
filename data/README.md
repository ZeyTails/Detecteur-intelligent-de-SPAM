# Données d'entraînement

Les jeux de données bruts utilisés pour entraîner le détecteur sont conservés
dans le dépôt afin que les utilisateurs puissent réentraîner le modèle après
un clonage. Les conditions de redistribution et la provenance des corpus
doivent toutefois être respectées avant toute publication sur un dépôt public.

## Fichiers nécessaires

Placer localement les deux fichiers suivants :

```text
data/raw/spam_assassin.csv
data/raw/spam_ham_dataset.csv
```

Le script `Spam.py` attend ces noms et ces emplacements. Les fichiers doivent
être obtenus depuis leurs sources autorisées en conservant leurs conditions de
réutilisation et leur provenance.

## Préparer les données

Depuis la racine du projet :

```bash
python Spam.py
```

Le script nettoie et fusionne les deux sources, puis génère localement :

```text
data/processed/spam_processed.csv
```

Ce fichier est une sortie générée. Il ne doit pas être modifié manuellement
ni ajouté au dépôt : il peut être recréé à tout moment avec `Spam.py`.

## Pourquoi le fichier traité n'est pas versionné ?

- le fichier préparé est beaucoup plus volumineux que les sources brutes ;
- il est entièrement reproductible à partir de `data/raw/` ;
- il contient des colonnes intermédiaires produites par le prétraitement ;
- le dépôt reste plus léger et se clone plus rapidement.

Les deux CSV de `data/raw/` et les petits CSV de démonstration de
`Mailpit/data/` peuvent donc être versionnés. Si un futur dataset dépasse
100 MiB par fichier, il faudra utiliser Git LFS ou une archive externe.
