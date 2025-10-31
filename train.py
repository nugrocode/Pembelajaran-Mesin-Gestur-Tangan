"""train.py

Latih model RandomForest (scikit-learn) terhadap CSV dataset yang dihasilkan oleh data_collection.py
Langkah:
 - Baca CSV
 - Preprocessing: normalisasi relatif terhadap pergelangan (landmark 0), dan scaling
 - Bagi train/val 80/20
 - Latih RandomForest, cetak akurasi, simpan model.pkl

Contoh:
 python train.py --input dataset.csv --model model.pkl
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import os

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def preprocess_row(row_values):
    """Normalisasi; row_values berupa list 63 (x,y,z)*21"""
    coords = np.array(row_values).reshape((21,3))  # (21,3)
    # Normalisasi: set origin ke landmark 0 (pergelangan)
    origin = coords[0].copy()
    coords[:, :2] = coords[:, :2] - origin[:2]
    # scale: bagi dengan max distance (euclidean) di xy untuk invarian skala
    dists = np.linalg.norm(coords[:, :2], axis=1)
    maxd = dists.max()
    if maxd == 0:
        maxd = 1.0
    coords[:, :2] = coords[:, :2] / maxd
    # flatten
    return coords.flatten().tolist()

def preprocess_df(df):
    X = []
    y = []
    for _, row in df.iterrows():
        values = row.iloc[:63].tolist()
        if any(pd.isna(values)):
            continue
        X.append(preprocess_row(values))
        y.append(str(row['label']))
    return np.array(X), np.array(y)

def plot_confusion(cm, labels, outpath='confusion.png'):
    """Plot confusion matrix dan simpan ke file."""
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)  # gunakan mappable dari imshow
    
    # Label sumbu
    ax.set_title('Confusion Matrix')
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Tampilkan nilai di tiap sel
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print('✅ Saved confusion matrix to', outpath)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='dataset.csv', help='CSV dataset')
    parser.add_argument('--model', '-m', default='model.pkl', help='Output model file (joblib)')
    parser.add_argument('--n_estimators', type=int, default=200, help='n_estimators for RandomForest')
    args = parser.parse_args()

    df = load_dataset(args.input)
    print('Loaded', len(df), 'rows')

    X, y = preprocess_df(df)
    print('After preprocessing -> X:', X.shape, 'y:', y.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators, n_jobs=-1, random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print('Validation accuracy:', acc)
    print('Classification report:\n', classification_report(y_val, y_pred))

    cm = confusion_matrix(y_val, y_pred, labels=np.unique(y))
    plot_confusion(cm, labels=np.unique(y), outpath='confusion.png')

    joblib.dump(clf, args.model)
    print('✅ Saved model to', args.model)


if __name__ == '__main__':
    main()
