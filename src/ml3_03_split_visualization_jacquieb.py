"""Визуализация разбиений train/test и train/valid/test для проекта ML3."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap


def visualize_train_test_split(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    ) -> None:
    """Строит boxplot и scatterplot для разбиения df_train/df_test."""
    df_train_tmp = df_train[["price"]].copy()
    df_train_tmp["set"] = "Train"
    
    df_test_tmp = df_test[["price"]].copy()
    df_test_tmp["set"] = "Test"
    
    combined = pd.concat([df_train_tmp, df_test_tmp])
    
    sns.set_style("dark")
    sns.relplot(height=3.38, aspect=3.38)
    sns.boxplot(x="set", y="price", data=combined, showmeans=True)
    plt.xlabel("")
    plt.show()
    
    print(
        "Размер df_train = ",
        df_train.shape,
        "\t" * 18,
        "Размер df_test =",
        df_test.shape,
        )
    sns.set_style("dark")
    sns.relplot(height=3.38, aspect=3.38)
    
    # sns.scatterplot(df_train, y="price", x="created", color="red", alpha=0.8)
    # sns.scatterplot(df_test, y="price", x="created", color="green", alpha=0.3)
    
    # 1. Добавляем label и явно указываем data=
    sns.scatterplot(data=df_train, y="price", x="created", color="red", alpha=0.8, label="Train")
    sns.scatterplot(data=df_test, y="price", x="created", color="green", alpha=0.3, label="Test")
    
    # 2. Вызываем легенду
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()


def visualize_train_valid_test_split(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    ) -> None:
    """Строит boxplot и scatterplot для разбиения df_train/df_valid/df_test."""
    df_train_tmp = df_train[["price"]].copy()
    df_train_tmp["set"] = "Train"
    
    df_valid_tmp = df_valid[["price"]].copy()
    df_valid_tmp["set"] = "Valid"
    
    df_test_tmp = df_test[["price"]].copy()
    df_test_tmp["set"] = "Test"
    
    combined = pd.concat([df_train_tmp, df_valid_tmp, df_test_tmp])
    
    sns.set_style("dark")
    sns.relplot(height=3.38, aspect=3.38)
    sns.boxplot(x="set", y="price", data=combined, showmeans=True)
    plt.xlabel("")
    plt.show()
    print(
        "Размер df_train = ",
        df_train.shape,
        "\t" * 8,
        "Размер df_valid =",
        df_valid.shape,
        "\t" * 8,
        "Размер df_test =",
        df_test.shape,
        )
    
    sns.set_style("dark")
    sns.relplot(height=3.38, aspect=3.38)
    
    # sns.scatterplot(df_train, y="price", x="created", color="red", alpha=0.8)
    # sns.scatterplot(df_valid, y="price", x="created", color="blue", alpha=0.8)
    # sns.scatterplot(df_test, y="price", x="created", color="green", alpha=0.3)
    
    # Добавляем label для каждой выборки
    sns.scatterplot(data=df_train, y="price", x="created", color="red", alpha=0.8, label="Train")
    sns.scatterplot(data=df_valid, y="price", x="created", color="blue", alpha=0.8, label="Valid")
    sns.scatterplot(data=df_test, y="price", x="created", color="green", alpha=0.3, label="Test")
    
    # Вызываем легенду
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()


def visualize_k_fold_split(
    df: pd.DataFrame,
    splits: list[tuple],
    ) -> None:
    """Строит heatmap индексов и boxplot price для K-Fold разбиения."""
    # ---------- Данные для heatmap ----------
    rows = []
    
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        rows.extend([{'fold': fold, 'index': idx, 'set': 'Train'} for idx in train_idx])
        
        rows.extend([{'fold': fold, 'index': idx, 'set': 'Test'} for idx in test_idx])
    
    heatmap_df = pd.DataFrame(rows)
    
    heatmap_df['value'] = heatmap_df['set'].map({'Train': 0, 'Test': 1})
    
    heatmap_matrix = heatmap_df.pivot(index='fold', columns='index', values='value')
    
    # ---------- Данные для boxplot ----------
    boxplot_parts = []
    
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        df_train = df.iloc[train_idx][['price']].copy()
        df_train['fold'] = fold
        df_train['set'] = 'Train'
        
        df_test = df.iloc[test_idx][['price']].copy()
        df_test['fold'] = fold
        df_test['set'] = 'Test'
        
        boxplot_parts.append(df_train)
        boxplot_parts.append(df_test)
    
    boxplot_df = pd.concat(boxplot_parts)
    
    # ---------- Общее полотно ----------
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 9), gridspec_kw={'height_ratios': [2, 1]})
    
    # График 1: heatmap
    sns.heatmap(heatmap_matrix, cmap=ListedColormap(['orange', 'green']), cbar=False, ax=axes[0])
    
    axes[0].set_title('K-Fold: Train/Test разбиение по индексам')
    axes[0].set_xlabel('Индекс строки')
    axes[0].set_ylabel('Fold')
    
    # График 2: boxplot
    sns.boxplot(data=boxplot_df, x='fold', y='price', hue='set', showmeans=True, ax=axes[1])
    
    axes[1].set_title('Распределение price в Train/Test по фолдам')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('price')
    
    plt.tight_layout()
    plt.show()
    
