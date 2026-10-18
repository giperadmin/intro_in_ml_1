"""Пользовательские разбиения кросс-валидации для проекта ML3.

Все функции возвращают позиционные индексы, предназначенные для ``DataFrame.iloc``:

    df_train = df.iloc[train_indices]
    df_test = df.iloc[test_indices]
"""

from __future__ import annotations

import numpy as np
import pandas as pd


RANDOM_STATE = 21


def _validate_dataframe(data: pd.DataFrame) -> None:
    """Проверяет, что на вход передан непустой pandas DataFrame."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data должен быть pandas DataFrame")
    if data.empty:
        raise ValueError("data не должен быть пустым")


def _validate_k(k: int, n_samples: int) -> None:
    """Проверяет, что k является корректным количеством фолдов."""
    if not isinstance(k, int):
        raise TypeError("k должен быть целым числом")
    if k < 2:
        raise ValueError("k должен быть не меньше 2")
    if k > n_samples:
        raise ValueError("k не может быть больше количества объектов")


def _validate_column(data: pd.DataFrame, column: str) -> None:
    """Проверяет, что нужная колонка есть в DataFrame."""
    if column not in data.columns:
        raise KeyError(f"В DataFrame нет колонки {column!r}")


def make_quantile_stratification_field(
    data: pd.DataFrame,
    target_field: str,
    n_bins: int = 5,
) -> pd.Series:
    """Создает дискретные квантили для стратификации непрерывной цели.

    Для регрессионной задачи нельзя напрямую использовать непрерывный target
    вроде ``price`` как ``stratify_field``: почти каждое значение может стать
    отдельным классом. Поэтому target сначала разбивается на интервалы по
    квантилям, а уже полученное поле можно передавать в
    ``make_stratified_k_fold_splits``.
    """
    _validate_dataframe(data)
    _validate_column(data, target_field)

    if not isinstance(n_bins, int):
        raise TypeError("n_bins должен быть целым числом")
    if n_bins < 2:
        raise ValueError("n_bins должен быть не меньше 2")

    target = data[target_field]

    if target.isna().any():
        raise ValueError(f"Поле {target_field!r} содержит пропуски")

    bins = pd.qcut(target, q=n_bins, labels=False, duplicates="drop")
    bins = bins.astype(int)
    bins.name = f"{target_field}_bin"

    if bins.nunique() < 2:
        raise ValueError(
            f"Не удалось разбить поле {target_field!r} минимум на 2 квантиля"
        )

    return bins


def make_k_fold_splits(
    data: pd.DataFrame,
    k: int,
    shuffle: bool = True,
    random_state: int = RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Возвращает позиционные индексы train/test для K-Fold.

    Параметры
    ---------
    data:
        Исходный DataFrame.
    k:
        Количество фолдов.
    shuffle:
        Нужно ли перемешивать позиции строк перед разбиением.
    random_state:
        Зерно генератора случайных чисел для детерминированного перемешивания.
    """
    _validate_dataframe(data)
    _validate_k(k, len(data))

    indices = np.arange(len(data))

    if shuffle:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(indices)

    fold_indices = np.array_split(indices, k)
    splits = []

    for fold_number in range(k):
        test_indices = fold_indices[fold_number]
        train_indices = np.concatenate(
            [fold for i, fold in enumerate(fold_indices) if i != fold_number]
        )
        splits.append((train_indices, test_indices))

    return splits


def make_group_k_fold_splits(
    data: pd.DataFrame,
    k: int,
    group_field: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Возвращает позиционные индексы train/test для Grouped K-Fold.

    Объекты с одинаковым значением ``group_field`` никогда не разделяются
    между train и test внутри одного фолда.
    """
    _validate_dataframe(data)
    _validate_k(k, len(data))
    _validate_column(data, group_field)

    groups = data[group_field].to_numpy()
    unique_groups, group_counts = np.unique(groups, return_counts=True)

    if k > len(unique_groups):
        raise ValueError("k не может быть больше количества уникальных групп")

    # Сначала раскладываем большие группы, чтобы фолды были ближе по числу строк.
    sorted_group_order = np.argsort(group_counts)[::-1]
    sorted_groups = unique_groups[sorted_group_order]

    fold_groups = [[] for _ in range(k)]
    fold_sizes = np.zeros(k, dtype=int)

    for group in sorted_groups:
        group_size = np.sum(groups == group)
        target_fold = int(np.argmin(fold_sizes))
        fold_groups[target_fold].append(group)
        fold_sizes[target_fold] += group_size

    all_indices = np.arange(len(data))
    splits = []

    for groups_in_fold in fold_groups:
        test_mask = np.isin(groups, groups_in_fold)
        test_indices = all_indices[test_mask]
        train_indices = all_indices[~test_mask]
        splits.append((train_indices, test_indices))

    return splits


def make_stratified_k_fold_splits(
    data: pd.DataFrame,
    k: int,
    stratify_field: str,
    shuffle: bool = True,
    random_state: int = RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Возвращает позиционные индексы train/test для Stratified K-Fold.

    Распределение ``stratify_field`` сохраняется примерно одинаковым
    в каждом test-фолде. Для регрессионной цели, например ``price``, сначала
    создайте дискретные бины через ``make_quantile_stratification_field``.
    """
    _validate_dataframe(data)
    _validate_k(k, len(data))
    _validate_column(data, stratify_field)

    values = data[stratify_field].to_numpy()
    all_indices = np.arange(len(data))
    rng = np.random.default_rng(random_state)

    fold_parts = [[] for _ in range(k)]

    for class_value in np.unique(values):
        class_indices = all_indices[values == class_value]

        if len(class_indices) < k:
            raise ValueError(
                f"Класс {class_value!r} содержит {len(class_indices)} объектов, "
                f"что меньше k={k}"
            )

        if shuffle:
            class_indices = rng.permutation(class_indices)

        class_folds = np.array_split(class_indices, k)

        for fold_number, class_fold in enumerate(class_folds):
            fold_parts[fold_number].append(class_fold)

    splits = []

    for fold_number in range(k):
        test_indices = np.concatenate(fold_parts[fold_number])
        train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=False)
        splits.append((train_indices, test_indices))

    return splits


def make_time_series_splits(
    data: pd.DataFrame,
    k: int,
    date_field: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Возвращает позиционные индексы train/test для time series split.

    Данные сортируются по ``date_field``. При ``k`` временных блоках функция
    возвращает ``k - 1`` разбиений:

    - разбиение 1: блок 1 как train, блок 2 как test;
    - разбиение 2: блоки 1-2 как train, блок 3 как test;
    - и так далее.
    """
    _validate_dataframe(data)
    _validate_k(k, len(data))
    _validate_column(data, date_field)

    dates = pd.to_datetime(data[date_field])
    sorted_positions = np.argsort(dates.to_numpy())
    time_blocks = np.array_split(sorted_positions, k)

    splits = []

    for test_block_number in range(1, k):
        train_indices = np.concatenate(time_blocks[:test_block_number])
        test_indices = time_blocks[test_block_number]
        splits.append((train_indices, test_indices))

    return splits
