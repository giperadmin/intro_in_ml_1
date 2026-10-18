"""Методы разбиения данных для задания 3 проекта ML3.

Функции возвращают готовые выборки ``DataFrame``:

    df_train, df_test = split_random_train_test(df, test_size=0.2)

По умолчанию во всех функциях задан ``random_state=21``. В разбиениях по дате
этот параметр оставлен для единого интерфейса, но не влияет на результат:
разбиение полностью определяется датами.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 21


def _validate_dataframe(data: pd.DataFrame) -> None:
    """Проверяет, что на вход передан непустой pandas DataFrame."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data должен быть pandas DataFrame")
    if data.empty:
        raise ValueError("data не должен быть пустым")


def _validate_ratio(name: str, value: float) -> None:
    """Проверяет, что параметр является долей строго между 0 и 1."""
    if not 0 < value < 1:
        raise ValueError(f"{name} должен быть долей строго между 0 и 1")


def _validate_column(data: pd.DataFrame, column: str) -> None:
    """Проверяет, что нужная колонка есть в DataFrame."""
    if column not in data.columns:
        raise KeyError(f"В DataFrame нет колонки {column!r}")


def split_random_train_test(
    data: pd.DataFrame,
    test_size: float,
    random_state: int = RANDOM_STATE,
    shuffle: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Случайно делит данные на train и test.

    Parameters
    ----------
    data:
        Исходный DataFrame.
    test_size:
        Доля объектов, которые попадут в test. Значение должно быть от 0 до 1.
    random_state:
        Зерно генератора случайных чисел для воспроизводимого разбиения.
    shuffle:
        Нужно ли перемешивать строки перед разбиением.
    """
    _validate_dataframe(data)
    _validate_ratio("test_size", test_size)
    
    df_train, df_test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        )
    
    return df_train.copy(), df_test.copy()


def split_random_train_validation_test(
    data: pd.DataFrame,
    validation_size: float,
    test_size: float,
    random_state: int = RANDOM_STATE,
    shuffle: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Случайно делит данные на train, validation и test.

    ``validation_size`` и ``test_size`` задаются как доли от полного датасета.
    Например, ``validation_size=0.2`` и ``test_size=0.2`` дают примерно
    60/20/20 для train/validation/test.
    """
    _validate_dataframe(data)
    _validate_ratio("validation_size", validation_size)
    _validate_ratio("test_size", test_size)
    
    if validation_size + test_size >= 1:
        raise ValueError("Сумма validation_size и test_size должна быть меньше 1")
    
    df_train_validation, df_test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        )
    
    adjusted_validation_size = validation_size / (1 - test_size)
    
    df_train, df_validation = train_test_split(
        df_train_validation,
        test_size=adjusted_validation_size,
        random_state=random_state,
        shuffle=shuffle,
        )
    
    return df_train.copy(), df_validation.copy(), df_test.copy()


def split_by_date_train_test(
    data: pd.DataFrame,
    date_split: str | pd.Timestamp,
    date_field: str = "created",
    random_state: int = RANDOM_STATE,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит данные на train и test по параметру date_split.

    В train попадают строки, где ``date_field < date_split``.
    В test попадают строки, где ``date_field >= date_split``.

    ``random_state`` не используется, потому что разбиение по дате
    детерминировано само по себе.
    """
    _validate_dataframe(data)
    _validate_column(data, date_field)
    
    prepared = data.copy()
    prepared[date_field] = pd.to_datetime(prepared[date_field])
    date_split = pd.to_datetime(date_split)
    
    df_train = prepared[prepared[date_field] < date_split].copy()
    df_test = prepared[prepared[date_field] >= date_split].copy()
    
    return df_train, df_test


def split_by_dates_train_validation_test(
    data: pd.DataFrame,
    validation_date: str | pd.Timestamp,
    test_date: str | pd.Timestamp,
    date_field: str = "created",
    random_state: int = RANDOM_STATE,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Делит данные на train, validation и test по двум датам.

    В train попадают строки, где ``date_field < validation_date``.
    В validation попадают строки, где
    ``validation_date <= date_field < test_date``.
    В test попадают строки, где ``date_field >= test_date``.

    ``random_state`` не используется, потому что разбиение по дате
    детерминировано само по себе.
    """
    _validate_dataframe(data)
    _validate_column(data, date_field)
    
    validation_date = pd.to_datetime(validation_date)
    test_date = pd.to_datetime(test_date)
    
    if validation_date >= test_date:
        raise ValueError("validation_date должна быть раньше test_date")
    
    prepared = data.copy()
    prepared[date_field] = pd.to_datetime(prepared[date_field])
    
    df_train = prepared[prepared[date_field] < validation_date].copy()
    df_validation = prepared[
        (prepared[date_field] >= validation_date)
        & (prepared[date_field] < test_date)
        ].copy()
    df_test = prepared[prepared[date_field] >= test_date].copy()
    
    return df_train, df_validation, df_test
