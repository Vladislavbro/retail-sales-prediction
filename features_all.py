import pandas as pd
import numpy as np


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Временные фичи на основе колонки 'dt'."""
    df = df.copy()
    dt = pd.to_datetime(df["dt"])

    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = dt.dt.month
    df["day_of_month"] = dt.dt.day
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)

    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"] = dt.dt.is_month_end.astype(int)

    df["days_since_start"] = (dt - dt.min()).dt.days

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_price_features(df: pd.DataFrame, windows: list[int] = [7, 14, 30]) -> pd.DataFrame:
    """Ценовые фичи на основе 'price'."""
    df = df.copy()
    df = df.sort_values(["nm_id", "dt"])
    g = df.groupby("nm_id")["price"]

    # Изменение цены
    df["price_diff"] = g.diff()
    df["price_pct_change"] = g.pct_change()

    # Скользящие статистики
    for w in windows:
        df[f"price_rmean_{w}d"] = g.transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f"price_rmin_{w}d"] = g.transform(lambda x: x.rolling(w, min_periods=1).min())
        df[f"price_rmax_{w}d"] = g.transform(lambda x: x.rolling(w, min_periods=1).max())

    # Отношение текущей цены к скользящей средней
    for w in windows:
        df[f"price_vs_rmean_{w}d"] = df["price"] / df[f"price_rmean_{w}d"]

    # Глубина скидки от исторического максимума
    df["price_cummax"] = g.transform("cummax")
    df["discount_depth"] = (df["price_cummax"] - df["price"]) / df["price_cummax"]
    df.drop(columns="price_cummax", inplace=True)

    # Перцентиль цены в expanding-окне
    df["price_rank_pct"] = g.transform(lambda x: x.expanding().rank(pct=True))

    # Дней с последнего изменения цены
    df["_pc"] = (g.diff().fillna(0) != 0).astype(int)
    df["days_since_price_change"] = (
        df.groupby("nm_id")["_pc"]
          .transform(lambda x: x.groupby(x.cumsum()).cumcount())
    )
    df.drop(columns="_pc", inplace=True)

    return df


def add_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Промо-фичи на основе 'is_promo'."""
    df = df.copy()
    df = df.sort_values(["nm_id", "dt"])
    g = df.groupby("nm_id")["is_promo"]

    # Сколько дней подряд идёт промо
    df["_not_promo"] = (df["is_promo"] == 0).astype(int)
    df["promo_days_in_row"] = (
        g.transform(lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
        * df["is_promo"]
    )
    df.drop(columns="_not_promo", inplace=True)

    # Дней с последнего промо
    df["_promo_ever"] = g.transform("cumsum")
    df["_promo_cumcount"] = g.transform(lambda x: x.groupby(x.cumsum()).cumcount())
    df["days_since_last_promo"] = df["_promo_cumcount"].where(df["_promo_ever"] > 0, np.nan)
    df.drop(columns=["_promo_ever", "_promo_cumcount"], inplace=True)

    # Доля промо-дней за последние 30 дней
    df["promo_freq_30d"] = g.transform(lambda x: x.rolling(30, min_periods=1).mean())

    # Начало / конец промо-периода
    promo_shifted = g.shift(fill_value=0)
    promo_shifted_fwd = g.shift(-1, fill_value=0)
    df["is_promo_start"] = ((df["is_promo"] == 1) & (promo_shifted == 0)).astype(int)
    df["is_promo_end"] = ((df["is_promo"] == 1) & (promo_shifted_fwd == 0)).astype(int)

    # Взаимодействие: промо × глубина скидки (если discount_depth уже есть)
    if "discount_depth" in df.columns:
        df["promo_x_discount"] = df["is_promo"] * df["discount_depth"]

    return df


def add_leftovers_features(df: pd.DataFrame, windows: list[int] = [7, 14]) -> pd.DataFrame:
    """Фичи по остаткам на складе ('prev_leftovers')."""
    df = df.copy()
    df = df.sort_values(["nm_id", "dt"])
    g = df.groupby("nm_id")["prev_leftovers"]

    # Скользящее среднее остатков
    for w in windows:
        df[f"left_rmean_{w}d"] = g.transform(lambda x: x.rolling(w, min_periods=1).mean())

    # Изменение остатков
    df["left_diff"] = g.diff()

    # Тренд остатков за 7 дней (наклон линейной регрессии)
    def rolling_slope(s, window=7):
        x = np.arange(window, dtype=float)
        x -= x.mean()
        result = np.full(len(s), np.nan)
        vals = s.values
        for i in range(window - 1, len(vals)):
            y = vals[i - window + 1 : i + 1].astype(float)
            if np.isnan(y).any():
                continue
            y -= y.mean()
            result[i] = np.dot(x, y) / np.dot(x, x)
        return pd.Series(result, index=s.index)

    df["left_trend_7d"] = df.groupby("nm_id")["prev_leftovers"].transform(rolling_slope)

    # Флаг низкого остатка (ниже 10-го перцентиля товара)
    q10 = df.groupby("nm_id")["prev_leftovers"].transform(
        lambda x: x.expanding().quantile(0.1)
    )
    df["is_low_stock"] = (df["prev_leftovers"] <= q10).astype(int)

    return df


def add_sales_lag_features(
    df: pd.DataFrame,
    lags: list[int] = [1, 7, 14, 28],
    windows: list[int] = [7, 14, 30],
) -> pd.DataFrame:
    """
    Лаги и скользящие статистики по продажам ('qty').
    Использовать только на train или с правильным разделением train/test,
    чтобы не допустить утечки данных.
    """
    df = df.copy()
    df = df.sort_values(["nm_id", "dt"])
    g = df.groupby("nm_id")["qty"]

    # Лаги
    for lag in lags:
        df[f"qty_lag_{lag}d"] = g.shift(lag)

    # Лаг на тот же день недели (неделю назад)
    df["qty_same_dow_lag"] = g.shift(7)

    # Скользящие средние и std
    for w in windows:
        shifted = g.shift(1)  # сдвиг на 1, чтобы не подглядывать текущий день
        df[f"qty_rmean_{w}d"] = (
            df.groupby("nm_id")[shifted.name if hasattr(shifted, "name") else "qty"]
              .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
        df[f"qty_rstd_{w}d"] = (
            df.groupby("nm_id")["qty"]
              .transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())
        )

    # Expanding mean (до текущего дня)
    df["qty_expanding_mean"] = g.transform(lambda x: x.shift(1).expanding().mean())

    return df


def add_item_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегатные товарные фичи по nm_id.
    Считаются по expanding-окну, чтобы избежать утечки на train.
    """
    df = df.copy()
    df = df.sort_values(["nm_id", "dt"])

    # Средние продажи товара (expanding, со сдвигом)
    if "qty" in df.columns:
        df["item_mean_qty"] = (
            df.groupby("nm_id")["qty"]
              .transform(lambda x: x.shift(1).expanding().mean())
        )

    # Средняя цена товара
    df["item_mean_price"] = (
        df.groupby("nm_id")["price"]
          .transform(lambda x: x.expanding().mean())
    )

    # Волатильность цены товара
    df["item_price_std"] = (
        df.groupby("nm_id")["price"]
          .transform(lambda x: x.expanding().std())
    )

    # Сколько дней товар в данных
    df["item_lifetime_days"] = df.groupby("nm_id").cumcount() + 1

    return df


def add_elasticity_features(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Псевдо-эластичность: корреляция цены и продаж в скользящем окне.
    Требует наличия 'qty'.
    """
    df = df.copy()
    df = df.sort_values(["nm_id", "dt"])

    if "qty" not in df.columns:
        return df

    # Rolling корреляция цена <-> qty
    def rolling_corr(group, w):
        return group["price"].rolling(w, min_periods=w // 2).corr(group["qty"])

    df[f"price_qty_corr_{window}d"] = (
        df.groupby("nm_id", group_keys=False)
          .apply(lambda g: rolling_corr(g, window), include_groups=False)
    )

    # Псевдо-эластичность: % изменения qty / % изменения price
    df["price_pct"] = df.groupby("nm_id")["price"].pct_change()
    df["qty_pct"] = df.groupby("nm_id")["qty"].pct_change()
    df["pseudo_elasticity"] = df["qty_pct"] / df["price_pct"].replace(0, np.nan)
    # Клиппинг выбросов
    df["pseudo_elasticity"] = df["pseudo_elasticity"].clip(-10, 10)
    df.drop(columns=["price_pct", "qty_pct"], inplace=True)

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Взаимодействия между признаками."""
    df = df.copy()

    # price × is_promo
    df["price_x_promo"] = df["price"] * df["is_promo"]

    # day_of_week × is_promo
    if "day_of_week" in df.columns:
        df["dow_x_promo"] = df["day_of_week"] * df["is_promo"]

    # prev_leftovers × is_promo
    df["left_x_promo"] = df["prev_leftovers"] * df["is_promo"]

    return df


# ──────────────────────────────────────────────
# Мастер-функция: применяет все блоки по порядку
# ──────────────────────────────────────────────

def add_all_features(df: pd.DataFrame, has_qty: bool = True) -> pd.DataFrame:
    """
    Последовательно применяет все блоки фичей.

    Args:
        df: исходный датафрейм с колонками nm_id, dt, price, is_promo, prev_leftovers [, qty]
        has_qty: True для train (есть qty), False для test
    """
    df = add_temporal_features(df)
    df = add_price_features(df)
    df = add_promo_features(df)
    df = add_leftovers_features(df)

    if has_qty:
        df = add_sales_lag_features(df)
        df = add_item_features(df)
        df = add_elasticity_features(df)
    else:
        df = add_item_features(df)  # часть фичей всё равно посчитается без qty

    df = add_interaction_features(df)

    return df
