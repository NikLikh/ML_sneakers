import os
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.exceptions import ConvergenceWarning

RANDOM_STATE = 42

class TSFeatures:
    """Генератор временных признаков для задачи продаж.

    Ключевые идеи:
    - строим лаги и скользящие статистики по каждому товару,
    - добавляем календарные и рыночные признаки,
    - оцениваем спорадический спрос (Кростон / Виллемейн),
    - добавляем кластерные признаки на уровне товаров.
    """

    REQUIRED_COLUMNS = ("nm_id", "dt", "price", "qty", "is_promo", "prev_leftovers")
    ITEM_STAT_COLUMNS = (
        "nm_qty_mean",
        "nm_qty_std",
        "nm_price_mean",
        "nm_price_std",
        "nm_zero_ratio_mean",
        "nm_e_mean",
        "nm_stockout_mean",
        "nm_has_promo",
    )

    def __init__(
        self,
        horizon= 14,
        lags = (14, 28),
        windows = (14, 28),
        n_clusters = 20,
        random_state = RANDOM_STATE,
    ):
        """Инициализирует конфигурацию генератора.

        Параметры:
        - horizon: горизонт прогноза (сколько дней вперёд).
        - lags: набор лагов по qty и price.
        - windows: окна скользящих статистик.
        - n_clusters: количество кластеров товаров.
        """
        self.horizon = int(horizon)
        self.lags = tuple(sorted({int(v) for v in lags}))
        self.windows = tuple(sorted({int(v) for v in windows}))
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)

        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        if not self.lags:
            raise ValueError("lags must not be empty")
        if not self.windows:
            raise ValueError("windows must not be empty")

        # Параметры для спорадического спроса
        self.croston_alpha = 0.15
        self.willemain_window = 56

        # Служебные буферы, заполняются при fit
        self.train_ = None
        self.train_features_ = None
        self.base_feature_columns_ = None
        self.feature_columns_ = None

        self.item_stats_ = None
        self.cluster_map_ = None
        self.cluster_price_map_ = None
        self.global_cluster_price_ = None

        self.scaler_ = None
        self.pca_ = None
        self.kmeans_ = None

    def _check_is_fitted(self) -> None:
        """Проверяет, что fit был вызван."""
        if self.train_ is None or self.item_stats_ is None or self.feature_columns_ is None:
            raise RuntimeError("TSFeatures is not fitted. Call fit() first.")

    @staticmethod
    def cols_check(df: pd.DataFrame, cols: tuple[str, ...] | list[str]) -> pd.DataFrame:
        """Гарантирует наличие колонок в датафрейме.

        Если колонка отсутствует — добавляется с NaN.
        """
        out = df.copy()
        for col in cols:
            if col not in out.columns:
                out[col] = np.nan
        return out

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготавливает данные: добавляет недостающие колонки и сортирует."""
        out = self.cols_check(df, self.REQUIRED_COLUMNS)
        out["dt"] = pd.to_datetime(out["dt"])
        out = out.sort_values(["nm_id", "dt"], kind="mergesort").reset_index(drop=True)
        return out

    @staticmethod
    def roll_by_group(values: pd.Series, key: pd.Series, window: int, func: str = "mean") -> pd.Series:
        """Скользящая агрегация по группам.

        values — серия значений,
        key — группирующий ключ (nm_id),
        window — длина окна.
        """
        return (
            values.groupby(key)
            .rolling(window=int(window), min_periods=2)
            .agg(func)
            .reset_index(0, drop=True)
        )

    @staticmethod
    def _safe_div(a, b, default: float = 0.0, clip_range: tuple[float, float] | None = None):
        """Безопасное деление с защитой от нулей и inf/NaN."""
        result = np.where(np.abs(b) < 1e-9, default, a / b)
        result = np.nan_to_num(result, nan=default, posinf=default, neginf=default)
        if clip_range is not None:
            result = np.clip(result, *clip_range)
        return result

    @staticmethod
    def _croston_sba_series(y, alpha: float = 0.15) -> np.ndarray:
        """Реализация Croston SBA по временной серии.

        Возвращает сглаженный прогноз для спорадического спроса.
        """
        y = np.asarray(y, dtype=float)
        y = np.clip(np.nan_to_num(y, nan=0.0), 0, None)

        n = len(y)
        out = np.zeros(n, dtype=np.float32)

        z = 0.0
        p = 1.0
        q = 1
        started = False

        for t in range(n):
            yt = y[t]
            if not started:
                if yt > 0:
                    z = yt
                    p = max(1.0, float(q))
                    started = True
                    out[t] = (1.0 - alpha / 2.0) * (z / max(p, 1e-6))
                    q = 1
                else:
                    out[t] = 0.0
                    q += 1
                continue

            out[t] = (1.0 - alpha / 2.0) * (z / max(p, 1e-6))
            if yt > 0:
                z = z + alpha * (yt - z)
                p = p + alpha * (q - p)
                q = 1
            else:
                q += 1

        return out

    @staticmethod
    def _sanitize_numeric(df: pd.DataFrame) -> pd.DataFrame:
        """Заменяет inf/NaN в числовых колонках на 0."""
        out = df.copy()
        num_cols = out.select_dtypes(include=[np.number]).columns
        out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    @staticmethod
    def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
        """Возвращает первую колонку из списка candidates, которая существует."""
        for col in candidates:
            if col in columns:
                return col
        return None

    def _resolve_reference_window(self, preferred: int = 28) -> int:
        """Выбирает окно для расчёта reference‑фичей (CV и т.п.)."""
        return preferred if preferred in self.windows else self.windows[-1]

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Генерирует все временные признаки без агрегации по товарам/кластеров."""
        out = df.copy()
        g = out.groupby("nm_id", sort=False)

        # --- Календарные признаки ---
        out["year"] = out["dt"].dt.year
        out["month"] = out["dt"].dt.month
        out["day"] = out["dt"].dt.day
        out["dayofweek"] = out["dt"].dt.dayofweek
        out["weekofyear"] = out["dt"].dt.isocalendar().week.astype(int)
        out["quarter"] = out["dt"].dt.quarter
        out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)

        # --- История по товару (с лагом горизонта) ---
        qty_hist = g["qty"].shift(self.horizon)
        price_hist = g["price"].shift(1).fillna(0)
        promo_hist = g["is_promo"].shift(1).fillna(0)
        left_hist = g["prev_leftovers"].shift(1)

        # --- Лаги ---
        for lag in self.lags:
            out[f"qty_lag_{lag}"] = g["qty"].shift(lag)
            out[f"price_lag_{lag}"] = g["price"].shift(lag)

        # --- Скользящие статистики ---
        for w in self.windows:
            out[f"qty_roll_mean_{w}"] = self.roll_by_group(qty_hist, out["nm_id"], w, "mean")
            out[f"qty_roll_std_{w}"] = self.roll_by_group(qty_hist, out["nm_id"], w, "std")
            out[f"qty_roll_min_{w}"] = self.roll_by_group(qty_hist, out["nm_id"], w, "min")
            out[f"qty_roll_max_{w}"] = self.roll_by_group(qty_hist, out["nm_id"], w, "max")

            out[f"price_roll_mean_{w}"] = self.roll_by_group(price_hist, out["nm_id"], w, "mean")
            out[f"price_roll_std_{w}"] = self.roll_by_group(price_hist, out["nm_id"], w, "std")
            out[f"price_roll_min_{w}"] = self.roll_by_group(price_hist, out["nm_id"], w, "min")
            out[f"price_roll_max_{w}"] = self.roll_by_group(price_hist, out["nm_id"], w, "max")

            out[f"promo_days_{w}"] = self.roll_by_group(promo_hist, out["nm_id"], w, "sum")

        # --- Динамика цены ---
        out["price_change_1"] = price_hist - g["price"].shift(2)
        out["price_pct_change_1"] = self._safe_div(
            out["price_change_1"],
            g["price"].shift(2).abs() + 1e-2,
            clip_range=(-10, 10),
        )
        out["discount_last_14_days"] = self.roll_by_group(
            (price_hist != g["price"].shift(2)).astype(int),
            out["nm_id"],
            14,
            "sum",
        )

        # --- Остатки ---
        out["leftover_speed_1"] = left_hist - g["prev_leftovers"].shift(2)
        out["left_roll_mean_7"] = self.roll_by_group(left_hist, out["nm_id"], 7, "mean")
        out["left_roll_mean_14"] = self.roll_by_group(left_hist, out["nm_id"], 14, "mean")

        out["qty_bought_mean_7"] = self.roll_by_group(qty_hist, out["nm_id"], 7, "mean")
        out["days_until_runout"] = self._safe_div(
            out["prev_leftovers"],
            out["qty_bought_mean_7"] + 1e-2,
            default=90,
            clip_range=(0, 90),
        )
        out["zero_sales_ratio_30"] = self.roll_by_group((qty_hist == 0).astype(int), out["nm_id"], 30, "mean")

        # --- Тренд спроса ---
        trend_lag_1 = self.horizon
        trend_lag_2 = self.horizon * 3
        out["trend_qty"] = self._safe_div(
            g["qty"].shift(trend_lag_1) - g["qty"].shift(trend_lag_2),
            float(max(trend_lag_2 - trend_lag_1, 1)),
            default=0,
        )

        # --- Вариабельность (CV) ---
        ref_window = self._resolve_reference_window(28)
        qty_roll_mean_col = f"qty_roll_mean_{ref_window}"
        qty_roll_std_col = f"qty_roll_std_{ref_window}"
        price_roll_mean_col = f"price_roll_mean_{ref_window}"
        price_roll_std_col = f"price_roll_std_{ref_window}"

        out["qty_cv_28"] = self._safe_div(
            out[qty_roll_std_col],
            out[qty_roll_mean_col] + 1e-2,
            clip_range=(0, 10),
        )
        out["price_cv_28"] = self._safe_div(
            out[price_roll_std_col],
            out[price_roll_mean_col] + 1e-2,
            clip_range=(0, 10),
        )

        # --- Эластичность (упрощённый коэффициент e) ---
        q1 = g["qty"].shift(self.horizon)
        q2 = g["qty"].shift(self.horizon + 1)
        p1 = g["price"].shift(self.horizon)
        p2 = g["price"].shift(self.horizon + 1)

        e = self._safe_div((q1 - q2) * p2, (p1 - p2) * (q2 + 1e-2), default=0)
        out["e_coef"] = np.abs(e)

        high = out["e_coef"].mean() + 2 * out["e_coef"].std()
        if not np.isfinite(high):
            high = out["e_coef"].max(skipna=True)
        if not np.isfinite(high):
            high = 0.0

        out["e_coef"] = out["e_coef"].clip(0, high)
        e_filler = out.loc[out["e_coef"] != 0, "e_coef"].mean()
        if pd.isna(e_filler):
            e_filler = 0
        out["e_coef"] = out["e_coef"].fillna(e_filler)
        out["e_mean_28"] = self.roll_by_group(out["e_coef"], out["nm_id"], 28, "mean")

        # --- Рыночные средние ---
        temp = pd.DataFrame({"dt": out["dt"], "q_hist": qty_hist})
        daily_q = temp.groupby("dt")["q_hist"].mean()
        daily_q_7 = daily_q.shift(1).rolling(7, min_periods=2).mean()
        daily_q_28 = daily_q.shift(1).rolling(28, min_periods=2).mean()
        out["market_qty_mean_7"] = out["dt"].map(daily_q_7)
        out["market_qty_mean_28"] = out["dt"].map(daily_q_28)

        # --- Давность последней продажи ---
        out["_tmp_dt_sale"] = out["dt"].where(qty_hist > 0)
        out["_last_sale_dt"] = out.groupby("nm_id", sort=False)["_tmp_dt_sale"].ffill()
        out["days_since_last_sale"] = (out["dt"] - out["_last_sale_dt"]).dt.days
        out["days_since_last_sale"] = out["days_since_last_sale"].fillna(999).clip(0, 999)
        out["exp_decay_recency"] = np.exp(-0.05 * out["days_since_last_sale"])
        out["log_days_since_last_sale"] = np.log1p(out["days_since_last_sale"])
        out["is_long_silence"] = (out["days_since_last_sale"] > 30).astype(int)
        out = out.drop(columns=["_tmp_dt_sale", "_last_sale_dt"])

        # --- Вероятность ненулевого спроса ---
        out["nonzero_flag"] = (qty_hist > 0).astype(int)
        for w in (7, 14, 28):
            out[f"prob_nonzero_{w}"] = self.roll_by_group(out["nonzero_flag"], out["nm_id"], w, "mean")
        out["hazard_proxy"] = out["prob_nonzero_28"]

        # --- Нормализация цены ---
        out["price_rank_pct"] = g["price"].transform(lambda s: s.rank(pct=True))
        out["price_zscore"] = self._safe_div(
            out["price"] - out[price_roll_mean_col],
            out[price_roll_std_col] + 1e-2,
            clip_range=(-5, 5),
        )

        # --- Эффект промо ---
        promo_stats = (
            pd.DataFrame(
                {
                    "nm_id": out["nm_id"],
                    "is_promo": out["is_promo"],
                    "qty_hist": qty_hist,
                }
            )
            .groupby(["nm_id", "is_promo"])["qty_hist"]
            .mean()
            .unstack(fill_value=0)
        )
        promo_stats["promo_lift"] = self._safe_div(
            promo_stats.get(1, 0),
            promo_stats.get(0, 1) + 1e-2,
            default=1,
            clip_range=(0, 10),
        )
        out = out.merge(promo_stats[["promo_lift"]], on="nm_id", how="left")
        out["promo_lift"] = out["promo_lift"].fillna(1).clip(0, 10)

        # --- Ожидаемый спрос = вероятность ненуля * средний ненулевой ---
        positive_qty = qty_hist.where(qty_hist > 0)
        out["pos_qty_mean_28"] = self.roll_by_group(positive_qty, out["nm_id"], 28, "mean")
        out["expected_demand_28"] = out["prob_nonzero_28"].fillna(0) * out["pos_qty_mean_28"].fillna(0)

        # --- Спайки спроса ---
        out["spike_ratio"] = self._safe_div(
            out[f"qty_roll_max_{ref_window}"],
            out[qty_roll_mean_col] + 1e-2,
            default=1,
            clip_range=(0, 20),
        )
        out["is_recent_spike"] = (
            qty_hist > out[qty_roll_mean_col] + 2 * out[qty_roll_std_col].fillna(0)
        ).astype(int)

        # --- Кростон / Виллемейн для спорадического спроса ---
        horizon = int(self.horizon)
        croston_alpha = float(self.croston_alpha)
        will_window = int(self.willemain_window)
        min_p = max(7, min(14, will_window))

        out["_qty_hist_int"] = out.groupby("nm_id", sort=False)["qty"].shift(horizon)

        croston_pred = np.zeros(len(out), dtype=np.float32)
        for idx in out.groupby("nm_id", sort=False).indices.values():
            arr = out.loc[idx, "_qty_hist_int"].fillna(0.0).to_numpy(dtype=float)
            croston_pred[idx] = self._croston_sba_series(arr, alpha=croston_alpha)
        out["croston_sba_qty"] = croston_pred

        gg = out.groupby("nm_id", group_keys=False, sort=False)
        roll_cnt = gg["_qty_hist_int"].transform(lambda s: s.rolling(will_window, min_periods=min_p).count())
        nz_cnt = gg["_qty_hist_int"].transform(lambda s: (s > 0).rolling(will_window, min_periods=min_p).sum())
        nz_mean = gg["_qty_hist_int"].transform(lambda s: s.where(s > 0).rolling(will_window, min_periods=min_p).mean())
        nz_std = gg["_qty_hist_int"].transform(lambda s: s.where(s > 0).rolling(will_window, min_periods=min_p).std())

        out["willemain_adi"] = self._safe_div(roll_cnt, nz_cnt.clip(lower=1), default=1, clip_range=(0, 100))
        out["willemain_cv2"] = self._safe_div(nz_std, nz_mean + 1e-6, default=0, clip_range=(0, 10)) ** 2

        out["willemain_class"] = np.select(
            [
                (out["willemain_adi"] < 1.32) & (out["willemain_cv2"] < 0.49),
                (out["willemain_adi"] >= 1.32) & (out["willemain_cv2"] < 0.49),
                (out["willemain_adi"] < 1.32) & (out["willemain_cv2"] >= 0.49),
                (out["willemain_adi"] >= 1.32) & (out["willemain_cv2"] >= 0.49),
            ],
            [0, 1, 2, 3],
            default=0,
        ).astype(np.int8)

        prob_nonzero = self._safe_div(1.0, out["willemain_adi"] + 1e-6, default=0, clip_range=(0, 1))
        will_expected = (prob_nonzero * nz_mean.fillna(0.0)).clip(0, None)

        out["willemain_fcst_qty"] = np.where(
            out["willemain_class"].isin([1, 3]),
            out["croston_sba_qty"],
            will_expected,
        ).astype(np.float32)

        out = out.drop(columns=["_qty_hist_int"])
        out = self._sanitize_numeric(out)
        return out

    def _build_item_stats(self, features: pd.DataFrame) -> pd.DataFrame:
        """Считает агрегаты по товару: средние, std, доля нулей и т.п."""
        qty_lag_col = self._first_existing(
            features.columns,
            [f"qty_lag_{self.horizon}", *[f"qty_lag_{lag}" for lag in self.lags]],
        )

        agg = {}
        if qty_lag_col is not None:
            agg["nm_qty_mean"] = (qty_lag_col, "mean")
            agg["nm_qty_std"] = (qty_lag_col, "std")

        if "price" in features.columns:
            agg["nm_price_mean"] = ("price", "mean")
            agg["nm_price_std"] = ("price", "std")

        if "zero_sales_ratio_30" in features.columns:
            agg["nm_zero_ratio_mean"] = ("zero_sales_ratio_30", "mean")
        if "e_mean_28" in features.columns:
            agg["nm_e_mean"] = ("e_mean_28", "mean")
        if "days_until_runout" in features.columns:
            agg["nm_stockout_mean"] = ("days_until_runout", "mean")
        if "is_promo" in features.columns:
            agg["nm_has_promo"] = ("is_promo", "max")

        if agg:
            stats = features.groupby("nm_id").agg(**agg).reset_index()
        else:
            stats = features[["nm_id"]].drop_duplicates().copy()

        for col in self.ITEM_STAT_COLUMNS:
            if col not in stats.columns:
                stats[col] = 0.0

        ordered = ["nm_id", *self.ITEM_STAT_COLUMNS]
        stats = stats[ordered]
        stats = self._sanitize_numeric(stats)
        return stats

    def _fit_clusters(self, features: pd.DataFrame) -> None:
        """Кластеризует товары по их средним характеристикам."""
        ref_window = self._resolve_reference_window(28)

        candidates = [
            f"qty_lag_{self.horizon}",
            *[f"qty_lag_{lag}" for lag in self.lags],
            f"qty_roll_mean_{ref_window}",
            f"qty_roll_std_{ref_window}",
            "price",
            f"price_roll_mean_{ref_window}",
            f"price_roll_std_{ref_window}",
            "is_promo",
            "prev_leftovers",
            "is_long_silence",
            "days_until_runout",
            "zero_sales_ratio_30",
            "e_mean_28",
            "promo_days_28",
            "trend_qty",
        ]
        cluster_cols = [c for c in dict.fromkeys(candidates) if c in features.columns]

        if not cluster_cols:
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            cluster_cols = [c for c in numeric_cols if c != "nm_id"]

        if not cluster_cols:
            unique_nm = features["nm_id"].drop_duplicates().tolist()
            self.cluster_map_ = {nm: 0 for nm in unique_nm}
            self.cluster_price_map_ = {0: float(np.nan_to_num(features.get("price", pd.Series([0])).mean(), nan=0.0))}
            self.global_cluster_price_ = self.cluster_price_map_[0]
            return

        X = features.groupby("nm_id")[cluster_cols].mean().replace([np.inf, -np.inf], np.nan).fillna(0.0)

        n_items = X.shape[0]
        if n_items == 0:
            self.cluster_map_ = {}
            self.cluster_price_map_ = {}
            self.global_cluster_price_ = 0.0
            return

        clusters = np.zeros(n_items, dtype=int)
        self.scaler_ = None
        self.pca_ = None
        self.kmeans_ = None

        if n_items >= 2:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
            X_for_cluster = X_scaled

            if X_scaled.shape[1] > 1 and X_scaled.shape[0] > 2:
                self.pca_ = PCA(n_components=0.95, random_state=self.random_state)
                X_for_cluster = self.pca_.fit_transform(X_scaled)

            n_clusters = max(1, min(self.n_clusters, n_items))
            if n_clusters > 1:
                self.kmeans_ = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    batch_size=1000,
                    random_state=self.random_state,
                )
                clusters = self.kmeans_.fit_predict(X_for_cluster)

        self.cluster_map_ = pd.Series(clusters, index=X.index).to_dict()

        cluster_price = (
            features.assign(cluster=features["nm_id"].map(self.cluster_map_))
            .groupby("cluster")["price"]
            .mean()
        )

        self.cluster_price_map_ = cluster_price.to_dict()
        global_price = cluster_price.mean()
        if not np.isfinite(global_price):
            global_price = features["price"].mean() if "price" in features.columns else 0.0
        if not np.isfinite(global_price):
            global_price = 0.0
        self.global_cluster_price_ = float(global_price)

    def _add_fitted_context(self, features: pd.DataFrame) -> pd.DataFrame:
        """Добавляет статистики по товару и кластерные признаки."""
        out = features.merge(self.item_stats_, on="nm_id", how="left")

        out["cluster"] = out["nm_id"].map(self.cluster_map_).fillna(-1).astype(int)
        out["category_avg_price"] = out["cluster"].map(self.cluster_price_map_)
        out["category_avg_price"] = out["category_avg_price"].fillna(self.global_cluster_price_)

        out["price_cluster"] = self._safe_div(
            out["price"],
            out["category_avg_price"] + 1e-2,
            default=1,
            clip_range=(0, 5),
        )
        out["price_to_item_mean"] = self._safe_div(
            out["price"],
            out["nm_price_mean"] + 1e-2,
            default=1,
            clip_range=(0, 5),
        )

        qty_lag_col = self._first_existing(
            out.columns,
            [f"qty_lag_{self.horizon}", *[f"qty_lag_{lag}" for lag in self.lags]],
        )
        qty_base = out[qty_lag_col] if qty_lag_col is not None else 0.0

        out["qty_to_item_mean"] = self._safe_div(
            qty_base,
            out["nm_qty_mean"] + 1e-2,
            default=1,
            clip_range=(0, 10),
        )

        out = self._sanitize_numeric(out)
        return out

    def _combine_history_with_new(self, new_df: pd.DataFrame) -> pd.DataFrame:
        """Объединяет историю и новые данные для расчёта лагов."""
        hist = self.train_.copy()
        hist["_is_new"] = 0

        new = new_df.copy()
        new["_is_new"] = 1

        all_cols = sorted(set(hist.columns).union(new.columns))
        hist = self.cols_check(hist, all_cols)
        new = self.cols_check(new, all_cols)

        combined = pd.concat([hist, new], ignore_index=True)
        combined = combined.sort_values(["nm_id", "dt", "_is_new"], kind="mergesort")
        combined = combined.drop_duplicates(subset=["nm_id", "dt"], keep="last")
        combined = combined.sort_values(["nm_id", "dt"], kind="mergesort").reset_index(drop=True)
        return combined

    def _align_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Выравнивает итоговый набор фич по колонкам train."""
        out = df.copy()
        dtype_map = self.train_features_.dtypes.to_dict() if self.train_features_ is not None else {}

        for col in self.feature_columns_:
            if col in out.columns:
                continue

            dtype = dtype_map.get(col)
            if dtype is not None and pd.api.types.is_datetime64_any_dtype(dtype):
                out[col] = pd.NaT
            elif dtype is not None and pd.api.types.is_numeric_dtype(dtype):
                out[col] = 0.0
            else:
                out[col] = None

        return out[self.feature_columns_]

    def fit(self, train_df: pd.DataFrame) -> "TSFeatures":
        """Обучает генератор: сохраняет историю, статистики и кластеры."""
        train = self.prepare(train_df)
        self.train_ = train.copy()

        train_core = self.build_features(train)
        self.base_feature_columns_ = train_core.columns.tolist()

        self.item_stats_ = self._build_item_stats(train_core)
        self._fit_clusters(train_core)

        train_full = self._add_fitted_context(train_core)
        self.train_features_ = train_full.reset_index(drop=True)
        self.feature_columns_ = self.train_features_.columns.tolist()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Строит фичи для новых данных, используя историю из fit."""
        self._check_is_fitted()

        new_df = self.prepare(df)
        combined = self._combine_history_with_new(new_df)

        features = self.build_features(combined)
        features = self._add_fitted_context(features)

        result = features.loc[features["_is_new"] == 1].copy()
        result = result.drop(columns=["_is_new"]).reset_index(drop=True)
        result = self._align_output_columns(result)
        result = self._sanitize_numeric(result)
        return result

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Удобный метод: fit + transform для train."""
        self.fit(train_df)
        return self.train_features_.copy()

    def get_feature_names_out(self) -> list[str]:
        """Возвращает список итоговых фич после fit."""
        self._check_is_fitted()
        return list(self.feature_columns_)
