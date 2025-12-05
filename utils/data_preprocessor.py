import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


def with_hist(required_cols: list[str]):
    def decorator(func):
        def wrapper(self, df: pd.DataFrame, *args, **kwargs):
            new_trans = df[required_cols].copy()
            new_trans["_is_new_data"] = True

            if not self.history.empty:
                prev_trans = self.history[required_cols].copy()
                prev_trans["_is_new_data"] = False
                all_trans = pd.concat([prev_trans, new_trans], ignore_index=True)
            else:
                all_trans = new_trans
            all_trans = all_trans.sort_values("_trans_time")

            result = func(self, all_trans, *args, **kwargs)

            mask = all_trans["_is_new_data"].to_numpy()
            result = result.iloc[mask]
            result.index = df.index

            return result

        return wrapper

    return decorator


class DataPreprocessor(TransformerMixin):
    def __init__(self):
        self.col_schema: dict[str, tuple[str, str]] = {
            "cc_num": ("_cc_num", "uint64"),
            "trans_date_trans_time": ("_trans_time", "object"),
            "merchant": ("c_merch", "category"),
            "category": ("c_cat", "category"),
            "amt": ("n_amt", "float32"),
            "gender": ("c_gender", "category"),
            "dob": ("d_dob", "object"),
            "job": ("c_job", "category"),
            "street": ("c_street", "category"),
            "city": ("c_city", "category"),
            "state": ("c_state", "category"),
            "lat": ("n_lat", "float32"),
            "long": ("n_long", "float32"),
            "city_pop": ("n_city_pop", "uint32"),
            "merch_lat": ("n_merch_lat", "float32"),
            "merch_long": ("n_merch_long", "float32"),
        }

        self.categories = [
            "c_merch",
            "c_cat",
            "c_street",
            "c_city",
            "c_state",
            "c_job",
            "c_gender",
            "c_channel",
            "c_cat_base",
            "c_cat_mode",
        ]

        self.frequencies = {
            "n_freq_1m": (1, "min"),
            "n_freq_5m": (5, "min"),
            "n_freq_1h": (1, "h"),
            "n_freq_1d": (1, "D"),
            "n_freq_7d": (7, "D"),
        }

        self.channels = {
            "shopping_net": "net",
            "misc_net": "net",
            "grocery_net": "net",
            "misc_pos": "pos",
            "shopping_pos": "pos",
            "grocery_pos": "pos",
            "personal_care": np.nan,
            "health_fitness": np.nan,
            "travel": np.nan,
            "kids_pets": np.nan,
            "food_dining": np.nan,
            "home": np.nan,
            "entertainment": np.nan,
            "gas_transport": np.nan,
        }

        self.history = pd.DataFrame()

    def fit(self, df, *args):
        new_df = df[
            ["trans_date_trans_time", "cc_num", "amt", "category", "merchant"]
        ].copy()
        new_df = self._apply_schema(new_df)

        new_df["_trans_time"] = pd.to_datetime(new_df["_trans_time"])
        new_df["c_cat_base"] = self._get_cat_base(new_df)
        new_df = new_df.drop("c_cat", axis=1)

        self.history = pd.concat([self.history, new_df])

        return self

    def transform(self, df, *args):
        import pandas as pd  # TODO:

        new_df = df[list(self.col_schema.keys())].copy()

        # Columns
        new_df = self._apply_schema(new_df)

        new_df["_trans_time"] = pd.to_datetime(new_df["_trans_time"])
        new_df["d_dob"] = pd.to_datetime(new_df["d_dob"])

        # Date and time
        new_df["n_hour"] = new_df["_trans_time"].dt.hour  # type: ignore
        new_df["n_day_of_week"] = new_df["_trans_time"].dt.dayofweek  # type: ignore
        new_df["n_month"] = new_df["_trans_time"].dt.month  # type: ignore

        new_df = self._cyc_enc(new_df, "n_hour", 24)
        new_df = self._cyc_enc(new_df, "n_day_of_week", 7)
        new_df = self._cyc_enc(new_df, "n_month", 12)

        # Periods
        new_df["b_weekend"] = new_df["n_day_of_week"] >= 5
        new_df["b_night"] = (new_df["n_hour"] >= 0) & (new_df["n_hour"] <= 5)

        # First and previos transactions
        new_df["n_s_last_cc"] = self._get_s_last_hist(new_df, "_cc_num")
        new_df["n_s_last_merch"] = self._get_s_last_hist(new_df, "c_merch")
        new_df["n_s_last_cc_merch"] = self._get_s_last_hist(
            new_df, ["_cc_num", "c_merch"]
        )

        # Frequencies
        new_df[list(self.frequencies.keys())] = self._get_freqs_hist(new_df)

        # Merchant
        new_df["c_cat_base"] = self._get_cat_base(new_df)
        new_df["c_channel"] = new_df["c_cat"].map(self.channels)

        # Personality
        new_df["d_age_at_trans"] = (new_df["_trans_time"] - new_df["d_dob"]).astype(
            "int64"
        ) // 10**9

        # Location
        new_df["n_dist_to_home_lat"] = new_df["n_merch_lat"] - new_df["n_lat"]
        new_df["n_dist_to_home_long"] = new_df["n_merch_long"] - new_df["n_long"]

        # Usual
        new_df["c_cat_mode"] = self._get_cat_mode_hist(new_df)
        new_df["n_amt_median"] = self._get_amt_med_hist(new_df)
        new_df["b_cat_mode"] = new_df["c_cat_base"] == new_df["c_cat_mode"]

        # Typization
        new_df[self.categories] = new_df[self.categories].astype("category")
        new_df["d_dob"] = pd.to_datetime(new_df["d_dob"]).astype("int64") // 10**9

        for cat in self.categories:
            new_df[cat] = new_df[cat].cat.add_categories("unknown")
            new_df[cat] = new_df[cat].fillna("unknown")

        # Drop unnecessary columns
        new_df = new_df.drop(
            [
                "_trans_time",
                "_cc_num",
            ],
            axis=1,
            errors="ignore",
        )

        return new_df

    def transform_fit(self, X):
        transformed = self.transform(X)
        self.fit(X)
        return transformed

    def _apply_schema(self, df: pd.DataFrame):
        for old_col, (new_col, dtype) in self.col_schema.items():
            if old_col in df.columns:
                df[new_col] = df[old_col].astype(dtype)  # type: ignore

        df = df.drop(
            list(self.col_schema.keys()),
            axis=1,
            errors="ignore",
        )

        return df

    def _cyc_enc(self, df: pd.DataFrame, feature: str, period: int):
        new_df = df.copy()
        new_df[f"{feature}_sin"] = np.sin(2 * np.pi * new_df[feature] / period).astype(
            "float16"
        )
        new_df[f"{feature}_cos"] = np.cos(2 * np.pi * new_df[feature] / period).astype(
            "float16"
        )
        return new_df

    @with_hist(["_trans_time", "_cc_num", "c_merch"])
    def _get_s_last_hist(
        self, df: pd.DataFrame, group_col: str | list[str], time_col="_trans_time"
    ):
        return df.groupby(group_col, observed=True)[time_col].diff().dt.seconds.astype("float32")  # type: ignore

    @with_hist(["_trans_time", "_cc_num", "n_amt"])
    def _get_freqs_hist(self, df: pd.DataFrame):
        result = pd.DataFrame()
        for feature, (window, unit) in self.frequencies.items():
            result.loc[:, feature] = self._get_freq(df, window, unit, "n_amt")

        return result

    @with_hist(["_trans_time", "_cc_num", "c_cat_base"])
    def _get_cat_mode_hist(self, df: pd.DataFrame):
        return self._get_cat_mode(df, "c_cat_base")

    @with_hist(["_trans_time", "_cc_num", "n_amt"])
    def _get_amt_med_hist(self, df: pd.DataFrame):
        return self._get_med(df, "n_amt")

    def _get_cat_base(self, df: pd.DataFrame) -> pd.Series:
        return (
            df["c_cat"].str.replace(r"_(pos|net)$", "", regex=True).astype("category")
        )

    def _get_freq(
        self,
        df: pd.DataFrame,
        window: int,
        unit: str,
        target: str,
        group_col: str = "_cc_num",
        time_col: str = "_trans_time",
    ) -> pd.Series:
        return (
            df[[group_col, time_col, target]]
            .sort_values([group_col, time_col])
            .groupby(group_col, observed=False)[[time_col, target]]
            .rolling(f"{window}{unit}", on=time_col)
            .count()[target]
            .reset_index(level=0, drop=True)
            .sort_index()
        )

    def _get_med(
        self, df: pd.DataFrame, target: str, group_col: str = "_cc_num"
    ) -> pd.Series:
        new_df = df[[group_col, target]].copy()

        result = (
            new_df.groupby(group_col, observed=False)[target]
            .expanding()
            .median()
            .groupby(level=0)
            .shift()
            .droplevel(0)
        )

        return pd.Series(result, index=new_df.index)

    def _get_cat_mode(
        self, df: pd.DataFrame, target: str, group_col: str = "_cc_num"
    ) -> pd.Series:
        new_df = df[[group_col, target]].copy()

        new_df[target] = new_df[target].astype("category")
        new_df["_cat_code"] = new_df[target].cat.codes
        codes = pd.Series(
            new_df[target].cat.categories.values,
            index=range(len(new_df[target].cat.categories)),
        )

        new_df["_cat_code_mode"] = (
            new_df.groupby(group_col, observed=False)["_cat_code"]
            .expanding()
            .apply(lambda x: x.mode().iloc[0])
            .groupby(level=0, observed=False)
            .shift()
            .droplevel(0)
        )

        return new_df["_cat_code_mode"].map(codes)
