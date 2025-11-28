import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


class DataPreprocessor(TransformerMixin):
    def fit(self, data, *args):
        new_data = data.copy()
        self.categories = [
            "merchant",
            "category",
            "street",
            "city",
            "state",
            "job",
            "gender",
            "channel",
            "category_base",
        ]
        new_data["trans_date_trans_time"] = pd.to_datetime(
            new_data["trans_date_trans_time"]
        )

        self.last_trans_cc_dict = (
            new_data.groupby("cc_num", observed=False)["trans_date_trans_time"]
            .max()
            .to_dict()
        )
        self.last_trans_merchant_dict = (
            new_data.groupby("merchant", observed=False)["trans_date_trans_time"]
            .max()
            .to_dict()
        )

        self.last_transactions_dict = {}
        grouped = new_data.groupby("cc_num", observed=False)["trans_date_trans_time"]
        for cc, trans in grouped:
            self.last_transactions_dict[cc] = trans.sort_values()

        self.home_long = new_data.groupby("cc_num", observed=False)["long"].median()
        self.home_lat = new_data.groupby("cc_num", observed=False)["lat"].median()

        new_data["category_base"] = (
            new_data["category"]
            .str.replace(r"_(pos|net)$", "", regex=True)
            .astype("category")
        )
        self.favourite_category = (
            new_data.groupby("cc_num", observed=False)["category_base"]
            .apply(pd.Series.mode)
            .reset_index(level=1, drop=True)
        )
        self.favourite_category = self.favourite_category[
            ~self.favourite_category.index.duplicated(keep="first")
        ]

        self.median_amt = new_data.groupby("cc_num", observed=False)["amt"].median()

        return self

    def transform(self, data: pd.DataFrame):
        new_data = data.copy()

        new_data["trans_date_trans_time"] = pd.to_datetime(
            new_data["trans_date_trans_time"]
        )
        new_data["dob"] = pd.to_datetime(new_data["dob"], format="%Y-%m-%d")

        # Basic time features
        new_data["hour"] = new_data["trans_date_trans_time"].dt.hour  # type: ignore
        new_data["day_of_week"] = new_data["trans_date_trans_time"].dt.dayofweek  # type: ignore
        new_data["month"] = new_data["trans_date_trans_time"].dt.month  # type: ignore

        new_data = self._encode_cyclic_features(new_data, "hour", 24)
        new_data = self._encode_cyclic_features(new_data, "day_of_week", 7)
        new_data = self._encode_cyclic_features(new_data, "month", 12)

        # Periods
        new_data["is_weekend"] = new_data["day_of_week"] >= 5
        new_data["is_night"] = (new_data["hour"] >= 0) & (new_data["hour"] <= 5)

        # Time since
        new_data = self._compute_time_since_last_trans_cc(new_data)
        new_data = self._compute_time_since_last_trans_merchant(new_data)
        new_data = self._compute_rolling_frequency(new_data)

        # Is new
        new_data["is_new_street"] = ~new_data.duplicated(
            ["cc_num", "street"], keep="first"
        )
        new_data["is_new_city"] = ~new_data.duplicated(["cc_num", "city"], keep="first")
        new_data["is_new_state"] = ~new_data.duplicated(
            ["cc_num", "state"], keep="first"
        )

        # Merchant
        new_data["category_base"] = new_data["category"].str.replace(
            r"_(pos|net)$", "", regex=True
        )
        new_data["channel"] = "other"
        new_data.loc[new_data["category"].str.endswith("_net"), "channel"] = "net"
        new_data.loc[new_data["category"].str.endswith("_pos"), "channel"] = "pos"

        # Personality
        new_data["age_at_trans"] = (new_data["trans_date_trans_time"] - new_data["dob"]).dt.total_seconds()  # type: ignore

        new_data["dist_to_home_long"] = new_data["long"] - new_data["cc_num"].map(
            self.home_long
        )
        new_data["dist_to_home_lat"] = new_data["lat"] - new_data["cc_num"].map(
            self.home_lat
        )

        new_data["is_fav_category"] = new_data["category_base"] == new_data[
            "cc_num"
        ].map(self.favourite_category)
        new_data["median_amt"] = new_data["cc_num"].map(self.median_amt)

        # Process bool
        bool_cols = new_data.select_dtypes(include="bool").columns
        new_data[bool_cols] = new_data[bool_cols].astype(int)

        # Typization
        new_data[self.categories] = new_data[self.categories].astype("category")
        new_data["dob"] = new_data["dob"].astype("int64") // 10**9

        # Drop unwanted columns
        new_data.drop(
            [
                "cc_num",
                "Unnamed: 0",
                "unix_time",
                "zip",
                "trans_num",
                "first",
                "last",
                "trans_date_trans_time",
            ],
            axis=1,
            inplace=True,
            errors="ignore",
        )

        return new_data.sort_index()

    def _encode_cyclic_features(self, df: pd.DataFrame, col: str, period: int):
        new_df = df.copy()
        new_df[f"{col}_sin"] = np.sin(2 * np.pi * new_df[col] / period)
        new_df[f"{col}_cos"] = np.cos(2 * np.pi * new_df[col] / period)
        return new_df

    def _compute_time_since_last_trans_cc(self, df: pd.DataFrame):
        new_df = df.copy().sort_values(["cc_num", "trans_date_trans_time"])
        new_df["time_since_last_trans_cc"] = (
            new_df.groupby("cc_num", observed=False)["trans_date_trans_time"]
            .diff()
            .dt.total_seconds()  # type: ignore
        )

        # Search for previos transaction
        mask = new_df["time_since_last_trans_cc"].isna()
        new_df.loc[mask, "time_since_last_trans_cc"] = new_df.loc[mask].apply(
            lambda row: (
                row["trans_date_trans_time"]
                - self.last_trans_cc_dict.get(
                    row["cc_num"], row["trans_date_trans_time"]
                )
            ).total_seconds(),
            axis=1,
        )

        # First transaction
        new_df["first_trans_cc"] = new_df["time_since_last_trans_cc"].isna()
        new_df["time_since_last_trans_cc"] = new_df["time_since_last_trans_cc"].fillna(
            0
        )

        return new_df

    def _compute_time_since_last_trans_merchant(self, df: pd.DataFrame):
        new_df = df.copy().sort_values(["cc_num", "trans_date_trans_time"])
        new_df["time_since_last_trans_mercahnt"] = (
            new_df.groupby("cc_num", observed=False)["trans_date_trans_time"]
            .diff()
            .dt.total_seconds()  # type: ignore
        )

        # Search for previos transaction
        mask = new_df["time_since_last_trans_mercahnt"].isna()
        new_df.loc[mask, "time_since_last_trans_mercahnt"] = new_df.loc[mask].apply(
            lambda row: (
                row["trans_date_trans_time"]
                - self.last_trans_merchant_dict.get(
                    row["cc_num"], row["trans_date_trans_time"]
                )
            ).total_seconds(),
            axis=1,
        )

        # First transaction
        new_df["first_trans_merchant"] = new_df["time_since_last_trans_mercahnt"].isna()
        new_df["time_since_last_trans_mercahnt"] = new_df[
            "time_since_last_trans_mercahnt"
        ].fillna(0)

        return df

    def _compute_rolling_frequency(
        self,
        df: pd.DataFrame,
        windows_units=[(1, "min"), (5, "min"), (1, "h"), (1, "D"), (7, "D")],
    ):
        new_df = df.copy().sort_values(["cc_num", "trans_date_trans_time"])

        for window, unit in windows_units:
            freq_col = f"frequency_{window}{unit}"
            result = pd.Series(0, index=df.index)

            for cc, group in new_df.groupby("cc_num", sort=False, observed=False):
                prev_trans = self.last_transactions_dict.get(
                    cc, pd.Series([], dtype="datetime64[ns]")
                )

                first_current_time = group["trans_date_trans_time"].min()
                prev_trans = prev_trans[prev_trans < first_current_time]
                curr_trans = group["trans_date_trans_time"]

                combined_index = pd.concat([prev_trans, curr_trans]).sort_values()
                combined_series = pd.Series(1, index=combined_index)

                counts = (
                    combined_series.rolling(f"{window}{unit}")
                    .count()
                    .iloc[-len(group) :]
                )
                result.loc[group.index] = counts.values

            new_df[freq_col] = result

        return new_df
