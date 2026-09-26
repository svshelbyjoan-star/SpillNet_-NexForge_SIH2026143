# ============================================================
# SPILLNET — FINAL INTEGRATED PROTOTYPE
# ============================================================
#
# Pipeline:
#
# Satellite spill detection
#        ↓
# Lagrangian backward tracking
#        ↓
# Estimated spill source
#        ↓
# AIS candidate filtering
#        ↓
# 18 AIS behavioural features
#        ↓
# XGBoost
#        ↓
# Ranked potential source vessels
#        ↓
# Forward Lagrangian prediction
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# 2. USER CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Change these two paths
# ------------------------------------------------------------

AIS_FILE = "synthetic_chennai_ais.csv"

OCEAN_FILE = (
    "cmems_mod_glo_phy_anfc_merged-uv_PT1H-i_"
    "uo-vo_78.00E-82.00E_10.00N-14.00N_"
    "0.49m_2026-09-08-2026-09-11.nc"
)


# ------------------------------------------------------------
# Spill detected by satellite/U-Net
#
# In the final dashboard these values will come from
# the satellite detection module.
# ------------------------------------------------------------

DETECTED_SPILL_LAT = 12.30
DETECTED_SPILL_LON = 80.20
DETECTED_SPILL_TIME = "2026-09-08 16:00:00+00:00"


# ------------------------------------------------------------
# Lagrangian settings
# ------------------------------------------------------------

BACKWARD_HOURS = 10
FORWARD_HOURS = 10


# ------------------------------------------------------------
# AIS candidate search settings
# ------------------------------------------------------------

SOURCE_RADIUS_KM = 5
TIME_WINDOW_MINUTES = 30

TOP_N_VESSELS = 5


# ============================================================
# 3. FINAL 18 FEATURES
# ============================================================

FEATURES = [

    "closest_distance_km",

    "distance_at_window_start_km",

    "distance_at_window_end_km",

    "distance_change_km",

    "closest_time_difference_min",

    "time_inside_source_radius_min",

    "ais_point_count",

    "average_sog",

    "maximum_sog",

    "speed_at_closest_approach",

    "speed_change",

    "average_course_change",

    "course_change_at_closest_approach",

    "approach_speed",

    "departure_speed",

    "maximum_ais_gap_min",

    "long_ais_gap_count",

    "moving_fraction"
]


# ============================================================
# 4. HAVERSINE DISTANCE
# ============================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    R = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(
        np.sqrt(a)
    )

    return R * c


# ============================================================
# 5. AIS VESSEL TYPE LABELS
# ============================================================

AIS_VESSEL_TYPE_MAP = {

    0: "Unknown",

    20: "Wing-in-Ground",

    30: "Fishing",

    31: "Towing",

    32: "Towing (Long/Wide Tow)",

    33: "Dredging / Underwater Ops",

    34: "Diving Operations",

    35: "Military",

    36: "Sailing",

    37: "Pleasure Craft",

    40: "High-Speed Craft",

    50: "Pilot Vessel",

    51: "Search & Rescue",

    52: "Tug",

    53: "Port Tender",

    54: "Anti-Pollution Vessel",

    55: "Law Enforcement",

    56: "Spare / Local Assignment",

    57: "Spare / Local Assignment",

    58: "Medical Transport",

    59: "Other Special Vessel",

    60: "Passenger",

    70: "Cargo",

    80: "Tanker",

    90: "Other",

    99: "Other / Unknown"
}


# ============================================================
# 6. LOAD AIS DATA
# ============================================================

def load_ais_data(
    file_path
):

    print("\n")
    print("=" * 70)
    print("LOADING AIS DATA")
    print("=" * 70)

    ais = pd.read_csv(
        file_path
    )

    ais["BaseDateTime"] = pd.to_datetime(
        ais["BaseDateTime"],
        utc=True
    )

    numeric_columns = [

        "LAT",
        "LON",
        "SOG",
        "COG",
        "Heading",
        "VesselType",
        "Status",
        "Length",
        "Width",
        "Draft",
        "Cargo"
    ]

    for col in numeric_columns:

        if col in ais.columns:

            ais[col] = pd.to_numeric(
                ais[col],
                errors="coerce"
            )

    ais = ais.sort_values(
        [
            "MMSI",
            "BaseDateTime"
        ]
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # Course change
    # --------------------------------------------------------

    ais["previous_COG"] = (
        ais
        .groupby("MMSI")["COG"]
        .shift(1)
    )

    ais["course_change_deg"] = (
        (
            ais["COG"]
            -
            ais["previous_COG"]
        )
        .abs()
        .clip(
            upper=180
        )
    )


    # --------------------------------------------------------
    # AIS time differences
    # --------------------------------------------------------

    ais["time_diff_seconds"] = (

        ais
        .groupby("MMSI")["BaseDateTime"]
        .diff()
        .dt.total_seconds()
    )


    ais["is_long_ais_gap"] = (

        ais["time_diff_seconds"]
        >
        30 * 60
    )


    # --------------------------------------------------------
    # Vessel labels
    # --------------------------------------------------------

    if "VesselType" in ais.columns:

        ais["VesselTypeLabel"] = (

            ais["VesselType"]
            .map(AIS_VESSEL_TYPE_MAP)
            .fillna("Unknown")
        )

    else:

        ais["VesselTypeLabel"] = "Unknown"


    print(
        "Rows:",
        len(ais)
    )

    print(
        "Unique vessels:",
        ais["MMSI"].nunique()
    )

    print(
        "Time range:",
        ais["BaseDateTime"].min(),
        "to",
        ais["BaseDateTime"].max()
    )

    return ais


# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================

def calculate_candidate_features(
    vessel_df,
    source_lat,
    source_lon,
    source_time,
    source_radius_km=5
):

    df = vessel_df.copy()

    df["BaseDateTime"] = pd.to_datetime(
        df["BaseDateTime"],
        utc=True
    )

    source_time = pd.to_datetime(
        source_time,
        utc=True
    )


    # --------------------------------------------------------
    # Distance to source
    # --------------------------------------------------------

    df["distance_to_source_km"] = (
        haversine_distance(

            df["LAT"].values,

            df["LON"].values,

            source_lat,

            source_lon
        )
    )


    # --------------------------------------------------------
    # Time difference
    # --------------------------------------------------------

    df["time_difference_min"] = (

        (
            df["BaseDateTime"]
            -
            source_time
        )
        .abs()
        .dt.total_seconds()
        /
        60
    )


    # --------------------------------------------------------
    # Closest point
    # --------------------------------------------------------

    closest_idx = (
        df["distance_to_source_km"]
        .idxmin()
    )

    closest_row = df.loc[
        closest_idx
    ]


    closest_distance = (
        closest_row[
            "distance_to_source_km"
        ]
    )

    closest_time_difference = (
        closest_row[
            "time_difference_min"
        ]
    )


    speed_at_closest = (
        closest_row["SOG"]
    )


    course_change_at_closest = (
        closest_row.get(
            "course_change_deg",
            np.nan
        )
    )


    # --------------------------------------------------------
    # Window distances
    # --------------------------------------------------------

    first_distance = (
        df.iloc[0][
            "distance_to_source_km"
        ]
    )

    last_distance = (
        df.iloc[-1][
            "distance_to_source_km"
        ]
    )

    distance_change = (
        first_distance
        -
        last_distance
    )


    # --------------------------------------------------------
    # Time inside source radius
    # --------------------------------------------------------

    inside_radius = df[
        df["distance_to_source_km"]
        <=
        source_radius_km
    ]


    if len(inside_radius) >= 2:

        time_inside_radius = (

            (
                inside_radius[
                    "BaseDateTime"
                ].max()

                -

                inside_radius[
                    "BaseDateTime"
                ].min()
            )
            .total_seconds()
            /
            60
        )

    else:

        time_inside_radius = 0.0


    # --------------------------------------------------------
    # AIS point count
    # --------------------------------------------------------

    ais_point_count = len(df)


    # --------------------------------------------------------
    # Speed features
    # --------------------------------------------------------

    average_sog = (
        df["SOG"].mean()
    )

    maximum_sog = (
        df["SOG"].max()
    )

    speed_change = (

        df.iloc[-1]["SOG"]
        -
        df.iloc[0]["SOG"]
    )


    # --------------------------------------------------------
    # Course change
    # --------------------------------------------------------

    if "course_change_deg" in df.columns:

        average_course_change = (

            df[
                "course_change_deg"
            ]
            .dropna()
            .mean()
        )

    else:

        average_course_change = np.nan


    # --------------------------------------------------------
    # Approach speed
    # --------------------------------------------------------

    before_closest = df[
        df["BaseDateTime"]
        <=
        closest_row["BaseDateTime"]
    ]

    if len(before_closest) > 0:

        approach_speed = (
            before_closest["SOG"].mean()
        )

    else:

        approach_speed = 0.0


    # --------------------------------------------------------
    # Departure speed
    # --------------------------------------------------------

    after_closest = df[
        df["BaseDateTime"]
        >=
        closest_row["BaseDateTime"]
    ]

    if len(after_closest) > 0:

        departure_speed = (
            after_closest["SOG"].mean()
        )

    else:

        departure_speed = 0.0


    # --------------------------------------------------------
    # AIS gaps
    # --------------------------------------------------------

    if "time_diff_seconds" in df.columns:

        maximum_ais_gap_min = (

            df["time_diff_seconds"]
            .max()
            /
            60
        )

        if "is_long_ais_gap" in df.columns:

            long_ais_gap_count = (

                df["is_long_ais_gap"]
                .sum()
            )

        else:

            long_ais_gap_count = 0

    else:

        time_diffs = (

            df["BaseDateTime"]
            .diff()
            .dt.total_seconds()
        )

        maximum_ais_gap_min = (

            time_diffs.max()
            /
            60
        )

        long_ais_gap_count = (

            time_diffs
            >
            30 * 60
        ).sum()


    # --------------------------------------------------------
    # Moving fraction
    # --------------------------------------------------------

    moving_fraction = (

        df["SOG"] > 0.5
    ).mean()


    # --------------------------------------------------------
    # Final feature dictionary
    # --------------------------------------------------------

    features = {

        "closest_distance_km":
            closest_distance,

        "distance_at_window_start_km":
            first_distance,

        "distance_at_window_end_km":
            last_distance,

        "distance_change_km":
            distance_change,

        "closest_time_difference_min":
            closest_time_difference,

        "time_inside_source_radius_min":
            time_inside_radius,

        "ais_point_count":
            ais_point_count,

        "average_sog":
            average_sog,

        "maximum_sog":
            maximum_sog,

        "speed_at_closest_approach":
            speed_at_closest,

        "speed_change":
            speed_change,

        "average_course_change":
            average_course_change,

        "course_change_at_closest_approach":
            course_change_at_closest,

        "approach_speed":
            approach_speed,

        "departure_speed":
            departure_speed,

        "maximum_ais_gap_min":
            maximum_ais_gap_min,

        "long_ais_gap_count":
            long_ais_gap_count,

        "moving_fraction":
            moving_fraction
    }


    return features


# ============================================================
# 8. LAGRANGIAN MODEL
# ============================================================

class LagrangianModel:

    def __init__(
        self,
        ocean_file
    ):

        print("\n")
        print("=" * 70)
        print("LOADING COPERNICUS OCEAN DATA")
        print("=" * 70)

        self.ds = xr.open_dataset(
            ocean_file
        )

        self.uo = self.ds["uo"].isel(
            depth=0
        )

        self.vo = self.ds["vo"].isel(
            depth=0
        )

        print(
            "Dataset loaded."
        )

        print(
            "Time points:",
            len(self.ds.time)
        )

        print(
            "Latitude:",
            float(self.ds.latitude.min()),
            "to",
            float(self.ds.latitude.max())
        )

        print(
            "Longitude:",
            float(self.ds.longitude.min()),
            "to",
            float(self.ds.longitude.max())
        )


    # --------------------------------------------------------
    # Get current at position/time
    # --------------------------------------------------------

    def get_current(
        self,
        lat,
        lon,
        time_index
    ):

        lat_index = (
            abs(
                self.ds.latitude
                -
                lat
            ).argmin()
        )

        lon_index = (
            abs(
                self.ds.longitude
                -
                lon
            ).argmin()
        )


        u = float(
            self.uo.isel(
                time=time_index,
                latitude=lat_index,
                longitude=lon_index
            ).values
        )

        v = float(
            self.vo.isel(
                time=time_index,
                latitude=lat_index,
                longitude=lon_index
            ).values
        )


        return u, v


    # --------------------------------------------------------
    # Dynamic Lagrangian run
    # --------------------------------------------------------

    def run(
        self,
        spill_lat,
        spill_lon,
        spill_time,
        backward_hours=10,
        forward_hours=10,
        plot=True
    ):

        spill_time = pd.to_datetime(
            spill_time,
            utc=True
        )


        # ----------------------------------------------------
        # Find nearest available dataset time
        # ----------------------------------------------------

        time_values = pd.to_datetime(
            self.ds.time.values,
            utc=True
        )


        time_differences = np.abs(
            time_values
            -
            spill_time
        )


        spill_time_index = int(
            np.argmin(
                time_differences
            )
        )


        actual_spill_time = (
            time_values[
                spill_time_index
            ]
        )


        print("\n")
        print("=" * 70)
        print("LAGRANGIAN TRACKING")
        print("=" * 70)

        print(
            "Detected spill:",
            spill_lat,
            spill_lon
        )

        print(
            "Requested time:",
            spill_time
        )

        print(
            "Nearest dataset time:",
            actual_spill_time
        )


        # ====================================================
        # BACKWARD TRACKING
        # ====================================================

        back_lat = [
            float(spill_lat)
        ]

        back_lon = [
            float(spill_lon)
        ]

        back_time = [
            actual_spill_time
        ]


        lat = float(
            spill_lat
        )

        lon = float(
            spill_lon
        )


        for step in range(
            1,
            backward_hours + 1
        ):

            time_index = (
                spill_time_index
                -
                step
            )


            if time_index < 0:

                break


            u, v = self.get_current(
                lat,
                lon,
                time_index
            )


            if np.isnan(u) or np.isnan(v):

                break


            dt = 3600


            # Reverse movement
            lat -= (
                v * dt
            ) / 111320.0


            lon -= (

                u * dt

                /

                (
                    111320.0
                    *
                    np.cos(
                        np.radians(lat)
                    )
                )
            )


            back_lat.append(
                lat
            )

            back_lon.append(
                lon
            )

            back_time.append(
                time_values[
                    time_index
                ]
            )


        # ====================================================
        # FORWARD TRACKING
        # ====================================================

        forward_lat = [
            float(spill_lat)
        ]

        forward_lon = [
            float(spill_lon)
        ]

        forward_time = [
            actual_spill_time
        ]


        lat = float(
            spill_lat
        )

        lon = float(
            spill_lon
        )


        for step in range(
            1,
            forward_hours + 1
        ):

            time_index = (
                spill_time_index
                +
                step
            )


            if time_index >= len(
                self.ds.time
            ):

                break


            u, v = self.get_current(
                lat,
                lon,
                time_index
            )


            if np.isnan(u) or np.isnan(v):

                break


            dt = 3600


            # Forward movement
            lat += (
                v * dt
            ) / 111320.0


            lon += (

                u * dt

                /

                (
                    111320.0
                    *
                    np.cos(
                        np.radians(lat)
                    )
                )
            )


            forward_lat.append(
                lat
            )

            forward_lon.append(
                lon
            )

            forward_time.append(
                time_values[
                    time_index
                ]
            )


        # ====================================================
        # RESULTS
        # ====================================================

        estimated_source = {

            "latitude":
                float(back_lat[-1]),

            "longitude":
                float(back_lon[-1]),

            "time":
                str(
                    pd.Timestamp(
                        back_time[-1]
                    ).isoformat()
                )
        }


        forward_prediction = {

            "latitude":
                float(forward_lat[-1]),

            "longitude":
                float(forward_lon[-1]),

            "time":
                str(
                    pd.Timestamp(
                        forward_time[-1]
                    ).isoformat()
                )
        }


        result = {

            "detected_spill": {

                "latitude":
                    float(spill_lat),

                "longitude":
                    float(spill_lon),

                "time":
                    str(
                        actual_spill_time.isoformat()
                    )
            },

            "estimated_source":
                estimated_source,

            "forward_prediction":
                forward_prediction,

            "backward_trajectory": [

                {
                    "latitude":
                        float(lat),

                    "longitude":
                        float(lon),

                    "time":
                        str(
                            pd.Timestamp(t)
                            .isoformat()
                        )
                }

                for lat, lon, t
                in zip(
                    back_lat,
                    back_lon,
                    back_time
                )
            ],

            "forward_trajectory": [

                {
                    "latitude":
                        float(lat),

                    "longitude":
                        float(lon),

                    "time":
                        str(
                            pd.Timestamp(t)
                            .isoformat()
                        )
                }

                for lat, lon, t
                in zip(
                    forward_lat,
                    forward_lon,
                    forward_time
                )
            ]
        }


        # ====================================================
        # PLOT
        # ====================================================

        if plot:

            plt.figure(
                figsize=(10, 8)
            )


            plt.plot(
                back_lon,
                back_lat,
                marker="o",
                linewidth=2,
                label="Backward trajectory"
            )


            plt.plot(
                forward_lon,
                forward_lat,
                marker="o",
                linewidth=2,
                label="Forward trajectory"
            )


            plt.scatter(
                back_lon[-1],
                back_lat[-1],
                marker="X",
                s=180,
                label="Estimated source"
            )


            plt.scatter(
                spill_lon,
                spill_lat,
                marker="*",
                s=250,
                label="Detected spill"
            )


            plt.scatter(
                forward_lon[-1],
                forward_lat[-1],
                marker="X",
                s=150,
                label="Future prediction"
            )


            plt.xlabel(
                "Longitude"
            )

            plt.ylabel(
                "Latitude"
            )

            plt.title(
                "SPILLNET — Lagrangian Oil-Spill Tracking"
            )

            plt.grid(
                True,
                alpha=0.3
            )

            plt.legend()

            plt.tight_layout()

            plt.show()


        return result


# ============================================================
# 9. FIND AIS CANDIDATES
# ============================================================

def find_ais_candidates(
    ais_df,
    source_lat,
    source_lon,
    source_time,
    source_radius_km=5,
    time_window_minutes=30
):

    data = ais_df.copy()


    source_time = pd.to_datetime(
        source_time,
        utc=True
    )


    data["BaseDateTime"] = pd.to_datetime(
        data["BaseDateTime"],
        utc=True
    )


    data["distance_to_source_km"] = (

        haversine_distance(

            data["LAT"].values,

            data["LON"].values,

            source_lat,

            source_lon
        )
    )


    data["time_difference_min"] = (

        (
            data["BaseDateTime"]
            -
            source_time
        )
        .abs()
        .dt.total_seconds()
        /
        60
    )


    candidate_points = data[

        (
            data["distance_to_source_km"]
            <=
            source_radius_km
        )

        &

        (
            data["time_difference_min"]
            <=
            time_window_minutes
        )

    ].copy()


    candidate_mmsis = (

        candidate_points[
            "MMSI"
        ]
        .dropna()
        .unique()
        .tolist()
    )


    return (
        candidate_mmsis,
        candidate_points
    )


# ============================================================
# 10. RANK AIS CANDIDATES
# ============================================================

def rank_ais_candidates(
    ais_df,
    model,
    source_lat,
    source_lon,
    source_time,
    source_radius_km=5,
    time_window_minutes=30
):

    data = ais_df.copy()


    source_time = pd.to_datetime(
        source_time,
        utc=True
    )


    candidate_mmsis, candidate_points = (
        find_ais_candidates(

            data,

            source_lat,

            source_lon,

            source_time,

            source_radius_km,

            time_window_minutes
        )
    )


    if len(candidate_mmsis) == 0:

        return pd.DataFrame()


    candidate_feature_rows = []


    time_start = (

        source_time
        -
        pd.Timedelta(
            minutes=time_window_minutes
        )
    )


    time_end = (

        source_time
        +
        pd.Timedelta(
            minutes=time_window_minutes
        )
    )


    for mmsi in candidate_mmsis:

        vessel_track = data[

            (data["MMSI"] == mmsi)

            &

            (
                data["BaseDateTime"]
                >=
                time_start
            )

            &

            (
                data["BaseDateTime"]
                <=
                time_end
            )

        ].copy()


        if vessel_track.empty:

            continue


        features = (
            calculate_candidate_features(

                vessel_track,

                source_lat,

                source_lon,

                source_time,

                source_radius_km
            )
        )


        features["MMSI"] = mmsi


        candidate_feature_rows.append(
            features
        )


    if not candidate_feature_rows:

        return pd.DataFrame()


    feature_df = pd.DataFrame(
        candidate_feature_rows
    )


    # --------------------------------------------------------
    # Clean model input
    # --------------------------------------------------------

    X = (

        feature_df[
            FEATURES
        ]

        .replace(
            [np.inf, -np.inf],
            np.nan
        )

        .fillna(0)
    )


    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    feature_df[
        "source_probability"
    ] = model.predict_proba(X)[:, 1]


    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    feature_df = (

        feature_df

        .sort_values(
            "source_probability",
            ascending=False
        )

        .reset_index(
            drop=True
        )
    )


    feature_df[
        "rank"
    ] = (
        feature_df.index + 1
    )


    return feature_df


# ============================================================
# 11. TRAIN V3 XGBOOST MODEL
# ============================================================

def train_v3_model(
    ais
):

    print("\n")
    print("=" * 70)
    print("TRAINING XGBOOST V3")
    print("=" * 70)


    NUM_EVENTS = 300

    np.random.seed(42)


    # --------------------------------------------------------
    # Select moving AIS points
    # --------------------------------------------------------

    moving_points = ais[
        ais["SOG"] >= 2
    ].copy()


    if len(moving_points) < NUM_EVENTS:

        raise ValueError(

            "Not enough moving AIS points "
            "to generate 300 training events."
        )


    moving_points = moving_points.sample(
        n=NUM_EVENTS,
        random_state=42
    ).reset_index(
        drop=True
    )


    realistic_events = []


    # --------------------------------------------------------
    # Generate realistic uncertain source events
    # --------------------------------------------------------

    for i, row in moving_points.iterrows():

        actual_lat = row["LAT"]

        actual_lon = row["LON"]

        actual_time = pd.to_datetime(
            row["BaseDateTime"],
            utc=True
        )


        location_error_km = np.random.uniform(
            0.2,
            2.0
        )


        time_error_min = np.random.uniform(
            -15,
            15
        )


        angle = np.random.uniform(
            0,
            2 * np.pi
        )


        lat_shift = (

            location_error_km
            *
            np.cos(angle)
            /
            111.0
        )


        lon_shift = (

            location_error_km
            *
            np.sin(angle)

            /

            (
                111.0
                *
                np.cos(
                    np.radians(
                        actual_lat
                    )
                )
            )
        )


        estimated_lat = (
            actual_lat
            +
            lat_shift
        )


        estimated_lon = (
            actual_lon
            +
            lon_shift
        )


        estimated_time = (

            actual_time

            +

            pd.Timedelta(
                minutes=time_error_min
            )
        )


        realistic_events.append({

            "event_id":
                f"V3_EVENT_{i+1:04d}",

            "actual_source_mmsi":
                row["MMSI"],

            "source_lat":
                estimated_lat,

            "source_lon":
                estimated_lon,

            "source_time":
                estimated_time
        })


    event_df = pd.DataFrame(
        realistic_events
    )


    # --------------------------------------------------------
    # Build candidate training data
    # --------------------------------------------------------

    training_rows = []


    for _, event in event_df.iterrows():

        event_id = event["event_id"]

        source_lat = event["source_lat"]

        source_lon = event["source_lon"]

        source_time = pd.to_datetime(
            event["source_time"],
            utc=True
        )

        actual_mmsi = event[
            "actual_source_mmsi"
        ]


        time_start = (

            source_time

            -

            pd.Timedelta(
                minutes=30
            )
        )


        time_end = (

            source_time

            +

            pd.Timedelta(
                minutes=30
            )
        )


        event_window = ais[

            (
                ais["BaseDateTime"]
                >=
                time_start
            )

            &

            (
                ais["BaseDateTime"]
                <=
                time_end
            )

        ].copy()


        if event_window.empty:

            continue


        event_window[
            "distance_to_source_km"
        ] = (

            haversine_distance(

                event_window[
                    "LAT"
                ].values,

                event_window[
                    "LON"
                ].values,

                source_lat,

                source_lon
            )
        )


        candidate_mmsis = (

            event_window[

                event_window[
                    "distance_to_source_km"
                ]
                <=
                5

            ][
                "MMSI"
            ]
            .unique()
        )


        for mmsi in candidate_mmsis:

            vessel_window = (
                event_window[
                    event_window[
                        "MMSI"
                    ]
                    ==
                    mmsi
                ].copy()
            )


            if vessel_window.empty:

                continue


            features = (
                calculate_candidate_features(

                    vessel_window,

                    source_lat,

                    source_lon,

                    source_time,

                    5
                )
            )


            label = int(
                mmsi == actual_mmsi
            )


            training_rows.append({

                "event_id":
                    event_id,

                "MMSI":
                    mmsi,

                **features,

                "label":
                    label
            })


    training_df = pd.DataFrame(
        training_rows
    )


    if training_df.empty:

        raise ValueError(
            "No candidate training data was generated."
        )


    training_df = (

        training_df

        .replace(
            [np.inf, -np.inf],
            np.nan
        )

        .fillna(0)
    )


    print(
        "Training rows:",
        len(training_df)
    )

    print(
        "Training events:",
        training_df[
            "event_id"
        ].nunique()
    )

    print(
        "Unique vessels:",
        training_df[
            "MMSI"
        ].nunique()
    )


    # --------------------------------------------------------
    # Event-level split
    # --------------------------------------------------------

    X = training_df[
        FEATURES
    ]

    y = training_df[
        "label"
    ]

    groups = training_df[
        "event_id"
    ]


    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )


    train_idx, test_idx = next(

        splitter.split(
            X,
            y,
            groups=groups
        )
    )


    X_train = X.iloc[
        train_idx
    ]

    X_test = X.iloc[
        test_idx
    ]


    y_train = y.iloc[
        train_idx
    ]

    y_test = y.iloc[
        test_idx
    ]


    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    model = XGBClassifier(

        n_estimators=300,

        max_depth=4,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="binary:logistic",

        eval_metric="logloss",

        random_state=42,

        n_jobs=-1
    )


    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    print("\n")
    print("V3 MODEL EVALUATION")
    print("=" * 70)


    print(
        "Accuracy :",
        round(
            accuracy_score(
                y_test,
                predictions
            ),
            4
        )
    )


    print(
        "Precision:",
        round(
            precision_score(
                y_test,
                predictions,
                zero_division=0
            ),
            4
        )
    )


    print(
        "Recall   :",
        round(
            recall_score(
                y_test,
                predictions,
                zero_division=0
            ),
            4
        )
    )


    print(
        "F1 Score :",
        round(
            f1_score(
                y_test,
                predictions,
                zero_division=0
            ),
            4
        )
    )


    print(
        "ROC-AUC  :",
        round(
            roc_auc_score(
                y_test,
                probabilities
            ),
            4
        )
    )


    return model


# ============================================================
# 12. SAVE MODEL
# ============================================================

def save_model(
    model
):

    os.makedirs(
        "SPILLNET_MODEL",
        exist_ok=True
    )


    model_path = (
        "SPILLNET_MODEL/"
        "xgboost_v3.json"
    )


    feature_path = (
        "SPILLNET_MODEL/"
        "features.json"
    )


    model.save_model(
        model_path
    )


    with open(
        feature_path,
        "w"
    ) as f:

        json.dump(
            FEATURES,
            f,
            indent=4
        )


    print("\n")
    print(
        "Model saved:"
    )

    print(
        model_path
    )

    print(
        feature_path
    )


# ============================================================
# 13. LOAD EXISTING MODEL IF AVAILABLE
# ============================================================

def get_model(
    ais
):

    model_path = (
        "SPILLNET_MODEL/"
        "xgboost_v3.json"
    )


    feature_path = (
        "SPILLNET_MODEL/"
        "features.json"
    )


    # --------------------------------------------------------
    # Existing model
    # --------------------------------------------------------

    if (
        os.path.exists(model_path)
        and
        os.path.exists(feature_path)
    ):

        print("\n")
        print(
            "Existing SPILLNET XGBoost model found."
        )


        model = XGBClassifier()

        model.load_model(
            model_path
        )


        with open(
            feature_path,
            "r"
        ) as f:

            saved_features = json.load(
                f
            )


        if saved_features != FEATURES:

            raise ValueError(
                "Saved model feature list "
                "does not match current pipeline."
            )


        print(
            "Model loaded successfully."
        )


        return model


    # --------------------------------------------------------
    # No model → train it
    # --------------------------------------------------------

    print("\n")
    print(
        "No trained model found."
    )

    print(
        "Training V3 model now..."
    )


    model = train_v3_model(
        ais
    )


    save_model(
        model
    )


    return model


# ============================================================
# 14. DASHBOARD OUTPUT
# ============================================================

def prepare_dashboard_output(
    ranked_results,
    ais_df,
    top_n=5
):

    if ranked_results.empty:

        return pd.DataFrame()


    results = (
        ranked_results
        .head(top_n)
        .copy()
    )


    vessel_info = (

        ais_df[
            [
                "MMSI",
                "VesselName",
                "VesselTypeLabel"
            ]
        ]
        .drop_duplicates(
            "MMSI"
        )
    )


    results = results.merge(

        vessel_info,

        on="MMSI",

        how="left"
    )


    output = results[

        [
            "rank",

            "MMSI",

            "VesselName",

            "VesselTypeLabel",

            "source_probability",

            "closest_distance_km",

            "closest_time_difference_min",

            "average_sog",

            "speed_at_closest_approach",

            "moving_fraction"
        ]

    ].copy()


    output["VesselName"] = (

        output["VesselName"]

        .fillna("Unknown")

        .astype(str)

        .str.strip()
    )


    output["VesselTypeLabel"] = (

        output[
            "VesselTypeLabel"
        ]

        .fillna("Unknown")
    )


    output["source_probability"] = (

        output[
            "source_probability"
        ]
        .round(4)
    )


    output[
        "closest_distance_km"
    ] = (

        output[
            "closest_distance_km"
        ]
        .round(3)
    )


    output[
        "closest_time_difference_min"
    ] = (

        output[
            "closest_time_difference_min"
        ]
        .round(2)
    )


    output[
        "average_sog"
    ] = (

        output[
            "average_sog"
        ]
        .round(2)
    )


    output[
        "speed_at_closest_approach"
    ] = (

        output[
            "speed_at_closest_approach"
        ]
        .round(2)
    )


    output[
        "moving_fraction"
    ] = (

        output[
            "moving_fraction"
        ]
        .round(3)
    )


    return output


# ============================================================
# 15. COMPLETE SPILLNET PIPELINE
# ============================================================

def run_spillnet_pipeline(

    spill_lat,

    spill_lon,

    spill_time,

    ais_df,

    lagrangian_model,

    xgboost_model,

    backward_hours=10,

    forward_hours=10,

    top_n=5,

    source_radius_km=5,

    time_window_minutes=30,

    plot=True

):


    # ========================================================
    # STEP 1 — LAGRANGIAN
    # ========================================================

    lagrangian_result = (

        lagrangian_model.run(

            spill_lat=spill_lat,

            spill_lon=spill_lon,

            spill_time=spill_time,

            backward_hours=backward_hours,

            forward_hours=forward_hours,

            plot=plot
        )
    )


    # ========================================================
    # STEP 2 — GET ESTIMATED SOURCE
    # ========================================================

    source = (
        lagrangian_result[
            "estimated_source"
        ]
    )


    print("\n")
    print("=" * 70)
    print("ESTIMATED SPILL SOURCE")
    print("=" * 70)


    print(
        "Latitude :",
        source["latitude"]
    )


    print(
        "Longitude:",
        source["longitude"]
    )


    print(
        "Time     :",
        source["time"]
    )


    # ========================================================
    # STEP 3 — AIS CANDIDATE RANKING
    # ========================================================

    ranked_results = (

        rank_ais_candidates(

            ais_df=ais_df,

            model=xgboost_model,

            source_lat=source[
                "latitude"
            ],

            source_lon=source[
                "longitude"
            ],

            source_time=source[
                "time"
            ],

            source_radius_km=source_radius_km,

            time_window_minutes=time_window_minutes
        )
    )


    # ========================================================
    # STEP 4 — DASHBOARD OUTPUT
    # ========================================================

    dashboard_output = (

        prepare_dashboard_output(

            ranked_results,

            ais_df,

            top_n
        )
    )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    final_result = {

        "detected_spill":
            lagrangian_result[
                "detected_spill"
            ],

        "estimated_source":
            lagrangian_result[
                "estimated_source"
            ],

        "forward_prediction":
            lagrangian_result[
                "forward_prediction"
            ],

        "backward_trajectory":
            lagrangian_result[
                "backward_trajectory"
            ],

        "forward_trajectory":
            lagrangian_result[
                "forward_trajectory"
            ],

        "candidate_vessels":
            dashboard_output.to_dict(
                orient="records"
            )
    }


    return final_result


# ============================================================
# 16. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("SPILLNET — NEXFORGE")
    print("Oil Spill Source Identification Prototype")
    print("=" * 70)


    # ========================================================
    # LOAD AIS
    # ========================================================

    if not os.path.exists(
        AIS_FILE
    ):

        raise FileNotFoundError(

            f"AIS file not found:\n"
            f"{AIS_FILE}\n\n"
            f"Place the AIS CSV in the same "
            f"folder as this Python script."
        )


    ais = load_ais_data(
        AIS_FILE
    )


    # ========================================================
    # LOAD LAGRANGIAN MODEL
    # ========================================================

    if not os.path.exists(
        OCEAN_FILE
    ):

        raise FileNotFoundError(

            f"Copernicus NetCDF file not found:\n"
            f"{OCEAN_FILE}\n\n"
            f"Place the .nc file in the same "
            f"folder as this Python script."
        )


    lagrangian_model = (
        LagrangianModel(
            OCEAN_FILE
        )
    )


    # ========================================================
    # GET / TRAIN XGBOOST
    # ========================================================

    xgboost_model = get_model(
        ais
    )


    # ========================================================
    # RUN COMPLETE PIPELINE
    # ========================================================

    result = run_spillnet_pipeline(

        spill_lat=
            DETECTED_SPILL_LAT,

        spill_lon=
            DETECTED_SPILL_LON,

        spill_time=
            DETECTED_SPILL_TIME,

        ais_df=
            ais,

        lagrangian_model=
            lagrangian_model,

        xgboost_model=
            xgboost_model,

        backward_hours=
            BACKWARD_HOURS,

        forward_hours=
            FORWARD_HOURS,

        top_n=
            TOP_N_VESSELS,

        source_radius_km=
            SOURCE_RADIUS_KM,

        time_window_minutes=
            TIME_WINDOW_MINUTES,

        plot=True
    )


    # ========================================================
    # PRINT FINAL OUTPUT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("SPILLNET FINAL OUTPUT")
    print("=" * 70)


    print("\nDETECTED SPILL")
    print("-" * 70)

    print(
        result[
            "detected_spill"
        ]
    )


    print("\nESTIMATED SOURCE")
    print("-" * 70)

    print(
        result[
            "estimated_source"
        ]
    )


    print("\nFORWARD OIL TRAJECTORY PREDICTION")
    print("-" * 70)

    print(
        result[
            "forward_prediction"
        ]
    )


    print("\nPOTENTIAL SOURCE VESSELS")
    print("-" * 70)


    candidate_vessels = (
        result[
            "candidate_vessels"
        ]
    )


    if not candidate_vessels:

        print(
            "No AIS candidate vessels "
            "were found within the search window."
        )

    else:

        dashboard_df = pd.DataFrame(
            candidate_vessels
        )


        print(
            dashboard_df.to_string(
                index=False
            )
        )


    # ========================================================
    # SAVE FINAL JSON
    # ========================================================

    with open(
        "spillnet_result.json",
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=4,
            default=str
        )


    print("\n")
    print("=" * 70)
    print(
        "Complete SPILLNET pipeline finished."
    )
    print(
        "Output saved as: spillnet_result.json"
    )
    print("=" * 70)
