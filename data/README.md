# SPILLNET Data

This directory documents the datasets used by the SPILLNET prototype.

The project uses a combination of satellite imagery, ocean-current data, real AIS data for model development/validation, and a synthetic Chennai AIS dataset for the prototype demonstration.

---

## 1. Satellite Data

Satellite imagery is used as the input for the oil-spill detection module.

### Purpose

The satellite data is processed by the oil-spill detection model to identify potential oil-spill regions and estimate their geographic location.

### Output

The spill detection module provides information such as:

- Detected spill location
- Latitude
- Longitude
- Detection timestamp
- Spill region/mask
- Detection confidence

The detected spill location is passed to the Lagrangian trajectory model.

---

# 2. Copernicus Marine Ocean-Current Data

The Lagrangian model uses ocean-current data from the Copernicus Marine Service.

### Dataset

**Global Ocean Physics Analysis and Forecast**

Dataset:

cmems_mod_glo_phy_anfc_merged-uv_PT1H-i

Variables used:

- uo — Eastward sea-water velocity
- vo— Northward sea-water velocity

Units:

m/s

The surface current layer was used for the prototype.

### Prototype Geographic Region

The Copernicus subset used for the Chennai-region prototype covers:

Longitude: 78°E – 82°E
Latitude : 10°N – 14°N

Prototype Time Range

The downloaded ocean-current subset covers:

2026-09-08 06:00 UTC
to
2026-09-11 06:00 UTC

with hourly current data.

Spill Scenario

The prototype spill was initialized at:

Latitude  : 12.30000°N
Longitude : 80.20000°E
Time      : 2026-09-08 16:00 UTC

The Lagrangian model performs:

Backward tracking to estimate the probable source
Forward tracking to predict future oil movement
Estimated Source

For the prototype scenario, the backward trajectory estimated:

Latitude  : 12.22076°N
Longitude : 80.13420°E
Time      : 2026-09-08 06:00 UTC

This estimated source location and time are then passed to the AIS module.

3. Real AIS Dataset

Real AIS data was used for developing and validating the AIS vessel-identification methodology.

Dataset Source

MarineCadastre AIS data

The downloaded dataset was:

AIS_178896334879276928_691-1788963349118.csv
Geographic Coverage

The MarineCadastre dataset used during development covers the Gulf of Mexico region:

Longitude: -90° to -89°W
Latitude : 28° to 29°N
Dataset Period

The requested AIS period was:

January 1–5, 2022

The downloaded records contain timestamps around the corresponding dataset period.

Dataset Size

The dataset contained approximately:

162,637 AIS records
318 unique vessels
17 original columns

Important AIS fields include:

MMSI
BaseDateTime
LAT
LON
SOG
COG
Heading
VesselName
VesselType
Status
Length
Width
Draft
Cargo
TransceiverClass
Purpose

The real AIS dataset was used for:

AIS trajectory processing
Vessel movement analysis
Feature engineering
Candidate-vessel generation
XGBoost development
Controlled model validation

The Gulf of Mexico dataset was used for development and validation of the AIS methodology. It should not be interpreted as Chennai AIS data.

4. Synthetic Chennai AIS Dataset

A synthetic AIS dataset was created specifically for the Chennai/Ennore prototype demonstration.

This was necessary because the development AIS dataset described above covers the Gulf of Mexico rather than the Chennai region.

Dataset
synthetic_chennai_ais.csv
Geographic Region

The synthetic vessels represent vessel traffic around the Chennai/Ennore coastal region.

Dataset Size
1,225 AIS records
25 vessels
5-minute AIS intervals
Time Range
2026-09-08 04:00 UTC
to
2026-09-08 08:00 UTC
Controlled Source Vessel

One synthetic vessel was intentionally constructed to pass through the estimated spill-source location.

MMSI       : 419000000
Vessel     : CHENNAI TRADER
Vessel Type: Tanker
Speed      : 10 knots
COG        : 45°
Controlled Source Location
Latitude  : 12.22076°N
Longitude : 80.13420°E
Time      : 2026-09-08 06:00 UTC

The vessel CHENNAI TRADER passes through this controlled source location at the source time.

Other synthetic vessels were generated at different distances from the source to create multiple AIS candidates for the vessel-ranking demonstration.

5. AIS Candidate Search Region

Once the Lagrangian model estimates the probable spill source, the AIS module searches for vessels around that source.

The prototype uses:

Spatial search radius : 5 km
Temporal window       : ±30 minutes

Therefore, AIS records are first filtered using:

Distance from estimated source ≤ 5 km
AND
Time difference from estimated source ≤ 30 minutes

The resulting vessels become candidate vessels for the XGBoost model.

6. AIS Features

For each candidate vessel, 18 AIS-derived features are calculated.

These include:

Closest distance to source
Distance at window start
Distance at window end
Distance change
Closest time difference
Time inside source radius
AIS point count
Average SOG
Maximum SOG
Speed at closest approach
Speed change
Average course change
Course change at closest approach
Approach speed
Departure speed
Maximum AIS gap
Long AIS gap count
Moving fraction

These features are passed to the XGBoost model to estimate source consistency and rank candidate vessels.

7. Data Flow

The data flow through SPILLNET is:

Satellite Imagery
       ↓
Oil Spill Detection
       ↓
Detected Spill Location + Time
       ↓
Copernicus Ocean Current Data
       ↓
Lagrangian Backward Tracking
       ↓
Estimated Spill Source
       ↓
AIS Data
       ↓
5 km / ±30 min Candidate Filtering
       ↓
18 AIS Features
       ↓
XGBoost
       ↓
Ranked Potential Source Vessels


# Synthetic Predictive Risk Mapping Dataset

The Predictive Risk Mapping module uses trajectory information from the SPILLNET oil-spill prediction pipeline together with a set of synthetic sensitive-zone data to estimate the potential impact of an oil spill on nearby regions.

### Dataset Overview

The risk-mapping dataset represents geographically sensitive marine and coastal zones that may be affected by the predicted oil trajectory.

The current demonstration dataset contains the following zone categories:

| Zone | Latitude | Longitude | Sensitivity | Priority | Radius (km) |
|------|----------|-----------|-------------|----------|-------------|
| Sensitive Mangrove Zone | 12.15 | 80.45 | 1.00 | 1.00 | 8 |
| Fishing Zone | 12.05 | 80.35 | 0.85 | 0.90 | 10 |
| Coastal Community | 12.00 | 80.55 | 0.80 | 0.85 | 8 |
| Port / Infrastructure | 12.25 | 80.70 | 0.70 | 0.90 | 8 |
| Marine Protected Area | 12.38 | 80.28 | 0.95 | 0.95 | 7 |
| Tourism / Beach Zone | 12.48 | 80.18 | 0.65 | 0.70 | 6 |
| Aquaculture Zone | 12.30 | 80.10 | 0.90 | 0.85 | 7 |
| Shipping Corridor | 12.55 | 80.40 | 0.55 | 0.80 | 9 |

> **Note:** The sensitive-zone coordinates and attributes are synthetic demonstration data created for the SIH 2026 prototype. They are not intended to represent official boundaries or real-world GIS datasets.

### Dataset Fields

- **Zone** – Name/category of the potentially affected region.
- **Latitude** – Central latitude of the zone.
- **Longitude** – Central longitude of the zone.
- **Sensitivity** – Relative environmental or socio-economic sensitivity of the zone.
- **Priority** – Relative response priority assigned to the zone.
- **Radius** – Approximate influence radius used for proximity-based risk calculation.

### Risk Calculation

The predicted oil trajectory generated by the SPILLNET pipeline is compared with the sensitive zones.

For each zone, the system calculates the minimum distance between the predicted oil trajectory and the zone.

A normalized proximity value is calculated as:

Proximity = max(0, 1 - Distance / Radius)

The risk score is then calculated using:

Risk Score = 100 × Proximity × Sensitivity × Priority

The resulting score is classified into three levels:

| Risk Level | Risk Score | Interpretation |
|------------|------------|----------------|
| HIGH | ≥ 65 | High predicted exposure / priority |
| MEDIUM | 35–64.99 | Moderate predicted exposure |
| LOW | < 35 | Lower predicted exposure |

### Dynamic Risk Mapping

The dashboard combines:

1. Detected spill location
2. Backward Lagrangian source trajectory
3. Forward Lagrangian oil trajectory
4. Sensitive-zone locations
5. Distance between the predicted trajectory and sensitive zones
6. Sensitivity and response-priority values

This produces a predictive risk map that helps identify which areas may require closer monitoring or prioritized response.

### Data Sources

The system uses:

- **CMEMS ocean-current data** for ocean velocity information.
- **SPILLNET predicted trajectories** for forward oil movement.
- **Synthetic sensitive-zone data** for the current prototype risk assessment.

The synthetic dataset is designed to demonstrate the risk-mapping workflow and can later be replaced with verified GIS layers such as protected areas, fishing grounds, ports, coastal settlements, aquaculture zones, and other authoritative spatial datasets.

### Purpose

The purpose of this dataset is not to claim that the listed locations are officially designated risk zones. Instead, it provides a controlled demonstration dataset for validating the predictive risk-mapping component of NEXFORGE and showing how predicted oil movement can be translated into location-specific response priorities.
