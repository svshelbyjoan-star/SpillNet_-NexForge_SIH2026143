# SPILLNET Data

This directory documents the datasets used by SPILLNET for satellite-based oil-spill detection, ocean-current modelling, AIS vessel analysis, and the Chennai-region prototype demonstration.

The project combines:

1. Sentinel-1 SAR satellite imagery
2. Copernicus Marine ocean-current data
3. Real AIS data from MarineCadastre
4. Synthetic Chennai/Ennore AIS data

---

## 1. Satellite Imagery Dataset

### Dataset Used

SPILLNET uses the **Sentinel-1 SAR Oil Spill Dataset** published on Zenodo by Trujillo-Acatitla et al.

The dataset was selected because it provides Sentinel-1 Synthetic Aperture Radar (SAR) imagery together with ground-truth masks specifically for oil-spill detection and segmentation.

The SIH26143 problem statement also specifies the Zenodo Sentinel-1 SAR Oil Spill Dataset as the satellite imagery data source.

### Dataset Structure

The complete dataset is divided into three parts:

| Part | Purpose | Data |
|---|---|---|
| Part I | Training and validation | Oil-spill images + masks |
| Part II | Training and validation | No-oil and look-alike images + masks |
| Part III | Testing | Oil, no-oil and look-alike images + masks |

### Image Characteristics

The Sentinel-1 SAR images are provided as:

- Sigma0 backscatter
- Decibel (dB) values
- VV and VH polarizations
- 2048 × 2048 × 2 dimensions
- TIFF format

The corresponding ground-truth masks are 2048 × 2048.

Only the Sentinel-1 SAR images are georeferenced; the ground-truth masks are provided as image matrices for model training, validation and testing.

### Part I — Oil Spill Images

Part I contains:

- 1,200 oil-spill Sentinel-1 SAR images
- 1,200 corresponding ground-truth masks

The masks use:

- `1` → oil-spill region
- `0` → background

### Part II — No-Oil and Look-Alike Images

Part II contains:

- 685 no-oil images
- 685 no-oil masks
- 685 look-alike images
- 685 look-alike masks

Look-alikes are dark SAR features that may resemble oil slicks but are not classified as oil spills.

For the oil-spill detection task, the look-alike ground truth is treated as background.

### Part III — Test Dataset

Part III contains the test data:

| Category | Images | Ground Truth |
|---|---:|---:|
| Oil spill | 150 | 150 |
| No oil | 150 | 150 |
| Look-alike | 150 | 150 |
| **Total** | **450** | **450** |

This provides a separate test set for evaluating the segmentation/detection model.

### Satellite Dataset Source

The three dataset parts are available through Zenodo:

- Part I — `10.5281/zenodo.8346860`
- Part II — `10.5281/zenodo.8253899`
- Part III — `10.5281/zenodo.13761290`

These datasets were published in 2024 by Trujillo-Acatitla, Tuxpan-Vargas, Ovando-Vázquez and Monterrubio-Martínez.

---

## 2. Prototype Spill Region

The satellite-detection module identifies the spill region from the satellite image.

For the Chennai-region prototype demonstration, the detected spill scenario used:

**Detected Spill Location**

Latitude  : 12.30000°N
Longitude : 80.20000°E
Time      : 2026-09-08 16:00 UTC

3. Copernicus Marine Ocean-Current Data

The Lagrangian module uses ocean-current data from the Copernicus Marine Service.

Dataset

Global Ocean Physics Analysis and Forecast

Dataset:

cmems_mod_glo_phy_anfc_merged-uv_PT1H-i
Variables Used
Variable	Description	Unit
uo	Eastward sea-water velocity	m/s
vo	Northward sea-water velocity	m/s

The surface current layer is used to model the movement of the oil.

Geographic Subset
Longitude : 78°E – 82°E
Latitude  : 10°N – 14°N
Temporal Coverage Used
2026-09-08 06:00 UTC
to
2026-09-11 06:00 UTC

The subset contains hourly ocean-current data.

Surface Layer

The prototype uses the surface layer at approximately:

Depth : 0.494 m
Lagrangian Processing

The detected spill location is used as the starting point.

The model performs:

Detected Spill
      ↓
Backward Tracking
      ↓
Estimated Source

and

Detected Spill
      ↓
Forward Tracking
      ↓
Predicted Oil Trajectory
Prototype Estimated Source

For the prototype scenario:

Latitude  : 12.22076°N
Longitude : 80.13420°E
Time      : 2026-09-08 06:00 UTC

The estimated source location and time are passed to the AIS vessel-identification module.

4. Real AIS Dataset
Dataset Source

SPILLNET uses MarineCadastre AIS data for AIS trajectory development and model validation.

The dataset used during development was:

AIS_178896334879276928_691-1788963349118.csv
Geographic Coverage

The downloaded MarineCadastre dataset covers a region in the Gulf of Mexico:

Longitude : -90° to -89°W
Latitude  : 28° to 29°N
Dataset Period

The requested AIS period was:

January 1–5, 2022
Dataset Size
AIS records  : 162,637
Unique MMSIs : 318
Columns      : 17
AIS Information Used

Important fields include:

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

The Gulf of Mexico AIS dataset is a development and validation dataset and is not represented as Chennai AIS data.

5. Synthetic Chennai AIS Dataset

Because the available MarineCadastre dataset covers the Gulf of Mexico rather than Chennai, a synthetic AIS dataset was created for the Chennai/Ennore prototype demonstration.

Dataset
synthetic_chennai_ais.csv
Dataset Size
AIS records : 1,225
Vessels     : 25
Interval    : 5 minutes
Time Range
2026-09-08 04:00 UTC
to
2026-09-08 08:00 UTC
Geographic Region

The synthetic vessels represent vessel traffic around the Chennai/Ennore coastal region.

Controlled Source Vessel

One vessel was intentionally generated to pass through the estimated spill-source location.

MMSI        : 419000000
Vessel Name : CHENNAI TRADER
Vessel Type : Tanker
Speed       : 10 knots
COG         : 45°
Controlled Source Point
Latitude  : 12.22076°N
Longitude : 80.13420°E
Time      : 2026-09-08 06:00 UTC

The CHENNAI TRADER trajectory passes through the controlled source point at the estimated source time.

Additional synthetic vessels were generated at different distances from the source so that the AIS module could demonstrate candidate-vessel filtering and XGBoost ranking.

The synthetic AIS dataset is explicitly used for prototype demonstration and is not presented as real vessel traffic data.

6. AIS Candidate Filtering

After the Lagrangian module estimates the probable spill source, the AIS module searches for vessels around the estimated source.

The prototype uses:

Spatial radius : 5 km
Time window    : ±30 minutes

A vessel becomes an AIS candidate when its AIS records satisfy:

Distance from estimated source ≤ 5 km
AND
Time difference from estimated source ≤ 30 minutes

The candidate vessels are then processed by the feature-engineering and XGBoost modules.

7. AIS Feature Engineering

For every candidate vessel, SPILLNET calculates 18 AIS-derived features.

Spatial Features
Closest distance to source
Distance at window start
Distance at window end
Distance change
Temporal Features
Closest time difference
Time inside source radius
AIS point count
Speed Features
Average SOG
Maximum SOG
Speed at closest approach
Speed change
Approach speed
Departure speed
Behaviour and Continuity Features
Average course change
Course change at closest approach
Maximum AIS gap
Long AIS gap count
Moving fraction

These features are supplied to the XGBoost model to estimate the source consistency of each candidate vessel.

8. Data Usage Summary
Dataset	Region	Purpose
Sentinel-1 SAR Oil Spill Dataset	Dataset-specific SAR scenes	Oil-spill detection and segmentation
Copernicus Marine	78–82°E, 10–14°N	Ocean-current and Lagrangian modelling
MarineCadastre AIS	Gulf of Mexico	AIS development and controlled validation
Synthetic Chennai AIS	Chennai/Ennore region	End-to-end prototype demonstration
9. SPILLNET Data Flow
Sentinel-1 SAR
      ↓
Oil Spill Detection
      ↓
Detected Spill Location + Time
      ↓
Copernicus Ocean Currents
      ↓
Lagrangian Backward / Forward Tracking
      ↓
Estimated Spill Source
      ↓
AIS Candidate Filtering
      ↓
18 AIS Features
      ↓
XGBoost
      ↓
Ranked Potential Source Vessels


## Predictive Risk Mapping Dataset

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

`Proximity = max(0, 1 - Distance / Radius)`

The risk score is then calculated using:

`Risk Score = 100 × Proximity × Sensitivity × Priority`

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
