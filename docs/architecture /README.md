# SPILLNET System Architecture

## Overview

SPILLNET is a software-based oil spill detection and vessel identification system developed for SIH 2026 problem statement SIH26143.

The system combines satellite imagery, oil-spill trajectory modelling, AIS vessel data, machine learning, and GIS-based visualization to identify potential vessels associated with a detected oil spill.

## System Architecture
Satellite Imagery
       │
       ▼
Oil Spill Detection
(U-Net / Deep Learning)
       │
       ▼
Detected Spill
Location + Time
       │
       ▼
Lagrangian Drift Model
       │
       ├── Backward Tracking
       │       │
       │       ▼
       │   Estimated Source
       │
       └── Forward Tracking
               │
               ▼
        Predicted Oil Trajectory
               │
               ▼
        AIS Vessel Identification
               │
               ▼
       Candidate Vessel Filtering
               │
               ▼
        Feature Engineering
               │
               ▼
            XGBoost
               │
               ▼
     Ranked Candidate Vessels
               │
               ▼
          GIS / Dashboard

Major Components

1. Satellite Oil Spill Detection
Satellite imagery is processed using a deep-learning based segmentation model to detect regions that are potentially affected by oil spills.

The detection stage provides:
Spill location
Detection timestamp
Spill mask
Spill area
Detection confidence


2. Lagrangian Oil Drift Modelling
The Lagrangian model estimates how the detected oil could have moved under ocean-current conditions.

Two directions are used:

Backward tracking to estimate the probable source location and time.
Forward tracking to predict the future movement of the detected spill.

Ocean current data is obtained from Copernicus Marine data products.


3. AIS Vessel Identification
The estimated source location and time are passed to the AIS module.

AIS data is filtered using:
Spatial proximity to the estimated source
Temporal proximity to the estimated release time
Candidate vessel tracks are then converted into behavioural and trajectory-based features.



4. XGBoost Candidate Ranking
An XGBoost classifier is used to estimate a source-consistency probability for each candidate vessel.
The model uses 18 AIS-derived features covering:

Spatial proximity
Temporal proximity
Vessel movement
Speed behaviour
Course behaviour
AIS continuity

The output is a ranked list of potential source vessels.


5. GIS and Dashboard
The system output can be visualized through a GIS/dashboard layer containing:

Detected spill location
Estimated source location
Oil trajectory
Candidate vessel locations
Vessel ranking
Source-consistency score



Data Flow
Satellite Data
      ↓
Spill Detection
      ↓
Spill Coordinates + Timestamp
      ↓
Lagrangian Model
      ↓
Estimated Source
      ↓
AIS Search Window
      ↓
Candidate Vessels
      ↓
18-Feature Extraction
      ↓
XGBoost
      ↓
Ranked Candidates
      ↓
GIS / Dashboard
