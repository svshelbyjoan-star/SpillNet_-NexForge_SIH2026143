# SPILLNET Workflow

## Overview

SPILLNET follows a sequential workflow that combines satellite-based oil spill detection, Lagrangian trajectory modelling, AIS analysis, machine learning, and GIS visualization.

## End-to-End Workflow

Satellite Image
      ↓
Oil Spill Detection
      ↓
Spill Location + Timestamp
      ↓
Lagrangian Backtracking
      ↓
Estimated Source Location + Time
      ↓
AIS Candidate Search
      ↓
18-Feature Extraction
      ↓
XGBoost Classification
      ↓
Candidate Vessel Ranking
      ↓
GIS Visualization
      ↓
Dashboard


Step 1 — Oil Spill Detection

Satellite imagery is processed by the spill detection module.

The module identifies a potential oil spill and provides its:

Geographic coordinates
Detection time
Spill region/mask
Confidence information


Step 2 — Source Estimation

The detected spill location and timestamp are passed to the Lagrangian model.

The model uses ocean-current data to simulate oil movement.

Backward Tracking

The model traces the oil backwards from the detected spill location.

Detected Spill
      ↓
Backward Drift
      ↓
Probable Source Location
      +
Estimated Release Time

This estimated source becomes the input for the AIS module.

Forward Tracking

The model also simulates the expected future movement of the oil.

Detected Spill
      ↓
Forward Drift
      ↓
Predicted Oil Trajectory


Step 3 — AIS Candidate Filtering

AIS data is searched around the estimated source.

The current prototype uses:

5 km spatial radius
±30 minute temporal window

Vessels whose AIS observations fall within this source-search region are selected as candidate vessels.


Step 4 — Feature Engineering
Each candidate vessel's AIS track is converted into 18 features.

Spatial Features
Closest distance to estimated source
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
Course and Behaviour Features
Average course change
Course change at closest approach
Moving fraction
AIS Continuity Features
Maximum AIS gap
Long AIS gap count


Step 5 — XGBoost Ranking
The extracted features are passed to the XGBoost model.
The model produces a source-consistency probability for each candidate vessel.
Candidates are sorted in descending order of this score.

Candidate Vessels
       ↓
Feature Matrix
       ↓
XGBoost
       ↓
Source-Consistency Scores
       ↓
Ranked Candidates


Step 6 — Final Output
The AIS module returns the highest-ranked candidate vessels.
The output contains information such as:

Rank
MMSI
Vessel name
Vessel type
Source-consistency score
Distance from estimated source
Time difference
Speed information
Movement fraction


Step 7 — GIS and Dashboard
The results are passed to the visualization layer.
The dashboard can display:

Detected spill
Estimated source
Backward trajectory
Forward oil trajectory
Candidate vessels
Candidate ranking
Source-consistency scores
