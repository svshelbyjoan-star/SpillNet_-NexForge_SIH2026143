# SPILLNET Prototype Demo

## Overview

This document describes the prototype demonstration flow for SPILLNET.

The prototype demonstrates the integration of:

- Oil spill source estimation
- Lagrangian ocean-current modelling
- AIS vessel identification
- XGBoost candidate ranking
- GIS/dashboard-ready output

## Demo Scenario

The prototype uses a simulated oil spill event in the Chennai coastal region.

The demonstration provides:

- Detected spill location
- Spill detection time
- Estimated source location
- Estimated source time
- Predicted oil trajectory
- AIS candidate vessels
- Source-consistency scores
- Ranked vessel candidates

## Demonstration Flow

1. Provide Spill Location and Time
              ↓
2. Run Lagrangian Model
              ↓
3. Estimate Probable Source
              ↓
4. Search AIS Data
              ↓
5. Extract Vessel Features
              ↓
6. Run XGBoost
              ↓
7. Rank Candidate Vessels
              ↓
8. Display Results



Lagrangian Demonstration

The Lagrangian module uses ocean-current data to simulate oil movement.

Backward Prediction

The model traces the detected spill backwards to estimate the probable source.

Forward Prediction

The model predicts the future movement of the spill from the detected location.

The demonstration visualizes both trajectories.

AIS Demonstration

The estimated source location and time are passed to the AIS module.

The prototype searches for vessels within the configured:

Spatial search radius
Temporal search window

Candidate vessel tracks are then converted into the 18 AIS-derived model features.

XGBoost Demonstration

The extracted candidate features are provided to the trained XGBoost model.

The model generates a source-consistency score for each candidate vessel.

Candidates are then ranked:

Rank 1 → Highest source-consistency score
Rank 2 → Second highest score
Rank 3 → Third highest score
...
Example Prototype Output
Estimated Source
Latitude  : 12.22076
Longitude : 80.13420
Time      : 2026-09-08 06:00 UTC

Potential Source Vessels

Rank 1
Vessel : CHENNAI TRADER
Type   : Tanker
Score  : 0.9358
Distance: 0.000 km

Rank 2
Vessel : GULF TRADER
Type   : Cargo
Score  : 0.0829
Distance: 1.481 km

The scores represent source consistency within the prototype model and should not be interpreted as proof of legal responsibility.

Prototype Validation

The AIS ranking model was evaluated using controlled realistic source-event simulations generated from real AIS trajectories.

For the V3 evaluation:

300 simulated source events
970 vessel-event candidate records
60 unseen test events
41 multi-candidate test events

The true source vessel was ranked:

Top-1 in 91.67% of unseen test events
Top-3 in 100% of unseen test events

Among the 41 unseen test events containing multiple candidate vessels:

Top-1: 87.8%
Top-3: 100%

These results describe controlled prototype evaluation and are not equivalent to validation against confirmed historical oil-spill attribution cases.

Demonstration Output

The prototype can provide dashboard-ready information including:

Spill coordinates
Estimated source coordinates
Source timestamp
Forward oil trajectory
Candidate vessel ranking
MMSI
Vessel name
Vessel type
Source-consistency score
Distance from estimated source
Time difference
Vessel speed information
