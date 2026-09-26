# Spill Detection Model

This directory contains documentation and model assets associated with the deep-learning-based oil-spill detection module of SPILLNET.

## Model

The spill detection module uses a **U-Net-based semantic segmentation model** to identify potential oil-spill regions in Sentinel-1 SAR imagery.

Unlike a simple image-level classifier, the segmentation model produces a pixel-level prediction of the suspected spill region.

## Input

The model processes Sentinel-1 SAR imagery containing:

- VV polarization
- VH polarization
- Sigma0 backscatter values

The satellite dataset used for development and evaluation is documented in:

`/data/README.md`

## Output

The model produces a predicted segmentation mask representing the detected oil-spill region.

The detected region is subsequently used to obtain the information required by the downstream SPILLNET modules, including:

- Spill location
- Spill region
- Approximate affected area
- Observation information

## SPILLNET Integration

The model forms the first computational stage of the SPILLNET pipeline:


Sentinel-1 SAR
      ↓
U-Net Segmentation
      ↓
Predicted Spill Mask
      ↓
Spill Detection Output
      ↓
Lagrangian Source Estimation


##Model Role

The U-Net model is responsible for:

Detecting potential oil-spill regions.
Producing pixel-level segmentation.
Providing the detected spill information to the trajectory module.

It does not perform vessel identification or source attribution.

SPILLNET uses **XGBoost** to estimate the source consistency of AIS candidate vessels identified near the probable spill source.

The model does not directly determine legal responsibility for an oil spill.

Instead, it produces a source-consistency score that is used to rank potential source vessels for further investigation.

## Input

Each candidate vessel is represented using 18 AIS-derived features.

### Spatial Features

- Closest distance to source
- Distance at window start
- Distance at window end
- Distance change

### Temporal Features

- Closest time difference
- Time inside source radius
- AIS point count

### Speed Features

- Average SOG
- Maximum SOG
- Speed at closest approach
- Speed change
- Approach speed
- Departure speed

### Behaviour Features

- Average course change
- Course change at closest approach
- Moving fraction

### AIS Continuity Features

- Maximum AIS gap
- Long AIS gap count

## Model Configuration

The XGBoost prototype uses:

n_estimators       = 300
max_depth          = 4
learning_rate      = 0.05
subsample          = 0.8
colsample_bytree   = 0.8
objective          = binary:logistic
eval_metric        = logloss
random_state       = 42
