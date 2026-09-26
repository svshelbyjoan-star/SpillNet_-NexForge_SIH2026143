# SPILLNET Data

This directory contains documentation and references for the datasets used by SPILLNET.

## Data Categories

### AIS Data

AIS data provides vessel movement information used for vessel identification and source-consistency analysis.

Relevant fields include:

- MMSI
- Timestamp
- Latitude
- Longitude
- Speed Over Ground (SOG)
- Course Over Ground (COG)
- Heading
- Vessel information

### Ocean Current Data

Ocean-current data is used by the Lagrangian model to estimate oil movement.

The prototype uses Copernicus Marine ocean-current data containing:

- Eastward velocity (`uo`)
- Northward velocity (`vo`)
- Time
- Latitude
- Longitude
- Depth

### Satellite Data

Satellite imagery is used by the oil-spill detection component to identify potential spill regions.

The resulting spill location and detection timestamp are passed to the trajectory and vessel-identification stages.

### Risk Mapping Data

Risk mapping may use geographic and coastal information to estimate potentially affected areas and support visualization.

## Data Handling

Large datasets and proprietary/model-specific files are not required to be committed directly to this repository.

The repository should contain documentation describing:

- Dataset source
- Dataset purpose
- Required format
- Processing requirements

## Prototype Data

The prototype may use controlled or synthetic data for demonstration and testing.

Synthetic AIS data should be clearly identified as synthetic and should not be presented as real-world vessel observations.
