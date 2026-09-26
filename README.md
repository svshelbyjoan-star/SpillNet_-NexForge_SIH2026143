SPILLNET — NEXFORGE

AI-Powered Marine Oil-Spill Decision Support System

Smart India Hackathon 2026
Problem Statement:SIH26143  
Theme: Disaster Management  
Domain:Marine Pollution Monitoring

About SPILLNET

Marine oil-spill response requires more than simply detecting an oil spill.

Response teams need to understand:

- Where is the spill?
- Where did the oil probably originate?
- Which vessels were present near the probable source?
- Where could the oil move next?
- Which areas may require attention?

SPILLNET is an AI-powered marine oil-spill decision-support platform developed by Team NEXFORGE.

The system integrates satellite imagery, ocean-current data, AIS vessel data, trajectory modelling and GIS analysis into a unified workflow.

---

Our Approach

SPILLNET follows five major stages:

### 1. DETECT

Satellite SAR imagery is processed using a U-Net deep-learning model to identify the suspected oil-spill region.

The system extracts the approximate spill location, affected region and observation time.

### 2. TRACE

The detected spill location may not be the exact point where the oil entered the sea because oil moves with ocean currents and environmental conditions.

A Lagrangian particle-based approach is used with ocean-current data to:

- Estimate a probable source region
- Estimate a probable source time window
- Reconstruct possible backward movement
- Simulate future oil movement

### 3. IDENTIFY

AIS (Automatic Identification System) data is analysed around the estimated source location and time.

Candidate vessels are filtered using:

- Spatial proximity
- Temporal proximity
- Vessel speed
- Speed changes
- Course changes
- Movement behaviour
- AIS observations around the source

The candidate vessels are then evaluated using an XGBoost machine-learning model and ranked using a source-consistency score.

The system provides potential source vessels for further investigation rather than making a legal attribution.

### 4. PREDICT

The trajectory model estimates the possible future movement of the oil.

This helps understand how the spill may spread under the available environmental conditions.

### 5. PRIORITIZE

The predicted trajectory is combined with GIS-based spatial analysis.

Potentially affected locations such as:

- Coastlines
- Ports
- Fishing areas
- Protected regions
- Coastal infrastructure

can be analysed to support response prioritization.

---

## 🏗️ System Architecture


Satellite SAR Imagery
        │
        ▼
   U-Net Detection
        │
        ▼
   Spill Location
        │
        ▼
Lagrangian Backtracking
        │
        ├──────────────► Probable Source
        │
        ▼
   AIS Candidate Search
        │
        ▼
 Feature Engineering
        │
        ▼
 XGBoost Vessel Ranking
        │
        ▼
 Potential Source Vessels
        │
        ▼
Forward Trajectory Prediction
        │
        ▼
     GIS Analysis
        │
        ▼
 Unified Decision Dashboard

Technology Stack
Artificial Intelligence
U-Net
XGBoost
Python
Data Processing
NumPy
Pandas
AIS data processing
Ocean & Trajectory Modelling
Lagrangian particle-based modelling
Copernicus Marine ocean-current data
Geospatial Analysis
GeoPandas
GIS
Spatial analysis
Backend & Platform
FastAPI
PostgreSQL / PostGIS
Frontend
React
