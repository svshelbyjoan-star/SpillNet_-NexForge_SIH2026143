import os
import json
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html

RESULT_PATH = r"C:\Users\sravy\spillnet_result.json"

# Synthetic sensitive-zone data used until real GIS layers are connected.
# These are demonstration coordinates around the Chennai/Puducherry coastal region.
SYNTHETIC_ZONES = [
    {"name":"Sensitive Mangrove Zone","lat":12.15,"lon":80.45,"type":"Mangrove","sensitivity":1.00,"priority":1.00,"radius_km":8.0},
    {"name":"Fishing Zone","lat":12.05,"lon":80.35,"type":"Fishing Zone","sensitivity":0.85,"priority":0.90,"radius_km":10.0},
    {"name":"Coastal Community","lat":12.00,"lon":80.55,"type":"Coastal Community","sensitivity":0.80,"priority":0.85,"radius_km":8.0},
    {"name":"Port / Infrastructure","lat":12.25,"lon":80.70,"type":"Port","sensitivity":0.70,"priority":0.90,"radius_km":8.0},
    {"name":"Marine Protected Area","lat":12.38,"lon":80.28,"type":"Protected Area","sensitivity":0.95,"priority":0.95,"radius_km":7.0},
    {"name":"Tourism / Beach Zone","lat":12.48,"lon":80.18,"type":"Tourism","sensitivity":0.65,"priority":0.70,"radius_km":6.0},
    {"name":"Aquaculture Zone","lat":12.30,"lon":80.10,"type":"Aquaculture","sensitivity":0.90,"priority":0.85,"radius_km":7.0},
    {"name":"Shipping Corridor","lat":12.55,"lon":80.40,"type":"Shipping","sensitivity":0.55,"priority":0.80,"radius_km":9.0},
]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*r*math.asin(math.sqrt(a))


def load_result():
    if not os.path.exists(RESULT_PATH):
        raise FileNotFoundError(
            f"Could not find {RESULT_PATH}. Run untitled3.py first."
        )
    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


result = load_result()
spill = result.get("detected_spill", {})
source = result.get("estimated_source", {})
pred = result.get("forward_prediction", {})
backward = result.get("backward_trajectory", [])
forward = result.get("forward_trajectory", [])
candidates = result.get("candidate_vessels", [])

# The pipeline currently returns two ranked source candidates. To keep the
# operational map useful, add three nearby AIS-context vessels as demo entries
# when fewer than five are available. These are clearly marked as synthetic.
SYNTHETIC_AIS = [
    {"rank": 3, "VesselName": "EASTERN STAR", "VesselTypeLabel": "Cargo", "source_probability": 0.091, "closest_distance_km": 3.2, "average_sog": 7.8, "lat": 12.34, "lon": 80.23},
    {"rank": 4, "VesselName": "COASTAL EXPRESS", "VesselTypeLabel": "Tanker", "source_probability": 0.074, "closest_distance_km": 4.7, "average_sog": 8.6, "lat": 12.27, "lon": 80.18},
    {"rank": 5, "VesselName": "BAY MERCHANT", "VesselTypeLabel": "Cargo", "source_probability": 0.052, "closest_distance_km": 6.1, "average_sog": 6.4, "lat": 12.40, "lon": 80.31},
]

all_vessels = [dict(v) for v in candidates]
while len(all_vessels) < 5:
    idx = len(all_vessels) - 2
    if 0 <= idx < len(SYNTHETIC_AIS):
        all_vessels.append(SYNTHETIC_AIS[idx])
    else:
        break

spill_lat = float(spill.get("latitude", 12.3))
spill_lon = float(spill.get("longitude", 80.2))
pred_lat = float(pred.get("latitude", spill_lat))
pred_lon = float(pred.get("longitude", spill_lon))

backward_df = pd.DataFrame(backward)
forward_df = pd.DataFrame(forward)

if not forward_df.empty:
    forward_df["time"] = pd.to_datetime(forward_df["time"])
    t0 = forward_df["time"].iloc[0]
    forward_df["hours"] = (forward_df["time"] - t0).dt.total_seconds() / 3600.0
    forward_df["movement_km"] = [
        haversine_km(spill_lat, spill_lon, a, o)
        for a, o in zip(forward_df["latitude"], forward_df["longitude"])
    ]
else:
    forward_df = pd.DataFrame(columns=["latitude","longitude","time","hours","movement_km"])


def calculate_risk(zones, trajectory):
    rows = []
    for z in zones:
        min_dist = float("inf")
        eta = None
        for _, p in trajectory.iterrows():
            d = haversine_km(p["latitude"], p["longitude"], z["lat"], z["lon"])
            if d < min_dist:
                min_dist = d
                eta = float(p["hours"])
        # Proximity factor: 1 at the zone centre, 0 at/outside the zone radius.
        proximity = max(0.0, 1.0 - min_dist / z["radius_km"])
        risk = 100.0 * proximity * z["sensitivity"] * z["priority"]
        if risk >= 65:
            level = "HIGH"
        elif risk >= 35:
            level = "MEDIUM"
        else:
            level = "LOW"
        rows.append({
            **z,
            "min_distance_km": min_dist,
            "eta_hours": eta,
            "risk_score": risk,
            "risk_level": level,
        })
    return pd.DataFrame(rows).sort_values("risk_score", ascending=False)


risk_df = calculate_risk(SYNTHETIC_ZONES, forward_df)

# Build a smooth synthetic predictive risk surface around the predicted trajectory.
# It is a visualization layer, while zone scores above are the auditable zone-level result.
def risk_surface():
    lat_min, lat_max = 11.85, 12.70
    lon_min, lon_max = 79.90, 80.85
    lats = np.linspace(lat_min, lat_max, 80)
    lons = np.linspace(lon_min, lon_max, 90)
    zz = np.zeros((len(lats), len(lons)))
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            # Contribution from predicted trajectory points.
            trajectory_risk = 0.0
            if not forward_df.empty:
                for _, p in forward_df.iterrows():
                    d = haversine_km(lat, lon, p["latitude"], p["longitude"])
                    trajectory_risk = max(trajectory_risk, math.exp(-(d*d)/(2*18.0**2)) * 55.0)
            # Contribution from synthetic sensitive zones.
            zone_risk = 0.0
            for z in SYNTHETIC_ZONES:
                d = haversine_km(lat, lon, z["lat"], z["lon"])
                zone_risk = max(zone_risk, 100*z["sensitivity"]*z["priority"]*math.exp(-(d*d)/(2*(z["radius_km"]*0.75)**2)))
            zz[i,j] = min(100.0, max(trajectory_risk, zone_risk))
    return lats, lons, zz

lats, lons, risk_grid = risk_surface()


def risk_map():
    # Keep the original-style geographic map: OSM basemap + clean markers/lines.
    fig = go.Figure()

    if not backward_df.empty:
        fig.add_trace(go.Scattermap(
            lat=backward_df["latitude"], lon=backward_df["longitude"],
            mode="lines+markers", name="Backward Source Trace",
            line=dict(width=3), marker=dict(size=5),
            hovertemplate="Source trace<br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>"
        ))

    if not forward_df.empty:
        fig.add_trace(go.Scattermap(
            lat=forward_df["latitude"], lon=forward_df["longitude"],
            mode="lines+markers", name="Predicted Oil Trajectory",
            line=dict(width=5), marker=dict(size=7),
            customdata=np.c_[forward_df["hours"], forward_df["movement_km"]],
            hovertemplate="Predicted oil<br>+%{customdata[0]:.1f} h<br>Distance: %{customdata[1]:.2f} km<br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>"
        ))

    # Risk zones: transparent circles make the highest-risk area visually obvious.
    for _, r in risk_df.iterrows():
        level = r["risk_level"]
        size = 24 if level == "HIGH" else 18 if level == "MEDIUM" else 12
        fig.add_trace(go.Scattermap(
            lat=[r["lat"]], lon=[r["lon"]], mode="markers+text",
            text=[f"{level} • {r['name']}" if level in ("HIGH", "MEDIUM") else ""],
            textposition="top center", name=f"{level} Risk Zone",
            showlegend=False, marker=dict(size=size, opacity=0.78),
            hovertemplate=(f"<b>{r['name']}</b><br>Risk: {r['risk_score']:.0f}/100"
                          f"<br>Level: {level}<br>Nearest trajectory: {r['min_distance_km']:.2f} km"
                          f"<br>ETA: {r['eta_hours']:.1f} h<extra></extra>")
        ))

    # Five AIS vessels. First entries are pipeline-ranked; remaining are synthetic context.
    vessel_lats = [12.2208, 12.235, 12.34, 12.27, 12.40]
    vessel_lons = [80.1342, 80.155, 80.23, 80.18, 80.31]
    for i, v in enumerate(all_vessels[:5]):
        lat = float(v.get("lat", vessel_lats[i]))
        lon = float(v.get("lon", vessel_lons[i]))
        actual = i < len(candidates)
        fig.add_trace(go.Scattermap(
            lat=[lat], lon=[lon], mode="markers+text",
            text=[v.get("VesselName", "Unknown")], textposition="bottom center",
            name="AIS Source Candidates" if i == 0 else None, showlegend=(i == 0),
            marker=dict(size=11, symbol="circle"),
            hovertemplate=(f"<b>{v.get('VesselName','Unknown')}</b><br>"
                           f"Type: {v.get('VesselTypeLabel','—')}<br>"
                           f"Source probability: {float(v.get('source_probability',0))*100:.2f}%<br>"
                           f"Distance: {float(v.get('closest_distance_km',0)):.2f} km<br>"
                           f"Avg speed: {float(v.get('average_sog',0)):.2f} kn<br>"
                           f"{'Pipeline-ranked candidate' if actual else 'Synthetic AIS context'}<extra></extra>")
        ))

    fig.add_trace(go.Scattermap(
        lat=[spill_lat], lon=[spill_lon], mode="markers", name="Detected Spill",
        marker=dict(size=18, symbol="star"),
        hovertemplate="<b>DETECTED SPILL</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>"
    ))
    fig.add_trace(go.Scattermap(
        lat=[float(source.get("latitude", spill_lat))], lon=[float(source.get("longitude", spill_lon))],
        mode="markers", name="Estimated Source", marker=dict(size=14, symbol="diamond"),
        hovertemplate="<b>ESTIMATED SOURCE</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>"
    ))
    fig.add_trace(go.Scattermap(
        lat=[pred_lat], lon=[pred_lon], mode="markers", name="Forecast Position",
        marker=dict(size=15, symbol="triangle"),
        hovertemplate="<b>FORECAST POSITION</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title="Oil Spill Source, Trajectory & Predictive Risk Map",
        height=680, margin=dict(l=0,r=0,t=55,b=0),
        map=dict(
            style="open-street-map",
            center=dict(lat=12.25, lon=80.25),
            zoom=8.2
        ),
        legend=dict(orientation="h", y=1.01, x=0),
        paper_bgcolor="white"
    )
    return fig


def trajectory_graph():
    fig = go.Figure()
    if not forward_df.empty:
        fig.add_trace(go.Scatter(
            x=forward_df["hours"], y=forward_df["movement_km"],
            mode="lines+markers", name="Predicted Movement"
        ))
    fig.update_layout(
        title="Predicted Spill Movement",
        xaxis_title="Forecast Time (hours)",
        yaxis_title="Distance from Detected Spill (km)",
        template="plotly_white", height=350
    )
    return fig


def zone_table():
    if risk_df.empty:
        return html.P("No risk zones available.")
    header = html.Tr([html.Th(c) for c in ["Zone", "Type", "Risk", "Level", "Nearest Distance", "ETA"]])
    rows = []
    for _, r in risk_df.iterrows():
        eta = "—" if pd.isna(r["eta_hours"]) else f"{r['eta_hours']:.1f} h"
        rows.append(html.Tr([
            html.Td(r["name"]), html.Td(r["type"]),
            html.Td(f"{r['risk_score']:.1f}/100"), html.Td(r["risk_level"]),
            html.Td(f"{r['min_distance_km']:.2f} km"), html.Td(eta)
        ]))
    return html.Table([html.Thead(header), html.Tbody(rows)], style={"width":"100%","borderCollapse":"collapse"})


def vessel_table():
    rows = []
    for v in all_vessels[:5]:
        prob = float(v.get("source_probability", 0))*100
        rows.append(html.Tr([
            html.Td(v.get("rank", "—")),
            html.Td(v.get("VesselName", "Unknown")),
            html.Td(v.get("VesselTypeLabel", "—")),
            html.Td(f"{prob:.2f}%"),
            html.Td(f"{float(v.get('closest_distance_km', 0)):.2f} km"),
            html.Td(f"{float(v.get('average_sog', 0)):.2f}"),
        ]))
    return html.Div([
        html.Table([
            html.Thead(html.Tr([html.Th(x) for x in ["Rank","Vessel","Type","Source Probability","Distance","Avg SOG"]])),
            html.Tbody(rows)
        ], style={"width":"100%","borderCollapse":"collapse"}),
        html.Small("Top 2 are pipeline-ranked candidates; remaining AIS entries are synthetic context for the demo.", style={"color":"#667085"})
    ])


def card(title, body):
    return html.Div([
        html.H3(title, style={"marginTop":0}), body
    ], style={"background":"white","padding":"20px","borderRadius":"14px","boxShadow":"0 2px 10px rgba(0,0,0,.08)"})


highest = risk_df.iloc[0] if not risk_df.empty else None
highest_text = f"{highest['risk_level']} RISK" if highest is not None else "N/A"
highest_score = f"{highest['risk_score']:.0f}/100" if highest is not None else "N/A"
highest_zone = highest["name"] if highest is not None else "No zone identified"
highest_eta = f"{highest['eta_hours']:.1f} h" if highest is not None and not pd.isna(highest["eta_hours"]) else "—"
forecast_hours = float(forward_df["hours"].max()) if not forward_df.empty else 0

app = Dash(__name__)
app.title = "NEXFORGE — Oil Spill Intelligence"

app.layout = html.Div([
    html.Div([
        html.H1("NEXFORGE", style={"margin":0}),
        html.Div("OIL SPILL INTELLIGENCE & RESPONSE SYSTEM", style={"letterSpacing":"2px"})
    ], style={"background":"#0b1f33","color":"white","padding":"24px 35px"}),

    html.Div([
        card("DETECTED SPILL", html.H2(f"{spill_lat:.4f}° N, {spill_lon:.4f}° E")),
        card("FORECAST HORIZON", html.H2(f"{forecast_hours:.1f} h")),
        card("TOP SOURCE VESSEL", html.H2(f"{float(candidates[0].get('source_probability',0))*100:.2f}%" if candidates else "N/A")),
        card("HIGHEST PREDICTIVE RISK", html.Div([html.H2(highest_text, style={"margin":"0 0 4px"}), html.Div(highest_zone, style={"fontWeight":"bold"}), html.Div(f"Risk score: {highest_score}  |  ETA: {highest_eta}", style={"marginTop":"6px","color":"#475467"})])),
    ], style={"display":"grid","gridTemplateColumns":"repeat(4,1fr)","gap":"16px","padding":"20px 35px"}),

    html.Div([
        card("🗺️ Predictive Risk Mapping", html.Div([
        dcc.Graph(figure=risk_map(), config={"displayModeBar":True}),
        html.Div([
            html.B("Risk legend: "), html.Span("HIGH = highest priority exposure • MEDIUM = monitor • LOW = lower predicted exposure"),
            html.Br(), html.B("Map layers: "), html.Span("OSM map + detected spill + backward source trace + forward oil trajectory + AIS vessels + risk zones")
        ], style={"padding":"8px 12px","fontSize":"13px","color":"#475467"})
    ])),
    ], style={"padding":"0 35px"}),

    html.Div([
        card("🎯 Risk-Zone Assessment", zone_table()),
        card("🚢 Probable Source Vessels", vessel_table()),
        card("⚠️ Response Intelligence", html.Div([
            html.Div([
                html.H2(highest_text, style={"margin":"0","fontSize":"30px"}),
                html.H3(highest_zone, style={"margin":"6px 0"}),
                html.Div(f"Risk score: {highest_score}", style={"fontSize":"22px","fontWeight":"bold"}),
                html.Div(f"Predicted arrival: {highest_eta}", style={"marginTop":"6px"}),
                html.Div(f"Nearest trajectory distance: {highest['min_distance_km']:.2f} km" if highest is not None else "", style={"marginTop":"4px"}),
            ], style={"padding":"12px","borderRadius":"10px","background":"#fff3cd" if highest is not None and highest['risk_level']=='HIGH' else "#f2f4f7"}),
            html.P("How to read it: HIGH = immediate priority, MEDIUM = monitor closely, LOW = lower predicted exposure. Synthetic zones are demo data."),
        ]))
    ], style={"display":"grid","gridTemplateColumns":"1.5fr 1.5fr 1fr","gap":"18px","padding":"20px 35px"}),

    html.Div([
        card("📈 Predicted Spill Movement", dcc.Graph(figure=trajectory_graph()))
    ], style={"padding":"0 35px 30px"})
], style={"background":"#f4f7fa","minHeight":"100vh","fontFamily":"Arial"})

if __name__ == "__main__":
    app.run(debug=True, port=8050)
