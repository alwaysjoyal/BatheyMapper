# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 15:37:14 2026

@author: Joyal
"""

# -*- coding: utf-8 -*-
# 3D Land Survey / Bathymetry Elevation Mapper — Streamlit Edition
# Ported from the original Tkinter desktop tool.
#
# Run locally with:
#     pip install -r requirements.txt
#     streamlit run streamlit_app.py

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.cm as cm
from pyproj import Transformer

st.set_page_config(page_title="3D Land Survey Elevation Mapper", page_icon="🌊", layout="wide")

LINKEDIN_URL = "https://www.linkedin.com/in/joyal-joseph-76674a23b/"

# ----------------------------------------------------------------------
# Session state init
# ----------------------------------------------------------------------
for key, default in {
    "df": None,
    "input_filename": None,
    "fig": None,
    "computed_elevation": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_xyz(uploaded_file) -> pd.DataFrame:
    """Read an XYZ/TXT/CSV file into a DataFrame with columns Easting, Northing, Elevation."""
    raw = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    # Try comma-delimited first, then whitespace
    for sep in [",", None]:
        try:
            df = pd.read_csv(io.StringIO(raw), sep=sep, engine="python", header=None, comment="#")
            df = df.iloc[:, :3]
            df.columns = ["Easting", "Northing", "Elevation"]
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            if len(df) > 0:
                return df.reset_index(drop=True)
        except Exception:
            continue
    raise ValueError("Could not parse file. Expected 3 numeric columns: Easting Northing Elevation.")


def utm_to_lonlat(easting, northing, zone, hemisphere):
    """Vectorized UTM -> lon/lat conversion using a modern pyproj Transformer (replaces deprecated pyproj.transform)."""
    south = hemisphere == "Southern"
    crs_utm = f"+proj=utm +zone={zone} +{'south' if south else ''} +ellps=WGS84 +units=m +no_defs"
    transformer = Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(np.asarray(easting), np.asarray(northing))
    return lon, lat


def elevation_annotations(title, subtitle, min_txt, max_txt,
                           utm_min, utm_max, geo_min, geo_max, show_utm, show_geo):
    anns = [
        dict(x=0.5, y=0.9, xref="paper", yref="paper", text=subtitle,
             showarrow=False, font=dict(size=12, color="black")),
        dict(x=0.0, y=0.2, xref="paper", yref="paper", text=min_txt,
             showarrow=False, font=dict(size=12, color="black"), xanchor="left"),
        dict(x=0.0, y=0.98, xref="paper", yref="paper", text=max_txt,
             showarrow=False, font=dict(size=12, color="black"), xanchor="left"),
    ]
    if show_utm:
        anns += [
            dict(x=0.0, y=0.15, xref="paper", yref="paper", text="UTM Coordinates:",
                 showarrow=False, font=dict(size=12, color="black"), xanchor="left"),
            dict(x=0.0, y=0.11, xref="paper", yref="paper", text=utm_min,
                 showarrow=False, font=dict(size=12, color="black"), xanchor="left"),
            dict(x=0.0, y=0.93, xref="paper", yref="paper", text="UTM Coordinates:",
                 showarrow=False, font=dict(size=12, color="black"), xanchor="left"),
            dict(x=0.0, y=0.89, xref="paper", yref="paper", text=utm_max,
                 showarrow=False, font=dict(size=12, color="black"), xanchor="left"),
        ]
    if show_geo:
        anns += [
            dict(x=0.0, y=0.07, xref="paper", yref="paper", text="Geographic Coordinates:",
                 showarrow=False, font=dict(size=12, color="black"), xanchor="left"),
            dict(x=0.0, y=0.03, xref="paper", yref="paper", text=geo_min,
                 showarrow=False, font=dict(size=12, color="black"), xanchor="left"),
            dict(x=0.0, y=0.85, xref="paper", yref="paper", text="Geographic Coordinates:",
                 showarrow=False, font=dict(size=12, color="black"), xanchor="left"),
            dict(x=0.0, y=0.81, xref="paper", yref="paper", text=geo_max,
                 showarrow=False, font=dict(size=12, color="black"), xanchor="left"),
        ]
    anns.append(
        dict(x=1, y=0, xref="paper", yref="paper",
             text=f'<a href="{LINKEDIN_URL}">Feedback: Joyal</a>',
             showarrow=False, font=dict(size=10, color="gray"),
             xanchor="right", yanchor="bottom")
    )
    return anns


def build_layout(fig, easting, northing, elevation, title, subtitle, annotations,
                  invert_z=False, z_axis_title="Elevation"):
    z_low, z_high = min(elevation) - 5, max(elevation) + 5
    z_range = [z_high, z_low] if invert_z else [z_low, z_high]
    fig.update_layout(
        scene=dict(
            xaxis_title="Easting",
            yaxis_title="Northing",
            zaxis_title=z_axis_title,
            zaxis=dict(range=z_range, autorange=False, nticks=7),
            aspectratio=dict(x=1, y=1, z=0.5),
            xaxis=dict(tickformat=".0f", tickvals=np.linspace(min(easting), max(easting), 6)),
            yaxis=dict(tickformat=".0f", tickvals=np.linspace(min(northing), max(northing), 6)),
        ),
        title=title,
        title_font_size=16,
        title_font_color="black",
        title_x=0.5,
        title_y=0.95,
        title_xanchor="center",
        title_yanchor="top",
        annotations=annotations,
        margin=dict(l=0, r=0, t=60, b=0),
        height=750,
    )
    return fig


# ----------------------------------------------------------------------
# Sidebar — data & settings
# ----------------------------------------------------------------------
st.sidebar.title("⛰️ Survey Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload survey file (.xyz, .txt, .csv)", type=["xyz", "txt", "csv"]
)

if uploaded_file is not None:
    if st.session_state["input_filename"] != uploaded_file.name:
        try:
            df = load_xyz(uploaded_file)
            st.session_state["df"] = df
            st.session_state["input_filename"] = uploaded_file.name
            st.session_state["computed_elevation"] = None
            st.sidebar.success(f"Loaded {len(df)} points from {uploaded_file.name}")
        except Exception as e:
            st.sidebar.error(f"Failed to load data: {e}")

df = st.session_state["df"]

st.sidebar.markdown("---")
st.sidebar.subheader("Reference Datum")
apply_datum = st.sidebar.checkbox("Apply datum adjustment", value=False)
datum_level = st.sidebar.number_input("Reference Datum Level", value=0.0, step=0.1, disabled=not apply_datum)
current_level = st.sidebar.number_input("Current Reference Level", value=0.0, step=0.1, disabled=not apply_datum)

st.sidebar.markdown("---")
st.sidebar.subheader("Coordinate Conversion")
do_conversion = st.sidebar.checkbox("Convert UTM → Lat/Lon", value=False)
utm_zone = st.sidebar.number_input("UTM Zone", min_value=1, max_value=60, value=43, disabled=not do_conversion)
hemisphere = st.sidebar.radio("Hemisphere", ["Northern", "Southern"], horizontal=True, disabled=not do_conversion)

st.sidebar.markdown("---")
st.sidebar.subheader("Plot Labels")
title = st.sidebar.text_input("Title", value="3D Elevation Survey")
subtitle = st.sidebar.text_input("Subtitle", value="")

st.sidebar.markdown("---")
st.sidebar.subheader("Display Options")
show_min = st.sidebar.checkbox("Show Min Elevation", value=True)
show_max = st.sidebar.checkbox("Show Max Elevation", value=True)
show_utm = st.sidebar.checkbox("Show UTM Coordinates", value=True)
show_geo = st.sidebar.checkbox("Show Geographic Coordinates", value=True)
show_markers = st.sidebar.checkbox("Show Min/Max Markers", value=True)
colorscale = st.sidebar.selectbox(
    "Colorscale", ["earth", "viridis", "turbo", "thermal", "ice", "portland", "algae"], index=0
)

plot_mode = st.sidebar.radio(
    "Plot Type", ["Mesh Surface", "Numeric Values (text)", "Point Cloud"], index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("Depth Axis")
invert_z = st.sidebar.checkbox(
    "Invert depth axis (deeper values at bottom)", value=False,
    help="For bathymetry: makes larger depth values sit lower on screen, resembling the seabed. "
         "Leave off for standard elevation data where higher = up."
)
z_axis_label = st.sidebar.text_input("Depth/Elevation axis label", value="Depth")

st.sidebar.markdown(
    f"<div style='text-align:right; color:gray; font-size:12px;'>"
    f"Feedback: <a href='{LINKEDIN_URL}' target='_blank'>Joyal</a></div>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------
st.title("🌊 3D Land Survey Elevation Mapper")

if df is None:
    st.info("**Easting   Northing   Depth**  \n(whitespace or comma separated)")
    st.stop()

# Apply datum adjustment (non-destructive: keep raw + computed separately)
if apply_datum:
    constant = datum_level - current_level
    elevation = (df["Elevation"] + constant).to_numpy()
else:
    constant = 0.0
    elevation = df["Elevation"].to_numpy()

st.session_state["computed_elevation"] = elevation

easting = df["Easting"].to_numpy()
northing = df["Northing"].to_numpy()

min_idx, max_idx = int(np.argmin(elevation)), int(np.argmax(elevation))
min_elev, max_elev = float(elevation[min_idx]), float(elevation[max_idx])

# Quick stats row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Points", f"{len(elevation):,}")
c2.metric("Min Elevation", f"{min_elev:.2f}")
c3.metric("Max Elevation", f"{max_elev:.2f}")
c4.metric("Mean Elevation", f"{np.mean(elevation):.2f}")

# Coordinate conversion (for annotations + optional preview map)
lon_min = lat_min = lon_max = lat_max = None
lon_all = lat_all = None
if do_conversion:
    try:
        lon_all, lat_all = utm_to_lonlat(easting, northing, utm_zone, hemisphere)
        lon_min, lat_min = lon_all[min_idx], lat_all[min_idx]
        lon_max, lat_max = lon_all[max_idx], lat_all[max_idx]
    except Exception as e:
        st.warning(f"Coordinate conversion failed: {e}")
        do_conversion = False

min_elevation_text = f"Min Elevation: {min_elev:.1f}" if show_min else ""
max_elevation_text = f"Max Elevation: {max_elev:.1f}" if show_max else ""
utm_min_text = f" Easting: {easting[min_idx]:.1f}, Northing: {northing[min_idx]:.1f}"
utm_max_text = f" Easting: {easting[max_idx]:.1f}, Northing: {northing[max_idx]:.1f}"
if do_conversion:
    geo_min_text = f" Longitude: {lon_min:.6f}, Latitude: {lat_min:.6f}"
    geo_max_text = f" Longitude: {lon_max:.6f}, Latitude: {lat_max:.6f}"
else:
    geo_min_text = geo_max_text = " N/A (enable UTM → Lat/Lon conversion)"

annotations = elevation_annotations(
    title, subtitle, min_elevation_text, max_elevation_text,
    utm_min_text, utm_max_text, geo_min_text, geo_max_text, show_utm, show_geo
)

# ----------------------------------------------------------------------
# Build the plot
# ----------------------------------------------------------------------
fig = go.Figure()

if plot_mode == "Mesh Surface":
    fig.add_trace(go.Mesh3d(
        x=easting, y=northing, z=elevation,
        intensity=elevation,
        colorscale=colorscale,
        opacity=1.0,
        colorbar=dict(title=z_axis_label),
    ))
elif plot_mode == "Point Cloud":
    fig.add_trace(go.Scatter3d(
        x=easting, y=northing, z=elevation,
        mode="markers",
        marker=dict(
            size=4, color=elevation, colorscale=colorscale, opacity=0.85,
            colorbar=dict(title=z_axis_label),
        ),
    ))
else:  # Numeric Values (text)
    norm = (elevation - min_elev) / (max_elev - min_elev) if max_elev > min_elev else np.zeros_like(elevation)
    try:
        cmap = cm.colormaps[colorscale]
    except KeyError:
        cmap = cm.colormaps["terrain"]
    terrain_colors = cmap(norm)
    rgb_colors = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in terrain_colors]
    fig.add_trace(go.Scatter3d(
        x=easting, y=northing, z=elevation,
        mode="markers+text",
        text=[f"{e:.1f}" for e in elevation],
        textposition="top center",
        textfont=dict(color=rgb_colors, size=10),
        marker=dict(color=elevation, colorscale=colorscale, size=6, opacity=0.8,
                    colorbar=dict(title=z_axis_label)),
    ))

if show_markers:
    fig.add_trace(go.Scatter3d(
        x=[easting[min_idx], easting[max_idx]],
        y=[northing[min_idx], northing[max_idx]],
        z=[min_elev, max_elev],
        mode="markers",
        marker=dict(symbol="circle", color="black", size=6),
        name="Min/Max Elevation",
    ))

fig = build_layout(fig, easting, northing, elevation, title, subtitle, annotations,
                    invert_z=invert_z, z_axis_title=z_axis_label)
st.session_state["fig"] = fig

st.plotly_chart(fig, use_container_width=True)

# Elevation histogram
with st.expander("📊 Elevation distribution"):
    st.bar_chart(pd.DataFrame({"Elevation": elevation}), y="Elevation")

# Optional lat/lon preview map
if do_conversion and lon_all is not None:
    with st.expander("🗺️ Geographic preview (converted points)"):
        st.map(pd.DataFrame({"lat": lat_all, "lon": lon_all}), size=3)

# ----------------------------------------------------------------------
# Downloads
# ----------------------------------------------------------------------
st.markdown("---")
st.subheader("Export")

dl1, dl2 = st.columns(2)

with dl1:
    html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
    out_name = (st.session_state["input_filename"] or "plot").rsplit(".", 1)[0] + ".html"
    st.download_button(
        "⬇️ Download interactive HTML plot", data=html_bytes,
        file_name=out_name, mime="text/html",
    )

with dl2:
    report_df = pd.DataFrame({
        "Easting": easting,
        "Northing": northing,
        "Input_Elevation": df["Elevation"].to_numpy(),
        "Computed_Elevation": elevation,
    })
    csv_bytes = report_df.to_csv(index=False, sep=" ", float_format="%.2f").encode("utf-8")
    report_name = "computed_" + (st.session_state["input_filename"] or "report").rsplit(".", 1)[0] + ".xyz"
    st.download_button(
        "⬇️ Download elevation report (.xyz)", data=csv_bytes,
        file_name=report_name, mime="text/plain",
    )

st.markdown(f"Feedback: [Joyal]({LINKEDIN_URL})")
