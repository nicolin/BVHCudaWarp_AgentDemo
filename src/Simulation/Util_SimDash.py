"""
UCLARC: Nicolin Govender
5/5/26

Viewer for checking the GPU simulation correctly does the spatial queries

Loads the complete semantic BVH (.npz) including Room Labels and Beds
Overlays agent trajectories and drops permanent, labeled markers for
Transmission Events based on the GPU physics engine output
"""

import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from enum import IntEnum
import cv2
import io
from PIL import Image

#=====================================================================
# Enums and Config
#=====================================================================
class InfectionStatus(IntEnum):
    SUSCEPTIBLE = 0
    EXPOSED     = 1
    INFECTED    = 2
    RECOVERED   = 3

STATUS_COLORS = {
    InfectionStatus.SUSCEPTIBLE: '#1E90FF', # Dodger Blue
    InfectionStatus.EXPOSED:     '#FF8C00', # Dark Orange
    InfectionStatus.INFECTED:    '#E32636', # Alizarin Crimson (Clear Red)
    InfectionStatus.RECOVERED:   '#32CD32'  # Lime Green
}
#=====================================================================


#=====================================================================
# A] Load Telemetry and Events
#=====================================================================
print("Loading Simulation Results")
df = pd.read_csv("../../Results/processed_telemetry.csv")
times = sorted(df['time'].unique())

# Pre-calculate max time for safe looping
max_time = times[-1] if times else 0

try:
    df_events = pd.read_csv("../../Results/transmission_events.csv")

    event_options = []
    if not df_events.empty and 'time' in df_events.columns:
        unique_events = sorted(df_events['time'].unique())
        event_options = [{'label': f"⚠️ Transmission at t={t}s", 'value': t} for t in unique_events]

except FileNotFoundError:
    print("transmission_events.csv not found. No event markers will be drawn")
    df_events = pd.DataFrame(columns=['time', 'infector', 'infectee', 'building_idx', 'floor', 'pos_x', 'pos_y', 'dist'])
    event_options = [{'label': "❌ Event File Missing", 'value': 0, 'disabled': True}]

try:
    df_doors = pd.read_csv("../../Data/doors_status.csv")
except FileNotFoundError:
    print("doors_status.csv not found. No door status will be drawn")
    df_doors = pd.DataFrame(columns=['time', 'door_id', 'is_open'])
#=====================================================================


#=====================================================================
# B] Load Semantic CAD Geometry (The BVH .npz)
#=====================================================================
print("Loading CAD Asset")
geom = np.load("../../Data/floorplan_simple_a.npz", allow_pickle=True)

# 1. Extract Walls for Plotly
verts = geom['wall_vertices']
indices = geom['wall_indices']
wall_x, wall_y = [], []
for i in range(0, len(indices), 3):
    p1, p2, p3 = verts[indices[i]], verts[indices[i+1]], verts[indices[i+2]]
    if p1[2] == 0.0 and p2[2] == 0.0: # Ground plane only
        wall_x.extend([p1[0], p2[0], None])
        wall_y.extend([p1[1], p2[1], None])

# 2. Extract Room Labels
room_coords = geom['room_coords']
room_names = geom['room_names']
room_x = [c[0] for c in room_coords]
room_y = [c[1] for c in room_coords]

# 3. Extract Beds (AABBs)
beds = geom['beds'] if 'beds' in geom else []

# 4. Extract Doors Dynamically from CAD
door_coords = {}
if 'doors' in geom:
    for i, door_pts in enumerate(geom['doors']):
        door_id = i + 1
        door_coords[door_id] = {
            'x': [door_pts[0][0], door_pts[1][0]],
            'y': [door_pts[0][1], door_pts[1][1]]
        }
#=====================================================================

#=====================================================================
# C] Build Dash Layout
#=====================================================================
app = dash.Dash(__name__)
app.title = "GPU Sim Test Viewer"

app.layout = html.Div(style={'backgroundColor': '#F8F9FA', 'color': '#212529', 'padding': '20px', 'fontFamily': 'sans-serif', 'minHeight': '100vh'}, children=[
    html.H2("Simulation Viewer", style={'textAlign': 'center', 'marginBottom': '0px', 'color': '#212529'}),

    # 1. Main Map Display
    dcc.Graph(id='live-map', style={'height': '750px'}),

    # 2. Animation Controls
    html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '20px', 'padding': '15px', 'backgroundColor': '#E9ECEF', 'borderRadius': '10px', 'border': '1px solid #DEE2E6'}, children=[
        html.Button("▶ Play / Pause", id='play-button', n_clicks=0, style={'padding': '10px 20px', 'fontSize': '16px', 'cursor': 'pointer', 'backgroundColor': '#007BFF', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'fontWeight': 'bold'}),

        # Video Export Button
        html.Button("🎬 Export MP4", id='export-button', n_clicks=0, style={'padding': '10px 20px', 'fontSize': '16px', 'cursor': 'pointer', 'backgroundColor': '#DC3545', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'fontWeight': 'bold'}),

        html.Div(style={'flexGrow': '1'}, children=[
            dcc.Slider(
                id='time-slider',
                min=min(times), max=max_time, step=None,
                marks={str(t): {'label': str(t), 'style': {'color': '#495057'}} for t in times if t % 10 == 0 or t == max_time},
                value=min(times)
            )
        ]),
        html.Div(id='time-display', style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#007BFF', 'minWidth': '150px', 'textAlign': 'right'}),

        # Event Dropdown
        html.Div(style={'minWidth': '300px'}, children=[
            dcc.Dropdown(
                id='event-dropdown',
                options=event_options,
                placeholder="🔍 Jump to Transmission",
                style={'color': '#212529'}
            )
        ])
    ]),

    # Status for Video Export
    html.Div(id='export-status', style={'textAlign': 'center', 'marginTop': '15px', 'fontWeight': 'bold', 'color': '#28A745', 'fontSize': '18px'}),

    dcc.Interval(id='anim-interval', interval=800, n_intervals=0, disabled=True)
])
#=====================================================================

#==================================================================================================
# D] App Callbacks & Rendering Logic
#==================================================================================================

#=====================================================================
@app.callback(
    Output('anim-interval', 'disabled'),
    Input('play-button', 'n_clicks'),
    State('anim-interval', 'disabled')
)
def toggle_play(n_clicks, currently_disabled):
    return not currently_disabled if n_clicks > 0 else True
#=====================================================================

#=====================================================================
@app.callback(
    Output('time-slider', 'value'),
    [Input('anim-interval', 'n_intervals'), Input('event-dropdown', 'value')],
    State('time-slider', 'value')
)
def update_time(n_intervals, selected_event_time, current_time):
    trigger = ctx.triggered_id

    if trigger == 'event-dropdown' and selected_event_time is not None:
        return selected_event_time

    idx = times.index(current_time)
    return times[(idx + 1) % len(times)]
#=====================================================================

#=====================================================================
def generate_frame(current_time):
    fig = go.Figure()

    # 5.1 Static Semantic Layer (Walls and Text)
    fig.add_trace(go.Scatter(x=wall_x, y=wall_y, mode='lines', line=dict(color='#343A40', width=3), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=room_x, y=room_y, mode='text', text=room_names, textfont=dict(color='rgba(0, 0, 0, 0.15)', size=24, family="Arial Black"), hoverinfo='skip', showlegend=False))

    # 5.2 Dynamic Environmental Layer (Doors and Beds)
    for d_id, coords in door_coords.items():
        is_open = False
        if not df_doors.empty:
            door_history = df_doors[(df_doors['door_id'] == d_id) & (df_doors['time'] <= current_time)]
            if not door_history.empty:
                is_open = door_history.iloc[-1]['is_open'] == 1

        color = '#28A745' if is_open else '#DC3545'
        dash_style = 'dot' if is_open else 'solid'

        fig.add_trace(go.Scatter(x=coords['x'], y=coords['y'], mode='lines', line=dict(color=color, width=4, dash=dash_style), name=f"Door {d_id}", hoverinfo='name', showlegend=False))

    for i, b in enumerate(beds):
        fig.add_shape(type="rect", x0=b[0], y0=b[1], x1=b[2], y1=b[3], line=dict(color="#4169E1", width=2), fillcolor="rgba(65, 105, 225, 0.1)")
        fig.add_annotation(x=(b[0]+b[2])/2, y=(b[1]+b[3])/2, text=f"Bed {i+1}", showarrow=False, font=dict(color="#4169E1", size=10))

    # 5.3 Agent Trajectories and Trails
    history_df = df[df['time'] <= current_time]
    for aid in history_df['agent_id'].unique():
        adata = history_df[history_df['agent_id'] == aid]
        fig.add_trace(go.Scatter(x=adata['pos_x'], y=adata['pos_y'], mode='lines', line=dict(color='rgba(100, 100, 100, 0.3)', width=2, dash='dot'), hoverinfo='skip', showlegend=False))

    # 5.4 Current Agent Positions
    current_df = df[df['time'] == current_time]
    fig.add_trace(go.Scatter(
        x=current_df['pos_x'], y=current_df['pos_y'], mode='markers+text',
        marker=dict(size=14, color=[STATUS_COLORS[s] for s in current_df['infection_status']], line=dict(color='white', width=1.5)),
        text=[f"A{int(aid)}" for aid in current_df['agent_id']], textposition="top center",
        textfont=dict(color='#212529', size=12, family='Arial, bold'),
        hovertext=[f"Building: {b} | Floor: {f}" for b, f in zip(current_df.get('building_idx', ['?']*len(current_df)), current_df.get('floor', ['?']*len(current_df)))],
        hoverinfo='text', showlegend=False
    ))

    # 5.5 Event Markers Transmissions
    past_events = df_events[df_events['time'] <= current_time]
    if not past_events.empty:
        event_texts = [f"<b>⚠️ Transmission</b><br>t={row['time']}s | A{int(row['infector'])} ➔ A{int(row['infectee'])}" for _, row in past_events.iterrows()]
        fig.add_trace(go.Scatter(
            x=past_events['pos_x'], y=past_events['pos_y'], mode='markers+text',
            marker=dict(size=25, color='#FFC107', symbol='star-triangle-up', line=dict(color='#D32F2F', width=2)),
            text=event_texts, textposition="bottom center", textfont=dict(color='#D32F2F', size=13, family='Arial, bold'),
            name="Transmission Event", hoverinfo='skip'
        ))

    # 5.6 Layout (Hardcoded ranges prevent the video camera from bouncing)
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="y", scaleratio=1, range=[-2, 22]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-2, 17]),
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF'
    )

    return fig
#=====================================================================

#=====================================================================
@app.callback(
    [Output('live-map', 'figure'), Output('time-display', 'children')],
    [Input('time-slider', 'value')]
)
def update_map(current_time):
    return generate_frame(current_time), f"Time: {current_time}s"
#=====================================================================

#=====================================================================
@app.callback(
    Output('export-status', 'children'),
    Input('export-button', 'n_clicks'),
    prevent_initial_call=True
)
def export_mp4(n_clicks):
    if n_clicks == 0: return ""

    print("🎬 Starting background MP4 render")
    output_path = "../../Results/Simulation_Render.mp4"

    # We render the subset of frames matching the slider marks to keep export fast
    render_times = [t for t in times if t % 10 == 0 or t == max_time]
    video_writer = None

    for t in render_times:
        fig = generate_frame(t)
        # Add a title specific to the video export
        fig.update_layout(title=dict(text=f"Time: {t}s", font=dict(size=20, color='black')))

        img_bytes = fig.to_image(format="png", width=1920, height=1080)
        image = Image.open(io.BytesIO(img_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if video_writer is None:
            h, w, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, 10, (w, h)) # 10 FPS

        video_writer.write(frame)

    if video_writer:
        video_writer.release()

    print("✅ MP4 Render Complete!")
    return f"✅ Success! Video saved to: {output_path}"
#=====================================================================


#==================================================================================================
if __name__ == '__main__':
    print("Launching Viewer")
    app.run(debug=True, port=8080)
#==================================================================================================
