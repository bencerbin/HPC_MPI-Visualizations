
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State

# -----------------------------
# Load data
# -----------------------------
with open("results_final.pkl", "rb") as f:
    results = pickle.load(f)

top_df = results["fantasy_movies"]
genre_df = results["genre_counts"]
G = results["graph"]

# -----------------------------
# 1. Time Series (binned)
# -----------------------------

top_df = top_df.copy()
top_df = top_df.reset_index(drop=True)

top_df["year_bin"] = (top_df["startYear"] // 5) * 5
grouped = top_df.groupby("year_bin")

max_idx = grouped["averageRating"].idxmax()
min_idx = grouped["averageRating"].idxmin()

summary = top_df.loc[max_idx][["year_bin", "primaryTitle"]].rename(
    columns={"primaryTitle": "max_title"}
)

summary_min = top_df.loc[min_idx][["year_bin", "primaryTitle"]].rename(
    columns={"primaryTitle": "min_title"}
)

summary_dict = summary.set_index("year_bin")["max_title"].to_dict()
summary_min_dict = summary_min.set_index("year_bin")["min_title"].to_dict()

#

print("\n=== SUMMARY (MAX) ===")
print(summary.head(10))
print("\nShape:", summary.shape)
print("\nUnique bins:", summary["year_bin"].nunique())
print("\nDuplicate bins (should be 0):")
print(summary["year_bin"].duplicated().sum())

print("\n=== SUMMARY (MIN) ===")
print(summary_min.head(10))
print("\nShape:", summary_min.shape)
print("\nUnique bins:", summary_min["year_bin"].nunique())
print("\nDuplicate bins (should be 0):")
print(summary_min["year_bin"].duplicated().sum())

#

custom_data = top_df["year_bin"].map(
    lambda y: [summary_dict.get(y, ""), summary_min_dict.get(y, "")]
)


# top_df = top_df.merge(summary, on="year_bin", how="left")
# top_df = top_df.merge(summary_min, on="year_bin", how="left")

fig_time = px.box(
    top_df,
    x="year_bin",
    y="averageRating",
    title="Distribution of Fantasy Movie Ratings Over Time",
    
)


fig_time.update_layout(
    template="plotly_white",
    xaxis_title="Year",
    yaxis_title="Rating"
)

fig_time.update_traces(
    customdata=custom_data,
    hovertemplate=
        "Rating: %{y}<br>" +
        "Max Movie: %{customdata[0]}<br>" +
        "Min Movie: %{customdata[1]}<extra></extra>"
)



# -----------------------------
# 2. Genre Bar Chart
# -----------------------------
# -----------------------------
# 2. Genre Analysis (Avg Rating)
# -----------------------------
genre_avg = top_df.copy()

# Split genres into individual rows
genre_avg = genre_avg.assign(
    Genre=genre_avg["genres"].str.split(",")
).explode("Genre")

# Compute avg rating + count
genre_summary = genre_avg.groupby("Genre").agg({
    "averageRating": "mean",
    "primaryTitle": "count"
}).reset_index()

genre_summary = genre_summary.rename(columns={
    "averageRating": "avg_rating",
    "primaryTitle": "movie_count"
})

top_movies_by_genre = (
    genre_avg.sort_values("averageRating", ascending=False)
    .groupby("Genre")
    .head(3)
    .groupby("Genre")["primaryTitle"]
    .apply(lambda x: "<br>".join(x))
    .to_dict()
)

genre_summary["top_movies"] = genre_summary["Genre"].map(top_movies_by_genre)
# Take top 10 by count (or rating if you prefer)
genre_summary = genre_summary.sort_values("movie_count", ascending=False).head(10)

fig_genre = px.bar(
    genre_summary,
    x="Genre",
    y="movie_count",
    title="Average Rating by Genre",
    labels={
        "avg_rating": "Average IMDb Rating"
    }
)

# 🔥 Add rich tooltips
fig_genre.update_traces(
    customdata=genre_summary[["movie_count"]],
    hovertemplate=
        "<b>%{x}</b><br>" +
        "Avg Rating: %{y:.2f}<br>" +
        "Movies: %{customdata[0]}<extra></extra>"
)

fig_genre.update_layout(template="plotly_white")
# -----------------------------
# 3. Network Graph
# -----------------------------
pos = nx.spring_layout(G, seed=42)

edge_x = []
edge_y = []

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5),
    hoverinfo='none',
    mode='lines'
)

node_x = []
node_y = []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

degrees = dict(G.degree())

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode='markers',
    marker=dict(size=6),
    text=[
        f"{node}<br>Connections: {degrees[node]}"
        for node in G.nodes()
    ],
    customdata=list(G.nodes()),
    hoverinfo='text'
)

fig_graph = go.Figure(data=[edge_trace, node_trace])
fig_graph.update_layout(
    title="Actor-Director Network",
    showlegend=False,
    template="plotly_white"
)

# -----------------------------
# DASH APP
# -----------------------------
app = Dash(__name__)

app.layout = html.Div([
    html.H1("IMDB Fantasy Dashboard", style={"textAlign": "center"}),

    dcc.Graph(figure=fig_time, id="fig_time"),
    dcc.Store(id="selected-node"),
    
    dcc.RangeSlider(
    id="year-slider",
    min=int(top_df["startYear"].min()),
    max=int(top_df["startYear"].max()),
    step=1,
    value=[1900, 2020],
    marks={int(y): str(y) for y in range(1890, 2025, 5)}
),
    
    dcc.Dropdown(
    id="genre-metric",
    options=[
        {"label": "Movie Count", "value": "movie_count"},
        {"label": "Average Rating", "value": "avg_rating"}
    ],
    value="movie_count",  # default
    clearable=False,
    style={"width": "300px", "margin": "10px auto"}
), 

    html.Div([

    html.P(
        "💡 Click a genre bar to filter the time series above",
        style={
            "textAlign": "center",
            "color": "gray",
            "fontSize": "14px",
            "marginBottom": "5px"
        }
    ),

    dcc.Graph(id="genre_graph")
]),


    dcc.Graph(id = "graph", figure=fig_graph),
])

@app.callback(
    Output("fig_time", "figure"),
    [
        Input("year-slider", "value"),
        Input("genre_graph", "clickData")
    ]
)
def update_time_chart(year_range, clickData):

    filtered_df = top_df[
        (top_df["startYear"] >= year_range[0]) &
        (top_df["startYear"] <= year_range[1])
    ]

    # 🎯 Apply genre filter if user clicks
    if clickData:
        selected_genre = clickData["points"][0]["x"]
        filtered_df = filtered_df[
            filtered_df["genres"].str.contains(selected_genre)
        ]

    fig = px.box(
        filtered_df,
        x="year_bin",
        y="averageRating",
        title="Fantasy Ratings Distribution Over Time"
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Year (binned)",
        yaxis_title="IMDb Ratings"
    )

    # Add detailed tooltips back
    fig.update_traces(
        customdata=filtered_df[["primaryTitle", "genres", "startYear"]],
        hovertemplate=
            "<b>%{customdata[0]}</b><br>" +
            "Genres: %{customdata[1]}<br>" +
            "Year: %{customdata[2]}<br>" +
            "Rating: %{y}<extra></extra>"
    )

    return fig

@app.callback(
    Output("genre_graph", "figure"),
    Input("genre-metric", "value")
)
def update_genre_chart(metric):
    
    fig = px.bar(
        genre_summary,
        x="Genre",
        y=metric,
        title=f"Top Secondary Genres (to Fantasy) by {'Average Rating' if metric=='avg_rating' else 'Movie Count'}",
    )

    fig.update_layout(template="plotly_white")
    
    fig.update_traces(
        customdata=genre_summary[["movie_count", "avg_rating"]],
        hovertemplate=
            "<b>%{x}</b><br>" +
            "Movies: %{customdata[0]}<br>" +
            "Avg Rating: %{customdata[1]:.2f}<extra></extra>"
    )

    return fig

@app.callback(
    Output("graph", "figure"),
    Output("selected-node", "data"),
    Input("graph", "clickData"),
    State("selected-node", "data"),
    prevent_initial_call=True   # 👈 ADD THIS
)

def update_network_viz(clickData, selected_node):

    # No click → default
    if clickData is None:
        return fig_graph, None

    # If already highlighted → ANY click resets
    if selected_node is not None:
        return fig_graph, None

    # Safe extraction
    if "customdata" not in clickData["points"][0]:
        return fig_graph, None

    clicked_node = clickData["points"][0]["customdata"]
    neighbors = list(G.neighbors(clicked_node))

    node_colors = []
    node_sizes = []

    for node in G.nodes():
        if node == clicked_node:
            node_colors.append("red")
            node_sizes.append(12)
        elif node in neighbors:
            node_colors.append("orange")
            node_sizes.append(9)
        else:
            node_colors.append("lightgray")
            node_sizes.append(6)

    degrees = dict(G.degree())

    new_node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        marker=dict(size=node_sizes, color=node_colors),
        text=[
            f"{node}<br>Connections: {degrees[node]}"
            for node in G.nodes()
        ],
        customdata=list(G.nodes()),
        hoverinfo='text'
    )

    fig = go.Figure(data=[edge_trace, new_node_trace])
    fig.update_layout(
        title="Actor-Director Network",
        showlegend=False,
        template="plotly_white"
    )

    return fig, clicked_node

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8051)