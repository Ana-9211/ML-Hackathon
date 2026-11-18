"""Folium helpers for drawing a trip: GPS fixes, nearby road network and routes.

Other parts of the project can import these to draw predicted routes in the
same style, e.g.

    m = trip_map(trip, net)
    add_route(m, net, predicted_arcs, "Predicted route", "#d62728")
    finish(m, "out.html")
"""
from __future__ import annotations

import folium
import numpy as np

from .network import RoadNetwork

LAYER_STYLE = {
    "gps": "#1f77b4",
    "network": "#9e9e9e",
    "reference": "#2ca02c",
}


def _merge_arcs(net: RoadNetwork, arcs) -> list[list[tuple[float, float]]]:
    """Join consecutive connected arcs into polylines (fewer map objects)."""
    lines, cur = [], []
    prev = None
    for a in arcs:
        a = int(a)
        if prev is not None and a == prev:
            continue
        f, t = net.arc_from[a], net.arc_to[a]
        if cur and net.arc_to[prev] == f:
            cur.append((net.node_lat[t], net.node_lon[t]))
        else:
            if cur:
                lines.append(cur)
            cur = [(net.node_lat[f], net.node_lon[f]), (net.node_lat[t], net.node_lon[t])]
        prev = a
    if cur:
        lines.append(cur)
    return lines


def trip_map(trip, net: RoadNetwork, network_radius_m: float = 80.0, max_gps_points: int = 3000,
             tiles: str = "OpenStreetMap", routes=None) -> folium.Map:
    """Map with GPS points and the road arcs near the trajectory.

    `routes` = list of (arc_ids, layer_name, colour). They are drawn BELOW the
    GPS points so the observed trajectory stays visible on top of the routes.
    """
    m = folium.Map(tiles=tiles, control_scale=True)
    m.fit_bounds([[trip.lat.min(), trip.lon.min()], [trip.lat.max(), trip.lon.max()]])

    px, py = net.proj.to_xy(trip.lon, trip.lat)
    step = max(1, int(np.ceil(len(px) / 4000)))
    near = np.unique(np.concatenate(net.arcs_near_many(px[::step], py[::step], network_radius_m)))
    roads = folium.FeatureGroup(name=f"Road network (arcs within {int(network_radius_m)} m)")
    for a in near:
        folium.PolyLine(net.arc_lonlat(a), color=LAYER_STYLE["network"], weight=2, opacity=0.6).add_to(roads)
    roads.add_to(m)

    for arcs, name, color in routes or []:
        add_route(m, net, arcs, name, color, weight=6)

    gstep = max(1, int(np.ceil(len(px) / max_gps_points)))
    gps = folium.FeatureGroup(name=f"GPS observations (every {gstep}. point)" if gstep > 1 else "GPS observations")
    for i in range(0, len(px), gstep):
        folium.CircleMarker((trip.lat[i], trip.lon[i]), radius=3, color="white", fill=True,
                            fill_color=LAYER_STYLE["gps"], fill_opacity=0.9, weight=0.5, tooltip=f"#{i}  t={trip.t[i]:.0f}s").add_to(gps)
    gps.add_to(m)
    folium.Marker((trip.lat[0], trip.lon[0]), tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((trip.lat[-1], trip.lon[-1]), tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    return m


def add_route(m: folium.Map, net: RoadNetwork, arcs, name: str, color: str, weight: int = 5,
              dash: str | None = None) -> None:
    layer = folium.FeatureGroup(name=name)
    for line in _merge_arcs(net, arcs):
        folium.PolyLine(line, color=color, weight=weight, opacity=0.6, dash_array=dash, tooltip=name).add_to(layer)
    layer.add_to(m)


def add_legend(m: folium.Map, entries: list[tuple[str, str]], title: str = "Legend") -> None:
    rows = "".join(
        f'<div><span style="display:inline-block;width:14px;height:4px;background:{c};'
        f'margin-right:6px;vertical-align:middle"></span>{label}</div>' for label, c in entries)
    html = (f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;'
            f'padding:8px 12px;border:1px solid #bbb;border-radius:6px;font:12px sans-serif">'
            f'<b>{title}</b>{rows}</div>')
    m.get_root().html.add_child(folium.Element(html))


def finish(m: folium.Map, path) -> None:
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(path))
