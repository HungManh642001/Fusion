"""Interactive Folium GIS Tactical Map Renderer for Satellite SAR & AIS Maritime Fusion.

Renders military GIS tactical map with:
- Dark Vessels (Red Marker / Alert)
- Spoofed AIS Ghost Ships (Yellow Marker / Warning)
- Cooperative Vessels (Green Marker)
- Coastal Surveillance Stations
"""

import os
from typing import Dict, List, Any
import folium
from folium import plugins


class GISTacticalMapRenderer:
    """Renders interactive HTML GIS maritime tactical map using Folium."""

    def __init__(self, default_lat: float = 30.563549, default_lon: float = 114.306087, zoom_start: int = 11):
        self.default_lat = default_lat
        self.default_lon = default_lon
        self.zoom_start = zoom_start

    def render_interactive_map(
        self,
        classified_results: Dict[str, List[Dict[str, Any]]],
        output_html_path: str = "output/naval_isr/tactical_sar_map.html",
        station_lat: float = 30.563549,
        station_lon: float = 114.306087,
    ) -> str:
        """Create and save an interactive HTML tactical map."""
        os.makedirs(os.path.dirname(output_html_path), exist_ok=True)

        m = folium.Map(
            location=[station_lat, station_lon],
            zoom_start=self.zoom_start,
            tiles="CartoDB dark_matter",
            control_scale=True,
        )

        # 1. Coastal Surveillance Station
        folium.Marker(
            location=[station_lat, station_lon],
            popup="<b>Trạm Giám sát Bờ Biển (Radar + ESM)</b><br>Tọa độ: 30.5635°N, 114.3061°E",
            tooltip="Trạm Giám sát Bờ Biển",
            icon=folium.Icon(color="blue", icon="tower-broadcast", prefix="fa"),
        ).add_to(m)

        # 2. Cooperative Vessels (Green)
        coop_group = folium.FeatureGroup(name="Tàu Hợp Tác (AIS Hợp Lệ)", show=True).add_to(m)
        for c in classified_results.get("cooperative", []):
            lat = c.get("sar_lat", c.get("lat"))
            lon = c.get("sar_lon", c.get("lon"))
            mmsi = c.get("mmsi", "Unknown")
            length = c.get("sar_length_m", 0.0)
            popup_html = (
                f"<div style='font-family: Arial; min-width: 160px;'>"
                f"<b style='color: #10b981;'>HỢP TÁC - HỢP LỆ</b><br>"
                f"<b>MMSI:</b> {mmsi}<br>"
                f"<b>Chiều dài:</b> {length:.1f} m<br>"
                f"<b>Tọa độ:</b> {lat:.4f}°N, {lon:.4f}°E"
                f"</div>"
            )
            folium.Marker(
                location=[lat, lon],
                popup=popup_html,
                tooltip=f"Tàu Hợp Tác (MMSI: {mmsi})",
                icon=folium.Icon(color="green", icon="ship", prefix="fa"),
            ).add_to(coop_group)

        # 3. Dark Vessels (Red Alert!)
        dark_group = folium.FeatureGroup(name="MỤC TIÊU ĐEN (TẮT AIS / TÀU CHIẾN)", show=True).add_to(m)
        for d in classified_results.get("dark_vessels", []):
            lat = d["lat"]
            lon = d["lon"]
            sid = d.get("detection_id", "SAR-?")
            length = d.get("length_m", 0.0)
            threat = d.get("threat_type", "Chưa rõ")
            popup_html = (
                f"<div style='font-family: Arial; min-width: 200px;'>"
                f"<b style='color: #ef4444; font-size: 14px;'>🚨 CẢNH BÁO: Dark Vessel (RED ALERT)</b><br>"
                f"<hr style='margin: 4px 0;'>"
                f"<b>Mã phát hiện SAR:</b> #{sid}<br>"
                f"<b>Chiều dài ước tính:</b> {length:.1f} m<br>"
                f"<b>Trạng thái AIS:</b> <span style='color: red;'>TẮT HOÀN TOÀN</span><br>"
                f"<b>Đánh giá:</b> {threat}<br>"
                f"<b>Tọa độ:</b> {lat:.4f}°N, {lon:.4f}°E"
                f"</div>"
            )
            folium.Marker(
                location=[lat, lon],
                popup=popup_html,
                tooltip=f"🚨 MỤC TIÊU ĐEN #{sid} (Dài {length:.1f}m)",
                icon=folium.Icon(color="red", icon="triangle-exclamation", prefix="fa"),
            ).add_to(dark_group)

            # Circle pulse indicator around dark target
            folium.Circle(
                location=[lat, lon],
                radius=800,
                color="#ef4444",
                weight=2,
                fill=True,
                fill_color="#ef4444",
                fill_opacity=0.25,
            ).add_to(dark_group)

        # 4. Spoofed Ghost Ships (Yellow Warning)
        ghost_group = folium.FeatureGroup(name="TÀU MA GIẢ MẠO (AIS KHÔNG CÓ THỰC)", show=True).add_to(m)
        for g in classified_results.get("spoofed_ghosts", []):
            lat = g["lat"]
            lon = g["lon"]
            mmsi = g.get("mmsi", "Ghost")
            sog = g.get("sog_knots", g.get("sog", 0.0))
            popup_html = (
                f"<div style='font-family: Arial; min-width: 180px;'>"
                f"<b style='color: #eab308; font-size: 13px;'>⚠️ CẢNH BÁO: Ghost Ship (YELLOW ALERT)</b><br>"
                f"<hr style='margin: 4px 0;'>"
                f"<b>MMSI Giả Mạo:</b> {mmsi}<br>"
                f"<b>Vận tốc khai báo:</b> {sog:.1f} kts<br>"
                f"<b>Phản xạ SAR:</b> <span style='color: orange;'>KHÔNG CÓ TÀU THỰC</span><br>"
                f"<b>Phân loại:</b> Tác chiến điện tử / Đánh lừa AIS"
                f"</div>"
            )
            folium.Marker(
                location=[lat, lon],
                popup=popup_html,
                tooltip=f"⚠️ TÀU MA AIS ({mmsi})",
                icon=folium.Icon(color="orange", icon="ghost", prefix="fa"),
            ).add_to(ghost_group)

        folium.LayerControl().add_to(m)
        m.save(output_html_path)
        return output_html_path
