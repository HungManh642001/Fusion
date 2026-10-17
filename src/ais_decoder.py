import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pyais import decode

EARTH_RADIUS = 6371000.0  # meters
KNOTS_TO_MS = 0.51444444

@dataclass
class AISMessage:
    ts: int  # Milliseconds timestamp
    mmsi: int
    lat: float
    lon: float
    sog: float  # knots
    cog: float  # degrees (0-360)

def extrapolate_kinematic(msg: AISMessage, target_ts: int) -> Tuple[float, float]:
    """
    Extrapolates vessel latitude and longitude to target_ts using kinematic dead-reckoning equations
    (Eq. 8 & 9 from Sensors 2025, 25, 2179).
    """
    dt = (target_ts - msg.ts) / 1000.0  # seconds
    if abs(dt) < 1e-6 or msg.sog is None or math.isnan(msg.sog):
        return msg.lat, msg.lon
    
    sog_ms = msg.sog * KNOTS_TO_MS
    cog_rad = math.radians(msg.cog if msg.cog is not None and not math.isnan(msg.cog) else 0.0)
    
    # Delta lat
    delta_lat = (180.0 * sog_ms * dt * math.cos(cog_rad)) / (math.pi * EARTH_RADIUS)
    lat_rt = msg.lat + delta_lat
    
    # Delta lon
    cos_lat = math.cos(math.radians(msg.lat))
    if abs(cos_lat) < 1e-6:
        cos_lat = 1e-6
    delta_lon = (180.0 * sog_ms * dt * math.sin(cog_rad)) / (math.pi * EARTH_RADIUS * cos_lat)
    lon_rt = msg.lon + delta_lon
    
    return lat_rt, lon_rt

class AISDecoder:
    """Decodes raw NMEA AIVDM/AIVDO files with timestamps."""
    def __init__(self, lat_bounds: Tuple[float, float] = (30.50, 30.62), lon_bounds: Tuple[float, float] = (114.20, 114.40)):
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds

    def decode_file(self, filepath: str) -> List[AISMessage]:
        messages = []
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(':', 1)
                if len(parts) != 2:
                    continue
                try:
                    ts = int(parts[0])
                except ValueError:
                    continue
                nmea = parts[1]
                if nmea.startswith('!AIVDM') or nmea.startswith('!AIVDO'):
                    try:
                        decoded = decode(nmea)
                        mmsi = getattr(decoded, 'mmsi', None)
                        lat = getattr(decoded, 'lat', None)
                        lon = getattr(decoded, 'lon', None)
                        sog = getattr(decoded, 'speed', None)
                        cog = getattr(decoded, 'course', None)
                        if mmsi and lat is not None and lon is not None:
                            # Verify valid bounds
                            if self.lat_bounds[0] <= lat <= self.lat_bounds[1] and self.lon_bounds[0] <= lon <= self.lon_bounds[1]:
                                messages.append(AISMessage(
                                    ts=ts,
                                    mmsi=int(mmsi),
                                    lat=float(lat),
                                    lon=float(lon),
                                    sog=float(sog) if sog is not None else 0.0,
                                    cog=float(cog) if cog is not None else 0.0
                                ))
                    except Exception:
                        continue
        return messages
