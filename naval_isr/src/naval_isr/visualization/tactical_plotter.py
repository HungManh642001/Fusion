"""Tactical Maritime Situation Plotter.

Renders 2D tactical surface picture with:
- Surveillance station origin and range rings
- Multi-target ground truth tracks
- Raw Radar returns and ESM Line-of-Bearing (LOB) rays
- Spoofed AIS offset vectors
- IMM filtered and Covariance Intersection fused tracks
- OSPA tracking performance metric curves
"""

import os
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np


class TacticalPlotter:
    """Renders high-contrast tactical situation display for Phase 1 reporting."""

    def __init__(self, style_dark: bool = True):
        self.style_dark = style_dark

    def plot_2d_situation(
        self,
        vessels: List[Any],
        radar_meas: List[Dict[str, Any]],
        esm_meas: List[Dict[str, Any]],
        fused_tracks: Dict[int, List[Dict[str, Any]]],
        output_path: str = "output/naval_isr/phase1_tactical_display.png",
    ) -> str:
        """Render 2D tactical Cartesian display with range rings and bearings."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 10), facecolor="#0f172a")
        ax.set_facecolor("#0b1120")

        # 1. Plot Range Rings from Coastal Station (0, 0)
        radii = [10000, 20000, 30000, 40000]  # in meters
        theta = np.linspace(0, 2 * np.pi, 200)
        for r in radii:
            ax.plot(
                r * np.cos(theta) / 1000.0,
                r * np.sin(theta) / 1000.0,
                color="#1e293b",
                linestyle="--",
                linewidth=1.0,
            )
            ax.text(
                0.5,
                (r / 1000.0) + 0.5,
                f"{r//1000} km",
                color="#64748b",
                fontsize=8,
            )

        # Plot Coastline / Station marker
        ax.scatter([0], [0], color="#38bdf8", s=150, marker="^", label="Coastal Station (Radar+ESM)")

        # 2. Plot ESM Line-of-Bearing (LOB) rays (Sample first 5 unique bearings)
        plotted_esm = 0
        for m in esm_meas:
            if plotted_esm >= 6:
                break
            b_rad = m["bearing_rad"]
            r_ray = 40.0  # km
            ax.plot(
                [0, r_ray * np.cos(b_rad)],
                [0, r_ray * np.sin(b_rad)],
                color="#f59e0b",
                linestyle=":",
                linewidth=1.2,
                alpha=0.6,
                label="ESM LOB (Passive Bearing)" if plotted_esm == 0 else "",
            )
            plotted_esm += 1

        # 3. Ground Truth Trajectories & Vessels
        colors = {"cooperative": "#10b981", "ais_spoofed": "#f97316", "dark_warship": "#ef4444"}
        labels = {
            "cooperative": "Target 1: Cargo (AIS Compliant)",
            "ais_spoofed": "Target 2: Smuggler (AIS Spoofed)",
            "dark_warship": "Target 3: Warship (Dark + Active Emitter)",
        }

        for v in vessels:
            traj = v.trajectory
            c = colors.get(v.vessel_type, "#94a3b8")
            lbl = labels.get(v.vessel_type, f"Target {v.track_id}")

            # Ground truth line
            ax.plot(traj[:, 1] / 1000.0, traj[:, 2] / 1000.0, color=c, linestyle="-", linewidth=2.0, label=lbl)
            # Final position marker
            ax.scatter(traj[-1, 1] / 1000.0, traj[-1, 2] / 1000.0, color=c, s=120, edgecolors="white", linewidth=1.5)

            # If spoofed, draw reported AIS position vs true position
            if v.ais_spoofed:
                spoofed_x = (traj[-1, 1] + 3500.0) / 1000.0
                spoofed_y = (traj[-1, 2] + 2500.0) / 1000.0
                ax.scatter(spoofed_x, spoofed_y, color="#eab308", s=100, marker="x", label="Reported AIS Spoof Pos")
                ax.plot(
                    [traj[-1, 1] / 1000.0, spoofed_x],
                    [traj[-1, 2] / 1000.0, spoofed_y],
                    color="#eab308",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.8,
                )

        # 4. Fused IMM-CI Tracks
        plotted_fused = False
        for tid, tlist in fused_tracks.items():
            if not tlist:
                continue
            xs = [item["state"][0] / 1000.0 for item in tlist]
            ys = [item["state"][2] / 1000.0 for item in tlist]
            ax.plot(
                xs,
                ys,
                color="#06b6d4",
                linestyle="-.",
                linewidth=1.8,
                label="IMM-CI Fused Track" if not plotted_fused else "",
            )
            plotted_fused = True

        ax.set_title("Naval Tactical Surface Picture: Active Radar + Passive ESM + AIS Fusion", color="#f8fafc", fontsize=14, pad=15)
        ax.set_xlabel("East (km)", color="#94a3b8", fontsize=11)
        ax.set_ylabel("North (km)", color="#94a3b8", fontsize=11)
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, color="#1e293b", linestyle=":", alpha=0.7)
        ax.set_xlim(-25, 25)
        ax.set_ylim(-5, 45)
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", loc="lower right", fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        return output_path

    def plot_ospa_metric(
        self,
        time_steps: List[float],
        ospa_curve: List[float],
        output_path: str = "output/naval_isr/phase1_ospa_metric.png",
    ) -> str:
        """Render OSPA metric curve over time."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0f172a")
        ax.set_facecolor("#0b1120")

        ax.plot(time_steps, ospa_curve, color="#38bdf8", linewidth=2.0, label="OSPA Metric (c=100m, p=1)")
        ax.axhline(np.mean(ospa_curve), color="#f59e0b", linestyle="--", linewidth=1.2, label=f"Mean OSPA: {np.mean(ospa_curve):.2f} m")

        ax.set_title("Multi-Target Tracking Precision: OSPA Evaluation Curve", color="#f8fafc", fontsize=13, pad=12)
        ax.set_xlabel("Time (seconds)", color="#94a3b8", fontsize=10)
        ax.set_ylabel("OSPA Distance (meters)", color="#94a3b8", fontsize=10)
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, color="#1e293b", linestyle=":", alpha=0.7)
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", loc="upper right")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        return output_path
