"""Dezert-Smarandache Theory (DSmT) Engine with PCR5 Proportional Conflict Redistribution.

Solves the classic Zadeh Paradox in military multi-sensor fusion:
When ESM and SAR sensors report highly contradictory hypotheses (e.g. ESM says Destroyer with 0.9 belief,
SAR dimensions say Frigate with 0.9 belief), classical Dempster-Shafer rule erroneously attributes
conflict mass to an unrelated third hypothesis (e.g. Commercial cargo).

DSmT PCR5 redistributes conflict energy back to the conflicting sources proportionally to their beliefs,
guaranteeing zero-paradox behavior under extreme electronic warfare & denial conditions.
"""

from typing import Dict, List, Optional
import numpy as np


class DSmTEngine:
    """DSmT evidence combination engine using Proportional Conflict Redistribution Rule 5 (PCR5)."""

    def __init__(self, classes: Optional[List[str]] = None):
        self.classes = classes or ["Destroyer", "Frigate", "Commercial", "Unknown"]

    def combine_pcr5(self, m1: Dict[str, float], m2: Dict[str, float]) -> Dict[str, float]:
        """Combine two independent basic belief assignments (BBAs) using the PCR5 rule.

        Formula for each hypothesis X:
        m_pcr5(X) = m1(X)*m2(X) + sum_{Y != X} [ (m1(X)^2 * m2(Y)) / (m1(X) + m2(Y))
                                               + (m2(X)^2 * m1(Y)) / (m2(X) + m1(Y)) ]
        """
        all_hypotheses = sorted(list(set(list(m1.keys()) + list(m2.keys()) + self.classes)))

        # Clean and fill missing keys with 0.0
        bba1 = {k: float(max(m1.get(k, 0.0), 0.0)) for k in all_hypotheses}
        bba2 = {k: float(max(m2.get(k, 0.0), 0.0)) for k in all_hypotheses}

        s1 = sum(bba1.values())
        s2 = sum(bba2.values())
        if s1 > 0:
            bba1 = {k: v / s1 for k, v in bba1.items()}
        if s2 > 0:
            bba2 = {k: v / s2 for k, v in bba2.items()}

        m_fused = {k: 0.0 for k in all_hypotheses}

        # 1. Conjunctive consensus for non-conflicting focal elements
        for x in all_hypotheses:
            m_fused[x] += bba1[x] * bba2[x]

        # 2. Proportional conflict redistribution for all disjoint pairs (X, Y)
        for i, x in enumerate(all_hypotheses):
            for j, y in enumerate(all_hypotheses):
                if i == j:
                    continue
                # Conflict term from m1(x) and m2(y)
                term1 = bba1[x] * bba2[y]
                if term1 > 1e-12:
                    denom = bba1[x] + bba2[y]
                    if denom > 1e-12:
                        m_fused[x] += (bba1[x] ** 2 * bba2[y]) / denom

                # Conflict term from m2(x) and m1(y)
                term2 = bba2[x] * bba1[y]
                if term2 > 1e-12:
                    denom = bba2[x] + bba1[y]
                    if denom > 1e-12:
                        m_fused[x] += (bba2[x] ** 2 * bba1[y]) / denom

        # Normalize final fused distribution
        total_mass = sum(m_fused.values())
        if total_mass > 0:
            m_fused = {k: float(v / total_mass) for k, v in m_fused.items()}

        return m_fused
