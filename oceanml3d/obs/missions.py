# Ported from NOSC (branch first_implementation) contrib/synthetic_obs/ - see docs/migration_nosc.md
"""
Registry of real altimetry missions, parameterized by the quantities that
actually drive ground-track sampling geometry: inclination, repeat-cycle
length and the number of orbits per repeat cycle. These are published,
near-constant mission-design parameters (not live ephemeris), consistent with
a closed-form repeat-orbit ground-track model (see orbits.py) rather than a
full SGP4/TLE propagator.

IMPORTANT CONVENTION - orbits vs passes:
    AVISO/CNES mission handbooks usually quote the repeat cycle in *passes*
    (half-revolutions: one ascending + one descending pass per orbit).
    `orbits_per_cycle` below is the number of FULL revolutions per cycle,
    i.e. passes / 2. Mixing the two conventions produces physically
    impossible orbits (a previous version of this file had Jason-3 at 254
    orbits/9.92 days = a 56-minute period, which is below the minimum
    possible LEO period of ~84 minutes) and roughly doubles the simulated
    sampling density - silently making every downstream OSSE optimistic.

    `validate_missions()` (called at import time) enforces a plausible LEO
    period for every entry so this class of error cannot silently recur.

Reference values (passes per cycle -> orbits_per_cycle):
    Jason-3 / TOPEX / Sentinel-6: 254 passes / 9.9156 d  -> 127 orbits (T ~ 112.4 min)
    Sentinel-3A/B:                770 passes / 27 d      -> 385 orbits (T ~ 101.0 min)
    SARAL (Envisat orbit):        1002 passes / 35 d     -> 501 orbits (T ~ 100.6 min)
    HY-2B:                        386 passes / 14 d      -> 193 orbits (T ~ 104.5 min)
    SWOT (science orbit):         584 passes / 20.8646 d -> 292 orbits (T ~ 102.9 min)
"""
from dataclasses import dataclass

# Plausible circular-LEO period band for altimetry constellations (~700-1400 km).
_MIN_PERIOD_S = 90.0 * 60.0
_MAX_PERIOD_S = 130.0 * 60.0


@dataclass(frozen=True)
class Mission:
    name: str
    inclination_deg: float
    repeat_cycle_days: float
    orbits_per_cycle: int
    swath_width_km: float = 0.0
    nadir_gap_km: float = 0.0
    ascending_node_lon0_deg: float = 0.0
    phase_offset_orbits: float = 0.0
    active_from: str | None = None    # first date with science data (YYYY-MM-DD), None = always
    active_to: str | None = None      # last date, None = still flying

    def is_active(self, date) -> bool:
        import numpy as np
        d = np.datetime64(str(date)[:10])
        return (self.active_from is None or d >= np.datetime64(self.active_from)) and \
               (self.active_to is None or d <= np.datetime64(self.active_to))

    @property
    def orbits_per_day(self) -> float:
        return self.orbits_per_cycle / self.repeat_cycle_days

    @property
    def orbital_period_s(self) -> float:
        return 86400.0 * self.repeat_cycle_days / self.orbits_per_cycle

    @property
    def nodal_days_per_cycle(self) -> int:
        """Integer number of nodal days per repeat cycle (the D of the
        standard (N, D) repeat-orbit description). For all supported
        missions this is the calendar cycle length rounded to the nearest
        integer (9.9156 -> 10, 20.8646 -> 21, ...)."""
        return int(round(self.repeat_cycle_days))

    @property
    def shift_per_orbit_deg(self) -> float:
        """Westward equatorial drift of the ascending node per orbit, from
        the repeat condition N x shift = D x 360 (exact cycle closure by
        construction). Deriving this from the sidereal day instead - as a
        previous version did - ignores the J2 nodal precession of the orbit
        plane and leaves a ~20 deg/cycle closure error."""
        return 360.0 * self.nodal_days_per_cycle / self.orbits_per_cycle


MISSIONS = {
    # Jason-3 / TOPEX-Poseidon / Sentinel-6 Michael Freilich reference orbit
    # Activity windows (approximate operational periods on the reference orbit; used only when
    # build_mask_dataset(historical=True)). Jason-2 flew the same orbit before Jason-3 (2008-2016).
    "jason2": Mission("jason2", inclination_deg=66.04, repeat_cycle_days=9.9156, orbits_per_cycle=127,
                      active_from="2008-07-04", active_to="2016-10-02"),
    "jason3": Mission("jason3", inclination_deg=66.04, repeat_cycle_days=9.9156, orbits_per_cycle=127,
                      active_from="2016-02-12"),
    "sentinel6": Mission("sentinel6", inclination_deg=66.04, repeat_cycle_days=9.9156, orbits_per_cycle=127,
                          phase_offset_orbits=63.5, active_from="2022-04-07"),  # interleaved half-cycle vs jason3
    # Sentinel-3A/B reference orbit
    "sentinel3a": Mission("sentinel3a", inclination_deg=98.65, repeat_cycle_days=27.0, orbits_per_cycle=385,
                          active_from="2016-06-01"),
    "sentinel3b": Mission("sentinel3b", inclination_deg=98.65, repeat_cycle_days=27.0, orbits_per_cycle=385,
                           phase_offset_orbits=192.5, active_from="2018-11-01"),
    # SARAL/AltiKa (ex-Envisat orbit; drifting orbit after 2016-07 approximated as the repeat one)
    "saral": Mission("saral", inclination_deg=98.55, repeat_cycle_days=35.0, orbits_per_cycle=501,
                     active_from="2013-03-14"),
    "envisat": Mission("envisat", inclination_deg=98.55, repeat_cycle_days=35.0, orbits_per_cycle=501,
                       active_to="2012-04-08"),
    # HY-2A / HY-2B
    "hy2a": Mission("hy2a", inclination_deg=99.34, repeat_cycle_days=14.0, orbits_per_cycle=193,
                    active_from="2011-10-01", active_to="2020-01-01"),
    "hy2b": Mission("hy2b", inclination_deg=99.34, repeat_cycle_days=14.0, orbits_per_cycle=193,
                    active_from="2018-11-01"),
    # Cryosat-2 (369-day repeat, ~30-day sub-cycle; approximated by its sub-cycle)
    "cryosat2": Mission("cryosat2", inclination_deg=92.0, repeat_cycle_days=29.0, orbits_per_cycle=421,
                        active_from="2010-07-01"),
    # SWOT KaRIn: wide-swath, two 50 km swaths either side of a 20 km nadir gap
    "swot": Mission("swot", inclination_deg=77.6, repeat_cycle_days=20.8646, orbits_per_cycle=292,
                     swath_width_km=50.0, nadir_gap_km=20.0, active_from="2023-07-21"),
}

# Nadir-only constellation resembling the "6 sats" merged product referenced
# elsewhere in this repo (process_data/mask/glorys_masking.ipynb).
SIX_SAT_NADIR = ["jason3", "sentinel6", "sentinel3a", "sentinel3b", "saral", "hy2b"]
# Historical constellation: every mission with an activity window; per-day selection by is_active().
HISTORICAL = ["jason2", "jason3", "sentinel6", "sentinel3a", "sentinel3b", "envisat", "saral", "hy2a", "hy2b", "cryosat2"]


def validate_missions(missions=None):
    """Raise if any mission implies a physically impossible orbital period."""
    missions = missions if missions is not None else MISSIONS.values()
    for mission in missions:
        period = mission.orbital_period_s
        if not (_MIN_PERIOD_S <= period <= _MAX_PERIOD_S):
            raise ValueError(
                f"Mission '{mission.name}': orbits_per_cycle={mission.orbits_per_cycle} over "
                f"{mission.repeat_cycle_days} days implies an orbital period of {period / 60:.1f} min, "
                f"outside the plausible LEO band [{_MIN_PERIOD_S / 60:.0f}, {_MAX_PERIOD_S / 60:.0f}] min. "
                f"Most likely cause: 'passes per cycle' (half-orbits) used where full orbits are expected "
                f"- divide by 2 (see module docstring)."
            )


validate_missions()
