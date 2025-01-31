"""
==============================================================================
Definition of MACH Tutorial wing flight points
==============================================================================
@File    :   STWFlightPoints.py
@Date    :   2023/10/05
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================

# ==============================================================================
# Extension modules
# ==============================================================================
from FlightPoint import FlightPoint

# ==============================================================================
# Cruise conditions
# ==============================================================================
# Standard cruise condition taken from https://en.wikipedia.org/wiki/Boeing_717#Specifications
CRUISE_ALTITUDE = 10.4e3  # meters
CRUISE_MACH = 0.77
standardCruise = FlightPoint(
    "cruise",
    loadFactor=1.0,
    fuelFraction=1.0,
    failureGroups=[],
    mach=CRUISE_MACH,
    altitude=CRUISE_ALTITUDE,
    alpha=3.25,
    evalFuncs=["lift", "drag", "cl", "cd"],
)

# ==============================================================================
# Low-speed mmaneouvre conditions
# ==============================================================================
# The low speed (Va) maneuver flight condition is taken from:
# https://www.flyradius.com/boeing-717/200-specifications-dimensions
# I converted the KCAS (263 @ 0 ft) value to a Mach number using https://aerotoolbox.com/airspeed-conversions/
# I then boost the flight speed by 15% so that we're not trying to simulate the aircraft right at CL max
SEALEVEL_VA_MACH = 0.398 * 1.15
MANEUVER_ALTITUDE = 0.0
MANEUVER_FUEL_LOAD_FRACTION = (
    0.0  # Perform the maneuver with zero fuel mass since we are not modelling the fuel's inertial relief in TACS
)

seaLevelLowSpeedPullUp = FlightPoint(
    "mnver_sealevel_va_pullup",
    loadFactor=2.5,
    fuelFraction=MANEUVER_FUEL_LOAD_FRACTION,
    failureGroups=["l_skin", "u_skin", "spar", "rib"],
    mach=SEALEVEL_VA_MACH,
    altitude=MANEUVER_ALTITUDE,
    alpha=8.7,
    evalFuncs=["lift", "drag", "cl", "cd"],
)

seaLevelLowSpeedPushDown = FlightPoint(
    "mnver_sealevel_va_pushdown",
    loadFactor=-1.0,
    fuelFraction=MANEUVER_FUEL_LOAD_FRACTION,
    failureGroups=["l_skin"],
    mach=SEALEVEL_VA_MACH,
    altitude=MANEUVER_ALTITUDE,
    alpha=-5.8,
    evalFuncs=["lift", "drag", "cl", "cd"],
)

# ==============================================================================
# High-speed manoeuvre conditions
# ==============================================================================
# These manoeuvres are performed at the cruise altitude. Following the FAR 25 V-n diagram, the 2.5g pullup is performed
# at the dive speed and the -1g pushdown is performed at the cruise speed.
# I'm assuming that the speed is limited by compressibility effects at the cruise altitude and so the dive speed is
# Md = Mc + 0.07 as is specified in 14 CFR 25.335(b)(2)
# (https://www.ecfr.gov/current/title-14/part-25/section-25.335#p-25.335(b)(2))

seaLevelHighSpeedPullUp = FlightPoint(
    "mnver_sealevel_va_pullup",
    loadFactor=2.5,
    fuelFraction=MANEUVER_FUEL_LOAD_FRACTION,
    failureGroups=["l_skin", "u_skin", "spar", "rib"],
    mach=CRUISE_MACH + 0.07,
    altitude=CRUISE_ALTITUDE,
    alpha=8.7,
    evalFuncs=["lift", "drag", "cl", "cd"],
)

seaLevelHighSpeedPushDown = FlightPoint(
    "mnver_sealevel_va_pushdown",
    loadFactor=-1.0,
    fuelFraction=MANEUVER_FUEL_LOAD_FRACTION,
    failureGroups=["l_skin"],
    mach=CRUISE_MACH,
    altitude=CRUISE_ALTITUDE,
    alpha=-5.8,
    evalFuncs=["lift", "drag", "cl", "cd"],
)

# ==============================================================================
# Define sets of flight points
# ==============================================================================
flightPointSets = {
    "cruise": [standardCruise],
    "mnver_sealevel_va_pullup": [seaLevelLowSpeedPullUp],
    "mnver_sealevel_va_pushdown": [seaLevelLowSpeedPushDown],
    "mnver_sealevel_vd_pullup": [seaLevelHighSpeedPullUp],
    "mnver_sealevel_vc_pushdown": [seaLevelHighSpeedPushDown],
    "3pt": [standardCruise, seaLevelLowSpeedPullUp, seaLevelLowSpeedPushDown],
    "2pt": [standardCruise, seaLevelLowSpeedPullUp],
    "5pt": [
        standardCruise,
        seaLevelLowSpeedPullUp,
        seaLevelLowSpeedPushDown,
        seaLevelHighSpeedPullUp,
        seaLevelHighSpeedPushDown,
    ],
    "maneuverOnly": [
        seaLevelLowSpeedPullUp,
        seaLevelLowSpeedPushDown,
        seaLevelHighSpeedPullUp,
        seaLevelHighSpeedPushDown,
    ],
}
