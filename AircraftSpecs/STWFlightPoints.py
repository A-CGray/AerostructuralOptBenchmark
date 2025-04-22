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
# Buffet conditions
# ==============================================================================
# The aircraft must be buffet free up to MMO and at 1.3g at the cruise speed,
# (https://safetyfirst.airbus.com/high-altitude-manual-flying/)
# I set these conditions at the aircraft ceiling instead of the cruise altitude to make them more challenging
# The MMO value is from:
# https://www.flyradius.com/boeing-717/200-specifications-dimensions
# dive speed is Md = MMO + 0.07 as is specified in 14 CFR 25.335(b)(2)
# (https://www.ecfr.gov/current/title-14/part-25/section-25.335#p-25.335(b)(2))
MAX_ALTITUDE = 11277.6  # 37,000 ft in meters
MMO = 0.82
buffetHighLift = FlightPoint(
    "buffet_high_lift",
    loadFactor=1.3,
    fuelFraction=1.0,
    failureGroups=[],
    mach=MMO,
    altitude=MAX_ALTITUDE,
    alpha=6.5,
    evalFuncs=["lift", "sepsensor", "sepsensorksarea"],
)
buffetHighSpeed = FlightPoint(
    "buffet_high_speed",
    loadFactor=1.0,
    fuelFraction=1.0,
    failureGroups=[],
    mach=MMO+0.07,
    altitude=MAX_ALTITUDE,
    alpha=4.25,
    evalFuncs=["lift", "sepsensor", "sepsensorksarea"],
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
# These manoeuvres are performed at 26,000 ft which is the altitude at which the flight speeds become Mach limited.
# Following the FAR 25 V-n diagram, the 2.5g pullup is performed at the dive speed and the -1g pushdown is performed at
# the cruise speed. I'm assuming that the speed is limited by compressibility effects at the cruise altitude and so the
# dive speed is Md = MMO + 0.07 as is specified in 14 CFR 25.335(b)(2)
# (https://www.ecfr.gov/current/title-14/part-25/section-25.335#p-25.335(b)(2))

HIGH_SPEED_MANEUVER_ALTITUDE = 7924.8  # 26,000 ft in m

highAltHighSpeedPullUp = FlightPoint(
    "mnver_highAlt_vd_pullup",
    loadFactor=2.5,
    fuelFraction=MANEUVER_FUEL_LOAD_FRACTION,
    failureGroups=["l_skin", "u_skin", "spar", "rib"],
    mach=MMO + 0.07,
    altitude=HIGH_SPEED_MANEUVER_ALTITUDE,
    alpha=5.0,
    evalFuncs=["lift", "drag", "cl", "cd"],
)

highAltHighSpeedPushDown = FlightPoint(
    "mnver_highAlt_vc_pushdown",
    loadFactor=-1.0,
    fuelFraction=MANEUVER_FUEL_LOAD_FRACTION,
    failureGroups=["l_skin"],
    mach=CRUISE_MACH,
    altitude=HIGH_SPEED_MANEUVER_ALTITUDE,
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
    "mnver_sealevel_vd_pullup": [highAltHighSpeedPullUp],
    "mnver_sealevel_vc_pushdown": [highAltHighSpeedPushDown],
    "buffet_high_lift": [buffetHighLift],
    "buffet_high_speed": [buffetHighSpeed],
    "3pt": [standardCruise, seaLevelLowSpeedPullUp, seaLevelLowSpeedPushDown],
    "2pt": [standardCruise, seaLevelLowSpeedPullUp],
    "5pt": [
        standardCruise,
        seaLevelLowSpeedPullUp,
        seaLevelLowSpeedPushDown,
        highAltHighSpeedPullUp,
        highAltHighSpeedPushDown,
    ],
    "maneuverOnly": [
        seaLevelLowSpeedPullUp,
        seaLevelLowSpeedPushDown,
    ],
    "buffet": [buffetHighLift, buffetHighSpeed],
    "cruise+buffet": [standardCruise, buffetHighLift, buffetHighSpeed],
    "5pt-buffet": [
        standardCruise,
        seaLevelLowSpeedPullUp,
        seaLevelLowSpeedPushDown,
        buffetHighLift,
        buffetHighSpeed,
    ],
}
