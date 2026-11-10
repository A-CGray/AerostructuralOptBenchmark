from baseclasses import AeroProblem
from typing import List, Optional


VALID_MASS_CONFIGS = [
    "landingGrossMass",
    "takeoffMass",
    "midCruiseMass",
    "cruiseStartMass",
]


class FlightPoint(AeroProblem):
    def __init__(
        self,
        name: str,
        loadFactor: float,
        fuelFraction: Optional[float] = 0.0,
        massConfig: Optional[str] = None,
        failureGroups: Optional[List[str]] = None,
        canUseVLM: Optional[bool] = False,
        CGFraction: float = 0.5,
        tailRotation: float = 0.0,
        **kwargs,
    ):
        """Define a flight condition, this is basically just an AeroProblem with a few extra attributes

        Parameters
        ----------
        name : string
            Name of the flight point, should be unique and currently must contain either "cruise" or "maneuver
        loadFactor : float
            Flight point load factor (how many G's the aircraft is pulling)
        fuelFraction : float, optional
            What fraction of the total fuel mass is the aircraft carrying at this flight point, by default zero
        massConfig : string
            Name of the mass configuration to use for this flight point, if not supplied then the aircraft mass is computed based on the fuel fraction
        failureGroups : list of strings
            Names of wingbox component groups for which to compute a failure constraint value at this flight point
        CGFraction : float, optional
            Where the centre of gravity is between the forward (0.0) and aft (1.0) CG limits, by default 0.5. It is
            not bounded, so a point can be run with its CG outside the nominal range
        tailRotation : float, optional
            Initial rotation of the horizontal tail about its hinge line in degrees, positive leading edge up, by
            default 0.0. It is the starting value of the flight point's tail rotation design variable, as alpha is
            for the angle of attack, and is only used in runs with the horizontal tail
        """
        super().__init__(name=name, **kwargs)

        self.loadFactor = loadFactor
        self.fuelFraction = fuelFraction
        if massConfig is not None and massConfig not in VALID_MASS_CONFIGS:
            raise ValueError(f"Invalid massConfig {massConfig}, must be one of {VALID_MASS_CONFIGS}")
        self.massConfig = massConfig
        self.failureGroups = failureGroups
        self.canUseVLM = canUseVLM
        self.CGFraction = CGFraction
        self.tailRotation = tailRotation


class LoadCase:
    def __init__(
        self,
        name: str,
        loadFactor: float,
        fuelFraction: Optional[float] = 0.0,
        massConfig: Optional[str] = None,
        failureGroups: Optional[List[str]] = None,
    ):
        """Define a load case, like a flight point but without the aerodynamic state information

        Parameters
        ----------
        name : string
            Name of the load case, should be unique and currently must contain either "cruise" or "maneuver
        loadFactor : float
            Load case load factor (how many G's the aircraft is pulling)
        fuelFraction : float, optional
            What fraction of the total fuel mass is the aircraft carrying at this load case, by default zero
        massConfig : string
            Name of the mass configuration to use for this flight point, if not supplied then the aircraft mass is computed based on the fuel fraction
        failureGroups : list of strings
            Names of wingbox component groups for which to compute a failure constraint value at this load case
        """
        self.name = name
        self.loadFactor = loadFactor
        self.fuelFraction = fuelFraction
        if massConfig is not None and massConfig not in VALID_MASS_CONFIGS:
            raise ValueError(f"Invalid massConfig {massConfig}, must be one of {VALID_MASS_CONFIGS}")
        self.massConfig = massConfig
        self.failureGroups = failureGroups
