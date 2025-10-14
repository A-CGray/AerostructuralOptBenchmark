from baseclasses import AeroProblem
from typing import List, Optional


class FlightPoint(AeroProblem):
    def __init__(
        self,
        name: str,
        loadFactor: float,
        fuelFraction: Optional[float] = 0.0,
        massConfig: Optional[str] = None,
        failureGroups: Optional[List[str]] = None,
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
        """
        super().__init__(name=name, **kwargs)

        self.loadFactor = loadFactor
        self.fuelFraction = fuelFraction
        self.massConfig = massConfig
        self.failureGroups = failureGroups


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
        self.massConfig = massConfig
        self.failureGroups = failureGroups
