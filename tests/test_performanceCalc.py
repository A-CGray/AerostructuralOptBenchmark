# ==============================================================================
# Standard Python modules
# ==============================================================================
import types
import unittest

# ==============================================================================
# External Python modules
# ==============================================================================
import openmdao.api as om

# ==============================================================================
# Extension modules
# ==============================================================================
from performanceCalc import FuelConsistencyGroup


class TestFuelConsistencyGroup(unittest.TestCase):
    def test_point_without_mass_config_gets_fuel_mass_difference(self):
        """
        Given a flight point with no mass configuration and a fuel fraction,
        when FuelConsistencyGroup is run,
        then its fuel mass difference equals the fuel mass minus the point's share of the fuel burn and the reserve fuel.
        """
        flightPoint = types.SimpleNamespace(name="testPoint", massConfig=None, fuelFraction=0.5)
        specs = {"reserveFuelMass": 100.0}
        prob = om.Problem()
        prob.model.add_subsystem(
            "fuelConsistency",
            FuelConsistencyGroup(aircraftSpecs=specs, flightPointSet=[flightPoint]),
            promotes=["*"],
        )
        prob.setup()
        prob.set_val("testPoint-fuelMass", 900.0, units="kg")
        prob.set_val("fuelBurn", 1000.0, units="kg")
        prob.run_model()
        self.assertAlmostEqual(prob.get_val("testPointFuelMassDiff", units="kg")[0], 300.0)


if __name__ == "__main__":
    unittest.main()
