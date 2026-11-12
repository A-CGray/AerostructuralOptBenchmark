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
import utils
from AircraftSpecs.FlightPoint import FlightPoint
from performanceCalc import (
    FuelAndMassConstraintGroup,
    FuelConsistencyGroup,
    LiftConstraintGroup,
    computeFuelTankUsage,
    getAeroForceName,
)
from utils import getPromotedInputNames

TEST_SPECS = {
    "reserveFuelMass": 100.0,
    "fuelDensity": 800.0,
    "wingboxFuelVolumeFraction": 0.8,
    "auxFuelVolume": 1.0,
}
TEST_POINT_NAME = "testPoint"


def createTestFlightPoint() -> FlightPoint:
    """Create a FlightPoint with no mass configuration, so its mass is computed from the fuel burn"""
    return FlightPoint(TEST_POINT_NAME, loadFactor=1.0, fuelFraction=0.5, mach=0.5, altitude=0.0, alpha=1.0)


class TestGetAeroForceName(unittest.TestCase):
    def test_joins_point_name_and_capitalised_force(self):
        """Assert that the force name is the flight point name and the capitalised force joined by an underscore."""
        self.assertEqual(getAeroForceName("cruise", "lift"), "cruise_Lift")
        self.assertEqual(getAeroForceName("cruise", "Drag"), "cruise_Drag")

    def test_utils_exports_the_performance_calc_function(self):
        """Assert that utils still provides getAeroForceName, as the same function performanceCalc defines."""
        self.assertIs(utils.getAeroForceName, getAeroForceName)


class TestLiftConstraintGroup(unittest.TestCase):
    def test_inputs_are_flight_point_output_names(self):
        """
        Given a flight point with no mass configuration,
        when LiftConstraintGroup is set up,
        then its external inputs are named after the flight point outputs that feed them, and the mass it computes
        itself is not one of them.
        """
        prob = om.Problem()
        prob.model.add_subsystem(
            "liftConstraints",
            LiftConstraintGroup(aircraftSpecs=TEST_SPECS, flightPointSet=[createTestFlightPoint()]),
            promotes=["*"],
        )
        prob.setup()
        self.assertEqual(
            set(getPromotedInputNames(prob)),
            {"landingGrossMass", "totalFuelBurn", getAeroForceName(TEST_POINT_NAME, "Lift")},
        )


class TestFuelConsistencyGroup(unittest.TestCase):
    def test_point_without_mass_config_gets_fuel_mass_difference(self):
        """
        Given a flight point with no mass configuration and a fuel fraction,
        when FuelConsistencyGroup is run,
        then its fuel mass difference equals the fuel mass minus the point's share of the fuel burn and the
        reserve fuel.
        """
        flightPoint = types.SimpleNamespace(name="testPoint", massConfig=None, fuelFraction=0.5)
        prob = om.Problem()
        prob.model.add_subsystem(
            "fuelConsistency",
            FuelConsistencyGroup(aircraftSpecs=TEST_SPECS, flightPointSet=[flightPoint]),
            promotes=["*"],
        )
        prob.setup()
        prob.set_val("testPoint-fuelMass", 900.0, units="kg")
        prob.set_val("totalFuelBurn", 1000.0, units="kg")
        prob.run_model()
        self.assertAlmostEqual(prob.get_val("testPointFuelMassDiff", units="kg")[0], 300.0)


class TestFuelAndMassConstraintGroup(unittest.TestCase):
    def test_point_without_mass_config_sets_up_with_lift_and_fuel_constraints(self):
        """
        Given a FlightPoint with no mass configuration,
        when FuelAndMassConstraintGroup is set up and run without the fuel volume constraint,
        then setup succeeds and the point gets both a lift difference and a fuel mass difference output.
        """
        prob = om.Problem()
        prob.model.add_subsystem(
            "constraints",
            FuelAndMassConstraintGroup(
                aircraftSpecs=TEST_SPECS, flightPointSet=[createTestFlightPoint()], includeFuelVolumeConstraint=False
            ),
            promotes=["*"],
        )
        prob.setup()
        prob.set_val("testPoint-fuelMass", 900.0, units="kg")
        prob.set_val("totalFuelBurn", 1000.0, units="kg")
        prob.run_model()
        self.assertAlmostEqual(prob.get_val("testPointFuelMassDiff", units="kg")[0], 300.0)
        self.assertEqual(prob.get_val("testPointLiftDiff").shape, (1,))

    def test_inputs_are_flight_point_output_names(self):
        """
        Given a flight point with no mass configuration and all three constraint groups included,
        when FuelAndMassConstraintGroup is set up,
        then its external inputs are exactly the flight point outputs that feed them, by name, in sorted order.
        """
        prob = om.Problem()
        prob.model.add_subsystem(
            "constraints",
            FuelAndMassConstraintGroup(aircraftSpecs=TEST_SPECS, flightPointSet=[createTestFlightPoint()]),
            promotes=["*"],
        )
        prob.setup()
        expectedInputs = [
            "landingGrossMass",
            "totalFuelBurn",
            getAeroForceName(TEST_POINT_NAME, "Lift"),
            f"{TEST_POINT_NAME}-fuelMass",
            "wingboxVolume",
        ]
        self.assertEqual(getPromotedInputNames(prob), sorted(expectedInputs))

    def test_fuel_tank_usage_uses_total_fuel_burn(self):
        """Assert that the fuel tank usage is computed from the total fuel burn and wingbox volume inputs."""
        totalFuelBurn = 1000.0
        wingboxVolume = 2.0
        prob = om.Problem()
        prob.model.add_subsystem(
            "constraints",
            FuelAndMassConstraintGroup(aircraftSpecs=TEST_SPECS, flightPointSet=[createTestFlightPoint()]),
            promotes=["*"],
        )
        prob.setup()
        prob.set_val("totalFuelBurn", totalFuelBurn, units="kg")
        prob.set_val("wingboxVolume", wingboxVolume, units="m**3")
        prob.run_model()
        expectedUsage = computeFuelTankUsage(
            fuelBurn=totalFuelBurn,
            wingboxVolume=wingboxVolume,
            reserveFuelMass=TEST_SPECS["reserveFuelMass"],
            fuelDensity=TEST_SPECS["fuelDensity"],
            wingboxVolumeFraction=TEST_SPECS["wingboxFuelVolumeFraction"],
            auxTankVolume=TEST_SPECS["auxFuelVolume"],
        )
        self.assertAlmostEqual(prob.get_val("fuelTankUsage")[0], expectedUsage)


if __name__ == "__main__":
    unittest.main()
