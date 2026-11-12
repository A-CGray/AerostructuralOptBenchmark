# ==============================================================================
# Standard Python modules
# ==============================================================================
import types
import unittest

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import openmdao.api as om
from hypothesis import given, settings
from hypothesis.strategies import floats

# ==============================================================================
# Extension modules
# ==============================================================================
import utils
from AircraftSpecs.FlightPoint import FlightPoint
from performanceCalc import (
    CorrectedDragComp,
    FuelAndMassConstraintGroup,
    FuelConsistencyGroup,
    LiftConstraintGroup,
    computeCorrectedDrag,
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
# Upper bounds of the corrected drag test inputs, a few times larger than any value the benchmark uses
MAX_TEST_DRAG = 1e6  # N
MAX_TEST_DRAG_COEFF = 0.1
MAX_TEST_REF_AREA = 500.0  # m^2
MAX_TEST_DYN_PRESSURE = 1e5  # Pa
CORRECTED_DRAG_RTOL = 1e-12


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


class TestCorrectedDrag(unittest.TestCase):
    @given(
        drag=floats(0.0, MAX_TEST_DRAG),
        extraDragCoeff=floats(0.0, MAX_TEST_DRAG_COEFF),
        refArea=floats(0.0, MAX_TEST_REF_AREA),
        dynPressure=floats(0.0, MAX_TEST_DYN_PRESSURE),
    )
    def test_drag_area_form_matches_coefficient_form(self, drag, extraDragCoeff, refArea, dynPressure):
        """
        Given an extra drag coefficient and the reference area it is normalised by,
        when computeCorrectedDrag is given their product as the extra drag area,
        then it returns the same corrected drag as the old coefficient form, drag + coefficient * area * dynamic
        pressure.
        """
        expectedDrag = drag + extraDragCoeff * refArea * dynPressure
        correctedDrag = computeCorrectedDrag(drag=drag, extraCDA=extraDragCoeff * refArea, dynPressure=dynPressure)
        np.testing.assert_allclose(correctedDrag, expectedDrag, rtol=CORRECTED_DRAG_RTOL)

    # Each example sets up a whole OpenMDAO problem, which can overrun Hypothesis's deadline on a busy machine
    @settings(deadline=None)
    @given(
        drag=floats(0.0, MAX_TEST_DRAG),
        extraDragCoeff=floats(0.0, MAX_TEST_DRAG_COEFF),
        refArea=floats(0.0, MAX_TEST_REF_AREA),
        dynPressure=floats(0.0, MAX_TEST_DYN_PRESSURE),
    )
    def test_component_matches_coefficient_form(self, drag, extraDragCoeff, refArea, dynPressure):
        """
        Given a CorrectedDragComp built with a drag area equal to an extra drag coefficient times its reference area,
        when the component is run with a drag input,
        then its correctedDrag output equals the old coefficient form, drag + coefficient * area * dynamic pressure.
        """
        prob = om.Problem()
        prob.model.add_subsystem(
            "dragCorrection",
            CorrectedDragComp(extraCDA=extraDragCoeff * refArea, dynPressure=dynPressure),
            promotes=["*"],
        )
        prob.setup()
        prob.set_val("drag", drag, units="N")
        prob.run_model()
        expectedDrag = drag + extraDragCoeff * refArea * dynPressure
        np.testing.assert_allclose(prob.get_val("correctedDrag")[0], expectedDrag, rtol=CORRECTED_DRAG_RTOL)

    def test_old_coefficient_options_are_rejected(self):
        """Assert that CorrectedDragComp rejects the extraDragCoeff and wingArea options as undeclared."""
        for oldOption in ["extraDragCoeff", "wingArea"]:
            with self.assertRaisesRegex(KeyError, oldOption):
                CorrectedDragComp(extraCDA=1.0, dynPressure=1.0, **{oldOption: 1.0})


if __name__ == "__main__":
    unittest.main()
