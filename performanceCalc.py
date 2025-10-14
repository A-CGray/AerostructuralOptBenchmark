"""
==============================================================================
Basic aircraft mission performance calculations
==============================================================================
@File    :   performanceCalc.py
@Date    :   2023/04/26
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
import sys

# ==============================================================================
# External Python modules
# ==============================================================================
import openmdao.api as om
import numpy as np
import jax
import jax.numpy as jnp


# ==============================================================================
# Extension modules
# ==============================================================================
from AircraftSpecs.FlightPoint import FlightPoint

# ==============================================================================
# Individual components
# ==============================================================================

jax.config.update("jax_enable_x64", True)  # Make jax use double precision
# Need to set XLA_PYTHON_CLIENT_PREALLOCATE=false otherwise every instance of jax will try to pre-allocate 75% of GPU
# memory (even when not using GPU)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


# --- Mass calculation components ---
def computeSegmentInitMass(lift, drag, finalMass, R, tsfc, climbAngle, v):
    """Compute the mass at the start of a segment given the mass at the end of the segment using the Breguet range equation

    Parameters
    ----------
    lift : float/complex
        Lift force
    drag : float/complex
        drag force
    finalWeight : float/complex
        Weight at end of segment
    R : float/complex
        Segment range
    tsfc : float/complex
        Thrust specific fuel consumption
    climbAngle : float/complex
        CLimb angle of segment in radians
    v : float/complex
        Flight speed
    """
    LoverD = np.sqrt((lift / drag) ** 2)
    initMass = finalMass * np.exp(R * tsfc / v * (np.cos(climbAngle) / LoverD + np.sin(climbAngle)))
    return initMass


class BreguetRangeSegmentComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("R", desc="Segment range")
        self.options.declare("tsfc", desc="Thrust-specific fuel consumption")
        self.options.declare("climbAngle", desc="Climb angle of segment")
        self.options.declare("v", desc="Flight speed")

    def setup(self):
        self.add_input("lift", shape=1, units="N")
        self.add_input("drag", shape=1, units="N")
        self.add_input("finalMass", shape=1, units="kg")
        self.add_output("initMass", shape=1, units="kg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        outputs["initMass"] = computeSegmentInitMass(
            lift=inputs["lift"],
            drag=inputs["drag"],
            finalMass=inputs["finalMass"],
            R=self.options["R"],
            tsfc=self.options["tsfc"],
            climbAngle=self.options["climbAngle"],
            v=self.options["v"],
        )
        # if self.comm.rank == 0:
        #     print(f"SegmentInitMass = {outputs['initMass'][0]: 11.7e}")


def elhamRegression(wingboxMass):
    # Estimate total mass of a wing based on mass of single wingbox using Elham's EMWET regression (see https://doi.org/10.1017/s0001924000008563)
    return 10.147 * wingboxMass**0.8162


class WingMassRegressionComp(om.ExplicitComponent):
    def setup(self):
        self.add_input("wingboxMass", shape=1, units="kg")
        self.add_output("wingMass", shape=1, units="kg")

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        outputs["wingMass"] = elhamRegression(inputs["wingboxMass"])
        # if self.comm.rank == 0:
        #     print(f"wingMass = {outputs['wingMass'][0]: 11.7e}")

    def compute_partials(self, inputs, partials):
        partials["wingMass", "wingboxMass"] = 10.147 * 0.8162 * inputs["wingboxMass"] ** (0.8162 - 1.0)


def computeLandingGrossMass(wingMass, payloadMass, airframeMass, reserveFuelMass):
    """Compute the landing mass of an aircraft

    Parameters
    ----------
    wingMass : float/complex
        Total mass of one wing
    payloadMass : float/complex
        Mass of the payload
    airframeMass : float/complex
        Mass of the airframe excluding the wings (e.g fuselage, engines, systems weight)
    reserveFuelMass : float/complex
        Reserve fuel mass that needs to be left over at the end of the mission

    Returns
    -------
    float/complex
        Aircraft landing gross mass
    """
    return 2 * wingMass + payloadMass + airframeMass + reserveFuelMass


class LandingGrossMass(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("payloadMass", desc="Mass of the payload")
        self.options.declare(
            "airframeMass",
            types=float,
            desc="Mass of the airframe excluding the wings (e.g fuselage, engines, systems weight)",
        )
        self.options.declare(
            "reserveFuelMass",
            types=float,
            desc="Reserve fuel mass that needs to be left over at the end of the mission",
        )

    def setup(self):
        self.add_input("wingMass", shape=1, units="kg")
        self.add_output("landingGrossMass", shape=1, units="kg")

    def setup_partials(self):
        self.declare_partials(of="landingGrossMass", wrt="wingMass", val=2.0)

    def compute(self, inputs, outputs):
        opt = self.options
        outputs["landingGrossMass"] = computeLandingGrossMass(
            wingMass=inputs["wingMass"],
            payloadMass=opt["payloadMass"],
            airframeMass=opt["airframeMass"],
            reserveFuelMass=opt["reserveFuelMass"],
        )
        # if self.comm.rank == 0:
        #     print(f"landingGrossMass = {outputs['landingGrossMass'][0]: 11.7e}")


def computeMidSegmentMass(initialMass, finalMass):
    """Compute the "mid-segment" Mass for a given segment

    Because the rate of fuel-burn in a segment is not constant,
    a geometric average of the start and end Massses is used to
    estimate the Mass at the mid-point of the segment

    Parameters
    ----------
    initialMass : float/complex
        Segment start mass
    finalMass : float/complex
        Segment end mass
    """
    return np.sqrt(finalMass * initialMass)


class MidSegmentMassComp(om.ExplicitComponent):
    def setup(self):
        self.add_input("initialMass", shape=1, units="kg")
        self.add_input("finalMass", shape=1, units="kg")
        self.add_output("midSegmentMass", shape=1, units="kg")

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        outputs["midSegmentMass"] = computeMidSegmentMass(
            initialMass=inputs["initialMass"], finalMass=inputs["finalMass"]
        )
        # if self.comm.rank == 0:
        #     print(f"midSegmentMass = {outputs['midSegmentMass'][0]: 11.7e}")

    def compute_partials(self, inputs, partials):
        partials["midSegmentMass", "initialMass"] = (
            0.5 * inputs["finalMass"] / np.sqrt(inputs["finalMass"] * inputs["initialMass"])
        )
        partials["midSegmentMass", "finalMass"] = (
            0.5 * inputs["initialMass"] / np.sqrt(inputs["finalMass"] * inputs["initialMass"])
        )


def computeCorrectedDrag(drag, extraDragCoeff, wingArea, dynPressure):
    return drag + extraDragCoeff * wingArea * dynPressure


# --- Lift and drag calculations ---
class CorrectedDragComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("extraDragCoeff", types=float, desc="Extra drag coefficient")
        self.options.declare("wingArea", types=float, desc="Wing area")
        self.options.declare("dynPressure", types=float, desc="Dynamic pressure")

    def setup(self):
        self.add_input("drag", shape=1, units="N")
        self.add_output("correctedDrag", shape=1, units="N")

    def setup_partials(self):
        self.declare_partials(of="correctedDrag", wrt="drag", val=1.0)

    def compute(self, inputs, outputs):
        outputs["correctedDrag"] = computeCorrectedDrag(
            drag=inputs["drag"],
            extraDragCoeff=self.options["extraDragCoeff"],
            wingArea=self.options["wingArea"],
            dynPressure=self.options["dynPressure"],
        )
        # if self.comm.rank == 0:
        #     print(f"correctedDrag = {outputs['correctedDrag'][0]: 11.7e}")


class LiftConstraintComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("loadFactor", types=float, desc="Load factor")
        self.options.declare("fuelFraction", types=float, desc="Mass of the aircraft", default=None)

    def setup(self):
        self.add_input("lift", shape=1, units="N")
        self.add_input("mass", shape=1, units="kg")
        if self.options["fuelFraction"] is not None:
            self.add_input("fuelMass", shape=1, units="kg")
        self.add_output("liftDiff", shape=1, units="N")

    def setup_partials(self):
        self.declare_partials(of="liftDiff", wrt="lift", val=2.0)
        self.declare_partials(of="liftDiff", wrt="mass", val=-self.options["loadFactor"] * 9.81)
        if self.options["fuelFraction"] is not None:
            self.declare_partials(of="liftDiff", wrt="fuelMass", val=-self.options["loadFactor"] * 9.81)

    def compute(self, inputs, outputs):
        mass = inputs["mass"]
        if self.options["fuelFraction"] is not None:
            mass += inputs["fuelMass"] * self.options["fuelFraction"]

        outputs["liftDiff"] = 2.0 * inputs["lift"] - mass * self.options["loadFactor"] * 9.81

        # if self.comm.rank == 0:
        #     print(f"liftDiff = {outputs['liftDiff'][0]: 11.7e}")


# --- Misc ---
def computeFuelTankUsage(
    fuelBurn,
    wingboxVolume,
    reserveFuelMass,
    fuelDensity,
    wingboxVolumeFraction,
    auxTankVolume,
):
    """Compute the percentage of available fuel tank volume used during a mission

    Parameters
    ----------
    fuelBurn : float/complex
        Mass of fuel burned during mission
    wingboxVolume : float/complex
        Volume of one wingbox
    reserveFuelMass : float/complex
        Mass of reserve fuel required at end of mission
    fuelDensity : float/complex
        Density of fuel
    wingboxVolumeFraction : float/complex
        Fraction of the wingbox which assumed to be fuel tank
    auxTankVolume : float/complex
        Volume of auxiliary fuel tanks not in wingbox

    Returns
    -------
    float/complex
        Fuel volume margin, 1.0 = Completely full, 0.0 = Completely empty
    """
    boxVolume = 2.0 * wingboxVolumeFraction * wingboxVolume
    fuelVolume = (fuelBurn + reserveFuelMass) / fuelDensity - auxTankVolume
    return fuelVolume / boxVolume


class FuelTankUsageComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("reserveFuelMass", desc="Mass of reserve fuel required at end of mission")
        self.options.declare("fuelDensity", desc="Density of fuel")
        self.options.declare(
            "wingboxVolumeFraction",
            desc="Fraction of the wingbox which assumed to be fuel tank",
        )
        self.options.declare("auxTankVolume", desc="Volume of auxiliary fuel tanks not in wingbox")

    def setup(self):
        self.add_input("fuelBurn", shape=1, units="kg")
        self.add_input("wingboxVolume", shape=1, units="m**3")
        self.add_output("fuelTankUsage", shape=1)

    def setup_partials(self):
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        outputs["fuelTankUsage"] = computeFuelTankUsage(
            fuelBurn=inputs["fuelBurn"],
            wingboxVolume=inputs["wingboxVolume"],
            reserveFuelMass=self.options["reserveFuelMass"],
            fuelDensity=self.options["fuelDensity"],
            wingboxVolumeFraction=self.options["wingboxVolumeFraction"],
            auxTankVolume=self.options["auxTankVolume"],
        )
        # if self.comm.rank == 0:
        #     print(f"fuelTankUsage = {outputs['fuelTankUsage'][0]: 11.7e}")


def computeWingLoading(wingArea, MTOM):
    """Compute the wing loading of an aircraft

    Parameters
    ----------
    MTOM : float/complex
        Aircraft maximum take-off mass
    wingArea : float/complex
        Planform area of a single wing

    Returns
    -------
    float/complex
        Wing loading in units of mass/area
    """
    return MTOM / (2.0 * wingArea)


class WingLoadingComp(om.ExplicitComponent):
    def setup(self):
        self.add_input("wingArea", shape=1, units="m**2")
        self.add_input("MTOM", shape=1, units="kg")
        self.add_output("wingLoading", shape=1, units="kg/m**2")

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs):
        outputs["wingLoading"] = computeWingLoading(wingArea=inputs["wingArea"], MTOM=inputs["MTOM"])
        # if self.comm.rank == 0:
        #     print(f"wingLoading = {outputs['wingLoading'][0]: 11.7e}")

    def compute_partials(self, inputs, partials):
        partials["wingLoading", "wingArea"] = -inputs["MTOM"] / (2.0 * inputs["wingArea"] ** 2)
        partials["wingLoading", "MTOM"] = 1.0 / (2.0 * inputs["wingArea"])


# ==============================================================================
# OpenMDAO group combining components needed to compute aircraft empty mass
# ==============================================================================
class AirframeMassGroup(om.Group):
    def initialize(self):
        self.options.declare("aircraftSpecs", types=dict)

    def setup(self):
        self.specs = self.options["aircraftSpecs"]

        # --- Compute wing mass from wingbox mass ---
        wingMassComp = WingMassRegressionComp()
        self.add_subsystem("WingMassRegression", wingMassComp, promotes=["*"])

        # --- Compute landing gross mass ---
        LGMComp = LandingGrossMass(
            payloadMass=self.specs["payloadMass"],
            airframeMass=self.specs["airframeMass"],
            reserveFuelMass=self.specs["reserveFuelMass"],
        )
        self.add_subsystem("MassSummation", LGMComp, promotes=["*"])


# ==============================================================================
# OpenMDAO group combining components needed to compute aircraft fuel burn
# ==============================================================================
class FuelBurnGroup(om.Group):
    def initialize(self):
        self.options.declare("aircraftSpecs", types=dict)
        self.options.declare("flightPoint")

    def setup(self):
        self.specs = self.options["aircraftSpecs"]
        self.flightPoint = self.options["flightPoint"]

        # --- Drag correction ---
        addedDragComp = CorrectedDragComp(
            extraDragCoeff=self.specs["extraDragCoeff"],
            wingArea=self.specs["refArea"],
            dynPressure=self.flightPoint.q,
        )
        self.add_subsystem(
            "dragCorrection",
            addedDragComp,
            promotes_inputs=[("drag", "cruiseDrag")],
            promotes_outputs=["*"],
        )

        # --- Breguet range calculations ---
        # First compute the cruise fuelburn to go from the landing gross mass to the weight at the start of cruise
        cruiseFuelburnComp = BreguetRangeSegmentComp(
            R=self.specs["range"],
            tsfc=self.specs["tsfc"],
            climbAngle=0.0,
            v=self.flightPoint.V,
        )
        self.add_subsystem(
            "CruiseFuelBurn",
            cruiseFuelburnComp,
            promotes_outputs=[("initMass", "cruiseStartMass")],
            promotes_inputs=[("lift", "cruiseLift"), ("finalMass", "landingGrossMass")],
        )
        self.connect("correctedDrag", "CruiseFuelBurn.drag")

        # Then compute the takeoff mass by using the cruise start mass as the final mass for the climb segment
        climbFuelburnComp = BreguetRangeSegmentComp(
            R=self.specs["climbRange"],
            tsfc=self.specs["tsfc"],
            climbAngle=self.specs["climbAngle"],
            v=self.specs["climbSpeed"],
        )
        self.add_subsystem(
            "climbFuelBurn",
            climbFuelburnComp,
            promotes_outputs=[("initMass", "takeoffMass")],
            promotes_inputs=[("lift", "cruiseLift")],
        )
        self.connect("cruiseStartMass", "climbFuelBurn.finalMass")
        self.connect("correctedDrag", "climbFuelBurn.drag")

        # Finally compute the fuelburn as the difference between the takeoff mass and the landing gross mass
        totalFuelBurnComp = om.AddSubtractComp(
            output_name="totalFuelBurn",
            input_names=["takeoffMass", "landingGrossMass"],
            scaling_factors=[1.0, -1.0],
        )
        self.add_subsystem("totalFuelBurnComp", totalFuelBurnComp, promotes=["*"])


class FuelDistributionComp(om.JaxExplicitComponent):
    """Given the total fuel mass required to be stored in the wingbox, and the volumes of each rib bay, compute the mass
    of fuel in each bay and the fraction of the total fuel volume used
    """

    def initialize(self):
        self.options.declare("fuelDensity", types=float, desc="Density of fuel")
        self.options.declare(
            "wingboxVolumeFraction",
            types=float,
            desc="Fraction of each rib bay assumed to be fuel tank",
        )
        self.options.declare("numRibBays", types=int)
        self.options.declare("maxSmoothingRelError", types=float, default=1e-4)

    def setup(self):
        self.add_input("bayVolumes", shape=self.options["numRibBays"], units="m**3")
        self.add_input("fuelMass", units="kg")
        self.add_output("bayFuelMasses", copy_shape="bayVolumes", units="kg")
        self.add_output("wingboxVolume")

    def get_self_statics(self):
        return (
            self.options["fuelDensity"],
            self.options["wingboxVolumeFraction"],
            self.options["maxSmoothingRelError"],
        )

    def compute_primal(self, bayVolumes, fuelMass):
        volFrac = self.options["wingboxVolumeFraction"]
        wingboxVolume = jnp.sum(bayVolumes)

        # The calculations below are a bit confusing, they could be done more simply with a for loop and some if
        # statements but then the code wouldn't be jittable/differentiable by jax

        # Start by assuming all bays are full
        bayFullFuelMasses = jnp.array(bayVolumes * volFrac * self.options["fuelDensity"]) * 2

        # Track the cumulative sum of fuel in the bays (flip the array so we are effectively filling from the wing tip inwards)
        cumulativeFuelMasses = jnp.flip(jnp.cumsum(jnp.flip(bayFullFuelMasses)))

        # Compute how much fuel is left to store if we fill up to each bay
        remainingFuelMass = fuelMass - cumulativeFuelMasses

        # If we add the remaining fuel mass to the full bay mass, then we will get the amount of fuel that would need to
        # be stored in each bay to reach the required fuel mass, assuming all outboard tanks are filled first. For most
        # bays this value will either be greater than the amount of fuel they can store (indicating we need to fill the
        # bays inboard of this one), or negative (indicating we don't need to fill this bay). The only bay that will
        # have a positive value less than the full bay mass is the bay where we stop filling. To get the actual mass in
        # each bay, we therefore need to clip these values between 0 and the full bay mass. However, we want to use a
        # smooth version of the clip function to avoid discontinuities in the derivatives.

        bayFuelMasses = bayFullFuelMasses + remainingFuelMass

        bayFuelMasses = 0.5 * self.smoothClip(
            bayFuelMasses,
            0.0,
            bayFullFuelMasses,
            maxRelError=self.options["maxSmoothingRelError"],
        )

        return bayFuelMasses, wingboxVolume

    @staticmethod
    def KSMax2(a, b, rho):
        """Elementwise KS maximum of two arrays

        Parameters
        ----------
        a : array_like
            First array
        b : array_like
            Second array
        rho : float/complex or array_like
            Rho value for KS aggregation, higher values give a closer, but less smooth, approximation to the true max

        Returns
        -------
        array_like
            Elementwise KS maximum of a and b
        """
        minVal = jnp.minimum(a, b)
        maxVal = jnp.maximum(a, b)
        return maxVal + 1 / rho * jnp.log(1 + jnp.exp(rho * (minVal - maxVal)))

    @staticmethod
    def smoothClip(x, lb, ub, maxRelError=1e-4):
        """A smooth approximation to the clip function using KS aggregation

        Parameters
        ----------
        x : array_like
            Values to be clipped
        lb : float/complex or array_like
            Lower bound, can be a single value or an array of same shape as x
        ub : float/complex
            Upper bound, can be a single value or an array of same shape as x
        maxRelError : float/complex
            The maximum error in this clipping will occur when x is at the lb or ub value. This parameter is used to
            pick the rho value used in the KS aggregation such that the error at these points is less than maxRelError,
            relative to the range (ub - lb).

        Returns
        -------
        float/complex
            Clipped values
        """
        # Convert lb and ub to arrays if they are single values
        if np.isscalar(lb):
            lb = jnp.full_like(x, lb)
        if np.isscalar(ub):
            ub = jnp.full_like(x, ub)

        # The maximum error in the two element KSMax function is 1/rho * log(2), which is roughly 0.7/rho and occurs when
        # the two inputs are equal. To be conservative, we will therefore choose rho = 1 / maxAllowedError
        width = ub - lb
        maxError = maxRelError * width
        rho = 1 / maxError

        # First do min of x and ub, KSMin = -KSMax(-f)
        clipped = -FuelDistributionComp.KSMax2(-x, -ub, rho)
        return FuelDistributionComp.KSMax2(clipped, lb, rho)


class FuelDistributionGroup(om.Group):
    def initialize(self):
        self.options.declare("aircraftSpecs", types=dict)
        self.options.declare("numRibBays", types=int)
        self.options.declare(
            "volumeVarName",
            types=str,
            desc="Name of the variable containing the rib bay volumes",
        )
        self.options.declare("maxSmoothingRelError", types=float, default=1e-4)

    def setup(self):
        specs = self.options["aircraftSpecs"]

        # Need a component to mux the scalar bay volumes computed by the geometry component into an array
        bayVolMuxer = om.MuxComp(vec_size=self.options["numRibBays"])
        bayVolMuxer.add_var(self.options["volumeVarName"], units="m**3")
        self.add_subsystem(
            "bayVolMuxer",
            bayVolMuxer,
            promotes=["*"],
        )

        # This is the component that computes the fuel masses in each bay from the total fuel mass and the bay volumes
        # fuelMass input will be automatically connected to the dvSys output through promotion
        self.add_subsystem(
            "fuelMassDistribution",
            FuelDistributionComp(
                fuelDensity=specs["fuelDensity"],
                wingboxVolumeFraction=specs["wingboxFuelVolumeFraction"],
                numRibBays=self.options["numRibBays"],
                maxSmoothingRelError=self.options["maxSmoothingRelError"],
            ),
            promotes_outputs=["*"],
            promotes_inputs=["fuelMass", ("bayVolumes", self.options["volumeVarName"])],
        )


class LiftConstraintGroup(om.Group):
    def initialize(self):
        self.options.declare("aircraftSpecs", types=dict)
        self.options.declare("flightPointSet", types=list)

    def setup(self):
        flightPointSet = self.options["flightPointSet"]

        for fp in flightPointSet:
            if isinstance(fp, FlightPoint):
                if fp.massConfig is not None:
                    fuelFraction = None
                    inputMassVar = fp.massConfig
                else:
                    fuelFraction = fp.fuelFraction
                    inputMassVar = f"{fp.name}_mass"
                    self.add_subsystem(
                        f"{fp.name}_mass",
                        om.ExecComp(f"{inputMassVar} = landingGrossMass - fuelBurn * {fuelFraction}"),
                        promotes=["*"],
                    )

                liftConComp = LiftConstraintComp(loadFactor=fp.loadFactor)
                self.add_subsystem(
                    f"{fp.name}LiftConstraint",
                    liftConComp,
                    promotes_inputs=[
                        ("mass", inputMassVar),
                        ("lift", f"{fp.name}Lift"),
                    ],
                    promotes_outputs=[("liftDiff", f"{fp.name}LiftDiff")],
                )


class FuelConsistencyGroup(om.Group):
    def initialize(self):
        self.options.declare("aircraftSpecs", types=dict)
        self.options.declare("flightPointSet", types=list)

    def setup(self):
        specs = self.options["aircraftSpecs"]
        flightPointSet = self.options["flightPointSet"]

        for fp in flightPointSet:
            if fp.massConfig is not None:
                execComp = om.ExecComp(
                    f"fuelMassDiff = fuelMass - (aircraftMass - landingGrossMass + {specs['reserveFuelMass']})",
                    fuelMass={"units": "kg"},
                    aircraftMass={"units": "kg"},
                    landingGrossMass={"units": "kg"},
                    fuelMassDiff={"units": "kg"},
                )
                self.add_subsystem(
                    f"{fp.name}FuelConsistency",
                    execComp,
                    promotes_inputs=[
                        ("fuelMass", f"{fp.name}-fuelMass"),
                        "landingGrossMass",
                        ("aircraftMass", fp.massConfig),
                    ],
                    promotes_outputs=[("fuelMassDiff", f"{fp.name}FuelMassDiff")],
                )
            else:
                execComp = om.ExecComp(
                    f"fuelMassDiff = fuelMass - ({fp.fuelFraction} * fuelBurn + {specs['reserveFuelMass']})",
                    promotes_inputs=[("fuelMass", f"{fp.name}-fuelMass"), "fuelBurn"],
                    promotes_outputs=[("fuelMassDiff", f"{fp.name}FuelMassDiff")],
                )


class FuelAndMassConstraintGroup(om.Group):
    def initialize(self):
        self.options.declare("aircraftSpecs", types=dict)
        self.options.declare("flightPointSet", types=list)

    def setup(self):
        specs = self.options["aircraftSpecs"]
        self.add_subsystem(
            "liftConstraints",
            LiftConstraintGroup(
                aircraftSpecs=specs,
                flightPointSet=self.options["flightPointSet"],
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "fuelConsistency",
            FuelConsistencyGroup(
                aircraftSpecs=specs,
                flightPointSet=self.options["flightPointSet"],
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "FuelTankUsage",
            FuelTankUsageComp(
                reserveFuelMass=specs["reserveFuelMass"],
                fuelDensity=specs["fuelDensity"],
                wingboxVolumeFraction=specs["wingboxFuelVolumeFraction"],
                auxTankVolume=specs["auxFuelVolume"],
            ),
            promotes=["*"],
        )


# Test the performance group derivatives
if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../AircraftSpecs"))
    from AircraftSpecs.STWSpecs import aircraftSpecs  # noqa: E402
    from AircraftSpecs.STWFlightPoints import flightPointSets  # noqa: E402

    prob = om.Problem()
    prob.model = FuelAndMassConstraintGroup(aircraftSpecs=aircraftSpecs, flightPointSet=flightPointSets["7pt"])
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=True, form="central", step=1e-8)
    om.n2(prob, show_browser=False)
