import os
import sys
import openmdao.api as om
import numpy as np
import niceplots
import matplotlib.pyplot as plt
from performanceCalc import FuelDistributionComp

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../AircraftSpecs"))
from AircraftSpecs.STWSpecs import aircraftSpecs  # noqa: E402

plt.style.use(niceplots.get_style())


# Now test the FuelDistribution component
prob = om.Problem()

bayVolumes = np.array(
    [
        0.585374,
        0.591549,
        0.588046,
        0.740573,
        0.687938,
        0.637246,
        0.588490,
        0.541665,
        0.496780,
        0.453839,
        0.412841,
        0.373786,
        0.336672,
        0.301498,
        0.268263,
        0.236969,
        0.207615,
        0.180200,
        0.154724,
        0.131188,
        0.109595,
        0.0899446,
    ]
)
numBays = len(bayVolumes)
totalVolume = np.sum(bayVolumes) * aircraftSpecs["wingboxFuelVolumeFraction"]
totalMass = totalVolume * aircraftSpecs["fuelDensity"]


class Group(om.Group):
    def setup(self):
        inputComp = om.IndepVarComp()
        inputComp.add_output("bayVolumes", bayVolumes, units="m**3")
        inputComp.add_output("fuelMass", 1.0, units="kg")
        self.add_subsystem("inputs", inputComp, promotes=["*"])
        self.add_subsystem(
            "model",
            FuelDistributionComp(
                fuelDensity=aircraftSpecs["fuelDensity"],
                wingboxVolumeFraction=aircraftSpecs["wingboxFuelVolumeFraction"],
                numRibBays=numBays,
                maxSmoothingRelError=1e-1,
            ),
            promotes=["*"],
        )


prob.model = Group()
prob.setup()

fuelMasses = np.linspace(0, 1.1 * totalMass * 2, 501)
bayFuelMasses = np.zeros((numBays, len(fuelMasses)))
for ii, fm in enumerate(fuelMasses):
    prob.set_val("fuelMass", fm, units="kg")
    prob.run_model()
    bayFuelMasses[:, ii] = prob.get_val("bayFuelMasses", units="kg")

# Plot the results
fig, ax = plt.subplots(figsize=(8, 8))

cmap = plt.get_cmap("cool")
for ii in range(numBays):
    color = cmap(ii / (numBays - 1))
    ax.plot(fuelMasses / 2, bayFuelMasses[ii, :], label=f"Rib Bay {ii + 1}", clip_on=False, color=color)

niceplots.adjust_spines(ax)
niceplots.label_line_ends(ax)
ax.set_xlabel("Total Fuel Mass (kg)")
ax.set_ylabel("Rib Bay Fuel Mass (kg)")
niceplots.save_figs(fig, "fuelDistributionDemo", ["pdf", "png"])
