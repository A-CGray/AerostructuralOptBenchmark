import os
from pbs4py import PBS
import numpy as np

runTime = 8

nas = PBS.nas(
    group_list="a1607",
    proc_type="cas",
    time=runTime,
    queue_name="normal",
    profile_file="",
)
nas.shell = "zsh"
nas.mail_options = "bae"

meshSize = int(9.7e5)
cellsPerProc = int(10e3)

runDir = "~/repos/AerostructuralOptBenchmark/runscripts"
baseOutputDir = "/nobackup/achris10/AerostructuralOptBenchmark"

initDVs = {
    "FixedPlanform": {
        "Linear": os.path.join(
            baseOutputDir,
            "FixedPlanformOpt",
            "FixedPlanformOpt-L2-Linear",
            "Outputs.pkl",
        ),
        "Nonlinear": os.path.join(
            baseOutputDir,
            "FixedPlanformOpt",
            "FixedPlanformOpt-L2-Nonlinear",
            "Outputs.pkl",
        ),
    },
    "VariablePlanform": {
        "Linear": os.path.join(
            baseOutputDir,
            "VariablePlanformOpt",
            "VariablePlanformOpt-L2-Linear-Part3",
            "Outputs.pkl",
        ),
        "Nonlinear": os.path.join(
            baseOutputDir,
            "VariablePlanformOpt",
            "VariablePlanformOpt-L2-Nonlinear-Part3",
            "Outputs.pkl",
        ),
    },
}

for optType in ["FixedPlanform", "VariablePlanform"]:
    for linType in ["Linear", "Nonlinear"]:
        idealNumProcs = meshSize // cellsPerProc
        numNodes = max(1, int(np.ceil(idealNumProcs / nas.ncpus_per_node)))
        numNodes = min(20, numNodes)
        totalProcs = numNodes * nas.ncpus_per_node

        nas.mpiexec = f"mpiexec_mpt -n {totalProcs}"
        nas.requested_number_of_nodes = numNodes

        geoDVs = "--shape --twist"
        if optType == "VariablePlanform":
            geoDVs += " --sweep --span --taper"

        jobName = f"PostOptPolar-{optType}Opt-L2-{linType}-test"
        outputDir = f"PostOptPolars/{jobName}"
        fullOutputDir = os.path.join(baseOutputDir, outputDir)
        linOption = "--nonlinear" if linType == "Nonlinear" else ""
        runCommand = f"""python aeroStructRun-MultipointParallel.py \\
--task polar \\
--addStructDVs \\
--addGeoDVs {geoDVs} \\
--flightPointSet cruise \\
--aeroLevel 2 --structLevel 1 {linOption} \\
--initDVs {initDVs[optType][linType]} \\
--output {outputDir}"""
        runCommand = nas.create_mpi_command(runCommand, output_root_name=os.path.join(fullOutputDir, jobName))

        jobBody = [
            f"mkdir -p {fullOutputDir}",
            f"cp {jobName}.pbs {fullOutputDir}/",
            f"cd {runDir}",
            runCommand,
        ]
        jobBody = [f"\n{line}" for line in jobBody]

        nas.write_job_file(
            job_filename=f"{jobName}.{nas.batch_file_extension}",
            job_name=jobName,
            job_body=jobBody,
        )
        # nas.launch(job_name=jobName, job_body=jobBody, blocking=False)
