import os
from pbs4py import PBS
import numpy as np

runTime = 72

nas = PBS.nas(group_list="a1607", proc_type="sky", time=runTime, queue_name="long", profile_file="")
nas.shell = "zsh"
nas.mail_options = "bae"

meshSize = int(9.7e5)
level = 2
cellsPerProc = int(10e3)

rangeScales = np.arange(0.5, 1.51, 0.25)

runDir = "~/repos/AerostructuralOptBenchmark/runscripts"
baseOutputDir = "/nobackup/achris10/AerostructuralOptBenchmark"

for rangeScale in rangeScales:
    if rangeScale != 1:
        for linType in ["Linear", "Nonlinear"]:
            initDVs = os.path.join(runDir, "DVs", f"VariablePlanformOpt-L2-{linType}.pkl")
            restartDict = os.path.join(
                baseOutputDir, f"VariablePlanformOpt/VariablePlanformOpt-L2-{linType}-Part3", "SNOPTRestart.pkl"
            )
            idealNumProcs = 3 * meshSize // cellsPerProc
            numNodes = max(1, int(np.ceil(idealNumProcs / nas.ncpus_per_node)))
            numNodes = min(20, numNodes)
            totalProcs = numNodes * nas.ncpus_per_node

            procs = np.array([0.21875, 0.528125, 0.253125])  # 70 169 81
            procs /= np.sum(procs)
            procs *= totalProcs
            procs = procs.astype(int)
            remainingProcs = totalProcs - np.sum(procs)
            if remainingProcs != 0:
                for ii in range(remainingProcs):
                    procs[-(ii + 1)] += 1
            procString = " ".join([str(i) for i in procs])

            nas.mpiexec = f"mpiexec_mpt -n {totalProcs}"
            nas.requested_number_of_nodes = numNodes

            jobName = f"VariablePlanformOpt-L{level}-{linType}-RangeScale{rangeScale}"
            outputDir = f"VariablePlanformOpt/{jobName}"
            fullOutputDir = os.path.join(baseOutputDir, outputDir)
            linOption = "--nonlinear" if linType == "Nonlinear" else ""
            runCommand = f"""python aeroStructRun-MultipointParallel.py \\
--task opt --optType fuelburn \\
--initPenalty 0.1 --violLimit 0.05 --hessianUpdate 60 \\
--timeLimit {(runTime * 3600 - 600)} \\
--addStructDVs \\
--addGeoDVs --shape --twist --sweep --span --taper \\
--maxWingLoading 600 \\
--flightPointSet 3pt \\
--rangeScale {rangeScale} \\
--procs {procString} \\
--aeroLevel {level} --structLevel 1 {linOption} \\
--initDVs {initDVs} \\
--restartDict {restartDict} \\
--output {outputDir}"""
            runCommand = nas.create_mpi_command(runCommand, output_root_name=os.path.join(fullOutputDir, jobName))

            jobBody = [f"mkdir -p {fullOutputDir}", f"cp {jobName}.pbs {fullOutputDir}/", f"cd {runDir}", runCommand]
            jobBody = [f"\n{line}" for line in jobBody]

            nas.write_job_file(
                job_filename=f"{jobName}.{nas.batch_file_extension}",
                job_name=jobName,
                job_body=jobBody,
            )
            # nas.launch(job_name=jobName, job_body=jobBody, blocking=False)
