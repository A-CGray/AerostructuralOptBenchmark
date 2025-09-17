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

paretoWeights = np.linspace(0, 1, 6)

runDir = "~/repos/AerostructuralOptBenchmark/runscripts"
baseOutputDir = "/nobackup/achris10/AerostructuralOptBenchmark"

for weight in paretoWeights:
    if weight != 1.0:
        for linType in ["Linear", "Nonlinear"]:
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

            prevJobName = f"VariablePlanformOpt-L{level}-{linType}-ParetoWeight-{weight:.2f}-Part2"
            jobName = prevJobName.replace("Part2", "Part3")
            outputDir = f"Fuelburn-TOGM-Pareto/{jobName}"
            fullOutputDir = os.path.join(baseOutputDir, outputDir)
            initDVs = os.path.join(baseOutputDir, "Fuelburn-TOGM-Pareto", prevJobName, "AeroStructOpt.hst")

            linOption = "--nonlinear" if linType == "Nonlinear" else ""
            runCommand = f"""python aeroStructRun-MultipointParallel.py \\
--task opt --optType pareto --paretoWeight {weight} \\
--initPenalty 0.1 --violLimit 0.05 --hessianUpdate 60 --stepLimit 0.01 \\
--timeLimit {(runTime * 3600 - 600)} \\
--addStructDVs \\
--addGeoDVs --shape --twist --sweep --span --taper \\
--maxWingLoading 600 \\
--flightPointSet 3pt \\
--procs {procString} \\
--aeroLevel {level} --structLevel 1 {linOption} \\
--initDVs {initDVs} \\
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
