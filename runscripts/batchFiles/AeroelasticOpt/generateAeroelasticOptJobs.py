import os
from pbs4py import PBS

runTime = 48

nas = PBS.nas(group_list="a1556", proc_type="sky", time=runTime, queue_name="long", profile_file="")
nas.shell = "zsh"
nas.mail_options = "bae"

meshSizes = [int(1.8e5), int(9.7e5), int(7.8e6)]
levels = [3, 2, 1]
cellsPerProc = int(10e3)

runDir = "~/repos/AerostructuralOptBenchmark/runscripts"
baseOutputDir = "/nobackup/achris10/AerostructuralOptBenchmark"

for linType in ["Linear", "Nonlinear"]:
    initDVs = os.path.join(runDir, "DVs", f"StructOpt-L1-{linType}.pkl")
    for level, meshSize in zip(levels, meshSizes):
        idealNumProcs = 2 * meshSize // cellsPerProc
        numNodes = max(1, idealNumProcs // nas.ncpus_per_node)
        numNodes = min(20, numNodes)

        nas.mpiexec = f"mpiexec_mpt -n {numNodes*nas.ncpus_per_node}"
        nas.requested_number_of_nodes = numNodes

        jobName = f"AeroelasticOpt-L{level}-{linType}"
        outputDir = f"AeroelasticOpt/{jobName}"
        fullOutputDir = os.path.join(baseOutputDir, outputDir)
        linOption = "--nonlinear" if linType == "Nonlinear" else ""
        runCommand = f"python aeroStructRun-MultipointParallel.py --task opt --optType structMass --initPenalty 0.1 --timeLimit {(runTime*3600 - 600)} --addStructDVs --flightPointSet maneuverOnly --aeroLevel {level} --structLevel 1 {linOption} --initDVs {initDVs} --output {outputDir}"
        runCommand = nas.create_mpi_command(runCommand, output_root_name=os.path.join(fullOutputDir, jobName))

        jobBody = [f"mkdir -p {fullOutputDir}", f"cp {jobName}.pbs {fullOutputDir}/", f"cd {runDir}", runCommand]
        jobBody = [f"\n{line}" for line in jobBody]

        nas.write_job_file(
            job_filename=f"{jobName}.{nas.batch_file_extension}",
            job_name=jobName,
            job_body=jobBody,
        )
        # nas.launch(job_name=jobName, job_body=jobBody, blocking=False)
