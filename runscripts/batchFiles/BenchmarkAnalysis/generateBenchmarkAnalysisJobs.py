import os
from pbs4py import PBS

nas = PBS.nas(group_list="a1556", proc_type="sky", time=2, queue_name="normal", profile_file="")
nas.shell = "zsh"
nas.mail_options = "bae"

meshSizes = [int(1.8e5), int(9.7e5), int(7.8e6)]
levels = [3, 2, 1]
cellsPerProc = int(20e3)

runDir = "~/repos/AerostructuralOptBenchmark/runscripts"
baseOutputDir = "/nobackup/achris10/AerostructuralOptBenchmark"

for linType in ["Linear", "Nonlinear"]:
    for level, meshSize in zip(levels, meshSizes):
        idealNumProcs = 3 * meshSize // cellsPerProc
        numNodes = max(1, idealNumProcs // nas.ncpus_per_node)
        numNodes = min(10, numNodes)

        nas.mpiexec = f"mpiexec_mpt -n {numNodes*nas.ncpus_per_node}"
        nas.requested_number_of_nodes = numNodes

        jobName = f"BenchmarkAnalysis-L{level}-{linType}"
        outputDir = f"BenchmarkAnalysis/{jobName}"
        fullOutputDir = os.path.join(baseOutputDir, outputDir)
        linOption = "--nonlinear" if linType == "Nonlinear" else ""
        runCommand = f"python aeroStructRun-MultipointParallel.py --task analysis --flightPointSet 3pt --aeroLevel {level} --structLevel 1 --output {outputDir} {linOption}"
        runCommand = nas.create_mpi_command(runCommand, output_root_name=os.path.join(fullOutputDir, jobName))

        jobBody = [f"mkdir -p {fullOutputDir}", f"cp {jobName}.pbs {fullOutputDir}/", f"cd {runDir}", runCommand]
        jobBody = [f"\n{line}" for line in jobBody]

        nas.write_job_file(
            job_filename=f"{jobName}.{nas.batch_file_extension}",
            job_name=jobName,
            job_body=jobBody,
        )
        # nas.launch(job_name=jobName, job_body=jobBody, blocking=False)
