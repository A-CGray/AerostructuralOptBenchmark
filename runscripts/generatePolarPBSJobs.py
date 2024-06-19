import os
from pbs4py import PBS

nas = PBS.nas(group_list="a1607", proc_type="sky", time=4, queue_name="normal", profile_file="")
nas.shell = "zsh"
nas.mail_options = "bae"

meshSizes = [int(1.8e5), int(9.7e5), int(7.8e6)]
levels = [3, 2, 1]
cellsPerProc = int(10e3)

runDir = "~/repos/AerostructuralOptBenchmark/runscripts"
baseOutputDir = "/nobackup/achris10/AerostructuralOptBenchmark"

batchFileDir = os.path.join("batchFiles", "polars")
os.makedirs(batchFileDir, exist_ok=True)

for level, meshSize in zip(levels, meshSizes):
    idealNumProcs = meshSize // cellsPerProc
    numNodes = max(1, idealNumProcs // nas.ncpus_per_node)
    numNodes = min(10, numNodes)

    nas.mpiexec = f"mpiexec_mpt -n {numNodes*nas.ncpus_per_node}"

    nas.requested_number_of_nodes = numNodes

    jobName = f"ADflowPolar-L{level}"
    outputDir = f"ADflowPolars/{jobName}"
    fullOutputDir = os.path.join(baseOutputDir, outputDir)
    runCommand = f"python adflowPolar.py --level {level} --output {outputDir} --task polar"
    runCommand = nas.create_mpi_command(runCommand, output_root_name=os.path.join(fullOutputDir, jobName))

    jobBody = [f"mkdir -p {fullOutputDir}", f"cp {jobName}.pbs {fullOutputDir}/", f"cd {runDir}", runCommand]

    nas.write_job_file(
        job_filename=os.path.join(batchFileDir, f"{jobName}.{nas.batch_file_extension}"),
        job_name=jobName,
        job_body=[f"\n{line}" for line in jobBody],
    )
    # nas.launch(job_name=jobName, job_body=[f"\n{line}" for line in jobBody], blocking=False)
