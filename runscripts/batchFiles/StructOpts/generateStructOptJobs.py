import os
from pbs4py import PBS

nas = PBS.nas(group_list="a1556", proc_type="sky", time=48, queue_name="long", profile_file="")
nas.shell = "zsh"
nas.mail_options = "bae"

levels = [4, 3, 2, 1]

runDir = "~/repos/AerostructuralOptBenchmark/runscripts"
baseOutputDir = "/nobackup/achris10/AerostructuralOptBenchmark"

for linType in ["Linear", "Nonlinear"]:
    for level in levels:
        numNodes = 1

        nas.mpiexec = f"mpiexec_mpt -n {numNodes*nas.ncpus_per_node}"
        nas.requested_number_of_nodes = numNodes

        jobName = f"StructOpt-L{level}-{linType}"
        outputDir = f"StructOpt/{jobName}"
        fullOutputDir = os.path.join(baseOutputDir, outputDir)
        linOption = "--nonlinear" if linType == "Nonlinear" else ""
        runCommand = f"python structRun.py --task opt --optType minMass  --addStructDVs --initPenalty 0.1 --structLevel 1 --output {outputDir} {linOption}"
        runCommand = nas.create_mpi_command(runCommand, output_root_name=os.path.join(fullOutputDir, jobName))

        jobBody = [f"mkdir -p {fullOutputDir}", f"cp {jobName}.pbs {fullOutputDir}/", f"cd {runDir}", runCommand]
        jobBody = [f"\n{line}" for line in jobBody]

        nas.write_job_file(
            job_filename=f"{jobName}.{nas.batch_file_extension}",
            job_name=jobName,
            job_body=jobBody,
        )
        # nas.launch(job_name=jobName, job_body=jobBody, blocking=False)
