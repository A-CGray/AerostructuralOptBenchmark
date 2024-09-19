#! /bin/bash

baseDir="/home/ali/BigBoi/ali/AerostructuralOptBenchmark/"
prevDVFile=""

for level in 4 3 2 1; do
	mkdir -p "${baseDir}/StructOpts/StructOptLinear-L${level}"
	mpiexec -n 14 python structRun.py \
--output StructOpts/StructOptLinear-L${level} \
--initDVs $prevDVFile \
--postInitDVs \
--tolFactor 1.0 \
--optimiser snopt \
--optIter 1000 \
--hessianUpdate 40 \
--initPenalty 0.1 \
--stepLimit 0.02 \
--feasibility 1e-06 \
--optimality 1e-06 \
--structScalingFactor 1.0 \
--geoScalingFactor 1.0 \
--addStructDVs \
--useComposite \
--usePanelLengthDVs \
--structLevel $level \
--structOrder 2 \
--thicknessAdjCon 0.0025 \
--heightAdjCon 0.01 \
--stiffAspectMax 30.0 \
--stiffAspectMin 5.0 \
--ksWeight 100.0 \
--ksType continuous \
--ffdLevel coarse \
--task opt \
--flightPointSet maneuverOnly \
--optType minMass | tee "${baseDir}/StructOpts/StructOptLinear-L${level}/StructOptLinear-L${level}.log"

	prevDVFile="${baseDir}/StructOpts/StructOptLinear-L${level}/Outputs.pkl"
done
