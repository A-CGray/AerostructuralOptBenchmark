#! /bin/bash

baseDir="/home/ali/BigBoi/ali/AerostructuralOptBenchmark/"
studyName="UnoStructOptTests"
studyDir="${baseDir}/${studyName}"

for alg in uno-filtersqp uno-filterslp uno-funnelsqp uno-ipopt; do
	mkdir -p "${studyDir}/StructOptLinear-${alg}"
	mpiexec -n 14 python structRun.py \
--output "${studyName}/StructOptLinear-${alg}" \
--postInitDVs \
--tolFactor 1.0 \
--optimiser ${alg} \
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
--structLevel 4 \
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
--optType minMass | tee "${studyDir}/StructOptLinear-${alg}/StructOptLinear-${alg}.log"
done
