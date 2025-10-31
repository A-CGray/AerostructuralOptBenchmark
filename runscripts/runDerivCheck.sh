#! /bin/bash
NCORES=$(wc -l < $PBS_NODEFILE)
mpiexec -n $NCORES python aeroStructRun-MultipointParallel.py \
--task derivCheck \
--flightPointSet cruise \
--addStructDVs --useFuelMassDVs --includeBFL \
--addGeoDVs --sweep --shape --twist \
--tolFactor 1e-2 \
--output aerostructDerivCheck \
--aeroLevel 3 --structLevel 3 \
--initDVs DVs/FixedPlanformOpt-L2-Linear.pkl \
