#! /bin/bash
NCORES=$(wc -l < $PBS_NODEFILE)
mpiexec -n $NCORES python aeroStructRun-MultipointParallel.py \
--task derivCheck \
--flightPointSet 2pt \
--addStructDVs --useFuelMassDVs --includeBFL \
--addGeoDVs --sweep --span \
--output aerostructDerivCheck \
--aeroLevel 3 --structLevel 3 \
--initDVs DVs/FixedPlanformOpt-L2-Linear.pkl
