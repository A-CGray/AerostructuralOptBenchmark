#! /bin/bash
mpiexec -n 7 python aeroStructRun-MultipointParallel.py --task check --flightPointSet 7pt --includeBFL --output aerostructCheck --initDVs DVs/FixedPlanformOpt-L2-Linear.pkl
