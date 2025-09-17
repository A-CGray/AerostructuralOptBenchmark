#! /bin/bash

# This script generates the meshes for the structural model
chordSpacing=(5 10 20 40)
spanSpacing=(3 5 10 20)
verticalSpacing=(3 5 10 20)

for i in ${!chordSpacing[@]}; do
    let "level = 4 - $i"
    fileName="wingbox-L$level-Order2"
    python generateWingboxMesh.py --name $fileName --nChord ${chordSpacing[$i]} --nSpan ${spanSpacing[$i]} --nVertical ${verticalSpacing[$i]}
    python generateFuelMassRBEs.py --level $level
done

rm -f wingbox*.dcel wingbox*.dat
