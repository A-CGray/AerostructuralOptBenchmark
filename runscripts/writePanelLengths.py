import dill

with open("Outputs.pkl", "rb") as f:
    outputs = dill.load(f)
structDVs = outputs["dvs.dv_struct"]

with open("structDVMap.pkl", "rb") as f:
    structDVMap = dill.load(f)

with open("PanelLengths.csv", "w") as f:
    for compName, dvNums in structDVMap.items():
        f.write(f"{compName},{structDVs[dvNums['panel length']]}\n")
