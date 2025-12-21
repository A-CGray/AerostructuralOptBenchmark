def writeOASMesh(mesh, filename):
    """
    Writes the OAS mesh in Tecplot .dat file format, for visualization and debugging purposes.

    Parameters
    ----------
    mesh[nx,ny,3] : numpy array
        The OAS mesh to be written.
    filename : str
        The file name including the .dat extension.
    """
    nChord = mesh.shape[0]
    nSpan = mesh.shape[1]
    with open(filename, "w") as f:
        f.write('TITLE = "OpenAeroStruct: Aerodynamic Mesh"\n')
        f.write('VARIABLES = "X", "Y", "Z"\n')
        f.write(f"Zone T=Mesh I={nChord}, J={nSpan},")
        f.write("DATAPACKING=BLOCK, VARLOCATION=([4,5]=CELLCENTERED)\n")
        # Write out node locations
        for k in range(3):
            for j in range(nSpan):
                for i in range(nChord):
                    f.write("%f " % (mesh[i, j, k]))
            f.write("\n")
