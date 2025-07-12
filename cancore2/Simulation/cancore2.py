
def configure_simulation():

    from cc3d.core.XMLUtils import ElementCC3D
    CompuCell3DElmnt=ElementCC3D("CompuCell3D",{"Revision":"2","Version":"4.6.0"})
    
    MetadataElmnt=CompuCell3DElmnt.ElementCC3D("Metadata")    
    # Basic properties simulation
    MetadataElmnt.ElementCC3D("NumberOfProcessors",{},"4")
    MetadataElmnt.ElementCC3D("DebugOutputFrequency",{},"10")
    # MetadataElmnt.ElementCC3D("NonParallelModule",{"Name":"Potts"})
    
    PottsElmnt=CompuCell3DElmnt.ElementCC3D("Potts")    
    # Basic properties of CPM (GGH) algorithm
    PottsElmnt.ElementCC3D("Dimensions",{"x":"100","y":"100","z":"1"})
    PottsElmnt.ElementCC3D("Steps",{},"1204")
    PottsElmnt.ElementCC3D("Temperature",{},"1.0")
    PottsElmnt.ElementCC3D("FluctuationAmplitude",{"CellType":"Medium"},"0.0")
    PottsElmnt.ElementCC3D("FluctuationAmplitude",{"CellType":"Lam"},"0.0")
    PottsElmnt.ElementCC3D("FluctuationAmplitude",{"CellType":"Col"},"0.0")
    PottsElmnt.ElementCC3D("FluctuationAmplitude",{"CellType":"ColL"},"1.0") 
    PottsElmnt.ElementCC3D("FluctuationAmplitude",{"CellType":"ColN"},"0.0") 
    PottsElmnt.ElementCC3D("FluctuationAmplitude",{"CellType":"Can1"},"4.0") 
    PottsElmnt.ElementCC3D("FluctuationAmplitude",{"CellType":"Can2"},"4.0") 
    PottsElmnt.ElementCC3D("FluctuationAmplitude",{"CellType":"Can3"},"4.0") 
    PottsElmnt.ElementCC3D("Flip2DimRatio",{},"2")
    PottsElmnt.ElementCC3D("NeighborOrder",{},"2")
    PottsElmnt.ElementCC3D("LatticeType",{},"Square")
    
    PluginElmnt=CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"CellType"})    
    # Listing all cell types in the simulation
    PluginElmnt.ElementCC3D("CellType",{"TypeId":"0","TypeName":"Medium"})
    PluginElmnt.ElementCC3D("CellType",{"TypeId":"1","TypeName":"Lam"})
    PluginElmnt.ElementCC3D("CellType",{"TypeId":"2","TypeName":"Col"})
    PluginElmnt.ElementCC3D("CellType",{"TypeId":"3","TypeName":"ColL"})
    PluginElmnt.ElementCC3D("CellType",{"TypeId":"4","TypeName":"ColN"})
    PluginElmnt.ElementCC3D("CellType",{"TypeId":"5","TypeName":"Can1"})
    PluginElmnt.ElementCC3D("CellType",{"TypeId":"6","TypeName":"Can2"})
    PluginElmnt.ElementCC3D("CellType",{"TypeId":"7","TypeName":"Can3"})
    
    CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"Volume"})
    
    CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"Surface"})
    
    PluginElmnt_1=CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"ExternalPotential"})    
    # External force applied to cell. Each cell has different force and force components have to be managed in Python.
    # e.g. cell.lambdaVecX=0.5; cell.lambdaVecY=0.1 ; cell.lambdaVecZ=0.3;
    PluginElmnt_1.ElementCC3D("Algorithm",{},"PixelBased")
    
    PluginElmnt_2=CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"CenterOfMass"})    
    # Module tracking center of mass of each cell
    
    PluginElmnt_3=CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"NeighborTracker"})    
    # Module tracking neighboring cells of each cell
    
    PluginElmnt_4=CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"PixelTracker"})    
    # Module tracking pixels of each cell
    
    PluginElmnt_5=CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"BoundaryPixelTracker"})    
    # Module tracking boundary pixels of each cell
    PluginElmnt_5.ElementCC3D("NeighborOrder",{},"2")
    
    PluginElmnt_6=CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"Contact"})
    # Specification of adhesion energies
    PluginElmnt_6.ElementCC3D("Energy",{"Type1":"Medium","Type2":"Medium"},"0.0")
    PluginElmnt_6.ElementCC3D("Energy",{"Type1":"Medium","Type2":"Lam"},"16.0")
    PluginElmnt_6.ElementCC3D("Energy",{"Type1":"Medium","Type2":"Col"},"16.0")
    PluginElmnt_6.ElementCC3D("Energy",{"Type1":"Medium","Type2":"ColL"},"16.0")
    PluginElmnt_6.ElementCC3D("Energy",{"Type1":"Medium","Type2":"ColN"},"16.0")
    PluginElmnt_6.ElementCC3D("Energy",{"id":"Md1","Type1":"Medium","Type2":"Can1"},"32.0")
    PluginElmnt_6.ElementCC3D("Energy",{"id":"Md2","Type1":"Medium","Type2":"Can2"},"32.0")
    PluginElmnt_6.ElementCC3D("Energy",{"id":"Md3","Type1":"Medium","Type2":"Can3"},"32.0")
    PluginElmnt_6.ElementCC3D("Energy",{"Type1":"Lam","Type2":"Lam"},"16.0")
    PluginElmnt_6.ElementCC3D("Energy",{"Type1":"Lam","Type2":"Col"},"16.0")
    PluginElmnt_6.ElementCC3D("Energy",{"Type1":"Lam","Type2":"ColL"},"16.0")
    PluginElmnt_6.ElementCC3D("Energy",{"Type1":"Lam","Type2":"ColN"},"16.0")
    PluginElmnt_6.ElementCC3D("Energy",{"id":"Lm1","Type1":"Lam","Type2":"Can1"},"16.0")
    PluginElmnt_6.ElementCC3D("Energy",{"id":"Lm2","Type1":"Lam","Type2":"Can2"},"16.0")
    PluginElmnt_6.ElementCC3D("Energy",{"id":"Lm3","Type1":"Lam","Type2":"Can3"},"16.0")
    PluginElmnt_6.ElementCC3D("Energy",{"Type1":"Col","Type2":"Col"},"16.0")
    PluginElmnt_6.ElementCC3D("Energy",{"Type1":"Col","Type2":"ColL"},"16.0")
    PluginElmnt_6.ElementCC3D("Energy",{"Type1":"Col","Type2":"ColN"},"16.0")
    PluginElmnt_6.ElementCC3D("Energy",{"id":"Co1","Type1":"Col","Type2":"Can1"},"32.0")
    PluginElmnt_6.ElementCC3D("Energy",{"id":"Co2","Type1":"Col","Type2":"Can2"},"32.0")
    PluginElmnt_6.ElementCC3D("Energy",{"id":"Co3","Type1":"Col","Type2":"Can3"},"32.0")
    PluginElmnt_6.ElementCC3D("Energy",{"Type1":"ColL","Type2":"ColL"},"16.0")
    PluginElmnt_6.ElementCC3D("Energy",{"Type1":"ColL","Type2":"ColN"},"16.0")
    PluginElmnt_6.ElementCC3D("Energy",{"id":"CL1","Type1":"ColL","Type2":"Can1"},"16.0")
    PluginElmnt_6.ElementCC3D("Energy",{"id":"CL2","Type1":"ColL","Type2":"Can2"},"16.0")
    PluginElmnt_6.ElementCC3D("Energy",{"id":"CL3","Type1":"ColL","Type2":"Can3"},"16.0")
    PluginElmnt_6.ElementCC3D("Energy",{"Type1":"ColN","Type2":"ColN"},"16.0")
    PluginElmnt_6.ElementCC3D("Energy",{"id":"CN1","Type1":"ColN","Type2":"Can1"},"32.0")
    PluginElmnt_6.ElementCC3D("Energy",{"id":"CN2","Type1":"ColN","Type2":"Can2"},"32.0")
    PluginElmnt_6.ElementCC3D("Energy",{"id":"CN3","Type1":"ColN","Type2":"Can3"},"32.0")
    PluginElmnt_6.ElementCC3D("Energy",{"id":"C11","Type1":"Can1","Type2":"Can1"},"32.0")
    PluginElmnt_6.ElementCC3D("Energy",{"id":"C12","Type1":"Can1","Type2":"Can2"},"32.0")
    PluginElmnt_6.ElementCC3D("Energy",{"id":"C13","Type1":"Can1","Type2":"Can3"},"32.0")
    PluginElmnt_6.ElementCC3D("Energy",{"id":"C22","Type1":"Can2","Type2":"Can2"},"32.0")
    PluginElmnt_6.ElementCC3D("Energy",{"id":"C23","Type1":"Can2","Type2":"Can3"},"32.0")
    PluginElmnt_6.ElementCC3D("Energy",{"id":"C33","Type1":"Can3","Type2":"Can3"},"32.0")
    PluginElmnt_6.ElementCC3D("NeighborOrder",{},"2")
     
    PluginElmnt_7=CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"Chemotaxis"})    
    # You may repeat ChemicalField element for each chemical field declared in the PDE solvers
    # Specification of chemotaxis properties of select cell types.
    ChemicalFieldElmnt=PluginElmnt_7.ElementCC3D("ChemicalField",{"Name":"EGF"})
    ChemicalFieldElmnt.ElementCC3D("ChemotaxisByType",{"Lambda":"40.0","Type":"Can1"})
    ChemicalFieldElmnt=PluginElmnt_7.ElementCC3D("ChemicalField",{"Name":"EGF"})
    ChemicalFieldElmnt.ElementCC3D("ChemotaxisByType",{"Lambda":"40.0","Type":"Can2"})
    ChemicalFieldElmnt=PluginElmnt_7.ElementCC3D("ChemicalField",{"Name":"EGF"})
    ChemicalFieldElmnt.ElementCC3D("ChemotaxisByType",{"Lambda":"40.0","Type":"Can3"})
    ChemicalFieldElmnt=PluginElmnt_7.ElementCC3D("ChemicalField",{"Name":"Oxy"})
    ChemicalFieldElmnt.ElementCC3D("ChemotaxisByType",{"Lambda":"2.0","Type":"Can1"})
    ChemicalFieldElmnt=PluginElmnt_7.ElementCC3D("ChemicalField",{"Name":"Oxy"})
    ChemicalFieldElmnt.ElementCC3D("ChemotaxisByType",{"Lambda":"2.0","Type":"Can2"})
    ChemicalFieldElmnt=PluginElmnt_7.ElementCC3D("ChemicalField",{"Name":"Oxy"})
    ChemicalFieldElmnt.ElementCC3D("ChemotaxisByType",{"Lambda":"2.0","Type":"Can3"})
    
    PluginElmnt_8=CompuCell3DElmnt.ElementCC3D("Plugin",{"Name":"Secretion"})    
    # Specification of secretion properties of select cell types.
    # You may repeat Field element for each chemical field declared in the PDE solvers
    # Specification of secretion properties of individual cells can be done in Python
    
    SteppableElmnt=CompuCell3DElmnt.ElementCC3D("Steppable",{"Type":"DiffusionSolverFE"})
    
    # Specification of PDE solvers
    DiffusionFieldElmnt=SteppableElmnt.ElementCC3D("DiffusionField",{"Name":"EGF"})
    DiffusionDataElmnt=DiffusionFieldElmnt.ElementCC3D("DiffusionData")
    DiffusionDataElmnt.ElementCC3D("FieldName",{},"EGF")
    DiffusionDataElmnt.ElementCC3D("GlobalDiffusionConstant",{},"0.20")
    DiffusionDataElmnt.ElementCC3D("GlobalDecayConstant",{},"0.0")
    # Additional options are:
    # DiffusionDataElmnt.ElementCC3D("InitialConcentrationExpression",{},"x*y")
    # DiffusionDataElmnt.ElementCC3D("ConcentrationFileName",{},"INITIAL CONCENTRATION FIELD - typically a file with path Simulation/NAME_OF_THE_FILE.txt")
    DiffusionDataElmnt.ElementCC3D("DiffusionCoefficient",{"CellType":"Lam"},"0.20")
    DiffusionDataElmnt.ElementCC3D("DiffusionCoefficient",{"CellType":"Col"},"0.20")
    DiffusionDataElmnt.ElementCC3D("DiffusionCoefficient",{"CellType":"ColL"},"0.20")
    DiffusionDataElmnt.ElementCC3D("DiffusionCoefficient",{"CellType":"ColN"},"0.20")
    DiffusionDataElmnt.ElementCC3D("DiffusionCoefficient",{"CellType":"Can1"},"0.20")
    DiffusionDataElmnt.ElementCC3D("DiffusionCoefficient",{"CellType":"Can2"},"0.20")
    DiffusionDataElmnt.ElementCC3D("DiffusionCoefficient",{"CellType":"Can3"},"0.20")
    DiffusionDataElmnt.ElementCC3D("DecayCoefficient",{"CellType":"Lam"},"0.0")
    DiffusionDataElmnt.ElementCC3D("DecayCoefficient",{"CellType":"Col"},"0.0")
    DiffusionDataElmnt.ElementCC3D("DecayCoefficient",{"CellType":"ColL"},"0.0")
    DiffusionDataElmnt.ElementCC3D("DecayCoefficient",{"CellType":"ColN"},"0.0")
    DiffusionDataElmnt.ElementCC3D("DecayCoefficient",{"CellType":"Can1"},"0.0")
    DiffusionDataElmnt.ElementCC3D("DecayCoefficient",{"CellType":"Can2"},"0.0")
    DiffusionDataElmnt.ElementCC3D("DecayCoefficient",{"CellType":"Can3"},"0.0")
    SecretionDataElmnt=DiffusionFieldElmnt.ElementCC3D("SecretionData")
    # When secretion is defined inside DissufionSolverFE all secretion constants are scaled automaticaly to account for the extra calls to the diffusion step when handling large diffusion constants
    # SecretionDataElmnt.ElementCC3D("Secretion",{"Type":"Can2"},"0.1")
    # SecretionDataElmnt.ElementCC3D("SecretionOnContact",{"SecreteOnContactWith":"Lam,Col,ColL,ColN,Can1,Can2","Type":"Can2"},"0.2")
    # SecretionDataElmnt.ElementCC3D("ConstantConcentration",{"Type":"Can2"},"0.1")
    BoundaryConditionsElmnt=DiffusionFieldElmnt.ElementCC3D("BoundaryConditions")
    PlaneElmnt=BoundaryConditionsElmnt.ElementCC3D("Plane",{"Axis":"X"})
    PlaneElmnt.ElementCC3D("ConstantDerivative",{"PlanePosition":"Min","Value":"0.0"})
    PlaneElmnt.ElementCC3D("ConstantDerivative",{"PlanePosition":"Max","Value":"0.0"})
    # Other options are (examples):
    # PlaneElmnt.ElementCC3D("Periodic")
    # PlaneElmnt.ElementCC3D("ConstantValue",{"PlanePosition":"Min","Value":"1.0"})
    PlaneElmnt_1=BoundaryConditionsElmnt.ElementCC3D("Plane",{"Axis":"Y"})
    PlaneElmnt_1.ElementCC3D("ConstantDerivative",{"PlanePosition":"Min","Value":"0.0"})
    PlaneElmnt_1.ElementCC3D("ConstantDerivative",{"PlanePosition":"Max","Value":"0.0"})
    # Other options are (examples):
    # PlaneElmnt_1.ElementCC3D("Periodic")
    # PlaneElmnt_1.ElementCC3D("ConstantValue",{"PlanePosition":"Min","Value":"1.0"})
    
    DiffusionFieldElmnt_1=SteppableElmnt.ElementCC3D("DiffusionField",{"Name":"MMP"})
    DiffusionDataElmnt_1=DiffusionFieldElmnt_1.ElementCC3D("DiffusionData")
    DiffusionDataElmnt_1.ElementCC3D("FieldName",{},"MMP")
    DiffusionDataElmnt_1.ElementCC3D("GlobalDiffusionConstant",{},"0.02")
    DiffusionDataElmnt_1.ElementCC3D("GlobalDecayConstant",{},"0.005")
    # Additional options are:
    # DiffusionDataElmnt_1.ElementCC3D("InitialConcentrationExpression",{},"x*y")
    # DiffusionDataElmnt_1.ElementCC3D("ConcentrationFileName",{},"INITIAL CONCENTRATION FIELD - typically a file with path Simulation/NAME_OF_THE_FILE.txt")
    DiffusionDataElmnt_1.ElementCC3D("DiffusionCoefficient",{"CellType":"Lam"},"0.02")
    DiffusionDataElmnt_1.ElementCC3D("DiffusionCoefficient",{"CellType":"Col"},"0.02")
    DiffusionDataElmnt_1.ElementCC3D("DiffusionCoefficient",{"CellType":"ColL"},"0.02")
    DiffusionDataElmnt_1.ElementCC3D("DiffusionCoefficient",{"CellType":"ColN"},"0.02")
    DiffusionDataElmnt_1.ElementCC3D("DiffusionCoefficient",{"CellType":"Can1"},"0.02")
    DiffusionDataElmnt_1.ElementCC3D("DiffusionCoefficient",{"CellType":"Can2"},"0.02")
    DiffusionDataElmnt_1.ElementCC3D("DiffusionCoefficient",{"CellType":"Can3"},"0.02")
    DiffusionDataElmnt_1.ElementCC3D("DecayCoefficient",{"CellType":"Lam"},"0.005")
    DiffusionDataElmnt_1.ElementCC3D("DecayCoefficient",{"CellType":"Col"},"0.005")
    DiffusionDataElmnt_1.ElementCC3D("DecayCoefficient",{"CellType":"ColL"},"0.005")
    DiffusionDataElmnt_1.ElementCC3D("DecayCoefficient",{"CellType":"ColN"},"0.005")
    DiffusionDataElmnt_1.ElementCC3D("DecayCoefficient",{"CellType":"Can1"},"0.005")
    DiffusionDataElmnt_1.ElementCC3D("DecayCoefficient",{"CellType":"Can2"},"0.005")
    DiffusionDataElmnt_1.ElementCC3D("DecayCoefficient",{"CellType":"Can3"},"0.005")
    SecretionDataElmnt_1=DiffusionFieldElmnt_1.ElementCC3D("SecretionData")
    # When secretion is defined inside DissufionSolverFE all secretion constants are scaled automaticaly to account for the extra calls to the diffusion step when handling large diffusion constants
    # SecretionDataElmnt_1.ElementCC3D("Secretion",{"Type":"Can2"},"0.1")
    # SecretionDataElmnt_1.ElementCC3D("SecretionOnContact",{"SecreteOnContactWith":"Lam,Col,ColL,ColN,Can1,Can2","Type":"Can2"},"0.2")
    # SecretionDataElmnt_1.ElementCC3D("ConstantConcentration",{"Type":"Can2"},"0.1")
    BoundaryConditionsElmnt_1=DiffusionFieldElmnt_1.ElementCC3D("BoundaryConditions")
    PlaneElmnt_2=BoundaryConditionsElmnt_1.ElementCC3D("Plane",{"Axis":"X"})
    PlaneElmnt_2.ElementCC3D("ConstantDerivative",{"PlanePosition":"Min","Value":"0.0"})
    PlaneElmnt_2.ElementCC3D("ConstantDerivative",{"PlanePosition":"Max","Value":"0.0"})
    # Other options are (examples):
    # PlaneElmnt_2.ElementCC3D("Periodic")
    # PlaneElmnt_2.ElementCC3D("ConstantValue",{"PlanePosition":"Min","Value":"1.0"})
    PlaneElmnt_3=BoundaryConditionsElmnt_1.ElementCC3D("Plane",{"Axis":"Y"})
    PlaneElmnt_3.ElementCC3D("ConstantDerivative",{"PlanePosition":"Min","Value":"0.0"})
    PlaneElmnt_3.ElementCC3D("ConstantDerivative",{"PlanePosition":"Max","Value":"0.0"})
    # Other options are (examples):
    # PlaneElmnt_3.ElementCC3D("Periodic")
    # PlaneElmnt_3.ElementCC3D("ConstantValue",{"PlanePosition":"Min","Value":"1.0"})
    
    DiffusionFieldElmnt_2=SteppableElmnt.ElementCC3D("DiffusionField",{"Name":"TIM"})
    DiffusionDataElmnt_2=DiffusionFieldElmnt_2.ElementCC3D("DiffusionData")
    DiffusionDataElmnt_2.ElementCC3D("FieldName",{},"TIM")
    DiffusionDataElmnt_2.ElementCC3D("GlobalDiffusionConstant",{},"0.04")
    DiffusionDataElmnt_2.ElementCC3D("GlobalDecayConstant",{},"0.005")
    # Additional options are:
    # DiffusionDataElmnt_2.ElementCC3D("InitialConcentrationExpression",{},"x*y")
    # DiffusionDataElmnt_2.ElementCC3D("ConcentrationFileName",{},"INITIAL CONCENTRATION FIELD - typically a file with path Simulation/NAME_OF_THE_FILE.txt")
    DiffusionDataElmnt_2.ElementCC3D("DiffusionCoefficient",{"CellType":"Lam"},"0.04")
    DiffusionDataElmnt_2.ElementCC3D("DiffusionCoefficient",{"CellType":"Col"},"0.04")
    DiffusionDataElmnt_2.ElementCC3D("DiffusionCoefficient",{"CellType":"ColL"},"0.04")
    DiffusionDataElmnt_2.ElementCC3D("DiffusionCoefficient",{"CellType":"ColN"},"0.04")
    DiffusionDataElmnt_2.ElementCC3D("DiffusionCoefficient",{"CellType":"Can1"},"0.04")
    DiffusionDataElmnt_2.ElementCC3D("DiffusionCoefficient",{"CellType":"Can2"},"0.04")
    DiffusionDataElmnt_2.ElementCC3D("DiffusionCoefficient",{"CellType":"Can3"},"0.04")
    DiffusionDataElmnt_2.ElementCC3D("DecayCoefficient",{"CellType":"Lam"},"0.005")
    DiffusionDataElmnt_2.ElementCC3D("DecayCoefficient",{"CellType":"Col"},"0.005")
    DiffusionDataElmnt_2.ElementCC3D("DecayCoefficient",{"CellType":"ColL"},"0.005")
    DiffusionDataElmnt_2.ElementCC3D("DecayCoefficient",{"CellType":"ColN"},"0.005")
    DiffusionDataElmnt_2.ElementCC3D("DecayCoefficient",{"CellType":"Can1"},"0.005")
    DiffusionDataElmnt_2.ElementCC3D("DecayCoefficient",{"CellType":"Can2"},"0.005")
    DiffusionDataElmnt_2.ElementCC3D("DecayCoefficient",{"CellType":"Can3"},"0.005")
    SecretionDataElmnt_2=DiffusionFieldElmnt_2.ElementCC3D("SecretionData")
    # When secretion is defined inside DissufionSolverFE all secretion constants are scaled automaticaly to account for the extra calls to the diffusion step when handling large diffusion constants
    # SecretionDataElmnt_2.ElementCC3D("Secretion",{"Type":"Can2"},"0.1")
    # SecretionDataElmnt_2.ElementCC3D("SecretionOnContact",{"SecreteOnContactWith":"Lam,Col,ColL,ColN,Can1,Can2","Type":"Can2"},"0.2")
    # SecretionDataElmnt_2.ElementCC3D("ConstantConcentration",{"Type":"Can2"},"0.1")
    BoundaryConditionsElmnt_2=DiffusionFieldElmnt_2.ElementCC3D("BoundaryConditions")
    PlaneElmnt_4=BoundaryConditionsElmnt_2.ElementCC3D("Plane",{"Axis":"X"})
    PlaneElmnt_4.ElementCC3D("ConstantDerivative",{"PlanePosition":"Min","Value":"0.0"})
    PlaneElmnt_4.ElementCC3D("ConstantDerivative",{"PlanePosition":"Max","Value":"0.0"})
    # Other options are (examples):
    # PlaneElmnt_4.ElementCC3D("Periodic")
    # PlaneElmnt_4.ElementCC3D("ConstantValue",{"PlanePosition":"Min","Value":"1.0"})
    PlaneElmnt_5=BoundaryConditionsElmnt_2.ElementCC3D("Plane",{"Axis":"Y"})
    PlaneElmnt_5.ElementCC3D("ConstantDerivative",{"PlanePosition":"Min","Value":"0.0"})
    PlaneElmnt_5.ElementCC3D("ConstantDerivative",{"PlanePosition":"Max","Value":"0.0"})
    # Other options are (examples):
    # PlaneElmnt_5.ElementCC3D("Periodic")
    # PlaneElmnt_5.ElementCC3D("ConstantValue",{"PlanePosition":"Min","Value":"1.0"})
    
    DiffusionFieldElmnt_3=SteppableElmnt.ElementCC3D("DiffusionField",{"Name":"Oxy"})
    DiffusionDataElmnt_3=DiffusionFieldElmnt_3.ElementCC3D("DiffusionData")
    DiffusionDataElmnt_3.ElementCC3D("FieldName",{},"Oxy")
    DiffusionDataElmnt_3.ElementCC3D("GlobalDiffusionConstant",{},"40.0")
    DiffusionDataElmnt_3.ElementCC3D("GlobalDecayConstant",{},"0.0")
    # Additional options are:
    # DiffusionDataElmnt_3.ElementCC3D("InitialConcentrationExpression",{},"x*y")
    # DiffusionDataElmnt_3.ElementCC3D("ConcentrationFileName",{},"INITIAL CONCENTRATION FIELD - typically a file with path Simulation/NAME_OF_THE_FILE.txt")
    DiffusionDataElmnt_3.ElementCC3D("DiffusionCoefficient",{"CellType":"Lam"},"40.0")
    DiffusionDataElmnt_3.ElementCC3D("DiffusionCoefficient",{"CellType":"Col"},"40.0")
    DiffusionDataElmnt_3.ElementCC3D("DiffusionCoefficient",{"CellType":"ColL"},"40.0")
    DiffusionDataElmnt_3.ElementCC3D("DiffusionCoefficient",{"CellType":"ColN"},"40.0")
    DiffusionDataElmnt_3.ElementCC3D("DiffusionCoefficient",{"CellType":"Can1"},"40.0")
    DiffusionDataElmnt_3.ElementCC3D("DiffusionCoefficient",{"CellType":"Can2"},"40.0")
    DiffusionDataElmnt_3.ElementCC3D("DiffusionCoefficient",{"CellType":"Can3"},"40.0")
    DiffusionDataElmnt_3.ElementCC3D("DecayCoefficient",{"CellType":"Lam"},"0.0")
    DiffusionDataElmnt_3.ElementCC3D("DecayCoefficient",{"CellType":"Col"},"0.0")
    DiffusionDataElmnt_3.ElementCC3D("DecayCoefficient",{"CellType":"ColL"},"0.0")
    DiffusionDataElmnt_3.ElementCC3D("DecayCoefficient",{"CellType":"ColN"},"0.0")
    DiffusionDataElmnt_3.ElementCC3D("DecayCoefficient",{"CellType":"Can1"},"0.0")
    DiffusionDataElmnt_3.ElementCC3D("DecayCoefficient",{"CellType":"Can2"},"0.0")
    DiffusionDataElmnt_3.ElementCC3D("DecayCoefficient",{"CellType":"Can3"},"0.0")
    SecretionDataElmnt_3=DiffusionFieldElmnt_3.ElementCC3D("SecretionData")
    # When secretion is defined inside DissufionSolverFE all secretion constants are scaled automaticaly to account for the extra calls to the diffusion step when handling large diffusion constants
    # SecretionDataElmnt_3.ElementCC3D("Secretion",{"Type":"Can2"},"0.1")
    # SecretionDataElmnt_3.ElementCC3D("SecretionOnContact",{"SecreteOnContactWith":"Lam,Col,ColL,ColN,Can1,Can2","Type":"Can2"},"0.2")
    # SecretionDataElmnt_3.ElementCC3D("ConstantConcentration",{"Type":"Can2"},"0.1")
    BoundaryConditionsElmnt_3=DiffusionFieldElmnt_3.ElementCC3D("BoundaryConditions")
    PlaneElmnt_6=BoundaryConditionsElmnt_3.ElementCC3D("Plane",{"Axis":"X"})
    PlaneElmnt_6.ElementCC3D("ConstantValue",{"PlanePosition":"Min","Value":"20.0"})
    PlaneElmnt_6.ElementCC3D("ConstantValue",{"PlanePosition":"Max","Value":"20.0"})
    # Other options are (examples):
    # PlaneElmnt_6.ElementCC3D("Periodic")
    # PlaneElmnt_6.ElementCC3D("ConstantDerivative",{"PlanePosition":"Min","Value":"1.0"})
    PlaneElmnt_7=BoundaryConditionsElmnt_3.ElementCC3D("Plane",{"Axis":"Y"})
    PlaneElmnt_7.ElementCC3D("ConstantValue",{"PlanePosition":"Min","Value":"20.0"})
    PlaneElmnt_7.ElementCC3D("ConstantValue",{"PlanePosition":"Max","Value":"20.0"})
    # Other options are (examples):
    # PlaneElmnt_7.ElementCC3D("Periodic")
    # PlaneElmnt_7.ElementCC3D("ConstantDerivative",{"PlanePosition":"Min","Value":"1.0"})
    
    # SteppableElmnt_1=CompuCell3DElmnt.ElementCC3D("Steppable",{"Type":"UniformInitializer"})    
    # Initial layout of cells in the form of rectangular slab
    
    # SteppableElmnt_2=CompuCell3DElmnt.ElementCC3D("Steppable",{"Type":"BlobInitializer"})
    # Initial layout of cells in the form of spherical (circular in 2D) blobs
    
    # SteppableElmnt_3=CompuCell3DElmnt.ElementCC3D("Steppable",{"Type":"PIFInitializer"})
    # Initial layout of cells using PIFF file. Piff files can be generated using PIFGEnerator
    # SteppableElmnt_3.ElementCC3D("PIFName",{},"Simulation/initializers/cancer.piff")

    CompuCellSetup.setSimulationXMLDescription(CompuCell3DElmnt)        
    CompuCellSetup.setSimulationXMLDescription(CompuCell3DElmnt)

            
from cc3d import CompuCellSetup
configure_simulation()            
 
from cancore2Steppables import ParameterSteppable
CompuCellSetup.register_steppable(steppable=ParameterSteppable(frequency=1))

from cancore2Steppables import CellLayoutSteppable
CompuCellSetup.register_steppable(steppable=CellLayoutSteppable(frequency=1))

from cancore2Steppables import CellGrowthSteppable
CompuCellSetup.register_steppable(steppable=CellGrowthSteppable(frequency=1))

from cancore2Steppables import MitosisSteppable
CompuCellSetup.register_steppable(steppable=MitosisSteppable(frequency=1))

from cancore2Steppables import CellMotilitySteppable
CompuCellSetup.register_steppable(steppable=CellMotilitySteppable(frequency=1))

from cancore2Steppables import HypoxiaSteppable
CompuCellSetup.register_steppable(steppable=HypoxiaSteppable(frequency=1))

from cancore2Steppables import MatrixDegradationSteppable
CompuCellSetup.register_steppable(steppable=MatrixDegradationSteppable(frequency=1))

from cancore2Steppables import SecretionSteppable
CompuCellSetup.register_steppable(steppable=SecretionSteppable(frequency=1))

from cancore2Steppables import RecorderSteppable
CompuCellSetup.register_steppable(steppable=RecorderSteppable(frequency=1))

CompuCellSetup.run()
