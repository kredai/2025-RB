
from cc3d.core.PySteppables import *
from pathlib import Path
import numpy as np; import math; import csv 
import random; from random import uniform

Rval = {{Rval}} #1.00 # growth rate modifier
Eadh = {{Eadh}} #32.0 # cell-ECM contact energy (higher is weaker)
Cadh = {{Cadh}} #32.0 # cell-cell contact energy
Ctax = {{Ctax}} #40.0 # global chemotaxis strength (also modifies random motion)
Msec = {{Msec}} #1.00 # MMP/TIMP secretion rate
SSox = {{SSox}} #0.00 # oxygen threshold (state-switching to Rval/4, Eadh/4, Cadh*4, Ctax*2, Msec*2, OxyUp/4)
pRes = {{pRes}} #0.00 # probability of acquiring hypoxia resistance (Oxy50*4, Eadh/4, Cadh*4, Ctax*2, Msec*2, OxyUp/4) 


class ParameterSteppable(SteppableBasePy):
    def __init__(self,frequency=1):
        SteppableBasePy.__init__(self,frequency)
     
    def start(self):
        # set cell parameters
        global Eadh
        global Cadh 
        global Ctax      
        for cell in self.cellList:
            if (cell.type == self.COL or cell.type == self.COLN):
                cell.targetVolume = cell.volume
                cell.lambdaVolume = 20.0
                cell.targetSurface = cell.surface
                cell.lambdaSurface = 80.0
            if cell.type == self.LAM:
                cell.targetVolume = cell.volume
                cell.lambdaVolume = 20.0
                cell.targetSurface = cell.surface
                cell.lambdaSurface = 80.0    
            if (cell.type == self.CAN1 or cell.type == self.CAN2 or cell.type == self.CAN3):
                cell.targetVolume = 12.0
                cell.lambdaVolume = 80.0
                cell.targetSurface = math.floor(4.0*math.sqrt(cell.targetVolume)) 
                cell.lambdaSurface = 20.0
            if cell.type == self.CAN1:
                self.get_xml_element("Md1").cdata = Eadh
                self.get_xml_element("Lm1").cdata = Eadh/2.0
                self.get_xml_element("Co1").cdata = Eadh
                self.get_xml_element("CL1").cdata = Eadh/2.0
                self.get_xml_element("CN1").cdata = Eadh
                self.get_xml_element("C11").cdata = Cadh
                self.get_xml_element("C12").cdata = Cadh*2.0
                self.get_xml_element("C13").cdata = Cadh*2.0
                cd = self.chemotaxisPlugin.addChemotaxisData(cell,"EGF")
                cd.setLambda(Ctax)
                cd = self.chemotaxisPlugin.addChemotaxisData(cell,"Oxy")
                cd.setLambda(Ctax/20.0)
            if cell.type == self.CAN2:
                self.get_xml_element("Md2").cdata = Eadh/4.0
                self.get_xml_element("Lm2").cdata = Eadh/8.0
                self.get_xml_element("Co2").cdata = Eadh/4.0
                self.get_xml_element("CL2").cdata = Eadh/8.0
                self.get_xml_element("CN2").cdata = Eadh/4.0
                self.get_xml_element("C22").cdata = Cadh*4.0
                self.get_xml_element("C23").cdata = Cadh*4.0
                cd = self.chemotaxisPlugin.addChemotaxisData(cell,"EGF")
                cd.setLambda(Ctax*2.0)
                cd = self.chemotaxisPlugin.addChemotaxisData(cell,"Oxy")
                cd.setLambda(Ctax/10.0)
            if cell.type == self.CAN3:
                self.get_xml_element("Md3").cdata = Eadh/4.0
                self.get_xml_element("Lm3").cdata = Eadh/8.0
                self.get_xml_element("Co3").cdata = Eadh/4.0
                self.get_xml_element("CL3").cdata = Eadh/8.0
                self.get_xml_element("CN3").cdata = Eadh/4.0
                self.get_xml_element("C33").cdata = Cadh*4.0
                cd = self.chemotaxisPlugin.addChemotaxisData(cell,"EGF")
                cd.setLambda(Ctax*2.0)
                cd = self.chemotaxisPlugin.addChemotaxisData(cell,"Oxy")
                cd.setLambda(Ctax/10.0)   


class CellLayoutSteppable(SteppableBasePy):
    def __init__(self,frequency=1):
        SteppableBasePy.__init__(self,frequency)
        self.runBeforeMCS=1
    
    def start(self):
        # create collagen fibres
        xmax = self.dim.x
        ymax = self.dim.y
        Eden = 1.00 # matrix density
        fnum = int(Eden*xmax*ymax/10) # number of fibres
        for i in range(fnum):
            pixList = []
            x1 = random.randint(-12,xmax-1)
            y1 = random.randint(-12,ymax-1)
            theta = math.pi*(0.125+ 0.250*random.random()) # pi/8 to 3pi/8 rad
            flen = 8.0 + 8.0*random.random() # fibre length
            x2 = int(x1 + flen*math.cos(theta))
            y2 = int(y1 + flen*math.sin(theta))
            tlen = max(abs(x2-x1),abs(y2-y1))  
            dx = (x2-x1)/float(tlen)
            dy = (y2-y1)/float(tlen)
            pixList.append((x1,y1))
            for j in range(abs(tlen)):
                x1 += dx; y1 += dy
                pixList.append((math.floor(x1),math.floor(y1)))
            if len(pixList) > 0:
                cell = self.new_cell(self.COL) # create a new cell
                for (x,y) in pixList:
                    if (x >= 0 and x < xmax and y >= 0 and y < ymax):
                        self.cell_field[x:x+1,y:y+1,0] = cell # add this pixel to the cell
        
        # create cancer cells
        x = (0+xmax-1)/2.0
        y = (0+ymax-1)/2.0
        i = round(x)-6
        while i < (round(x)+5):
            i = i+3
            j = round(y)-6
            while j < (round(y)+5):
                j = j+3
                cell = self.new_cell(self.CAN1) # create a new cell
                self.cell_field[i-3:i,j-3:j,0] = cell # add this pixel to the cell       
                
        # create basement membrane
        min_radius = 6.0
        max_radius = 8.0
        for i in range(0,xmax-1):
            for j in range(0,ymax-1):
                if ((i-x)*(i-x) + (j-y)*(j-y)) < max_radius**2:
                    if ((i-x)*(i-x) + (j-y)*(j-y)) > min_radius**2:
                        cell = self.new_cell(self.LAM) # create a new cell
                        self.cell_field[i:i+1,j:j+1,0] = cell # add this pixel to the cell
       
                
class CellGrowthSteppable(SteppableBasePy):
    def __init__(self,frequency=1):
        SteppableBasePy.__init__(self,frequency)
        
    def step(self,mcs):
        fieldEGF = self.field.EGF
        fieldOxy = self.field.Oxy
        global Rval
        Egf50 = 4.0; Oxy50 = 4.0 
        Gmod = 1.0/30 # cells divide every 240 MCS (in theory)
        for cell in self.cellList:
            if cell.type == self.CAN1:
                EGFc=fieldEGF[int(round(cell.xCOM)),int(round(cell.yCOM)),int(round(cell.zCOM))]
                Oxyc=fieldOxy[int(round(cell.xCOM)),int(round(cell.yCOM)),int(round(cell.zCOM))]
                # EGFc can double growth, Oxyc can zero-out growth (Hill functions with n=2, half-max listed)
                cell.targetVolume += Rval * Gmod * (1 + 1/(1 + (Egf50/max(0.01, EGFc))**2)) * (1/(1 + (Oxy50/max(0.01, Oxyc))**2)) 
                cell.lambdaVolume = 80.0
            if cell.type == self.CAN2:
                EGFc=fieldEGF[int(round(cell.xCOM)),int(round(cell.yCOM)),int(round(cell.zCOM))]
                Oxyc=fieldOxy[int(round(cell.xCOM)),int(round(cell.yCOM)),int(round(cell.zCOM))]
                # EGFc can double growth, Oxyc can zero-out growth (Hill functions with n=2, half-max listed)
                cell.targetVolume += (Rval/4.0) * Gmod * (1 + 1/(1 + (Egf50/max(0.01, EGFc))**2)) * (1/(1 + (Oxy50/max(0.01, Oxyc))**2)) 
                cell.lambdaVolume = 80.0
            if cell.type == self.CAN3:
                EGFc=fieldEGF[int(round(cell.xCOM)),int(round(cell.yCOM)),int(round(cell.zCOM))]
                Oxyc=fieldOxy[int(round(cell.xCOM)),int(round(cell.yCOM)),int(round(cell.zCOM))]
                # EGFc can double growth, Oxyc can zero-out growth (Hill functions with n=2, half-max listed)
                cell.targetVolume += Rval * Gmod * (1 + 1/(1 + (Egf50/max(0.01, EGFc))**2)) * (1/(1 + ((Oxy50/4.0)/max(0.01, Oxyc))**2)) 
                cell.lambdaVolume = 80.0
            if (cell.type == self.CAN1 or cell.type == self.CAN2 or cell.type == self.CAN3):
                cell.targetSurface = math.floor(4.0*math.sqrt(cell.targetVolume)) 
                cell.lambdaSurface = 20.0
 

class MitosisSteppable(MitosisSteppableBase):
    def __init__(self,frequency=1):
        MitosisSteppableBase.__init__(self,frequency)
        
    def step(self, mcs):
        cells_to_divide=[]
        for cell in self.cellList:
            if (cell.type == self.CAN1 or cell.type == self.CAN2 or cell.type == self.CAN3):
                if cell.volume > 16.0: # cell width is >16um
                    cells_to_divide.append(cell)
        for cell in cells_to_divide:
            self.divide_cell_random_orientation(cell)
            # Other valid options
            # self.divide_cell_orientation_vector_based(cell,1,1,0)
            # self.divide_cell_along_major_axis(cell)
            # self.divide_cell_along_minor_axis(cell)
    def update_attributes(self):
        global pRes
        # reducing parent target volume
        self.parent_cell.targetVolume /= 2.0 
        self.parent_cell.targetSurface = math.floor(4.0*math.sqrt(self.parent_cell.targetVolume))            
        self.clone_parent_2_child()
        if self.parent_cell.type == self.CAN3:
            if random.random() > pRes: 
                self.child_cell.type = self.CAN2 # lose hypoxia resistance        
        # for more control of what gets copied from parent to child use cloneAttributes function
        # self.clone_attributes(source_cell=self.parent_cell, target_cell=self.child_cell, no_clone_key_dict_list=[attrib1, attrib2])
 

class CellMotilitySteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self,frequency)
     
    def start(self):
        global Ctax
        # iterating over all cells in simulation
        for cell in self.cell_list:
            # Make sure ExternalPotential plugin is loaded
            if cell.type == self.CAN1:
                # negative lambdaVecX makes force point in the positive direction
                # force component along X axis
                cell.lambdaVecX = Ctax * uniform(-0.25, 0.25)
                # force component along Y axis
                cell.lambdaVecY = Ctax * uniform(-0.25, 0.25)
                # cell.lambdaVecZ=0.0 # force component along Z axis
            if cell.type == self.CAN2:
                cell.lambdaVecX = Ctax*2.0 * uniform(-0.25, 0.25)
                cell.lambdaVecY = Ctax*2.0 * uniform(-0.25, 0.25)
            if cell.type == self.CAN3:
                cell.lambdaVecX = Ctax*2.0 * uniform(-0.25, 0.25)
                cell.lambdaVecY = Ctax*2.0 * uniform(-0.25, 0.25)
     
    def step(self, mcs):
        for cell in self.cellList:
            if cell.type == self.CAN1:
                cell.lambdaVecX = Ctax * uniform(-0.25, 0.25)
                cell.lambdaVecY = Ctax * uniform(-0.25, 0.25)
            if cell.type == self.CAN2:
                cell.lambdaVecX = Ctax*2.0 * uniform(-0.25, 0.25)
                cell.lambdaVecY = Ctax*2.0 * uniform(-0.25, 0.25)
            if cell.type == self.CAN3:
                cell.lambdaVecX = Ctax*2.0 * uniform(-0.25, 0.25)
                cell.lambdaVecY = Ctax*2.0 * uniform(-0.25, 0.25)


class HypoxiaSteppable(SteppableBasePy):
    def __init__(self,frequency=1):
        SteppableBasePy.__init__(self,frequency)
        self.runBeforeMCS=1

    def step(self,mcs):
        global SSox
        global pRes
        fieldOxy = self.field.Oxy
        hypC = SSox # Oxyc threshold for hypoxia
        for cell in self.cellList:
            if cell.type == self.CAN1:
                Oxyc = fieldOxy[cell.xCOM,cell.yCOM,cell.zCOM]
                mcs = self.simulator.getStep()
                if Oxyc >= hypC:
                    cell.dict["mcsO"] = mcs
                if hasattr(cell,"mcsO"):
                    mcsO = cell.dict["mcsO"]
                else: mcsO = 200
                if (Oxyc < hypC and mcs > (mcsO+20) and SSox > 0.0):
                    if random.random() > pRes:
                        cell.type = self.CAN2 # don't gain hypoxia resistance
                    else: cell.type = self.CAN3
                
  
class MatrixDegradationSteppable(SteppableBasePy):
    def __init__(self,frequency=1):
        SteppableBasePy.__init__(self,frequency)
        
    def step(self,mcs):
        #collagen degradation
        fieldMMP = self.field.MMP
        fieldTIM = self.field.TIM
        MMPt = 4.0 # MMPc-TIMc threshold for matrix degradation
        for cell in self.cellList:
            if (cell.type == self.LAM or cell.type == self.COL or cell.type == self.COLN):
                MMPc = fieldMMP[cell.xCOM,cell.yCOM,cell.zCOM]
                TIMc = fieldTIM[cell.xCOM,cell.yCOM,cell.zCOM]
                if TIMc > 0.01:
                    if (MMPc - TIMc) >= MMPt:
                        cell.type = self.COLL
                        cell.targetVolume = cell.volume
                        cell.lambdaVolume = 20.0
                        cell.targetSurface = cell.surface
                        cell.lambdaSurface = 80.0                        
                        if hasattr(cell,"numL"):
                            cell.dict["numL"] += 1
                        else: cell.dict["numL"] = 1
                        mcs = self.simulator.getStep()
                        cell.dict["mcsL"] = mcs                             
        for cell in self.cellList:
            if cell.type == self.COLL:
                numL = cell.dict["numL"]
                mcs = self.simulator.getStep()
                mcsL = cell.dict["mcsL"]
                if (mcs >= (mcsL+20) and (MMPc - TIMc) < MMPt):
                    cell.type = self.COLN
                    cell.targetVolume = cell.volume
                    cell.lambdaVolume = 20.0
                    cell.targetSurface = cell.surface
                    cell.lambdaSurface = 80.0
                elif (mcs >= (mcsL+60) or numL >= 6):
                    self.delete_cell(cell)
 

class SecretionSteppable(SecretionBasePy):
    def __init__(self,frequency=1):
        SecretionBasePy.__init__(self,frequency)
        
    def step(self,mcs):
        global Msec
        EGFsecretor = self.getFieldSecretor("EGF")
        MMPsecretor = self.getFieldSecretor("MMP")
        TIMsecretor = self.getFieldSecretor("TIM")
        Oxysecretor = self.getFieldSecretor("Oxy")
        for cell in self.cellList:
            if cell.type == self.CAN1:
                EGFsecretor.uptakeInsideCell(cell,0.5,0.5)
                MMPsecretor.secreteOutsideCellAtBoundaryOnContactWith(cell,Msec,[self.LAM,self.COL,self.COLN])
                TIMsecretor.secreteOutsideCellAtBoundaryOnContactWith(cell,Msec,[self.LAM,self.COL,self.COLN])
                Oxysecretor.uptakeInsideCell(cell,4.0,1.0)   
            if cell.type == self.CAN2:
                EGFsecretor.uptakeInsideCell(cell,0.5,0.5)
                MMPsecretor.secreteOutsideCellAtBoundaryOnContactWith(cell,Msec*2.0,[self.LAM,self.COL,self.COLN])
                TIMsecretor.secreteOutsideCellAtBoundaryOnContactWith(cell,Msec*2.0,[self.LAM,self.COL,self.COLN])
                Oxysecretor.uptakeInsideCell(cell,1.0,1.0)
            if cell.type == self.CAN3:
                EGFsecretor.uptakeInsideCell(cell,0.5,0.5)
                MMPsecretor.secreteOutsideCellAtBoundaryOnContactWith(cell,Msec*2.0,[self.LAM,self.COL,self.COLN])
                TIMsecretor.secreteOutsideCellAtBoundaryOnContactWith(cell,Msec*2.0,[self.LAM,self.COL,self.COLN])
                Oxysecretor.uptakeInsideCell(cell,1.0,1.0)                 
            if cell.type==self.COL:
                EGFsecretor.secreteInsideCellAtBoundaryOnContactWith(cell,4.0,[self.COLL])
            if cell.type==self.COLL:
                numL = cell.dict["numL"]
                EGFsecretor.secreteInsideCell(cell,2.0/numL)
                MMPsecretor.uptakeInsideCell(cell,1.0,1.0)
                TIMsecretor.uptakeInsideCell(cell,0.5,1.0)


class RecorderSteppable(SteppableBasePy):
    def __init__(self,frequency=1):
        SteppableBasePy.__init__(self,frequency)
    
    def finish(self):
        # write parameters to file
        global Rval; global Eadh; global Cadh; global Ctax; global Msec; global SSox; global pRes
        sim_path = Path(self.output_dir)
        path = ((sim_path.parent).parent).joinpath("!temp.dat")
        if not path.exists():
            with open(str(path), "w", newline="") as csvfile:
                pass
        with open(str(path), "r", newline="") as csvfile:
            paramreader = csv.reader(csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL)
            itr = -1 + sum(1 for row in paramreader)
        with open(str(path), "a", newline="") as csvfile:
            if itr <= 0:
                itr += 1
                paramwriter = csv.writer(csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL)
                paramwriter.writerow(["Iteration","ResistanceProbability","StateSwitchingThreshold","CellGrowthModifier","CellECMContactEnergy","CellCellContactEnergy","ChemotaxisModifier","MMPSecretionModifier","NormoxicCellCount","HypoxicCellCount","ResistorCellCount","UniformityScore"]) 
        nCan1 = 0; nCan2 = 0; nCan3 = 0
        # compute uniformity matrix
        u11 = 0.0; u12 = 0.0; u13 = 0.0; u14 = 0.0; u22 = 0.0; u23 = 0.0; u24 = 0.0; u33 = 0.0; u34 = 0.0; u44 = 0.0
        for cell in self.cellList:
            if cell.type == self.CAN1:
                s = cell.surface; nCan1 += 1
                neighborList = self.getCellNeighborDataList(cell)
                u11 += neighborList.commonSurfaceAreaWithCellTypes(cell_type_list=[self.CAN1])
                u12 += neighborList.commonSurfaceAreaWithCellTypes(cell_type_list=[self.CAN2])
                u13 += neighborList.commonSurfaceAreaWithCellTypes(cell_type_list=[self.CAN3])
                u14 += s - neighborList.commonSurfaceAreaWithCellTypes(cell_type_list=[self.CAN1,self.CAN2,self.CAN3])
            if cell.type == self.CAN2:
                s = cell.surface; nCan2 += 1
                neighborList = self.getCellNeighborDataList(cell)
                u22 += neighborList.commonSurfaceAreaWithCellTypes(cell_type_list=[self.CAN2])
                u23 += neighborList.commonSurfaceAreaWithCellTypes(cell_type_list=[self.CAN3])
                u24 += s - neighborList.commonSurfaceAreaWithCellTypes(cell_type_list=[self.CAN1,self.CAN2,self.CAN3])
            if cell.type == self.CAN3:
                s = cell.surface; nCan3 += 1
                neighborList = self.getCellNeighborDataList(cell)
                u33 += neighborList.commonSurfaceAreaWithCellTypes(cell_type_list=[self.CAN3])
                u34 += s - neighborList.commonSurfaceAreaWithCellTypes(cell_type_list=[self.CAN1,self.CAN2,self.CAN3])
        uMat = np.array([[u11,u12,u13,u14],[u12,u22,u23,u24],[u13,u23,u33,u34],[u14,u24,u34,u44]])/(nCan1+nCan2+nCan3)
        Unif = round(np.linalg.norm(uMat, ord=1),2)
        with open(str(path), "a", newline="") as csvfile:
            paramwriter = csv.writer(csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL)
            paramwriter.writerow([f"{itr:03d}",f"{pRes:.2f}",f"{SSox:.2f}",f"{Rval:.2f}",f"{Eadh:.1f}",f"{Cadh:.1f}",f"{Ctax:.1f}",f"{Msec:.2f}",nCan1,nCan2,nCan3,f"{Unif:.2f}"])
    
