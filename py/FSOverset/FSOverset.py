# Import modules
import numpy

import Converter.PyTree as C
import Connector.PyTree as X
import Converter.Internal as Internal
import Converter.Mpi as Cmpi
import Connector.Mpi as Xmpi
import Generator.PyTree as G
import Transform.PyTree as T
import Geom.PyTree as D
import RigidMotion.PyTree as R
import CPlot.PyTree as CPlot
import CPlot.Decorator as Decorator

from FSDataManager import (
    FSClac,
    FSIntArray, FSStringArray, FSFloatArray,
    FSUnstructVolumeCellTypes,
    FSDataSpecArray, FSDatasetInfo,
    FSError, FS_AT_CADGroupID,
    FSMeshEnums, FSDataName
)

import FSOversetBlanking # needed for extractActiveSubMesh and copySolution

from FSCGNSConverter.FSCGNSConverter import FSCGNSConverter

import math

__all__ = [
    'FSOverset', 'generateBlankingMask', 'extractPyTree',
    'generateDiscParasFromMesh', 'generateBCDictFromMesh',
    'extractActiveSubMesh', 'copySolution'
]

__DEG2RAD__ = math.pi/180.
__RAD2DEG__ = 180./math.pi

# ---------------------------------------------------------------------------- #
# Classes
# ---------------------------------------------------------------------------- #

class FSOverset:

    def __init__(self, clac, fsmesh, meshKey, blankingDict={}):
        self.clac = clac
        self.fsmesh = fsmesh
        self.meshKey = meshKey
        self.blankingDict = blankingDict
        self.pyTree = None

        if meshKey in blankingDict:
            if len(blankingDict[meshKey]) > 0:
                self.pyTree = extractPyTree(clac=clac, fsmesh=fsmesh)
        self.cellNName = 'cellN' # cellN located at the nodes 

        self.fsVolumeCellTypes = FSIntArray(0)
        for cellType in FSUnstructVolumeCellTypes:
            if self.fsmesh.HasCellType(cellType):
                self.fsVolumeCellTypes.Append(cellType)

    def computeBlanking(self, blankingMaskDict, blankingType='center_in'):
        validBlankingTypes = ['center_in', 'node_in', 'cell_intersect']
        if blankingType not in validBlankingTypes:
            raise ValueError("computeBlanking: invalid blankingType (%s). Possible values are %s"%(blankingType, validBlankingTypes))
        
        meshKeyTarget = self.meshKey
        blankingDict = self.blankingDict
        if self.pyTree is not None:
            C._deleteEmptyZones(self.pyTree)

            for maskKey in blankingDict[meshKeyTarget]:
                bodiesL = Internal.getZones(blankingMaskDict[maskKey])
                self.pyTree = X.blankCellsTri(self.pyTree, [bodiesL], [], blankingType=blankingType, cellNName=self.cellNName) 

            # Create an FSDM dataset for cellN obtained in Cassiopee
            cellNList = Internal.getNodesFromName(self.pyTree, self.cellNName)
            cellN = [n_cellN[1] for n_cellN in cellNList]
            np_cellN = numpy.concatenate(cellN)
            Internal._rmNodesFromType(self.pyTree, 'FlowSolution_t')

            if not self.fsmesh.HasUnstructDataset(self.cellNName):
                fsVarNames = FSStringArray(1)
                fsVarNames[0] = self.cellNName
                fsVarSpecs = FSDataSpecArray(1)
                self.fsmesh.InitUnstructDataset(
                    self.cellNName,
                    FSDatasetInfo(fsVarNames, fsVarSpecs, self.fsVolumeCellTypes)
                )

            fs_var = self.fsmesh.GetUnstructDataset(self.cellNName).GetValues()
            numpy.copyto(
                numpy.array(fs_var.Buffer(), copy=False),
                np_cellN[:,None],
                casting='same_kind'
            )

        return None

# ---------------------------------------------------------------------------- #
# FSOverset Functions
# ---------------------------------------------------------------------------- #

# offsetDict : mandatory (can be zero) to specify if a BC defines a blanking mask or not.
def generateBlankingMask(clac, fsmesh, meshKey, offsetDict, offsetFromBC='BCOverset', dim=3, localDir='./', check=False):
    """Generate a blanking mask from a specified BC"""
    validBCNames = ['BCOverset', 'BCWall']
    if offsetFromBC not in validBCNames:
        raise ValueError("generateBlankingMask: invalid offsetFromBC (%s). Possible values are %s"%(offsetFromBC, validBCNames))

    tb = None

    # Conversion of the curvilinear mesh of the body ('standard' conversion)
    if meshKey in offsetDict:
        bcDict = generateBCDictFromMesh(fsmesh)
        if any(bcDict[bc].startswith(offsetFromBC) for bc in bcDict):
            if fsmesh.HasUnstructDataset('UndeformedCoordinates'): coordsName = 'UndeformedCoordinates'
            else: coordsName = 'Coordinates'
            convObj = FSCGNSConverter(clac=clac, fsmesh=fsmesh, bcDict=bcDict, coordsName=coordsName, datasets=[])
            convObj.convert2CGNS()

            z_body = Internal.getZones(convObj.pyTree)[0]
            z_body[0] += f'.{Cmpi.rank:d}'
            C._deleteFlowSolutions__(z_body)

            # Extract the BC
            wall = C.extractBCOfType(z_body, offsetFromBC)
            del z_body
            if wall != []:        
                elts = Internal.getNodesFromType(wall, 'Elements_t')
                for elt in elts:
                    if elt[0].startswith('GridElements'):
                        Internal._rmNode(wall, elt)
                tb = C.convertArray2Tetra(wall)
                del wall

                tb = T.join(tb)
                Cmpi._setProc(tb, Cmpi.rank)
                param = Internal.getNodeFromName1(tb, '.Solver#Param')
                Internal.newDataArray('meshKey', parent=param, value=meshKey)

    # Create bodies per meshKey
    tb = Cmpi.allgatherZones(tb)
    if Cmpi.master and check: C.convertPyTree2File(tb, localDir+'wall.plt')
    bodies = {}
    for zone in Internal.getZones(tb):
        meshKeyNode = Internal.getNodeFromName(zone, 'meshKey')
        meshKeyLocal = Internal.getValue(meshKeyNode)
        if meshKeyLocal not in bodies.keys():
            bodies[meshKeyLocal] = zone
        else:
           bodies[meshKeyLocal] = T.join(bodies[meshKeyLocal], zone)
           bodies[meshKeyLocal] = G.close(bodies[meshKeyLocal])

    # Create offset bodies per meshKey
    blankingMaskDict = bodies.copy()

    sign_offset = 1. if offsetFromBC == 'BCWall' else -1.
    for meshKeyLocal in blankingMaskDict:
        offsetdist = offsetDict[meshKeyLocal]
        if offsetdist > 0.:
            BB = G.bbox(blankingMaskDict[meshKeyLocal])
            xmin = BB[0]; ymin = BB[1]; zmin = BB[2]
            xmax = BB[3]; ymax = BB[4]; zmax = BB[5]
            dmax = max((xmax-xmin), (ymax-ymin), (zmax-zmin))
            ppul = 50./dmax
            if Cmpi.master: print('generateBlankingMask: generating offset (meshKey=%s) with ppul=%f and dmax=%f'%(meshKeyLocal, ppul, dmax))
            blankingMaskDict[meshKeyLocal] = D.offsetSurface(blankingMaskDict[meshKeyLocal], offset=sign_offset*offsetDict[meshKeyLocal], pointsPerUnitLength=ppul, algo=0, dim=dim)[0]
            if Cmpi.master and check: C.convertPyTree2File(blankingMaskDict[meshKeyLocal], localDir+'offset_%s_%s.plt'%(offsetFromBC, meshKeyLocal))
            blankingMaskDict[meshKeyLocal] = C.convertArray2Tetra(blankingMaskDict[meshKeyLocal])
        else:
            Internal._rmNodesByName1(bodies[meshKeyLocal], '.Solver#Param')
            blankingMaskDict[meshKeyLocal] = C.convertArray2Tetra(bodies[meshKeyLocal])

        blankingMaskDict[meshKeyLocal] = G.close(blankingMaskDict[meshKeyLocal])

    return blankingMaskDict

def extractPyTree(clac, fsmesh):
    """Extract a pyTree mesh from a fsmesh"""
    # Conversion of the blanked mesh ('light' conversion -> only the volume element types)
    convObj = FSCGNSConverter(clac=clac, fsmesh=fsmesh)
    convObj.convert2CGNS(forOverset=True)
    z = Internal.getZones(convObj.pyTree)[0]
    z = C.breakConnectivity(z)
    t = C.newPyTree(['Base', z])
    return t

def display(clac, fsmesh, meshKey, variables, dataset='State', it=0, displayDict={}, localDir='./', saveTree=False):
    """Display flow solution using Cassiopee"""
    # get display information
    colormap = displayDict.get('colormap', 24) # default: jet
    isoEdges = displayDict.get('isoEdges', 0.) # line width of isolines
    isoScales = displayDict.get('isoScales', {}) # dict of {varname: [varname, niso, min, max]}
    ppw = displayDict.get('ppw', 1000) # pixels per height
    mpl = displayDict.get('mpl', False) # pixels per height

    # automatically set camera information
    xlim = displayDict['xlim']
    ylim = displayDict['ylim']
    zplane = displayDict['zplane']
    posCam, posEye, dirCam, viewAngle, exportResolution = Decorator.getInfo2DMode(xlim, ylim, zplane, ppw)

    # conversion
    bcDict = generateBCDictFromMesh(fsmesh)
    convObj = FSCGNSConverter(clac=clac, fsmesh=fsmesh, bcDict=bcDict, datasets=['State'])
    convObj.convert2CGNS(forFFDX=True) # get NGon array
    zone = Internal.getZones(convObj.pyTree)[0]
    Cmpi._setProc(zone, Cmpi.rank)
    zone[0] = '%s_%d'%(meshKey, Cmpi.rank)
    Xmpi._connectMatchNGon(zone)

    listOfMeshKeys = set(Cmpi.allgather(meshKey))
    listOfMeshKeys = sorted(listOfMeshKeys)
    listOfZones = []
    for meshKeyLocal in listOfMeshKeys:
        listOfZones.extend([
            meshKeyLocal,
            zone if meshKey == meshKeyLocal else []
        ])
    
    t = C.newPyTree(listOfZones)

    # get correct flow container
    Internal.__FlowSolutionCenters__ = 'FlowSolution#%s'%dataset
    varList = C.getVarNames(t, excludeXYZ=True, loc='centers')[0]
    t = Cmpi.center2Node(t, var=varList)
    Internal._rmNodesByName(t, Internal.__FlowSolutionCenters__)
    Internal.__FlowSolutionCenters__ = 'FlowSolution#Centers'

    # add cellN
    X._applyBCOverlaps(t, depth=2, loc='nodes', val=0, cellNName='cellN')

    # save solution tree
    if saveTree: Cmpi.convertPyTree2File(t, localDir+'solution_iter%04d.cgns'%it)

    # force (x,y) plane
    T._rotate(t, (0,0,0), (1,0,0), -90.) # from (x,z) to (x,y)

    # display
    for v in variables:
        filename = localDir+'%s_it%04d.png'%(v, it)
        export = CPlot.decorator if mpl else filename

        if v not in isoScales:
            vmin = Cmpi.getMinValue(t, v)
            vmax = Cmpi.getMaxValue(t, v)
            isoScales[v] = [v, 25, vmin, vmax] # default CPlot values

        CPlot.display(t, mode='Scalar', scalarField=v,
            dim=2, export=export, isoScales=isoScales[v], isoEdges=isoEdges,
            offscreen=7, bgColor=0, colormap=colormap,
            viewAngle=viewAngle,
            posCam=posCam, posEye=posEye, dirCam=dirCam,
            exportResolution=exportResolution)
        
        if mpl and Cmpi.master:
            fig, ax = Decorator.createSubPlot(box=True, figsize=(7,6), dpi=100, xlim=xlim, ylim=ylim)
            cbar = Decorator.createColorBar(fig, ax, title=v, discrete=True, nticks=5, labelFormat='%.2f', size='3%')
            Decorator.savefig(filename, pad=0.1, dpi=200)
    
    return None

# ---------------------------------------------------------------------------- #
# FSOversetBlanking Functions
# ---------------------------------------------------------------------------- #

def extractActiveSubMesh(fsdatamanager, meshKeyOrig, meshKeyActive):
    """Extract the active mesh (blanked mesh) from the original mesh (non-blanked mesh) based on cellN dataset"""
    # cellN = 0: blanked cells
    # cellN = 1: non-blanked cells
    dataManagerOps = (('BlankMesh', {'MeshKeyOrig'     : meshKeyOrig,
                                     'MeshKeyActive'   : meshKeyActive,
                                     'DatasetNameCellNature': 'cellN',
                                     'QuantityNameCellNature': 'cellN',
                                     'CellNatureActive': 1,
                                     'AddFacesMarker': 56,
                                     'AddFacesBC': 'BCOverset',
                                    }),)
    fsdatamanager.DoOps(dataManagerOps) or FSError.PrintAndExit()
    return None

def copySolution(fsdatamanager, meshKeyOrig, meshKeyActive):
    """Copy the flow solution from the active mesh (blanked mesh) to the original mesh (non-blanked mesh)"""
    # cellN = 0: blanked cells
    # cellN = 1: non-blanked cells
    dataManagerOps = (('CopyDataOfBlankedMesh', {'MeshKeyOrig'    : meshKeyOrig,
                                                 'MeshKeyActive'  : meshKeyActive,
                                                 'AttributeNameDataAvailable': 'DataPresent',
                                                 'InitBlankedCellData': 2,
                                          }),)
    fsdatamanager.DoOps(dataManagerOps) or FSError.PrintAndExit()
    return None

# ---------------------------------------------------------------------------- #
# Helper Functions
# ---------------------------------------------------------------------------- #

def getBoundaryTreatmentsFromMesh__(fsmesh):
    markers = FSIntArray()
    fsmesh.GatherCellAttributeValues(FS_AT_CADGroupID, markers) or FSError.PrintAndExit()

    treatments = {}
    for marker in markers: # get BCType and associated boundary markers
        treatmentType = str(fsmesh.GetCellAttributeValueName(FS_AT_CADGroupID, marker))
        treatmentType = ''.join([i for i in treatmentType if not i.isdigit()]) # remove integer(s) from name
        if treatmentType in treatments: treatments[treatmentType].append(marker)
        else: treatments[treatmentType] = [marker]
    
    return treatments

def generateDiscParasFromMesh(fsmesh, discParaDict=None):
    """Extract the boundary marker names from the fsmesh and generates the matching discParas"""
    treatments = getBoundaryTreatmentsFromMesh__(fsmesh)
    paraDict = discParaDict.copy() if discParaDict is not None else {}
    paraDict['boundary treatments'] = [
        {'treatment type': key, 'boundary markers': value} for key, value in treatments.items()
    ]
    return paraDict

def generateBCDictFromMesh(fsmesh):
    """Extract the boundary marker names from the fsmesh and generates the matching bcdict"""
    treatments = getBoundaryTreatmentsFromMesh__(fsmesh)
    bcDict = {}
    for treatmentType, markers in treatments.items():
        for marker in markers: bcDict[marker] = treatmentType
    return bcDict

def getWallBoundaryMarkers(fsmesh):
    """Get all wall boundary markers from the fsmesh"""
    treatments = getBoundaryTreatmentsFromMesh__(fsmesh)
    wallMarkers = []
    for key, value in treatments.items():
        if 'Wall' in key: wallMarkers.extend(value)
    return wallMarkers

def getClacInfo(meshDict):
    """Distribute the meshes across all available processors"""
    globalClac = FSClac()
    globalProcID = globalClac.GetProcID()
    nGlobalProcs = globalClac.GetNProcs()

    nMeshes = len(meshDict)

    # first security check
    if nGlobalProcs < nMeshes:
        raise ValueError('FSOverset: the number of MPI processes must be greater or equal to the number of meshes (nMeshes = %d)'%nMeshes)

    # compute total mesh weight and sort meshKeys per weight
    for meshKeyLocal in meshDict: 
        if 'meshProcessorWeight' not in meshDict[meshKeyLocal]: 
            meshDict[meshKeyLocal]['meshProcessorWeight'] = 1.0 # default value
    weightTotal = sum(meshDict[meshKeyLocal]['meshProcessorWeight'] for meshKeyLocal in meshDict)
    sortedMeshKeys = sorted(meshDict.keys(), key=lambda x: meshDict[x]['meshProcessorWeight'], reverse=True)

    # initialize balancingDict
    balancingDict = {meshKeyLocal: 0 for meshKeyLocal in meshDict}
    for meshKeyLocal in meshDict:
        nProcs = math.floor(nGlobalProcs*meshDict[meshKeyLocal]['meshProcessorWeight']/weightTotal)
        balancingDict[meshKeyLocal] = max(1, nProcs) # at least one proc per mesh

    # correct balancingDict based on nGlobalProcs
    diffBalancing = nGlobalProcs - sum(balancingDict.values())
    # too many procs: remove values from lightest to heaviest meshKey
    if diffBalancing < 0:
        pos = -1
        while diffBalancing < 0:
            meshKeyLocal = sortedMeshKeys[pos]
            if balancingDict[meshKeyLocal] > 1:
                balancingDict[meshKeyLocal] -= 1
                diffBalancing += 1
            pos -= 1
    # too few procs: add values from heaviest to lightest meshKey
    elif diffBalancing > 0:
        pos = 0
        while diffBalancing > 0:
            meshKeyLocal = sortedMeshKeys[pos]
            balancingDict[meshKeyLocal] += 1
            diffBalancing -= 1
            pos += 1

    # get final meshKey per proc
    threshold = 0
    meshKey = None
    for meshKeyLocal in sortedMeshKeys:
        threshold += balancingDict[meshKeyLocal]
        if globalProcID < threshold: 
            meshKey = meshKeyLocal
            break
    
    # get local and master clacs
    clac = FSClac()
    colors = {meshKeyLocal: pos for pos, meshKeyLocal in enumerate(sortedMeshKeys)}
    meshColor = colors[meshKey] # meshColor must be an integer
    globalClac.DivideIntoGroups(meshColor, clac)

    master = clac.GetProcID() == 0
    masterClac = FSClac()
    globalClac.DivideIntoGroups(master, masterClac)

    return meshKey, meshColor, clac, globalClac, masterClac

def getMeshKeys(meshKey, blankingDict):
    """Get orig and active mesh keys"""
    meshKeyOrig = 'none'
    meshKeyActive = 'none'
    if meshKey in blankingDict:
        if blankingDict[meshKey]: # not None or []
            meshKeyOrig = meshKey + '_orig'
            meshKeyActive = meshKey + '_active'

    return meshKeyOrig, meshKeyActive

def initGridVelocity(fsmesh, meshKey, motionDict):
    """Initialize the GridVelocity dataset"""
    if meshKey in motionDict:
        if not fsmesh.HasUnstructDataset('GridVelocity'):
            nNodes = fsmesh.GetNCells(FSMeshEnums.CT_Node)
            gridVelNames = FSStringArray(3)
            gridVelNames[0] = FSDataName.GridVelocity().X()
            gridVelNames[1] = FSDataName.GridVelocity().Y()
            gridVelNames[2] = FSDataName.GridVelocity().Z()
            gridVelSpecs = FSDataSpecArray(3)
            gridVelSpecs[0].Velocity()
            gridVelSpecs[1].Velocity()
            gridVelSpecs[2].Velocity()
            gridVels = FSFloatArray(nNodes, 3)
            gridVels.Fill(0.0)
            fsmesh.InitUnstructDataset('GridVelocity', FSDatasetInfo(gridVelNames, gridVelSpecs, FSMeshEnums.CT_Node), gridVels)

    return None

def copyGrid2GridInit(fsmesh, meshKey, motionDict, blankingMaskDict=None):
    """Initialize the UndeformedCoordinates dataset from the original Coordinates dataset"""

    if meshKey in motionDict:
        if not fsmesh.HasUnstructDataset('UndeformedCoordinates'):
            coordsDataset = fsmesh.GetUnstructDataset('Coordinates')
            coords = coordsDataset.GetValues()
            undeformedNames = FSStringArray(3)
            undeformedNames[0] = FSDataName.Coordinates().X()
            undeformedNames[1] = FSDataName.Coordinates().Y()
            undeformedNames[2] = FSDataName.Coordinates().Z()
            undeformedSpecs = FSDataSpecArray(3)
            undeformedSpecs[0].Length()
            undeformedSpecs[1].Length()
            undeformedSpecs[2].Length()
            undeformed = coords
            fsmesh.InitUnstructDataset('UndeformedCoordinates', FSDatasetInfo(undeformedNames, undeformedSpecs, FSMeshEnums.CT_Node), undeformed)
    
    if blankingMaskDict is not None:
        for maskKeyLocal in blankingMaskDict:
            if maskKeyLocal in motionDict:
                z = blankingMaskDict[maskKeyLocal]
                R._copyGrid2GridInit(z, mode=1)

    return None

def copyGridInit2Grid(fsmesh):
    """Copy UndeformedCoordinates to Coordinates"""
    if fsmesh.HasUnstructDataset('UndeformedCoordinates'):
        refCoords = fsmesh.GetUnstructDataset('UndeformedCoordinates').GetValues()
        gridCoords = fsmesh.GetUnstructDataset('Coordinates').GetValues()
        numpy.copyto(
                numpy.array(gridCoords.Buffer(), copy=False),
                numpy.array(refCoords.Buffer(), copy=False),
                casting='same_kind'
        )

    return None

def evalPositionFSMesh__(fsmesh, meshKey, time, motionDict):
    tx, ty, tz = motionDict[meshKey]['transl_speed']
    cx, cy, cz = motionDict[meshKey]['axis_pnt']
    kx, ky, kz = motionDict[meshKey]['axis_vct']
    omega = motionDict[meshKey]['angular_frq']

    if 'ampl_angle' in motionDict[meshKey]: # oscillation
        alphaMean = motionDict[meshKey]['mean_angle']
        alphaAmpl = motionDict[meshKey]['ampl_angle']
        alpha = alphaMean + alphaAmpl * math.sin(omega * time)
    else: # rotation
        alpha = omega * time * __RAD2DEG__
    
    cosalpha = math.cos(alpha * __DEG2RAD__)
    sinalpha = math.sin(alpha * __DEG2RAD__)

    copyGridInit2Grid(fsmesh)
    gridCoords = fsmesh.GetUnstructDataset('Coordinates').GetValues()

    np_gridCoords = numpy.array(gridCoords.Buffer(), copy=False)

    # center vector
    c = numpy.array([cx, cy, cz])

    # rotation axis vector
    k = numpy.array([kx, ky, kz])

    # translation speed vector
    t = numpy.array([tx, ty, tz])

    # position vector
    cm = np_gridCoords - c

    # k x CM
    kcm_cross = numpy.cross(k, cm)

    # k . CM
    # Element-by-element multiplication (with broadcasting) + sum of the components along axis 1
    kcm_dot = numpy.sum(k*cm, axis=1, keepdims=True) # keepdims=True to return (nnodes, 1) array

    # rotation (Rodrigues' rotation formula) + translation
    np_gridCoords[:] = (c + cosalpha*cm + (1 - cosalpha)*kcm_dot*k + sinalpha*kcm_cross) + time*t

    return None

def evalPositionMask__(mask, meshKey, time, motionDict):
    tx, ty, tz = motionDict[meshKey]['transl_speed']
    cx, cy, cz = motionDict[meshKey]['axis_pnt']
    kx, ky, kz = motionDict[meshKey]['axis_vct']
    omega = motionDict[meshKey]['angular_frq']

    if 'ampl_angle' in motionDict[meshKey]: # oscillation
        alphaMean = motionDict[meshKey]['mean_angle']
        alphaAmpl = motionDict[meshKey]['ampl_angle']
        alpha = alphaMean + alphaAmpl * math.sin(omega * time)
    else: # rotation
        alpha = omega * time * __RAD2DEG__

    R._copyGridInit2Grid(mask)
    T._rotate(mask, (cx,cy,cz), (kx,ky,kz), alpha, vectors=[])
    T._translate(mask, (tx*time, ty*time, tz*time))

    return None

def evalPosition(fsmesh, meshKey, time, motionDict, blankingMaskDict=None):
    """Move the fsmesh and all masks based on motionDict and time"""
    if meshKey in motionDict:
        evalPositionFSMesh__(fsmesh, meshKey, time, motionDict)
    
    if blankingMaskDict is not None:
        for maskKeyLocal in blankingMaskDict:
            if maskKeyLocal in motionDict:
                evalPositionMask__(blankingMaskDict[maskKeyLocal], maskKeyLocal, time, motionDict)

    return None

def evalGridSpeedFSMesh__(fsmesh, meshKey, time, motionDict):
    tx, ty, tz = motionDict[meshKey]['transl_speed']
    cx, cy, cz = motionDict[meshKey]['axis_pnt']
    kx, ky, kz = motionDict[meshKey]['axis_vct']
    omega = motionDict[meshKey]['angular_frq']

    if 'ampl_angle' in motionDict[meshKey]: # oscillation
        alphaAmpl = motionDict[meshKey]['ampl_angle']
        alphaDot = omega * alphaAmpl * math.cos(omega * time) # derivative of alpha w.r.t to time
        alphaDot *= __DEG2RAD__ # radians per sec.
    else: # rotation
        alphaDot = omega

    gridCoords = fsmesh.GetUnstructDataset('Coordinates').GetValues() # grid has already been moved
    gridVels = fsmesh.GetUnstructDataset('GridVelocity').GetValues()

    np_gridCoords = numpy.array(gridCoords.Buffer(), copy=False)
    np_gridVels = numpy.array(gridVels.Buffer(), copy=False)

    # center vector
    c = numpy.array([cx, cy, cz])

    # rotation axis vector
    k = numpy.array([kx, ky, kz])

    # translation speed vector
    t = numpy.array([tx, ty, tz])

    # position vector
    cm = np_gridCoords - c

    # k x CM
    kcm_cross = numpy.cross(k, cm)

    # new grid speed
    np_gridVels[:] = t + alphaDot*kcm_cross
    
    return None

def evalGridSpeed(fsmesh, meshKey, time, motionDict):
    """Update the fsmesh grid velocities based on motionDict and time"""
    if meshKey in motionDict:
        evalGridSpeedFSMesh__(fsmesh, meshKey, time, motionDict)

    return None