from FSDataManager import FSError, FSDataLog, FSDataManager

from CODA import DiscretizationFactory, TimeIntegrationFactory
from CODA import StopNumIterations, StopRelativeReduction
from CODA import MonitorTabular, MonitorSelection
from CODA.CODAHelpers import BuildDiscretizationParameterTrees, BuildTimeIntegrationParameterTrees

from FSOverset.FSOverset import FSOverset, generateBlankingMask, extractActiveSubMesh, copySolution, generateDiscParasFromMesh, getClacInfo, getMeshKeys, display
from FSCGNSConverter.FSCGNSConverter import buildMeshOps

# mesh settings
localDirIn = 'INPUT/'
localDirOut = 'OUTPUT/STEADY/'

# solver settings
targetResidualReduction = 1.0e-8
maximumNumberOfIterations = 200

meshDict = {
    'background': {'meshFilename': localDirIn+'background.h5', 'meshProcessorWeight': 2.},
    'naca': {'meshFilename': localDirIn+'naca.h5', 'meshProcessorWeight': 1.},
}
offsetDict = {
    'naca': 0.3
}
blankingDict = {
    'background': ['naca']
}
displayDict = {
    'variables': ['Density'],
    'xlim': [-1.5, 2.5],
    'ylim': [-1.0, 1.0],
    'zplane': 0.0,
    'mpl': False
}

discSelectionParaDict = {
    "PDE" : "Euler",
    "spatial scheme" : "FV",
    "convection scheme" : "Roe upwinding",
    "order" : 2,
}
discParaDict = {
    "testing": {
        "willfully ignore excessive load imbalance among domains w.r.t. the number of faces": True,
    },
    "reference state": {
        "flow speed specification": {
            "type": "Mach number based",
            "Mach": 0.755,
        },
        "flow direction specification": {
            "type": "aerodynamic flow angles",
            "angle of attack": 0.0,
        },
    },
    "preprocessing" : {
        "maximum relative surface integral" : 1e-13,
        "non-local boundary treatments" : {
            "epsilon for in-element check" : 1e-13,
        },
    },
    "reconstruction": {
        "face gradient augmentation": "cell-to-face",
        "face gradient augmentation for the Jacobian matrix": "cell-to-face",
        "type": "limited linear",
        "gradient limiter": {
            "type": "full limiting",
        },
    },
    "boundary integral quantities": {
        "Coef_Area": 1.0,
        "Moment_Center": [0.25, 0.0, 0.0],
        "Coef_Length": 1.0,
    },
}
timeIntegrationParaDict = {
    "time integration method": "linearized implicit Euler",
    "time step": {
        "type": "local",
        "CFL": {
            "type": "SER ramp max refmax",
            "initial CFL number": 1.0,
            "maximum CFL number": 1000,
            "SER exponent": 0.5,
        },
    }
}

## ====================================
## Get clacs, fsmesh, etc.
## ====================================

# Get clacs
meshKey, meshColor, clac, globalClac, masterClac = getClacInfo(meshDict)
meshFilename = meshDict[meshKey]['meshFilename']
meshKeyActive, meshKeyOrig = getMeshKeys(meshKey, blankingDict)

# Get orig mesh
dm = FSDataManager(globalClac)
fsmeshOrig = dm.GetMesh(meshKeyOrig, clac, True)
meshOps = buildMeshOps(meshFilename, verbose=False)
fsmeshOrig.DoOps(meshOps) or FSError.PrintAndExit()

# Get active mesh
fsmeshActive = dm.GetMesh(meshKeyActive, clac, True)

## ====================================
## initialize FSOverset
## ====================================

blankingMaskDict = generateBlankingMask(
    clac=clac, fsmesh=fsmeshOrig,
    offsetDict=offsetDict,
    meshKey=meshKey,
    localDir=localDirOut,
    offsetFromBC='BCWall',
    check=False)

blankingObj = FSOverset(clac=clac, fsmesh=fsmeshOrig, meshKey=meshKey, blankingDict=blankingDict)
# need to run it once to initialize fsmeshActive and create the local numbering
blankingObj.computeBlanking(blankingMaskDict=blankingMaskDict)
extractActiveSubMesh(dm, meshKeyOrig, meshKeyActive)

## ====================================
## Set up CODA Dics & Settings
## ====================================

fsmeshActive.CreateLocalNumbering()

discParaDict = generateDiscParasFromMesh(fsmeshActive, discParaDict)

discSelectionParas, discParas = BuildDiscretizationParameterTrees(discSelectionParaDict, discParaDict)
disc = DiscretizationFactory.GetSingleton().Create(discSelectionParas, globalClac, fsmeshActive, discParas)

timeIntegrationParasAllLevels = BuildTimeIntegrationParameterTrees([timeIntegrationParaDict])
timeIntegrationParas = timeIntegrationParasAllLevels[0]
timeIntegration = TimeIntegrationFactory.GetSingleton().Create(disc, timeIntegrationParasAllLevels)

state = disc.CreateZeroFieldVector()
disc.InitializeFieldVector({'type': 'free stream'}, state)

residualNames = ['DensityResidual', 'MomentumResidual', 'EnergyStagnationDensityResidual']
monitorVariables = ['CFL'] + ['%sReduction'%res for res in residualNames]
reductionCallback = StopRelativeReduction(residualNames, targetResidualReduction)

iterationCallbacks = (reductionCallback | StopNumIterations(maximumNumberOfIterations)) + MonitorTabular(
    globalClac,
    disc.GetStateVariableNames(),
    timeIntegrationParas['state backup controller'],
    monitorVariables=monitorVariables,
    monitorSelection=MonitorSelection(monitorWallClockTime=False),
    monitorPeriod=1,
)

dataLog = FSDataLog(globalClac)

## ====================================
## Compute loop
## ====================================

# solution process
status = timeIntegration.Iterate([iterationCallbacks], state, dataLog)

# copy solution to active grids
state.ExportToFSMesh(disc.GetMeshInterface(), fsmeshActive, 'State') or FSError.PrintAndExit()

# copy solution to original grids
copySolution(dm, meshKeyOrig, meshKeyActive)

# export image with Cassiopee
if displayDict is not None:
    iterations = dataLog.GetDataArray('TimeIntegration', 'Iteration')
    niter = iterations.Size()
    it = iterations[niter-1]

    display(globalClac, fsmeshActive, meshKey, displayDict['variables'], dataset='State', it=it, displayDict=displayDict, localDir=localDirOut, saveTree=True)

# export convergence history
dataLog.ExportDataTECPLOT(localDirOut+'monitor.dat', 'l2-norms') or FSError.PrintAndExit()

# export flow solution
fsmeshActive.ExportMeshHDF5(HDF5Filename=localDirOut+'solution_%s_active.h5'%meshKey, FilePerProcess=False) or FSError.PrintAndExit()
fsmeshOrig.ExportMeshHDF5(HDF5Filename=localDirOut+'solution_%s_orig.h5'%meshKey, FilePerProcess=False) or FSError.PrintAndExit()