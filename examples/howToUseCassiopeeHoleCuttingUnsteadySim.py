from FSDataManager import FSClac, FSError, FSDataLog, FSDataManager

from CODA import DiscretizationFactory, TimeIntegrationFactory
from CODA import StopNumIterations, StopRelativeReduction
from CODA import MonitorTabular, MonitorSelection, MonitorIntegrals
from CODA.CODAHelpers import BuildDiscretizationParameterTrees, BuildTimeIntegrationParameterTrees

from FSOverset.FSOverset import FSOverset, generateBlankingMask, extractActiveSubMesh, copySolution, generateDiscParasFromMesh
from FSOverset.FSOverset import initGridVelocity, copyGrid2GridInit, evalPosition, evalGridSpeed, getWallBoundaryMarkers, display, getClacInfo, getMeshKeys
from FSCGNSConverter.FSCGNSConverter import buildMeshOps

import math
import sys

motionType = sys.argv[1].lower()
if motionType not in ['rotation', 'oscillation']:
    raise ValueError('FSOverset: incorrect motionType (%s). Possible values are "rotation" and "oscillation".'%motionType)

# mesh settings
localDirIn = 'INPUT/'
localDirOut = 'OUTPUT/%s/'%motionType.upper()

# flow settings
if motionType == 'oscillation':
    Mach, gamma, chord = 0.755, 1.4, 1.0
    k = 0.0814 # reduced frequency
    tx, ty, tz = (0., 0., 0.) # translation vector
    cx, cy, cz = (0.25, 0., 0.) # center of rotation
    kx, ky, kz = (0., 1., 0.) # axis vector
    alphaMean, alphaAmpl = -0.016, -2.51  # degrees, mean angle and max angle amplitude
else: # rotation
    Mach, gamma, chord = 0.2, 1.4, 1.0
    k = 0.0814 # reduced frequency
    tx, ty, tz = (0., 0., 0.) # translation vector
    cx, cy, cz = (0.25, 0., 0.) # center of rotation
    kx, ky, kz = (0., 1., 0.) # axis vector

Uinf = Mach * math.sqrt(gamma) # freestream speed
omega = 2 * k * Uinf / chord # angular frequency
period = 2 * math.pi / omega # oscillatory period

# solver settings
TSOP = 200 # time steps per period
nperiod = 1 # number of periods
targetResidualReduction = 1.0e-8
maximumNumberOfIterations = 200

time_step = period / TSOP
niter = nperiod * TSOP
time = 0. # initalization

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

# set up motionDict
if motionType == 'oscillation':
    motionDict = {
        'naca': {
            'transl_speed': [tx, ty, tz],
            'axis_pnt': [cx, cy, cz],
            'axis_vct': [kx, ky, kz],
            'angular_frq': omega,
            'mean_angle': alphaMean,
            'ampl_angle': alphaAmpl
        }
    }
else: # rotation
    motionDict = {
        'naca': {
            'transl_speed': [tx, ty, tz],
            'axis_pnt': [cx, cy, cz],
            'axis_vct': [kx, ky, kz],
            'angular_frq': omega
        }
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
            "Mach": Mach,
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

innerTimeIntegrationParaDict = {
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
outerTimeIntegrationParaDict = {
    "time integration method": "[(E)S]DIRK",
    "time information logging name": "PhysicalTime",
    "Butcher tableau": {"identifier": "Alexander22"},
    "time step": {
        "type": "constant",
        "size": time_step,
    },
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
## Set up CODA Dicts & Settings
## ====================================

fsmeshActive.CreateLocalNumbering()

discParaDict = generateDiscParasFromMesh(fsmeshActive, discParaDict)

discSelectionParas, discParas = BuildDiscretizationParameterTrees(discSelectionParaDict, discParaDict)
discFactory = DiscretizationFactory.GetSingleton()
disc = discFactory.Create(discSelectionParas, globalClac, fsmeshActive, discParas)

timeIntegrationParasAllLevels = BuildTimeIntegrationParameterTrees([outerTimeIntegrationParaDict, innerTimeIntegrationParaDict])
timeIntegrationParasOuter = timeIntegrationParasAllLevels[0]
timeIntegrationParasInner = timeIntegrationParasAllLevels[1]
timeFactory = TimeIntegrationFactory.GetSingleton()
timeIntegration = timeFactory.Create(disc, timeIntegrationParasAllLevels)

residualNames = ['DensityResidual', 'MomentumResidual', 'EnergyStagnationDensityResidual']
monitorVariables = ['CFL'] + ['%sReduction'%res for res in residualNames]
reductionCallback = StopRelativeReduction(residualNames, targetResidualReduction)

wallMarkers = getWallBoundaryMarkers(fsmesh=fsmeshActive)
monitorIntegralsCallbacks = MonitorIntegrals(
    'boundary',
    [
        {
            'varnames': ['CoefDrag', 'CoefLift', 'CoefMomentY'],
            'markers': wallMarkers,
            'levels': [0], # outerIntegration
            'key': 'AerodynamicCoefficients',
        },
    ],
    globalClac=globalClac,
    meshLocalClac=clac,
    meshClacMaster=masterClac,
    monitorPeriod=1,
    timeLoggingName='PhysicalTime',
    multiMesh=True
)

iterationCallbacksInner = (reductionCallback | StopNumIterations(maximumNumberOfIterations)) + MonitorTabular(
    globalClac,
    disc.GetStateVariableNames(),
    timeIntegrationParasInner['state backup controller'],
    monitorVariables=monitorVariables,
    monitorSelection=MonitorSelection(monitorWallClockTime=True),
    monitorPeriod=1,
    includeTimeColumn=True,
)

iterationCallbacksOuter = StopNumIterations(1) + monitorIntegralsCallbacks

dataLog = FSDataLog(globalClac)

# init. undeformed grid coordinates and grid velocities
initGridVelocity(fsmeshOrig, meshKey, motionDict)
copyGrid2GridInit(fsmeshOrig, meshKey, motionDict, blankingMaskDict)

## ====================================
## Compute loop
## ====================================

for i in range(niter):
    time += time_step

    # update grid coordinates and grid velocities
    evalPosition(fsmeshOrig, meshKey, time, motionDict, blankingMaskDict)
    evalGridSpeed(fsmeshOrig, meshKey, time, motionDict)

    # update blanking    
    blankingObj.computeBlanking(blankingMaskDict=blankingMaskDict)
    extractActiveSubMesh(dm, meshKeyOrig, meshKeyActive)

    # update CODA settings
    fsmeshActive.HasLocalNumbering() or fsmeshActive.CreateLocalNumbering()
    disc = discFactory.Create(discSelectionParas, globalClac, fsmeshActive, discParas)
    timeIntegration = timeFactory.Create(disc, timeIntegrationParasAllLevels)
    if i == 0:
        state = disc.CreateZeroFieldVector()
        disc.InitializeFieldVector({'type': 'free stream'}, state)
    else:
        state = disc.CreateZeroFieldVector()
        state.ImportFromFSMesh(disc.GetMeshInterface(), fsmeshActive, 'State') or FSError.PrintAndExit()
    
    # solution process
    status = timeIntegration.Iterate([iterationCallbacksOuter, iterationCallbacksInner], state, dataLog)

    # copy solution to active grids
    state.ExportToFSMesh(disc.GetMeshInterface(), fsmeshActive, 'State') or FSError.PrintAndExit()

    # copy solution to original grids
    copySolution(dm, meshKeyOrig, meshKeyActive)

    # export image with Cassiopee
    if displayDict is not None:
        display(globalClac, fsmeshActive, meshKey, displayDict['variables'], dataset='State', it=i+1, displayDict=displayDict, localDir=localDirOut, saveTree=True)

    # export flow solution
    fsmeshActive.ExportMeshHDF5(HDF5Filename=localDirOut+'solution_%s_active_iter%04d.h5'%(meshKey, i+1), FilePerProcess=False) or FSError.PrintAndExit()
    fsmeshOrig.ExportMeshHDF5(HDF5Filename=localDirOut+'solution_%s_orig_iter%04d.h5'%(meshKey, i+1), FilePerProcess=False) or FSError.PrintAndExit()

# export data logs
dataLog.ExportDataXML(localDirOut+'datalog.xml') or FSError.PrintAndExit()

# convergence history output
dataLog.ExportDataTECPLOT(localDirOut+'datalog.dat') or FSError.PrintAndExit()