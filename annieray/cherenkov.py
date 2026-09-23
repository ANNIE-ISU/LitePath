
import numpy as np
import math

print("CherenkovAlgoTest.py is Running \n __________________________ ")

#Constants
c = 299792458 #Speed of light in m/s

#Rounding Function
def truncate(number, decimals = 0):
    factor = 10 ** decimals

    return math.trunc(number*factor)/factor

#Takes in angles to make coefficients for x,y,z components
#Expects a multi row array for alpha
def getComp(thetaC,alpha):
    lam = np.sin(thetaC)*np.cos(alpha)
    kap = np.sin(thetaC)*np.sin(alpha)
    eta = np.cos(thetaC)

    return lam,kap,eta

#Takes in a vector and generates two orthonormal vectors as outputs 
def getBasis(vector):
    mu1,mu2,mu3 = vector
    if mu3 !=0: #Prevent division by zero error
        b1One = 1
        b1Two = 1
        b1Three = -(mu1+mu2)/mu3

        b1 = np.array([b1One,b1Two,b1Three])
        b1 = b1/np.linalg.norm(b1) #Normalizing
    
        b2 = np.cross(vector,b1)
        b2 = b2/np.linalg.norm(b2) #Normalizing
    elif mu2 != 0: #Prevent division by zero error
        b1One = 1
        b1Two = -(mu3+mu1)/mu2
        b1Three = 1

        b1 = np.array([b1One,b1Two,b1Three])
        b1 = b1/np.linalg.norm(b1) #Normalizing
    
        b2 = np.cross(vector,b1)
        b2 = b2/np.linalg.norm(b2) #Normalizing
    elif mu1 != 0: #Prevent division by zero error
        b1One = -(mu3+mu2)/mu1
        b1Two = 1
        b1Three = 1

        b1 = np.array([b1One,b1Two,b1Three])
        b1 = b1/np.linalg.norm(b1) #Normalizing
    
        b2 = np.cross(vector,b1)
        b2 = b2/np.linalg.norm(b2) #Normalizing 
    else:
        print('Non physical vector input: ')
     
    return b1,b2

#Takes in the vector to generate two orthogonal vectors along with two angles
#Calls the previous two functions to generate the global photon vector
#Expects a multi row array for alpha
def getPhotonVec(vector,b1,b2,thetaC,alpha):
    lam, kap,eta = getComp(thetaC,alpha)
    normVector = np.asarray(vector) / np.linalg.norm(vector)
    vector = (lam[:, np.newaxis] * b1
              + kap[:, np.newaxis] * b2
              + eta * normVector)

    return vector


#User set default parameters
beta = 0.999999 # Speed of the muon as a fraction of the speed of light
n =1.34 #Refractive index of medium
trackPrec = 2 #control the step size of the track (currently cm steps along track)
LengthTravel = 4
defaultWavelength = 1234 #Just here because the original file has a singular specified wavelength
def generate_cherenkov_photons(
    muon_pos,
    muonDirec,
    photons_per_cm: int = 150,
    thetaC: float = 0.73,
    rng: np.random.Generator | None=None,
    wavelength: float = defaultWavelength,
    track_length: float = 4.0,
    ) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    if len(muon_pos) != 4:
        muonStart = tuple([muon_pos[0],muon_pos[1],muon_pos[2],0])
    else:
        muonStart = muon_pos

    n_steps = int(track_length * 10**trackPrec) + 1
    muonSpeed = beta*c

    b1,b2 = getBasis(muonDirec)

    muonArray = rng.uniform(0, track_length, size=n_steps * photons_per_cm)
    muonArray.sort()

    createTime = (muonArray / muonSpeed) * 10**9 + muonStart[3] * 10**9

    photonAlpha = rng.uniform(0, 2 * np.pi, size=n_steps * photons_per_cm)

    photStartPosArray = muonArray[:, np.newaxis] * np.array(muonDirec)
    photonDirec = getPhotonVec(muonDirec, b1, b2, thetaC, photonAlpha)

    origins = np.empty((n_steps * photons_per_cm, 3), dtype=np.float32)
    directions = np.empty((n_steps * photons_per_cm, 3), dtype=np.float32)

    origins[:,0] = 1000 * photStartPosArray[:,0] + muonStart[0]
    origins[:,1] = 1000 * photStartPosArray[:,1] + muonStart[1]
    origins[:,2] = 1000 * photStartPosArray[:,2] + muonStart[2]

    directions[:,0] = photonDirec[:,0]
    directions[:,1] = photonDirec[:,1]
    directions[:,2] = photonDirec[:,2]

    return origins, directions, createTime.astype(np.float32)


def generate_isotropic_photons(
    point: tuple[float, float, float, float],
    n_photons: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origins = np.full((n_photons, 3), point[:3], dtype=np.float32)
    theta = np.arccos(rng.uniform(-1, 1, size=n_photons))
    phi = rng.uniform(0, 2 * np.pi, size=n_photons)
    directions = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ]).astype(np.float32)
    create_times = np.full(n_photons, point[3] * 1e9, dtype=np.float32)
    return origins, directions, create_times


def generate_spot_photons(
    point: tuple[float, float, float, float],
    direction: tuple[float, float, float],
    n_photons: int,
    spread_deg: float = 2.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a directed beam of *n_photons* from a point source.

    Mirrors the interactive LAPPD viewer's "spot gun": photons originate at
    *point* and travel roughly along *direction*, scattered away from the
    axis by a Rayleigh distribution scaled by *spread_deg*.

    Parameters
    ----------
    point : (x, y, z, t0)
        Source position and time (t0 in *seconds*, converted to ns internally).
    direction : (dx, dy, dz)
        Nominal beam direction (need not be unit length).
    n_photons : int
        Number of photons to generate.
    spread_deg : float
        Rayleigh scale of the angular dispersion in degrees.
    rng : np.random.Generator or None

    Returns
    -------
    (origins, directions, create_times): (N, 3) float32, (N, 3) float32, (N,) float32.
    """
    if rng is None:
        rng = np.random.default_rng()

    dx0, dy0, dz0 = direction[:3]
    norm = math.sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0)
    if norm < 1e-12:
        dy0, norm = 0.0, 1.0
        dz0 = -1.0
    dx0, dy0, dz0 = dx0 / norm, dy0 / norm, dz0 / norm
    muon_dir = np.array([dx0, dy0, dz0], dtype=np.float64)

    if abs(dx0) < 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(muon_dir, ref)
    u /= np.linalg.norm(u)
    v = np.cross(muon_dir, u)

    spread_rad = math.radians(spread_deg)
    theta = rng.rayleigh(scale=spread_rad, size=n_photons)
    phi = rng.uniform(0.0, 2.0 * math.pi, size=n_photons)
    mu = (np.cos(theta)[:, None] * muon_dir
          + np.sin(theta)[:, None] * np.cos(phi)[:, None] * u
          + np.sin(theta)[:, None] * np.sin(phi)[:, None] * v)
    mu /= np.linalg.norm(mu, axis=1, keepdims=True)

    origins = np.full((n_photons, 3), point[:3], dtype=np.float32)
    directions = mu.astype(np.float32)
    t0_ns = point[3] * 1e9 if len(point) > 3 else 0.0
    create_times = np.full(n_photons, t0_ns, dtype=np.float32)
    return origins, directions, create_times
