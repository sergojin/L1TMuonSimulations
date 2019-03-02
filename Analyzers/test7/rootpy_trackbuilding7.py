import numpy as np
np.random.seed(2023)

import os, sys
from six.moves import range, zip, map, filter

from rootpy.plotting import Hist, Hist2D, Graph, Efficiency
from rootpy.tree import Tree, TreeChain, TreeModel, FloatCol, IntCol, ShortCol
from rootpy.io import root_open
#from rootpy.memory.keepalive import keepalive
from ROOT import gROOT, TH1
gROOT.SetBatch(True)
TH1.AddDirectory(False)


# ______________________________________________________________________________
# Analyzer

# Enums
kDT, kCSC, kRPC, kGEM, kME0 = 0, 1, 2, 3, 4

# Globals
eta_bins = (1.2, 1.55, 1.7, 1.8, 1.98, 2.15, 2.4)
eta_bins = eta_bins[::-1]
pt_bins = (-0.5, -0.365, -0.26, -0.155, -0.07, 0.07, 0.155, 0.26, 0.365, 0.5)
nlayers = 12  # 5 (CSC) + 4 (RPC) + 3 (GEM)
#superstrip_size = 32

assert(len(eta_bins) == 6+1)
assert(len(pt_bins) == 9+1)


# Functions
def delta_phi(lhs, rhs):  # in radians
  rad = lhs - rhs
  while rad <  -np.pi:  rad += np.pi*2
  while rad >= +np.pi:  rad -= np.pi*2
  return rad

def delta_theta(lhs, rhs):  # in radians
  rad = lhs - rhs
  return rad

def range_phi_deg(deg):
  while deg <  -180.:
    deg += 360.
  while deg >= +180.:
    deg -= 360.
  return deg

def calc_phi_loc_deg_from_glob(glob, sector):
  # glob in deg, sector [1-6]
  glob = range_phi_deg(glob)
  loc = glob - 15. - (60. * (sector-1))
  return loc

def calc_phi_loc_int(glob, sector):
  # glob in deg, sector [1-6]
  loc = calc_phi_loc_deg_from_glob(glob, sector)
  if (loc + 22.) < 0.:
    loc += 360.
  loc = (loc + 22.) * 60.
  phi_int = int(round(loc))
  return phi_int

def calc_phi_loc_deg(bits):
  loc = float(bits)/60. - 22.
  return loc

def calc_phi_glob_deg(loc, sector):
  # loc in deg, sector [1-6]
  glob = loc + 15. + (60. * (sector-1))
  if glob >= 180.:
    glob -= 360.
  return glob

def calc_theta_int(theta, endcap):
  # theta in deg, endcap [-1,+1]
  if endcap == -1:
    theta = 180. - theta
  theta = (theta - 8.5) * 128./(45.0-8.5)
  theta_int = int(round(theta))
  return theta_int

def calc_theta_rad_from_eta(eta):
  theta = np.arctan2(1.0, np.sinh(eta))
  return theta

def calc_theta_deg_from_eta(eta):
  return np.rad2deg(calc_theta_rad_from_eta(eta))

def calc_theta_deg_from_int(theta_int):
  theta_deg = float(theta_int) * (45.0-8.5)/128. + 8.5;
  return theta_deg

def calc_eta_from_theta_deg(theta_deg, endcap):
  # theta in deg, endcap [-1,+1]
  theta_rad = np.deg2rad(theta_deg)
  eta = -1. * np.log(np.tan(theta_rad/2.))
  if endcap == -1:
    eta = -eta
  return eta

def extrapolate_to_emtf(phi, invpt, eta):  # phi in radians
  # 1.204 is the magic constant at eta of 1.9
  eta_sf = np.sinh(1.9) / np.sinh(np.abs(eta))
  return phi - 1.204 * invpt * eta_sf

def find_sector(phi):  # phi in radians
  dphi = delta_phi(phi, np.pi/12)  # sector 1 starts at 15 deg
  dphi = int(np.floor(dphi/(np.pi/3)))  # divide by 60 deg
  if dphi < 0:
    sector = 7 + dphi
  else:
    sector = 1 + dphi
  return sector

def find_endcap(eta):
  endcap = +1 if eta >= 0. else -1
  return endcap

def find_endsec(endcap, sector):
  endsec = (sector - 1) if endcap == 1 else (sector - 1 + 6)
  return endsec

def find_pt_bin(pt):
  ipt = np.digitize((pt,), pt_bins[1:])[0]  # skip lowest edge
  ipt = np.clip(ipt, 0, len(pt_bins)-2)
  return ipt

def find_eta_bin(eta):
  ieta = np.digitize((np.abs(eta),), eta_bins[1:])[0]  # skip lowest edge
  ieta = np.clip(ieta, 0, len(eta_bins)-2)
  return ieta


# More functions

# From https://stackoverflow.com/a/31539746
def weighted_percentile(data, percents, weights=None):
  ''' percents in units of 1%
  weights specifies the frequency (count) of data.
  '''
  if weights is None:
    return np.percentile(data, percents)
  ind=np.argsort(data)
  d=data[ind]
  w=weights[ind]
  p=1.*w.cumsum()/w.sum()*100
  y=np.interp(percents, p, d)
  return y

def is_valid_for_run2(hit):
  is_csc = (hit.type == kCSC)
  is_rpc = (hit.type == kRPC)
  is_irpc = (hit.type == kRPC) and ((hit.station == 3 or hit.station == 4) and hit.ring == 1)
  return (is_csc or (is_rpc and not is_irpc))

# Decide EMTF hit layer number
class EMTFLayer(object):
  def __init__(self):
    lut = np.zeros((5,5,5), dtype=np.int32) - 99  # (type, station, ring) -> layer
    lut[1,1,4] = 0  # ME1/1a
    lut[1,1,1] = 0  # ME1/1b
    lut[1,1,2] = 1  # ME1/2
    lut[1,1,3] = 1  # ME1/3
    lut[1,2,1] = 2  # ME2/1
    lut[1,2,2] = 2  # ME2/2
    lut[1,3,1] = 3  # ME3/1
    lut[1,3,2] = 3  # ME3/2
    lut[1,4,1] = 4  # ME4/1
    lut[1,4,2] = 4  # ME4/2
    lut[2,1,2] = 5  # RE1/2
    lut[2,2,2] = 6  # RE2/2
    lut[2,3,1] = 7  # RE3/1
    lut[2,3,2] = 7  # RE3/2
    lut[2,3,3] = 7  # RE3/3
    lut[2,4,1] = 8  # RE4/1
    lut[2,4,2] = 8  # RE4/2
    lut[2,4,3] = 8  # RE4/3
    lut[3,1,1] = 9  # GE1/1
    lut[3,2,1] = 10 # GE2/1
    lut[4,1,1] = 11 # ME0
    self.lut = lut

  def __call__(self, hit):
    index = (hit.type, hit.station, hit.ring)
    entry = self.lut[index]
    return entry

find_emtf_layer = EMTFLayer()

# Decide EMTF hit zones
class EMTFZone(object):
  def __init__(self):
    lut = np.zeros((5,5,5,6,2), dtype=np.int32) - 99  # (type, station, ring) -> [zone] x [min_theta,max_theta]
    lut[1,1,4][0] = 4,17   # ME1/1a
    lut[1,1,4][1] = 16,26  # ME1/1a
    lut[1,1,4][2] = 24,37  # ME1/1a
    lut[1,1,4][3] = 34,43  # ME1/1a
    lut[1,1,4][4] = 40,53  # ME1/1a
    lut[1,1,1][0] = 4,17   # ME1/1b
    lut[1,1,1][1] = 16,26  # ME1/1b
    lut[1,1,1][2] = 24,37  # ME1/1b
    lut[1,1,1][3] = 34,43  # ME1/1b
    lut[1,1,1][4] = 40,53  # ME1/1b
    lut[1,1,2][4] = 46,54  # ME1/2
    lut[1,1,2][5] = 52,88  # ME1/2
    lut[1,1,3][4] = 46,54  # ME1/3
    lut[1,1,3][5] = 52,88  # ME1/3
    #
    lut[1,2,1][0] = 4,17   # ME2/1
    lut[1,2,1][1] = 16,25  # ME2/1
    lut[1,2,1][2] = 24,36  # ME2/1
    lut[1,2,1][3] = 34,43  # ME2/1
    lut[1,2,1][4] = 40,49  # ME2/1
    lut[1,2,2][5] = 53,88  # ME2/2
    #
    lut[1,3,1][0] = 4,17   # ME3/1
    lut[1,3,1][1] = 16,25  # ME3/1
    lut[1,3,1][2] = 24,36  # ME3/1
    lut[1,3,1][3] = 34,40  # ME3/1
    lut[1,3,2][4] = 44,54  # ME3/2
    lut[1,3,2][5] = 52,88  # ME3/2
    #
    lut[1,4,1][0] = 4,17   # ME4/1
    lut[1,4,1][1] = 16,25  # ME4/1
    lut[1,4,1][2] = 24,35  # ME4/1
    lut[1,4,2][3] = 38,43  # ME4/2
    lut[1,4,2][4] = 41,54  # ME4/2
    lut[1,4,2][5] = 52,88  # ME4/2
    #
    lut[2,1,2][5] = 52,84  # RE1/2
    lut[2,2,2][5] = 56,76  # RE2/2
    lut[2,3,1][0] = 4,20   # RE3/1
    lut[2,3,1][1] = 20,24  # RE3/1
    lut[2,3,1][2] = 24,32  # RE3/1
    lut[2,3,2][3] = 40,40  # RE3/2
    lut[2,3,2][4] = 40,52  # RE3/2
    lut[2,3,2][5] = 48,84  # RE3/2
    lut[2,3,3][3] = 40,40  # RE3/3
    lut[2,3,3][4] = 40,52  # RE3/3
    lut[2,3,3][5] = 48,84  # RE3/3
    lut[2,4,1][0] = 8,16   # RE4/1
    lut[2,4,1][1] = 16,28  # RE4/1
    lut[2,4,1][2] = 24,28  # RE4/1
    lut[2,4,2][3] = 36,44  # RE4/2
    lut[2,4,2][4] = 44,52  # RE4/2
    lut[2,4,2][5] = 52,84  # RE4/2
    lut[2,4,3][3] = 36,44  # RE4/3
    lut[2,4,3][4] = 44,52  # RE4/3
    lut[2,4,3][5] = 52,84  # RE4/3
    #
    lut[3,1,1][1] = 17,26  # GE1/1
    lut[3,1,1][2] = 24,37  # GE1/1
    lut[3,1,1][3] = 35,45  # GE1/1
    lut[3,1,1][4] = 40,52  # GE1/1
    lut[3,2,1][0] = 7,19   # GE2/1
    lut[3,2,1][1] = 18,24  # GE2/1
    lut[3,2,1][2] = 23,35  # GE2/1
    lut[3,2,1][3] = 34,45  # GE2/1
    lut[3,2,1][4] = 40,46  # GE2/1
    #
    lut[4,1,1][0] = 4,17   # ME0
    lut[4,1,1][1] = 16,23  # ME0
    self.lut = lut

  def __call__(self, hit):
    index = (hit.type, hit.station, hit.ring)
    entry = self.lut[index]
    answer = (entry[:,0] <= hit.emtf_theta) & (hit.emtf_theta <= entry[:,1])
    zones = np.nonzero(answer)
    if isinstance(zones, tuple):
      zones = zones[0]
    return zones

find_emtf_zones = EMTFZone()

# Decide EMTF hit bend
class EMTFBend(object):
  def __init__(self):
    self.lut = np.array([5, -5, 4, -4, 3, -3, 2, -2, 1, -1, 0], dtype=np.int32)

  def __call__(self, hit):
    if hit.type == kCSC:
      #clct = int(hit.pattern)
      #bend = self.lut[clct]
      bend = hit.bend
      if hit.station == 1:
        # Special case for ME1/1a:
        # rescale the bend to the same scale as ME1/1b
        if hit.ring == 4:
          bend *= 0.026331/0.014264
      elif hit.station in (2,3,4):
        if -8 <= bend <= 8:
          bend = 0
        elif bend > 8:
          bend = +1
        else:
          bend = -1
      bend *= hit.endcap
    elif hit.type == kGEM:
      bend = hit.bend
      bend *= hit.endcap
    elif hit.type == kME0:
      bend = hit.bend
    else:
      bend = 0
    return bend

find_emtf_bend = EMTFBend()

class EMTFZee(object):
  def __init__(self):
    self.lut = np.array([599.0, 696.8, 827.1, 937.5, 1027, 708.7, 790.9, 968.8, 1060, 566.4, 794.8, 539.3], dtype=np.float32)
    assert(self.lut.shape[0] == nlayers)

  def __call__(self, hit):
    return self.lut[hit.emtf_layer]

find_emtf_zee = EMTFZee()

class EMTFPhi(object):
  def __init__(self):
    pass

  def __call__(self, hit):
    if hit.type == kCSC:
      if hit.station == 1:
        if hit.ring == 1:
          bend_corr_lut = (-1.3861, 1.3692)  # ME1/1b (r,f)
        elif hit.ring == 4:
          bend_corr_lut = (-1.6419, 1.6012)  # ME1/1a (r,f)
        else:
          bend_corr_lut = (-0.9237, 0.8287)  # ME1/2 (r,f)
        bend_corr = bend_corr_lut[int(hit.fr)] * hit.bend
        bend_corr = bend_corr if hit.endcap == 1 else (bend_corr * -1)
        bend_corr = int(round(bend_corr))
        return hit.emtf_phi + bend_corr
      else:
        pass
    else:
      pass
    return hit.emtf_phi

find_emtf_phi = EMTFPhi()

class EMTFLayerPartner(object):
  def __init__(self):
    self.lut = np.array([2, 2, 0, 0, 0, 0, 2, 3, 4, 0, 2, 0], dtype=np.int32)
    assert(self.lut.shape[0] == nlayers)

  def __call__(self, emtf_layer, zone):
    partner = self.lut[emtf_layer]
    if zone >= 5:  # zones 5,6, use ME1/2
      if partner == 0:
        partner = 1
    return partner

find_emtf_layer_partner = EMTFLayerPartner()

# Decide EMTF road quality (by pT)
class EMTFRoadQuality(object):
  def __init__(self):
    self.best_ipt = find_pt_bin(0.)

  def __call__(self, ipt):
    return self.best_ipt - np.abs(ipt - self.best_ipt)

find_emtf_road_quality = EMTFRoadQuality()

# Decide EMTF road sort code
class EMTFRoadSortCode(object):
  def __init__(self):
    # 10     9      8      7    6      5      4      3      2..0
    # ME1/1, GE1/1, ME1/2, ME2, GE2/1, ME3&4, RE1&2, RE3&4, qual
    self.hits_to_mlayer = (10,8,7,5,5,4,4,3,3,9,6,10)

  def __call__(self, mode, qual, hits):
    code = 0
    for hit in hits:
      hit_lay = hit.emtf_layer
      mlayer = self.hits_to_mlayer[hit_lay]
      code |= (1 << mlayer)
    code |= qual
    return code

find_emtf_road_sort_code = EMTFRoadSortCode()

# Decide EMTF road mode
def is_emtf_singlemu(mode):
  return mode in (11,13,14,15)

def is_emtf_doublemu(mode):
  return mode in (7,10,12) + (11,13,14,15)

def is_emtf_muopen(mode):
  return mode in (3,5,6,9) + (7,10,12) + (11,13,14,15)

def is_emtf_single(mode):
  return bool(mode & (1 << 3))

# Decide EMTF legit hit
def is_emtf_legit_hit(hit):
  def check_bx(hit):
    if hit.type == kCSC:
      return hit.bx in (-1,0)
    else:
      return hit.bx == 0
  #
  def check_quality(hit):
    if hit.type == kRPC:
      return hit.quality <= 9  # cluster width
    else:
      return True
  #
  return check_bx(hit) and check_quality(hit)


# ______________________________________________________________________________
# Classes

class Particle(object):
  def __init__(self, pt, eta, phi, q):
    self.pt = pt
    self.eta = eta
    self.phi = phi
    self.q = q

  def to_parameters(self):
    parameters = np.array((np.true_divide(self.q, self.pt), self.phi, self.eta), dtype=np.float32)
    return parameters

class PatternBank(object):
  def __init__(self, bankfile):
    with np.load(bankfile) as data:
      patterns_phi = data['patterns_phi']
      #patterns_theta = data['patterns_theta']
      patterns_theta = np.zeros_like(patterns_phi)
      patterns_match = data['patterns_match']
    self.x_array = patterns_phi
    self.y_array = patterns_theta
    self.z_array = patterns_match
    assert(self.x_array.dtype == np.int32)
    assert(self.y_array.dtype == np.int32)
    assert(self.z_array.dtype == np.int32)
    assert(self.x_array.shape == (len(pt_bins)-1, len(eta_bins)-1, nlayers, 3))
    assert(self.y_array.shape == (len(pt_bins)-1, len(eta_bins)-1, nlayers, 3))
    assert(self.z_array.shape == (len(pt_bins)-1, len(eta_bins)-1, nlayers, 3))

class Hit(object):
  def __init__(self, _id, emtf_layer, emtf_phi, emtf_theta, emtf_bend, time, sim_tp):
    self.id = _id  # (_type, station, ring, sector, fr, bx)
    self.emtf_layer = emtf_layer
    self.emtf_phi = emtf_phi
    self.emtf_theta = emtf_theta
    self.emtf_bend = emtf_bend
    self.time = time
    self.sim_tp = sim_tp

  def get_ring(self):
    return self.id[2]

  def get_sector(self):
    return self.id[3]

  def get_fr(self):
    return self.id[4]

  def get_bx(self):
    return self.id[5]

class Road(object):
  def __init__(self, _id, hits, mode, mode_csc, quality, sort_code, theta_median):
    self.id = _id  # (endcap, sector, ipt, ieta, iphi)
    self.hits = hits
    self.mode = mode
    self.mode_csc = mode_csc
    self.quality = quality
    self.sort_code = sort_code
    self.theta_median = theta_median

  def to_variables(self):
    amap = {}
    (endcap, sector, ipt, ieta, iphi) = self.id
    road_info = (ipt, ieta, iphi)
    #np.random.shuffle(self.hits)  # randomize the order
    for hit in self.hits:
      hit_lay = hit.emtf_layer
      if hit_lay not in amap:
        amap[hit_lay] = hit
    #
    arr = np.zeros((nlayers * 7) + 3, dtype=np.float32)
    arr[0*nlayers:6*nlayers] = np.nan  # emtf_phi, emtf_theta, emtf_bend, time, ring, fr
    arr[6*nlayers:7*nlayers] = 1.0     # mask
    arr[7*nlayers:]          = road_info
    for lay, hit in amap.iteritems():
      ind = [i*nlayers + lay for i in xrange(6)]
      arr[ind] = (hit.emtf_phi, hit.emtf_theta, hit.emtf_bend, hit.get_bx(), hit.get_ring(), hit.get_fr())
      ind = (6*nlayers + lay)
      arr[ind] = 0.0
    return arr

class Track(object):
  def __init__(self, _id, hits, mode, xml_pt, pt, q, emtf_phi, emtf_theta, ndof, chi2):
    assert(pt > 0.)
    self.id = _id  # (endcap, sector)
    self.hits = hits
    self.mode = mode
    self.xml_pt = xml_pt
    self.pt = pt
    self.q = q
    self.emtf_phi = emtf_phi
    self.emtf_theta = emtf_theta
    self.ndof = ndof
    self.chi2 = chi2
    self.phi = calc_phi_glob_deg(calc_phi_loc_deg(emtf_phi), _id[1])
    self.eta = calc_eta_from_theta_deg(calc_theta_deg_from_int(emtf_theta), _id[0])

def particles_to_parameters(particles):
  parameters = np.zeros((len(particles), 3), dtype=np.float32)
  for i, part in enumerate(particles):
    parameters[i] = part.to_parameters()
  return parameters

def roads_to_variables(roads):
  variables = np.zeros((len(roads), (nlayers * 7) + 3), dtype=np.float32)
  for i, road in enumerate(roads):
    variables[i] = road.to_variables()
  return variables


# Pattern recognition module
class PatternRecognition(object):
  def __init__(self, bank):
    self.bank = bank
    self.cache = dict()  # cache for pattern results

  def _create_road_hit(self, hit):
    hit_id = (hit.type, hit.station, hit.ring, hit.endsec, hit.fr, hit.bx)
    hit_sim_tp = (hit.sim_tp1 == 0 and hit.sim_tp2 == 0)
    myhit = Hit(hit_id, hit.lay, hit.emtf_phi, hit.emtf_theta, hit.emtf_bend, hit.time, hit_sim_tp)
    return myhit

  def _apply_patterns_in_zone(self, zone, hit_lay):
    result = self.cache.get((zone, hit_lay), None)
    if result is not None:
      return result

    # Retrieve patterns with (ipt, ieta, lay, pattern)
    patterns_x0 = self.bank.x_array[:, zone, hit_lay, 0, np.newaxis]
    patterns_x1 = self.bank.x_array[:, zone, hit_lay, 2, np.newaxis]
    patterns_iphi = np.arange(-23, 23+1, dtype=np.int32)
    mask = (patterns_x0 <= patterns_iphi) & (patterns_iphi <= patterns_x1)
    result = np.transpose(np.nonzero(mask))
    self.cache[(zone, hit_lay)] = result
    return result

  def _apply_patterns(self, endcap, sector, sector_hits):
    amap = {}  # road_id -> list of 'myhit'

    # Loop over hits
    for ihit, hit in enumerate(sector_hits):
      hit_x = (hit.emtf_phi+16)//32  # divide by 'quadstrip' unit (4 * 8)
      #hit_y = hit.emtf_theta
      hit_lay = hit.lay
      hit_zones = hit.zones

      myhit = None

      # Loop over the zones that the hit is belong to
      for zone in hit_zones:
        result = self._apply_patterns_in_zone(zone, hit_lay)

        for index in result:
          ipt, iphi = index
          iphi = hit_x - (iphi - 23)  # iphi 0 starts at -23
          ieta = zone
          if 0 <= iphi <= 4928//32:  # divide by 'quadstrip' unit (4 * 8)
            # Create 'myhit'
            if myhit is None:
              myhit = self._create_road_hit(hit)

            # Associate 'myhit' to road ids
            road_id = (endcap, sector, ipt, ieta, iphi)
            amap.setdefault(road_id, []).append(myhit)  # append hit to road

    # Create a road
    roads = []
    for road_id, road_hits in amap.iteritems():
      (endcap, sector, ipt, ieta, iphi) = road_id
      road_mode = 0
      road_mode_csc = 0
      road_mode_me0 = 0
      tmp_road_hits = []
      tmp_thetas = []

      for hit in road_hits:
        (_type, station, ring, endsec, fr, bx) = hit.id
        road_mode |= (1 << (4 - station))
        if _type == kCSC or _type == kME0:
          road_mode_csc |= (1 << (4 - station))

        if _type == kME0:
          road_mode_me0 |= (1 << 2)
        elif _type == kCSC and station == 1 and (ring == 1 or ring == 4):
          road_mode_me0 |= (1 << 1)
        elif _type == kCSC and station >= 2:
          road_mode_me0 |= (1 << 0)

        tmp_road_hits.append(hit)
        if _type == kCSC:
          tmp_thetas.append(hit.emtf_theta)

      # Apply modified SingleMu requirement
      if (is_emtf_singlemu(road_mode) and is_emtf_muopen(road_mode_csc)) or (road_mode_me0 >= 6):
        road_quality = find_emtf_road_quality(ipt)
        road_sort_code = find_emtf_road_sort_code(road_mode, road_quality, tmp_road_hits)
        tmp_theta = np.median(tmp_thetas, overwrite_input=True)

        myroad = Road(road_id, tmp_road_hits, road_mode, road_mode_csc, road_quality, road_sort_code, tmp_theta)
        roads.append(myroad)
    return roads

  def run(self, hits):
    roads = []

    # Split by sector
    sector_mode_array = np.zeros((12,), dtype=np.int32)
    sector_hits_array = np.empty((12,), dtype=np.object)
    for ind in np.ndindex(sector_hits_array.shape):
      sector_hits_array[ind] = []

    legit_hits = filter(is_emtf_legit_hit, hits)

    for ihit, hit in enumerate(legit_hits):
      hit.endsec = find_endsec(hit.endcap, hit.sector)
      hit.lay = find_emtf_layer(hit)
      assert(hit.lay != -99)

      if (hit.type == kCSC or hit.type == kME0):
        sector_mode_array[hit.endsec] |= (1 << (4 - hit.station))
      sector_hits_array[hit.endsec].append(hit)

    # Loop over sector processors
    for endcap in (-1, +1):
      for sector in (1, 2, 3, 4, 5, 6):
        endsec = find_endsec(endcap, sector)
        sector_mode = sector_mode_array[endsec]
        sector_hits = sector_hits_array[endsec]

        # Provide early exit (using csc-only 'mode')
        if not is_emtf_muopen(sector_mode):
          continue

        # Remove all RPC hits
        #sector_hits = [hit for hit in sector_hits if hit.type != kRPC]

        # Remove all non-Run 2 hits
        only_use_run2 = False
        if only_use_run2:
          sector_hits = filter(is_valid_for_run2, sector_hits)

        # Loop over sector hits
        for ihit, hit in enumerate(sector_hits):
          hit.old_emtf_phi = hit.emtf_phi
          #hit.emtf_zee = find_emtf_zee(hit)
          hit.emtf_phi = find_emtf_phi(hit)
          hit.emtf_bend = find_emtf_bend(hit)
          hit.zones = find_emtf_zones(hit)

        # Apply patterns
        sector_roads = self._apply_patterns(endcap, sector, sector_hits)
        roads += sector_roads
    return roads


# Road cleaning module
class RoadCleaning(object):
  def __init__(self):
    pass

  # https://stackoverflow.com/a/30396816
  def _iter_from_middle(self, lst):
    try:
      middle = len(lst)//2
      yield lst[middle]
      for shift in xrange(1, middle+1):
        # order is important!
        yield lst[middle - shift]
        yield lst[middle + shift]
    except IndexError: # occures on lst[len(lst)] or for empty list
      raise StopIteration

  def _groupby(self, data):
    def is_adjacent(prev, curr, length):
      # adjacent if (x,y,z) == (x,y,z+length)
      return prev[:-1] == curr[:-1] and (prev[-1] + length) == curr[-1]

    if data:
      data.sort()
      myiter = iter(data)
      prev = curr = next(myiter)
      # Iterate over data
      while True:
        group = []
        stop = False
        # Iterate until the next value is different
        while is_adjacent(prev, curr, len(group)):
          try:
            group.append(curr)
            curr = next(myiter)
          except StopIteration:
            stop = True
            break
        # Output group
        yield group
        prev = curr
        if stop:
          return

  def _sortby(self, clean_roads, groupinfo):
    def select_bx_zero(road):
      bx_counter1 = 0  # count hits with BX <= -1
      bx_counter2 = 0  # count hits with BX <= 0
      bx_counter3 = 0  # count hits with BX > 0
      layer_has_been_used = set()
      for hit in road.hits:
        if hit.emtf_layer not in layer_has_been_used:
          layer_has_been_used.add(hit.emtf_layer)
          if hit.get_bx() <= -1:
            bx_counter1 += 1
          if hit.get_bx() <= 0:
            bx_counter2 += 1
          if hit.get_bx() > 0:
            bx_counter3 += 1
      #trk_bx_zero = (bx_counter1 < 2 and bx_counter2 >= 2)
      trk_bx_zero = (bx_counter1 < 3 and bx_counter2 >= 2 and bx_counter3 < 2)
      return trk_bx_zero

    clean_roads = list(filter(select_bx_zero, clean_roads))

    if clean_roads:
      clean_roads.sort(key=lambda road: road.sort_code, reverse=True)

      # Iterate over clean_roads
      for i, road in enumerate(clean_roads):
        keep = True
        gi = groupinfo[road.id]

        # No intersect between two ranges (x1, x2), (y1, y2): (x2 < y1) || (x1 > y2)
        # Intersect: !((x2 < y1) || (x1 > y2)) = (x2 >= y1) and (x1 <= y2)
        for j, road_to_check in enumerate(clean_roads[:i]):
          gj = groupinfo[road_to_check.id]
          _get_endsec = lambda x: x[:2]
          # Allow +/-2 due to extrapolation-to-EMTF error
          if (_get_endsec(road.id) == _get_endsec(road_to_check.id)) and (gi[1]+2 >= gj[0]) and (gi[0]-2 <= gj[1]):
            keep = False
            break

        if keep:
          # Do not share ME1/1, ME1/2 or ME0
          for j, road_to_check in enumerate(clean_roads[:i]):
            hits_i = [(hit.emtf_layer, hit.emtf_phi) for hit in road.hits if hit.emtf_layer in (0,1,11)]
            hits_j = [(hit.emtf_layer, hit.emtf_phi) for hit in road_to_check.hits if hit.emtf_layer in (0,1,11)]
            if set(hits_i).intersection(hits_j):
              keep = False
              break

        if keep:
          yield road
      return

  def run(self, roads):
    # road_id = (endcap, sector, ipt, ieta, iphi)
    amap = {road.id : road for road in roads}

    # pick median in each iphi group
    clean_roads = []
    groupinfo = {}

    # Loop over road clusters
    for group in self._groupby(amap.keys()):

      # Loop over roads in road clusters, starting from middle
      for index in self._iter_from_middle(xrange(len(group))):
        road_id = group[index]
        road = amap[road_id]
        keep = True
        if (0 <= index-1) and road.sort_code < amap[group[index-1]].sort_code:
          keep = False
        if (index+1 < len(group)) and road.sort_code < amap[group[index+1]].sort_code:
          keep = False
        if keep:
          break

      _get_iphi = lambda x: x[4]
      g = (_get_iphi(group[0]), _get_iphi(group[-1]))  # first and last road_id's in the iphi group
      groupinfo[road_id] = g
      clean_roads.append(road)

    # sort the roads + kill the siblings
    sorted_clean_roads = []
    for road in self._sortby(clean_roads, groupinfo):
      sorted_clean_roads.append(road)
    return sorted_clean_roads


# Road slimming module
class RoadSlimming(object):
  def __init__(self, bank):
    self.bank = bank

  def run(self, roads):
    slim_roads = []

    for road in roads:
      ipt, ieta, iphi = road.id[2:]

      prim_match_lut = self.bank.z_array[ipt, ieta, :, 1]

      tmp_phi = (iphi * 32)  # multiply by 'quadstrip' unit (4 * 8)

      #_select_csc = lambda x: (x.emtf_layer <= 4)
      #tmp_thetas = [hit.emtf_theta for hit in road.hits if _select_csc(hit)]
      #tmp_theta = np.median(tmp_thetas, overwrite_input=True)
      tmp_theta = road.theta_median

      hits_array = np.empty((nlayers,), dtype=np.object)
      for ind in np.ndindex(hits_array.shape):
        hits_array[ind] = []

      best_phi_array = np.full((nlayers,), tmp_phi, dtype=np.int32)
      best_theta_array = np.full((nlayers,), tmp_theta, dtype=np.int32)

      for hit in road.hits:
        hit_lay = hit.emtf_layer
        hits_array[hit_lay].append(hit)

      # Assume going through ME1, ME2, ... in order
      for hit_lay in xrange(nlayers):
        mean_dphi = prim_match_lut[hit_lay]
        hit_lay_p = find_emtf_layer_partner(hit_lay, ieta)

        # Make pairs of (hit1, hit2)
        # Want to pick the best hit1, given the selection of hit2 and mean_dphi
        pairs = []
        if hits_array[hit_lay]:
          for hit1 in hits_array[hit_lay]:
            if hits_array[hit_lay_p]:
              for hit2 in hits_array[hit_lay_p]:
                dphi = abs((hit1.emtf_phi - hit2.emtf_phi) - mean_dphi)
                dtheta = abs(hit1.emtf_theta - tmp_theta)
                pairs.append((hit1, hit2, dphi, dtheta))
                #print hit1.emtf_phi, hit2.emtf_phi, abs((hit1.emtf_phi - hit2.emtf_phi)), dphi
                continue  # end loop over hit2
            else:
              dphi = abs((hit1.emtf_phi - best_phi_array[hit_lay_p]) - mean_dphi)
              dtheta = abs(hit1.emtf_theta - tmp_theta)
              pairs.append((hit1, hit1, dphi, dtheta))
            continue  # end loop over hit1

          # Find best pair, which is min (dtheta, dphi)
          best_pair = min(pairs, key=lambda x: (x[3], x[2]))
          best_hit = best_pair[0]
          hits_array[hit_lay] = [best_hit]
          best_phi_array[hit_lay] = best_hit.emtf_phi
          best_theta_array[hit_lay] = best_hit.emtf_theta
        else:
          # No hit in layer, put in the best estimate
          best_phi_array[hit_lay] = best_phi_array[hit_lay_p] + mean_dphi
          best_theta_array[hit_lay] = tmp_theta

      slim_road_hits = []
      for hit_lay in xrange(nlayers):
        if hits_array[hit_lay]:
          assert(len(hits_array[hit_lay]) == 1)
          slim_road_hits.append(hits_array[hit_lay][0])

      slim_road = Road(road.id, slim_road_hits, road.mode, road.mode_csc, road.quality, road.sort_code, road.theta_median)
      slim_roads.append(slim_road)
    return slim_roads


# pT assignment module
class PtAssignment(object):
  def __init__(self, kerasfile):
    (model_file, model_weights_file) = kerasfile

    adjust_scale = 3

    reg_pt_scale = 100.

    # Get encoder
    from nn_encode import Encoder

    # Get custom objects
    from nn_models import update_keras_custom_objects
    update_keras_custom_objects()

    # Load Keras model
    from nn_models import load_my_model, update_keras_custom_objects
    update_keras_custom_objects()
    self.loaded_model = load_my_model(model_file.replace('.json',''), model_weights_file.replace('.h5',''))
    self.loaded_model.trainable = False
    assert not self.loaded_model.updates

    def create_encoder(x):
      nentries = x.shape[0]
      y = np.zeros((nentries, 3), dtype=np.float32)  # dummy
      encoder = Encoder(x, y, adjust_scale=adjust_scale, reg_pt_scale=reg_pt_scale)
      return encoder
    self.create_encoder = create_encoder

    def predict(x):
      y = self.loaded_model.predict(x)
      assert len(y) == 2
      y[0] /= reg_pt_scale
      y = np.moveaxis(np.asarray(y),0,-1)  # shape (2, n, 1) -> shape (n, 1, 2)
      return y
    self.predict = predict

  def run(self, x):
    x_new = np.array([], dtype=np.float32)
    y = np.array([], dtype=np.float32)
    z = np.array([], dtype=np.float32)
    if len(x) == 0:
      return (x_new, y, z)

    encoder = self.create_encoder(x)
    x_new = encoder.get_x()
    y = self.predict(x_new)
    x_mask = encoder.get_x_mask()
    return (x_new, y, x_mask)


# Track producer module
class TrackProducer(object):
  def __init__(self):
    self.s_min = 0.
    self.s_max = 60.
    self.s_nbins = 120
    self.s_step = (self.s_max - self.s_min)/self.s_nbins
    self.s_lut =[ 1.8005,  1.5194,  1.5708,  1.8247,  2.1989,  2.6489,  3.1625,  3.7251,
                  4.3240,  4.9595,  5.6337,  6.3424,  7.0590,  7.7485,  8.4050,  9.0398,
                  9.6598, 10.2800, 10.9236, 11.6060, 12.3216, 13.0521, 13.7887, 14.5427,
                 15.2964, 16.0232, 16.7303, 17.4535, 18.2066, 19.0044, 19.8400, 20.6934,
                 21.5215, 22.3143, 23.1066, 23.8221, 24.4586, 25.1335, 25.9083, 26.7333,
                 27.5310, 28.2623, 28.9778, 29.7226, 30.5507, 31.4670, 32.4541, 33.5263,
                 34.5659, 35.5155, 36.4457, 37.4019, 38.3762, 39.3604, 40.3595, 41.3763,
                 42.3333, 43.2434, 44.2686, 45.5962, 47.0878, 48.3783, 49.4891, 50.5445,
                 51.4431, 52.2846, 53.1180, 53.9492, 54.7793, 55.6090, 56.4384, 57.2676,
                 58.0967, 58.9257, 59.7547, 60.5836, 61.4125, 62.2413, 63.0702, 63.8990,
                 64.7278, 65.5566, 66.3854, 67.2142, 68.0430, 68.8718, 69.7006, 70.5293,
                 71.3581, 72.1869, 73.0157, 73.8444, 74.6732, 75.5020, 76.3307, 77.1595,
                 77.9882, 78.8170, 79.6458, 80.4745, 81.3033, 82.1321, 82.9608, 83.7896,
                 84.6183, 85.4471, 86.2759, 87.1046, 87.9334, 88.7621, 89.5909, 90.4197,
                 91.2484, 92.0772, 92.9059, 93.7347, 94.5635, 95.3922, 96.2210, 97.0497]
    #self.s_lut = np.linspace(self.s_min, self.s_max, num=self.s_nbins+1)[:-1]
    self.s_step = np.asarray(self.s_step)
    self.s_lut = np.asarray(self.s_lut)

  def get_trigger_pt(self, x, y_meas):
    xml_pt = np.abs(1.0/y_meas)
    if xml_pt <= 2.:  # do not use the LUT if below 2 GeV
      return xml_pt

    def digitize(x, bins=(self.s_nbins, self.s_min, self.s_max)):
      x = np.clip(x, bins[1], bins[2]-1e-5)
      binx = (x - bins[1]) / (bins[2] - bins[1]) * bins[0]
      return binx.astype(np.int32)

    def interpolate(x, x0, x1, y0, y1):
      y = (x - x0) / (x1 - x0) * (y1 - y0) + y0
      return y

    binx = digitize(xml_pt)
    if binx == self.s_nbins-1:  # check boundary
      binx -= 1

    x0, x1 = binx * self.s_step, (binx+1) * self.s_step
    y0, y1 = self.s_lut[binx], self.s_lut[binx+1]
    pt = interpolate(xml_pt, x0, x1, y0, y1)
    return pt

  def pass_trigger(self, strg, ndof, mode, theta_median, y_meas, y_discr, discr_pt_cut=14.):
    ipt1 = strg
    ipt2 = find_pt_bin(y_meas)
    quality1 = find_emtf_road_quality(ipt1)
    quality2 = find_emtf_road_quality(ipt2)

    if mode in (11,13,14,15) and quality2 <= (quality1+1):
      if np.abs(1.0/y_meas) > 14:
        trigger = (y_discr > 0.8597) # 98.0% coverage
      elif np.abs(1.0/y_meas) > discr_pt_cut:
        trigger = (y_discr > 0.7217) # 98.0% coverage
      else:
        trigger = (y_discr >= 0.)  # True
    else:
      trigger = (y_discr < 0.)  # False
    return trigger

  def run(self, slim_roads, variables, predictions, other_vars):

    # __________________________________________________________________________
    # Extra pieces
    nvariables = (nlayers * 6) + 3 - 36

    discr_pt_cut = 8.

    def get_theta_median_from_x(x):
      assert(x.shape[0] == nvariables)
      theta_median = x[-1] # last variable
      theta_median = (theta_median * 83) + 3
      return theta_median.astype(np.int32)

    def get_zone_from_x(x):
      assert(x.shape[0] == nvariables)
      zone = x[-2] # second last variable
      zone = (zone * 5) + 0
      return zone.astype(np.int32)

    def get_straightness_from_x(x):
      assert(x.shape[0] == nvariables)
      straightness = x[-3] # third last variable
      straightness = (straightness * 4) + 4
      return straightness.astype(np.int32)

    def get_ndof_from_x_mask(x_mask):
      assert(x_mask.shape[0] == nlayers)
      assert(x_mask.dtype == np.bool)
      valid = ~x_mask
      return valid.sum()

    def get_mode_from_x_mask(x_mask):
      assert(x_mask.shape[0] == nlayers)
      assert(x_mask.dtype == np.bool)
      valid = ~x_mask
      mode = np.int32(0)
      if np.any([valid[0], valid[1], valid[5], valid[9], valid[11]]):   # ME1/1, ME1/2, RE1/2, GE1/1, ME0
        mode |= (1<<3)
      if np.any([valid[2], valid[6], valid[10]]):  # ME2, RE2, GE2/1
        mode |= (1<<2)
      if np.any([valid[3], valid[7]]):  # ME3, RE3
        mode |= (1<<1)
      if np.any([valid[4], valid[8]]):  # ME4, RE4
        mode |= (1<<0)

      # Apply modified SingleMu requirement
      mode_me0 = np.int32(0)
      if valid[11]: # ME0
        mode_me0 |= (1 << 2)
      if valid[0]:  # ME1/1
        mode_me0 |= (1 << 1)
      if np.any([valid[2], valid[3], valid[4]]):  # ME2, ME3, ME4
        mode_me0 |= (1 << 0)
      if mode not in (11,13,14,15) and mode_me0 == 7:
        mode = 11  # pretend as mode 11
      return mode

    # __________________________________________________________________________
    assert(len(slim_roads) == len(variables))
    assert(len(slim_roads) == len(predictions))
    assert(len(slim_roads) == len(other_vars))

    tracks = []

    for myroad, myvars, mypreds, myother in zip(slim_roads, variables, predictions, other_vars):
      assert(len(myvars.shape) == 1)
      x = myvars
      x_mask = myother
      y_meas = np.asscalar(mypreds[...,0])
      y_discr = np.asscalar(mypreds[...,1])

      theta_median = get_theta_median_from_x(x)
      zone = get_zone_from_x(x)
      strg = get_straightness_from_x(x)
      ndof = get_ndof_from_x_mask(x_mask)
      mode = get_mode_from_x_mask(x_mask)

      passed = self.pass_trigger(strg, ndof, mode, theta_median, y_meas, y_discr, discr_pt_cut=discr_pt_cut)
      xml_pt = np.abs(1.0/y_meas)
      pt = self.get_trigger_pt(x, y_meas)

      if passed:
        trk_q = np.sign(y_meas)
        trk_emtf_phi = myroad.id[4]
        trk_emtf_theta = myroad.theta_median
        trk = Track(myroad.id, myroad.hits, mode, xml_pt, pt, trk_q, trk_emtf_phi, trk_emtf_theta, ndof, y_discr)
        tracks.append(trk)
    return tracks


# ______________________________________________________________________________
# Book histograms
histograms = {}

# Efficiency
eff_pt_bins = (0., 0.5, 1., 1.5, 2., 3., 4., 5., 6., 7., 8., 10., 12., 14., 16., 18., 20., 22., 24., 27., 30., 34., 40., 48., 60., 80., 120.)

for k in ("denom", "numer"):
  hname = "eff_vs_genpt_%s" % k
  histograms[hname] = Hist(eff_pt_bins, name=hname, title="; gen p_{T} [GeV]", type='F')
  histograms[hname].Sumw2()

  hname = "eff_vs_geneta_%s" % k
  histograms[hname] = Hist(70, 1.1, 2.5, name=hname, title="; gen |#eta|", type='F')
  histograms[hname].Sumw2()

  hname = "eff_vs_genphi_%s" % k
  histograms[hname] = Hist(64, -3.2, 3.2, name=hname, title="; gen #phi", type='F')
  histograms[hname].Sumw2()

# Rates
hname = "nevents"
histograms[hname] = Hist(5, 0, 5, name=hname, title="; count", type='F')
for m in ("emtf", "emtf2023"):
  hname = "highest_%s_absEtaMin1.24_absEtaMax2.4_qmin12_pt" % m
  histograms[hname] = Hist(100, 0., 100., name=hname, title="; p_{T} [GeV]; entries", type='F')
  hname = "highest_%s_absEtaMin1.24_absEtaMax1.65_qmin12_pt" % m
  histograms[hname] = Hist(100, 0., 100., name=hname, title="; p_{T} [GeV]; entries", type='F')
  hname = "highest_%s_absEtaMin1.65_absEtaMax2.15_qmin12_pt" % m
  histograms[hname] = Hist(100, 0., 100., name=hname, title="; p_{T} [GeV]; entries", type='F')
  hname = "highest_%s_absEtaMin2.15_absEtaMax2.4_qmin12_pt" % m
  histograms[hname] = Hist(100, 0., 100., name=hname, title="; p_{T} [GeV]; entries", type='F')

  for l in xrange(14,22+1):
    hname = "%s_ptmin%i_qmin12_eta" % (m,l)
    histograms[hname] = Hist(10, 1.55, 2.55, name=hname, title="; |#eta|; entries", type='F')

# Effie
for m in ("emtf", "emtf2023"):
  for l in (0, 10, 15, 20, 30, 40, 50):
    for k in ("denom", "numer"):
      hname = "%s_eff_vs_genpt_l1pt%i_%s" % (m,l,k)
      histograms[hname] = Hist(eff_pt_bins, name=hname, title="; gen p_{T} [GeV]", type='F')
      hname = "%s_eff_vs_geneta_l1pt%i_%s" % (m,l,k)
      histograms[hname] = Hist(70, 1.1, 2.5, name=hname, title="; gen |#eta| {gen p_{T} > 20 GeV}", type='F')
      hname = "%s_eff_vs_geneta_genpt30_l1pt%i_%s" % (m,l,k)
      histograms[hname] = Hist(70, 1.1, 2.5, name=hname, title="; gen |#eta| {gen p_{T} > 30 GeV}", type='F')

  hname = "%s_l1pt_vs_genpt" % m
  histograms[hname] = Hist2D(100, -0.5, 0.5, 300, -0.5, 0.5, name=hname, title="; gen 1/p_{T} [1/GeV]; 1/p_{T} [1/GeV]", type='F')
  hname = "%s_l1ptres_vs_genpt" % m
  histograms[hname] = Hist2D(100, -0.5, 0.5, 300, -1, 2, name=hname, title="; gen 1/p_{T} [1/GeV]; #Delta(p_{T})/p_{T}", type='F')


# ______________________________________________________________________________
# Settings

# Get number of events
#maxEvents = -1
#maxEvents = 4000000
maxEvents = 1000

# Condor or not
use_condor = ('CONDOR_EXEC' in os.environ)

# Analysis mode
#analysis = 'verbose'
#analysis = 'application'
analysis = 'rates'
#analysis = 'effie'
#analysis = 'mixing'
#analysis = 'images'
if use_condor:
  analysis = sys.argv[1]

# Job
jobid = 0
if use_condor:
  jobid = int(sys.argv[2])

print('[INFO] Using cmssw         : %s' % os.environ['CMSSW_VERSION'])
print('[INFO] Using condor        : %i' % use_condor)
print('[INFO] Using max events    : %i' % maxEvents)
print('[INFO] Using analysis mode : %s' % analysis)
print('[INFO] Using job id        : %s' % jobid)

# Other stuff
bankfile = 'pattern_bank.20.npz'

kerasfile = ['model.r3.1.json', 'model_weights.r3.1.h5']

infile_r = None  # input file handle

def load_pgun():
  global infile_r
  infile = 'ntuple_SingleMuon_Toy_2GeV_add.6.root'
  if use_condor:
    infile = 'root://cmsio5.rc.ufl.edu//store/user/jiafulow/L1MuonTrigger/P2_10_1_5/SingleMuon_Toy_2GeV/'+infile
  infile_r = root_open(infile)
  tree = infile_r.ntupler.tree
  #tree = TreeChain('ntupler/tree', [infile])
  print('[INFO] Opening file: %s' % infile)

  # Define collection
  tree.define_collection(name='hits', prefix='vh_', size='vh_size')
  tree.define_collection(name='tracks', prefix='vt_', size='vt_size')
  tree.define_collection(name='particles', prefix='vp_', size='vp_size')
  #tree.define_collection(name='evt_info', prefix='ve_', size='ve_size')
  return tree

def load_pgun_batch(j):
  #global infile_r
  #infile_r = root_open('pippo.root', 'w')

  jj = np.split(np.arange(2000), 200)[j]
  infiles = []
  for j in jj:
    infiles.append('root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_1_5/SingleMuon_Toy_2GeV/ParticleGuns/CRAB3/180813_212614/%04i/ntuple_SingleMuon_Toy_%i.root' % ((j+1)/1000, (j+1)))
    infiles.append('root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_1_5/SingleMuon_Toy2_2GeV/ParticleGuns/CRAB3/180813_212740/%04i/ntuple_SingleMuon_Toy2_%i.root' % ((j+1)/1000, (j+1)))

  tree = TreeChain('ntupler/tree', infiles)
  print('[INFO] Opening file: %s' % ' '.join(infiles))

  # Define collection
  tree.define_collection(name='hits', prefix='vh_', size='vh_size')
  tree.define_collection(name='tracks', prefix='vt_', size='vt_size')
  tree.define_collection(name='particles', prefix='vp_', size='vp_size')
  #tree.define_collection(name='evt_info', prefix='ve_', size='ve_size')
  return tree

def load_minbias_batch(j):
  global infile_r
  pufiles = ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_1_5/ntuple_SingleNeutrino_PU200/ParticleGuns/CRAB3/180925_011729/0000/ntuple_SingleNeutrino_PU200_%i.root' % (i+1) for i in xrange(63)]
  #pufiles = ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_1_5/ntuple_SingleNeutrino_PU140/ParticleGuns/CRAB3/180925_011552/0000/ntuple_SingleNeutrino_PU140_%i.root' % (i+1) for i in xrange(56)]
  infile = pufiles[j]
  infile_r = root_open(infile)
  tree = infile_r.ntupler.tree
  print('[INFO] Opening file: %s' % infile)

  # Define collection
  tree.define_collection(name='hits', prefix='vh_', size='vh_size')
  tree.define_collection(name='tracks', prefix='vt_', size='vt_size')
  tree.define_collection(name='particles', prefix='vp_', size='vp_size')
  tree.define_collection(name='evt_info', prefix='ve_', size='ve_size')
  return tree

def load_minbias_batch_for_mixing(j):
  global infile_r
  pufiles = []
  # For training purposes
  pufiles += ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_1_5/ntuple_SingleNeutrino_PU140/ParticleGuns/CRAB3/180925_011552/0000/ntuple_SingleNeutrino_PU140_%i.root' % (i+1) for i in xrange(20)]  # up to 20/56
  pufiles += ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_1_5/ntuple_SingleNeutrino_PU200/ParticleGuns/CRAB3/180925_011729/0000/ntuple_SingleNeutrino_PU200_%i.root' % (i+1) for i in xrange(30)]  # up to 30/63
  #pufiles += ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_1_5/ntuple_SingleMuon_PU140/ParticleGuns/CRAB3/180925_011845/0000/ntuple_SingleMuon_PU140_%i.root' % (i+1) for i in xrange(25)]
  #pufiles += ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_1_5/ntuple_SingleMuon_PU200/ParticleGuns/CRAB3/180925_012012/0000/ntuple_SingleMuon_PU200_%i.root' % (i+1) for i in xrange(26)]
  #pufiles += ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_1_5/ntuple_SingleElectron_PU140/ParticleGuns/CRAB3/180925_012138/0000/ntuple_SingleElectron_PU140_%i.root' % (i+1) for i in xrange(28)]  # all jobs failed
  pufiles += ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_1_5/ntuple_SingleElectron_PU200/ParticleGuns/CRAB3/180925_012258/0000/ntuple_SingleElectron_PU200_%i.root' % (i+1) for i in xrange(27)]
  pufiles += ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_1_5/ntuple_SinglePhoton_PU140/ParticleGuns/CRAB3/180925_012419/0000/ntuple_SinglePhoton_PU140_%i.root' % (i+1) for i in xrange(27)]
  pufiles += ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_1_5/ntuple_SinglePhoton_PU200/ParticleGuns/CRAB3/180925_012545/0000/ntuple_SinglePhoton_PU200_%i.root' % (i+1) for i in xrange(27)]
  # For testing purposes (SingleNeutrino, PU200)
  pufiles += ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_1_5/ntuple_SingleNeutrino_PU200/ParticleGuns/CRAB3/180925_011729/0000/ntuple_SingleNeutrino_PU200_%i.root' % (i+1) for i in xrange(30,63)]  # from 30/63

  infile = pufiles[j]
  infile_r = root_open(infile)
  tree = infile_r.ntupler.tree
  print('[INFO] Opening file: %s' % infile)

  # Define collection
  tree.define_collection(name='hits', prefix='vh_', size='vh_size')
  tree.define_collection(name='tracks', prefix='vt_', size='vt_size')
  tree.define_collection(name='particles', prefix='vp_', size='vp_size')
  tree.define_collection(name='evt_info', prefix='ve_', size='ve_size')
  return tree

def unload_tree():
  global infile_r
  try:
    infile_r.close()
  except:
    pass


# ______________________________________________________________________________
# Analysis: verbose
if analysis == 'verbose':
  tree = load_pgun()

  # Loop over events
  for ievt, evt in enumerate(tree):
    if maxEvents != -1 and ievt == maxEvents:
      break

    print("Processing event: {0}".format(ievt))

    # Hits
    for ihit, hit in enumerate(evt.hits):
      print(".. hit  {0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11}".format(ihit, hit.bx, hit.type, hit.station, hit.ring, hit.sector, hit.fr, hit.sim_phi, hit.sim_theta, hit.time, hit.sim_tp1, hit.sim_tp2))
    # Tracks
    for itrk, trk in enumerate(evt.tracks):
      print(".. trk  {0} {1} {2} {3} {4} {5} {6} {7}".format(itrk, trk.sector, trk.mode, trk.pt, trk.phi, trk.eta, trk.theta, trk.q))
    # Gen particles
    for ipart, part in enumerate(evt.particles):
      print(".. part {0} {1} {2} {3} {4} {5}".format(ipart, part.pt, part.phi, part.eta, part.theta, part.q))

  # End loop over events
  unload_tree()




# ______________________________________________________________________________
# Analysis: application
elif analysis == 'application':
  #tree = load_pgun()
  tree = load_pgun_batch(jobid)

  # Workers
  bank = PatternBank(bankfile)
  recog = PatternRecognition(bank)
  clean = RoadCleaning()
  slim = RoadSlimming(bank)
  out_particles = []
  out_roads = []
  npassed, ntotal = 0, 0

  # Event range
  n = -1

  # ____________________________________________________________________________
  # Loop over events
  for ievt, evt in enumerate(tree):
    if n != -1 and ievt == n:
      break

    roads = recog.run(evt.hits)
    clean_roads = clean.run(roads)
    slim_roads = slim.run(clean_roads)
    assert(len(clean_roads) == len(slim_roads))

    part = evt.particles[0]  # particle gun
    part.invpt = np.true_divide(part.q, part.pt)

    if len(slim_roads) > 0:
      mypart = Particle(part.pt, part.eta, part.phi, part.q)
      out_particles.append(mypart)
      out_roads.append(slim_roads[0])

    is_important = lambda part: (1.24 <= abs(part.eta) <= 2.4) and (part.bx == 0) and (part.pt > 5.)

    is_possible = lambda hits: any([((hit.type == kCSC or hit.type == kME0) and hit.station == 1) for hit in hits]) and \
        any([(hit.type == kCSC and hit.station >= 2) for hit in hits])

    if ievt < 20 or (len(clean_roads) == 0 and is_important(part) and is_possible(evt.hits)):
      print("evt {0} has {1} roads and {2} clean roads".format(ievt, len(roads), len(clean_roads)))
      print(".. part invpt: {0} pt: {1} eta: {2} phi: {3}".format(part.invpt, part.pt, part.eta, part.phi))
      part.ipt = find_pt_bin(part.invpt)
      part.ieta = find_eta_bin(part.eta)
      #part.exphi = emtf_extrapolation(part)
      #part.sector = find_sector(part.exphi)
      #part.endcap = find_endcap(part.eta)
      #part.emtf_phi = calc_phi_loc_int(np.rad2deg(part.exphi), part.sector)
      #part.emtf_theta = calc_theta_int(calc_theta_deg_from_eta(part.eta), part.endcap)
      #part_road_id = (part.endcap, part.sector, part.ipt, part.ieta, (part.emtf_phi+16)//32)
      #part_nhits = sum([1 for hit in evt.hits if hit.endcap == part.endcap and hit.sector == part.sector])
      #print(".. part road id: {0} nhits: {1} exphi: {2} emtf_phi: {3}".format(part_road_id, part_nhits, part.exphi, part.emtf_phi))
      for ihit, hit in enumerate(evt.hits):
        hit_id = (hit.type, hit.station, hit.ring, find_endsec(hit.endcap, hit.sector), hit.fr, hit.bx)
        hit_sim_tp = (hit.sim_tp1 == 0 and hit.sim_tp2 == 0)
        print(".. .. hit {0} id: {1} lay: {2} ph: {3} th: {4} tp: {5}".format(ihit, hit_id, find_emtf_layer(hit), hit.emtf_phi, hit.emtf_theta, hit_sim_tp))
      for iroad, myroad in enumerate(sorted(roads, key=lambda x: x.id)):
        print(".. road {0} id: {1} nhits: {2} mode: {3} qual: {4} sort: {5}".format(iroad, myroad.id, len(myroad.hits), myroad.mode, myroad.quality, myroad.sort_code))
      for iroad, myroad in enumerate(clean_roads):
        print(".. croad {0} id: {1} nhits: {2} mode: {3} qual: {4} sort: {5}".format(iroad, myroad.id, len(myroad.hits), myroad.mode, myroad.quality, myroad.sort_code))
        for ihit, myhit in enumerate(myroad.hits):
          print(".. .. hit {0} id: {1} lay: {2} ph: {3} th: {4} tp: {5}".format(ihit, myhit.id, myhit.emtf_layer, myhit.emtf_phi, myhit.emtf_theta, myhit.sim_tp))
      for iroad, myroad in enumerate(slim_roads):
        print(".. sroad {0} id: {1} nhits: {2} mode: {3} qual: {4} sort: {5}".format(iroad, myroad.id, len(myroad.hits), myroad.mode, myroad.quality, myroad.sort_code))
        for ihit, myhit in enumerate(myroad.hits):
          print(".. .. hit {0} id: {1} lay: {2} ph: {3} th: {4} tp: {5}".format(ihit, myhit.id, myhit.emtf_layer, myhit.emtf_phi, myhit.emtf_theta, myhit.sim_tp))

    # Quick efficiency
    if is_important(part):
      trigger = len(clean_roads) > 0
      ntotal += 1
      if trigger:
        npassed += 1
      #else:
      #  print("evt {0} FAIL".format(ievt))

      hname = "eff_vs_genpt_denom"
      histograms[hname].fill(part.pt)
      if trigger:
        hname = "eff_vs_genpt_numer"
        histograms[hname].fill(part.pt)

      if part.pt > 20.:
        hname = "eff_vs_geneta_denom"
        histograms[hname].fill(abs(part.eta))
        if trigger:
          hname = "eff_vs_geneta_numer"
          histograms[hname].fill(abs(part.eta))
        hname = "eff_vs_genphi_denom"
        histograms[hname].fill(part.phi)
        if trigger:
          hname = "eff_vs_genphi_numer"
          histograms[hname].fill(part.phi)

  # End loop over events
  unload_tree()

  # ____________________________________________________________________________
  # Plot histograms
  outfile = 'histos_tba.root'
  if use_condor:
    outfile = 'histos_tba_%i.root' % jobid
  print('[INFO] Creating file: %s' % outfile)
  with root_open(outfile, 'recreate') as f:
    for hname in ["eff_vs_genpt", "eff_vs_geneta", "eff_vs_genphi"]:
      denom = histograms[hname + "_denom"]
      numer = histograms[hname + "_numer"]
      eff = Efficiency(numer, denom, name=hname)
      eff.SetStatisticOption(0)  # kFCP
      eff.SetConfidenceLevel(0.682689492137)  # one sigma
      eff.Write()
    print('[INFO] npassed/ntotal: %i/%i = %f' % (npassed, ntotal, float(npassed)/ntotal))

  # ____________________________________________________________________________
  # Save objects
  outfile = 'histos_tba.npz'
  if use_condor:
    outfile = 'histos_tba_%i.npz' % jobid
  print('[INFO] Creating file: %s' % outfile)
  if True:
    assert(len(out_particles) == len(out_roads))
    parameters = particles_to_parameters(out_particles)
    variables = roads_to_variables(out_roads)
    np.savez_compressed(outfile, parameters=parameters, variables=variables)




# ______________________________________________________________________________
# Analysis: rates

elif analysis == 'rates':
  tree = load_minbias_batch(jobid)

  # Workers
  bank = PatternBank(bankfile)
  recog = PatternRecognition(bank)
  clean = RoadCleaning()
  slim = RoadSlimming(bank)
  ptassign = PtAssignment(kerasfile)
  trkprod = TrackProducer()
  out_variables = []
  out_predictions = []

  # Event range
  n = -1

  # ____________________________________________________________________________
  # Loop over events
  for ievt, evt in enumerate(tree):
    if n != -1 and ievt == n:
      break

    roads = recog.run(evt.hits)
    clean_roads = clean.run(roads)
    slim_roads = slim.run(clean_roads)
    variables = roads_to_variables(slim_roads)
    variables_mod, predictions, other_vars = ptassign.run(variables)
    emtf2023_tracks = trkprod.run(slim_roads, variables_mod, predictions, other_vars)

    found_high_pt_tracks = any(map(lambda trk: trk.pt > 20., emtf2023_tracks))

    #if found_high_pt_tracks:
    #  out_variables.append(variables)
    #  out_predictions.append(predictions)

    if found_high_pt_tracks:
      print("evt {0} has {1} roads, {2} clean roads, {3} old tracks, {4} new tracks".format(ievt, len(roads), len(clean_roads), len(evt.tracks), len(emtf2023_tracks)))
      for ipart, part in enumerate(evt.particles):
        if part.pt > 5.:
          part.invpt = np.true_divide(part.q, part.pt)
          print(".. part invpt: {0} pt: {1} eta: {2} phi: {3}".format(part.invpt, part.pt, part.eta, part.phi))
      for iroad, myroad in enumerate(clean_roads):
        print(".. croad {0} id: {1} nhits: {2} mode: {3} qual: {4} sort: {5}".format(iroad, myroad.id, len(myroad.hits), myroad.mode, myroad.quality, myroad.sort_code))
        #for ihit, myhit in enumerate(myroad.hits):
        #  print(".. .. hit {0} id: {1} lay: {2} ph: {3} th: {4} tp: {5}".format(ihit, myhit.id, myhit.emtf_layer, myhit.emtf_phi, myhit.emtf_theta, myhit.sim_tp))
      for itrk, mytrk in enumerate(emtf2023_tracks):
        print(".. trk {0} id: {1} nhits: {2} mode: {3} pt: {4} ndof: {5} chi2: {6}".format(itrk, mytrk.id, len(mytrk.hits), mytrk.mode, mytrk.pt, mytrk.ndof, mytrk.chi2))
        for ihit, myhit in enumerate(mytrk.hits):
          print(".. .. hit {0} id: {1} lay: {2} ph: {3} th: {4} tp: {5}".format(ihit, myhit.id, myhit.emtf_layer, myhit.emtf_phi, myhit.emtf_theta, myhit.sim_tp))
      for itrk, mytrk in enumerate(evt.tracks):
        print(".. otrk {0} id: {1} mode: {2} pt: {3}".format(itrk, (mytrk.endcap, mytrk.sector), mytrk.mode, mytrk.pt))


    # Fill histograms
    histograms["nevents"].fill(1.0)

    def fill_highest_pt():
      highest_pt = -999999.
      for itrk, trk in enumerate(tracks):
        if select(trk):
          if highest_pt < trk.pt:  # using scaled pT
            highest_pt = trk.pt
      if highest_pt > 0.:
        highest_pt = min(100.-1e-3, highest_pt)
        histograms[hname].fill(highest_pt)

    def fill_eta():
      h = histograms[hname]
      eta_bins = [False] * (10+2)
      for itrk, trk in enumerate(tracks):
        if select(trk):
          b = h.FindFixBin(abs(trk.eta))
          eta_bins[b] = True
      for b in xrange(len(eta_bins)):
        if eta_bins[b]:
          h.fill(h.GetBinCenter(b))

    tracks = evt.tracks
    #
    select = lambda trk: trk and (1.24 <= abs(trk.eta) <= 2.4) and (trk.bx == 0) and (trk.mode in (11,13,14,15))
    hname = "highest_emtf_absEtaMin1.24_absEtaMax2.4_qmin12_pt"
    fill_highest_pt()
    select = lambda trk: trk and (1.24 <= abs(trk.eta) < 1.65) and (trk.bx == 0) and (trk.mode in (11,13,14,15))
    hname = "highest_emtf_absEtaMin1.24_absEtaMax1.65_qmin12_pt"
    fill_highest_pt()
    select = lambda trk: trk and (1.65 <= abs(trk.eta) < 2.15) and (trk.bx == 0) and (trk.mode in (11,13,14,15))
    hname = "highest_emtf_absEtaMin1.65_absEtaMax2.15_qmin12_pt"
    fill_highest_pt()
    select = lambda trk: trk and (2.15 <= abs(trk.eta) <= 2.4) and (trk.bx == 0) and (trk.mode in (11,13,14,15))
    hname = "highest_emtf_absEtaMin2.15_absEtaMax2.4_qmin12_pt"
    fill_highest_pt()
    for l in xrange(14,22+1):
      select = lambda trk: trk and (1.24 <= abs(trk.eta) <= 2.4) and (trk.bx == 0) and (trk.mode in (11,13,14,15)) and (trk.pt > float(l))
      hname = "emtf_ptmin%i_qmin12_eta" % (l)
      fill_eta()

    tracks = emtf2023_tracks
    #
    select = lambda trk: trk and (1.24 <= abs(trk.eta) <= 2.4)
    hname = "highest_emtf2023_absEtaMin1.24_absEtaMax2.4_qmin12_pt"
    fill_highest_pt()
    select = lambda trk: trk and (1.24 <= abs(trk.eta) < 1.65)
    hname = "highest_emtf2023_absEtaMin1.24_absEtaMax1.65_qmin12_pt"
    fill_highest_pt()
    select = lambda trk: trk and (1.65 <= abs(trk.eta) < 2.15)
    hname = "highest_emtf2023_absEtaMin1.65_absEtaMax2.15_qmin12_pt"
    fill_highest_pt()
    select = lambda trk: trk and (2.15 <= abs(trk.eta) <= 2.4)
    hname = "highest_emtf2023_absEtaMin2.15_absEtaMax2.4_qmin12_pt"
    fill_highest_pt()
    for l in xrange(14,22+1):
      select = lambda trk: trk and (1.24 <= abs(trk.eta) <= 2.4) and (trk.pt > float(l))
      hname = "emtf2023_ptmin%i_qmin12_eta" % (l)
      fill_eta()

  # End loop over events
  unload_tree()

  # ____________________________________________________________________________
  # Plot histograms
  outfile = 'histos_tbb.root'
  if use_condor:
    outfile = 'histos_tbb_%i.root' % jobid
  print('[INFO] Creating file: %s' % outfile)
  with root_open(outfile, 'recreate') as f:
    hnames = []
    hname = "nevents"
    hnames.append("nevents")
    for m in ("emtf", "emtf2023"):
      hname = "highest_%s_absEtaMin1.24_absEtaMax2.4_qmin12_pt" % m
      hnames.append(hname)
      hname = "highest_%s_absEtaMin1.24_absEtaMax1.65_qmin12_pt" % m
      hnames.append(hname)
      hname = "highest_%s_absEtaMin1.65_absEtaMax2.15_qmin12_pt" % m
      hnames.append(hname)
      hname = "highest_%s_absEtaMin2.15_absEtaMax2.4_qmin12_pt" % m
      hnames.append(hname)
      for l in xrange(14,22+1):
        hname = "%s_ptmin%i_qmin12_eta" % (m,l)
        hnames.append(hname)
    for hname in hnames:
      h = histograms[hname]
      h.Write()




# ______________________________________________________________________________
# Analysis: effie
elif analysis == 'effie':
  #tree = load_pgun()
  tree = load_pgun_batch(jobid)

  # Workers
  bank = PatternBank(bankfile)
  recog = PatternRecognition(bank)
  clean = RoadCleaning()
  slim = RoadSlimming(bank)
  ptassign = PtAssignment(kerasfile)
  trkprod = TrackProducer()

  # Event range
  n = -1

  # ____________________________________________________________________________
  # Loop over events
  for ievt, evt in enumerate(tree):
    if n != -1 and ievt == n:
      break

    part = evt.particles[0]  # particle gun
    part.invpt = np.true_divide(part.q, part.pt)

    roads = recog.run(evt.hits)
    clean_roads = clean.run(roads)
    slim_roads = slim.run(clean_roads)
    variables = roads_to_variables(slim_roads)
    variables_mod, predictions, other_vars = ptassign.run(variables)
    emtf2023_tracks = trkprod.run(slim_roads, variables_mod, predictions, other_vars)

    if ievt < 20 and False:
      print("evt {0} has {1} roads, {2} clean roads, {3} old tracks, {4} new tracks".format(ievt, len(roads), len(clean_roads), len(evt.tracks), len(emtf2023_tracks)))
      for itrk, mytrk in enumerate(emtf2023_tracks):
        y = np.true_divide(part.q, part.pt)
        y_pred = np.true_divide(mytrk.q, mytrk.xml_pt)
        print(".. {0} {1}".format(y, y_pred))

    # Fill histograms
    def fill_efficiency_pt():
      trigger = any([select(trk) for trk in tracks])  # using scaled pT
      denom = histograms[hname + "_denom"]
      numer = histograms[hname + "_numer"]
      if (1.24 <= abs(part.eta) <= 2.4) and (part.bx == 0):
        denom.fill(part.pt)
        if trigger:
          numer.fill(part.pt)

    def fill_efficiency_eta():
      trigger = any([select(trk) for trk in tracks])  # using scaled pT
      denom = histograms[hname + "_denom"]
      numer = histograms[hname + "_numer"]
      if (part.bx == 0):
        denom.fill(abs(part.eta))
        if trigger:
          numer.fill(abs(part.eta))

    def fill_resolution():
      trigger = any([select(trk) for trk in tracks])  # using scaled pT
      if (part.bx == 0) and trigger:
        trk = tracks[0]
        trk.invpt = np.true_divide(trk.q, trk.xml_pt)  # using unscaled pT
        histograms[hname1].fill(part.invpt, trk.invpt)
        histograms[hname2].fill(abs(part.invpt), (abs(1.0/trk.invpt) - abs(1.0/part.invpt))/abs(1.0/part.invpt))

    for l in (0, 10, 15, 20, 30, 40, 50):
      select = lambda trk: trk and (1.24 <= abs(trk.eta) <= 2.4) and (trk.mode in (11,13,14,15)) and (trk.pt > float(l))
      tracks = evt.tracks
      #
      hname = "emtf_eff_vs_genpt_l1pt%i" % (l)
      fill_efficiency_pt()
      if part.pt > 20.:
        hname = "emtf_eff_vs_geneta_l1pt%i" % (l)
        fill_efficiency_eta()
      if part.pt > 30.:
        hname = "emtf_eff_vs_geneta_genpt30_l1pt%i" % (l)
        fill_efficiency_eta()
      if l == 0:
        hname1 = "emtf_l1pt_vs_genpt"
        hname2 = "emtf_l1ptres_vs_genpt"
        fill_resolution()

      select = lambda trk: trk and (1.24 <= abs(trk.eta) <= 2.4) and (trk.pt > float(l))
      tracks = emtf2023_tracks
      #
      hname = "emtf2023_eff_vs_genpt_l1pt%i" % (l)
      fill_efficiency_pt()
      if part.pt > 20.:
        hname = "emtf2023_eff_vs_geneta_l1pt%i" % (l)
        fill_efficiency_eta()
      if part.pt > 30.:
        hname = "emtf2023_eff_vs_geneta_genpt30_l1pt%i" % (l)
        fill_efficiency_eta()
      if l == 0:
        hname1 = "emtf2023_l1pt_vs_genpt"
        hname2 = "emtf2023_l1ptres_vs_genpt"
        fill_resolution()

  # End loop over events
  unload_tree()

  # ____________________________________________________________________________
  # Plot histograms
  outfile = 'histos_tbc.root'
  if use_condor:
    outfile = 'histos_tbc_%i.root' % jobid
  print('[INFO] Creating file: %s' % outfile)
  with root_open(outfile, 'recreate') as f:
    hnames = []
    for m in ("emtf", "emtf2023"):
      for l in (0, 10, 15, 20, 30, 40, 50):
        for k in ("denom", "numer"):
          hname = "%s_eff_vs_genpt_l1pt%i_%s" % (m,l,k)
          hnames.append(hname)
          hname = "%s_eff_vs_geneta_l1pt%i_%s" % (m,l,k)
          hnames.append(hname)
          hname = "%s_eff_vs_geneta_genpt30_l1pt%i_%s" % (m,l,k)
          hnames.append(hname)
      hname = "%s_l1pt_vs_genpt" % m
      hnames.append(hname)
      hname = "%s_l1ptres_vs_genpt" % m
      hnames.append(hname)
    for hname in hnames:
      h = histograms[hname]
      h.Write()




# ______________________________________________________________________________
# Analysis: mixing
elif analysis == 'mixing':
  #tree = load_minbias_batch(jobid)
  tree = load_minbias_batch_for_mixing(jobid)

  # Workers
  bank = PatternBank(bankfile)
  recog = PatternRecognition(bank)
  clean = RoadCleaning()
  slim = RoadSlimming(bank)
  #ptassign = PtAssignment(kerasfile)
  #trkprod = TrackProducer()
  out_particles = []
  out_roads = []
  npassed, ntotal = 0, 0

  # Event range
  n = -1

  # ____________________________________________________________________________
  # Loop over events
  for ievt, evt in enumerate(tree):
    if n != -1 and ievt == n:
      break

    roads = recog.run(evt.hits)
    clean_roads = clean.run(roads)
    slim_roads = slim.run(clean_roads)
    #variables = roads_to_variables(slim_roads)
    #variables_mod, predictions, other_vars = ptassign.run(variables)
    #emtf2023_tracks = trkprod.run(slim_roads, variables_mod, predictions, other_vars)

    def find_highest_part_pt():
      highest_pt = -999999.
      for ipart, part in enumerate(evt.particles):
        if select(part):
          if highest_pt < part.pt:
            highest_pt = part.pt
      if highest_pt > 0.:
        highest_pt = min(100.-1e-3, highest_pt)
        return highest_pt

    select = lambda part: (1.24 <= abs(part.eta) <= 2.4) and (part.bx == 0)
    highest_part_pt = find_highest_part_pt()

    def find_highest_track_pt():
      highest_pt = -999999.
      for itrk, trk in enumerate(evt.tracks):
        if select(trk):
          if highest_pt < trk.pt:  # using scaled pT
            highest_pt = trk.pt
      if highest_pt > 0.:
        highest_pt = min(100.-1e-3, highest_pt)
        return highest_pt

    select = lambda trk: trk and (1.24 <= abs(trk.eta) <= 2.4) and (trk.bx == 0) and (trk.mode in (11,13,14,15))
    highest_track_pt = find_highest_track_pt()

    if len(slim_roads) > 0:
      part = (jobid, ievt, highest_part_pt, highest_track_pt)
      out_particles += [part for _ in xrange(len(slim_roads))]
      out_roads += slim_roads

    debug_event_list = [2826, 2937, 3675, 4581, 4838, 5379, 7640]

    if ievt < 20 or ievt in debug_event_list:
      print("evt {0} has {1} roads, {2} clean roads, {3} old tracks, {4} new tracks".format(ievt, len(roads), len(clean_roads), len(evt.tracks), '?'))
      #for iroad, myroad in enumerate(sorted(roads, key=lambda x: x.id)):
      #  print(".. road {0} id: {1} nhits: {2} mode: {3} qual: {4} sort: {5}".format(iroad, myroad.id, len(myroad.hits), myroad.mode, myroad.quality, myroad.sort_code))
      for iroad, myroad in enumerate(clean_roads):
        print(".. croad {0} id: {1} nhits: {2} mode: {3} qual: {4} sort: {5}".format(iroad, myroad.id, len(myroad.hits), myroad.mode, myroad.quality, myroad.sort_code))
        for ihit, myhit in enumerate(myroad.hits):
          print(".. .. hit {0} id: {1} lay: {2} ph: {3} th: {4} tp: {5}".format(ihit, myhit.id, myhit.emtf_layer, myhit.emtf_phi, myhit.emtf_theta, myhit.sim_tp))

  # End loop over events
  unload_tree()

  # ____________________________________________________________________________
  # Save objects
  outfile = 'histos_tbd.npz'
  if use_condor:
    outfile = 'histos_tbd_%i.npz' % jobid
  print('[INFO] Creating file: %s' % outfile)
  if True:
    assert(len(out_roads) == len(out_particles))
    variables = roads_to_variables(out_roads)
    aux = np.array(out_particles, dtype=np.float32)
    np.savez_compressed(outfile, variables=variables, aux=aux)



