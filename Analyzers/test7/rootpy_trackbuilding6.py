import numpy as np
np.random.seed(2023)

import os, sys
from itertools import izip
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
eta_bins = (1.2, 1.4, 1.55, 1.7, 1.8, 1.98, 2.15, 2.5)
eta_bins = eta_bins[::-1]
pt_bins = (-0.50, -0.333333, -0.25, -0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20, 0.25, 0.333333, 0.50)
nlayers = 12  # 5 (CSC) + 4 (RPC) + 3 (GEM)
superstrip_size = 16

assert(len(eta_bins) == 7+1)
assert(len(pt_bins) == 13+1)


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

def calc_phi_loc_int_1(glob, sector):
  # glob in deg, sector [1-6]
  loc = calc_phi_loc_deg_from_glob(glob, sector)
  if (loc + 22.) < 0.:
    loc += 360.
  loc = (loc + 22.) * 60.
  #phi_int = int(round(loc))
  phi_int = loc  # no cast to int
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
  eta_sf = np.sinh(1.9) / np.sinh(abs(eta))
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
  ieta = np.digitize((abs(eta),), eta_bins[1:])[0]  # skip lowest edge
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

# Decide EMTF hit layer number
class EMTFLayer(object):
  def __init__(self):
    lut = np.zeros((5,5,5), dtype=np.int32) - 99
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

  def get(self, hit):
    index = (hit.type, hit.station, hit.ring)
    return self.lut[index]

anemtflayer = EMTFLayer()
def emtf_layer(hit):
  return anemtflayer.get(hit)

# Decide EMTF hit bend
class EMTFBend(object):
  def __init__(self):
    self.lut = np.array([5, -5, 4, -4, 3, -3, 2, -2, 1, -1, 0], dtype=np.int32)

  def get(self, hit):
    if hit.type == kCSC:
      #clct = int(hit.pattern)
      #bend = self.lut[clct]
      bend = hit.bend
      bend *= hit.endcap
    elif hit.type == kGEM:
      bend = hit.bend
      bend *= hit.endcap
    else:  # kRPC, kME0
      bend = hit.bend
    return bend

anemtfbend = EMTFBend()
def emtf_bend(hit):
  return anemtfbend.get(hit)

# Decide EMTF road quality (by pT)
class EMTFRoadQuality(object):
  def __init__(self):
    self.best_ipt = find_pt_bin(0.)

  def get(self, ipt):
    return self.best_ipt - abs(ipt - self.best_ipt)

anemtfroadquality = EMTFRoadQuality()
def emtf_road_quality(ipt):
  return anemtfroadquality.get(ipt)

# Decide EMTF road sort code
class EMTFRoadSortCode(object):
  def __init__(self):
    pass

  def get(self, road_mode, road_quality, road_hits):
    #def madorsky_code(mode, qual):
    #  code = 0
    #  code |= ((mode >> 3) & 1) << 6
    #  code |= ((qual >> 2) & 1) << 5
    #  code |= ((mode >> 2) & 1) << 4
    #  code |= ((qual >> 1) & 1) << 3
    #  code |= ((mode >> 1) & 1) << 2
    #  code |= ((qual >> 0) & 1) << 1
    #  code |= ((mode >> 0) & 1) << 0
    #return code

    def mlayer_code(hits, qual):
      # 11     10     9      8      7    6      5    4    3      2..0
      # GE1/1, ME1/1, ME1/2, GE2/1, ME2, RE1&2, ME3, ME4, RE3&4, qual
      hits_to_mlayer = (10,9,7,5,4,6,6,3,3,11,8,11)
      code = 0
      for hit in hits:
        hit_lay = hit.emtf_layer
        mlayer = hits_to_mlayer[hit_lay]
        code |= (1 << mlayer)
      code |= qual
      return code
    return mlayer_code(road_hits, road_quality)

anemtfroadsortcode = EMTFRoadSortCode()
def emtf_road_sort_code(road_mode, road_quality, road_hits):
  return anemtfroadsortcode.get(road_mode, road_quality, road_hits)

# Decide EMTF road mode
class EMTFRoadMode(object):
  def __init__(self):
    self.singlemu = (11,13,14,15)
    self.doublemu = (7,10,12) + self.singlemu
    self.muopen = (3,5,6,9) + self.doublemu

anemtfroadmode = EMTFRoadMode()
def emtf_is_singlemu(mode):
  return mode in anemtfroadmode.singlemu
def emtf_is_doublemu(mode):
  return mode in anemtfroadmode.doublemu
def emtf_is_muopen(mode):
  return mode in anemtfroadmode.muopen

# Extrapolate from paramter space to EMTF space
class EMTFExtrapolation(object):
  def __init__(self):
    self.eta_bins = (13, 1.2, 2.5)
    self.pt_bins = (200, -0.5, 0.5)
    self.loaded = False

  def _find_bin(self, x, bins):
    x = np.clip(x, bins[1], bins[2]-1e-8)
    binx = (x - bins[1]) / (bins[2] - bins[1]) * bins[0]
    return int(binx)

  def _find_eta_bin(self, part):
    x = abs(part.eta)
    return self._find_bin(x, self.eta_bins)

  def _find_pt_bin(self, part):
    x = part.invpt
    return self._find_bin(x, self.pt_bins)

  def get(self, part):
    if not self.loaded:
      with np.load(bankfile) as data:
        self.lut = data['patterns_exphi']
        self.loaded = True
    index = (self._find_eta_bin(part), self._find_pt_bin(part))
    c = self.lut[index]
    dphi = c * (part.invpt * np.sinh(1.8587) / np.sinh(abs(part.eta)))
    exphi = part.phi + dphi  # in radians
    return exphi

anemtfextrapolation = EMTFExtrapolation()
def emtf_extrapolation(part):
  return anemtfextrapolation.get(part)


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

class Pattern(object):
  def __init__(self, xmin, xmed, xmax, ymin, ymed, ymax):
    self.xmin = xmin
    self.xmed = xmed
    self.xmax = xmax
    self.ymin = ymin
    self.ymed = ymed
    self.ymax = ymax

class PatternBank(object):
  def __init__(self, bankfile):
    with np.load(bankfile) as data:
      patterns_phi = data['patterns_phi']
      patterns_theta = data['patterns_theta']
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
  def __init__(self, _id, bx, emtf_layer, emtf_phi, emtf_theta, emtf_bend, time, sim_tp):
    self.id = _id  # (_type, station, ring, fr)
    self.bx = bx
    self.emtf_layer = emtf_layer
    self.emtf_phi = emtf_phi
    self.emtf_theta = emtf_theta
    self.emtf_bend = emtf_bend
    self.time = time
    self.sim_tp = sim_tp

  def get_ring(self):
    return self.id[2]

  def get_fr(self):
    return self.id[3]

class Road(object):
  def __init__(self, _id, hits, mode, mode_csc, quality, sort_code):
    self.id = _id  # (endcap, sector, ipt, ieta, iphi)
    self.hits = hits
    self.mode = mode
    self.mode_csc = mode_csc
    self.quality = quality
    self.sort_code = sort_code

  def to_variables(self):
    amap = {}
    #np.random.shuffle(self.hits)  # randomize the order
    for hit in self.hits:
      hit_lay = hit.emtf_layer
      if hit_lay not in amap:
        amap[hit_lay] = hit
    #
    hits_phi = np.zeros(nlayers, dtype=np.float32) + np.nan
    hits_theta = np.zeros(nlayers, dtype=np.float32) + np.nan
    hits_bend = np.zeros(nlayers, dtype=np.float32) + np.nan
    hits_time = np.zeros(nlayers, dtype=np.float32) + np.nan
    hits_ring = np.zeros(nlayers, dtype=np.float32) + np.nan
    hits_fr = np.zeros(nlayers, dtype=np.float32) + np.nan
    hits_mask = np.zeros(nlayers, dtype=np.float32) + 1.0
    for lay, hit in amap.iteritems():
      hits_phi[lay] = hit.emtf_phi
      hits_theta[lay] = hit.emtf_theta
      hits_bend[lay] = hit.emtf_bend
      hits_time[lay] = hit.bx  #FIXME: use hit.time?
      hits_ring[lay] = hit.get_ring()
      hits_fr[lay] = hit.get_fr()
      hits_mask[lay] = 0.0
    #
    (endcap, sector, ipt, ieta, iphi) = self.id
    road_info = (ipt, ieta, iphi)
    variables = np.hstack((hits_phi, hits_theta, hits_bend, hits_time, hits_ring, hits_fr, hits_mask, road_info))
    return variables

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

  def _apply_patterns(self, endcap, sector, ipt_range, ieta_range, iphi_range, sector_hits):

    # Retrieve patterns with (ipt, ieta, lay, pattern)
    ipt_slice = slice(ipt_range[0], ipt_range[-1]+1, None)
    ieta_slice = slice(ieta_range[0], ieta_range[-1]+1, None)
    pattern_x = self.bank.x_array[ipt_slice, ieta_slice, np.newaxis, :, :]
    pattern_y = self.bank.y_array[ipt_slice, ieta_slice, np.newaxis, :, :]
    pattern_iphi = np.arange(iphi_range[0], iphi_range[-1]+1, dtype=np.int32)

    # Loop over hits
    amap = {}  # road_id -> list of 'myhit'

    for ihit, hit in enumerate(sector_hits):
      # Make hit coordinates
      #hit_x = hit.emtf_phi - (pattern_iphi * 32 - 16)  # multiply by 'quadstrip' unit (4 * 8)
      hit_x = hit.emtf_phi - (pattern_iphi * 16 - 8)  # multiply by 'doublestrip' unit (2 * 8)
      hit_y = hit.emtf_theta
      hit_lay = hit.lay

      # Match patterns
      mask = (pattern_x[...,hit_lay,0] <= hit_x) & (hit_x <= pattern_x[...,hit_lay,2]) & (pattern_y[...,hit_lay,0] <= hit_y) & (hit_y <= pattern_y[...,hit_lay,2])

      # Create a hit (for output)
      hit_id = (hit.type, hit.station, hit.ring, hit.fr)
      hit_sim_tp = (hit.sim_tp1 == 0 and hit.sim_tp2 == 0)
      myhit = Hit(hit_id, hit.bx, hit_lay, hit.emtf_phi, hit.emtf_theta, emtf_bend(hit), hit.time, hit_sim_tp)

      # Associate hits to road ids
      for index, condition in np.ndenumerate(mask):
        if condition:  # good hit
          ipt = ipt_range[index[0]]
          ieta = ieta_range[index[1]]
          iphi = iphi_range[index[2]]
          road_id = (endcap, sector, ipt, ieta, iphi)
          amap.setdefault(road_id, []).append(myhit)  # append hit to road

    # Create a road
    roads = []
    for road_id, road_hits in amap.iteritems():
      # Try BX window (-1,0)
      road_mode = 0
      road_mode_csconly = 0
      tmp_road_hits = []
      for hit in road_hits:
        if hit.bx in (-1,0):
          (_type, station, ring, fr) = hit.id
          road_mode |= (1 << (4 - station))
          if _type == kCSC:
            road_mode_csconly |= (1 << (4 - station))
          tmp_road_hits.append(hit)

      #if not (emtf_is_singlemu(road_mode) and emtf_is_muopen(road_mode_csconly)):
      #  # Try BX window (0,+1)
      #  road_mode = 0
      #  tmp_road_hits = []
      #  for hit in road_hits:
      #    if hit.bx in (0,+1):
      #      (_type, station, ring, fr) = hit.id
      #      road_mode |= (1 << (4 - station))
      #      if _type == kCSC:
      #        road_mode_csconly |= (1 << (4 - station))
      #      tmp_road_hits.append(hit)

      if (emtf_is_singlemu(road_mode) and emtf_is_muopen(road_mode_csconly)):
        (endcap, sector, ipt, ieta, iphi) = road_id
        road_quality = emtf_road_quality(ipt)
        road_sort_code = emtf_road_sort_code(road_mode, road_quality, tmp_road_hits)
        myroad = Road(road_id, tmp_road_hits, road_mode, road_mode_csconly, road_quality, road_sort_code)
        roads.append(myroad)
    return roads

  def run(self, hits, part=None):
    roads = []

    fake_modes = np.zeros(12, dtype=np.int32)  # provide early exit
    for ihit, hit in enumerate(hits):
      if hit.bx in (-1,0):
        hit.endsec = find_endsec(hit.endcap, hit.sector)
        hit.lay = emtf_layer(hit)
        assert(hit.lay != -99)
        if hit.type == kCSC:  # at least 2 CSC hits
          fake_modes[hit.endsec] |= (1 << (4 - hit.station))

    # Loop over sector processors
    for endcap in (-1, +1):
      for sector in (1, 2, 3, 4, 5, 6):
        endsec = find_endsec(endcap, sector)
        fake_mode = fake_modes[endsec]
        early_exit = np.count_nonzero((fake_mode & (1<<3), fake_mode & (1<<2), fake_mode & (1<<1), fake_mode & (1<<0))) < 2  # at least 2 CSC hits
        if early_exit:  continue

        # Patterns to run
        ipt_range = xrange(len(pt_bins))
        ieta_range = xrange(len(eta_bins))
        #iphi_range = xrange(4928/32)  # divide by 'quadstrip' unit (4 * 8)
        iphi_range = xrange(4928/16)  # divide by 'doublestrip' unit (2 * 8)

        # Hits
        sector_hits = [hit for hit in hits if hit.bx in (-1,0) and hit.endsec == endsec]

        # Remove all RPC hits
        #sector_hits = [hit for hit in sector_hits if hit.type != kRPC]

        # Cheat using gen particle info
        if part is not None:
          part.ipt = find_pt_bin(part.invpt)
          part.ieta = find_eta_bin(part.eta)
          if part.ipt != find_pt_bin(0.):  # don't use MC info at the highest pT because of muon showering
            sector_hits = [hit for hit in hits if hit.bx in (-1,0) and hit.endsec == endsec and (hit.sim_tp1 == 0 and hit.sim_tp2 == 0)]
          #ipt_range = [x for x in xrange(part.ipt-1, part.ipt+1+1) if 0 <= x < len(pt_bins)-1]
          ipt_range = xrange(0,(len(pt_bins)-1)//2+1) if part.q < 0 else xrange((len(pt_bins)-1)//2, len(pt_bins)-1)
          ieta_range = [x for x in xrange(part.ieta-1, part.ieta+1+1) if 0 <= x < len(eta_bins)-1]
          tmp_phis = [hit.emtf_phi for hit in sector_hits if hit.type == kCSC and hit.station >= 2]
          if len(tmp_phis) == 0:  continue
          #tmp_phi = np.mean(tmp_phis)
          tmp_phi = np.median(tmp_phis, overwrite_input=True)
          #iphi = int(tmp_phi/32)  # divide by 'quadstrip' unit (4 * 8)
          iphi = int(tmp_phi/16)  # divide by 'doublestrip' unit (2 * 8)
          #iphi_range = xrange(max(0,iphi-12), min(4928/16,iphi+12+1))
          iphi_range = xrange(max(0,iphi-36), min(4928/16,iphi+36+1))

        sector_roads = self._apply_patterns(endcap, sector, ipt_range, ieta_range, iphi_range, sector_hits)
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
          if hit.bx <= -1:
            bx_counter1 += 1
          if hit.bx <= 0:
            bx_counter2 += 1
          if hit.bx > 0:
            bx_counter3 += 1
      #trk_bx_zero = (bx_counter1 < 2 and bx_counter2 >= 2)
      trk_bx_zero = (bx_counter1 < 3 and bx_counter2 >= 2 and bx_counter3 < 2)
      return trk_bx_zero

    def select_two_csc(road):
      layer_has_been_used = set()
      for hit in road.hits:
        if hit.emtf_layer not in layer_has_been_used:
          layer_has_been_used.add(hit.emtf_layer)
      count = 0
      for lay in layer_has_been_used:
        if lay in [0,1,2,3,4]:
          count += 1
      trk_two_csc = (count >= 2)
      return trk_two_csc

    clean_roads = filter(select_bx_zero, clean_roads)
    #clean_roads = filter(select_two_csc, clean_roads)

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
          if (_get_endsec(road.id) == _get_endsec(road_to_check.id)) and (gi[1]+4 >= gj[0]) and (gi[0]-4 <= gj[1]):
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
      for index in self._iter_from_middle(range(len(group))):
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
      road_id = road.id
      ipt, ieta, iphi = road_id[2:]
      if ieta <= 4:
        pairings = dict([(0,2), (1,2), (2,0), (3,0), (4,0), (5,0), (6,2), (7,3), (8,4), (9,0), (10,2), (11,0)])
      else:
        pairings = dict([(0,2), (1,2), (2,1), (3,1), (4,1), (5,1), (6,2), (7,3), (8,4), (9,1), (10,2), (11,1)])

      #tmp_phi = (iphi * 32 - 16)  # multiply by 'quadstrip' unit (4 * 8)
      tmp_phi = (iphi * 16 - 8)  # multiply by 'doublestrip' unit (2 * 8)

      _select_csc = lambda x: (x.emtf_layer <= 4)
      tmp_thetas = [hit.emtf_theta for hit in road.hits if _select_csc(hit)]
      tmp_theta = np.median(tmp_thetas, overwrite_input=True)

      hits_array = np.empty((nlayers,), dtype=np.object)
      for ind in np.ndindex(hits_array.shape):
        hits_array[ind] = []

      best_phi_array = np.zeros((nlayers,), dtype=np.float32) + tmp_phi
      best_theta_array = np.zeros((nlayers,), dtype=np.float32) + tmp_theta

      for hit in road.hits:
        hit_lay = hit.emtf_layer
        hits_array[hit_lay].append(hit)

      # Assume going through ME1, ME2, ... in order
      for hit_lay in xrange(nlayers):
        mean_dphi = self.bank.z_array[ipt, ieta, hit_lay, 1]
        hit_lay_p = pairings[hit_lay]
        if hit_lay != 0 and hit_lay != 1:
          assert(hit_lay > hit_lay_p)

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
          best_theta_array[hit_lay] = best_theta_array[hit_lay_p]


      slim_road_hits = []
      for hit_lay in xrange(nlayers):
        if hits_array[hit_lay]:
          assert(len(hits_array[hit_lay]) == 1)
          slim_road_hits.append(hits_array[hit_lay][0])

      slim_road = Road(road.id, slim_road_hits, road.mode, road.mode_csc, road.quality, road.sort_code)
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

    def create_encoder(x):
      nentries = x.shape[0]
      dummy = np.zeros((nentries, 3), dtype=np.float32)
      encoder = Encoder(x, dummy, adjust_scale=adjust_scale, reg_pt_scale=reg_pt_scale)
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

    ndof = (encoder.get_x_mask() == False).sum(axis=1)  # num of hits
    ndof = ndof[:, np.newaxis]
    return (x_new, y, ndof)


# Track producer module
class TrackProducer(object):
  def __init__(self):
    self.s_min = 0.
    self.s_max = 60.
    self.s_nbins = 120
    self.s_step = (self.s_max - self.s_min)/self.s_nbins
    self.s_lut =[ 1.7735,  1.4907,  1.5448,  1.7995,  2.1694,  2.6076,  3.1008,  3.6336,
                  4.1887,  4.7578,  5.3386,  5.9338,  6.5453,  7.1688,  7.8041,  8.4512,
                  9.1112,  9.7853, 10.4714, 11.1732, 11.8939, 12.6312, 13.3879, 14.1624,
                 14.9124, 15.6381, 16.3544, 17.0820, 17.8456, 18.6531, 19.4701, 20.2475,
                 20.9810, 21.7350, 22.5501, 23.3048, 23.9189, 24.4705, 25.0511, 25.7179,
                 26.4861, 27.3111, 28.1933, 29.0985, 30.0049, 30.9373, 31.8868, 32.9006,
                 34.0259, 35.1165, 36.1163, 37.1042, 38.1683, 39.2137, 40.2477, 41.2547,
                 42.2844, 43.4362, 44.8745, 46.4813, 48.3087, 49.6675, 50.5817, 51.4416,
                 52.2907, 53.1361, 53.9798, 54.8228, 55.6652, 56.5074, 57.3494, 58.1913,
                 59.0331, 59.8748, 60.7165, 61.5581, 62.3997, 63.2413, 64.0829, 64.9244,
                 65.7659, 66.6075, 67.4490, 68.2905, 69.1320, 69.9735, 70.8150, 71.6565,
                 72.4980, 73.3395, 74.1810, 75.0225, 75.8640, 76.7055, 77.5470, 78.3885,
                 79.2299, 80.0714, 80.9129, 81.7544, 82.5959, 83.4374, 84.2788, 85.1203,
                 85.9618, 86.8033, 87.6448, 88.4862, 89.3277, 90.1692, 91.0107, 91.8522,
                 92.6936, 93.5351, 94.3766, 95.2181, 96.0596, 96.9010, 97.7425, 98.5840]
    #self.s_lut = np.linspace(self.s_min, self.s_max, num=self.s_nbins+1)[:-1]

  def run(self, slim_roads, variables, predictions, other_vars):
    assert(len(slim_roads) == len(variables))
    assert(len(slim_roads) == len(predictions))
    assert(len(slim_roads) == len(other_vars))

    discr_pt_cut = 14.

    tracks = []

    for myroad, myvars, mypreds, myother in izip(slim_roads, variables, predictions, other_vars):
      # Unpack variables
      assert(len(myvars.shape) == 1)
      assert(myvars.shape[0] == (nlayers * 6) + 8)

      x = myvars
      ndof = np.asscalar(myother)
      y_meas = np.asscalar(mypreds[...,0])
      y_discr = np.asscalar(mypreds[...,1])

      passed = self.pass_trigger(x, ndof, y_meas, y_discr, discr_pt_cut=discr_pt_cut)
      trk_xml_pt = np.abs(1.0/y_meas)
      trk_pt = self.get_trigger_pt(x, y_meas)

      if passed:
        trk_q = np.sign(y_meas)
        trk_mode = 0
        x_mode_vars = np.equal(x[nlayers*6+3:nlayers*6+8], 1)
        for i, x_mode_var in enumerate(x_mode_vars):
          if i == 0:
            station = 1
          else:
            station = i
          if x_mode_var:
            trk_mode |= (1 << (4 - station))

        trk_emtf_phi = myroad.id[4]
        trk_emtf_theta = int(x[(nlayers*6) + 2] * 83) + 3

        trk = Track(myroad.id, myroad.hits, trk_mode, trk_xml_pt, trk_pt, trk_q, trk_emtf_phi, trk_emtf_theta, ndof, y_discr)
        tracks.append(trk)
    return tracks

  def get_trigger_pt(self, x, y_meas):
    #zone = int(x[(nlayers*6) + 1] * 6)

    xml_pt = np.abs(1.0/y_meas)
    if xml_pt <= 2.:
      return xml_pt

    def digitize(x, bins=(self.s_nbins, self.s_min, self.s_max)):
      x = np.clip(x, bins[1], bins[2]-1e-8)
      binx = (x - bins[1]) / (bins[2] - bins[1]) * bins[0]
      return int(binx)

    def interpolate(x, x0, x1, y0, y1):
      y = (x - x0) / (x1 - x0) * (y1 - y0) + y0
      return y

    binx = digitize(xml_pt)
    if (binx+1) >= self.s_nbins:  # check boundary
      binx = self.s_nbins-2

    x0, x1 = binx * self.s_step, (binx+1) * self.s_step
    y0, y1 = self.s_lut[binx], self.s_lut[binx+1]
    pt = interpolate(xml_pt, x0, x1, y0, y1)
    return pt

  def pass_trigger(self, x, ndof, y_meas, y_discr, discr_pt_cut=14.):
    trk_mode = 0
    x_mode_vars = np.equal(x[nlayers*6+3:nlayers*6+8], 1)
    for i, x_mode_var in enumerate(x_mode_vars):
      if i == 0:
        station = 1
      else:
        station = i
      if x_mode_var:
        trk_mode |= (1 << (4 - station))

    straightness = int(x[(nlayers*6) + 0] * 6) + 6

    ipt1 = straightness
    ipt2 = find_pt_bin(y_meas)
    quality1 = emtf_road_quality(ipt1)
    quality2 = emtf_road_quality(ipt2)

    if trk_mode in (11,13,14,15) and quality2 <= (quality1+1):
      if np.abs(1.0/y_meas) > discr_pt_cut:
        if ndof <= 3:
          #trigger = (y_discr > 0.8)
          trigger = (y_discr > 0.9945)  # 90% coverage
          #trigger = (y_discr > 0.9999)  # 95% coverage
        else:
          #trigger = (y_discr > 0.5393)
          trigger = (y_discr > 0.9503) # 98.5% coverage
          #trigger = (y_discr > 0.9929) # 99% coverage
      else:
        trigger = (y_discr >= 0.)  # True
    else:
      trigger = (y_discr < 0.)  # False
    return trigger


# ______________________________________________________________________________
# Book histograms
histograms = {}

# Efficiency
#eff_pt_bins = (0., 0.5, 1., 2., 3., 4., 5., 6., 8., 10., 12., 14., 16., 18., 20., 22., 24., 26., 28., 30., 35., 40., 45., 50., 60., 80., 120.)
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
  hname = "highest_%s_absEtaMin0_absEtaMax2.5_qmin12_pt" % m
  histograms[hname] = Hist(100, 0., 100., name=hname, title="; p_{T} [GeV]; entries", type='F')
  hname = "highest_%s_absEtaMin1.2_absEtaMax1.65_qmin12_pt" % m
  histograms[hname] = Hist(100, 0., 100., name=hname, title="; p_{T} [GeV]; entries", type='F')
  hname = "highest_%s_absEtaMin1.65_absEtaMax2.15_qmin12_pt" % m
  histograms[hname] = Hist(100, 0., 100., name=hname, title="; p_{T} [GeV]; entries", type='F')
  hname = "highest_%s_absEtaMin2.15_absEtaMax2.5_qmin12_pt" % m
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
      histograms[hname] = Hist(70, 1.1, 2.5, name=hname, title="; gen |#eta| {gen p_{T} > %i GeV}" % (l), type='F')

  hname = "%s_l1pt_vs_genpt" % m
  histograms[hname] = Hist2D(100, -0.5, 0.5, 300, -0.5, 0.5, name=hname, title="; gen 1/p_{T} [1/GeV]; 1/p_{T} [1/GeV]", type='F')
  hname = "%s_l1ptres_vs_genpt" % m
  histograms[hname] = Hist2D(100, -0.5, 0.5, 300, -2, 2, name=hname, title="; gen 1/p_{T} [1/GeV]; #Delta(p_{T})/p_{T}", type='F')


# ______________________________________________________________________________
# Settings

# Get number of events
#maxEvents = -1
maxEvents = 4000000
#maxEvents = 1000

# Condor or not
use_condor = ('CONDOR_EXEC' in os.environ)

# Analysis mode
#analysis = 'verbose'
#analysis = 'training'
#analysis = 'application'
#analysis = 'rates'
#analysis = 'effie'
#analysis = 'mixing'
analysis = 'images'
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
bankfile = 'histos_tb.17.npz'

kerasfile = ['model.17.json', 'model_weights.17.h5']

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
  return tree

def load_pgun_batch(j):
  global infile_r
  infile_r = root_open('pippo.root', 'w')

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
  return tree

def load_minbias_batch(j):
  global infile_r
  pufiles = ['root://cmsxrootd-site.fnal.gov//store/group/l1upgrades/L1MuonTrigger/P2_10_1_5/ntuple_SingleNeutrino_PU200/ParticleGuns/CRAB3/180813_210720/0000/ntuple_SingleNeutrino_PU200_%i.root' % (i+1) for i in xrange(63)]
  infile = pufiles[j]
  infile_r = root_open(infile)
  tree = infile_r.ntupler.tree
  print('[INFO] Opening file: %s' % infile)

  # Define collection
  tree.define_collection(name='hits', prefix='vh_', size='vh_size')
  tree.define_collection(name='tracks', prefix='vt_', size='vt_size')
  tree.define_collection(name='particles', prefix='vp_', size='vp_size')
  return tree

def unload_tree():
  global infile_r
  infile_r.close()


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
# Analysis: training
elif analysis == 'training':
  tree = load_pgun()

  # 3-dimensional arrays of lists
  # [ipt][ieta][lay]
  patterns_phi = np.empty((len(pt_bins)-1, len(eta_bins)-1, nlayers), dtype=np.object)
  patterns_theta = np.empty((len(pt_bins)-1, len(eta_bins)-1, nlayers), dtype=np.object)
  patterns_match = np.empty((len(pt_bins)-1, len(eta_bins)-1, nlayers), dtype=np.object)  # used for matching
  for ind in np.ndindex(patterns_phi.shape):
    patterns_phi[ind] = []
    patterns_theta[ind] = []
    patterns_match[ind] = []

  # 2-dimensional arrays of lists
  # [ieta][ipt]
  e = EMTFExtrapolation()
  patterns_exphi = np.empty((e.eta_bins[0], e.pt_bins[0]), dtype=np.object)
  for ind in np.ndindex(patterns_exphi.shape):
    patterns_exphi[ind] = []

  # ____________________________________________________________________________
  # Loop over events
  for ievt, evt in enumerate(tree):
    if maxEvents != -1 and ievt == maxEvents:
      break

    if (ievt % 1000 == 0):  print("Processing event: {0}".format(ievt))

    part = evt.particles[0]  # particle gun
    part.invpt = np.true_divide(part.q, part.pt)
    #part.exphi = extrapolate_to_emtf(part.phi, part.invpt, part.eta)
    part.exphi = emtf_extrapolation(part)
    part.sector = find_sector(part.exphi)
    part.endcap = find_endcap(part.eta)
    #part.emtf_phi = calc_phi_loc_int(np.rad2deg(part.exphi), part.sector)
    part.emtf_phi = calc_phi_loc_int_1(np.rad2deg(part.exphi), part.sector)  # no cast to int
    part.emtf_theta = calc_theta_int(calc_theta_deg_from_eta(part.eta), part.endcap)

    if ievt < 20:
      print("evt {0} has {1} particles and {2} hits".format(ievt, len(evt.particles), len(evt.hits)))
      print(".. part invpt: {0} pt: {1} eta: {2} phi: {3} exphi: {4} sec: {5} ph: {6} th: {7}".format(part.invpt, part.pt, part.eta, part.phi, part.exphi, part.sector, part.emtf_phi, part.emtf_theta))

    part.ipt = find_pt_bin(part.invpt)
    part.ieta = find_eta_bin(part.eta)
    the_patterns_phi = patterns_phi[part.ipt,part.ieta]
    the_patterns_theta = patterns_theta[0,part.ieta]  # no binning in pt
    the_patterns_match = patterns_match[part.ipt,part.ieta]
    e = EMTFExtrapolation()
    the_patterns_exphi = patterns_exphi[e._find_eta_bin(part), e._find_pt_bin(part)]

    # Loop over hits
    cached_hits = {}  # used for matching

    for ihit, hit in enumerate(evt.hits):
      hit_lay = emtf_layer(hit)
      hit.lay = hit_lay
      assert(hit_lay != -99)
      if ievt < 20:
        print(".. hit {0} type: {1} st: {2} ri: {3} fr: {4} lay: {5} sec: {6} ph: {7} th: {8}".format(ihit, hit.type, hit.station, hit.ring, hit.fr, hit_lay, hit.sector, hit.emtf_phi, hit.emtf_theta))

      if hit.endcap == part.endcap and hit.sector == part.sector and (hit.sim_tp1 == 0 and hit.sim_tp2 == 0):
        the_patterns_phi[hit_lay].append(hit.emtf_phi - part.emtf_phi)
        if part.pt >= 4.:  # use >=4 GeV muons for theta windows
          the_patterns_theta[hit_lay].append(hit.emtf_theta)
        if hit_lay not in cached_hits:
          cached_hits[hit_lay] = hit

        if hit.type == kCSC and hit.station == 2:  # extrapolation to EMTF using ME2
          dphi = delta_phi(np.deg2rad(hit.sim_phi), part.phi)
          dphi /= (part.invpt * np.sinh(1.8587) / np.sinh(abs(part.eta)))
          the_patterns_exphi.append(dphi)

    # Find pairs of hits
    if part.ieta <= 4:
      pairings = dict([(0,2), (1,2), (2,0), (3,0), (4,0), (5,0), (6,2), (7,3), (8,4), (9,0), (10,2), (11,0)])
    else:
      pairings = dict([(0,2), (1,2), (2,1), (3,1), (4,1), (5,1), (6,2), (7,3), (8,4), (9,1), (10,2), (11,1)])

    for ihit, hit in enumerate(evt.hits):
      if hit.endcap == part.endcap and hit.sector == part.sector and (hit.sim_tp1 == 0 and hit.sim_tp2 == 0):
        hit_lay = hit.lay
        hit_lay_p = pairings[hit_lay]
        if hit_lay_p in cached_hits:
          hit_p = cached_hits[hit_lay_p]
          the_patterns_match[hit_lay].append(hit.emtf_phi - hit_p.emtf_phi)

  # End loop over events
  unload_tree()

  # ____________________________________________________________________________
  # Plot histograms
  print('[INFO] Creating file: histos_tb.root')
  with root_open('histos_tb.root', 'recreate') as f:
    for i in xrange(len(pt_bins)-1):
      for j in xrange(len(eta_bins)-1):
        for k in xrange(nlayers):
          hname = "patterns_phi_%i_%i_%i" % (i,j,k)
          h1a = Hist(201, -402, 402, name=hname, title=hname, type='F')
          for x in patterns_phi[i,j,k]:  h1a.fill(x)
          h1a.Write()

          hname = "patterns_theta_%i_%i_%i" % (i,j,k)
          h1b = Hist(81, -40.5, 40.5, name=hname, title=hname, type='F')
          for x in patterns_theta[i,j,k]:  h1b.fill(x)
          h1b.Write()

  # ____________________________________________________________________________
  # Save objects
  print('[INFO] Creating file: histos_tb.npz')
  if True:
    patterns_phi_tmp = patterns_phi
    patterns_theta_tmp = patterns_theta
    patterns_match_tmp = patterns_match
    patterns_phi = np.zeros((len(pt_bins)-1, len(eta_bins)-1, nlayers, 3), dtype=np.int32)
    patterns_theta = np.zeros((len(pt_bins)-1, len(eta_bins)-1, nlayers, 3), dtype=np.int32)
    patterns_match = np.zeros((len(pt_bins)-1, len(eta_bins)-1, nlayers, 3), dtype=np.int32)
    #
    for i in xrange(len(pt_bins)-1):
      for j in xrange(len(eta_bins)-1):
        for k in xrange(nlayers):
          patterns_phi_tmp_ijk = patterns_phi_tmp[i,j,k]
          if len(patterns_phi_tmp_ijk) > 1000:
            if k == 9 or k == 10 or k == 11:  # keep more GEMs
              x = np.percentile(patterns_phi_tmp_ijk, [3.5, 50, 96.5], overwrite_input=True)
            else:
              x = np.percentile(patterns_phi_tmp_ijk, [5, 50, 95], overwrite_input=True)
            x = [int(round(xx)) for xx in x]
            if (x[2] - x[0]) < 32:
              old_x = x[:]
              while (x[2] - x[0]) < 32:  # make sure the range is larger than twice the 'doublestrip' unit
                x[0] -= 1
                x[2] += 1
              print(".. phi (%i,%i,%i) expanded from [%i,%i] to [%i,%i]" % (i,j,k,old_x[0],old_x[2],x[0],x[2]))
            patterns_phi[i,j,k] = x

          #patterns_theta_tmp_ijk = patterns_theta_tmp[i,j,k]
          patterns_theta_tmp_ijk = patterns_theta_tmp[0,j,k]  # no binning in pt
          if len(patterns_theta_tmp_ijk) > 1000:
            x = np.percentile(patterns_theta_tmp_ijk, [2.5, 50, 97.5], overwrite_input=True)
            x = [int(round(xx)) for xx in x]
            patterns_theta[i,j,k] = x

          patterns_match_tmp_ijk = patterns_match_tmp[i,j,k]
          if len(patterns_match_tmp_ijk) > 1000:
            x = np.percentile(patterns_match_tmp_ijk, [5, 50, 95], overwrite_input=True)
            x = [int(round(xx)) for xx in x]
            patterns_match[i,j,k] = x


    # Mask layers by (ieta, lay)
    valid_layers = [
      (6,1), (6,2), (6,3), (6,4), (6,5), (6,6), (6,7), (6,8),
      (5,1), (5,2), (5,3), (5,4), (5,5), (5,6), (5,7), (5,8),
      (4,0), (4,1), (4,2), (4,3), (4,4), (4,7), (4,8), (4,9), (4,10),
      (3,0), (3,2), (3,3), (3,4), (3,7), (3,8), (3,9), (3,10),
      (2,0), (2,2), (2,3), (2,4), (2,7), (2,8), (2,9), (2,10),
      (1,0), (1,2), (1,3), (1,4), (1,7), (1,8), (1,9), (1,10), (1,11),
      (0,0), (0,2), (0,3), (0,4), (0,7), (0,8), (0,10), (0,11),
    ]
    mask = np.ones_like(patterns_phi, dtype=np.bool)
    for valid_layer in valid_layers:
      mask[:,valid_layer[0],valid_layer[1],:] = False
    patterns_phi[mask] = 0
    patterns_theta[mask] = 0
    patterns_match[mask] = 0

    # extrapolation to EMTF using ME2
    overwrite_extrapolation = True
    smooth_extrapolation = True
    if overwrite_extrapolation:
      patterns_exphi_tmp = patterns_exphi
      patterns_exphi = np.zeros(patterns_exphi_tmp.shape, dtype=np.float32)
      for index, x in np.ndenumerate(patterns_exphi_tmp):
        if x:
          patterns_exphi[index] = np.median(x, overwrite_input=True)
      if smooth_extrapolation:
        from scipy.interpolate import Rbf
        patterns_exphi_tmp = patterns_exphi
        patterns_exphi = np.zeros_like(patterns_exphi_tmp, dtype=np.float32)
        e = EMTFExtrapolation()
        x = [e.pt_bins[1] + (i+0.5)/e.pt_bins[0]*(e.pt_bins[2] - e.pt_bins[1]) for i in xrange(e.pt_bins[0])]
        for index in np.ndindex(e.eta_bins[0]):
          assert(len(x) == len(patterns_exphi_tmp[index]))
          rbf = Rbf(x, patterns_exphi_tmp[index], smooth = 0.3, function='multiquadric')
          patterns_exphi[index] = rbf(x)
    else:
      with np.load(bankfile) as data:
        patterns_exphi = data['patterns_exphi']

    outfile = 'histos_tb.npz'
    np.savez_compressed(outfile, patterns_phi=patterns_phi, patterns_theta=patterns_theta, patterns_match=patterns_match, patterns_exphi=patterns_exphi)




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

    part = evt.particles[0]  # particle gun
    part.invpt = np.true_divide(part.q, part.pt)

    ## Cheat using gen particle info
    #roads = recog.run(evt.hits, part)
    #clean_roads = clean.run(roads)

    roads = recog.run(evt.hits)
    clean_roads = clean.run(roads)
    slim_roads = slim.run(clean_roads)
    assert(len(clean_roads) == len(slim_roads))

    if len(slim_roads) > 0:
      mypart = Particle(part.pt, part.eta, part.phi, part.q)
      out_particles.append(mypart)
      out_roads.append(slim_roads[0])

    if ievt < 20 or len(clean_roads) == 0:
      print("evt {0} has {1} roads and {2} clean roads".format(ievt, len(roads), len(clean_roads)))
      print(".. part invpt: {0} pt: {1} eta: {2} phi: {3}".format(part.invpt, part.pt, part.eta, part.phi))
      part.ipt = find_pt_bin(part.invpt)
      part.ieta = find_eta_bin(part.eta)
      part.exphi = emtf_extrapolation(part)
      part.sector = find_sector(part.exphi)
      part.endcap = find_endcap(part.eta)
      part.emtf_phi = calc_phi_loc_int(np.rad2deg(part.exphi), part.sector)
      part.emtf_theta = calc_theta_int(calc_theta_deg_from_eta(part.eta), part.endcap)
      part_road_id = (part.endcap, part.sector, part.ipt, part.ieta, part.emtf_phi/16)
      part_nhits = sum([1 for hit in evt.hits if hit.endcap == part.endcap and hit.sector == part.sector])
      print(".. part road id: {0} nhits: {1} exphi: {2} emtf_phi: {3}".format(part_road_id, part_nhits, part.exphi, part.emtf_phi))
      for ihit, hit in enumerate(evt.hits):
        if hit.endcap == part.endcap and hit.sector == part.sector:
          hit_id = (hit.type, hit.station, hit.ring, hit.fr)
          print(".. .. hit {0} id: {1} lay: {2} ph: {3} th: {4} bx: {5} tp: {6}/{7}".format(ihit, hit_id, emtf_layer(hit), hit.emtf_phi, hit.emtf_theta, hit.bx, hit.sim_tp1, hit.sim_tp2))
      for iroad, myroad in enumerate(sorted(roads, key=lambda x: x.id)):
        print(".. road {0} id: {1} nhits: {2} mode: {3} qual: {4} sort: {5}".format(iroad, myroad.id, len(myroad.hits), myroad.mode, myroad.quality, myroad.sort_code))
      for iroad, myroad in enumerate(clean_roads):
        print(".. croad {0} id: {1} nhits: {2} mode: {3} qual: {4} sort: {5}".format(iroad, myroad.id, len(myroad.hits), myroad.mode, myroad.quality, myroad.sort_code))
        for ihit, myhit in enumerate(myroad.hits):
          print(".. .. hit {0} id: {1} lay: {2} ph: {3} th: {4} bx: {5} tp: {6}".format(ihit, myhit.id, myhit.emtf_layer, myhit.emtf_phi, myhit.emtf_theta, myhit.bx, myhit.sim_tp))
      for iroad, myroad in enumerate(slim_roads):
        print(".. sroad {0} id: {1} nhits: {2} mode: {3} qual: {4} sort: {5}".format(iroad, myroad.id, len(myroad.hits), myroad.mode, myroad.quality, myroad.sort_code))
        for ihit, myhit in enumerate(myroad.hits):
          print(".. .. hit {0} id: {1} lay: {2} ph: {3} th: {4} bx: {5} tp: {6}".format(ihit, myhit.id, myhit.emtf_layer, myhit.emtf_phi, myhit.emtf_theta, myhit.bx, myhit.sim_tp))

    # Quick efficiency
    if (1.24 <= abs(part.eta) <= 2.4) and (part.bx == 0) and part.pt > 5.:
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
  print('[INFO] Creating file: histos_tba.root')
  with root_open('histos_tba.root', 'recreate') as f:
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
  print('[INFO] Creating file: histos_tba.npz')
  if True:
    assert(len(out_particles) == len(out_roads))
    parameters = particles_to_parameters(out_particles)
    variables = roads_to_variables(out_roads)
    outfile = 'histos_tba.npz'
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
        #  print(".. .. hit {0} id: {1} lay: {2} ph: {3} th: {4} bx: {5} tp: {6}".format(ihit, myhit.id, myhit.emtf_layer, myhit.emtf_phi, myhit.emtf_theta, myhit.bx, myhit.sim_tp))
      for itrk, mytrk in enumerate(emtf2023_tracks):
        print(".. trk {0} id: {1} nhits: {2} mode: {3} pt: {4} ndof: {5} chi2: {6}".format(itrk, mytrk.id, len(mytrk.hits), mytrk.mode, mytrk.pt, mytrk.ndof, mytrk.chi2))
        for ihit, myhit in enumerate(mytrk.hits):
          print(".. .. hit {0} id: {1} lay: {2} ph: {3} th: {4} bx: {5} tp: {6}".format(ihit, myhit.id, myhit.emtf_layer, myhit.emtf_phi, myhit.emtf_theta, myhit.bx, myhit.sim_tp))
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
    select = lambda trk: trk and (0. <= abs(trk.eta) <= 2.5) and (trk.bx == 0) and (trk.mode in (11,13,14,15))
    hname = "highest_emtf_absEtaMin0_absEtaMax2.5_qmin12_pt"
    fill_highest_pt()
    select = lambda trk: trk and (1.2 <= abs(trk.eta) < 1.65) and (trk.bx == 0) and (trk.mode in (11,13,14,15))
    hname = "highest_emtf_absEtaMin1.2_absEtaMax1.65_qmin12_pt"
    fill_highest_pt()
    select = lambda trk: trk and (1.65 <= abs(trk.eta) < 2.15) and (trk.bx == 0) and (trk.mode in (11,13,14,15))
    hname = "highest_emtf_absEtaMin1.65_absEtaMax2.15_qmin12_pt"
    fill_highest_pt()
    select = lambda trk: trk and (2.15 <= abs(trk.eta) <= 2.5) and (trk.bx == 0) and (trk.mode in (11,13,14,15))
    hname = "highest_emtf_absEtaMin2.15_absEtaMax2.5_qmin12_pt"
    fill_highest_pt()
    for l in xrange(14,22+1):
      select = lambda trk: trk and (0. <= abs(trk.eta) <= 2.5) and (trk.bx == 0) and (trk.mode in (11,13,14,15)) and (trk.pt > float(l))
      hname = "emtf_ptmin%i_qmin12_eta" % (l)
      fill_eta()

    tracks = emtf2023_tracks
    #
    select = lambda trk: trk and (0. <= abs(trk.eta) <= 2.5)
    hname = "highest_emtf2023_absEtaMin0_absEtaMax2.5_qmin12_pt"
    fill_highest_pt()
    select = lambda trk: trk and (1.2 <= abs(trk.eta) < 1.65)
    hname = "highest_emtf2023_absEtaMin1.2_absEtaMax1.65_qmin12_pt"
    fill_highest_pt()
    select = lambda trk: trk and (1.65 <= abs(trk.eta) < 2.15)
    hname = "highest_emtf2023_absEtaMin1.65_absEtaMax2.15_qmin12_pt"
    fill_highest_pt()
    select = lambda trk: trk and (2.15 <= abs(trk.eta) <= 2.5)
    hname = "highest_emtf2023_absEtaMin2.15_absEtaMax2.5_qmin12_pt"
    fill_highest_pt()
    for l in xrange(14,22+1):
      select = lambda trk: trk and (0. <= abs(trk.eta) <= 2.5) and (trk.pt > float(l))
      hname = "emtf2023_ptmin%i_qmin12_eta" % (l)
      fill_eta()

  # End loop over events
  unload_tree()

  # ____________________________________________________________________________
  # Plot histograms
  print('[INFO] Creating file: histos_tbb.root')
  with root_open('histos_tbb.root', 'recreate') as f:
    hnames = []
    hname = "nevents"
    hnames.append("nevents")
    for m in ("emtf", "emtf2023"):
      hname = "highest_%s_absEtaMin0_absEtaMax2.5_qmin12_pt" % m
      hnames.append(hname)
      hname = "highest_%s_absEtaMin1.2_absEtaMax1.65_qmin12_pt" % m
      hnames.append(hname)
      hname = "highest_%s_absEtaMin1.65_absEtaMax2.15_qmin12_pt" % m
      hnames.append(hname)
      hname = "highest_%s_absEtaMin2.15_absEtaMax2.5_qmin12_pt" % m
      hnames.append(hname)
      for l in xrange(14,22+1):
        hname = "%s_ptmin%i_qmin12_eta" % (m,l)
        hnames.append(hname)
    for hname in hnames:
      h = histograms[hname]
      h.Write()

  # ____________________________________________________________________________
  # Save objects
  #print('[INFO] Creating file: histos_tbb.npz')
  #if True:
  #  variables = np.vstack(out_variables)
  #  predictions = np.vstack(out_predictions)
  #  outfile = 'histos_tbb.npz'
  #  np.savez_compressed(outfile, variables=variables, predictions=predictions)




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
      select = lambda trk: trk and (0. <= abs(trk.eta) <= 2.5) and (trk.mode in (11,13,14,15)) and (trk.pt > float(l))
      tracks = evt.tracks
      #
      hname = "emtf_eff_vs_genpt_l1pt%i" % (l)
      fill_efficiency_pt()
      if part.pt > 20.:
        hname = "emtf_eff_vs_geneta_l1pt%i" % (l)
        fill_efficiency_eta()
      if l == 0:
        hname1 = "emtf_l1pt_vs_genpt"
        hname2 = "emtf_l1ptres_vs_genpt"
        fill_resolution()

      select = lambda trk: trk and (0. <= abs(trk.eta) <= 2.5) and (trk.pt > float(l))
      tracks = emtf2023_tracks
      #
      hname = "emtf2023_eff_vs_genpt_l1pt%i" % (l)
      fill_efficiency_pt()
      if part.pt > 20.:
        hname = "emtf2023_eff_vs_geneta_l1pt%i" % (l)
        fill_efficiency_eta()
      if l == 0:
        hname1 = "emtf2023_l1pt_vs_genpt"
        hname2 = "emtf2023_l1ptres_vs_genpt"
        fill_resolution()

  # End loop over events
  unload_tree()

  # ____________________________________________________________________________
  # Plot histograms
  print('[INFO] Creating file: histos_tbc.root')
  with root_open('histos_tbc.root', 'recreate') as f:
    hnames = []
    for m in ("emtf", "emtf2023"):
      for l in (0, 10, 15, 20, 30, 40, 50):
        for k in ("denom", "numer"):
          hname = "%s_eff_vs_genpt_l1pt%i_%s" % (m,l,k)
          hnames.append(hname)
          hname = "%s_eff_vs_geneta_l1pt%i_%s" % (m,l,k)
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
  tree = load_minbias_batch(jobid)

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
      highest_pt = np.full((12,), -999999., dtype=np.float32)
      for ipart, part in enumerate(evt.particles):
        if select(part):
          part.invpt = np.true_divide(part.q, part.pt)
          #part.exphi = extrapolate_to_emtf(part.phi, part.invpt, part.eta)
          part.exphi = emtf_extrapolation(part)
          part.sector = find_sector(part.exphi)
          part.endcap = find_endcap(part.eta)
          endsec = find_endsec(part.endcap, part.sector)
          if highest_pt[endsec] < part.pt:
            highest_pt[endsec] = part.pt
      #highest_pt = np.clip(highest_pt, -999999., 100.-1e-3)
      return highest_pt

    select = lambda part: (1.24 <= abs(part.eta) <= 2.4) and (part.bx == 0)
    highest_part_pt = find_highest_part_pt()

    def find_highest_track_pt():
      highest_pt = np.full((12,), -999999., dtype=np.float32)
      for itrk, trk in enumerate(evt.tracks):
        if select(trk):
          endsec = find_endsec(trk.endcap, trk.sector)
          if highest_pt[endsec] < trk.pt:  # using scaled pT
            highest_pt[endsec] = trk.pt
      #highest_pt = np.clip(highest_pt, -999999., 100.-1e-3)
      return highest_pt

    select = lambda trk: trk and (0. <= abs(trk.eta) <= 2.5) and (trk.bx == 0) and (trk.mode in (11,13,14,15))
    highest_track_pt = find_highest_track_pt()

    def find_aux_info(road):
      endsec = find_endsec(road.id[0], road.id[1])
      aux = (jobid, ievt, highest_part_pt[endsec], highest_track_pt[endsec])
      return aux

    if len(slim_roads) > 0:
      out_particles += [find_aux_info(road) for road in slim_roads]
      out_roads += slim_roads

    debug_event_list = [2826, 2937, 3675, 4581, 4838, 5379, 7640]

    if ievt < 20 or ievt in debug_event_list:
      print("evt {0} has {1} roads, {2} clean roads, {3} old tracks, {4} new tracks".format(ievt, len(roads), len(clean_roads), len(evt.tracks), '?'))
      #for iroad, myroad in enumerate(sorted(roads, key=lambda x: x.id)):
      #  print(".. road {0} id: {1} nhits: {2} mode: {3} qual: {4} sort: {5}".format(iroad, myroad.id, len(myroad.hits), myroad.mode, myroad.quality, myroad.sort_code))
      for iroad, myroad in enumerate(clean_roads):
        print(".. croad {0} id: {1} nhits: {2} mode: {3} qual: {4} sort: {5}".format(iroad, myroad.id, len(myroad.hits), myroad.mode, myroad.quality, myroad.sort_code))
        for ihit, myhit in enumerate(myroad.hits):
          print(".. .. hit {0} id: {1} lay: {2} ph: {3} th: {4} bx: {5} tp: {6}".format(ihit, myhit.id, myhit.emtf_layer, myhit.emtf_phi, myhit.emtf_theta, myhit.bx, myhit.sim_tp))

  # End loop over events
  unload_tree()

  # ____________________________________________________________________________
  # Save objects
  print('[INFO] Creating file: histos_tbd.npz')
  if True:
    assert(len(out_roads) == len(out_particles))
    variables = roads_to_variables(out_roads)
    aux = np.array(out_particles, dtype=np.float32)
    outfile = 'histos_tbd.npz'
    np.savez_compressed(outfile, variables=variables, aux=aux)




# ______________________________________________________________________________
# Analysis: images
elif analysis == 'images':

  # Make EMTF image
  # A m-by-n matrix has m rows and n columns
  class EMTFImage(object):
    def __init__(self, superstrip_size):
      self.superstrip_size = superstrip_size

      self.sector_hits_capacity = 50

      l_lut = np.zeros((5,5,5), dtype=np.int32) - 99  # (type, station, ring) -> layer
      l_lut[1,1,4] = 0  # ME1/1a
      l_lut[1,1,1] = 0  # ME1/1b
      l_lut[1,1,2] = 1  # ME1/2
      l_lut[1,1,3] = 1  # ME1/3
      l_lut[1,2,1] = 2  # ME2/1
      l_lut[1,2,2] = 2  # ME2/2
      l_lut[1,3,1] = 3  # ME3/1
      l_lut[1,3,2] = 3  # ME3/2
      l_lut[1,4,1] = 4  # ME4/1
      l_lut[1,4,2] = 4  # ME4/2
      l_lut[2,1,2] = 1  # RE1/2
      l_lut[2,2,2] = 2  # RE2/2
      l_lut[2,3,1] = 3  # RE3/1
      l_lut[2,3,2] = 3  # RE3/2
      l_lut[2,3,3] = 3  # RE3/3
      l_lut[2,4,1] = 4  # RE4/1
      l_lut[2,4,2] = 4  # RE4/2
      l_lut[2,4,3] = 4  # RE4/3
      l_lut[3,1,1] = 0  # GE1/1
      l_lut[3,2,1] = 2  # GE2/1
      l_lut[4,1,1] = 0  # ME0
      self.l_lut = l_lut
      assert(self.l_lut.max() < nlayers)

      m_lut = np.zeros((5,5,5,2), dtype=np.int32) - 99  # (type, station, ring, fr) -> row
      m_lut[1,1,4,0] = 4  # ME1/1a (R)
      m_lut[1,1,4,1] = 3  # ME1/1a (F)
      m_lut[1,1,1,0] = 4  # ME1/1b (R)
      m_lut[1,1,1,1] = 3  # ME1/1b (F)
      m_lut[1,1,2,0] = 1  # ME1/2 (R)
      m_lut[1,1,2,1] = 0  # ME1/2 (F)
      m_lut[1,1,3,:] = 0  # ME1/3
      m_lut[1,2,1,:] = 6  # ME2/1
      m_lut[1,2,2,:] = 6  # ME2/2
      m_lut[1,3,1,:] = 7  # ME3/1
      m_lut[1,3,2,:] = 7  # ME3/2
      m_lut[1,4,1,:] = 9  # ME4/1
      m_lut[1,4,2,:] = 9  # ME4/2
      m_lut[2,1,2,:] = 2  # RE1/2
      m_lut[2,2,2,:] = 5  # RE2/2
      m_lut[2,3,1,:] = 8  # RE3/1
      m_lut[2,3,2,:] = 8  # RE3/2
      m_lut[2,3,3,:] = 8  # RE3/3
      m_lut[2,4,1,:] = 10 # RE4/1
      m_lut[2,4,2,:] = 10 # RE4/2
      m_lut[2,4,3,:] = 10 # RE4/3
      m_lut[3,1,1,:] = 2  # GE1/1
      m_lut[3,2,1,:] = 5  # GE2/1
      m_lut[4,1,1,:] = 0  # ME0
      self.m_lut = m_lut
      self.m_size = 11
      assert(self.m_lut.max() < self.m_size)

      self.n_size = 5040 // self.superstrip_size

      t_lut = np.zeros((11,7,2), dtype=np.int32) - 99  # (row,zone) -> (min theta, max theta)
      t_lut[0,0] = 4,17   # ME0
      t_lut[0,1] = 17,23  # ME0
      t_lut[1,0] = 4,17   # ME0
      t_lut[1,1] = 17,23  # ME0
      t_lut[2,1] = 17,26  # GE1/1
      t_lut[2,2] = 24,37  # GE1/1
      t_lut[2,3] = 35,45  # GE1/1
      t_lut[2,4] = 40,52  # GE1/1
      t_lut[3,0] = 4,17   # ME1/1 (F)
      t_lut[3,1] = 16,26  # ME1/1 (F)
      t_lut[3,2] = 24,37  # ME1/1 (F)
      t_lut[3,3] = 34,43  # ME1/1 (F)
      t_lut[3,4] = 40,53  # ME1/1 (F)
      t_lut[4,0] = 4,17   # ME1/1 (R)
      t_lut[4,1] = 16,26  # ME1/1 (R)
      t_lut[4,2] = 24,37  # ME1/1 (R)
      t_lut[4,3] = 34,43  # ME1/1 (R)
      t_lut[4,4] = 40,53  # ME1/1 (R)
      #
      t_lut[0,4] = 46,54  # ME1/2 (F)
      t_lut[0,5] = 52,67  # ME1/2 (F)
      t_lut[0,6] = 65,87  # ME1/2 (F)
      t_lut[1,4] = 46,54  # ME1/2 (R)
      t_lut[1,5] = 52,67  # ME1/2 (R)
      t_lut[1,6] = 65,87  # ME1/2 (R)
      t_lut[2,5] = 52,72  # RE1/2
      t_lut[2,6] = 68,84  # RE1/2
      #
      t_lut[5,0] = 7,19   # GE2/1
      t_lut[5,1] = 18,24  # GE2/1
      t_lut[5,2] = 23,35  # GE2/1
      t_lut[5,3] = 34,45  # GE2/1
      t_lut[5,4] = 40,46  # GE2/1
      t_lut[5,5] = 56,68  # RE2/2+3
      t_lut[5,6] = 64,76  # RE2/2+3
      t_lut[6,0] = 4,17   # ME2/1
      t_lut[6,1] = 16,25  # ME2/1
      t_lut[6,2] = 24,36  # ME2/1
      t_lut[6,3] = 34,43  # ME2/1
      t_lut[6,4] = 40,49  # ME2/1
      t_lut[6,5] = 52,67  # ME2/2
      t_lut[6,6] = 65,87  # ME2/2
      #
      t_lut[7,0] = 4,17   # ME3/1
      t_lut[7,1] = 16,25  # ME3/1
      t_lut[7,2] = 24,36  # ME3/1
      t_lut[7,3] = 34,40  # ME3/1
      t_lut[7,4] = 44,54  # ME3/2
      t_lut[7,5] = 52,67  # ME3/2
      t_lut[7,6] = 64,87  # ME3/2
      t_lut[8,0] = 4,20   # RE3/1
      t_lut[8,1] = 20,24  # RE3/1
      t_lut[8,2] = 24,32  # RE3/1
      t_lut[8,3] = 40,40  # RE3/2+3
      t_lut[8,4] = 40,52  # RE3/2+3
      t_lut[8,5] = 48,72  # RE3/2+3
      t_lut[8,6] = 60,84  # RE3/2+3
      #
      t_lut[9,0] = 4,17   # ME4/1
      t_lut[9,1] = 16,25  # ME4/1
      t_lut[9,2] = 24,35  # ME4/1
      t_lut[9,3] = 38,43  # ME4/2
      t_lut[9,4] = 41,54  # ME4/2
      t_lut[9,5] = 52,67  # ME4/2
      t_lut[9,6] = 64,87  # ME4/2
      t_lut[10,0] = 8,16  # RE4/1
      t_lut[10,1] = 16,28 # RE4/1
      t_lut[10,2] = 24,28 # RE4/1
      t_lut[10,3] = 36,44 # RE4/2+3
      t_lut[10,4] = 44,52 # RE4/2+3
      t_lut[10,5] = 52,64 # RE4/2+3
      t_lut[10,6] = 64,84 # RE4/2+3
      self.t_lut = t_lut

      self.zone_size = 7

      # ME1/1, ME1/2, ME2, ME3, ME4, RE1
      # RE2, RE3, RE4, GE1/1, GE2/1, ME0
      s_bend_lut =[-0.059926, -0.065415, -0.149701,  0.086035,  0.108797,  1.000000,
                    1.000000,  1.000000,  1.000000, -0.515074, -0.599277, -0.073872,]
      self.s_bend_lut = np.array(s_bend_lut, dtype=np.float32)

      s_bend_sign_lut = np.sign(self.s_bend_lut)
      self.s_bend_sign_lut = np.array(s_bend_sign_lut, dtype=np.float32)

      s_bend_max_lut = [28, 40, 12, 20, 20, 0, 0, 0, 0, 3, 1, 18]
      self.s_bend_max_lut = np.array(s_bend_max_lut, dtype=np.float32)

    def get_layer(self, hit):
      index = (hit.type, hit.station, hit.ring)
      lay = self.l_lut[index]
      return lay

    def get_row(self, hit):
      index = (hit.type, hit.station, hit.ring, hit.fr)
      m = self.m_lut[index]
      assert(m < self.m_size)
      return m

    def get_col(self, hit):
      hit_phi = np.asarray(hit.emtf_phi, dtype=np.int32)
      n = hit_phi // self.superstrip_size
      assert(n < self.n_size)
      return n

    def get_zones(self, hit):
      m = self.get_row(hit)
      t = self.t_lut[m]
      hit_theta = np.asarray(hit.emtf_theta, dtype=np.int32)
      zones = np.logical_and(t[:,0] <= hit_theta, hit_theta <= t[:,1])
      zones = np.where(zones)[0]
      assert((zones < self.zone_size).all())
      return zones

    def get_chn_bend(self, hit):
      if hit.type == kCSC:
        bend = hit.bend
        bend *= hit.endcap
      elif hit.type == kGEM:
        bend = hit.bend
        bend *= hit.endcap
      else:
        bend = hit.bend
      lay = self.get_layer(hit)
      #s = self.s_bend_lut[lay]
      #bend *= s
      bend_sign = self.s_bend_sign_lut[lay]
      bend_max = self.s_bend_max_lut[lay]
      if bend_max == 0:
        bend = 0.5
      else:
        bend = np.clip(float(bend), -bend_max, bend_max)
        bend = 0.5 + (bend_sign * bend)/(2 * bend_max)
      return bend

    def get_chn_theta(self, hit):
      theta = np.clip(float(hit.emtf_theta), 4., 86.)
      theta = (theta - 3.)/83.
      return theta

    def get_chn_mphi(self, hit):
      n = self.get_col(hit)
      mphi = float(hit.emtf_phi) - (n * self.superstrip_size)
      mphi /= self.superstrip_size
      return mphi

    def get_chn(self, hit):
      chn = (self.get_chn_bend(hit),
             self.get_chn_theta(hit),
             self.get_chn_mphi(hit))
      chn = np.asarray(chn, dtype=np.float32)
      assert((0.0 <= chn).all())
      assert((chn <= 1.0).all())
      return chn

    def is_intime(self, hit):
      if hit.type == kCSC:
        return hit.bx == -1 or hit.bx == 0
      else:
        return hit.bx == 0

    def __call__(self, hits):
      amap = {}
      for ihit, hit in enumerate(hits):
        if self.is_intime(hit):
          m = self.get_row(hit)
          n = self.get_col(hit)
          zones = self.get_zones(hit)
          for z in zones:
            m_z = m + (z * self.m_size)
            amap.setdefault((m_z,n), []).append(hit)

      image_pixels = np.zeros((self.sector_hits_capacity,2), dtype=np.int32) - 99
      image_channels = np.zeros((self.sector_hits_capacity,3), dtype=np.float32) + np.nan

      i = 0
      for k, v in amap.iteritems():
        if i == self.sector_hits_capacity:
          break
        (m,n) = k
        hits_mn = v
        h = hits_mn[0]  # arbitrarily pick the first hit
        chn = self.get_chn(h)

        image_pixels[i] = (m,n)
        image_channels[i] = chn
        i += 1

      return image_pixels, image_channels

  make_emtf_image = EMTFImage(superstrip_size=superstrip_size)


  # Assign EMTF label
  class EMTFLabel(object):
    def __init__(self):
      self.pt_bins = np.linspace(-0.525, 0.525, num=22)
      self.eta_bins = eta_bins
      self.phi_bins = np.linspace(-40, 40, num=129)  # uGMT uses 360./576

    def find_pt_bin(self, part):
      pt = np.true_divide(part.q, part.pt)
      ipt = np.digitize((pt,), self.pt_bins[1:])[0]  # skip lowest edge
      ipt = np.clip(ipt, 0, len(self.pt_bins)-2)
      return ipt

    def find_sector_phi(self, glob, sector):  # glob in deg, sector in [1,6]
      # sector 1 starts at 15 deg
      loc = glob - 15. - (60. * (sector-1))
      # but here I'm assuming sector 1 centers at 40 deg
      center = 40. - 15.
      dphi = delta_phi(loc, center)
      return dphi

    def find_phi_bin(self, part, sector):
      phi = self.find_sector_phi(np.rad2deg(part.phi), (sector%6)+1)
      iphi = np.digitize((phi,), self.phi_bins[1:])[0]  # skip lowest edge
      iphi = np.clip(iphi, 0, len(self.phi_bins)-2)
      return iphi

    def find_eta_bin(self, part):
      eta = abs(part.eta)
      ieta = np.digitize((eta,), self.eta_bins[1:])[0]  # skip lowest edge
      ieta = np.clip(ieta, 0, len(self.eta_bins)-2)
      return ieta

    def __call__(self, part, sector):
      label = (self.find_pt_bin(part),
               self.find_phi_bin(part, sector),
               self.find_eta_bin(part))
      label = np.asarray(label, dtype=np.int32)
      return label

  assign_emtf_label = EMTFLabel()

  # ____________________________________________________________________________
  tree = load_pgun()  #FIXME
  #tree = load_pgun_batch(jobid)

  out_image_pixels = []
  out_image_channels = []
  out_labels = []
  out_parameters = []

  # Event range
  #n = -1
  n = maxEvents  #FIXME

  # ____________________________________________________________________________
  # Loop over events
  for ievt, evt in enumerate(tree):
    if n != -1 and ievt == n:
      break

    if (ievt % 1000 == 0):  print("Processing event: {0}".format(ievt))

    # Skip events with very few hits
    if not len(evt.hits) >= 4:
      continue

    # Skip events without ME1 hits
    has_ME1 = False
    for ihit, hit in enumerate(evt.hits):
      if hit.type == kCSC and hit.station == 1:
        has_ME1 = True
        break
      elif hit.type == kME0 and hit.station == 1:
        has_ME1 = True
        break
    if not has_ME1:
      continue

    part = evt.particles[0]  # particle gun
    part.invpt = np.true_divide(part.q, part.pt)

    # Find the best sector
    sector_cnt_array = np.zeros((12,), dtype=np.int32)
    sector_hits_array = np.empty((12,), dtype=np.object)
    for ind in np.ndindex(sector_hits_array.shape):
      sector_hits_array[ind] = []

    for ihit, hit in enumerate(evt.hits):
      #print(".. hit  {0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11}".format(ihit, hit.bx, hit.type, hit.station, hit.ring, hit.sector, hit.fr, hit.sim_phi, hit.sim_theta, hit.time, hit.sim_tp1, hit.sim_tp2))

      assert(hit.emtf_phi < 5040)  # 84*60
      endsec = find_endsec(hit.endcap, hit.sector)
      sector_cnt_array[endsec] += 1
      sector_hits_array[endsec].append(hit)

    # Get the best sector hits
    best_sector = np.argmax(sector_cnt_array)
    sector_hits = sector_hits_array[best_sector]

    # The workhorse
    image_pixels, image_channels = make_emtf_image(sector_hits)
    out_image_pixels.append(image_pixels)
    out_image_channels.append(image_channels)

    labels = assign_emtf_label(part, best_sector)
    out_labels.append(labels)

    parameters = np.asarray((np.true_divide(part.q, part.pt), part.phi, part.eta, best_sector), dtype=np.float32)
    out_parameters.append(parameters)

    if ievt < 40 and part.pt > 5.:
      print("evt {0}".format(ievt))
      # Hits
      for ihit, hit in enumerate(sector_hits):
        print(".. hit  {0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11}".format(ihit, hit.bx, hit.type, hit.station, hit.ring, hit.sector, hit.fr, hit.emtf_phi, hit.emtf_theta, hit.time, hit.sim_tp1, hit.sim_tp2))
      # Gen particles
      for ipart, part in enumerate(evt.particles):
        print(".. part {0} {1} {2} {3} {4} {5}".format(ipart, part.pt, part.phi, part.eta, part.theta, part.q))
      # Zones
      n_zones = make_emtf_image.zone_size
      n_rows = make_emtf_image.m_size
      for zone in xrange(n_zones):
        # Count hits in zone
        rows = [px[0] for px in image_pixels if px[0] != -99 and (zone*n_rows) <= px[0] < ((zone+1)*n_rows)]
        print(".. zone {0} {1}/{2}".format(zone, len(rows), len(sector_hits)))
      print(".. zone info", map(make_emtf_image.get_zones, sector_hits))

  # End loop over events
  unload_tree()


  # ____________________________________________________________________________
  # Save objects
  print('[INFO] Creating file: histos_tbe.npz')
  if True:
    image_pixels   = np.asarray(out_image_pixels, dtype=np.int32)
    image_channels = np.asarray(out_image_channels, dtype=np.float32)
    labels         = np.asarray(out_labels, dtype=np.int32)
    parameters     = np.asarray(out_parameters, dtype=np.float32)
    outfile = 'histos_tbe.npz'
    np.savez_compressed(outfile, image_pixels=image_pixels, image_channels=image_channels, labels=labels, parameters=parameters)


