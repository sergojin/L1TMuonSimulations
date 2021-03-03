import numpy as np
from enum import Enum 

nlayers = 16  # 5 (CSC) + 4 (RPC) + 3 (GEM)

#nvariables = (nlayers * 6) + 3 - 36 
#nvariables = (nlayers * 6) - 36 
nvariables = 23 # 6 dphi, 6 dtheta, 4 bends, 1 FR bit, track theta, ME11 bit, 4 RPC bits

#nvariables_input = (nlayers * 7) + 3

nvariables_input = (nlayers * (10+1)) + 3

nparameters_input = 6

# ______________________________________________________________________________
class Encoder(object):

  def __init__(self, x, y, adjust_scale=0, reg_pt_scale=1.0, reg_dxy_scale=1.0,
               drop_ge11=True, drop_ge21=True, drop_me0=True, drop_irpc=True, drop_dt=True):
    
      if x is None or y is None:
          raise Exception('Invalid input x or y')

      assert(x.shape[1] == nvariables_input)
      if y.shape[1] == 1:
          y = np.zeros((y.shape[0], nparameters_input), dtype=np.float32)
      else:
          assert(y.shape[1] == nparameters_input)
      assert(x.shape[0] == y.shape[0])  
    
      self.nentries = x.shape[0]
      self.x_orig  = x
      self.y_orig  = y
      self.x_copy  = x.copy()
      self.y_copy  = y.copy()

      # Get views
      # Each layer has 6 sets of features (phi, theta, bend, time, ring, fr) and 1 set of mask
      # Additionally, each road has 3 more features.
      # Some inputs are not actually used.
      #self.x_phi   = self.x_copy[:, nlayers*0:nlayers*1]
      #self.x_theta = self.x_copy[:, nlayers*1:nlayers*2]
      #self.x_bend  = self.x_copy[:, nlayers*2:nlayers*3]
      #self.x_time  = self.x_copy[:, nlayers*3:nlayers*4]
      #self.x_ring  = self.x_copy[:, nlayers*4:nlayers*5]
      #self.x_fr    = self.x_copy[:, nlayers*5:nlayers*6]
      #self.x_mask  = self.x_copy[:, nlayers*6:nlayers*7].astype(np.bool)  # this makes a copy
      #self.x_road  = self.x_copy[:, nlayers*7:nlayers*8]  # ipt, ieta, iphi
      #self.y_pt    = self.y_copy[:, 0]  # q/pT
      #self.y_phi   = self.y_copy[:, 1]
      #self.y_eta   = self.y_copy[:, 2]
      #self.y_vx        = self.y_copy[:, 3]
      #self.y_vy        = self.y_copy[:, 4]
      #self.y_vz        = self.y_copy[:, 5]
        
      self.x_phi       = self.x_copy[:, nlayers*0:nlayers*1]
      self.x_theta     = self.x_copy[:, nlayers*1:nlayers*2]
      self.x_bend      = self.x_copy[:, nlayers*2:nlayers*3]
      self.x_qual      = self.x_copy[:, nlayers*3:nlayers*4]
      self.x_time      = self.x_copy[:, nlayers*4:nlayers*5]
      self.x_ring      = self.x_copy[:, nlayers*5:nlayers*6]
      self.x_fr        = self.x_copy[:, nlayers*6:nlayers*7]
      self.x_old_phi   = self.x_copy[:, nlayers*7:nlayers*8]
      self.x_old_bend  = self.x_copy[:, nlayers*8:nlayers*9]
      self.x_ext_theta = self.x_copy[:, nlayers*9:nlayers*10]
      self.x_mask      = self.x_copy[:, nlayers*10:nlayers*11].astype(np.bool)  # this makes a copy
      self.x_road      = self.x_copy[:, nlayers*11:nlayers*12]  # ipt, ieta, iphi
      self.y_pt        = self.y_copy[:, 0]  # q/pT
      self.y_phi       = self.y_copy[:, 1]
      self.y_eta       = self.y_copy[:, 2]
      self.y_vx        = self.y_copy[:, 3]
      self.y_vy        = self.y_copy[:, 4]
      self.y_vz        = self.y_copy[:, 5]  
        
      
      # Scale q/pT for training
      self.y_pt *= reg_pt_scale  
        
      self.y_dxy  = self.y_vy * np.cos(self.y_phi) - self.y_vx * np.sin(self.y_phi)
      self.y_dxy *= reg_dxy_scale

      # Drop detectors
      x_dropit = self.x_mask
      if drop_ge11:
        x_dropit[:, 9] = 1  # 9: GE1/1
      if drop_ge21:
        x_dropit[:, 10] = 1 # 10: GE2/1
      if drop_me0:
        x_dropit[:, 11] = 1 # 11: ME0
      if drop_irpc:
        x_ring_tmp = self.x_ring.astype(np.int32)
        x_ring_tmp = (x_ring_tmp == 2) | (x_ring_tmp == 3)
        x_dropit[~x_ring_tmp[:,7], 7] = 1  # 7: RE3, neither ring2 nor ring3
        x_dropit[~x_ring_tmp[:,8], 8] = 1  # 8: RE4, neither ring2 nor ring3        
      if drop_dt:
        x_dropit[:, 12:16] = 1 # 12,13,14,15: MB1,2,3,4

      self.x_phi      [x_dropit] = np.nan
      self.x_theta    [x_dropit] = np.nan
      self.x_bend     [x_dropit] = np.nan
      self.x_qual     [x_dropit] = np.nan
      self.x_time     [x_dropit] = np.nan
      self.x_ring     [x_dropit] = np.nan
      self.x_fr       [x_dropit] = np.nan
      self.x_old_phi  [x_dropit] = np.nan
      self.x_old_bend [x_dropit] = np.nan
      self.x_ext_theta[x_dropit] = np.nan
      self.x_mask [x_dropit] = 1  

     
      # Make event weight
      #self.w       = np.ones(self.y_pt.shape, dtype=np.float32)
      self.w       = np.abs(self.y_pt)/0.2 + 1.0

      # Straightness & zone
      self.x_straightness = self.x_road[:, 0][:, np.newaxis]
      self.x_zone         = self.x_road[:, 1][:, np.newaxis]

      # Subtract median phi from hit phis
      self.x_phi_median    = self.x_road[:, 2] * 32  # multiply by 'quadstrip' unit (4 * 8)
      self.x_phi_median    = self.x_phi_median[:, np.newaxis]
      #self.x_phi          -= self.x_phi_median

      # Subtract median theta from hit thetas
      self.x_theta_median  = np.nanmedian(self.x_theta[:,:5], axis=1)  # CSC only
      #self.x_theta_median[np.isnan(self.x_theta_median)] = np.nanmedian(self.x_theta[np.isnan(self.x_theta_median)], axis=1)  # use all types
      self.x_theta_median  = self.x_theta_median[:, np.newaxis]
      #self.x_theta        -= self.x_theta_median

      # Modify ring and F/R definitions
      x_ring_tmp = self.x_ring.astype(np.int32)
      self.x_ring[(x_ring_tmp == 2) | (x_ring_tmp == 3)] = +1 # ring 2,3 -> +1
      self.x_ring[(x_ring_tmp == 1) | (x_ring_tmp == 4)] = -1 # ring 1,4 -> -1
      x_fr_tmp = self.x_fr.astype(np.int32)
      self.x_fr[(x_fr_tmp == 1)] = +1  # front chamber -> +1
      self.x_fr[(x_fr_tmp == 0)] = -1 # rear chamber  -> -1    
    
      #Start current EMTF input calculations  
      self.x_dphi = np.zeros((self.nentries, 6))
      self.x_dtheta = np.zeros((self.nentries, 6))

      self.x_phi_emtf = np.zeros((self.nentries, 4))
      self.x_theta_emtf = np.zeros((self.nentries, 4))
      self.x_bend_emtf = np.zeros((self.nentries, 4))
      self.x_fr_emtf = np.zeros((self.nentries, 1))
      self.x_track_theta = np.zeros((self.nentries, 1))  
      self.x_ME11ring = np.zeros((self.nentries, 1)) 
      self.x_RPCbit = np.zeros((self.nentries, 4))   
    
      for i in range(self.nentries):
        if ~np.isnan(self.x_old_phi[i,0]) :
           self.x_phi_emtf[i,0] = self.x_old_phi[i,0]
           self.x_theta_emtf[i,0] = self.x_theta[i,0]
           self.x_bend_emtf[i,0] = self.x_old_bend[i,0]
           self.x_fr_emtf[i,0] = self.x_fr[i,0]
           self.x_ME11ring [i,0] = 0. 
        elif ~np.isnan(self.x_old_phi[i,1]) :   
           self.x_phi_emtf[i,0] = self.x_old_phi[i,1]
           self.x_theta_emtf[i,0] = self.x_theta[i,1]
           self.x_bend_emtf[i,0] = self.x_old_bend[i,1] 
           self.x_fr_emtf[i,0] = self.x_fr[i,1]
           self.x_ME11ring [i,0] = 1. 
        elif ~np.isnan(self.x_old_phi[i,5]):
           self.x_phi_emtf[i,0] = self.x_old_phi[i,5]  
           self.x_theta_emtf[i,0] = self.x_theta[i,5]
           self.x_bend_emtf[i,0] = 0.
           self.x_fr_emtf[i,0] = 0.
           self.x_RPCbit[i,0] = 1.

        if ~np.isnan(self.x_old_phi[i,2]):
           self.x_phi_emtf[i,1] = self.x_old_phi[i,2]
           self.x_theta_emtf[i,1] = self.x_theta[i,2]
           self.x_bend_emtf[i,1] = self.x_old_bend[i,2]
        elif ~np.isnan(self.x_old_phi[i,6]):               
           self.x_phi_emtf[i,1] = self.x_old_phi[i,6]               
           self.x_theta_emtf[i,1] = self.x_theta[i,6]   
           self.x_bend_emtf[i,1] = 0.
           self.x_RPCbit[i,1] = 1.
 
        if ~np.isnan(self.x_old_phi[i,3]):
           self.x_phi_emtf[i,2] = self.x_old_phi[i,3]
           self.x_theta_emtf[i,2] = self.x_theta[i,3] 
           self.x_bend_emtf[i,2] = self.x_old_bend[i,3] 
        elif ~np.isnan(self.x_old_phi[i,7]):               
           self.x_phi_emtf[i,2] = self.x_old_phi[i,7]               
           self.x_theta_emtf[i,2] = self.x_theta[i,7]  
           self.x_bend_emtf[i,2] = 0. 
           self.x_RPCbit[i,2] = 1.
        
        if ~np.isnan(self.x_old_phi[i,4]):
           self.x_phi_emtf[i,3] = self.x_old_phi[i,4]
           self.x_theta_emtf[i,3] = self.x_theta[i,4]  
           self.x_bend_emtf[i,3] = self.x_old_bend[i,4]  
        elif ~np.isnan(self.x_old_phi[i,8]):               
           self.x_phi_emtf[i,3] = self.x_old_phi[i,8]     
           self.x_theta_emtf[i,3] = self.x_theta[i,8]
           self.x_bend_emtf[i,3] = 0.     
           self.x_RPCbit[i,3] = 1.
        
        if ~np.isnan(self.x_theta[i,2]):
            self.x_track_theta [i,0] = self.x_theta[i,2] 
        elif ~np.isnan(self.x_theta[i,3]):    
            self.x_track_theta [i,0] = self.x_theta[i,3]                 
        elif ~np.isnan(self.x_theta[i,4]):    
            self.x_track_theta [i,0] = self.x_theta[i,4]    
            
                
        self.x_dphi[i,0] = self._dphi(self.x_phi_emtf[i,0],self.x_phi_emtf[i,1])
        self.x_dphi[i,1] = self._dphi(self.x_phi_emtf[i,0],self.x_phi_emtf[i,2])
        self.x_dphi[i,2] = self._dphi(self.x_phi_emtf[i,0],self.x_phi_emtf[i,3])
        self.x_dphi[i,3] = self._dphi(self.x_phi_emtf[i,1],self.x_phi_emtf[i,2])
        self.x_dphi[i,4] = self._dphi(self.x_phi_emtf[i,1],self.x_phi_emtf[i,3])
        self.x_dphi[i,5] = self._dphi(self.x_phi_emtf[i,2],self.x_phi_emtf[i,3])

     
        self.x_dtheta[i,0] = self._dtheta(self.x_theta_emtf[i,0],self.x_theta_emtf[i,1])
        self.x_dtheta[i,1] = self._dtheta(self.x_theta_emtf[i,0],self.x_theta_emtf[i,2])
        self.x_dtheta[i,2] = self._dtheta(self.x_theta_emtf[i,0],self.x_theta_emtf[i,3])
        self.x_dtheta[i,3] = self._dtheta(self.x_theta_emtf[i,1],self.x_theta_emtf[i,2])
        self.x_dtheta[i,4] = self._dtheta(self.x_theta_emtf[i,1],self.x_theta_emtf[i,3])
        self.x_dtheta[i,5] = self._dtheta(self.x_theta_emtf[i,2],self.x_theta_emtf[i,3])

      self.x_mask[:,:]= 0
    
      print "phi ME11", self.x_old_phi[0:10,0]
      print "phi ME12", self.x_old_phi[0:10,1]
      print "phi ME2", self.x_old_phi[0:10,2]
      print "phi RE2", self.x_old_phi[0:10,6]
      print "delta phi 1-2", self.x_dphi[0:10,0]
      
      #print "theta ME11", self.x_theta[0:10,0]
      #print "theta ME12",  self.x_theta[0:10,1]
      #print "theta ME2", self.x_theta[0:10,2]  
      #print "theta RE2", self.x_theta[0:10,6]  
      #print self.x_dtheta[0:10,0]        
    
      # Remove NaN
      self._handle_nan_in_x(self.x_copy)
      #self._handle_nan_in_x(self.x_gem_csc_bend)
     
      return

  # Copied from scikit-learn
  def _handle_zero_in_scale(self, scale):
    scale[scale == 0.0] = 1.0
    return scale

  def _handle_nan_in_x(self, x):
    x[np.isnan(x)] = 0.0
    return x

  def _dphi(self, x, y):
    if ( x!=0 and y!=0):
       delta = x-y
    else: 
       delta = 0.
    return delta

  def _dtheta(self, x, y):
    if ( x!=0 and y!=0):
       delta = x-y
    else: 
       delta =0.
    return delta

  def get_x(self, drop_columns_of_zeroes=True):
    #x_new = np.hstack((self.x_phi, self.x_theta, self.x_bend,
    #                   self.x_time, self.x_ring, self.x_fr,
    #                   self.x_straightness, self.x_zone, self.x_theta_median))
    #x_new = np.hstack((self.x_phi, self.x_theta, self.x_bend,
    #                   self.x_time, self.x_ring, self.x_fr, self.x_dphi, self.x_dtheta, self.x_bend_emtf, self.x_fr_emtf))
    
    x_new = np.hstack((self.x_dphi, self.x_dtheta, self.x_bend_emtf, self.x_fr_emtf, self.x_track_theta, self.x_ME11ring, self.x_RPCbit))
    
    # Drop input nodes
    #if drop_columns_of_zeroes:
      #drop_phi    = [nlayers*0 + x for x in xrange(0,0)]  # keep everyone
      #drop_theta  = [nlayers*1 + x for x in xrange(0,0)]  # keep everyone
      #drop_bend   = [nlayers*2 + x for x in xrange(5,11)] # no bend for RPC, GEM
      #drop_time   = [nlayers*3 + x for x in xrange(0,12)] # no time for everyone
      #drop_ring   = [nlayers*4 + x for x in xrange(5,12)] # ring for only ME2, ME3, ME4
      #drop_ring  += [nlayers*4 + x for x in xrange(0,2)]  # ^
      #drop_fr     = [nlayers*5 + x for x in xrange(2,11)] # fr for only ME1/1, ME1/2, ME0
    
    #  drop_phi    = [nlayers*0 + x for x in xrange(0,12)]  # keep everyone
    #  drop_theta  = [nlayers*1 + x for x in xrange(0,12)]  # keep everyone
    #  drop_bend   = [nlayers*2 + x for x in xrange(0,12)] # no bend for RPC, GEM
    #  drop_time   = [nlayers*3 + x for x in xrange(0,12)] # no time for everyone
    #  drop_ring   = [nlayers*4 + x for x in xrange(0,12)] # ring for only ME2, ME3, ME4
    #  drop_fr     = [nlayers*5 + x for x in xrange(0,12)] # fr for only ME1/1, ME1/2, ME0

      #drop_pattern = [nlayers*6 + x for x in xrange(0,3)]  
        
    #  x_dropit = np.zeros(x_new.shape[1], dtype=np.bool)
    #  for i in drop_phi + drop_theta + drop_bend + drop_time + drop_ring + drop_fr:
    #    x_dropit[i] = True

      #x_dropit_test = np.all(x_new == 0, axis=0)  # find columns of zeroes
      #assert(list(x_dropit) == list(x_dropit_test))

    #  x_new = x_new[:, ~x_dropit]
      #print x_new[0:10,:]
    return x_new

  def get_x_mask(self):
    x_mask = self.x_mask.copy()
    return x_mask

  def get_x_road(self):
    x_road = np.hstack((self.x_straightness, self.x_zone, self.x_theta_median))
    return x_road

  def get_y(self):
    y_new = self.y_pt.copy()
    return y_new

  def get_y_corrected_for_eta(self):
    y_new = self.y_pt * (np.sinh(1.8587) / np.sinh(np.abs(self.y_eta)))
    return y_new

  def get_dxy(self):
    dxy_new = self.y_dxy.copy()
    return dxy_new

  def get_dz(self):
    dz_new = self.y_vz.copy()
    return dz_new

  def get_w(self):
    w_new = self.w.copy()
    return w_new

  def save_encoder(self, filepath):
    np.savez_compressed(filepath, x_mean=self.x_mean, x_std=self.x_std)

  def load_endcoder(self, filepath):
    loaded = np.load(filepath)
    self.x_mean = loaded['x_mean']
    self.x_std = loaded['x_std']
