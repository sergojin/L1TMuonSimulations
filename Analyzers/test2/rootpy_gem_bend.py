import numpy as np
np.random.seed(2023)

from rootpy.plotting import Hist, Hist2D, Legend, Canvas
from rootpy.tree import Tree, TreeModel, FloatCol, IntCol, ShortCol
from rootpy.io import root_open


# ______________________________________________________________________________
# Tree models
#   see: http://www.rootpy.org/auto_examples/tree/model.html

class Hit(TreeModel):
  pass

class Track(TreeModel):
  pass

class Particle(TreeModel):
  pass


# ______________________________________________________________________________
# Analyzer

# Open file
#infile = root_open('ntuple.0.root')
infile = root_open('ntuple.1.root')
tree = infile.ntupler.tree

# Book histograms
histograms = {}
histogram2Ds = {}

# pT vs gen pT
hname = "h2_pt_vs_genpt"
histogram2Ds[hname] = Hist2D(50, 0.0, 0.5, 50, 0.0, 0.5, name=hname, title="; gen 1/p_{T} [1/GeV]; EMTFv5 1/p_{T} [1/GeV]", type='F')

# GEM-CSC bend vs gen pT
for i in xrange(3):
  # i=0: rear, i=1: front, i=2: all
  hname = "h2_bend_vs_genpt_fr%i" % i
  histogram2Ds[hname] = Hist2D(50, 0.0, 0.5, 71, -1.025, 2.525, name=hname, title="; gen 1/p_{T} [1/GeV]; GEM-CSC bend [deg]", type='F')

  hname = "h2_common_bend_vs_genpt_fr%i" % i
  histogram2Ds[hname] = Hist2D(50, 0.0, 0.5, 71, -1.025, 2.525, name=hname, title="; gen 1/p_{T} [1/GeV]; GEM-CSC bend [deg]", type='F')

# GEM-CSC bend when triggered
for i in xrange(3):
  # i=0: gen pT <= 20, i=1: gen pT > 20, i=2: all
  hname = "h_bend_l1pt20uns_gen%i" % i
  histograms[hname] = Hist(71, -1.025, 2.525, name=hname, title="; GEM-CSC bend [deg]", type='F')

# GEM-CSC scaled bend residuals
for i in xrange(6):
  # i=0..6: 2, 3, 5, 10, 20, 50 GeV
  hname = "h_bend_s_pt%i" % i
  histograms[hname] = Hist(101, -0.505, 0.505, name=hname, title="; residual of (scaled bend - EMTFv5 1/p_{T}) [1/GeV]", type='F')


# Define collection
tree.define_collection(name='hits', prefix='vh_', size='vh_size')
tree.define_collection(name='tracks', prefix='vt_', size='vt_size')
tree.define_collection(name='genparticles', prefix='vp_', size='vp_size')

# Enums
kDT, kCSC, kRPC, kGEM = 0, 1, 2, 3

# Lambdas
unscale_pt = lambda x: x/1.3

get_common_bend = lambda x, y: (x*2.29072) if y else x

get_scaled_bend = lambda x: (x*0.24363)


# Loop over events
for ievt, evt in enumerate(tree):
  if (ievt % 1000 == 0):  print("Processing event: {0}".format(ievt))

  # ____________________________________________________________________________
  # Filter

  # Skip events if not exactly one gen particle
  if len(evt.genparticles) != 1:  continue

  # Skip events if not exactly one track
  if len(evt.tracks) != 1:  continue

  mytrk = evt.tracks[0]
  mypart = evt.genparticles[0]

  # Skip event if no station 1 hit
  mode = mytrk.mode
  if not (mode & (1<<3)):  continue

  endcap = mytrk.endcap
  sector = mytrk.sector

  # Skip event if no ME1/1 hit
  myhits_me11 = [hit for hit in evt.hits if hit.endcap == mytrk.endcap and hit.sector == mytrk.sector and hit.station == 1 and (hit.ring == 1 or hit.ring == 4) and hit.type == kCSC]
  if not len(myhits_me11): continue

  # Skip event if no ME1/1 hit
  myhits_ge11 = [hit for hit in evt.hits if hit.endcap == mytrk.endcap and hit.sector == mytrk.sector and hit.station == 1 and hit.ring == 1 and hit.type == kGEM]
  if not len(myhits_ge11): continue

  # ____________________________________________________________________________
  # Verbose

  verbose = False

  if verbose:
    # Hits
    for ihit, hit in enumerate(evt.hits):
      print(".. hit  {0} {1} {2} {3} {4} {5} {6} {7}".format(ihit, hit.bx, hit.type, hit.station, hit.ring, hit.sim_phi, hit.sim_theta, hit.fr))

    # Tracks
    for itrk, trk in enumerate(evt.tracks):
      print(".. trk  {0} {1} {2} {3} {4} {5} {6}".format(itrk, trk.pt, trk.phi, trk.eta, trk.theta, trk.q, trk.mode))

    # Gen particles
    for ipart, part in enumerate(evt.genparticles):
      print(".. part {0} {1} {2} {3} {4} {5}".format(ipart, part.pt, part.phi, part.eta, part.theta, part.q))

  # ____________________________________________________________________________
  # Make plots

  myhit_me11 = np.random.choice(myhits_me11)
  myhit_ge11 = np.random.choice(myhits_ge11)
  mybend = myhit_me11.sim_phi - myhit_ge11.sim_phi
  if (mypart.q > 0):  mybend = -mybend
  myfr = myhit_me11.fr
  mybend_common = get_common_bend(mybend, myfr)
  mybend_scaled = get_scaled_bend(mybend_common)

  myptbin = -1
  if ((1.0/2 - 0.05) < 1.0/mypart.pt <= (1.0/2)):
    myptbin = 0;
  elif ((1.0/3 - 0.03333) < 1.0/mypart.pt <= (1.0/3)):
    myptbin = 1;
  elif ((1.0/5 - 0.03333) < 1.0/mypart.pt <= (1.0/5)):
    myptbin = 2;
  elif ((1.0/10 - 0.02) < 1.0/mypart.pt <= (1.0/10)):
    myptbin = 3;
  elif ((1.0/20 - 0.02) < 1.0/mypart.pt <= (1.0/20)):
    myptbin = 4;
  elif ((1.0/50 - 0.01) < 1.0/mypart.pt <= (1.0/50)):
    myptbin = 5;

  # pT vs gen pT
  hname = "h2_pt_vs_genpt"
  histogram2Ds[hname].fill(1.0/mypart.pt, 1.0/unscale_pt(mytrk.pt))

  # bend vs gen pT
  hname = "h2_bend_vs_genpt_fr%i" % myfr
  histogram2Ds[hname].fill(1.0/mypart.pt, mybend)
  hname = "h2_bend_vs_genpt_fr%i" % 2  # inclusive
  histogram2Ds[hname].fill(1.0/mypart.pt, mybend)

  # common bend vs gen pT
  hname = "h2_common_bend_vs_genpt_fr%i" % myfr
  histogram2Ds[hname].fill(1.0/mypart.pt, mybend_common)
  hname = "h2_common_bend_vs_genpt_fr%i" % 2  # inclusive
  histogram2Ds[hname].fill(1.0/mypart.pt, mybend_common)

  # GEM-CSC bend when triggered
  mytrigger = unscale_pt(mytrk.pt) > 20.
  mytrigger_gen = mypart.pt > 20
  if mytrigger:
    hname = "h_bend_l1pt20uns_gen%i" % mytrigger_gen
    histograms[hname].fill(mybend_common)
    hname = "h_bend_l1pt20uns_gen%i" % 2  # inclusive
    histograms[hname].fill(mybend_common)

  # GEM-CSC scaled bend residuals
  if myptbin != -1:
    hname = "h_bend_s_pt%i" % myptbin
    histograms[hname].fill(mybend_scaled - 1.0/unscale_pt(mytrk.pt))

  continue  # end loop over event

# ______________________________________________________________________________
# Drawer

from drawer import *
mydrawer = MyDrawer()
options = mydrawer.options

# Print
for hname, h in histograms.iteritems():
  h.Draw("COLZ")
  gPad.Print(options.outdir + hname + ".png")
for hname, h in histogram2Ds.iteritems():
  h.Draw("COLZ")
  gPad.Print(options.outdir + hname + ".png")

# Make ratio of bend vs gen pT
denom_ifr = 1
hname = "h2_bend_vs_genpt_fr%i" % denom_ifr
h2 = histogram2Ds[hname].Clone(hname + "_tmp")
h2.RebinX(4)
prof = h2.ProfileX(hname + "_tmp_pfx", 1, -1, "s")
proj = prof.ProjectionX(hname + "_tmp_px", "e")
denom = proj
#
numer_ifr = 1 - denom_ifr
hname = "h2_bend_vs_genpt_fr%i" % numer_ifr
h2 = histogram2Ds[hname].Clone(hname + "_tmp")
h2.RebinX(4)
prof = h2.ProfileX(hname + "_tmp_pfx", 1, -1, "s")
proj = prof.ProjectionX(hname + "_tmp_px", "e")
numer = proj
#
numer.Divide(denom)
numer.Draw()
numer.Fit("pol0", "", "", 0.04, 0.32)
hname = numer.GetName()
gPad.Print(options.outdir + hname + ".png")

# Make ratio of bend vs gen pT [2]
hname = "h2_common_bend_vs_genpt_fr%i" % 2  # inclusive
h2 = histogram2Ds[hname].Clone(hname + "_tmp")
h2.RebinX(4)
prof = h2.ProfileX(hname + "_tmp_pfx", 1, -1, "s")
proj = prof.ProjectionX(hname + "_tmp_px", "e")
denom = proj
#
hname = "h2_pt_vs_genpt"
h2 = histogram2Ds[hname].Clone(hname + "_tmp")
h2.RebinX(4)
prof = h2.ProfileX(hname + "_tmp_pfx", 1, -1, "s")
proj = prof.ProjectionX(hname + "_tmp_px", "e")
numer = proj
#
numer.Divide(denom)
numer.Draw()
numer.Fit("pol0", "", "", 0.04, 0.32)
hname = numer.GetName()
gPad.Print(options.outdir + hname + ".png")

# Make overlay of GEM-CSC bend when triggered
hname = "h_bend_l1pt20uns_gen%i" % 2  # inclusive
h1a = histograms[hname]
h1a.linecolor = 'black'
h1a.linewidth = 2
hname = "h_bend_l1pt20uns_gen%i" % 1
h1b = histograms[hname]
h1b.linecolor = 'red'
h1b.linewidth = 2
h1a.Draw("hist")
h1b.Draw("same hist")
hname = "h_bend_l1pt20uns_gen%i" % 99
gPad.Print(options.outdir + hname + ".png")

# Make overlay of GEM-CSC scaled bend residuals
labels = ["2 GeV", "3 GeV", "5 GeV", "10 GeV", "20 GeV", "50 GeV"]
leg = Legend(len(labels), leftmargin=0.55, textfont=42, textsize=0.03, entryheight=0.04, entrysep=0.01)
leg.SetShadowColor(0)
leg.SetBorderSize(0)
#
for i in xrange(6):
  hname = "h_bend_s_pt%i" % i
  h1a = histograms[hname]
  h1a.linecolor = options.palette[i]
  h1a.linewidth = 2
  h1a.Scale(1.0/h1a.Integral())  # normalize
  if i == 0:
    ymax = 0.3
    h1a.SetMaximum(ymax)
    h1a.GetYaxis().SetTitle("(normalized)")
    h1a.Draw("hist")
  else:
    h1a.Draw("same hist")
  leg.AddEntry(h1a, labels[i], "l")
#
leg.Draw()
hname = "h_bend_s_pt%i" % 99
gPad.Print(options.outdir + hname + ".png")
