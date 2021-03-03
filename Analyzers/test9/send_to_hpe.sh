#scp histos_tba.14.npz histos_tbd.14.npz jlow@15.226.54.11:/scratch/CMS/L1MuonTrigger/P2_10_1_5/SingleMuon_Toy_2GeV/
#scp histos_tba.15.npz histos_tbd.15.npz jlow@15.226.54.11:/scratch/CMS/L1MuonTrigger/P2_10_1_5/SingleMuon_Toy_2GeV/
#scp histos_tba.16.npz histos_tbd.16.npz jlow@15.226.54.11:/scratch/CMS/L1MuonTrigger/P2_10_1_5/SingleMuon_Toy_2GeV/
#scp histos_tba.17.npz histos_tbd.17.npz histos_tbe.17.npz jlow@15.226.54.11:/scratch/CMS/L1MuonTrigger/P2_10_1_5/SingleMuon_Toy_2GeV/
#scp histos_tba.18.npz histos_tbd.18.npz histos_tbe.18.npz jlow@15.226.54.11:/scratch/CMS/L1MuonTrigger/P2_10_1_5/SingleMuon_Toy_2GeV/

tar czf hpe_default.tgz nn_*.py mykeras6.ipynb tdrstyle.mplstyle model.h5 model.json model_weights.h5
ssh jlow@15.226.54.11 "tar -C /home/uf/jlow/jftest3/slurm/ -xzf -" < hpe_default.tgz
