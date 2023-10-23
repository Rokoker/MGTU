import LCHM_matan
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

for r in range(1,35,1):
    print("Расстояние до цели ", r)
    LLS = LCHM_matan.LLS()
    a_gen, t_gen = LLS.generator(0)
    a_canal, t_canal = LLS.canal(a_gen, t_gen,r,0)
    a_buger, t_buger = LLS.canal_buger(a_canal,t_canal,0)
    a_buger_bgs, t_buger_bgs = LLS.canal_BGS(a_buger, t_buger,0,noise)
    a_loc, t_loc = LLS.loc_sig_for_photodetector(a_gen,t_canal,0)
    freq, spec, time = LLS.p_detector(a_loc, a_buger_bgs, t_buger_bgs, graf_time = 0, graf_spec = 0)
    R_target, OSH =  LLS.find_freq_bien(spec,freq,0)
       

    