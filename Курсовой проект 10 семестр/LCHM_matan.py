# Фаил, задающий всю логику и проводящий расчеты
import matplotlib.pyplot as plt
import numpy as np
import pylab
import math
import random
import scipy.fft as sp
from scipy import signal
from scipy.optimize import curve_fit
from statistics import mean
from scipy import special
import scipy.stats as sct
import time

# Зададим класс, внутри которого будем проводить расчеты
class LLS(object):

    #Зададим параметры лазера исходя из условия
    c = 3 * 10**9                   # Скорость света
    imp_A = 10                      # Амплитуда импульса
    imp_lambda = 905 * 10**(-6)     # Длина волны импульса
    imp_f = c / imp_lambda          # Частота импульса
    df = 10**9                      # Частота девицаии импульса
    #df = 0.2 * imp_f
    print("Девиация ", df, "Герц")
    print("Частота ", imp_f, "Герц")
    imp_t = 50 * 10**(-9)           # Длителььность импульса 
    disc_f = imp_f*50               # Частота дискретизации     
    t_int = 1/disc_f                # Интервал взятия отсчетов
    B = df/(imp_t/4)                # Скорость изменения частоты
    

    def generator(self, graf):
        
        sig_t=np.arange(0,self.imp_t/2 + self.t_int,self.t_int)                 # Первая временная область для отображения импульса, частота растет
        print("Максимальная дальность до цели = ", sig_t[-1]*self.c/2, " м")    # Из условия определенности        
        sig_t_2 = np.arange(self.imp_t/2, self.imp_t + self.t_int,self.t_int)   # Вторая временная область для отображения импульса, частота падает
        amp_imp=[] # Массив амплитуда сигнала
        arr_frq=[] # Массив мгновенных значений частоты

        # Формируем амплитуды в первой временной области
        for time in sig_t:
            frq_1 = self.imp_f + self.B * time/2 
            amp_imp.append(self.imp_A*np.exp(1j*2*np.pi*(frq_1*time)))
            last = time
            arr_frq.append(frq_1)
        
        # Поиск фазы для правильного в момент изменения роста/уменьшения частоты 
        phasa = amp_imp[-1]
        phasa = math.atan2(phasa.imag,phasa.real)
        
        # Формируем амплитуды во второй временной области
        for time in sig_t_2:
            time = time - last
            frq_2 = frq_1 - self.B * time/2 
            amp_imp.append(self.imp_A*np.exp(1j*2*np.pi*(frq_2*time)+phasa*1j))
            arr_frq.append(frq_2)

        sig_t = np.append(sig_t, sig_t_2) # Объеденяем временные области     

        if graf == 1:
            plt.title("График изменения частоты от времени")
            plt.ylabel("Частота,Гц")
            plt.xlabel("Время, с")
            plt.plot(sig_t,arr_frq)
            plt.show()         
        
        return  amp_imp,sig_t
    
    def canal(self,izl_sig, sig_t,R,graf):                   # Канал без потерь с целью
        t_zad = 2*R/self.c                              # Задержка по веремени в зависимосимости от расстяния R       
        for time in range(0, len(sig_t)):
            sig_t[time] =sig_t[time] + t_zad            # Перенесли время сигнала на время приема
        do_sig_t=np.arange(0,t_zad,self.t_int)       # Временная область до приема сигнала
        do_sig_a=[]                                     # Амплитудная область до приема сигнала
        for time in range(len(do_sig_t)):
            do_sig_a.append(0)                          # Амплитуда на приемнике до приема сигнала равна нулю, заполнили нулями
        do_sig_a.extend(izl_sig)                        # Совмещение амплитудных областей до приема сигнала и приема сигнала
        do_sig_t = np.append(do_sig_t,sig_t)            # Совмещение временных областей до приема сигнала и приема сигнала
        if graf == 1:
            plt.title("Амплитуда сигнала в зависимости от времени")
            plt.ylabel("Амплитуда,В")
            plt.xlabel("Время, с")
            plt.plot(do_sig_t,do_sig_a)
            plt.show()
        sog=[]
        for i in range(len(do_sig_a)):
            sog.append((do_sig_a[i].real))
        sig_spec = (sp.rfft((sog))/len(sog))
        sig_freq = (sp.rfftfreq(len(sog), d = self.t_int))

        if graf == 1:
            plt.title("Спектр сигнала")
            plt.ylabel("Мощность, В/Гц")
            plt.xlabel("Время, с")
            plt.plot(sig_freq,np.abs(sig_spec))
            plt.show()
                     
        return  do_sig_a, do_sig_t                  # Вернули амплитуды сигнала и его временную область
    
    
    def canal_buger(self,amp_sig, sig_t,graf):
        # Канал с затуханием
        # Так как на момент выполнения данного метода у нас амплитдная область содержит только амплитуду принятого сигнала
        # То мы можем наложить маску затухания на всю область, затухание согласно экспоненте, подробнее в документации   
        sig_a=[]     
        for time in range(len(sig_t)):
            sig_a.append(amp_sig[time]*math.exp(-0.012*sig_t[time]*self.c))
        if graf == 1:
            plt.title("Амплитуда сигнала, при затухании")
            plt.ylabel("Амплитуда, В")
            plt.xlabel("Время, с")
            plt.plot(sig_t, sig_a)
            plt.show()                    
        return  sig_a, sig_t
    
    
    def canal_BGS(self,amp_sig,time_sig,graf,noise):      # Наложение БГШ
        sig=[]
        for time in range(0, len(amp_sig)):
            sig.append(amp_sig[time] + random.gauss(0,noise))  # Добавление шума к амплитуде     
        if graf == 1:
            plt.title("Амплитуда сигнала + шум")
            plt.ylabel("Амплитуда, В")
            plt.xlabel("Время, с")
            plt.plot(time_sig, sig)
            plt.show()
        sog=[]
        for i in range(len(sig)):
            sog.append(abs(sig[i]))
        sig_spec = (sp.rfft((sog))/len(sog))
        sig_freq = (sp.rfftfreq(len(sog), d = self.t_int))

        if graf == 1:
            plt.title("Спектр сигнала + шум")
            plt.ylabel("Мощность, В/Гц")
            plt.xlabel("Время, с")
            plt.plot(sig_freq,20*np.log10(np.abs(sig_spec)))
            plt.show() 
                
        return  sig, time_sig                                    # Вернули амплитуды сигнала с шумом
    

    def loc_sig_for_photodetector(self,sig_a,sig_t,graf):
        sig_loc=[] # Массив под локальный сигнал
        # Цил заполняет все время моделирования циклично сгенерированным импульсом
        for i in range(len(sig_t)):
            if i < len(sig_a):
                sig_loc.append(sig_a[i])
            else:
                sig_loc.append(sig_a[i-len(sig_a)])
        if graf == 1:
            plt.title("Амплитуда локального сигнала на фотодетекторе")
            plt.ylabel("Амплитуда, В")
            plt.xlabel("Время, с")
            plt.plot(sig_t, sig_loc)
            plt.show()     
        return sig_loc, sig_t
    

    def p_detector(self,sig_loc, sig_priema, sig_t, graf_time, graf_spec):
        if graf_time == 1:
            plt.title("Ампилутуда сигнала на входе фотодетектора")
            plt.ylabel("Амплитуда, В")
            plt.xlabel("Время, с")
            plt.plot(sig_t, sig_loc)
            plt.plot(sig_t, sig_priema)
            plt.show() 
        itog=[] # Массив на входе фотодетектора
        for i in range(len(sig_priema)):
            itog.append(abs((sig_loc[i]+sig_priema[i])**2))
                            #(-sig_loc[i]+sig_priema[i])**2)) # Смешиваем амплитуды сигналов
            #itog.append(abs((sig_loc[i]**2+sig_priema[i]**2))) # Смешиваем амплитуды сигналов
        if graf_time == 1:
            plt.title("Ампилутуда сиганал после фотодетектора")
            plt.ylabel("Амплитуда, В")
            plt.xlabel("Время, с")
            plt.plot(sig_t, itog)
            plt.show() 

        sig_spec = (sp.rfft((itog))/len(itog))
        sig_freq = (sp.rfftfreq(len(itog), d = self.t_int))

        if graf_spec == 1:
            plt.title("Спект сигнала после фотодетектора")
            plt.ylabel("Мощность, В/Гц")
            plt.xlabel("Время, с")
            plt.plot(sig_freq,20*np.log10(np.abs(sig_spec)))
            plt.show() 
        return sig_freq, sig_spec, sig_t

    def find_freq_bien(self,spec,freq,graf):
        abs_a_spec=[]
        
        # Приведем к модулю, для дальейших вычеслений
        for i in range(len(spec)):
            abs_a_spec.append(abs(spec[i]))
        abs_a_spec[0]=1
        # Ограничим спектр обработки, чтобы игнорировать нулевую гармонику
        for index,frq in enumerate(freq):
            if  frq>4 * 10**9:
                obrez_abs_a_spec = abs_a_spec[:index]
                ind_2 = index
                break

        for index,frq in enumerate(freq):
            if  2*10**7<frq:
                obrez_abs_a_spec = obrez_abs_a_spec[index:]
                mac_ampl_spec = max(obrez_abs_a_spec)
                ind_1 = index
                break
        
        index = abs_a_spec.index(mac_ampl_spec) # Индекс максимальной грамноники, в исходном спектре 

        Fb = freq[index] # Частота биений
        R_target = Fb * self.c * self.imp_t /16 / self.df # Поиск дальности до цели
        #print("Частота биений ", Fb, "Гц")
        #print("Расстояние до цели ", R_target , "м")
        if graf == 1:
            plt.title("Спектр сиганала")
            plt.ylabel("Мощность, В/Гц")
            plt.xlabel("Время, с")
            plt.plot(freq,20*np.log10(abs_a_spec))
            plt.show()
                
        center=[]
        min_porog=[]
        resh=[]
        sred=[]
        for index in range(ind_1,ind_2,1):
             # Окно для принятия решения
            ignore=[]# Игнорируемы точки
            school=[]# Точки для поиска ско
            resh_tochka=abs_a_spec[index]  

            for ign in range(0,9):
                
                try:
                    ignore.append(abs_a_spec[index+ign]) # Область игнорируемых точек
                except IndexError:                # В оба направления в количестве 10 точек
                    pass                          # Обработчик ошибок для неполного заполенения в граничных условиях
                
                try:
                    ignore.append(abs_a_spec[index-ign])
                except IndexError:
                    pass
                   
                
            for sch in range(9,80):                # Область точек, в которых вычисляем СКО
                                                    # В оба направления в количестве 60 точек
                                                    # Обработчик ошибок для неполного заполнения в граничных условиях
                try:
                    school.append(abs_a_spec[index+sch])                   
                except IndexError:
                    pass

                try:
                    school.append(abs_a_spec[index-sch])                   
                except IndexError:
                    #school.append(100) 
                    pass
                
            Fl=5*10**-8  # Вероятность ложной тревоги
           
            sigma,mean = self.find_sko(school)  # Находим СКО и мат. ожидание
            
            porog = sct.norm.ppf(q=1-Fl,scale = sigma)+mean  # Вычисления порога с фикисрованной ложной тревогой, но разным СКО
                         
            if resh_tochka >= porog:   #  Принятие решения о наличии сигнала в области решающих точек
                min_porog.append(porog)
                center.append(index)   # Добавление в массив точек, где превышен порог
                resh.append(resh_tochka)
                sred.append(mean)
                plt.plot(freq[index],20*np.log10(resh_tochka), "x", color ='r')   # Крестик по критерию Н-П
        try:
            maximum_ampl = max(resh)
            #print(resh)
            #print(maximum_ampl)
            for index, ampl in enumerate(resh):
                if ampl == maximum_ampl:
                    i_dex = index
                    break
            F_bien = freq[center[i_dex]]
            y=list(reversed(abs_a_spec))
            shym=y[:1000]
            sred_shym = np.mean(shym)
            #print(sred_shym)
            #print(maximum_ampl)
            OSH = maximum_ampl/sred_shym
            R_target = F_bien * self.c * self.imp_t /8 / self.df # Поиск дальности до цели
            print("Частота биений ", F_bien, "Гц")
            print("Расстояние до цели ", R_target , "м")
            print("Отношение сигнал - шум ", OSH)
        except ValueError:
            R_target=-1
            OSH = 0 
            pass
        if graf == 1:
            plt.title("Спект сигнала после фотодетектора")
            plt.ylabel("Мощность, В/Гц")
            plt.xlabel("Время, с")        
            plt.plot(freq,20*np.log10(abs_a_spec))
         
            plt.show()
        return R_target, OSH
    
    def find_sko(self,sig):  # Функция поиска СКО
        try:            
            sigma  = np.std(sig)            
            mean = np.mean(sig) # Математическое ожидание
        except RuntimeWarning:
            pass    
        return sigma,mean # Возвращаем значения СКО и мат. ожидание   

            