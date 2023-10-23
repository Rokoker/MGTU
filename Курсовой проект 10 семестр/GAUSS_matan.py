# Импорт бибилиотек
import matplotlib.pyplot as plt
import numpy as np
import pylab
import math
import random
import scipy.fft as sp
from scipy import signal
from scipy.optimize import curve_fit
from statistics import mean
import scipy.stats as sct




class LLS(object):
    
    #Зададим параметры лазера исходя из условия
    c = 3 * 10**9   # Скорость света

    # Параметры затухания сигнала в среде (Модель сильного тумана)
    # Где alpha - коэф. общего ослабления сигнала
    # betta - коэф. переизлучения сигнала 
    alpha=0.012
    betta=0.002
    koef_appertura =  1/8            # koef_appertura - коэф. показывающий долю переотраженного сигнала, попадающего в аппертуру приемника
                                     # т.к. переотражение происходит по полному теленому углу, а примем мы лишь некоторую часть
                                     # Позже будет необходимо будет пересчитать в зависимости от расстояния до цели и диаметра пучка импульса
    koef_target = 1/2                #- коэф. отраженного импульса от нашей цели
                                     # В дальнейшем пересчитать от угла наклона цели и ее поверхности
    imp_A = 20                       # Амплитуда импульса
    a = 4*10**(-9)                   # Коэф определяющий ширину гауссовского импульса
        
    def generator(self):
        izl_sig_A=[]                    # Массив под амплитуды        
        imp_lambda = 905 * 10**(-6)     # Длина волны импульса        
        imp_f = self.c / imp_lambda     # Частота импульса        
        self.imp_t = 50 * 10**(-9)      # Длителььность импульса        
        self.t_int = self.imp_t/10000   # Интервал взятия отсчетов        
        self.disc_f = 1/self.t_int      # Частота дискретизации, пересчитана        
                                        # из интервала взятия отсчетов                                        
        sig_t=np.arange(0,self.imp_t + self.t_int,self.t_int)  # Временная область для отображения импульса
        self.n_sig= len(sig_t)                    # Количество отсчетов для отображения импульса
        sig_a_for_SF=[]                           # Массив амплитуд, для дальнейшей свертки в СФ        
        for time in sig_t:            
            amp_imp = self.imp_A*math.exp(-(time-self.imp_t/2)**2/(2*self.a**2))            
            izl_sig_A.append(amp_imp)                                                 # Массив с амплитудой излученного сигнала
        sig_a_for_SF= izl_sig_A[:]     # Мы точно знаем, что сигнал затухнет, за время своей длительности
        self.sig_for_por=izl_sig_A[:]                        # поэтому учитывает коэф alpha
        return  izl_sig_A, sig_t,sig_a_for_SF                # Вернули амплитуды сигнала и его временную область


    def canal(self,izl_sig, sig_t,R):                   # Канал без потерь с целью
        t_zad = 2*R/self.c                              # Задержка по веремени в зависимосимости от расстяния R
        self.r_obsh=R
        time_after_target = 2*1000/self.c                                               
        for time in range(0, len(sig_t)):
            sig_t[time] =sig_t[time] + t_zad            # Перенесли время сигнала на время приема
               
        do_sig_t=np.arange(0,t_zad,1/self.disc_f)       # Временная область до приема сигнала
        self.ind_target = len(do_sig_t)-1 
        do_sig_a=[]                                     # Амплитудная область до приема сигнала

        for time in range(len(do_sig_t)):
            do_sig_a.append(0)                          # Амплитуда на приемнике до приема сигнала равна нулю, заполнили нулями
        self.otchet_start_priema = len(do_sig_t)
        do_sig_a.extend(izl_sig)                        # Совмещение амплитудных областей до приема сигнала и приема сигнала
        self.otchet_last_priema = len(do_sig_a)
        do_sig_t = np.append(do_sig_t,sig_t)            # Совмещение временных областей до приема сигнала и приема сигнала

        after_sig_t = np.arange(do_sig_t[-1],time_after_target,1/self.disc_f)       # Временная область после приема сигнала
        after_sig_a = []                                                  # Амплитудная область после приема сигнала

        for time in range(len(after_sig_t)):
            after_sig_a.append(0)                    # Амплитуда на приемнике после приема сигнала равна нулю, заполнили нулями

        do_sig_a.extend(after_sig_a)                 # Совмещение амплитудных областей (до приема, приема сигнала) с областью после приема сигнала
        
        do_sig_t = np.append(do_sig_t,after_sig_t)   # Совмещение временных областей до приема сигнала и приема сигнала

        
        return  do_sig_a, do_sig_t                  # Вернули амплитуды сигнала и его временную область
                    

    def canal_buger(self,sig_a, sig_t,R):
        # Канал с затуханием
        # Так как на момент выполнения данного метода у нас амплитдная область содержит только амплитуду принятого сигнала
        # То мы можем наложить маску затухания на всю область, затухание согласно экспоненте, подробнее в документации
        
        for time in range(len(sig_t)):
            sig_a[time]=sig_a[time]*math.exp(-self.alpha*sig_t[time]*self.c) 
        for time in range(len(sig_t)):                         
            sig_a[time]=self.otr_target(sig_a,time)  # Реакция цели
        return  sig_a, sig_t

    def otr_target(self,ampl,x):        # Импульсная характеристика цели
        
        #otv = ampl[x]*0.25+ampl[x-1]*0.125+ampl[x-2]*0.125  # Дискретная импульсная характеристика
        otv = ampl[x]*0.5        
        return otv
    

    def canal_BGS(self,sig_a):      # Наложение БГШ
                
        for time in range(0, len(sig_a)):
            sig_a[time] = sig_a[time] + random.gauss(0,1)             # Добавление шума к амплитуде
   
        return  sig_a                             #вернули амплитуды сигнала
    
    def oba_prin_sig(self,amp_ras,amp_sig):
        # Сложим амплитуды переотраженного импульса ( от среды и от цели)
        return amp_ras+amp_sig

    def svertka(self,y_priem,y_izl):                                       # функция свертки
        N=len(y_priem)        # для нормировки
        sig_after_SF = signal.fftconvolve(y_priem,y_izl,mode = "same")     # находим сигнал после свертки
                                                                           # после прохождения СФ
        return sig_after_SF/N

    def N_P(self,sig,time_sig, mod = "No_POR"):
        kurs.SNR(sig,time_sig)
        sig = signal.resample(sig, 500)   # Передескритизация сигнала
        time_sig = np.linspace(0, time_sig[-1], 500, endpoint=False)      
        plt.figure()      
        center=[]
        min_porog=[]
        for index in range(0,len(sig),1):
            resh=[] # Окно для принятия решения
            ignore=[]# Игнорируемы точки
            school=[]# Точки для поиска ско
            resh.append(sig[index])      
            
            for ign in range(0,10):
                
                try:
                    ignore.append(sig[index+ign]) # Область игнорируемых точек
                except IndexError:                # В оба направления в количестве 10 точек
                    pass                          # Обработчик ошибок для неполного заполенения в граничных условиях
                
                try:
                    ignore.append(sig[index-ign])
                except IndexError:
                    pass                   
                
            for sch in range(10,70):                # Область точек, в которых вычисляем СКО
                                                    # В оба направления в количестве 60 точек
                                                    # Обработчик ошибок для неполного заполнения в граничных условиях
                try:
                    school.append(sig[index+sch])                   
                except IndexError:
                    pass

                try:
                    school.append(sig[index-sch])                   
                except IndexError:
                    pass

            Fl=5*10**-9  # Вероятность ложной тревоги          
            sigma,mean = self.find_sko(school)  # Находим СКО и мат. ожидание           
            porog = sct.norm.ppf(q=1-Fl,scale = sigma)+mean  # Вычисления порога с фикисрованной ложной тревогой, но разным СКО     
            
            for k in range(len(resh)):                
                if resh[k] >= porog:   #  Принятие решения о наличии сигнала в области решающих точек
                    min_porog.append(porog)
                    center.append(index)   # Добавление в массив точек, где превышен порог                    
                    plt.plot(time_sig[index+k],resh[k], "x", color ='r')   # Крестик по критерию Н-П
                    
        plt.vlines(time_sig, ymin=0, ymax=sig)  # Вывод графика сигнала
        if mod == "POR":
            plt.title("Сигнал с ПОР")
        elif mod =="MINUS_POR":
            plt.title("Сигнал с удаленной ПОР")
        else:
            plt.title("Сигнал без ПОР")  
        plt.xlabel("Время, с")
        plt.ylabel("Амплитуда, В")
        plt.show()        
        
        try:
            try:               
                S = time_sig[center[int(len(center)/2)]]/2*self.c-35  # Вычисление расстояние до цели
                if time_sig[center[-1]]/2*self.c-35>self.r_obsh+50:
                    print("Fl")
                print("Растояние до цели, оценено по моменту первой точки, где выолнился критерий Н-П")
                print(f"До цели {S} метров")
                a=1
                return a
            except UnboundLocalError:
                print("Цель не обнаружена")
                return 0
        except IndexError:
                print("Цель не обнаружена")
                return 0
        
    def POR(self,sig_a,sig_t): # Нахождение импульсной характеристики ПОР и ее свертка с сигналом
        # Три массива под разные амплитуда рассеяния
        # amp_ras - итоговый массив амплитуд ПОР
        amp_ras=np.zeros(len(sig_a))
        # amp_ras_alpha - полное рассеяние импульса
        amp_ras_alpha=np.zeros(len(sig_a))
        # amp_ras_betta - рассеяние имульса только на переотражение
        amp_ras_betta=np.zeros(len(sig_a))
        for time in range(len(sig_t)):            
            amp_ras_alpha[time]=1*math.exp(-self.alpha*sig_t[time]*self.c)              # Амплитуда имульса, после ослабления                
            amp_ras_betta[time]=1*math.exp(-(self.alpha-self.betta)*sig_t[time]*self.c) # Амплитуда импульса, после ослабления только
                                                                                                 # на переотражение                   

            amp_ras[time]=(amp_ras_betta[time]-amp_ras_alpha[time])
                                                            #*self.koef_appertura  # Разница двух предыдущих амплитуд, дает амплитуду,
                                                            # переотраженную от среды
                                                            # koef_appertura - коэф. показывающий долю переотраженного сигнала,
                                                            # попадающего в аппертуру приемника
        self.kek=amp_ras[:]
        for time in range(len(sig_t)):                                                    # т.к. переотражение происходит по полному теленому углу, а примем мы лишь некоторую часть
            if time >= self.otchet_start_priema:               
                amp_ras[time]=self.otr_target(amp_ras,time)
                # Разница двух предыдущих амплитуд, дает амплитуду,
                                                            # переотраженную от среды
                                                            # koef_appertura - коэф. показывающий долю переотраженного сигнала,
                                                            # попадающего в аппертуру приемника
                                                            # т.к. переотражение происходит по полному теленому углу, а примем мы лишь некоторую часть
        vuh = signal.fftconvolve(amp_ras,self.sig_for_por, mode = "same")/len(self.sig_for_por) # Свертка импульсной характеристики и излученного сигнала
        
        return vuh  
        
    def spec(self,y,label, mod = 1):                                 # Функция отображения спектра сигнала
        y=sp.fftshift(sp.fft(y)/len(y))
        frq = sp.fftshift(sp.fftfreq(len(y), d = self.t_int))            # Нормировка оси частот
        # Построение графиков
        plt.figure()
        plt.title(label)
        plt.xlabel("Частота,Гц")
        plt.ylabel("Амплитуда, В")
        plt.plot(frq,np.abs(y.real))
        plt.show()

    def graf(self,y,x,Label):                               # Функция построения графиков временной области
        plt.figure()
        plt.title(Label)
        plt.xlabel("Время,с")
        plt.ylabel("Амплитуда, В")
        plt.plot(x,y)
        plt.draw()


    def find_sko(self,sig):  # Функция поиска СКО
        try:            
            sigma  = np.std(sig)            
            mean = np.mean(sig) # Математическое ожидание
        except RuntimeWarning:
            pass    
        return sigma,mean # Возвращаем значения СКО и мат. ожидание

    def minus_por(self,sig,time,naid):
        sig = signal.resample(sig, 500)   # Передескритизация сигнала
        time = np.linspace(0, time[-1], 500, endpoint=False) # Переводиму шкалу времени
        # Делаем это так как моделируем работу АЦП
        y_max = max(sig) # Находим максимальную амплитуду сигнала
        for i in range(len(sig)):
            if y_max == sig[i]:
                i_max = i  # Находим индекс, который представляет максимальную амплитуду сигнала
                break
        for i in range(-10,10,1):
            if sig[i_max + i] > 0.707*sig[i_max]:  # Условие, что максимальная амплитуда, которую мы нашли, распространяется на 10 отсчетов влево и вправо
                i_por = i_max                       # И уровень этих отсчетов больше 0.707 от максимальной амплитуды
                o = sig[i_max]                      # Если это условие верно, сохраняем индекс максимальной амплитуды и ее значение
            else:
                i_por = 0       # Иначе считаем, что ПОР не внесет существенного влияния на определение сигнала
                o=0             # Следовательно приравниваем индекс и амплитуду ПОР к 0
        
        if o == 0:    # Если помеху не удалось обнаружить предыдущим методом, значит амплитуда сигнала больше уровня помехи              
            seek=sig[:i_max-10]     # Но т.к. помеха все еще присутствует, значит второй максимум, слева от первого максимума будет являться пор         
            o = max(seek)           # Ограничим поиск этого максиума на 10 отсчетов влево от первого максиума, исключая возможность принятия сигннала, как помехи
   
        plt.vlines(time[i_por], ymin=0, ymax = sig[i_por], color = "r") # Выведем центр найденой ПОР
        plt.plot(time,sig)
        plt.show()

        # Проведем новый расчет сигнала, чтобы вычесть из него помеху
        # Для этого создадим массив амплитуд нормированной ПОР, который после домножим на амплитуду ПОР, найденную раньше      
        sig_for_minus = kurs.svertka(self.kek,sig_a_for_SF)
        sig_for_minus = sig_for_minus / max(sig_for_minus)
        sig_for_minus = signal.resample(sig_for_minus, 500)
        sig_for_minus = sig_for_minus * o
        for i in range(len(sig)):
            sig_for_minus[i] = sig[i] - sig_for_minus[i] # Вычтем из существующего сигнала модель найденной ПОР             
        plt.plot(time,sig_for_minus)
        plt.show()        
        naid= kurs.N_P(sig_for_minus,t_target,  mod = "MINUS_POR") # Воспользуемся методом поиска сигнала
        

    def SNR(self,ampl, time):

        sig = signal.resample(ampl, 500)   # Передескритизация сигнала
        time = np.linspace(0, time[-1], 500, endpoint=False) # Пересчитали время
        # Делаем это, т.к. моделируем работу АЦП
        ser_imp = self.imp_t/2  # Находим середину импульса
        t_zad = 2*self.r_obsh/self.c      # Находим время задержки сигнала, отраженного от цели                       
        time_seredina_imp = t_zad + ser_imp  # Находим в итоговой временной оси середину импульса
        ignore =[]
        school =[]
        for i in range(len(time)):
            if  time[i] < time_seredina_imp and time[i+1]> time_seredina_imp:
                index = i
                break      
        E=0
        for i in range(-2,3,1):
            E = sig[i+index]**2+ E
        E = E/5            
        for ign in range(0,10):                
            try:
                ignore.append(sig[index+ign]) # Область игнорируемых точек
            except IndexError:                # В оба направления в количестве 10 точек
                pass                          # Обработчик ошибок для неполного заполенения в граничных условиях
                
            try:
                ignore.append(sig[index-ign])
            except IndexError:
                pass                
        for sch in range(10,70):                # Область точек, в которых вычисляем СКО
                                                    # В оба направления в количестве 60 точек
                                                    # Обработчик ошибок для неполного заполнения в граничных условиях
            try:
                school.append(sig[index+sch])                   
            except IndexError:
                pass

            try:
                school.append(sig[index-sch])                   
            except IndexError:
                pass            
        N=0
        for i in range(len(school)):
            N = N + school[i]**2
        N=N/120
        print("Отношение сигнал шум ", E/N)
            
           
detec=0
for r in range(10,180,10):
    r = 140
    if __name__ == "__main__":
        
        print(f"Заданное расстояние {r}")
        kurs = LLS()    
        a_gen, t_gen, sig_a_for_SF = kurs.generator()  # Генерация сигнала
                                                                   
        #kurs.graf(a_gen,t_gen,"Сгенерированный импульс")
        #kurs.spec(a_gen,"Спектр сгенерированного импульса",0)
        
        a_target,t_target = kurs.canal(a_gen,t_gen,r)   # Генерация всей временной шкалы
        #kurs.graf(a_target,t_target,"Принятый сигнал от цели")
        #kurs.spec(a_target,"Спектр принятого сигнал от цели",0)
        
        a_lose_target, t_target = kurs.canal_buger(a_target,t_target,r)   # Учет затухания
        #kurs.graf(a_lose_target,t_target,"Принятый сигнал от цели, с затуханием")
        #kurs.spec(a_lose_target,"Спектр принятого сигнал от цели, с затуханием",0)

        sig_and_BGS = kurs.canal_BGS(a_lose_target)  # Сложение ПОР и Сигнала с Шумом
        #kurs.graf(sig_and_BGS,t_target,"Принятый сигнал от цели с шумом")
        #kurs.spec(sig_and_BGS,"Спектр принятого сигнала от цели с шумом")
        
        sig_after_SF = kurs.svertka(sig_and_BGS,sig_a_for_SF)  # Свертка\СФ
        #kurs.graf(sig_after_SF,t_target,"Принятый сигнал от цели с шумом после СФ")
        #kurs.spec(sig_after_SF,"Спектр принятого сигнала от цели с шумом после СФ")       

        naid= kurs.N_P(sig_after_SF,t_target)       
        
        #########
        # С ПОР #
        #########

        a_POR = kurs.POR(a_lose_target,t_target)   # Генерация ПОР
        a_PORR = kurs.svertka(a_POR,sig_a_for_SF)
        #kurs.graf(a_PORR,t_target,"Принятая ПОР")
        #kurs.spec(a_POR,"Спектр принятой ПОР")
        
        a_POR_and_sig_ang_BGS=kurs.oba_prin_sig(a_POR,sig_and_BGS)      # Сложение ПОР и Сигнала
        #kurs.graf(a_POR_and_sig_ang_BGS,t_target,"ПОР и сигнал от цели с шумом")

        sig_after_SF = kurs.svertka(a_POR_and_sig_ang_BGS,sig_a_for_SF)
       
        naid= kurs.N_P(sig_after_SF,t_target,  mod = "POR")       # Детектирование по локальному минимуму и критерию Н-П

        kurs.minus_por(sig_after_SF,t_target,naid)
        
        
        print("\n")

  

    
    
    

    

