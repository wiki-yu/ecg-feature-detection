import numpy as np
import matplotlib.pyplot as plt

try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib
import scipy.signal as signal


class Detectors:
    """ECG feature detection algorithms
    maximum & minimum deflection
    maximum minimum dv/dt
    """
    
    def __init__(self, fs = False):
        self.fs = fs
        self.delay = 0

    def ecg_detector(self, unfiltered_ecg):
        f1 = 48/self.fs
        f2 = 52/self.fs
        b, a = signal.butter(4, [f1*2, f2*2], btype='bandstop')
        # print("butter b: ", b)
        # print("butter a: ", a)
        filtered_ecg = signal.lfilter(b, a, unfiltered_ecg)
        print("filtered_ecg shape: ", np.shape(filtered_ecg))

        ori_diff = np.zeros(len(filtered_ecg))
        for i in range(1, len(ori_diff)):
            ori_diff[i] = filtered_ecg[i] - filtered_ecg[i-1]

        diff = np.zeros(len(filtered_ecg))
        for i in range(4, len(diff)):
            diff[i] = filtered_ecg[i] - filtered_ecg[i-4]
        print("shape diff[i]: ", np.shape(diff))

        ci = [1, 4, 6, 4, 1]        
        low_pass = signal.lfilter(ci, 1, diff)
        print("low pass shape: ", np.shape(low_pass))
   
        low_pass[:int(0.2*self.fs)] = 0
        print("low pass shape: ", np.shape(low_pass))
        plt.figure()
        plt.plot(low_pass)
        plt.title("low_pass2 signal")
      
        ms200 = int(0.2*self.fs)
        ms1200 = int(1.2*self.fs)        
        ms160 = int(0.16*self.fs)
        neg_threshold = int(0.01*self.fs)

        # Real time electrocardiogram QRS detection using combined adaptive threshold
        M = 0
        M_list = []
        neg_m = []
        MM = []
        M_slope = np.linspace(1.0, 0.6, ms1200-ms200)

        QRS = []
        r_peaks = []
        r_bottoms = []
        max_dvdts = []
        min_dvdts = [] 

        counter = 0

        thi_list = []
        thi = False
        thf_list = []
        thf = False
        newM5 = False

        # fs=360, neg_threshold=3
        # fs=1000, neg_threshold=10
        print("neg_threshold: ", neg_threshold)

        for i in range(len(low_pass)):
            # M
            # Initially M = 0.6*max(Y) is set for the first 5s of the sigal, where at least 2 QRS complexes should occur
            if i < 5 * self.fs:
                M = 0.6 * np.max(low_pass[:i+1])
                MM.append(M)
                # A buffer with 5 steep-slope threshold values is preset
                if len(MM) > 5:
                    MM.pop(0)
                # print("MM:", i, QRS)
            # No detection is allowed 200 ms after the current one
            elif QRS and i < QRS[-1] + ms200:
                # a new value of M5 is calculated:
                newM5 = 0.6 * np.max(low_pass[QRS[-1]: i])
                # The estimated newM5 value can become quite high, if steep slope premature ventricular contraction 
                # or artifact appeared, and for that reason it is limited to
                if newM5 > 1.5 * MM[-1]:
                    newM5 = 1.1 * MM[-1]
                # print("2elif QRS!!!!!!!!!!!!!!!!!!", i)

            elif newM5 and QRS and i == QRS[-1] + ms200:
                MM.append(newM5)
                if len(MM)>5:
                    MM.pop(0)    
                # M is calculated as an average value of MM
                M = np.mean(MM)    
                # print("3elif QRS:@@@@@@@@@@@@@@@@@@@@@@", i)
            
            elif QRS and i > QRS[-1] + ms200 and i < QRS[-1] + ms1200:
                M = np.mean(MM) * M_slope[i-(QRS[-1] + ms200)]
                # print("4elif QRS:#####################", i)

            elif QRS and i > QRS[-1] + ms1200:
                M = 0.6*np.mean(MM)
                # print("5elif QRS:$$$$$$$$$$$$$$$$$$$$$$$$$", i)

            M_list.append(M)
            neg_m.append(-M)

            # potential peak shows up
            if not QRS and low_pass[i] > M:
                print("########################### ", i)
                QRS.append(i)
                thi_list.append(i)
                thi = True
            
            elif QRS and i > QRS[-1] + ms200 and low_pass[i] > M:
                print("************************** ", i)
                QRS.append(i)
                thi_list.append(i)
                thi = True

            # following process after potential peak shows up 
            # within ms160 after the peak
            if thi and i < thi_list[-1] + ms160:
                print("111, i: ", i)
                if low_pass[i] < -M and low_pass[i-1] > -M:
                    print("## close to bottom, i:{}, -M:{}, low_pass[i]:{} ".format(i, -M, low_pass[i]))
                    #thf_list.append(i)
                    thf = True
                    
                if thf and low_pass[i] < -M:
                    thf_list.append(i)
                    counter += 1
                    print("&& counter+1 ##, i:{}, -M:{}, low_pass[i]:{}, counter:{} ".format(i, -M, low_pass[i], counter))

                
                elif low_pass[i] > -M and thf:
                    print("&& climb from bottom, i:{}, -M:{}, low_pass[i]:{} ".format(i, -M, low_pass[i]))
                    counter = 0
                    thi = False
                    thf = False
            # processing outside the peak area
            elif thi and i > thi_list[-1] + ms160:
                print("222, i: ", i)
                counter = 0
                thi = False
                thf = False                                        
            
            if counter > neg_threshold:
                print("saving!!!")
                unfiltered_section = unfiltered_ecg[thi_list[-1] - int(0.01*self.fs): i]
                # print("np.argmax(unfiltered_section)：", np.argmax(unfiltered_section))
                # print("thi_list[-1]: ", thi_list[-1])
                # print("int(0.01*self.fs): ", int(0.01*self.fs))
                r_peaks.append(self.delay + np.argmax(unfiltered_section) + thi_list[-1] - int(0.01*self.fs))
                r_bottoms.append(self.delay + np.argmin(unfiltered_section) + thi_list[-1] - int(0.01*self.fs))

                ori_diff_section = ori_diff[thi_list[-1] - int(0.01*self.fs): i]
                print("ori_diff_section: ", ori_diff_section)

                max_dvdts.append(self.delay + np.argmax(ori_diff_section) + thi_list[-1] - int(0.01*self.fs))
                min_dvdts.append(self.delay + np.argmin(ori_diff_section) + thi_list[-1] - int(0.01*self.fs))
                counter = 0
                thi = False
                thf = False
        print("M_list: ", np.shape(M_list))

        # removing the 1st detection as it 1st needs the QRS complex amplitude for the threshold
        # r_peaks.pop(0)
        plt.figure(figsize=(20, 4))
        plt.plot(unfiltered_ecg)
        plt.plot(r_peaks, unfiltered_ecg[r_peaks], 'ro')
        plt.plot(r_bottoms, unfiltered_ecg[r_bottoms], 'go')
        plt.plot(max_dvdts, unfiltered_ecg[max_dvdts], 'bo')
        plt.plot(min_dvdts, unfiltered_ecg[min_dvdts], 'yo')
        plt.title("Detected features")
        plt.show()
        print("r peaks: ", r_peaks)
        return r_peaks