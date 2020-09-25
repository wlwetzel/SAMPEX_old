sys.path.append('/home/wyatt/pyModules')
import sys
import IRBEM
import numpy as np
import pandas as pd
import math as m
from scipy import signal
sys.path.append('/home/wyatt/Downloads/irbem-code-r620-trunk/python')
import SAMP_Data
import SAMPEXreb
import datetime
import matplotlib.animation as animation
from Wavelet import wavelet,wave_signif,invertWave
from scipy.integrate import simps

"""
TODO:

fix SAMPEXreb.quick_read, which no longer exists cause im stupid
Im putting it in SAMP_Data, and I need to rerganize stuff
Also need to rework IRBEM into this
"""

def to_equatorial(position,pitch):
    lstar = model.make_lstar(position,{'Kp':0})
    bmin = lstar['bmin']
    blocal = lstar['blocal']
    eq_pitch = np.arcsin(np.sqrt(np.sin(np.deg2rad(pitch))**2 * bmin[0] / blocal[0]))
    return np.rad2deg(eq_pitch)

def readInTimes(start,end):
    workDir = '/home/wyatt/Documents/SAMPEX/'
    filename = "/home/wyatt/Documents/SAMPEX/data/HILThires/"
    if len(str(start.dayofyear))==1:
        day = '00'+str(start.dayofyear)
    elif len(str(start.dayofyear))==2:
        day = '0'+str(start.dayofyear)
    else:
        day = str(start.dayofyear)
    filename = filename + 'State1/hhrr1993' + day+'.txt'
    data = SAMP_Data.quick_read(filename,start,end)
    times = data.index.values
    if data.empty:
        #SAMPEX occasionally misses data, we'll just skip this section
        return 'Pass'
    return times

def findPitches(times,interpolate=True):

    if type(times) == str:
        print('SAMPEX is likely missing data from this period')
        emptyDF = pd.DataFrame(data=None)
        return emptyDF

    real_Start = pd.to_datetime(times[0],utc=True)
    real_End = pd.to_datetime(times[-1],utc=True)

    dataObj = SAMP_Data.OrbitData(date=real_Start)
    pitchInfo = dataObj.read_time_range(real_Start,real_End,parameters=['A11', 'A21', 'A31', 'A12', 'A22',
    'A32', 'A13', 'A23', 'A33','GEI_X','GEI_Y','GEI_Z'])

    if interpolate:
        try:
            pitchInfo = pitchInfo.resample('100ms').asfreq()
            pitchInfo = pitchInfo.interpolate(method='polynomial',order=7)
            return pitchInfo
        except:
            #sometimes the sampex data is missing inconsistently so that my other
            #methods of skipping bad data don't work
            print('exceptsasfd')
            emptyDF = pd.DataFrame(data=None)
            return emptyDF
    else:
        return pitchInfo

def wavAnalysis(data):
    mean = np.mean(data)
    maxScale=1.5
    variance = np.std(data, ddof=1) ** 2
    n = len(data)
    dt = .1
    dj = .05
    s0 = 2*dt
    j1 = int(12 / dj)
    mother = 'MORLET'
    pad = 1
    dataSeries = pd.Series(data=data)
    lag1 = dataSeries.autocorr(lag=1)

    wave, period, scale, coi = wavelet(data, dt, pad, dj, s0, j1, mother)
    power = np.abs(wave)**2

    signif = wave_signif(([variance]), dt=dt, sigtest=0, scale=scale,
                                            lag1=lag1, mother=mother)
    sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])  # expand signif --> (J+1)x(N) array
    sig95 = power / sig95  # where ratio > 1, power is significant
    mask = sig95>1
    denoisedWav = mask * wave
    denoisedDat = invertWave(mother, scale, dt, dj, denoisedWav) + mean
    mask = scale<maxScale
    mask = mask[:,np.newaxis].dot(np.ones(n)[np.newaxis,:])
    denoisedWav *=mask
    newdat = invertWave(mother, scale, dt, dj, denoisedWav)
    return newdat,denoisedDat

def peakFind(start,end):
    workDir = '/home/wyatt/Documents/SAMPEX/'
    filename = "/home/wyatt/Documents/SAMPEX/data/HILThires/"
    if len(str(start.dayofyear))==1:
        day = '00'+str(start.dayofyear)
    elif len(str(start.dayofyear))==2:
        day = '0'+str(start.dayofyear)
    else:
        day = str(start.dayofyear)
    filename = filename + 'State1/hhrr1993' + day+'.txt'
    data = SAMP_Data.quick_read(filename,start,end)
    data = data.drop(columns=['Rate5','Rate6'])

    if data.empty:
        #bug with missing sampex data, functions further downstream can handle
        #empty dataFrames, wavelet cannot
        print('No data found, SAMPEX is likely missing this time period')
        emptyDF = pd.DataFrame(data=None,columns=['Rate1','Rate2','Rate3','Rate4'])
        return emptyDF

    times = data.index
    burstDF = pd.DataFrame(data=None,index=times)

    cols = ['Rate1','Rate2','Rate3','Rate4']

    for col in cols:
        shortScale, _ = wavAnalysis(data[col].to_numpy())
        burstDF[col] = shortScale


    peakList = [] # for holding the dataframes for each col, they wont be the same
                # length to start so this is easiest


    baseDF = data - burstDF
    for col in cols:
        peaks , _ = signal.find_peaks(burstDF[col],height=100)
        peakTimes = times[peaks]
        peakDF = burstDF[col].loc[peakTimes]
        peakDF = peakDF.drop(peakDF[peakDF / baseDF[col].loc[peakTimes]<1].index )
        peakList.append(peakDF)

    col1 = peakList[0].index
    col2 = peakList[1].index
    col3 = peakList[2].index
    col4 = peakList[3].index
    dt = pd.Timedelta('00:00:00.3')
    peakDF = pd.DataFrame(columns =[ 'Rate1','Rate2','Rate3','Rate4'])
    # associate peaks with one another
    for i in range(len(col1.values)):
        for j in range(len(col2.values)):
            if abs(col1[i] - col2[j]) < dt:
                for k in range(len(col3.values)):
                    if abs(col1[i] - col3[k]) < dt:
                        for l in range(len(col4.values)):
                            if (
                                abs(col1[i] - col4[l]) < dt and
                                abs(col2[j] - col3[k]) < dt and
                                abs(col2[j] - col4[l]) < dt and
                                abs(col3[k] - col4[l]) < dt
                                ):
                                tempDF = pd.DataFrame(data={'Rate1':peakList[0].iloc[i] ,
                                                            'Rate2':peakList[1].iloc[j] ,
                                                            'Rate3':peakList[2].iloc[k] ,
                                                            'Rate4':peakList[3].iloc[l]
                                                            }, index = [col1[i]])
                                peakDF = peakDF.append(tempDF)

    pitchInfo = findPitches(times)

    if pitchInfo.empty:
        #bug with missing sampex data, functions further downstream can handle
        #empty dataFrames
        print('No data found, SAMPEX is likely missing data this time period')
        emptyDF = pd.DataFrame(data=None,columns=['Rate1','Rate2','Rate3','Rate4'])
        return emptyDF

    peakTimes = peakDF.index
    pitchAtPeaks = pitchInfo.loc[peakTimes]
    for col in pitchAtPeaks.columns:
        peakDF[col] = pitchAtPeaks[col]

    peakDF = peakDF.drop(columns=['Year','Day-of-year','Sec_of_day'])
    return peakDF

def getBins(alpha,beta,flux=1):
    """Handles negative angles, which don't represent actual pitch angles-they
    are mirrored onto the positive side

    Parameters
    ----------
    flux : float
        Flux in a 100ms bin.
        Default Value is for generating look direction data, should only be used
        as outArr,_ = getBins(alpha,beta)
    alpha : float
        'Left' limit of FOV of detector.
    beta : float
        'Right' limit of FOV of detector.

    Returns
    -------
    np.array,np.array
        The divided angle data, and the weights corresponding to the burst

    """
    if flux==1:
        binNum = 10
    else:
        binNum=100

    if (alpha < 0 and beta > 0 ) or (alpha > 0 and beta < 0 ):
        #choose bins proportionally
        aBins = int(abs(alpha)/  (abs(alpha) + abs(beta) ) * binNum)
        bBins = binNum - aBins
        alphaDat = np.linspace(0,abs(alpha),aBins)
        betaDat = np.linspace(0,abs(beta),bBins)
        weights = np.array(binNum * [flux/binNum])
        outArr = np.append(alphaDat, betaDat)
        return outArr,weights
    else:
        outArr = np.linspace(alpha, beta,binNum)
        weights = np.array([flux/binNum] * binNum)
        return outArr,weights

def binning(peakDF,pitchAtPeaks):
    """
    binning each microburst
    """
    #if peakDF is empty, we didnt find any microbursts and we should just pass
    if peakDF.empty:
        weightList = np.array([])
        binList = np.array([])
    else:
        binList = []
        weightList = []
        names = pd.DataFrame(data={'Rate1':['det_1_Alpha','det_1_Beta'] ,
                                   'Rate2':['det_2_Alpha','det_2_Beta'],
                                   'Rate3':['det_3_Alpha','det_3_Beta'] ,
                                   'Rate4':['det_4_Alpha','det_4_Beta']})
        cols = ['Rate1','Rate2','Rate3','Rate4']

        for col in cols:
            for i in range(len(peakDF.index)):
                testPitches = pitchAtPeaks.iloc[i]
                alpha = testPitches[names[col][0]]
                beta = testPitches[names[col][1]]
                flux = peakDF.iloc[i].loc[col]
                bins,weights = getBins(alpha, beta,flux)
                binList.append(bins)
                weightList.append(weights)

        weightList = np.array(weightList).flatten()
        binList = np.array(binList).flatten()
    return weightList,binList

if __name__ == '__main__':
    month = '03'
    eventList = []
    hours = ['0' + str(i) if i<10 else str(i) for i in range(24)]
    days = ['0' + str(i) if i<10 else str(i) for i in range(1,32)]
    for day in days:
        for hour in hours:
            eventList.append([pd.Timestamp('1993-'+month +'-'+day+ ' ' +hour+ ':00:00' , tz = 'utc'),
                              pd.Timestamp('1993-'+month +'-'+day+ ' ' +hour+ ':29:59' , tz = 'utc')])
            eventList.append([pd.Timestamp('1993-'+month +'-'+day+ ' ' +hour+ ':29:59' , tz = 'utc'),
                              pd.Timestamp('1993-'+month +'-'+day+ ' ' +hour+ ':59:59' , tz = 'utc')])
    #%%
    """
    writing data
    """
    #%%
    i=1
    with open('/home/wyatt/Documents/SAMPEX/generated_Data/burst_Mar93.csv','a') as file:
        for event in eventList:
            print(i/48)
            i+=1
            start = event[0]
            end = event[1]
            peakDF = peakFind(start, end)
            print(peakDF)
            peakDF.to_csv(file,header = False)
