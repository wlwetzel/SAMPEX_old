import matplotlib.pyplot as plt
import pandas as pd
from SAMP_Data import *
from scipy import signal
import plotly.express as px
path = '/home/wyatt/Documents/SAMPEX/test_dat.csv'
from itertools import groupby
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
# year = 1992
# day = 278
# date = datetime_from_doy(year,day)
# date = '1992278'
# obj = HiltData(date=date)
# df = px.data.iris()
# data = obj.read(14200,14400)
# data = data[['Rate1','Rate2','Rate3','Rate4']]
# data.to_csv(path)

data = pd.read_csv(path)
time = data.index
x = data['Rate1'].to_numpy()
indices = np.array(list(range(len(x))))
line = lambda x,m,b: m*x+b
def peak_algo(x):
    """
    TODO: need to add condition that peaks be mostly decreasing
    """
    peaks, _ = signal.find_peaks(x,prominence=(500,None))
    peaks = list(peaks)
    #groups successive peaks
    grouped = [list(g) for k, g in groupby(peaks[:-1],lambda x: (peaks[peaks.index(x)+1] -peaks[peaks.index(x)])<5) if k]
    grouped = [item for item in grouped if len(item)>2]
    # filtered = [item for item in grouped if np.all(np.diff(x[item]) >=0)]
    filtered = list(itertools.chain.from_iterable(grouped))
    prominences = signal.peak_prominences(x,filtered)[0]
    #see if a line with negative slope can be reasonably fit
    line_list = []
    for group in grouped:
        popt,pcov = curve_fit(line,group,x[group])
        line_list.append(popt)
    return filtered,prominences,line_list
use_peaks = True
if use_peaks:
    peaks,prominences,line_list = peak_algo(x)
    heights = x[peaks]-prominences

    fig = px.line(x)
    fig.add_scatter( x=peaks,y=x[peaks] ,mode='markers')
    for i in range(len(peaks)):
        fig.add_shape(
                # Line Vertical
                dict(
                    type="line",
                    x0=peaks[i],
                    y0=heights[i],
                    x1=peaks[i],
                    y1=x[peaks[i]],
                    line=dict(
                        color="Black",
                        width=1
                    )
        ))
    fig.update_yaxes(type="log")
    for item in line_list:
        yvals = line(indices,item[0],item[1])
        fig.add_scatter(x=indices,y=yvals)
    fig.show()
# t=np.linspace(0,2,20)
# sawtooth = (signal.sawtooth(np.pi *5*t,width=.5) + 1)*np.exp(-t) + 1.5*np.exp(-t)

def corr_algo(x):
    """
    use a sample decreasing sawtooth and use cross correlation
    """
    sawtooth = np.array([1.5,2.9,1.7,1,1.5,1.9,1.2,.7,1,1.3,.75,.5,.65,.85,.5,.3,.45,.6,.4,.2])
    corr = signal.correlate(sawtooth,x)

    return corr

fs = 1/.1 #sample frequency

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=10):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y
corr_al = False
if corr_al:
    filtered_data = butter_highpass_filter(x,.1,fs)
    corr = signal.correlate(filtered_data,x[1120:1150])
    corr = corr_algo(filtered_data)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    indices = [i for i in range(len(x))]
    fig.add_trace(
        go.Scatter(x=indices, y=x, name="yaxis data"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=indices, y=corr, name="yaxis2 data"),
        secondary_y=True,
    )
    # fig.update_yaxes(type="log")

    fig.show()
