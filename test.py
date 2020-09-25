import sys
import pandas as pd
sys.path.append('/home/wyatt/pyModules')
from wavelet import *
path = '/home/wyatt/Documents/SAMPEX/test_dat.csv'
import plotly.express as px
import plotly.graph_objs as go
data = pd.read_csv(path)
data = data['Rate1']
def wave_subtract(data):
    mean = np.mean(data)
    maxScale=60
    variance = np.std(data, ddof=1) ** 2
    n = len(data)
    dt = .1
    dj = .02
    s0 = 2*dt
    j1 = int(12 / dj)
    mother = 'MORLET'
    pad = 1
    dataSeries = pd.Series(data=data)
    lag1 = dataSeries.autocorr(lag=1)

    wave, period, scale, coi = wavelet(data, dt, pad, dj, s0, j1, mother)
    power = np.abs(wave)**2
    # denoisedWav = wave
    signif = wave_signif(([variance]), dt=dt, sigtest=0, scale=scale,
    lag1=lag1, mother=mother)
    sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])  # expand signif --> (J+1)x(N) array
    sig95 = power / sig95  # where ratio > 1, power is significant
    mask = sig95>1
    denoisedWav = mask * wave
    power = np.abs(denoisedWav)**2
    mask = scale<maxScale
    mask = mask[:,np.newaxis].dot(np.ones(n)[np.newaxis,:])
    denoisedWav *=mask
    denoisedDat = invert_wavelet(mother, scale, dt, dj, denoisedWav)
    return denoisedDat

# n = len(data.to_numpy())
# denoisedDat = wave_subtract(data.to_numpy())
# fig = px.line(x = list(range(n)),y=denoisedDat)
# fig.add_scatter(x = list(range(n)),y=data)

rolled = data.rolling(50,win_type='parzen',center=True).mean()
rolled2 = rolled.rolling(100,win_type='parzen',center=True).mean()
n = len(rolled)
subtracted = data-.8*rolled
fig = px.line(data)
fig.add_scatter(x=list(range(n)),y=rolled)
fig.add_scatter(x=list(range(n)),y=rolled2)
# fig.update_yaxes(type='log')
fig.show()
