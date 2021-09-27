import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def linear(x, *params):
    y = np.zeros_like(x)
    k = params[0]
    y0 = params[1]
    y = y0 + k * x
    return y

def linearFitter(x, y, *y0):
    guess = [1, 0]
    if len(y0) == 0:
        try:
            print("Try fitting without bounds")
            popt, pcov = curve_fit(linear, x, y, p0=guess)
        except:
            print("Cannot fit")
            popt = guess
            pcov = 0
    else:
        y0 = y0[0]
        try:
            popt, pcov = curve_fit(linear, x, y, p0=guess, bounds=([-np.inf, y0-0.001], [np.inf, y0+0.001]))
        except ValueError:
            print("Cannot fit with bounds")
            try:
                print("Try fitting without bounds")
                popt, pcov = curve_fit(linear, x, y, p0=guess)
            except:
                print("Still cannot fit")
                popt = guess
                pcov = 0
        except RuntimeError:
            print("The least-squares minimization failed")
            popt = guess
            pcov = 0

    return popt, pcov

def pickPeak(center, guess):
    m = []
    for n in np.arange(0, len(guess), 3):
        diff = [abs(x-guess[n]) for x in center]
        m.extend([diff.index(min(diff))])
    return m

def singleGaussian(x, *params):
    y = np.zeros_like(x)
    ctr = params[0]
    amp = params[1]
    wid = params[2]
    y0 = params[3]
    y = y0 + amp * np.exp( -((x - ctr)/wid)**2)
    return y

def singleGaussianFitter(cbins, counts, pts):
    binNum = cbins.shape[0]
    binSize = cbins[1]-cbins[0]

    guess = np.zeros(4)
    guess[0] = pts[0]
    guess[1] = 300
    guess[2] = 1.0
    guess[3] = pts[1]
    
    ##These are global fitting constraints
    lB = [0, 0, 0, guess[3]*0.8]
    hB = [np.inf, np.inf, np.inf, guess[3]*1.2]

    try:
        popt, pcov = curve_fit(singleGaussian, cbins, counts, p0=guess, bounds=(lB, hB))
    except ValueError:
        print("Cannot fit with bounds")
        try:
            print("Try fitting without bounds")
            popt, pcov = curve_fit(singleGaussian, cbins, counts, p0=guess)
        except:
            print("Still cannot fit")
            popt = guess
            pcov = 0
    except RuntimeError:
        print("The least-squares minimization failed")
        popt = guess
        pcov = 0

    return popt, pcov

##y0 is fixed to zero in this function
def multipleGaussian(x, *params):
    y = np.zeros_like(x)
    for i in np.arange(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
    return y

##This can take variable numbers of peaks
def multipleGaussianFitter(cbins, counts, pts):
    binNum = cbins.shape[0]
    binSize = cbins[1]-cbins[0]
    
    guess = np.zeros(3*len(pts))
    for n in np.arange(len(pts)):
        guess[3*n] = pts[n][0]
        guess[3*n+1] = pts[n][1]
        guess[3*n+2] = 3*binSize

    ##These are global fitting constraints
    lowBound = [[0.02, 0, 0], [0.20, 0, 0], [0.50, 0, 0]]
    highBound = [[0.07, np.inf, np.inf], [0.23, np.inf, np.inf], [0.53, np.inf, np.inf]]

    ##Figure out how many and which sets of bounds to use
    lB = []
    hB = []
    m = pickPeak([0.04, 0.20, 0.51], guess)
    for n in m:
        lB.extend(lowBound[n])
        hB.extend(highBound[n])

    try:
        popt, pcov = curve_fit(multipleGaussian, cbins, counts, p0=guess, bounds=(lB, hB))
    except ValueError:
        print("Cannot fit with bounds")
        try:
            print("Try fitting without bounds")
            popt, pcov = curve_fit(multipleGaussian, cbins, counts, p0=guess)
        except:
            print("Still cannot fit")
            popt = guess
            pcov = 0
    except RuntimeError:
        print("The least-squares minimization failed")
        popt = guess
        pcov = 0
        
    return popt, pcov

def multipleGaussianPlotter(ccbins, popt):
    m = pickPeak([0.04, 0.20, 0.51], popt)
    colors = ['b', 'g', '#db1d1d']
    
    for n in np.arange(len(m)):
        params = popt[3*n:3*n+3]
        fit = multipleGaussian(ccbins, *params)
        plt.plot(ccbins, fit, linewidth=1.5, color=colors[m[n]])
    
    fit = multipleGaussian(ccbins, *popt)
    plt.plot(ccbins, fit, linewidth=1.5, color='k')

def exponential(x, *params):
    y = np.zeros_like(x)
    amp = params[0]
    tau = params[1]
    y0 = params[2]
    y = y0 + amp * np.exp(-x/tau)
    return y

def exponentialFitter(cbins, counts):
    binNum = cbins.shape[0]
    binSize = cbins[1]-cbins[0]

    guess = np.zeros(3)
    guess[0] = np.max(counts)
    guess[1] = 3*binSize
    guess[2] = 0.0

    ##These are global fitting constraints
    lB = [0, 0, 0]
    hB = [np.inf, np.inf, np.min(counts)]

    try:
        popt, pcov = curve_fit(exponential, cbins[np.argmax(counts):], counts[np.argmax(counts):], \
                               p0=guess, bounds=(lB, hB))
    except ValueError:
        print("Cannot fit with bounds")
        try:
            print("Try fitting without bounds")
            popt, pcov = curve_fit(exponential, cbins[2:], counts[2:], p0=guess)
        except:
            print("Still cannot fit")
            popt = guess
            pcov = 0
    except RuntimeError:
        print("The least-squares minimization failed")
        popt = guess
        pcov = 0
        
    return popt, pcov


