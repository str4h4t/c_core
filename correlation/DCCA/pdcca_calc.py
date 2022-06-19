import numpy as np
from numpy.matlib import repmat

def sliding_window(xx, k):
    idx = np.arange(k)[None, :] + np.arange(len(xx) - k + 1)[:, None]
    return xx[idx], idx


def compute_dpcca_others(cdata, k):
    nsamples, nvars = cdata.shape

    cdata = cdata - cdata.mean(axis=0)
    xx = np.cumsum(cdata, axis=0)

    F2_dfa_x = np.zeros(nvars)
    allxdif = []
    for ivar in range(nvars):  # do for all vars
        xx_swin, idx = sliding_window(xx[:, ivar], k)
        nwin = xx_swin.shape[0]
        b1, b0 = np.polyfit(np.arange(k), xx_swin.T, deg=1)  # linear fit (UPDATE if needed)

        x_hatx = repmat(b1, k, 1).T * repmat(range(k), nwin, 1) + repmat(b0, k, 1).T

        xdif = xx_swin - x_hatx
        allxdif.append(xdif)
        F2_dfa_x[ivar] = (xdif ** 2).mean()

    dcca = np.zeros([nvars, nvars])
    for i in range(nvars):  # do for all vars
        for j in range(nvars):  # do for all vars
            F2_dcca = (allxdif[i] * allxdif[j]).mean()
            dcca[i, j] = F2_dcca / np.sqrt(F2_dfa_x[i] * F2_dfa_x[j])

    C = np.linalg.inv(dcca)

    # (Clear but slow version)
    dpcca = np.zeros([nvars,nvars])
    for i in range(nvars):
       for j in range(nvars):
           dpcca[i,j] = -C[i,j]/np.sqrt(C[i,i]*C[j,j])

    # DPCCA (oneliner version)
    mydiag = np.sqrt(np.abs(np.diag(C)))
    dpcca = (-C / repmat(mydiag, nvars, 1).T) / repmat(mydiag, nvars, 1) + 2 * np.eye(nvars)

    # Include correlation and partial corr just for comparison ;)
    # Compute Corr

    # Get parCorr
    cov = np.cov(cdata.T)
    C0 = np.linalg.inv(cov)
    mydiag = np.sqrt(np.abs(np.diag(C0)))
    parCorr = (-C0 / repmat(mydiag, nvars, 1).T) / repmat(mydiag, nvars, 1) + 2 * np.eye(nvars)

    return dcca, dpcca, parCorr


def executor(data, k):
    cdata = data.T
    #k = 6
    dcca, dpcca, parcorr = compute_dpcca_others(cdata,k)
    return dcca, dpcca, parcorr