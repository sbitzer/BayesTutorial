#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates Bayesian estimation of a two-dimensional mean as data
points arrive sequentially. Depending on the chosen prior and data 
distributions it can show many different effects, e.g.:
    - narrowing of posterior for more data points
    - that posterior will include true mean most of the time (calibrated)
    - that posterior will not include true mean most of the time (uncalibrated)
    - that posterior is influenced by prior, but data overwrites that influence

When run, the script opens a figure window in which increasingly more data 
points and the corresponding posterior is shown as one clicks into the plotted
axis.
    
Created on Wed Nov 23 16:44:02 2016

@author: bitzer
"""

import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import lines
plt.style.use('bmh')


#%% data generated from prior
prior_mu = np.zeros(2)
prior_std = 3
prior_corr = 0
prior_cov = np.array( [[prior_std**2, prior_std**2 * prior_corr],
                       [prior_std**2 * prior_corr, prior_std**2]] )

#real_mu = np.r_[1.0, -4.0]
real_mu = np.random.multivariate_normal(prior_mu, prior_cov)
real_std = 2
real_corr = 0
real_cov = np.array( [[real_std**2, real_std**2 * real_corr],
                       [real_std**2 * real_corr, real_std**2]] )

N = 20
data = np.random.multivariate_normal(real_mu, real_cov, size=N)


#%% initialise plot
def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

def get_cov_ellipse_params(cov, volume=.9):
    """Parameters for drawing an ellipse enclosing a desired probability mass.
    
    Returns width, height and angle of an ellipse enclosing *volume* based on 
    the specified covariance matrix (*cov*).

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        volume : The volume inside the ellipse; defaults to 0.5
        
    based on a function by Noah Haskell Silbert:
    http://www.nhsilbert.net/source/2014/06/bivariate-normal-ellipse-plotting-in-python/
    """

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    width, height = 2 * np.sqrt(chi2.ppf(volume,2)) * np.sqrt(vals)
    
    return width, height, theta


figsz = (10, 10)
fig = plt.figure(figsize=figsz)
ax = plt.axes(aspect='equal')
plt.axis([-10, 10, -10, 10])

# common ellipse properties
ellprops = {'facecolor': 'none', 'lw': 2}

# plot prior
width, height, theta = get_cov_ellipse_params(prior_cov)
prior_ell = Ellipse(xy=prior_mu, width=width, height=height, angle=theta, 
                    edgecolor=next(ax._get_lines.prop_cycler)['color'], 
                    **ellprops)
ax.add_artist(prior_ell)

# handles for legend
lh = [lines.Line2D([], [], color=prior_ell.get_ec(), label='prior')]

# plot dots
dots = plt.plot(0, 0, '.', visible=False)[0]
lh.append(lines.Line2D([], [], ls='none', marker='.', color=dots.get_color(), 
                       label='data'))

# sample mean
smean = plt.plot(0, 0, '+', mew=2, visible=False)[0]
lh.append(lines.Line2D([], [], ls='none', marker='+', color=smean.get_color(), 
                       label='sample mean'))

# plot posterior of mean
width, height, theta = get_cov_ellipse_params(prior_cov)
post_ell = Ellipse(xy=prior_mu, width=width, height=height, angle=theta, 
                   edgecolor=next(ax._get_lines.prop_cycler)['color'], 
                   visible=False, **ellprops)
ax.add_artist(post_ell)
lh.append(lines.Line2D([], [], color=post_ell.get_ec(), label='posterior'))

# plot real covariance
width, height, theta = get_cov_ellipse_params(real_cov)
lik_ell = Ellipse(xy=prior_mu, width=width, height=height, angle=theta, 
                   edgecolor=next(ax._get_lines.prop_cycler)['color'], 
                   ls='--', label='real cov', **ellprops)
ax.add_artist(lik_ell)
lh.append(lines.Line2D([], [], ls='--', color=lik_ell.get_ec(), label='real cov'))

plt.legend(handles=lh)

plt.title('ready! click to proceed!')


#%% visualise posterior after seeing increasingly many data points
def compute_posterior(dp, mu0, cov0, cov_lik):
    prec0 = np.linalg.inv(cov0)
    prec_lik = np.linalg.inv(cov_lik)
    cov_pos = np.linalg.inv( prec0 + prec_lik )
    
    return np.dot(cov_pos, np.dot(prec_lik, dp) + np.dot(prec0, mu0)), cov_pos

class clicker:
    def __init__(self, data, mu, cov, dots, post_ell, lik_ell, smean):
        self.data = data
        self.N = data.shape[0]
        self.mu = mu
        self.cov = cov
        
        self.post_ell = post_ell
        self.lik_ell = lik_ell
        self.dots = dots
        self.smean = smean
        self.cid = dots.figure.canvas.mpl_connect('button_press_event', self)

        self.n = -1
        
        
    def __call__(self, event):
        if event.inaxes!=self.dots.axes: 
            return

        if self.n < 0:
            self.dots.set_visible(True)
            self.smean.set_visible(True)
            self.post_ell.set_visible(True)
        
        self.n += 1
        
        if self.n == self.N:
            self.lik_ell.center = real_mu
            self.lik_ell.axes.plot(real_mu[0], real_mu[1], 'x', mew=2,
                                   color=self.lik_ell.get_ec())
            
            self.dots.axes.set_title('')
            self.dots.figure.canvas.draw()
        elif self.n < self.N:
            # update dots
            self.dots.set_data(self.data[:self.n+1, 0], self.data[:self.n+1, 1])
            
            # update sample mean
            self.smean.set_data(np.mean(self.data[:self.n+1, 0]), 
                                np.mean(self.data[:self.n+1, 1]))
            
            # compute posterior for current n
            self.mu, self.cov = compute_posterior(data[self.n, :], self.mu, 
                                                  self.cov, real_cov)
            
            # update posterior ellipse
            width, height, theta = get_cov_ellipse_params(self.cov)
            self.post_ell.center = self.mu
            self.post_ell.width = width
            self.post_ell.height = height
            self.post_ell.angle = theta
            
            # update figure
            self.dots.axes.set_title('%d more' % (self.N-self.n))
            self.dots.figure.canvas.draw()



clicker_instance = clicker(data, prior_mu, prior_cov, dots, post_ell, lik_ell, 
                           smean)

plt.show()