# -*- coding: utf-8 -*-
#
# Copyright (c) 2007-2016 Guillaume Pellerin <yomguy@parisson.com>
# Copyright (c) 2013-2016 Thomas Fillon <thomas@parisson.com>

# This file is part of TimeSide.

# TimeSide is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.

# TimeSide is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with TimeSide.  If not, see <http://www.gnu.org/licenses/>.

# Authors:
# - Antoine Liutkus
# - Fabian-Robert Stöter
# - Thomas Fillon <thomas@parisson.com>

from timeside.core import implements, interfacedoc
from timeside.core.processor import Processor
from timeside.core.api import IEffect
from timeside.core.tools.parameters import store_parameters

from timeside.core.preprocessors import downmix_to_mono, frames_adapter

import numpy as np
from scipy.signal import medfilt

import stft


def logit(w,t,s):
    return 1./(1. + np.exp(-s*(w-t)))


class DummyDenoise(Processor):

    """A dummy analyzer returning random samples from audio frames"""
    implements(IEffect)
    # Define Parameters

    _schema = {'$schema': 'http://json-schema.org/schema#',
               'properties': {'bandwidth': {'default': 100,
                                            'type': 'double'}
                          },
               'type': 'object'}

    @interfacedoc
    @store_parameters
    def __init__(self, bandwidth=100):
        super(DummyDenoise, self).__init__()

        self.bandwidth = 100  # in Hz
        self.eps = 1e-10 # for scale parameter estimation & Wiener mask
        self.fstart = 400  # in Hz (below that threshold: no noise)

        self.fft_size = 2048
        self.fft_hopsize = 1024
        #self.input_blocksize = 2**16
        #self.input_stepsize = 2**16

    @interfacedoc
    def setup(self, channels=None, samplerate=None,
              blocksize=None, totalframes=None):
        super(DummyDenoise, self).setup(channels, samplerate,
                                       blocksize, totalframes)

        self.bandwidth_bins = self.bandwidth * self.fft_size / self.source_samplerate
        self.bandwidth_bins = int(np.floor(self.bandwidth_bins / 2 )*2 + 1)
        self.fmin = self.fstart*self.fft_size / self.source_samplerate


    @staticmethod
    @interfacedoc
    def id():
        return "dummy_denoise"

    @staticmethod
    @interfacedoc
    def name():
        return "Dummy denoise"

    def process(self, frames, eod=False):

        
        
        X = stft.stft(frames, self.fft_size, self.fft_hopsize)

        V = np.abs(X)
        target_spectrum = medfilt(V, kernel_size=[self.bandwidth_bins,1,1])
        noise_spectrum = np.maximum(0,V - target_spectrum)
        mask = target_spectrum/(target_spectrum+noise_spectrum)
        mask = logit(mask,0.5,20)
        
        mask[:self.fmin,...]=1.
        S = X*(1.-mask)

        frames = stft.istft(S, 1, self.fft_hopsize, shape=frames.shape)
        

        return frames, eod
