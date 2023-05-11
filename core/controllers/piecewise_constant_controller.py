from .controller import Controller
from numpy import rint, argmin
from math import floor


class PiecewiseConstantController(Controller):
    def __init__(self, dynamics, h, ut, round_mode=True):
        Controller.__init__(self, dynamics)
        # assuming ut shape = (batch, T, m)
        self.ut = ut
        self.h = h
        self.round_mode_on = round_mode

    def forward(self, x, t):
        #technically floor should be correct but floating point arithmetic
        #can make you choose the wrong slice

        if type(self.h) is float:
            if self.round_mode_on:
                t_idx = int(rint(t / self.h))
            else:
                t_idx = floor(t / self.h)
                if t_idx == self.ut.shape[1]:
                    #off by one can happen because of floating point errors
                    t_idx -= 1
        else:
            t_idx = argmin(abs(self.h - t))
        if t_idx >= self.ut.shape[1]:
            raise OverflowError('[ERROR] Controller called outside horizon.')
        return self.ut[:, t_idx]
