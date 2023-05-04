from .controller import Controller
import torch as th

class MPCController(Controller):
    def __init__(self, dynamics, trajopt):
        super().__init__(dynamics)
        self.trajopt = trajopt
        self.dynamics = dynamics
        self.xt_prev = None
        self.ut_prev = None

    def forward(self, x, t):
        ws_ut = ws_xt = None
        if self.xt_prev is not None:
            ws_xt = th.cat([
             x[:, None],
             self.xt_prev[:, 2:],
             self.xt_prev[:, -1][:, None]
            ], dim=1)
        if self.ut_prev is not None:
            ws_ut = th.cat([
                self.ut_prev[:, 1:],
                self.ut_prev[:, -1][:, None]
            ], dim=1)

        xt, ut = self.trajopt(x, t,
                              xt_prev=ws_xt,
                              ut_prev=ws_ut)
        self.xt_prev = xt
        self.ut_prev = ut
        return ut[:, 0]
