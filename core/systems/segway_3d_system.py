from torch import cat, cos, diag, float64, norm, \
    sin, stack, tensor, zeros, set_default_dtype, pow
from torch import tanh
from torch import atan2
from torch.nn import Module, Parameter
import torch as th

from core.dynamics import SystemDynamics, AffineDynamics
from core.util import  default_fig
from numpy import pi
set_default_dtype(float64)

model = th.tensor([44.798,  # mb
                      2.485,  # mw
                      0.055936595310797,  # Jw
                      -0.02322718759275,  # a2
                      0.166845864363019,  # c2
                      3.604960049044268,  # A2
                      3.836289730154863,  # B2
                      1.069672194414735,  # C2
                      1.261650363363571,  # K
                      0.195,  # r
                      0.5,  # L
                      9.81,  # gGravity
                      0.,  # FricCoeff 3.185188257847262
                      1.0e-3,  # velEps
                      1.225479467549329  # FricCoeff 1.225479467549329
                      ], requires_grad=True)

class Segway3D(SystemDynamics, AffineDynamics, Module):
    def __init__(self):
        SystemDynamics.__init__(self, 7, 2)
        Module.__init__(self)

    # void dynamics(const double t,
    #               const double X[STATE_LENGTH],
    #               const double U[INPUT_LENGTH],
    #                     double xDot[STATE_LENGTH])
    # {
    # 	double g[STATE_LENGTH*INPUT_LENGTH];
    # 	double Fric;
    # 	double a_tmp;
    # 	double b_a_tmp;
    # 	double f_tmp;
    # 	double b_f_tmp;
    # 	double c_f_tmp;
    # 	double d_f_tmp;
    # 	double e_f_tmp;
    # 	double f_f_tmp;
    # 	double g_f_tmp;
    # 	double h_f_tmp;
    # 	double i_f_tmp;
    # 	double j_f_tmp;
    # 	double k_f_tmp;
    # 	double l_f_tmp;
    # 	double m_f_tmp;
    # 	double n_f_tmp;
    # 	double o_f_tmp;
    # 	double p_f_tmp;
    # 	double q_f_tmp;
    # 	double r_f_tmp;
    # 	double s_f_tmp;
    # 	double t_f_tmp;
    # 	double u_f_tmp;
    # 	double v_f_tmp;
    # 	double f_tmp_tmp;
    # 	double b_f_tmp_tmp;
    # 	double w_f_tmp;
    # 	double x_f_tmp;
    #
    # /*  */
    def forward(self, X, U, t):
        Fric = X[3] - X[6] * model[9];
        Fric = model[12] * tanh(Fric / model[13]) + model[14] * Fric;
        a_tmp = cos(X[5]);
        b_a_tmp = sin(X[5]);
        xDot_0 = X[3] * cos(X[2]);
        xDot_1 = X[3] * sin(X[2]);
        xDot_2 = X[4];
        f_tmp = model[3] * model[3];
        b_f_tmp = model[9] * model[9];
        c_f_tmp = model[4] * model[4];
        d_f_tmp = model[0] * model[0];
        e_f_tmp = 4.0 * f_tmp;
        f_f_tmp = 4.0 * c_f_tmp;
        g_f_tmp = X[4] * X[4];
        h_f_tmp = X[6] * X[6];
        i_f_tmp = 4.0 * h_f_tmp + 3.0 * g_f_tmp;
        j_f_tmp = cos(2.0 * X[5]);
        k_f_tmp = cos(3.0 * X[5]);
        l_f_tmp = pow(model[3], 3.0);
        m_f_tmp = 4.0 * model[6] * model[4] * model[0];
        n_f_tmp = pow(model[4], 3.0);
        o_f_tmp = sin(2.0 * X[5]);
        p_f_tmp = model[5] * model[4] * model[0] * model[9] * g_f_tmp;
        q_f_tmp = -model[4] * model[7] * model[0] * model[9] * g_f_tmp;
        r_f_tmp = sin(3.0 * X[5]);
        s_f_tmp = 3.0 * f_tmp * model[4] * d_f_tmp * model[9] * g_f_tmp;
        t_f_tmp = -4.0 * model[3] * model[4];
        u_f_tmp = 2.0 * model[3] * model[4];
        v_f_tmp = f_tmp * d_f_tmp;
        f_tmp_tmp = v_f_tmp * b_f_tmp;
        b_f_tmp_tmp = c_f_tmp * d_f_tmp * b_f_tmp;
        f_f_tmp = 1.0 / (
                (((((((((4.0 * model[6] * model[2] + e_f_tmp * model[2] *
                         model[0]) + f_f_tmp * model[2] * model[0]) + 2.0 *
                       model[6] * model[0] *
                       b_f_tmp) + f_tmp_tmp) + b_f_tmp_tmp) + 4.0 * model[6] *
                    model[1] * b_f_tmp)
                   + e_f_tmp * model[0] * model[1] * b_f_tmp) + f_f_tmp *
                  model[0] * model[1] * b_f_tmp) + (f_tmp + -c_f_tmp) *
                 d_f_tmp * b_f_tmp * j_f_tmp) + u_f_tmp * d_f_tmp * b_f_tmp *
                o_f_tmp);
        w_f_tmp = 2.0 * f_tmp;
        xDot_3 = 0.5 * model[9] * f_f_tmp * (
                ((((((((((((((((((((((-8.0 * model[6] *
                                      Fric + -8.0 * f_tmp * Fric * model[
                                          0]) + -8.0 * c_f_tmp * Fric * model[
                                         0]) +
                                    model[0] * model[9] * ((((-8.0 * model[
                        4] * Fric + model[3] * (-model[5] +
                                                model[7]) * g_f_tmp) + 4.0 *
                                                             model[3] * model[
                                                                 6] * (
                                                                     h_f_tmp + g_f_tmp)) +
                                                            l_f_tmp * model[
                                                                0] * i_f_tmp) +
                                                           model[3] * c_f_tmp *
                                                           model[0] * i_f_tmp) *
                                    a_tmp) + t_f_tmp * model[11] * d_f_tmp *
                                   model[9] * j_f_tmp) + model[3] *
                                  model[5] * model[0] * model[
                                      9] * g_f_tmp * k_f_tmp) + -model[3] *
                                 model[7] *
                                 model[0] * model[
                                     9] * g_f_tmp * k_f_tmp) + l_f_tmp * d_f_tmp *
                                model[9] *
                                g_f_tmp * k_f_tmp) + -3.0 * model[
                                   3] * c_f_tmp * d_f_tmp * model[9] *
                               g_f_tmp * k_f_tmp) + 8.0 * model[3] * Fric *
                              model[0] * model[9] * b_a_tmp)
                             + m_f_tmp * h_f_tmp * model[
                                 9] * b_a_tmp) + e_f_tmp * model[4] * d_f_tmp *
                            h_f_tmp * model[
                                9] * b_a_tmp) + 4.0 * n_f_tmp * d_f_tmp * h_f_tmp *
                           model[9]
                           * b_a_tmp) + p_f_tmp * b_a_tmp) + m_f_tmp * model[
                             9] * g_f_tmp * b_a_tmp) +
                        q_f_tmp * b_a_tmp) + s_f_tmp * b_a_tmp) + 3.0 * n_f_tmp * d_f_tmp *
                      model[9]
                      * g_f_tmp * b_a_tmp) + w_f_tmp * model[11] * d_f_tmp *
                     model[9] * o_f_tmp) +
                    -2.0 * c_f_tmp * model[11] * d_f_tmp * model[
                        9] * o_f_tmp) + p_f_tmp *
                   r_f_tmp) + q_f_tmp * r_f_tmp) + s_f_tmp * r_f_tmp) + -n_f_tmp * d_f_tmp *
                model[9] * g_f_tmp * r_f_tmp);
        e_f_tmp = model[10] * model[10];
        i_f_tmp = -2.0 * f_tmp;
        k_f_tmp = 2.0 * c_f_tmp;
        l_f_tmp = i_f_tmp * model[0];
        m_f_tmp = k_f_tmp * model[0];
        n_f_tmp = f_tmp * model[0];
        c_f_tmp = c_f_tmp * model[0];
        p_f_tmp = model[4] * model[0];
        q_f_tmp = model[2] * e_f_tmp;
        e_f_tmp = e_f_tmp * model[1];
        r_f_tmp = 2.0 * (model[7] + n_f_tmp);
        s_f_tmp = 2.0 * (model[5] + c_f_tmp);
        u_f_tmp = u_f_tmp * model[0] ;
        xDot_4 = b_f_tmp * X[4] * (
                (-2.0 * model[3] * model[0] * X[3] * a_tmp + t_f_tmp *
                 model[0] * X[6] * j_f_tmp) + -2.0 * (
                        p_f_tmp * X[3] + (((model[5] + -model[7])
                                           + l_f_tmp) + m_f_tmp) * X[
                            6] * a_tmp) * b_a_tmp) * (1.0 / ((((q_f_tmp +
                                                                e_f_tmp * b_f_tmp) + r_f_tmp * b_f_tmp * (
                                                                       a_tmp * a_tmp)) + s_f_tmp *
                                                              b_f_tmp * (
                                                                      b_a_tmp * b_a_tmp)) + u_f_tmp * b_f_tmp * o_f_tmp));
        xDot_5 = X[6];
        t_f_tmp = 4.0 * model[4] * model[11];
        k_f_tmp = k_f_tmp * model[2] * model[0];
        m_f_tmp = m_f_tmp * model[1] * b_f_tmp;
        x_f_tmp = -(model[4] * model[4]) * d_f_tmp;
        xDot_6 = f_f_tmp * (
                (((((((((((((((((((8.0 * Fric * model[2] + 4.0 * Fric *
                                   model[0] * b_f_tmp) + 8.0 * Fric * model[
                                      1] * b_f_tmp) + 2.0 * model[0] *
                                 (2.0 * model[4] * Fric * model[9] + model[3] *
                                  model[11] * (2.0 * model[2] +
                                               (model[0] + 2.0 * model[
                                                   1]) * b_f_tmp)) * a_tmp) + -2.0 *
                                model[3] * model[4]
                                * model[0] * (
                                        model[0] * h_f_tmp * b_f_tmp + -2.0 * (
                                            model[2] + model[1] *
                                            b_f_tmp) * g_f_tmp) * j_f_tmp) + t_f_tmp *
                               model[2] * model[0] * b_a_tmp) +
                              -4.0 * model[3] * Fric * model[0] * model[
                                  9] * b_a_tmp) + 2.0 * model[4] *
                             model[
                                 11] * d_f_tmp * b_f_tmp * b_a_tmp) + t_f_tmp *
                            model[0] * model[1] *
                            b_f_tmp * b_a_tmp) + v_f_tmp * h_f_tmp * b_f_tmp * o_f_tmp) + x_f_tmp *
                          h_f_tmp * b_f_tmp * o_f_tmp) + -2.0 * model[5] *
                         model[2] * g_f_tmp *
                         o_f_tmp) + 2.0 * model[7] * model[
                            2] * g_f_tmp * o_f_tmp) + i_f_tmp * model
                       [2] * model[
                           0] * g_f_tmp * o_f_tmp) + k_f_tmp * g_f_tmp * o_f_tmp) + -
                     model
                     [5] * model[0] * b_f_tmp * g_f_tmp * o_f_tmp) + model[7] *
                    model[0] *
                    b_f_tmp * g_f_tmp * o_f_tmp) + -2.0 * model[5] * model[1]
                   * b_f_tmp * g_f_tmp * o_f_tmp) + 2.0 * model[7] * model[1]
                  * b_f_tmp * g_f_tmp * o_f_tmp) + l_f_tmp * model[1] *
                 b_f_tmp * g_f_tmp * o_f_tmp) + m_f_tmp * g_f_tmp * o_f_tmp);
        t_f_tmp = x_f_tmp * b_f_tmp;
        l_f_tmp = (((((((2.0 * model[6] * model[2] + w_f_tmp * model[2] * model[
            0]) +
                        k_f_tmp) + model[6] * model[0] * b_f_tmp) + f_tmp_tmp) +
                     b_f_tmp_tmp) + 2.0 * model[6] * model[
                        1] * b_f_tmp) + w_f_tmp *
                   model[0] * model[1] * b_f_tmp) + m_f_tmp;
        j_f_tmp = -f_tmp * d_f_tmp * b_f_tmp;
        i_f_tmp = model[3] * model[4] * d_f_tmp * b_f_tmp * o_f_tmp;
        g_f_tmp = p_f_tmp * model[9] * a_tmp;
        h_f_tmp = -model[3] * model[0] * model[9] * b_a_tmp;
        Fric = model[8] * model[9] * (
                (((model[6] + n_f_tmp) + c_f_tmp) + g_f_tmp) +
                h_f_tmp);
        g_3 = Fric * (1.0 / (((l_f_tmp + t_f_tmp * (a_tmp * a_tmp)) + j_f_tmp *
                               (b_a_tmp * b_a_tmp)) + i_f_tmp));
        g_10 = Fric * (
            1.0 / (((l_f_tmp + t_f_tmp * (a_tmp * a_tmp)) + j_f_tmp *
                    (b_a_tmp * b_a_tmp)) + i_f_tmp));
        t_f_tmp = r_f_tmp * model[9];
        l_f_tmp = q_f_tmp * (1.0 / model[9]) + e_f_tmp * model[9];
        j_f_tmp = s_f_tmp * model[9];
        i_f_tmp = u_f_tmp * model[9] * o_f_tmp;
        g_4 = -model[8] * model[10] * (
            1.0 / (((l_f_tmp + t_f_tmp * (a_tmp * a_tmp))
                    + j_f_tmp * (b_a_tmp * b_a_tmp)) + i_f_tmp));
        g_11 = model[8] * model[10] * (
            1.0 / (((l_f_tmp + t_f_tmp * (a_tmp * a_tmp))
                    + j_f_tmp * (b_a_tmp * b_a_tmp)) + i_f_tmp));
        g_0 = th.tensor(0.0);
        g_1 = th.tensor(0.0);
        g_2 = th.tensor(0.0);
        g_5 = th.tensor(0.0);

        g_7 = th.tensor(0.0);
        g_8 = th.tensor(0.0);
        g_9 = th.tensor(0.0);
        g_12 = th.tensor(0.0);
        t_f_tmp = -2.0 * model[8] * (
                (((2.0 * model[2] + model[0] * b_f_tmp) + 2.0 *
                  model[1] * b_f_tmp) + g_f_tmp) + h_f_tmp) * f_f_tmp;
        g_6 = t_f_tmp;
        g_13 = t_f_tmp;

        g_u1 = th.stack([g_0, g_1, g_2, g_3, g_4, g_5, g_6])
        g_u2 = th.stack([g_7, g_8, g_9, g_10, g_11, g_12, g_13])
        g = th.stack([g_u1, g_u2], dim=1)
        xdot = th.stack([xDot_0, xDot_1, xDot_2, xDot_3, xDot_4, xDot_5,
                         xDot_6])
        return xdot + g @ U