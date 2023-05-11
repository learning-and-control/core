from matplotlib.pyplot import figure
from numpy import array
import numpy as np
import torch
from torch import cos, eye, float64, sin, stack, zeros, tensor, atan2, \
    unbind, ones_like, tan, zeros_like, ones
from torch.nn import Module, Parameter
from core.dynamics import FullyActuatedRoboticDynamics, ObservableDynamics
from core.util import default_fig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
g = 9.80665  # gravitational acceleration

class Quadrotor(FullyActuatedRoboticDynamics, Module, ObservableDynamics):
    """Quadrotor object.

    Conventions are from:

    'Modelling, Identification and Control of a Quadrotor Helicopter'

    Conventions:
    [1] The default inertial frame is fixed and NWU (north, west, up).
    [2] Forward direction is x, left is y in body frame.
    [3] Forward and rear rotors are numbered 1 and 3 respectively. Left and
        right rotors are numbered 4 and 2 respectively. The even rotors
        rotate CCW and the odd rotors rotate CW. When aligned with the inertial
        frame, this means x is aligned with N, y is aligned with W, and
        z is aligned with U.

                          1 ^ x
                            |
                     y      |
                     4 <----o----> 2
                            |
                            |
                            v 3

    [4] Let the quadrotor state be in R^12. The origin and axes of the body
        frame should coincide with the barycenter of the quadrotor and the
        principal axes. Angle states follow ZYX Euler angle conventions.
        (x, y, z): Cartesian position in inertial frame
        (phi, theta, psi): angular position in inertial frame (RPY respectively)
        (u, v, w): linear velocity in body frame
        (p, q, r): angular velocities in body frame
    [5] Vector conventions
        s: State of the quadrotor in order of (o, alpha, dob, dalphab), where
           o is the position and alpha is the vector of angular positions in
           the INERTIAL frame. dob and dalphab are the linear and angular
           velocities in the BODY frame.
        u: Virtual input in order of (ft, taux, tauy, tauz), where ft is the
           total thrust in the +z direction and tauxyz are the angular torques.
           These are all with respect to the BODY frame.
    """

    def __init__(
        self,
        mass: float,
        I: np.ndarray,
        kf: float,
        km: float,
        l: float,
        Jtp: float | None = None,
    ) -> None:
        """Initialize a quadrotor.

        Parameters
        ----------
        mass : float
            Mass of the quadrotor.
        I : np.ndarray, shape=(3,)
            Principal moments of inertia.
        kf : float
            Thrust factor. Internal parameter for rotor speed to body force and
            moment conversion.
        km : float
            Drag factor. Internal parameter for rotor speed to body force and
            moment conversion.
        l : float
            Distance from rotors to center of quadrotor.
        Jtp : Optional[float]
            Total rotational moment of inertia about the propellor axes. Used
            for computing the gyroscopic effect.
        """
        assert mass > 0.0
        assert I.shape == (3,)
        assert np.all(I > 0.0)
        assert kf > 0.0
        assert km > 0.0
        if Jtp is not None:
            assert Jtp > 0.0

        # 12 states, 4 inputs, is a spatially 3D system
        # full-state observation, but the angles are observed as sin/cos pairs: 15-dim
        FullyActuatedRoboticDynamics.__init__(self, 6, 4)
        ObservableDynamics.__init__(self, 15)
        _Jtp = Parameter(tensor(Jtp, dtype=float64, device=device)) if Jtp is not None else None
        self.params = (
            Parameter(tensor(mass, dtype=float64, device=device)),
            Parameter(tensor(I, dtype=float64, device=device)),
            Parameter(tensor(kf, dtype=float64, device=device)),
            Parameter(tensor(km, dtype=float64, device=device)),
            Parameter(tensor(l, dtype=float64, device=device)),
            _Jtp,
        )

    def get_observation(self, s: torch.Tensor) -> torch.Tensor:
        """Returns observations where angles are in sin/cos form."""
        p = s[..., :3]  # x, y, z
        alpha = s[..., 3:6]  # phi, theta, psi
        do_b = s[..., 6:9]  # u, v, w [NOTE] u is notationally overloaded w/input
        dalpha_b = s[..., 9:12]  # p, q, r

        phi = alpha[..., 0]
        theta = alpha[..., 1]
        psi = alpha[..., 2]
        sincos_alpha = torch.stack(
            (
                cos(phi),
                sin(phi),
                cos(theta),
                sin(theta),
                cos(psi),
                sin(psi),
            ),
            dim=-1,
        )  # (..., 6)

        return concatenate((p, sincos_alpha, do_b, dalpha_b), dim=-1)

    def to_principal_coordinates(self, s: torch.Tensor) -> torch.Tensor:
        """Returns coordinates where angles are wrapped."""
        p = s[..., :3]  # x, y, z
        alpha = s[..., 3:6]  # phi, theta, psi
        do_b = s[..., 6:9]  # u, v, w
        dalpha_b = s[..., 9:12]  # p, q, r

        phi = alpha[..., 0]
        theta = alpha[..., 1]
        psi = alpha[..., 2]
        atan2_alpha = torck.stack(
            (
                atan2(cos(phi), sin(phi)),
                atan2(cos(theta), sin(theta)),
                atan2(cos(psi), sin(psi)),
            ),
            dim=-1,
        )  # (..., 3)

        return concatenate((p, atan2_alpha, do_b, dalpha_b), dim=-1)

    @property
    def V(self) -> torch.Tensor:
        """Matrix converting squared rotor speeds to virtual forces/moments.

        Controller design occurs using virtual force/torque inputs, but true
        input is rotor speeds, so once the virtual inputs are designed, they
        are converted (in an invertible way) using _V.

        u = V @ wsq
        """
        _, _, kf, km, l, _ = self.params

        _V = torch.tensor(
            [
                [kf, kf, kf, kf],
                [0.0, -kf * l, 0.0, kf * l],
                [-kf * l, 0.0, kf * l, 0.0],
                [-km, km, -km, km],
            ],
            dtype=float64,
            device=device,
        )

        return _V

    @property
    def invV(self) -> torch.tensor:
        """Matrix converting virtual forces/moments to squared rotor speeds.

        wsq = invV @ u
        """
        _, _, kf, km, l, _ = self.params

        _invV = (
            torch.tensor(
                [
                    [1.0 / kf, 0.0, -2.0 / (kf * l), -1.0 / km],
                    [1.0 / kf, -2.0 / (kf * l), 0.0, 1 / km],
                    [1.0 / kf, 0.0, 2.0 / (kf * l), -1.0 / km],
                    [1.0 / kf, 2.0 / (kf * l), 0.0, 1.0 / km],
                ],
                dtype=float64,
                device=device,
            )
            / 4.0
        )

        return _invV

    def Rwb(self, alpha: torch.Tensor) -> torch.Tensor:
        """Rotation matrix from BODY to WORLD frame (ZYX Euler).

        Parameters
        ----------
        alpha : np.ndarray, shape=(..., 3)
            Roll, pitch, yaw vector.

        Returns
        -------
        R : np.ndarray, shape=(..., 3, 3)
            Rotation matrix from BODY to WORLD frame.
        """
        assert alpha.shape[-1] == 3

        phi = alpha[..., 0]
        theta = alpha[..., 1]
        psi = alpha[..., 2]

        cphi = cos(phi)
        cth = cos(theta)
        cpsi = cos(psi)
        sphi = sin(phi)
        sth = sin(theta)
        spsi = sin(psi)

        row1 = torch.stack(
            (
                cth * cpsi,
                sphi * sth * cpsi - cphi * spsi,
                cphi * sth * cpsi + sphi * spsi,
            ),
            dim=-1,
        )  # (..., 3)
        row2 = torch.stack(
            (
                cth * spsi,
                sphi * sth * spsi + cphi * cpsi,
                cphi * sth * spsi - sphi * cpsi,
            ),
            dim=-1,
        )  # (..., 3)
        row3 = torch.stack((-sth, sphi * cth, cphi * cth), dim=-1)  # (..., 3)
        R = torch.stack((row1, row2, row3), dim=-2)  # (..., 3, 3)

        return R

    def Twb(self, alpha: torch.Tensor) -> torch.Tensor:
        """Angular velocity transformation matrix from BODY to WORLD frame (ZYX Euler).

        Parameters
        ----------
        alpha : torch.Tensor, shape=(..., 3)
            Roll, pitch, yaw vector.

        Returns
        -------
        T : torch.Tensor, shape=(..., 3, 3)
            Angular velocity transformation matrix from BODY to WORLD frame.
        """
        assert alpha.shape[-1] == 3

        phi = alpha[..., 0]
        theta = alpha[..., 1]

        cphi = cos(phi)
        cth = cos(theta)
        sphi = sin(phi)
        tth = tan(theta)

        row1 = torch.stack((ones_like(cth), sphi * tth, cphi * tth), dim=-1)
        row2 = torch.stack((zeros_like(cth), cphi, -sphi), dim=-1)
        row3 = torch.stack((zeros_like(cth), sphi / cth, cphi / cth), dim=-1)
        T = torch.stack((row1, row2, row3), dim=-2)

        return T

    def fdyn(self, t: float, s: torch.Tensor) -> torch.Tensor:
        """Quadrotor autonomous dynamics.

        Parameters
        ----------
        t : float
            Time. Unused, included for API compliance.
        s : torch.Tensor, shape=(..., 12)
            State of quadrotor.

        Returns
        -------
        _fdyn : torch.Tensor, shape=(..., )
            Time derivatives of states from autonomous dynamics.
        """
        assert s.shape[-1] == 12

        # states
        alpha = s[..., 3:6]  # phi, theta, psi
        do_b = s[..., 6:9]  # u, v, w
        dalpha_b = s[..., 9:12]  # p, q, r

        # moments of inertia
        Ix, Iy, Iz = self.params[1]

        # body -> world transformations
        Rwb = self.Rwb(alpha)
        Twb = self.Twb(alpha)

        # velocities
        do = (Rwb @ do_b.unsqueeze(-1)).squeeze(-1)
        dalpha = (Twb @ dalpha_b.unsqueeze(-1)).squeeze(-1)

        # accelerations
        u = do_b[..., 0]
        v = do_b[..., 1]
        w = do_b[..., 2]
        p = dalpha_b[..., 0]
        q = dalpha_b[..., 1]
        r = dalpha_b[..., 2]
        phi = alpha[..., 0]
        th = alpha[..., 1]

        ddo_b = torch.stack(
            (
                r * v - q * w + g * sin(th),
                p * w - r * u - g * sin(phi) * cos(th),
                q * u - p * v - g * cos(th) * cos(phi),
            ),
            dim=-1,
        )  # (..., 3)
        ddalpha_b = torch.stack(
            (
                ((Iy - Iz) * q * r) / Ix,
                ((Iz - Ix) * p * r) / Iy,
                ((Ix - Iy) * p * q) / Iz,
            ),
            dim=-1,
        ) # (..., 3)

        _fdyn = torch.concatenate((do, dalpha, ddo_b, ddalpha_b), dim=-1)
        return _fdyn

    def gdyn(self, t: float, s: torch.Tensor) -> torch.Tensor:
        """Quadrotor control dynamics.

        Parameters
        ----------
        t : float
            Time. Unused, included for API compliance.
        s : torch.Tensor, shape=(..., 12)
            State of quadrotor.

        Returns
        -------
        _gdyn : torch.Tensor, shape=(..., 12, 4)
            Matrix representing affine control dynamics.
        """
        assert s.shape[-1] == 12

        # mass and inertias
        m, I = self.params[:2]
        Ix, Iy, Iz = I

        # accelerations
        ddo_b = zeros(s.shape[:-1] + (3, 4), dtype=float64, device=s.device)
        ddo_b[..., 2, 0] = 1.0 / m

        ddalpha_b = zeros(s.shape[:-1] + (3, 4), dtype=float64, device=s.device)
        ddalpha_b[..., 0, 1] = 1.0 / Ix
        ddalpha_b[..., 1, 2] = 1.0 / Iy
        ddalpha_b[..., 2, 3] = 1.0 / Iz

        _gdyn = torch.concatenate(
            (
                zeros(s.shape[:-1] + (6, 4), dtype=float64, device=s.device),
                ddo_b,
                ddalpha_b,
            ),
            dim=-2,
        )
        return _gdyn

    def forward(
        self,
        s: torch.Tensor,
        u: torch.Tensor,
        t: float,
    ) -> np.ndarray:
        """Quadrotor dynamics function.

        Parameters
        ----------
        s : torch.Tensor, shape=(..., 12)
            State of the quadrotor.
        u : torch.Tensor, shape=(..., 4)
            Virtual input of the quadrotor.
        t : float
            Time. Unused, included for API compliance.

        Returns
        -------
        ds : torch.Tensor, shape=(..., 12)
            Time derivative of the states.
        """
        assert s.shape[-1] == 12
        assert u.shape[-1] == 4

        fdyn = self.fdyn(t, s)
        gdyn = self.gdyn(t, s)

        # (..., 12, 4)
        ds_gyro = zeros(s.shape[:-1] + (12,), dtype=float64, device=s.device)

        # check whether gyroscopic effects are modeled. Note: this is
        # functionally treated as a disturbance, since if it is directly
        # modeled for control, the system is no longer control affine. However,
        # including this in simulations can help test for robustness.
        Jtp = self.params[-1]
        if Jtp is not None:
            Ix, Iy, _ = self.params[1]
            p = s[..., 9]
            q = s[..., 10]

            wsq = self.invV @ u
            assert torch.all(wsq >= 0.0)
            w = torch.sqrt(wsq)
            w[..., 0] *= -1
            w[..., 2] *= -1
            Omega = torch.sum(w, dim=-1)  # net prop speeds

            ds_gyro[..., 9] = -Jtp * q * Omega / Ix
            ds_gyro[..., 10] = Jtp * p * Omega / Iy

        ds = fdyn + (gdyn @ u.unsqueeze(-1)).squeeze(-1) + ds_gyro
        return ds

    def A(self, s: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        """Linearized autonomous dynamics about (s, input).

        Parameters
        ----------
        s : torch.Tensor, shape=(..., 12)
            State.
        input : torch.Tensor, shape=(..., 4)
            Virtual input of the quadrotor. Unused, included for API compliance.

        Returns
        -------
        _A : torch.Tensor, shape=(..., 12, 12)
            Linearized autonomous dynamics about (s, input).
        """
        assert s.shape[-1] == 12

        Ix, Iy, Iz = self.params[1]

        phi = s[..., 3]
        theta = s[..., 4]
        psi = s[..., 5]
        u = s[..., 6]
        v = s[..., 7]
        w = s[..., 8]
        p = s[..., 9]
        q = s[..., 10]
        r = s[..., 11]

        cphi = cos(phi)
        cth = cos(theta)
        cpsi = cos(psi)
        sphi = sin(phi)
        sth = sin(theta)
        spsi = sin(psi)
        tth = tan(theta)

        _A = zeros(s.shape[:-1] + (12, 12))

        _A[..., 0, 3:9] = torch.stack(
            [
                v * (sphi * spsi + cphi * cpsi * sth)
                + w * (cphi * spsi - cpsi * sphi * sth),
                w * cphi * cpsi * cth - u * cpsi * sth + v * cpsi * cth * sphi,
                w * (cpsi * sphi - cphi * spsi * sth)
                - v * (cphi * cpsi + sphi * spsi * sth)
                - u * cth * spsi,
                cpsi * cth,
                cpsi * sphi * sth - cphi * spsi,
                sphi * spsi + cphi * cpsi * sth,
            ],
            dim=-1
        )
        _A[..., 1, 3:9] = torch.stack(
            [
                -v * (cpsi * sphi - cphi * spsi * sth)
                - w * (cphi * cpsi + sphi * spsi * sth),
                w * cphi * cth * spsi - u * spsi * sth + v * cth * sphi * spsi,
                w * (sphi * spsi + cphi * cpsi * sth)
                - v * (cphi * spsi - cpsi * sphi * sth)
                + u * cpsi * cth,
                cth * spsi,
                cphi * cpsi + sphi * spsi * sth,
                cphi * spsi * sth - cpsi * sphi,
            ],
            dim=-1
        )
        _A[..., 2, 3:9] = torch.stack(
            [
                v * cphi * cth - w * cth * sphi,
                -u * cth - w * cphi * sth - v * sphi * sth,
                zeros(s.shape[:-1], dtype=float64, device=device),
                -sth,
                cth * sphi,
                cphi * cth,
            ],
            dim=-1,
        )
        _A[..., 3, 3:5] = torch.stack(
            [
                q * cphi * tth - r * sphi * tth,
                r * cphi * (tth**2.0 + 1.0) + q * sphi * (tth**2.0 + 1.0),
            ],
            dim=-1
        )
        _A[..., 3, 9:12] = torch.stack(
            [ones(s.shape[:-1]), sphi * tth, cphi * tth],
            dim=-1,
        )
        _A[..., 4, 3] = -r * cphi - q * sphi
        _A[..., 4, 10:12] = torch.stack([cphi, -sphi], dim=-1)
        _A[..., 5, 3:5] = torch.stack(
            [
                (q * cphi) / cth - (r * sphi) / cth,
                (r * cphi * sth) / cth**2 + (q * sphi * sth) / cth**2,
            ],
            dim=-1,
        )
        _A[..., 5, 10:12] = torch.stack(
            [sphi / cth, cphi / cth],
            dim=-1
        )
        _A[..., 6, 4] = g * cth
        _A[..., 6, 7:12] = torch.stack(
            [r, -q, zeros(s.shape[:-1]), -w, v],
            dim=-1
        )
        _A[..., 7, 3:12] = torch.stack(
            [
                -g * cphi * cth,
                g * sphi * sth,
                zeros(s.shape[:-1]),
                -r,
                zeros(s.shape[:-1]),
                p,
                w,
                zeros(s.shape[:-1]),
                -u,
            ],
            dim=-1,
        )
        _A[..., 8, 3:12] = torch.stack(
            [
                g * cth * sphi,
                g * cphi * sth,
                zeros(s.shape[:-1]),
                q,
                -p,
                zeros(s.shape[:-1]),
                -v,
                u,
                zeros(s.shape[:-1]),
            ],
            dim=-1,
        )
        _A[..., 9, 10:12] = torch.stack(
            [(r * (Iy - Iz)) / Ix, (q * (Iy - Iz)) / Ix],
            dim=-1
        )
        _A[..., 10, 9:12] = torch.stack(
            [-(r * (Ix - Iz)) / Iy, zeros(s.shape[:-1]), -(p * (Ix - Iz)) / Iy],
            dim=-1,
        )
        _A[..., 11, 9:11] = torch.stack(
            [(q * (Ix - Iy)) / Iz, (p * (Ix - Iy)) / Iz],
            dim=-1,
        )

        return _A

    def B(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Linearized control dynamics about (s, u).

        Parameters
        ----------
        s : np.ndarray, shape=(..., 12)
            State. Unused, included for API compliance.
        u : np.ndarray, shape=(..., 4)
            Virtual input of the quadrotor. Unused, included for API compliance.

        Returns
        -------
        _B : np.ndarray, shape=(..., 12, 4)
            Linearized control dynamics about (s, u).
        """
        m = self.params[0]
        Ix, Iy, Iz = self.params[1]
        _B = zeros(s.shape[:-1] + (12, 4), dtype=float64, device=device)

        _B[..., 8, 0] = 1.0 / m
        _B[..., 9, 1] = 1.0 / Ix
        _B[..., 10, 2] = 1.0 / Iy
        _B[..., 11, 3] = 1.0 / Iz

        return _B

    def jacobian(self, x, u, t):
        return self.A(x, u), self.B(x, u)
