import math
import numpy as np
import torch
%pip install matplotlib
%pip install scipy
%pip install statsmodels
%pip install seaborn
import time
from collections import deque
import matplotlib.pyplot as plt
import scipy.stats as stats  # for Gaussian fitting
from scipy.signal import welch
import torch, torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize


class Positive(nn.Module):
    def forward(self, θ):
        # shift by log2 so the *effective* weight is ≃0 when θ = 0
        return F.softplus(θ) - math.log(2.0)

    # optional but nice: lets you overwrite the tensor later if you ever need to
    def right_inverse(self, W):
        eps = 1e-6
        return torch.log(torch.exp(W + math.log(2.0)) - 1.0 + eps)


########################################################
#             UTILITY / GENERATION FUNCTIONS
########################################################

def location_to_index(loc_deg, n, space_size=180):
    if n <= 1:
        return 0
    frac = loc_deg / float(space_size - 1)
    return int(round(frac * (n - 1)))


def index_to_location(idx, n, space_size=180):
    if n <= 1:
        return 0
    frac = idx / float(n - 1)
    return frac * (space_size - 1)


def make_gaussian_vector_batch_gpu(center_indices, size=180, sigma=5.0, device=None):
    xs = torch.arange(size, dtype=torch.float32, device=device)
    centers = center_indices.view(-1, 1)
    dist = torch.abs(xs - centers)
    return torch.exp(-0.5 * (dist / sigma) ** 2)


def generate_event_loc_seq_batch(
        batch_size=32,
        space_size=180,
        offset_probability=0.1
):
    """
    Creates a batch of location sequences (each T=15 steps).
    Each event is assigned a random location and a modality:
      - 60% Bimodal (B)
      - 20% Audio-only (A)
      - 20% Visual-only (V)
    Probability of starting an event = 0.2
    No overlap between events (duration D=3).
    """
    T = 15
    D = 3
    p_start = 0.2

    loc_seqs = []
    mod_seqs = []
    offset_applied = []
    seq_lengths = []

    for _ in range(batch_size):
        loc_seq = [999] * T
        mod_seq = ['X'] * T

        has_offset = (np.random.rand() < offset_probability)
        t = 0
        while t <= T - D:
            if np.random.rand() < p_start:
                loc_val = np.random.randint(0, space_size)
                r = np.random.rand()
                if r < 0.6:
                    event_mod = 'B'
                elif r < 0.8:
                    event_mod = 'A'
                else:
                    event_mod = 'V'

                for tau in range(D):
                    loc_seq[t + tau] = loc_val
                    mod_seq[t + tau] = event_mod
                t += D
            else:
                t += 1

        loc_seqs.append(loc_seq)
        mod_seqs.append(mod_seq)
        offset_applied.append(has_offset)
        seq_lengths.append(T)

    return loc_seqs, mod_seqs, offset_applied, seq_lengths


def generate_av_batch_tensor(
        loc_seqs,
        mod_seqs,
        offset_applied,
        n=180,
        space_size=180,
        sigma_in=5.0,
        noise_std=0.01,
        loc_jitter_std=0.0,
        stimulus_intensity=1.0,
        device=None,
        max_len=None
):
    """
    Generates audio (xA_batch) and visual (xV_batch) tensors
    (batch_size, T, n_neurons).
    """
    batch_size = len(loc_seqs)

    if max_len is None:
        max_len = max(len(seq) for seq in loc_seqs)

    xA_batch = torch.zeros((batch_size, max_len, n), dtype=torch.float32, device=device)
    xV_batch = torch.zeros((batch_size, max_len, n), dtype=torch.float32, device=device)
    valid_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

    for b in range(batch_size):
        loc_seq = loc_seqs[b]
        mod_seq = mod_seqs[b]
        offset = offset_applied[b]
        T = len(loc_seq)
        valid_mask[b, :T] = True

        for t in range(T):
            loc_t_deg = loc_seq[t]
            if loc_t_deg == 999 or mod_seq[t] == 'X':
                continue

            if loc_jitter_std > 0.0:
                loc_t_deg += np.random.normal(0, loc_jitter_std)
                loc_t_deg = np.clip(loc_t_deg, 0, space_size - 1)

            visual_offset = 0.0
            if offset:
                # small random offset for visual location if offset_applied
                visual_offset = np.random.uniform(-3, 3)

            center_idxA = location_to_index(loc_t_deg, n, space_size)
            center_idxV = location_to_index(loc_t_deg + visual_offset, n, space_size)

            gaussA = make_gaussian_vector_batch_gpu(
                torch.tensor([center_idxA], device=device),
                n, sigma_in, device
            ) * stimulus_intensity

            gaussV = make_gaussian_vector_batch_gpu(
                torch.tensor([center_idxV], device=device),
                n, sigma_in, device
            ) * stimulus_intensity

            if mod_seq[t] == 'A':
                gaussV.zero_()
            elif mod_seq[t] == 'V':
                gaussA.zero_()

            gaussA += torch.randn(1, n, device=device) * noise_std
            gaussV += torch.randn(1, n, device=device) * noise_std

            xA_batch[b, t] = gaussA[0]
            xV_batch[b, t] = gaussV[0]

    return xA_batch, xV_batch, valid_mask


########################################################
#       MULTI-BATCH GPU IZHIKEVICH NETWORK CLASS
########################################################

class MultiBatchAudVisMSINetworkTime(nn.Module):
    """
    Implements a multi-layer spiking network
    (Audio, Visual, MSI excitatory, MSI inhibitory, and Readout) with:
      - A->MSI & V->MSI split into AMPA/NMDA (both excitatory).
      - A->MSI_inh & V->MSI_inh also split into AMPA/NMDA (excitatory).
      - Dedicated MSI_inh -> MSI_exc GABA projection.
      - Dedicated inhibitory projection from unimodal layers (A_inh, V_inh) directly to MSI excit.
      - Tsodyks-Markram short-term depression on AMPA synapses.
      - Conduction delays, STDP for early layers, supervised readout training.
      - Lateral (surround) inhibition in MSI excit.
    """

    def __init__(
            self,
            n_neurons=30,
            batch_size=32,
            lr_unimodal=1e-4,
            lr_msi=1e-4,
            lr_readout=1e-4,
            sigma_in=5.0,
            sigma_teacher=3.0,
            noise_std=0.1,
            single_modality_prob=0.3,
            v_thresh=0.25,
            dt=0.1,
            tau_m=5.0,
            n_substeps=100,
            loc_jitter_std=0.0,
            space_size=180,
            conduction_delay_a2msi=5,
            conduction_delay_v2msi=5,
            conduction_delay_msi2out=5
    ):
        super().__init__()
        self.n = n_neurons  # number of excitatory neurons
        self.batch_size = batch_size
        self.space_size = space_size
        self.sigma_in = sigma_in

        # teacher scheduling
        self.sigma_teacher_init = 6.0
        self.sigma_teacher_final = 2.0
        self.curriculum_epochs = 10

        self.lr_uni = lr_unimodal
        self.lr_msi = lr_msi
        self.lr_out = lr_readout
        self.noise_std = noise_std
        self.v_thresh = v_thresh
        self.dt = dt
        self.tau_m = tau_m
        self.n_substeps = n_substeps
        self.loc_jitter_std = loc_jitter_std

        def pos_init(shape, scale=3.0):
            """
            Return a Parameter θ such that
              W = softplus(θ) – log(2)
            has mean ≈ 0  (because softplus(0)=log 2).
            """
            theta = scale * torch.randn(*shape, device=self.device)  # θ ~ N(0,σ²)
            return nn.Parameter(theta, requires_grad=False)  # θ is stored

        # define an MSI inhibitory subpopulation
        self.n_inh = int(0.3 * n_neurons)
        if self.n_inh < 1:
            self.n_inh = 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.input_scaling = 60.0

        # ------------- Weights: In -> Uni(A/V) --------------
        self.W_inA = pos_init((self.n, self.n), 0.01)
        self.W_inV = pos_init((self.n, self.n), 0.01)

        # ------------- A->MSI (exc) and V->MSI (exc) --------------
        init_a2msi = torch.tensor(0.005 * np.random.randn(self.n, self.n),
                                  dtype=torch.float32, device=self.device)
        init_v2msi = torch.tensor(0.005 * np.random.randn(self.n, self.n),
                                  dtype=torch.float32, device=self.device)

        self.W_a2msi_AMPA = pos_init((self.n, self.n), 0.005 * 0.05 * 0.8)
        self.W_a2msi_NMDA = pos_init((self.n, self.n), 0.005 * 0.95 * 0.8)
        self.W_v2msi_AMPA = pos_init((self.n, self.n), 0.005 * 0.05 * 0.8)
        self.W_v2msi_NMDA = pos_init((self.n, self.n), 0.005 * 0.95 * 0.8)

        # Original feedforward weight boost
        self.W_a2msi_AMPA.mul_(1)
        self.W_a2msi_NMDA.mul_(3)
        self.W_v2msi_AMPA.mul_(1)
        self.W_v2msi_NMDA.mul_(3)

        # ------------- Dedicated Inhibitory Projections from unimodal to MSI excit -------------
        self.W_inA_inh = torch.tensor(0.002 * np.random.randn(self.n, self.n),
                                      dtype=torch.float32, device=self.device)
        self.W_inV_inh = torch.tensor(0.002 * np.random.randn(self.n, self.n),
                                      dtype=torch.float32, device=self.device)

        # ------------- A->MSI_inh (exc) and V->MSI_inh (exc) --------------
        init_a2msi_inh = torch.tensor(0.005 * np.random.randn(self.n_inh, self.n),
                                      dtype=torch.float32, device=self.device)
        init_v2msi_inh = torch.tensor(0.005 * np.random.randn(self.n_inh, self.n),
                                      dtype=torch.float32, device=self.device)

        self.W_a2msiInh_AMPA = pos_init((self.n_inh, self.n), 0.005 * 0.05 * 5.0)
        self.W_a2msiInh_NMDA = pos_init((self.n_inh, self.n), 0.005 * 0.95 * 15.0)
        self.W_v2msiInh_AMPA = pos_init((self.n_inh, self.n), 0.005 * 0.05 * 5.0)
        self.W_v2msiInh_NMDA = pos_init((self.n_inh, self.n), 0.005 * 0.95 * 15.0)
        # ------------- MSI_inh -> MSI_exc (GABA) --------------

        # ------------- MSI_inh -> MSI_exc (GABA) --------------
        self.W_msiInh2Exc_GABA = torch.tensor(0.003 * np.random.randn(self.n, self.n_inh),
                                              dtype=torch.float32, device=self.device)

        self.register_buffer(
            "W_msiInh2Exc_GABA_init", self.W_msiInh2Exc_GABA.clone()
        )

        # ------------- MSI -> Out --------------
        self.W_msi2out = torch.tensor(0.01 * np.random.randn(self.n, self.n),
                                      dtype=torch.float32, device=self.device)

        # ----- inhibitory iSTDP parameters -----
        self.rho0 = 50.0 / 1000  # target rate per sub-step (≈5 Hz)
        self.eta_i = 2e-5  # learning-rate
        self.tau_post_i = 2000.0  # decay of postsyn trace (ms)

        self.allow_inhib_plasticity = True

        self.step_counter = 0
        self.inhib_scaling_T = 2000

        # One trace per *excitatory* postsynaptic neuron
        self.post_i_trace = torch.zeros((self.batch_size, self.n),
                                        dtype=torch.float32,
                                        device=self.device)
        self.rate_avg_tau = 5000.0  # ms (50 s network time)
        self.post_rate_avg = torch.zeros((self.batch_size, self.n),
                                         device=self.device)

        # ------------- Biases --------------
        self.b_uniA = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        self.b_uniV = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        self.b_msi = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        self.b_msi_inh = torch.zeros(self.n_inh, dtype=torch.float32, device=self.device)
        self.b_out = torch.zeros(self.n, dtype=torch.float32, device=self.device)

        self.EI_history = []  # will store ratios for plots/debug

        # ------------- STDP traces --------------
        self.pre_trace_inA = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.post_trace_inA = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.pre_trace_inV = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.post_trace_inV = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.pre_trace_a2msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.post_trace_a2msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.pre_trace_v2msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.post_trace_v2msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)

        # ------------- Izhikevich params --------------
        # For unimodal excit
        self.aA, self.bA, self.cA, self.dA = 0.1, 0.2, -65.0, 2.0
        self.aV, self.bV, self.cV, self.dV = 0.1, 0.2, -65.0, 2.0
        # MSI excit
        self.aM, self.bM, self.cM, self.dM = 0.1, 0.2, -65.0, 2.0
        # MSI inh (fast spiking)
        self.aMi, self.bMi, self.cMi, self.dMi = 0.1, 0.2, -65.0, 2.0
        # Out
        self.aO, self.bO, self.cO, self.dO = 0.1, 0.2, -65.0, 2.0

        # ------------- Membrane potentials & recovery --------------
        # Unimodal excit
        self.v_uniA = torch.full((self.batch_size, self.n), self.cA, device=self.device)
        self.u_uniA = self.bA * self.v_uniA
        self.v_uniV = torch.full((self.batch_size, self.n), self.cV, device=self.device)
        self.u_uniV = self.bV * self.v_uniV

        # MSI excit
        self.v_msi = torch.full((self.batch_size, self.n), self.cM, device=self.device)
        self.u_msi = self.bM * self.v_msi

        # MSI inh
        self.v_msi_inh = torch.full((self.batch_size, self.n_inh), self.cMi, device=self.device)
        self.u_msi_inh = self.bMi * self.v_msi_inh

        # Out
        self.v_out = torch.full((self.batch_size, self.n), self.cO, device=self.device)
        self.u_out = self.bO * self.v_out

        # ------------- Spikes from last substep --------------
        self._latest_sA = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self._latest_sV = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self._latest_sMSI = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self._latest_sMSI_inh = torch.zeros((self.batch_size, self.n_inh), dtype=torch.float32, device=self.device)
        self._latest_sOut = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)

        # ------------- Synaptic currents --------------
        self.tau_syn = 2.5
        self.I_A = torch.zeros((self.batch_size, self.n), device=self.device)
        self.I_V = torch.zeros((self.batch_size, self.n), device=self.device)
        self.I_M = torch.zeros((self.batch_size, self.n), device=self.device)
        self.I_M_inh = torch.zeros((self.batch_size, self.n_inh), device=self.device)
        self.I_O = torch.zeros((self.batch_size, self.n), device=self.device)

        # ------------- Conduction delay buffers --------------
        self.conduction_delay_a2msi = conduction_delay_a2msi
        self.conduction_delay_v2msi = conduction_delay_v2msi
        self.conduction_delay_msi2out = conduction_delay_msi2out

        # unimodal->MSI excit
        self.buffer_a2msi = deque([
            torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
            for _ in range(self.conduction_delay_a2msi)
        ], maxlen=self.conduction_delay_a2msi)
        self.buffer_v2msi = deque([
            torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
            for _ in range(self.conduction_delay_v2msi)
        ], maxlen=self.conduction_delay_v2msi)

        # unimodal->MSI inh (dedicated inhibitory path):
        self.conduction_delay_inA_inh = 5
        self.conduction_delay_inV_inh = 5
        self.buffer_inA_inh = deque([
            torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
            for _ in range(self.conduction_delay_inA_inh)
        ], maxlen=self.conduction_delay_inA_inh)
        self.buffer_inV_inh = deque([
            torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
            for _ in range(self.conduction_delay_inV_inh)
        ], maxlen=self.conduction_delay_inV_inh)

        # unimodal->MSI_inh excit:
        self.conduction_delay_a2msi_inh = 5
        self.conduction_delay_v2msi_inh = 5
        self.buffer_a2msi_inh = deque([
            torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
            for _ in range(self.conduction_delay_a2msi_inh)
        ], maxlen=self.conduction_delay_a2msi_inh)
        self.buffer_v2msi_inh = deque([
            torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
            for _ in range(self.conduction_delay_v2msi_inh)
        ], maxlen=self.conduction_delay_v2msi_inh)

        # MSI_inh->MSI_exc
        self.conduction_delay_msi_inh2exc = 2
        self.buffer_msi_inh2exc = deque([
            torch.zeros((self.batch_size, self.n_inh), dtype=torch.float32, device=self.device)
            for _ in range(self.conduction_delay_msi_inh2exc)
        ], maxlen=self.conduction_delay_msi_inh2exc)

        # MSI->Out
        self.buffer_msi2out = deque([
            torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
            for _ in range(self.conduction_delay_msi2out)
        ], maxlen=self.conduction_delay_msi2out)

        ################################################################
        # NMDA parameters and state variables
        ################################################################
        self.gNMDA = 0.3
        self.tau_nmda = 70.0
        self.nmda_alpha = 0.05
        self.mg_k = 0.1
        self.Erev_nmda = 10.0
        self.tau_nmdaVolt = 100.0
        self.v_nmda_rest = -65.0
        self.nmda_vrest_offset = 7.0
        self.mg_vhalf = -50.0

        # Two dendrites for mg-block gating in MSI excit:
        self.dend_coupling_alpha = 0.05
        self.v_dend_A = torch.full((self.batch_size, self.n), self.cM, device=self.device)
        self.v_dend_V = torch.full((self.batch_size, self.n), self.cM, device=self.device)

        # For NMDA gating in MSI excit:
        self.nmda_m = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.v_nmda = torch.full((self.batch_size, self.n), self.v_nmda_rest, dtype=torch.float32, device=self.device)

        # For MSI_inh, we'll also do NMDA gating if desired:
        self.v_dend_inhA = torch.full((self.batch_size, self.n_inh), self.cMi, device=self.device)
        self.v_dend_inhV = torch.full((self.batch_size, self.n_inh), self.cMi, device=self.device)
        self.nmda_m_inh = torch.zeros((self.batch_size, self.n_inh), dtype=torch.float32, device=self.device)
        self.v_nmda_inh = torch.full((self.batch_size, self.n_inh), self.v_nmda_rest, dtype=torch.float32,
                                     device=self.device)

        ################################################################
        # Short-term depression (Tsodyks-Markram) for AMPA
        ################################################################
        # For the A->MSI and V->MSI excit path:
        self.R_a = torch.ones((self.batch_size, self.n), device=self.device)
        self.u_a = torch.full((self.batch_size, self.n), 0.2, device=self.device)
        self.R_v = torch.ones((self.batch_size, self.n), device=self.device)
        self.u_v = torch.full((self.batch_size, self.n), 0.2, device=self.device)

        # For the A->MSI_inh and V->MSI_inh excit path:
        self.R_a_inh = torch.ones((self.batch_size, self.n_inh), device=self.device)
        self.u_a_inh = torch.full((self.batch_size, self.n_inh), 0.2, device=self.device)
        self.R_v_inh = torch.ones((self.batch_size, self.n_inh), device=self.device)
        self.u_v_inh = torch.full((self.batch_size, self.n_inh), 0.2, device=self.device)

        self.tau_rec = 800.0
        self.tau_fac = 20.0

        exc_names = [
            "W_inA", "W_inV",
            "W_a2msi_AMPA", "W_a2msi_NMDA",
            "W_v2msi_AMPA", "W_v2msi_NMDA",
            "W_a2msiInh_AMPA", "W_a2msiInh_NMDA",
            "W_v2msiInh_AMPA", "W_v2msiInh_NMDA",
        ]
        for name in exc_names:
            parametrize.register_parametrization(self, name, Positive())

        # ------------- surround inhibition in MSI excit (Mexican-hat prior) --------------
        self.g_GABA = 10  # ① global scale  ↑  (was 0.4)

        sigma_inh = 12.0  # ② broader surround
        W_hat = torch.zeros((self.n, self.n), device=self.device)
        for i in range(self.n):
            for j in range(self.n):
                d = min(abs(i - j), self.n - abs(i - j))
                W_hat[i, j] = (
                        math.exp(-(d / sigma_inh) ** 2)  # centre
                        - 0.8 * math.exp(-(d / (4 * sigma_inh)) ** 2)  # surround
                )
        W_hat.clamp_(min=0)
        W_hat *= 6  # ③ three-fold weight boost

        eps = 0.05  # add mild noise
        self.W_MSI_inh = nn.Parameter(  # make it learnable
            W_hat + eps * torch.randn_like(W_hat), requires_grad=False
        )
        self.register_buffer("W_MSI_inh_init", W_hat)  # keep a clean copy

        # self.g_GABA *= 15
        #
        # # 2) Recurrent MSI_inh -> MSI_exc weights
        # self.W_msiInh2Exc_GABA.mul_(20)
        #
        # # 3) Feed‑forward inhibition from A and V layers
        # self.W_inA_inh.mul_(20)
        # self.W_inV_inh.mul_(20)

        self.auto_calibrate_input_gain(high_bound=4000)

        # ------------------------------------------------------------------
        #                       calibration helpers
        # ------------------------------------------------------------------

    def set_inhib_plasticity(self, enable: bool):
        self.allow_inhib_plasticity = enable

    def _probe_spike_sum(self):
        """
        Return MSI spike sum for a 3step Gaussian pulse delivered to the
        centre neuron. Uses the current input_scaling.
        """
        self.EI_history.clear()
        self.R_a.fill_(1.0);
        self.R_v.fill_(1.0)
        self.R_a_inh.fill_(1.0);
        self.R_v_inh.fill_(1.0)
        self.reset_state(batch_size=1)
        pulse = torch.zeros((1, self.n), device=self.device)
        pulse[0, self.n // 2] = 2.0
        total = 0
        for _ in range(3):
            _, _, sM, _ = self.update_all_layers_batch(pulse, pulse)
            total += sM.sum().item()
        return total

    def auto_calibrate_input_gain(self, target=100, tol=3, max_iter=12, high_bound=200):
        """
        Binarysearch self.input_scaling until the singlepulse probe
        produces target � tol MSI spikes.
        """
        print("Calibrating input gain &")
        low, high = 0.0, high_bound
        best_gain = self.input_scaling
        best_err = float('inf')
        was_training = self.training
        self.eval()

        for _ in range(max_iter):
            self.input_scaling = 0.5 * (low + high)
            spikes = self._probe_spike_sum()
            err = abs(spikes - target)
            print(f"  try {self.input_scaling:4.2f} � spike sum {int(spikes)}")

            if err < best_err:
                best_err, best_gain = err, self.input_scaling
            if err <= tol:
                break
            if spikes > target + tol:
                high = self.input_scaling
            else:
                low = self.input_scaling

        self.input_scaling = best_gain

        if was_training:
            self.train()
        print("  mean EI ratio after calibration:",
              np.mean(self.EI_history[-self.n_substeps:]))
        print(f" input_scaling final = {self.input_scaling:4.2f}  "
              f"(spike sum {int(self._probe_spike_sum())})")

    def iSTDP_homeo(self, W, pre_spk, post_spk, lr=1e-4, rho=0.05):
        """
    Vogels-Abbott rule:
    Δw =  η  ·  pre · (post − ρ)
    Keeps postsynaptic rate near target ρ (fraction of sub-steps).
    """

        # Δw = η · (post − ρ) · preᵀ
        dw = lr * torch.bmm(
            (post_spk - rho).unsqueeze(2),  # (B,n_post,1)
            pre_spk.unsqueeze(1)  # (B,1,n_pre)
        ).mean(dim=0)

        with torch.no_grad():
            W.add_(dw)

        # Clamp individual weights ≥0 and L2-norm of each *row* ≤ initial norm

        with torch.no_grad():
            # row-wise L2 clamp to initial norm
            row_norms = W.norm(p=2, dim=1, keepdim=True)
            init_norms = self.W_msiInh2Exc_GABA_init.norm(p=2, dim=1, keepdim=True)
            scale = (row_norms > init_norms).float()
            W.div_(row_norms + 1e-8)
            W.mul_(init_norms * (1 - scale) + row_norms * scale)
            W.clamp_(min=0)

    def reset_state(self, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size

        # reset unimodal
        self.v_uniA = torch.full((self.batch_size, self.n), self.cA, dtype=torch.float32, device=self.device)
        self.u_uniA = self.bA * self.v_uniA
        self.v_uniV = torch.full((self.batch_size, self.n), self.cV, dtype=torch.float32, device=self.device)
        self.u_uniV = self.bV * self.v_uniV

        # reset MSI excit
        self.v_msi = torch.full((self.batch_size, self.n), self.cM, dtype=torch.float32, device=self.device)
        self.u_msi = self.bM * self.v_msi

        # reset MSI inh
        self.v_msi_inh = torch.full((self.batch_size, self.n_inh), self.cMi, dtype=torch.float32, device=self.device)
        self.u_msi_inh = self.bMi * self.v_msi_inh

        # reset out
        self.v_out = torch.full((self.batch_size, self.n), self.cO, dtype=torch.float32, device=self.device)
        self.u_out = self.bO * self.v_out

        self.I_A = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.I_V = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.I_M = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.I_M_inh = torch.zeros((self.batch_size, self.n_inh), dtype=torch.float32, device=self.device)
        self.I_O = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)

        self._latest_sA = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self._latest_sV = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self._latest_sMSI = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self._latest_sMSI_inh = torch.zeros((self.batch_size, self.n_inh), dtype=torch.float32, device=self.device)
        self._latest_sOut = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)

        self.pre_trace_inA = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.post_trace_inA = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.pre_trace_inV = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.post_trace_inV = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.pre_trace_a2msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.post_trace_a2msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.pre_trace_v2msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.post_trace_v2msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)

        # clear conduction buffers
        self.buffer_a2msi.clear()
        self.buffer_v2msi.clear()
        self.buffer_inA_inh.clear()
        self.buffer_inV_inh.clear()
        self.buffer_a2msi_inh.clear()
        self.buffer_v2msi_inh.clear()
        self.buffer_msi_inh2exc.clear()
        self.buffer_msi2out.clear()

        for _ in range(self.conduction_delay_a2msi):
            self.buffer_a2msi.append(torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device))
        for _ in range(self.conduction_delay_v2msi):
            self.buffer_v2msi.append(torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device))
        for _ in range(self.conduction_delay_inA_inh):
            self.buffer_inA_inh.append(torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device))
        for _ in range(self.conduction_delay_inV_inh):
            self.buffer_inV_inh.append(torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device))
        for _ in range(self.conduction_delay_a2msi_inh):
            self.buffer_a2msi_inh.append(
                torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device))
        for _ in range(self.conduction_delay_v2msi_inh):
            self.buffer_v2msi_inh.append(
                torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device))
        for _ in range(self.conduction_delay_msi_inh2exc):
            self.buffer_msi_inh2exc.append(
                torch.zeros((self.batch_size, self.n_inh), dtype=torch.float32, device=self.device))
        for _ in range(self.conduction_delay_msi2out):
            self.buffer_msi2out.append(torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device))

        # reset NMDA gating
        self.nmda_m = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.v_nmda = torch.full((self.batch_size, self.n), self.v_nmda_rest, dtype=torch.float32, device=self.device)
        self.nmda_m_inh = torch.zeros((self.batch_size, self.n_inh), dtype=torch.float32, device=self.device)
        self.v_nmda_inh = torch.full((self.batch_size, self.n_inh), self.v_nmda_rest, dtype=torch.float32,
                                     device=self.device)

        self.v_dend_A = torch.full((self.batch_size, self.n), self.cM, device=self.device)
        self.v_dend_V = torch.full((self.batch_size, self.n), self.cM, device=self.device)
        self.v_dend_inhA = torch.full((self.batch_size, self.n_inh), self.cMi, device=self.device)
        self.v_dend_inhV = torch.full((self.batch_size, self.n_inh), self.cMi, device=self.device)

        # reset STP
        self.R_a = torch.ones((self.batch_size, self.n), device=self.device)
        self.u_a = torch.full((self.batch_size, self.n), 0.2, device=self.device)
        self.R_v = torch.ones((self.batch_size, self.n), device=self.device)
        self.u_v = torch.full((self.batch_size, self.n), 0.2, device=self.device)

        self.R_a_inh = torch.ones((self.batch_size, self.n_inh), device=self.device)
        self.u_a_inh = torch.full((self.batch_size, self.n_inh), 0.2, device=self.device)
        self.R_v_inh = torch.ones((self.batch_size, self.n_inh), device=self.device)
        self.u_v_inh = torch.full((self.batch_size, self.n_inh), 0.2, device=self.device)

        self.post_i_trace = torch.zeros((self.batch_size, self.n),
                                        dtype=torch.float32,
                                        device=self.device)

        # NEW – keep firing-rate tracker in sync
        self.post_rate_avg = torch.zeros((self.batch_size, self.n),
                                         dtype=torch.float32,
                                         device=self.device)

    def update_all_layers_batch(self, xA_batch, xV_batch, valid_mask=None, record_voltages=False):
        """
        Runs sub-steps of simulation for one time-step's input xA, xV.
        Fix: proper shape handling for A->MSI_inh and V->MSI_inh short-term depression + matrix multiplies
        """
        batch_size = xA_batch.shape[0]

        if valid_mask is not None:
            mask = valid_mask.view(batch_size, 1)
            xA_batch = xA_batch * mask
            xV_batch = xV_batch * mask

        sA = torch.zeros((batch_size, self.n), dtype=torch.float32, device=self.device)
        sV = torch.zeros((batch_size, self.n), dtype=torch.float32, device=self.device)
        sM = torch.zeros((batch_size, self.n), dtype=torch.float32, device=self.device)
        sMi = torch.zeros((batch_size, self.n_inh), dtype=torch.float32, device=self.device)
        sO = torch.zeros((batch_size, self.n), dtype=torch.float32, device=self.device)

        decay_factor = 1.0 - self.dt / self.tau_syn
        spike_threshold = 30.0

        xA_expanded = xA_batch.unsqueeze(1)  # (B,1,n)
        xV_expanded = xV_batch.unsqueeze(1)
        W_inA_expanded = self.W_inA.unsqueeze(0).expand(batch_size, self.n, self.n)  # (B,n,n)
        W_inV_expanded = self.W_inV.unsqueeze(0).expand(batch_size, self.n, self.n)

        I_A_input = self.input_scaling * (
                torch.bmm(xA_expanded, W_inA_expanded).squeeze(1) + self.b_uniA
        )
        I_V_input = self.input_scaling * (
                torch.bmm(xV_expanded, W_inV_expanded).squeeze(1) + self.b_uniV
        )

        if record_voltages:
            sM_substeps = []

        for _ in range(self.n_substeps):
            # Decay syn currents
            self.I_A.mul_(decay_factor)
            self.I_V.mul_(decay_factor)
            self.I_M.mul_(decay_factor)
            self.I_M_inh.mul_(decay_factor)
            self.I_O.mul_(decay_factor)

            # Add unimodal input
            self.I_A.add_(I_A_input)
            self.I_V.add_(I_V_input)

            # pop from buffers
            delayed_spikes_a2msi = self.buffer_a2msi.popleft()  # (B,n)
            delayed_spikes_v2msi = self.buffer_v2msi.popleft()  # (B,n)
            delayed_spikes_inA_inh = self.buffer_inA_inh.popleft()  # (B,n)
            delayed_spikes_inV_inh = self.buffer_inV_inh.popleft()  # (B,n)
            delayed_spikes_a2msi_inh = self.buffer_a2msi_inh.popleft()  # (B,n)
            delayed_spikes_v2msi_inh = self.buffer_v2msi_inh.popleft()  # (B,n)
            delayed_spikes_msi_inh2exc = self.buffer_msi_inh2exc.popleft()  # (B,n_inh)
            delayed_spikes_msi2out = self.buffer_msi2out.popleft()  # (B,n)

            # ------------- A->MSI excit (AMPA+NMDA) -------------
            W_a2msi_AMPA_expanded = self.W_a2msi_AMPA.unsqueeze(0).expand(batch_size, self.n, self.n)
            W_v2msi_AMPA_expanded = self.W_v2msi_AMPA.unsqueeze(0).expand(batch_size, self.n, self.n)
            W_a2msi_NMDA_expanded = self.W_a2msi_NMDA.unsqueeze(0).expand(batch_size, self.n, self.n)
            W_v2msi_NMDA_expanded = self.W_v2msi_NMDA.unsqueeze(0).expand(batch_size, self.n, self.n)

            # Tsodyks for A->MSI AMPA
            self.R_a += (1.0 - self.R_a) * (self.dt / self.tau_rec)
            use_A = self.u_a * self.R_a
            self.R_a -= use_A * delayed_spikes_a2msi  # shape (B,n) elementwise

            # compute actual current
            # multiply W*(use_A * spikes) like original code
            I_M_a_AMPA = torch.bmm(
                W_a2msi_AMPA_expanded,
                (use_A * delayed_spikes_a2msi).unsqueeze(2)  # (B,n,1)
            ).squeeze(2)

            # Tsodyks for V->MSI AMPA
            self.R_v += (1.0 - self.R_v) * (self.dt / self.tau_rec)
            use_V = self.u_v * self.R_v
            self.R_v -= use_V * delayed_spikes_v2msi

            I_M_v_AMPA = torch.bmm(
                W_v2msi_AMPA_expanded,
                (use_V * delayed_spikes_v2msi).unsqueeze(2)
            ).squeeze(2)

            self.I_M.add_(self.input_scaling * (I_M_a_AMPA + I_M_v_AMPA + self.b_msi))

            # NMDA portion
            nmda_a = torch.bmm(W_a2msi_NMDA_expanded, delayed_spikes_a2msi.unsqueeze(2)).squeeze(2)
            nmda_v = torch.bmm(W_v2msi_NMDA_expanded, delayed_spikes_v2msi.unsqueeze(2)).squeeze(2)

            self.nmda_m.mul_(1.0 - self.dt / self.tau_nmda)
            self.nmda_m.add_(self.nmda_alpha * (nmda_a + nmda_v))

            d_va = self.dend_coupling_alpha * (self.v_msi - self.v_dend_A) / self.tau_m
            d_vv = self.dend_coupling_alpha * (self.v_msi - self.v_dend_V) / self.tau_m
            self.v_dend_A += self.dt * d_va
            self.v_dend_V += self.dt * d_vv

            dv_nmda = ((self.v_msi + self.nmda_vrest_offset) - self.v_nmda) / self.tau_nmdaVolt
            self.v_nmda += self.dt * dv_nmda

            mg_A = 1.0 / (1.0 + torch.exp(-self.mg_k * (self.v_dend_A - self.mg_vhalf)))
            mg_V = 1.0 / (1.0 + torch.exp(-self.mg_k * (self.v_dend_V - self.mg_vhalf)))

            I_nmda = self.gNMDA * self.nmda_m * (mg_A + mg_V) * (self.Erev_nmda - self.v_msi)
            self.I_M.add_(I_nmda)

            # ------------- Dedicated Inhibitory Projections from unimodal to MSI excit -------------
            W_inA_inh_expanded = self.W_inA_inh.unsqueeze(0).expand(batch_size, self.n, self.n)
            W_inV_inh_expanded = self.W_inV_inh.unsqueeze(0).expand(batch_size, self.n, self.n)

            I_inA_inh = torch.bmm(W_inA_inh_expanded, delayed_spikes_inA_inh.unsqueeze(2)).squeeze(2)
            I_inV_inh = torch.bmm(W_inV_inh_expanded, delayed_spikes_inV_inh.unsqueeze(2)).squeeze(2)
            self.I_M.sub_(self.input_scaling * (I_inA_inh + I_inV_inh))

            # ------------- A->MSI_inh excit (AMPA+NMDA) -------------
            # short-term depression
            # First handle resource usage: we sum all input spikes for each batch item,
            # then reduce the resource for each inh neuron by that fraction * use_A_inh.
            self.R_a_inh += (1.0 - self.R_a_inh) * (self.dt / self.tau_rec)
            use_A_inh = self.u_a_inh * self.R_a_inh
            spike_sum_a = delayed_spikes_a2msi_inh.sum(dim=1, keepdim=True)  # (B,1)
            self.R_a_inh -= use_A_inh * spike_sum_a  # broadcast => (B,n_inh)

            # Now compute the actual AMPA current via matrix multiply:
            #   W is shape (n_inh,n), expand => (B,n_inh,n)
            #   input spikes is shape (B,n), we do bmm with (B,n_inh,n)*(B,n,1)->(B,n_inh,1)->(B,n_inh)
            # Then we multiply elementwise by "use_A_inh" to scale each post-neuron's amplitude
            Wa2miA_expanded = self.W_a2msiInh_AMPA.unsqueeze(0).expand(batch_size, self.n_inh, self.n)
            preA_inh_3d = delayed_spikes_a2msi_inh.unsqueeze(2)  # (B,n,1)
            raw_inp_a_AMPA = torch.bmm(Wa2miA_expanded, preA_inh_3d).squeeze(2)  # (B,n_inh)
            I_Mi_a_AMPA = self.input_scaling * (use_A_inh * raw_inp_a_AMPA)

            # NMDA for A->MSI_inh
            Wa2miN_expanded = self.W_a2msiInh_NMDA.unsqueeze(0).expand(batch_size, self.n_inh, self.n)
            raw_inp_a_NMDA = torch.bmm(Wa2miN_expanded, preA_inh_3d).squeeze(2)

            self.nmda_m_inh.mul_(1.0 - self.dt / self.tau_nmda)
            self.nmda_m_inh.add_(self.nmda_alpha * raw_inp_a_NMDA)

            # V->MSI_inh excit (similar approach)
            self.R_v_inh += (1.0 - self.R_v_inh) * (self.dt / self.tau_rec)
            use_V_inh = self.u_v_inh * self.R_v_inh
            spike_sum_v = delayed_spikes_v2msi_inh.sum(dim=1, keepdim=True)
            self.R_v_inh -= use_V_inh * spike_sum_v

            Wv2miA_expanded = self.W_v2msiInh_AMPA.unsqueeze(0).expand(batch_size, self.n_inh, self.n)
            preV_inh_3d = delayed_spikes_v2msi_inh.unsqueeze(2)
            raw_inp_v_AMPA = torch.bmm(Wv2miA_expanded, preV_inh_3d).squeeze(2)
            I_Mi_v_AMPA = self.input_scaling * (use_V_inh * raw_inp_v_AMPA)

            Wv2miN_expanded = self.W_v2msiInh_NMDA.unsqueeze(0).expand(batch_size, self.n_inh, self.n)
            raw_inp_v_NMDA = torch.bmm(Wv2miN_expanded, preV_inh_3d).squeeze(2)
            self.nmda_m_inh.add_(self.nmda_alpha * raw_inp_v_NMDA)

            # Combine AMPA for MSI_inh
            self.I_M_inh.add_(I_Mi_a_AMPA + I_Mi_v_AMPA + self.b_msi_inh)

            # NMDA gating for MSI_inh
            d_viA = self.dend_coupling_alpha * (self.v_msi_inh - self.v_dend_inhA) / self.tau_m
            d_viV = self.dend_coupling_alpha * (self.v_msi_inh - self.v_dend_inhV) / self.tau_m
            self.v_dend_inhA += self.dt * d_viA
            self.v_dend_inhV += self.dt * d_viV

            dv_nmda_inh = ((self.v_msi_inh + self.nmda_vrest_offset) - self.v_nmda_inh) / self.tau_nmdaVolt
            self.v_nmda_inh += self.dt * dv_nmda_inh

            mg_iA = 1.0 / (1.0 + torch.exp(-self.mg_k * (self.v_dend_inhA - self.mg_vhalf)))
            mg_iV = 1.0 / (1.0 + torch.exp(-self.mg_k * (self.v_dend_inhV - self.mg_vhalf)))

            I_nmda_inh = self.gNMDA * self.nmda_m_inh * (mg_iA + mg_iV) * (self.Erev_nmda - self.v_msi_inh)
            self.I_M_inh.add_(I_nmda_inh)

            # ------------- MSI_inh -> MSI_ex (GABA) -------------
            W_msiInh2Exc_expanded = self.W_msiInh2Exc_GABA.unsqueeze(0).expand(batch_size, self.n, self.n_inh)
            I_M_inh2exc = torch.bmm(W_msiInh2Exc_expanded, delayed_spikes_msi_inh2exc.unsqueeze(2)).squeeze(2)
            self.I_M.sub_(self.input_scaling * I_M_inh2exc)

            # ------------- MSI->Out -------------
            W_msi2out_expanded = self.W_msi2out.unsqueeze(0).expand(batch_size, self.n, self.n)
            I_O_msi = torch.bmm(W_msi2out_expanded, delayed_spikes_msi2out.unsqueeze(2)).squeeze(2)
            self.I_O.add_(self.input_scaling * (I_O_msi + self.b_out))

            # ------------- Lateral surround in MSI excit -------------
            I_GABA = torch.matmul(self._latest_sMSI, self.W_MSI_inh.t())
            self.I_M.sub_(self.g_GABA * I_GABA)

            # ------------- Izhikevich updates -------------
            # Unimodal A
            dVA = (0.04 * self.v_uniA.pow(2) + 5.0 * self.v_uniA + 140.0 - self.u_uniA + self.I_A)
            self.v_uniA += self.dt * dVA
            self.u_uniA += self.dt * (self.aA * (self.bA * self.v_uniA - self.u_uniA))
            new_sA = (self.v_uniA >= spike_threshold).float()
            self.v_uniA[self.v_uniA >= spike_threshold] = self.cA
            self.u_uniA[self.v_uniA == self.cA] += self.dA

            # Unimodal V
            dVV = (0.04 * self.v_uniV.pow(2) + 5.0 * self.v_uniV + 140.0 - self.u_uniV + self.I_V)
            self.v_uniV += self.dt * dVV
            self.u_uniV += self.dt * (self.aV * (self.bV * self.v_uniV - self.u_uniV))
            new_sV = (self.v_uniV >= spike_threshold).float()
            self.v_uniV[self.v_uniV >= spike_threshold] = self.cV
            self.u_uniV[self.v_uniV == self.cV] += self.dV

            # MSI excit
            dVM = (0.04 * self.v_msi.pow(2) + 5.0 * self.v_msi + 140.0 - self.u_msi + self.I_M)
            self.v_msi += self.dt * dVM
            self.u_msi += self.dt * (self.aM * (self.bM * self.v_msi - self.u_msi))
            new_sM = (self.v_msi >= spike_threshold).float()
            self.v_msi[self.v_msi >= spike_threshold] = self.cM
            self.u_msi[self.v_msi == self.cM] += self.dM

            # Exponential decay of the postsynaptic trace
            decay = torch.exp(
                torch.tensor(-self.dt / self.tau_post_i, device=self.device)
            )
            self.post_i_trace.mul_(decay)

            # Add the current spikes (0/1)
            self.post_i_trace.add_(new_sM)

            alpha_rate = self.dt / self.rate_avg_tau
            self.post_rate_avg.mul_(1.0 - alpha_rate).add_(alpha_rate * new_sM)

            # MSI inh
            dVMi = (0.04 * self.v_msi_inh.pow(2) + 5.0 * self.v_msi_inh + 140.0
                    - self.u_msi_inh + self.I_M_inh)
            self.v_msi_inh += self.dt * dVMi
            self.u_msi_inh += self.dt * (self.aMi * (self.bMi * self.v_msi_inh - self.u_msi_inh))
            new_sMi = (self.v_msi_inh >= spike_threshold).float()
            self.v_msi_inh[self.v_msi_inh >= spike_threshold] = self.cMi
            self.u_msi_inh[self.v_msi_inh == self.cMi] += self.dMi

            # --- Vogels-2011 inhibitory STDP --------------------------------
            # shapes: new_sMi  (B, n_inh)   post_i_trace  (B, n)

            if self.allow_inhib_plasticity:
                pre_i = new_sMi  # presyn spikes
                post_j = self.post_i_trace  # running postsyn trace

                # outer-product: (B, n, n_inh)
                delta = self.eta_i * (post_j - self.rho0).unsqueeze(2) * pre_i.unsqueeze(1)

                # average over batch and apply
                self.W_msiInh2Exc_GABA += delta.mean(dim=0)

                # keep weights within [0, w_max]
                w_max = 0.05  # or self.w_inh_max
                self.W_msiInh2Exc_GABA.clamp_(0.0, w_max)

            # Out
            dVO = (0.04 * self.v_out.pow(2) + 5.0 * self.v_out + 140.0 - self.u_out + self.I_O)
            self.v_out += self.dt * dVO
            self.u_out += self.dt * (self.aO * (self.bO * self.v_out - self.u_out))
            new_sO = (self.v_out >= spike_threshold).float()
            self.v_out[self.v_out >= spike_threshold] = self.cO
            self.u_out[self.v_out == self.cO] += self.dO

            # ----- homeostatic iSTDP (uses current burst of spikes) -----
            # ------------------  homeostatic iSTDP  ------------------

            if self.training and self.allow_inhib_plasticity:
                # * post = spikes of excitatory neurons (n_exc)
                self.iSTDP_homeo(
                    self.W_msiInh2Exc_GABA,
                    pre_spk=new_sMi,
                    post_spk=new_sM,
                    lr=1e-4 * self.dt / self.tau_syn
                )

            if valid_mask is not None:
                mask = valid_mask.view(-1, 1)
                new_sA *= mask
                new_sV *= mask
                new_sM *= mask
                new_sMi *= mask
                new_sO *= mask

            self._latest_sA = new_sA
            self._latest_sV = new_sV
            self._latest_sMSI = new_sM
            self._latest_sMSI_inh = new_sMi
            self._latest_sOut = new_sO

            self.buffer_a2msi.append(new_sA.clone())
            self.buffer_v2msi.append(new_sV.clone())
            self.buffer_inA_inh.append(new_sA.clone())
            self.buffer_inV_inh.append(new_sV.clone())
            self.buffer_a2msi_inh.append(new_sA.clone())
            self.buffer_v2msi_inh.append(new_sV.clone())
            self.buffer_msi_inh2exc.append(new_sMi.clone())
            self.buffer_msi2out.append(new_sM.clone())

            sA, sV, sM, sMi, sO = new_sA, new_sV, new_sM, new_sMi, new_sO

            if record_voltages:
                sM_substeps.append(new_sM.clone())

            self.step_counter += 1

        # --- slow homeostatic scaling of inhibitory rows ---
        if self.allow_inhib_plasticity:
            if self.step_counter % self.inhib_scaling_T == 0:  # every 200 ms (given dt=0.1 ms, n_substeps=100)
                scale_err = (self.post_rate_avg - self.rho0) / self.rho0  # (B,n)
                eta_scale = 0.02
                factor = 1 + eta_scale * torch.clamp(scale_err, min=-0.5, max=+0.5)
                factor = factor.mean(dim=0)
                self.W_msiInh2Exc_GABA.mul_(factor.unsqueeze(1))
                self.W_inA_inh.mul_(factor.unsqueeze(1))
                self.W_inV_inh.mul_(factor.unsqueeze(1))

        if record_voltages:
            sM_substeps_tensor = torch.stack(sM_substeps, dim=0)
            volt_dict = {'sM_substeps': sM_substeps_tensor}
            return sA, sV, sM, sO, volt_dict
        else:
            # --- fast global E/I ratio probe (optional) ---

            if not self.training:  # skip during inference if desired
                I_exc = self.I_M.clamp(min=0).mean()  # total depolarising drive
                I_inh = (-self.I_M.clamp(max=0)).mean() + 1e-5
                self.EI_history.append((I_exc / I_inh).item())

            return sA, sV, sM, sO

    def stdp_update_batch(self, W, post_acts_batch, pre_acts_batch,
                          post_trace_batch, pre_trace_batch,
                          lr, tau_pre=0.9, tau_post=0.9,
                          A_plus=1.0, A_minus=1.0):
        """
        STDP update for a batch of pre/post spikes.
        """
        batch_size = post_acts_batch.shape[0]

        pre_trace_batch.mul_(tau_pre).add_(pre_acts_batch)
        post_trace_batch.mul_(tau_post).add_(post_acts_batch)

        dW_batch = torch.zeros_like(W)
        for i in range(batch_size):
            post_acts = post_acts_batch[i]
            pre_acts = pre_acts_batch[i]
            post_tr = post_trace_batch[i]
            pre_tr = pre_trace_batch[i]

            dW_plus = A_plus * torch.outer(post_acts, pre_tr)
            dW_minus = A_minus * torch.outer(post_tr, pre_acts)
            dW_batch.add_(dW_plus - dW_minus)

        dW_batch.div_(batch_size)
        update_val = lr * dW_batch
        W.add_(update_val)
        return pre_trace_batch, post_trace_batch, update_val

    def normalize_rows_gpu(self, W):
        norms = torch.norm(W, dim=1, keepdim=True)
        norms[norms == 0] = 1.0
        W.div_(norms)

    ########################################################
    #           UNSUPERVISED & SUPERVISED TRAINING
    ########################################################

    def train_unsupervised_batch(self, n_sequences, batch_size=32):
        """
        Generates random AV sequences & applies STDP to early layers.
        Only uses In->Uni and Uni->MSI excit for STDP.
        """
        while n_sequences > 0:
            actual_batch_size = min(batch_size, n_sequences)
            loc_seqs, mod_seqs, offset_applied, seq_lengths = generate_event_loc_seq_batch(
                batch_size=actual_batch_size,
                space_size=self.space_size,
                offset_probability=0.1
            )
            max_len = max(seq_lengths)

            xA_batch, xV_batch, valid_mask = generate_av_batch_tensor(
                loc_seqs, mod_seqs,
                offset_applied,
                n=self.n,
                space_size=self.space_size,
                sigma_in=self.sigma_in,
                noise_std=self.noise_std,
                loc_jitter_std=self.loc_jitter_std,
                device=self.device,
                max_len=max_len
            )

            self.reset_state(actual_batch_size)

            for t in range(max_len):
                xA_t = xA_batch[:, t]
                xV_t = xV_batch[:, t]
                valid_t = valid_mask[:, t]

                sA, sV, sMSI, _ = self.update_all_layers_batch(xA_t, xV_t, valid_t)

                # STDP: In->Uni(V)
                self.pre_trace_inV, self.post_trace_inV, _ = self.stdp_update_batch(
                    self.W_inV, sV, xV_t,
                    self.post_trace_inV, self.pre_trace_inV, self.lr_uni
                )
                self.normalize_rows_gpu(self.W_inV)

                # STDP: In->Uni(A)
                self.pre_trace_inA, self.post_trace_inA, _ = self.stdp_update_batch(
                    self.W_inA, sA, xA_t,
                    self.post_trace_inA, self.pre_trace_inA, self.lr_uni
                )
                self.normalize_rows_gpu(self.W_inA)

                # STDP: Uni(V)->MSI (AMPA+NMDA)
                W_v2msi_total = self.W_v2msi_AMPA + self.W_v2msi_NMDA
                self.pre_trace_v2msi, self.post_trace_v2msi, _ = self.stdp_update_batch(
                    W_v2msi_total, sMSI, sV,
                    self.post_trace_v2msi, self.pre_trace_v2msi, self.lr_msi
                )
                self.normalize_rows_gpu(W_v2msi_total)
                with torch.no_grad():
                    self.W_v2msi_AMPA.copy_(0.2 * W_v2msi_total)
                    self.W_v2msi_NMDA.copy_(0.8 * W_v2msi_total)

                # STDP: Uni(A)->MSI (AMPA+NMDA)
                W_a2msi_total = self.W_a2msi_AMPA + self.W_a2msi_NMDA
                self.pre_trace_a2msi, self.post_trace_a2msi, _ = self.stdp_update_batch(
                    W_a2msi_total, sMSI, sA,
                    self.post_trace_a2msi, self.pre_trace_a2msi, self.lr_msi
                )
                self.normalize_rows_gpu(W_a2msi_total)
                with torch.no_grad():
                    self.W_a2msi_AMPA.copy_(0.2 * W_a2msi_total)
                    self.W_a2msi_NMDA.copy_(0.8 * W_a2msi_total)

            # ⬇️ keep surround inhibition close to its prior
            lam = 1e-3
            with torch.no_grad():
                self.W_MSI_inh.add_(-lam * (self.W_MSI_inh - self.W_MSI_inh_init))

            n_sequences -= actual_batch_size

    def train_readout_batch(self, n_sequences, conditions=['both'],
                            condition_probs=None, batch_size=32,
                            epoch_idx: int = 0):
        """
        Generates random AV sequences, picks a condition
        (audio_only, visual_only, or both). Applies teacher signal
        at the final step of each event for supervised training
        on MSI excit -> readout.
        """
        if condition_probs is None:
            condition_probs = [1.0 / len(conditions)] * len(conditions)
        else:
            total = sum(condition_probs)
            condition_probs = [p / total for p in condition_probs]

        while n_sequences > 0:
            actual_batch_size = min(batch_size, n_sequences)
            loc_seqs, mod_seqs, offset_applied, seq_lengths = generate_event_loc_seq_batch(
                batch_size=actual_batch_size,
                space_size=self.space_size,
                offset_probability=0.1
            )
            batch_conds = np.random.choice(conditions, size=actual_batch_size, p=condition_probs)

            max_len = max(seq_lengths)
            xA_batch, xV_batch, valid_mask = generate_av_batch_tensor(
                loc_seqs, mod_seqs,
                offset_applied,
                n=self.n,
                space_size=self.space_size,
                sigma_in=self.sigma_in,
                noise_std=self.noise_std,
                loc_jitter_std=self.loc_jitter_std,
                device=self.device,
                max_len=max_len
            )

            # Zero out certain modalities depending on condition
            for i, condition in enumerate(batch_conds):
                if condition == 'audio_only':
                    xV_batch[i].zero_()
                elif condition == 'visual_only':
                    xA_batch[i].zero_()

            self.reset_state(actual_batch_size)

            sMSI_list = []
            err_list = []

            for t in range(max_len):
                xA_t = xA_batch[:, t]
                xV_t = xV_batch[:, t]
                valid_t = valid_mask[:, t]

                self.update_all_layers_batch(xA_t, xV_t, valid_t)
                sMSI = self._latest_sMSI.clone()

                # Identify final step of an event
                has_teacher = torch.zeros(actual_batch_size, dtype=torch.bool, device=self.device)
                for i_seq in range(actual_batch_size):
                    if valid_t[i_seq] and t < len(loc_seqs[i_seq]):
                        if loc_seqs[i_seq][t] != 999:
                            if (t == len(loc_seqs[i_seq]) - 1) or (loc_seqs[i_seq][t + 1] == 999):
                                has_teacher[i_seq] = True

                if not has_teacher.any():
                    continue

                sMSI_t = sMSI[has_teacher]

                teacher_vecs = []
                idx_map = has_teacher.nonzero(as_tuple=True)[0]
                for i_idx in idx_map:
                    loc_val = loc_seqs[i_idx.item()][t]
                    sigma_now = (
                        self.sigma_teacher_init
                        if epoch_idx < self.curriculum_epochs
                        else self.sigma_teacher_final
                    )
                    teacher_idx = location_to_index(loc_val, self.n, self.space_size)
                    tv = make_gaussian_vector_batch_gpu(
                        torch.tensor([teacher_idx], device=self.device),
                        self.n, sigma_now, self.device
                    )
                    teacher_vecs.append(tv[0])

                teacher_stack = torch.stack(teacher_vecs)

                raw_out = torch.matmul(sMSI_t, self.W_msi2out.t())
                out_acts = torch.clamp(raw_out, min=0)

                err = teacher_stack - out_acts
                sMSI_list.append(sMSI_t)
                err_list.append(err)

            if len(sMSI_list) == 0:
                n_sequences -= actual_batch_size
                continue

            sMSI_all = torch.cat(sMSI_list, dim=0)
            err_all = torch.cat(err_list, dim=0)

            dW = self.lr_out * torch.matmul(err_all.t(), sMSI_all)
            torch.clamp_(dW, -0.25, 0.25)
            self.W_msi2out.add_(dW)

            n_sequences -= actual_batch_size

    def evaluate_batch(self, n_sequences, condition='both', batch_size=32, stimulus_intensity=1.0):
        """
        Generates random AV sequences, runs them, and computes localization
        error at the final step. Returns average absolute error in degrees.
        """
        errors = []
        while n_sequences > 0:
            actual_batch_size = min(batch_size, n_sequences)
            loc_seqs, mod_seqs, offset_applied, seq_lengths = generate_event_loc_seq_batch(
                batch_size=actual_batch_size,
                space_size=self.space_size,
                offset_probability=0.1
            )
            max_len = max(seq_lengths)

            xA_batch, xV_batch, valid_mask = generate_av_batch_tensor(
                loc_seqs, mod_seqs,
                offset_applied,
                n=self.n,
                space_size=self.space_size,
                sigma_in=self.sigma_in,
                noise_std=self.noise_std,
                loc_jitter_std=self.loc_jitter_std,
                stimulus_intensity=stimulus_intensity,
                device=self.device,
                max_len=max_len
            )

            if condition == 'audio_only':
                xV_batch.zero_()
            elif condition == 'visual_only':
                xA_batch.zero_()

            self.reset_state(actual_batch_size)
            batch_errors = [[] for _ in range(actual_batch_size)]

            for t in range(max_len):
                xA_t = xA_batch[:, t]
                xV_t = xV_batch[:, t]
                valid_t = valid_mask[:, t]

                self.update_all_layers_batch(xA_t, xV_t, valid_t)

                for i_seq in range(actual_batch_size):
                    if valid_t[i_seq] and t < len(loc_seqs[i_seq]):
                        if loc_seqs[i_seq][t] != 999:
                            # final step of event if next step is 999 or out of range
                            if (t == len(loc_seqs[i_seq]) - 1) or (loc_seqs[i_seq][t + 1] == 999):
                                sMSI_i = self._latest_sMSI[i_seq]
                                raw_out = torch.matmul(self.W_msi2out, sMSI_i) + self.b_out
                                out_acts = torch.clamp(raw_out, min=0)
                                pred_idx = torch.argmax(out_acts).item()
                                pred_deg = index_to_location(pred_idx, self.n, self.space_size)
                                loc_deg = loc_seqs[i_seq][t]
                                error = abs(pred_deg - loc_deg)
                                batch_errors[i_seq].append(error)

            for seq_errs in batch_errors:
                if seq_errs:
                    errors.append(np.mean(seq_errs))

            n_sequences -= actual_batch_size

        if not errors:
            return 0.0
        return np.mean(errors)

    # ─────────────────────────────────────────────────────────────
    #    QUICK TOGGLE: keep only the Mexican-hat surround active
    # ─────────────────────────────────────────────────────────────
    def isolate_surround(self, enable: bool = True):
        """
        If *enable* is True, this zeroes out:
           • feed-forward A→MSI_ex & V→MSI_ex inhibition
           • MSI_inh → MSI_ex gate
        and freezes inhibitory plasticity, leaving the Mexican-hat
        surround (W_MSI_inh + g_GABA) untouched.
        Call again with False to restore learning.
        """
        z = 0.0 if enable else 1.0
        self.W_inA_inh.mul_(z)
        self.W_inV_inh.mul_(z)
        self.W_msiInh2Exc_GABA.mul_(z)
        self.allow_inhib_plasticity = not enable


########################################################
#             INVERSE EFFECTIVENESS & ANALYSIS
########################################################

def eval_random_seqs_event_gpu(net, n_seq=30, condition='both', stimulus_intensity=1.0, batch_size=32):
    return net.evaluate_batch(
        n_sequences=n_seq,
        condition=condition,
        batch_size=batch_size,
        stimulus_intensity=stimulus_intensity
    )


def calculate_spike_rates_gpu(net, n_seq=30, condition='both', stimulus_intensity=1.0, batch_size=32):
    """
    Runs random sequences, measuring avg spike rate in each layer (A, V, MSI excit, Out).
    (If you want MSI_inh rates, you could similarly track net._latest_sMSI_inh.)
    """
    total_a_spikes = 0.0
    total_v_spikes = 0.0
    total_msi_spikes = 0.0
    total_out_spikes = 0.0
    total_neuron_steps = 0

    while n_seq > 0:
        actual_batch_size = min(batch_size, n_seq)
        loc_seqs, mod_seqs, offset_applied, seq_lengths = generate_event_loc_seq_batch(
            batch_size=actual_batch_size,
            space_size=net.space_size,
            offset_probability=0.1
        )
        max_len = max(seq_lengths)

        xA_batch, xV_batch, valid_mask = generate_av_batch_tensor(
            loc_seqs, mod_seqs,
            offset_applied,
            n=net.n,
            space_size=net.space_size,
            sigma_in=net.sigma_in,
            noise_std=net.noise_std,
            loc_jitter_std=net.loc_jitter_std,
            stimulus_intensity=stimulus_intensity,
            device=net.device,
            max_len=max_len
        )

        if condition == 'audio_only':
            xV_batch.zero_()
        elif condition == 'visual_only':
            xA_batch.zero_()

        net.reset_state(actual_batch_size)

        for t in range(max_len):
            xA_t = xA_batch[:, t]
            xV_t = xV_batch[:, t]
            valid_t = valid_mask[:, t]

            net.update_all_layers_batch(xA_t, xV_t, valid_t)

            total_a_spikes += net._latest_sA.sum().item()
            total_v_spikes += net._latest_sV.sum().item()
            total_msi_spikes += net._latest_sMSI.sum().item()
            total_out_spikes += net._latest_sOut.sum().item()

            total_neuron_steps += (actual_batch_size * net.n)

        n_seq -= actual_batch_size

    a_rate = total_a_spikes / total_neuron_steps
    v_rate = total_v_spikes / total_neuron_steps
    msi_rate = total_msi_spikes / total_neuron_steps
    out_rate = total_out_spikes / total_neuron_steps

    return {
        'audio': a_rate,
        'visual': v_rate,
        'msi': msi_rate,
        'output': out_rate
    }


########################################################
#       FIXED DURATION & MSI ENHANCEMENT TEST
########################################################

def generate_fixed_duration_event_sequences(space_size=180, T=30, D=10, n_per_loc=20):
    """
    Generates sequences for each location in [0..space_size-1], each with
    an event of duration D=10 steps. We do n_per_loc sequences per location.
    """
    loc_seqs = []
    mod_seqs = []
    offset_applied = []
    seq_lengths = []

    for loc in range(space_size):
        for _ in range(n_per_loc):
            start_t = np.random.randint(0, T - D + 1)
            loc_seq = [999] * T
            mod_seq = ['X'] * T

            for t in range(start_t, start_t + D):
                loc_seq[t] = loc
                mod_seq[t] = 'B'

            loc_seqs.append(loc_seq)
            mod_seqs.append(mod_seq)
            offset_applied.append(False)
            seq_lengths.append(T)

    return loc_seqs, mod_seqs, offset_applied, seq_lengths


def check_msi_enhancement(net, space_size=180, T=30, D=10, n_per_loc=20):
    """
    1) Plot 3 histograms (Audio, Visual, Bimodal) of errors in [-20,20],
       with fitted Gaussians
    2) Then a 4th subplot: bar chart of sensitivities [S_A, S_V, S_AV, S_AV_ideal].
    """
    loc_seqs, mod_seqs, offset_applied, seq_lengths = generate_fixed_duration_event_sequences(
        space_size=space_size, T=T, D=D, n_per_loc=n_per_loc
    )
    max_len = max(seq_lengths)
    total_sequences = len(loc_seqs)

    xA_batch, xV_batch, valid_mask = generate_av_batch_tensor(
        loc_seqs, mod_seqs, offset_applied,
        n=net.n,
        space_size=net.space_size,
        sigma_in=net.sigma_in,
        noise_std=net.noise_std,
        loc_jitter_std=net.loc_jitter_std,
        stimulus_intensity=1.0,
        device=net.device,
        max_len=max_len
    )

    def run_and_get_signed_errors(xA, xV, condition='both'):
        xA_cond = xA.clone()
        xV_cond = xV.clone()
        if condition == 'audio_only':
            xV_cond.zero_()
        elif condition == 'visual_only':
            xA_cond.zero_()

        batch_eval_size = 256
        all_errors = []

        for start_idx in range(0, total_sequences, batch_eval_size):
            end_idx = start_idx + batch_eval_size
            xA_chunk = xA_cond[start_idx:end_idx]
            xV_chunk = xV_cond[start_idx:end_idx]
            valid_chunk = valid_mask[start_idx:end_idx]

            n_in_chunk = xA_chunk.shape[0]
            net.reset_state(n_in_chunk)

            msi_spike_history = torch.zeros((max_len, n_in_chunk, net.n), device=net.device)

            for t in range(max_len):
                xA_t = xA_chunk[:, t]
                xV_t = xV_chunk[:, t]
                valid_t = valid_chunk[:, t]

                net.update_all_layers_batch(xA_t, xV_t, valid_t)
                msi_spike_history[t] = net._latest_sMSI

            for seq_i in range(n_in_chunk):
                global_idx = start_idx + seq_i
                loc_seq_i = loc_seqs[global_idx]
                times = [tt for tt, val in enumerate(loc_seq_i) if val != 999]
                if len(times) == 0:
                    continue
                final_t = times[-1]

                sMSI_i = msi_spike_history[final_t, seq_i, :]
                raw_out = torch.matmul(net.W_msi2out, sMSI_i) + net.b_out
                out_acts = torch.clamp(raw_out, min=0)
                pred_idx = torch.argmax(out_acts).item()
                pred_deg = index_to_location(pred_idx, net.n, net.space_size)
                loc_deg = loc_seq_i[final_t]

                diff = pred_deg - loc_deg
                if diff > 90:
                    diff -= 180
                elif diff < -90:
                    diff += 180
                all_errors.append(diff)

        return all_errors

    errors_audio = run_and_get_signed_errors(xA_batch, xV_batch, 'audio_only')
    errors_visual = run_and_get_signed_errors(xA_batch, xV_batch, 'visual_only')
    errors_bimodal = run_and_get_signed_errors(xA_batch, xV_batch, 'both')

    x_min, x_max = -20, 20
    bin_edges = np.linspace(x_min, x_max, 81)

    plt.figure(figsize=(24, 5))

    # Audio hist
    plt.subplot(1, 4, 1)
    errorsA_filtered = [e for e in errors_audio if x_min <= e <= x_max]
    plt.hist(errorsA_filtered, bins=bin_edges, density=True, edgecolor='black')
    muA, stdA = stats.norm.fit(errorsA_filtered)
    x_vals = np.linspace(x_min, x_max, 200)
    pdfA = stats.norm.pdf(x_vals, muA, stdA)
    plt.plot(x_vals, pdfA, 'r-', linewidth=2)
    plt.title(f"Audio-only (std={stdA:.2f})", fontsize=12)
    plt.xlabel("Signed Error (deg)")
    plt.xlim(x_min, x_max)

    # Visual hist
    plt.subplot(1, 4, 2)
    errorsV_filtered = [e for e in errors_visual if x_min <= e <= x_max]
    plt.hist(errorsV_filtered, bins=bin_edges, density=True, edgecolor='black', color='orange')
    muV, stdV = stats.norm.fit(errorsV_filtered)
    pdfV = stats.norm.pdf(x_vals, muV, stdV)
    plt.plot(x_vals, pdfV, 'r-', linewidth=2)
    plt.title(f"Visual-only (std={stdV:.2f})", fontsize=12)
    plt.xlabel("Signed Error (deg)")
    plt.xlim(x_min, x_max)

    # Bimodal hist
    plt.subplot(1, 4, 3)
    errorsB_filtered = [e for e in errors_bimodal if x_min <= e <= x_max]
    plt.hist(errorsB_filtered, bins=bin_edges, density=True, edgecolor='black', color='green')
    muB, stdB = stats.norm.fit(errorsB_filtered)
    pdfB = stats.norm.pdf(x_vals, muB, stdB)
    plt.plot(x_vals, pdfB, 'r-', linewidth=2)
    plt.title(f"Bimodal (std={stdB:.2f})", fontsize=12)
    plt.xlabel("Signed Error (deg)")
    plt.xlim(x_min, x_max)

    # Sensitivity bar chart
    plt.subplot(1, 4, 4)
    sA = 0.0 if stdA == 0 else 1.0 / stdA
    sV = 0.0 if stdV == 0 else 1.0 / stdV
    sB = 0.0 if stdB == 0 else 1.0 / stdB
    sAV_ideal = np.sqrt(sA ** 2 + sV ** 2)

    bar_labels = ["Audio", "Visual", "Bimodal", "Bim-ideal"]
    bar_vals = [sA, sV, sB, sAV_ideal]

    plt.bar(range(len(bar_vals)), bar_vals, color=["blue", "orange", "green", "red"], alpha=0.7)
    plt.xticks(range(len(bar_vals)), bar_labels)
    plt.ylabel("Sensitivity (1/std)")
    plt.title(f"Sensitivities\nBim: {sB:.2f}, Ideal: {sAV_ideal:.2f}", fontsize=11)

    plt.tight_layout()
    plt.show()


########################################################
#        FLASH-SOUND: TEMPORAL BINDING (NO BURST COUNT)
########################################################

def generate_two_event_offset_seq(loc, T=60, D=5, offset=0, space_size=180):
    """
    A positive offset → visual lags audio by <offset> macro steps (10 ms each);
    a negative offset → visual leads; 0 → simultaneous.
    """
    loc_seq = [999] * T
    mod_seq = ['X'] * T
    aud_on = 0 if offset >= 0 else abs(offset)
    vis_on = 0 if offset <= 0 else offset
    for t in range(aud_on, aud_on + D):
        loc_seq[t] = loc;
        mod_seq[t] = 'A'
    for t in range(vis_on, vis_on + D):
        loc_seq[t] = loc;
        mod_seq[t] = 'V' if mod_seq[t] == 'X' else 'B'
    return loc_seq, mod_seq


# -------------------------------------------------------------------
# NEW -- temporal-integration sweep (creates one big batch and returns
#        population spike raster + integrated counts)
# -------------------------------------------------------------------
def run_temporal_integration(net, offsets, loc=90, T=60, D=5):
    """
    offsets  : list of integers (macro-step units, 1 step = 10 ms)
    returns  : dict with 'spike_raster'  (T×len(offsets) ndarray) and
               'int_spikes'   (len(offsets) vector, 0-100 ms window)
    """
    loc_seqs, mod_seqs, off, lens = generate_flash_sound_batch(
        offsets, loc=loc, T=T, D=D, space_size=net.space_size
    )
    max_len = max(lens)
    xA, xV, mask = generate_av_batch_tensor(
        loc_seqs, mod_seqs, off,
        n=net.n, space_size=net.space_size,
        sigma_in=net.sigma_in, noise_std=0.0,  # no extra noise for clarity
        device=net.device, max_len=max_len
    )

    net.reset_state(len(offsets))
    spike_raster = torch.zeros((max_len, len(offsets)), device=net.device)

    for t in range(max_len):
        net.update_all_layers_batch(xA[:, t], xV[:, t], mask[:, t])
        spike_raster[t] = net._latest_sMSI.sum(dim=1)  # pop spike count

    # integrate spikes in a 0-100 ms window starting at *whichever*
    # modality arrives last (window length = D + 5 macro steps)
    win = slice(0, D + 5)
    int_spikes = spike_raster[win].sum(dim=0).cpu().numpy()

    return {'spike_raster': spike_raster.cpu().numpy(),
            'int_spikes': int_spikes,
            'offsets_ms': [o * 10 for o in offsets]}


def plot_temporal_binding(results):
    import matplotlib.pyplot as plt
    rast = results['spike_raster'];
    ints = results['int_spikes']
    offs = results['offsets_ms']
    # heat-map
    plt.figure(figsize=(8, 4))
    plt.imshow(rast, origin='lower', aspect='auto',
               extent=[offs[0], offs[-1], 0, 10 * rast.shape[0]])
    plt.colorbar(label='MSI pop-spikes / 10 ms')
    plt.xlabel('Audio – Visual onset (ms)')
    plt.ylabel('Time (ms)')
    plt.title('MSI activity vs AV asynchrony')
    # binding curve
    plt.figure(figsize=(4, 3))
    plt.plot(offs, ints, 'o-')
    plt.axvline(0, ls='--', c='k')
    plt.xlabel('Audio – Visual onset (ms)')
    plt.ylabel('Integrated spikes (0–100 ms)')
    plt.title('Temporal binding window')
    plt.tight_layout()
    plt.show()


def generate_flash_sound_batch(
        offsets,
        loc=90,
        T=50,
        D=5,
        space_size=180
):
    loc_seqs = []
    mod_seqs = []
    offset_applied = []
    seq_lengths = []

    for off in offsets:
        seq_loc, seq_mod = generate_two_event_offset_seq(
            loc=loc, T=T, D=D, offset=off, space_size=space_size
        )
        loc_seqs.append(seq_loc)
        mod_seqs.append(seq_mod)
        offset_applied.append(False)
        seq_lengths.append(T)

    return loc_seqs, mod_seqs, offset_applied, seq_lengths


def run_flash_sound_experiment(net, offsets, T=50, D=5, loc=90, batch_size=None):
    if batch_size is None:
        batch_size = len(offsets)

    loc_seqs, mod_seqs, offset_applied, seq_lengths = generate_flash_sound_batch(
        offsets=offsets, loc=loc, T=T, D=D, space_size=net.space_size
    )
    max_len = max(seq_lengths)

    xA_batch, xV_batch, valid_mask = generate_av_batch_tensor(
        loc_seqs, mod_seqs, offset_applied,
        n=net.n,
        space_size=net.space_size,
        sigma_in=net.sigma_in,
        noise_std=net.noise_std,
        loc_jitter_std=net.loc_jitter_std,
        device=net.device,
        max_len=max_len
    )

    net.reset_state(batch_size)

    spike_history = torch.zeros((max_len, batch_size, net.n), device=net.device)

    for t in range(max_len):
        xA_t = xA_batch[:, t]
        xV_t = xV_batch[:, t]
        valid_t = valid_mask[:, t]

        net.update_all_layers_batch(xA_t, xV_t, valid_t)
        spike_history[t] = net._latest_sMSI

    return {
        'spike_history': spike_history,
        'offsets': offsets,
        'loc_seqs': loc_seqs,
        'mod_seqs': mod_seqs
    }


def visualize_flash_sound_msi(net, offsets=None, T=50, D=5, loc=90, threshold=5):
    if offsets is None:
        offsets = list(range(21))  # 0..20

    results = run_flash_sound_experiment(net, offsets, T=T, D=D, loc=loc)
    spike_history = results['spike_history']
    Tdim = spike_history.shape[0]
    B = len(offsets)

    total_spikes = spike_history.sum(dim=2).cpu().numpy()  # shape (T, B)

    n_cols = 5
    n_rows = int(np.ceil(B / n_cols))

    plt.figure(figsize=(4 * n_cols, 3 * n_rows))

    for i, off in enumerate(offsets):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.plot(total_spikes[:, i], color='blue', label=f'Offset={off}')
        plt.title(f"Offset={off}")
        plt.xlabel("Time step")
        plt.ylabel("Summed MSI spikes")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------
#               QUICK EI‑BALANCE / IRREGULARITY PROBE
# --------------------------------------------------------------
# --------------------------------------------------------------
#  QUICK EI‑BALANCE / IRREGULARITY PROBE  (updated version)
# --------------------------------------------------------------
def single_cell_cv(
        net,
        macro_steps: int = 300,
        drive_std: float = 0.02,
        warmup_steps: int = 20,
        min_spikes: int = 5,
        max_retries: int = 3,
):
    """Return ISI‑CV of *some* MSI excitatory neuron, retrying if silent.

    * If the first chosen neuron fires < *min_spikes* after warm‑up, we pick
      another neuron with the highest spike count among the MSI population and
      rerun.  After *max_retries* attempts, returns `np.nan.
    * Drive is *Gaussian* noise delivered to both sensory layers.
    """
    B = 1
    for attempt in range(max_retries):
        neuron_idx = np.random.randint(net.n)
        net.reset_state(batch_size=B)
        spike_times = []
        sub_counter = 0
        for macro in range(macro_steps + warmup_steps):
            jitter = drive_std * torch.randn((B, net.n), device=net.device)
            _, _, _, _, vdict = net.update_all_layers_batch(
                jitter, jitter, record_voltages=True
            )
            fired = vdict["sM_substeps"][:, 0, neuron_idx].nonzero(as_tuple=True)[0]
            if fired.numel():
                spike_times.extend((fired + sub_counter).tolist())
            sub_counter += net.n_substeps
        spike_times = np.array(spike_times)
        spike_times = spike_times[spike_times > warmup_steps * net.n_substeps]
        if spike_times.size >= min_spikes:
            isi = np.diff(spike_times)
            return float(np.std(isi) / (np.mean(isi) + 1e-9))
    # still not enough spikes
    return float("nan")


# ---------------------------------------------------------------------------
# 2.  QUICK EI PROBE – unchanged except full sub‑step count
# ---------------------------------------------------------------------------

def quick_ei_probe(
        net,
        n_macro_steps: int = 200,
        bg_lambda_hz: float = 800.0,
        bg_amplitude: float = 0.2,
        store_logs: bool = True,
):
    net.reset_state(batch_size=1)
    dt_macro = net.dt * net.n_substeps  # ms
    p_event = bg_lambda_hz * dt_macro * 1e-3
    ei_log, cv_log, spk_log = [], [], []
    for _ in range(n_macro_steps):
        poisson_mask = (torch.rand(1, net.n, device=net.device) < p_event)
        bg_current = bg_amplitude * poisson_mask.float()
        _, _, _, _, vdict = net.update_all_layers_batch(
            bg_current, bg_current, record_voltages=True
        )
        pop_spikes = vdict["sM_substeps"].sum().item()
        spk_log.append(pop_spikes)
        I_exc = torch.clamp(net.I_M, min=0) + net.I_A + net.I_V
        I_inh = torch.clamp(net.I_M, max=0)
        ei_log.append((I_inh.abs().sum() / (I_exc.sum() + 1e-6)).item())
        if len(spk_log) > 1:
            cv_log.append(np.std(spk_log) / (np.mean(spk_log) + 1e-6))
    mean_ei = float(np.mean(ei_log))
    mean_fr = np.mean(spk_log) / (net.n * dt_macro * 1e-3 * n_macro_steps)
    mean_cv = float(np.mean(cv_log)) if cv_log else float("nan")
    if store_logs:
        net._ei_probe_log = ei_log
        net._pop_cv_log = cv_log
        net._pop_spk_log = spk_log
    return mean_ei, mean_cv, mean_fr


def instrument_network(net, *, exc_threshold: float = 5e-4):
    """
    Monkey-patch net.update_all_layers_batch so that each call records:
        • net._ei_ratio_log – |I_inh| / I_exc   (skips “silent” items)
        • net._pop_cv_log   – population CV(ISI)

    The guard now works **per batch-item**, eliminating the rare 30-50×
    outliers you saw with global-sum pooling.

    Parameters
    ----------
    net : MultiBatchAudVisMSINetworkTime
    exc_threshold : float
        Minimum excitatory current **per batch-item** required for that
        item to be included in the ratio (default = 5 × 10⁻⁴).
    """
    if getattr(net, "_ei_patched", False):  # only patch once
        return net

    net._ei_ratio_log = []
    net._pop_cv_log = []

    _orig = net.update_all_layers_batch  # keep original method

    # -----------------------------------------------------------------------
    #  FULL wrapped() FUNCTION
    # -----------------------------------------------------------------------
    def wrapped(self, *args, **kwargs):
        """
        Proxy that calls the original update, then logs EI-ratio and
        population spike irregularity for the *active* batch items.
        """
        out = _orig(*args, **kwargs)

        # ── currents (batch, neuron) ───────────────────────────────────────
        I_exc = self.I_A + self.I_V + torch.clamp(self.I_M, min=0)
        I_inh = torch.clamp(-self.I_M, min=0)

        # per-item excitation summed over neurons --------------------------
        exc_per_item = I_exc.sum(dim=1)  # shape (B,)
        active_mask = exc_per_item > exc_threshold  # bool mask (B,)

        if active_mask.any():  # at least one active seq
            total_exc = exc_per_item[active_mask].sum().item()
            total_inh = I_inh.sum(dim=1)[active_mask].sum().item()
            self._ei_ratio_log.append(total_inh / total_exc)

            pop_counts = (self._latest_sMSI * active_mask.unsqueeze(1)).sum(dim=1)
            μ = pop_counts.mean().item() + 1e-9
            cv = pop_counts.std(unbiased=False).item() / μ
            self._pop_cv_log.append(cv)

        return out

    # -----------------------------------------------------------------------

    # hot-swap the instance method
    net.update_all_layers_batch = wrapped.__get__(net, net.__class__)
    net._ei_patched = True

    # flush residual currents / delay buffers so the first logged values
    # reflect the *current* connectivity state
    net.reset_state()

    return net


###############################################################################
# EXPERIMENT 1 – QUICK EI BALANCE PROBE
###############################################################################

def ei_balance_probe(net, n_sequences: int = 50, batch_size: int = 64):
    """Run *n_sequences* random AV sequences and plot EI‑ratio histogram."""

    instrument_network(net)
    net._ei_ratio_log.clear()
    net._pop_cv_log.clear()

    while n_sequences > 0:
        bs = min(batch_size, n_sequences)
        loc_seqs, mod_seqs, offset_applied, seq_lengths = generate_event_loc_seq_batch(bs)
        max_len = max(seq_lengths)
        xA, xV, mask = generate_av_batch_tensor(loc_seqs, mod_seqs, offset_applied,
                                                n=net.n, space_size=net.space_size,
                                                sigma_in=net.sigma_in, noise_std=net.noise_std,
                                                device=net.device, max_len=max_len)
        net.reset_state(bs)
        for t in range(max_len):
            net.update_all_layers_batch(xA[:, t], xV[:, t], mask[:, t])
        n_sequences -= bs

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(net._ei_ratio_log, bins=40, color='slateblue', edgecolor='black')
    plt.xlabel('|I_inh| / I_exc')
    plt.ylabel('count')
    plt.title('EI‑ratio distribution')

    plt.subplot(1, 2, 2)
    plt.plot(net._pop_cv_log)
    plt.xlabel('time step')
    plt.ylabel('CV(ISI)')
    plt.title('Population irregularity')
    plt.tight_layout();
    plt.show()


###############################################################################
# EXPERIMENT 2 – FUNCTIONAL PERTURBATION OF INHIBITION
###############################################################################

def perturb_inhibition_sweep(
        net,
        scales=(1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0),
        batch_size=64,
        rescale_input=False,
):
    # pristine copies
    W_gate0, W_A0, W_V0 = (net.W_msiInh2Exc_GABA.clone(),
                           net.W_inA_inh.clone(),
                           net.W_inV_inh.clone())
    g_gaba0 = net.g_GABA

    # logs
    err_b, err_a, err_v, gain = [], [], [], []

    for g in scales:
        # scale every inhibitory knob
        net.W_msiInh2Exc_GABA = W_gate0 * g
        net.W_inA_inh = W_A0 * g
        net.W_inV_inh = W_V0 * g
        net.g_GABA = g_gaba0 * g

        if rescale_input:
            net.auto_calibrate_input_gain(target=100, max_iter=6)

        # three errors
        e_b = net.evaluate_batch(120, 'both', batch_size=batch_size)
        e_a = net.evaluate_batch(120, 'audio_only', batch_size=batch_size)
        e_v = net.evaluate_batch(120, 'visual_only', batch_size=batch_size)

        err_b.append(e_b)
        err_a.append(e_a)
        err_v.append(e_v)
        gain.append(min(e_a, e_v) - e_b)

    # restore state
    net.W_msiInh2Exc_GABA, net.W_inA_inh, net.W_inV_inh = W_gate0, W_A0, W_V0
    net.g_GABA = g_gaba0

    # figure
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(scales, err_b, 'o-', label='Bimodal err')
    ax1.plot(scales, err_a, '^-', label='Audio err', alpha=0.6)
    ax1.plot(scales, err_v, 'v-', label='Visual err', alpha=0.6)
    ax1.set_xlabel('Inhibitory weight scale')
    ax1.set_ylabel('Error (deg)');
    ax1.grid(True);
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(scales, gain, 's--', color='tab:red', label='MSI benefit')
    ax2.axhline(0, ls=':', color='tab:red')
    ax2.set_ylabel('MSI benefit (deg)')
    ax2.legend(loc='lower right')

    fig.tight_layout();
    plt.show()

    return {'scale': scales, 'bimodal_err': err_b,
            'audio_err': err_a, 'visual_err': err_v,
            'msi_gain': gain}


###############################################################################
# EXPERIMENT 3 – γ‑POWER VS INHIBITION
###############################################################################

def gamma_power_vs_inhibition(net, scales=(1, 0.6, 0.3, 0), duration=500):
    orig_W = net.W_msiInh2Exc_GABA.clone()
    instrument_network(net)
    gauss = torch.zeros((1, net.n), device=net.device)
    gauss[:, net.n // 2] = 3.0
    traces = []
    for g in scales:
        net.W_msiInh2Exc_GABA = orig_W * g
        net.reset_state(1)
        trace = []
        for _ in range(duration):
            net.update_all_layers_batch(gauss, gauss)
            trace.append(net._latest_sMSI.sum().item())
        traces.append(np.asarray(trace))
    net.W_msiInh2Exc_GABA = orig_W

    plt.figure(figsize=(8, 4))
    for tr, g in zip(traces, scales):
        f, Pxx = welch(tr, fs=1.0, nperseg=256)
        plt.semilogy(f, Pxx, label=f'g={g}')
    plt.xlim(0, 0.5)
    plt.xlabel('frequency')
    plt.ylabel('PSD')
    plt.legend();
    plt.title('γ‑band power vs inhibition');
    plt.tight_layout();
    plt.show()


########################################################
#       *** ADDED INVERSE EFFECTIVENESS TEST CALL ***
########################################################

def analyze_inverse_effectiveness(net, intensities=[0.002, 0.003, 0.004, 0.005],
                                  recording_time=30):
    """
    Analyze inverse effectiveness by measuring firing rates across different intensities.
    We'll measure the MSI excit population. (If you want MSI_inh, add code similarly.)
    """
    old_noise_std = net.noise_std
    net.noise_std = 0.0

    old_gNMDA = net.gNMDA

    conditions = ['audio_only', 'visual_only', 'both']
    T = recording_time
    D = 15
    start_time = 5

    def create_input_batch(intensities_list, conds, Tsteps, Dwin, start_t):
        B = len(intensities_list) * len(conds)
        xA_b = torch.zeros((B, Tsteps, net.n), device=net.device)
        xV_b = torch.zeros((B, Tsteps, net.n), device=net.device)

        for i, intensity in enumerate(intensities_list):
            for j, cond in enumerate(conds):
                b_idx = i * len(conds) + j
                center_idx = net.n // 2
                if cond in ['audio_only', 'both']:
                    gaussA = make_gaussian_vector_batch_gpu(
                        torch.tensor([center_idx], device=net.device),
                        net.n, net.sigma_in, net.device
                    )[0] * intensity
                    for tau in range(start_t, start_t + Dwin):
                        xA_b[b_idx, tau] = gaussA
                if cond in ['visual_only', 'both']:
                    gaussV = make_gaussian_vector_batch_gpu(
                        torch.tensor([center_idx], device=net.device),
                        net.n, net.sigma_in, net.device
                    )[0] * intensity
                    for tau in range(start_t, start_t + Dwin):
                        xV_b[b_idx, tau] = gaussV
        return xA_b, xV_b

    def run_network_and_get_spikes(xA, xV, set_gNMDA=None):
        B = xA.shape[0]
        if set_gNMDA is not None:
            net.gNMDA = set_gNMDA

        net.reset_state(batch_size=B)
        spike_counts = np.zeros(B, dtype=np.float32)

        for t in range(T):
            _, _, _, _, volt_dict = net.update_all_layers_batch(
                xA[:, t], xV[:, t], record_voltages=True
            )
            sM_sub = volt_dict['sM_substeps'].sum(dim=(0, 2))  # shape (B,)
            spike_counts += sM_sub.cpu().numpy()

        return spike_counts

    xA_batch_on, xV_batch_on = create_input_batch(intensities, conditions, T, D, start_time)
    xA_batch_off, xV_batch_off = create_input_batch(intensities, conditions, T, D, start_time)

    # 1) NMDA ON
    spike_counts_on = run_network_and_get_spikes(xA_batch_on, xV_batch_on, set_gNMDA=old_gNMDA)
    # 2) NMDA OFF
    spike_counts_off = run_network_and_get_spikes(xA_batch_off, xV_batch_off, set_gNMDA=0.0)

    net.gNMDA = old_gNMDA
    total_time_steps = T * net.n_substeps

    N_int = len(intensities)
    N_cond = len(conditions)
    rates_on = spike_counts_on / (net.n * total_time_steps)
    rates_off = spike_counts_off / (net.n * total_time_steps)

    rates_on_2d = rates_on.reshape(N_int, N_cond)
    rates_off_2d = rates_off.reshape(N_int, N_cond)

    audio_rates_nmda_on = rates_on_2d[:, 0]
    visual_rates_nmda_on = rates_on_2d[:, 1]
    bimodal_rates_nmda_on = rates_on_2d[:, 2]

    audio_rates_nmda_off = rates_off_2d[:, 0]
    visual_rates_nmda_off = rates_off_2d[:, 1]
    bimodal_rates_nmda_off = rates_off_2d[:, 2]

    def compute_additivity_index(a_rate, v_rate, b_rate):
        eps = 1e-3
        sum_uv = np.maximum(a_rate + v_rate, eps)
        return b_rate / sum_uv

    additivity_index_nmda_on = compute_additivity_index(
        audio_rates_nmda_on, visual_rates_nmda_on, bimodal_rates_nmda_on
    )
    additivity_index_nmda_off = compute_additivity_index(
        audio_rates_nmda_off, visual_rates_nmda_off, bimodal_rates_nmda_off
    )

    log_intensities = np.log10(np.array(intensities) * 1000.0)

    net.noise_std = old_noise_std

    return {
        'intensities': intensities,
        'log_intensities': log_intensities.tolist(),
        'audio_rates_on': audio_rates_nmda_on.tolist(),
        'visual_rates_on': visual_rates_nmda_on.tolist(),
        'bimodal_rates_on': bimodal_rates_nmda_on.tolist(),
        'additivity_index_on': additivity_index_nmda_on.tolist(),
        'audio_rates_off': audio_rates_nmda_off.tolist(),
        'visual_rates_off': visual_rates_nmda_off.tolist(),
        'bimodal_rates_off': bimodal_rates_nmda_off.tolist(),
        'additivity_index_off': additivity_index_nmda_off.tolist()
    }


###############################################################################
# EXPERIMENT 4 – INVERSE EFFECTIVENESS CHECK
###############################################################################

def inverse_effectiveness_check(net, intensities=(0.002, 0.0022, 0.0025, 0.0028, 0.003, 0.0032,
                                                  0.0035, 0.0038, 0.004, 0.0042, 0.0045,
                                                  0.0047, 0.005, 0.0052, 0.0055, 0.0057, 0.0057,
                                                  0.006, 0.0062, 0.0065, 0.0067, 0.007, 0.0072, 0.0075, 0.0077,
                                                  0.008, 0.0082, 0.0085, 0.0087,
                                                  0.009, 0.0092, 0.0094, 0.0096, 0.0098, 0.01, 0.0102, 0.0105,
                                                  0.0107, 0.011, 0.0112, 0.0115, 0.0118, 0.012, 0.0122, 0.0125,
                                                  0.0128, 0.013, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055,
                                                  0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1, 0.15,
                                                  0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,
                                                  0.75, 0.8, 0.85, 0.9, 0.95, 1)):
    instrument_network(net)
    res = analyze_inverse_effectiveness(net, list(intensities))
    print(f"Mean EI ratio during run: {np.mean(net._ei_ratio_log):.2f}")
    plt.figure(figsize=(4, 3))
    plt.plot(res['additivity_index_on'], 'o-')
    plt.axhline(1, ls='--', c='k');
    plt.ylabel('b / (a+v)');
    plt.xlabel('intensity index');
    plt.title('Inverse effectiveness');
    plt.tight_layout();
    plt.show()
    return res


# ─────────────────────────────────────────────────────────────
#  ASSAY A – RF-SHARPENING BY SURROUND
# ─────────────────────────────────────────────────────────────
def rf_width_probe(net, stim_intensity=0.3, n_steps=5):
    """
    Fires a single-pixel Gaussian at every spatial position and
    returns the mean FWHM (neurons) of the MSI hill.
    """
    widths = []
    gauss = torch.zeros((1, net.n), device=net.device)
    for idx in range(net.n):
        gauss.zero_()
        gauss[0, idx] = stim_intensity
        net.reset_state(1)
        for _ in range(n_steps):
            net.update_all_layers_batch(gauss, gauss)
        hill = net._latest_sMSI[0].cpu()
        widths.append((hill > 0.5 * hill.max()).sum().item())
    return float(np.mean(widths))


# ─────────────────────────────────────────────────────────────
#  ASSAY B – WINNER-TAKE-ALL  (hill-integrated read-out)
# ─────────────────────────────────────────────────────────────
def wta_probe(net, sep_deg=20, strong_amp=1.0, weak_amp=0.5,
              n_steps=60, window=2):
    """
    Two simultaneous Gaussians, ‘strong’ and ‘weak’, separated by
    *sep_deg*.  Returns the *hill-integrated* spike count for each
    after n_steps.

    window = ±neurons around each centre that are summed (default 2).
    """
    # 1) build the two hills ------------------------------------------------
    mid = net.n // 2
    offset = int(round(sep_deg / (net.space_size / net.n)))
    idx1, idx2 = mid, (mid + offset) % net.n

    g1 = make_gaussian_vector_batch_gpu(torch.tensor([idx1], device=net.device),
                                        net.n, 5.0, net.device)[0] * strong_amp
    g2 = make_gaussian_vector_batch_gpu(torch.tensor([idx2], device=net.device),
                                        net.n, 5.0, net.device)[0] * weak_amp
    stim = (g1 + g2).unsqueeze(0)  # add batch axis

    # 2) temporarily weaken ST-depression so the strong hill
    #    doesn't burn out before the read-out ------------------------------
    u_save, v_save = net.u_a.clone(), net.u_v.clone()
    net.u_a.fill_(0.05);
    net.u_v.fill_(0.05)

    # 3) run the simulation -----------------------------------------------
    # 3) run & ACCUMULATE spikes over time  ──────────────────────────────
    net.reset_state(1)
    hist = torch.zeros(net.n, device=net.device)  # NEW

    for _ in range(n_steps):
        net.update_all_layers_batch(stim, stim)
        hist += net._latest_sMSI[0]  # accumulate

    # 4) restore STP ------------------------------------------------------

    # 4) restore STP parameters -------------------------------------------
    net.u_a.copy_(u_save);
    net.u_v.copy_(v_save)

    # 5) integrate spikes in ±window neurons ------------------------------
    def hill_sum(idx):
        rng = [(idx + i) % net.n for i in range(-window, window + 1)]
        return hist[rng].sum().item()

    return hill_sum(idx1), hill_sum(idx2)


# ─────────────────────────────────────────────────────────────
#  ASSAY C – SIZE-TUNING / SURROUND SUPPRESSION
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
#  ASSAY C – SIZE-TUNING / SURROUND-SUPPRESSION  (updated)
# ─────────────────────────────────────────────────────────────
def size_tuning_curve(net, widths=range(1, 25, 2), stim_amp=0.4,
                      n_steps=800, window=None):
    """
    Present a square patch centred on the map midline, expand its
    half-width across *widths*, and accumulate MSI spikes.

    Parameters
    ----------
    net : MultiBatchAudVisMSINetworkTime
    widths : iterable[int]
        Half-width of the patch in *neurons* (deg ≈ neuron index).
    stim_amp : float
        Amplitude of the constant drive delivered to A and V layers.
    n_steps : int
        Number of macro-steps to run (dt * n_substeps each).
        With dt=0.1 ms and n_substeps=100, n_steps=800 ≈ 80 ms.
    window : int | None
        If given, sum spikes only within ±window neurons around the
        centre; otherwise sum the whole population.

    Returns
    -------
    list[float]
        Accumulated spike counts, one per width in *widths*.
    """
    responses = []
    mid = net.n // 2

    for r in widths:
        # 1) build the stimulus -------------------------------------------
        stim = torch.zeros((1, net.n), device=net.device)
        stim[0, mid - r: mid + r + 1] = stim_amp

        # 2) run the network & accumulate spikes --------------------------
        net.reset_state(1)
        hist = torch.zeros(net.n, device=net.device)

        for _ in range(n_steps):
            net.update_all_layers_batch(stim, stim)
            hist += net._latest_sMSI[0]  # accumulate sub-steps

        # 3) store summed response ----------------------------------------
        if window is None:
            responses.append(hist.sum().item())
        else:
            rng = [(mid + i) % net.n for i in range(-window, window + 1)]
            responses.append(hist[rng].sum().item())

    return responses


# ------------------------------------------------------------------
# Helper #1: build a clean A-only / V-only / A+V pulse batch
# ------------------------------------------------------------------
def build_pulse_batch(net, intensity, pulse_len=15, start=5):
    """
    Returns (xA, xV) with shape (1, T, n) where both modalities get the
    same Gaussian pulse centred on the map midline.  Everything else is 0.
    """
    # a few silent steps before and after the pulse
    T = start + pulse_len + 5
    xA = torch.zeros((1, T, net.n), device=net.device)
    xV = torch.zeros((1, T, net.n), device=net.device)

    centre = net.n // 2
    gauss = make_gaussian_vector_batch_gpu(
        torch.tensor([centre], device=net.device),
        net.n, net.sigma_in, net.device
    )[0] * intensity

    for t in range(start, start + pulse_len):
        xA[0, t] = gauss  # audio channel
        xV[0, t] = gauss  # visual channel

    return xA, xV


# ------------------------------------------------------------------
# Helper #2: run the network on that batch and get the spike-rate
# ------------------------------------------------------------------
def pulse_rate(net, xA, xV, layer='msi'):
    """
    Executes the network on (xA, xV) and returns spikes · neuron⁻¹ · s⁻¹
    for the requested layer.  Only time-steps that actually contain the
    stimulus contribute to the rate.
    """
    B, T, _ = xA.shape
    net.reset_state(B)

    spike_sum = torch.zeros(B, device=net.device)
    active_steps = torch.zeros(B, device=net.device)  # how many frames carry input

    for t in range(T):
        net.update_all_layers_batch(xA[:, t], xV[:, t])

        # was there any drive on this macro-step?
        active_now = ((xA[:, t].abs().sum(dim=1) +
                       xV[:, t].abs().sum(dim=1)) > 0).float()
        active_steps += active_now

        if layer == 'msi':
            spike_sum += net._latest_sMSI.sum(dim=1)
        elif layer == 'audio':
            spike_sum += net._latest_sA.sum(dim=1)
        elif layer == 'visual':
            spike_sum += net._latest_sV.sum(dim=1)
        elif layer == 'output':
            spike_sum += net._latest_sOut.sum(dim=1)
        else:
            raise ValueError(f"Unknown layer '{layer}'")

    # convert to spikes · neuron⁻¹ · s⁻¹
    # (n_substeps × dt  = 1 ms, so n_substeps steps = 1 ms)
    total_ms = active_steps * net.n_substeps * net.dt  # ms
    total_ms[total_ms == 0] = 1e-6  # avoid /0
    rate = (spike_sum / (net.n * total_ms * 1e-3)).cpu().numpy()  # → Hz

    # when B == 1 just return a scalar
    return rate[0] if B == 1 else rate


import numpy as np
import matplotlib.pyplot as plt


def run_inverse_effectiveness_test(
        net,
        intensities=[0.002, 0.003, 0.004, 0.006, 0.008, 0.01, 0.015,
                     0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5, 1.0],
        n_seq=50,
        batch_size=32,
        do_plot=True
):
    """
    Measures the network's MSI (multisensory integration) at different stimulus
    intensities and plots an 'inverse-effectiveness' curve:
      - x-axis: unimodal strength (avg of A-only & V-only firing rates).
      - y-axis: MEI = (Bimodal - UnimodalMean) / UnimodalMean

    Args:
      net          : Your MultiBatchAudVisMSINetworkTime (or similar).
      intensities  : List of stimulus intensities to test.
      n_seq        : How many random sequences to run for each intensity.
      batch_size   : Batch size to use when calling calculate_spike_rates_gpu.
      do_plot      : Whether to display the MEI vs unimodal-strength plot.

    Returns:
      A dict with:
        'intensities'   : the list of intensities tested
        'audio_rates'   : measured MSI-layer firing rates (A-only)
        'visual_rates'  : measured MSI-layer firing rates (V-only)
        'bimodal_rates' : measured MSI-layer firing rates (A+V)
        'unimodal_mean' : average of audio and visual rates
        'MEI'           : (bimodal - uni_mean) / uni_mean
    """
    audio_rates = []
    visual_rates = []
    bimodal_rates = []

    # 1) For each intensity, measure MSI spiking in audio-only, visual-only, and both
    for alpha in intensities:
        # audio-only
        a_dict = calculate_spike_rates_gpu(net,
                                           n_seq=n_seq,
                                           condition='audio_only',
                                           stimulus_intensity=alpha,
                                           batch_size=batch_size)
        a_rate = a_dict['msi']  # population rate in MSI layer

        # visual-only
        v_dict = calculate_spike_rates_gpu(net,
                                           n_seq=n_seq,
                                           condition='visual_only',
                                           stimulus_intensity=alpha,
                                           batch_size=batch_size)
        v_rate = v_dict['msi']

        # bimodal
        b_dict = calculate_spike_rates_gpu(net,
                                           n_seq=n_seq,
                                           condition='both',
                                           stimulus_intensity=alpha,
                                           batch_size=batch_size)
        b_rate = b_dict['msi']

        audio_rates.append(a_rate)
        visual_rates.append(v_rate)
        bimodal_rates.append(b_rate)

    audio_rates = np.array(audio_rates)
    visual_rates = np.array(visual_rates)
    bimodal_rates = np.array(bimodal_rates)

    # 2) Compute unimodal-mean and MEI
    unimodal_mean = 0.5 * (audio_rates + visual_rates)
    MEI = (bimodal_rates - unimodal_mean) / (unimodal_mean + 1e-9)

    # 3) Plot if requested
    if do_plot:
        plt.figure(figsize=(6, 4))
        plt.plot(unimodal_mean, MEI, 'o-')
        plt.title("Inverse-Effectiveness Test")
        plt.xlabel("Unimodal Strength (avg of Audio & Visual rates)")
        plt.ylabel("MEI = (Bimodal - UniMean) / UniMean")
        plt.grid(True)
        plt.show()

    # 4) Return all results in a dict
    return {
        'intensities': intensities,
        'audio_rates': audio_rates,
        'visual_rates': visual_rates,
        'bimodal_rates': bimodal_rates,
        'unimodal_mean': unimodal_mean,
        'MEI': MEI
    }


def run_av_pair(net, aud_deg=0, vis_deg=0, *,
                duration=30,  # macro-steps (≈3 ms each if dt=0.1 ms & n_substeps=100)
                intensity=0.4,
                return_error=False):
    """Fire synchronous A+V Gaussians and read out either
       MSI population activity or localisation error.
    """
    # --- build the two Gaussians ----------------------------------------
    idx_a = location_to_index(aud_deg, net.n, net.space_size)
    idx_v = location_to_index(vis_deg % net.space_size,
                              net.n, net.space_size)

    gA = make_gaussian_vector_batch_gpu(
        torch.tensor([idx_a], device=net.device),
        net.n, net.sigma_in, net.device)[0] * intensity
    gV = make_gaussian_vector_batch_gpu(
        torch.tensor([idx_v], device=net.device),
        net.n, net.sigma_in, net.device)[0] * intensity

    xA = gA.unsqueeze(0)  # add batch axis
    xV = gV.unsqueeze(0)

    # --- run -------------------------------------------------------------
    net.reset_state(batch_size=1)
    pop_spikes = 0.0
    for _ in range(duration):
        net.update_all_layers_batch(xA, xV)
        pop_spikes += net._latest_sMSI.sum().item()  # all neurons, all sub-steps

    if return_error:  # localisation error instead
        sM = net._latest_sMSI[0]
        raw = torch.matmul(net.W_msi2out, sM) + net.b_out
        pred_idx = torch.argmax(torch.clamp(raw, min=0)).item()
        pred_deg = index_to_location(pred_idx, net.n, net.space_size)
        err = abs(pred_deg - aud_deg)  # unsigned error
        return err
    else:
        return pop_spikes


import numpy as np, scipy.optimize as opt

def generate_spatial_offset_batch(
    offsets,
    loc=90,
    T=60,
    D=5,
    space_size=180
):
    """
    For each offset in 'offsets', we create one sequence in which:
      - Audio is at 'loc'
      - Visual is at 'loc + offset' (wrapped if needed)
      - Both occur simultaneously for D steps, within a T-step sequence.

    Returns:
      loc_seqs, mod_seqs, offset_applied, seq_lengths
    """
    loc_seqs = []
    mod_seqs = []
    offset_applied = []
    seq_lengths = []

    for off in offsets:
        # Wrap the visual location if it falls outside [0, space_size)
        vis_loc = (loc + off) % space_size

        seq_loc = [999]*T
        seq_mod = ['X']*T
        # Let the bursts occur from t=0..D-1 (you can shift them later if desired)
        for t in range(D):
            seq_loc[t] = loc      # audio
            seq_mod[t] = 'A'
        for t in range(D):
            # If this time slot is already 'A', we combine into 'B'
            if seq_mod[t] == 'A':
                seq_mod[t] = 'B'
            else:
                seq_loc[t] = vis_loc
                seq_mod[t] = 'V'

        loc_seqs.append(seq_loc)
        mod_seqs.append(seq_mod)
        offset_applied.append(False)  # not used here, but part of generate_av_batch_tensor’s signature
        seq_lengths.append(T)

    return loc_seqs, mod_seqs, offset_applied, seq_lengths

def run_spatial_integration(net, offsets, loc=90, T=60, D=5):
    """
    offsets  : list of integer (or float) offsets in 'degrees' of space
    loc      : the reference location (degrees)
    T        : total macro-steps to run
    D        : duration (macro-steps) of simultaneous A+V bursts
    returns  : dict with fields:
                 'spike_raster' -> shape (T, len(offsets)) with population-summed spikes at each time
                 'int_spikes'   -> 1D array of integrated spikes over a 0–100 ms window
                 'offsets_deg'  -> the actual offsets in degrees
    """
    # 1) Build batch sequences for each offset
    loc_seqs, mod_seqs, offset_applied, seq_lengths = generate_spatial_offset_batch(
        offsets, loc=loc, T=T, D=D, space_size=net.space_size
    )
    max_len = max(seq_lengths)  # should be T

    # 2) Convert to audio/visual input tensors
    xA, xV, mask = generate_av_batch_tensor(
        loc_seqs, mod_seqs, offset_applied,
        n=net.n,
        space_size=net.space_size,
        sigma_in=net.sigma_in,
        noise_std=0.0,  # keep minimal noise if you want a clean measure
        device=net.device,
        max_len=max_len
    )

    # 3) Run the network
    B = len(offsets)
    net.reset_state(batch_size=B)

    # We'll record the total (population) spikes in MSI for each offset at each time
    spike_raster = torch.zeros((max_len, B), device=net.device)

    for t in range(max_len):
        net.update_all_layers_batch(xA[:, t], xV[:, t], mask[:, t])
        # sum across MSI neurons => shape (B,)
        spike_raster[t] = net._latest_sMSI.sum(dim=1)

    # 4) Integrate spikes in (e.g.) the first 10 macro-steps = 0..100 ms if dt=0.1 and n_substeps=100
    #    (You can adjust the exact window to match your code’s assumptions.)
    #    Let’s pick the same 0–100 ms slice you used in temporal_integration.
    window = slice(0, D+5)  # e.g. D steps plus a few
    int_spikes = spike_raster[window].sum(dim=0).cpu().numpy()

    return {
        'spike_raster': spike_raster.cpu().numpy(),  # shape (T, B)
        'int_spikes': int_spikes,
        'offsets_deg': offsets
    }

def plot_spatial_binding(results):
    import matplotlib.pyplot as plt

    rast = results['spike_raster']   # shape (T, B)
    ints = results['int_spikes']
    offs = results['offsets_deg']
    Tdim, B = rast.shape

    # Heatmap
    plt.figure(figsize=(8, 4))
    plt.imshow(rast, aspect='auto', origin='lower',
               extent=[offs[0], offs[-1], 0, Tdim],  # x from first->last offset, y from 0->T
               cmap='viridis')
    plt.colorbar(label='MSI pop-spikes / time-step')
    plt.xlabel('Audio–Visual offset (degrees)')
    plt.ylabel('Time (macro-steps)')
    plt.title('MSI activity vs spatial offset')

    # Binding curve
    plt.figure(figsize=(4, 3))
    plt.plot(offs, ints, 'o-')
    plt.axvline(0, ls='--', c='k')
    plt.xlabel('Audio–Visual offset (degrees)')
    plt.ylabel('Integrated spikes (0–100 ms window)')
    plt.title('Spatial binding window')
    plt.tight_layout()
    plt.show()



########################################################
#                MAIN TRAINING SCRIPT
########################################################
from scipy.stats import binned_statistic
from statsmodels.nonparametric.smoothers_lowess import lowess


def run_training(
        batch_size=256,
        n_unsup_epochs=15,
        sequences_per_unsup_epoch=1000,
        n_readout_epochs=6,
        sequences_per_readout_epoch=3000
):
    """
    1) Initialize net
    2) Unsupervised STDP
    3) Evaluate readout error
    4) Supervised readout training
    5) Evaluate final error
    6) Inverse effectiveness test
    7) MSI enhancement test
    """
    print(f"Initializing network with batch size {batch_size}...")
    net = MultiBatchAudVisMSINetworkTime(
        n_neurons=180,
        batch_size=batch_size,
        lr_unimodal=5e-5,
        lr_msi=5e-5,
        lr_readout=1e-3,
        sigma_in=5.0,
        sigma_teacher=2.0,  # (not used directly in final, replaced by scheduling)
        noise_std=0.1,
        single_modality_prob=0.5,
        v_thresh=0.25,
        dt=0.1,
        tau_m=20.0,
        n_substeps=100,
        loc_jitter_std=0,
        space_size=180,
        conduction_delay_a2msi=5,
        conduction_delay_v2msi=6,
        conduction_delay_msi2out=5
    )

    import matplotlib.pyplot as plt

    init_W_inA = net.W_inA.clone().cpu().numpy()
    init_W_inV = net.W_inV.clone().cpu().numpy()

    net.set_inhib_plasticity(True)

    # Unsupervised STDP
    print("\n--- STDP training (unsupervised) ---")
    unsup_start = time.time()
    for epoch in range(n_unsup_epochs):
        epoch_start = time.time()
        net.train_unsupervised_batch(sequences_per_unsup_epoch, batch_size=batch_size)
        epoch_time = time.time() - epoch_start
        print(f"  Unsup Epoch {epoch + 1}/{n_unsup_epochs} - Time: {epoch_time:.2f}s")
    unsup_time = time.time() - unsup_start
    print(f"Unsupervised training completed in {unsup_time:.2f}s")

    net.set_inhib_plasticity(False)

    net.reset_state()


    # Evaluate readout error
    print("\nEvaluating unsupervised readout performance...")
    init_err_both = net.evaluate_batch(100, condition='both', batch_size=batch_size)
    init_err_audio = net.evaluate_batch(100, condition='audio_only', batch_size=batch_size)
    init_err_vis = net.evaluate_batch(100, condition='visual_only', batch_size=batch_size)
    print(f"  Both:        {init_err_both:.2f} deg")
    print(f"  Audio-only:  {init_err_audio:.2f} deg")
    print(f"  Visual-only: {init_err_vis:.2f} deg")

    # Supervised readout training
    print("\n--- Readout training (unimodal + bimodal mixture) ---")
    readout_start = time.time()
    conditions = ['audio_only', 'visual_only', 'both']
    condition_probs = [0.1, 0.1, 0.8]

    for epoch in range(n_readout_epochs):
        epoch_start = time.time()
        net.train_readout_batch(sequences_per_readout_epoch,
                                conditions=conditions,
                                condition_probs=condition_probs,
                                batch_size=batch_size,
                                epoch_idx=epoch)

        # homeostatic row-normalisation once per epoch
        # net.normalize_rows_gpu(net.W_msiInh2Exc_GABA)
        # net.normalize_rows_gpu(net.W_inA_inh)
        # net.normalize_rows_gpu(net.W_inV_inh)
        net.normalize_rows_gpu(net.W_msi2out)

        err_b = net.evaluate_batch(30, condition='both', batch_size=batch_size)
        err_a = net.evaluate_batch(30, condition='audio_only', batch_size=batch_size)
        err_v = net.evaluate_batch(30, condition='visual_only', batch_size=batch_size)

        epoch_time = time.time() - epoch_start
        print(f"  Readout Epoch {epoch + 1}/{n_readout_epochs} - Time: {epoch_time:.2f}s")
        print(f"    Errors => Both:{err_b:.2f}, Audio:{err_a:.2f}, Visual:{err_v:.2f}")
    readout_time = time.time() - readout_start
    print(f"Readout training completed in {readout_time:.2f}s")

    total_time = unsup_time + readout_time
    print(f"\nTotal training time: {total_time:.2f}s")

    if __name__ == "__main__":
    offsets_deg = range(-60, 61, 10)  # from -30 to +30 in steps of 5 deg
    spatial_res = run_spatial_integration(net, offsets_deg, loc=90, T=60, D=5)
    plot_spatial_binding(spatial_res)


    # MSI enhancement test
    print("\n--- Testing MSI enhancement with 30ms sequences & 10ms events ---")
    check_msi_enhancement(net, space_size=180, T=30, D=10, n_per_loc=20)

    return {
        'network': net,
        'init_weights': {
            'W_inA': init_W_inA,
            'W_inV': init_W_inV,
        },
        'errors': {
            'unsup_batchsize': {
                'both': init_err_both,
                'audio': init_err_audio,
                'visual': init_err_vis
            },
            'final_batch1': {}
        },
        'timing': {
            'unsupervised': unsup_time,
            'readout': readout_time,
            'total': total_time
        }
    }


# Example usage (main execution):

if __name__ == "__main__":
    results = run_training()
    net = results['network']

    # --------- NEW: temporal integration sweep ----------
    offsets = list(range(-10, 11))  # −100 … +100 ms in 10 ms steps
    res_ti = run_temporal_integration(net, offsets, loc=90, T=60, D=5)
    plot_temporal_binding(res_ti)

    perturb_inhibition_sweep(net)

    # ── 1. Isolate the surround  ───────────────────────────
    net.isolate_surround(True)

    # ── 2. RF width check  ─────────────────────────────────
    width_iso = rf_width_probe(net)
    print(f"[ASSAY A] mean RF width (surround-only) : {width_iso:.1f} neurons")

    # control: turn *off* surround as well
    g_old = net.g_GABA
    net.g_GABA = 0.0
    width_nogaba = rf_width_probe(net)
    net.g_GABA = g_old
    print(f"[ASSAY A] mean RF width (no inhibition) : {width_nogaba:.1f} neurons")

    # ── 3. Winner-take-all  ───────────────────────────────
    w_strong, w_weak = wta_probe(net)
    print(f"[ASSAY B] spike strong vs weak : {w_strong:.1f} / {w_weak:.1f}")

    # ── 4. Size-tuning curve  ─────────────────────────────
    widths = list(range(1, 25, 2))
    resp = size_tuning_curve(net, widths)
    print("[ASSAY C] size-tuning (mid-patch widths) :", resp)

    # Flash-Sound test
    offsets = list(range(21))
    visualize_flash_sound_msi(net, offsets=offsets, T=50, D=10, loc=90)
