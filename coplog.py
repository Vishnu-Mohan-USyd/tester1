import math

import numpy as np
import torch
import time
from collections import deque
import matplotlib.pyplot as plt
import scipy.stats as stats  # for Gaussian fitting
from scipy.signal import welch
import os


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

class MultiBatchAudVisMSINetworkTime:
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

        # define an MSI inhibitory subpopulation
        self.n_inh = int(0.3 * n_neurons)
        if self.n_inh < 1:
            self.n_inh = 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.input_scaling = 60.0

        # ------------- Weights: In -> Uni(A/V) --------------
        self.W_inA = torch.tensor(0.01 * np.random.randn(self.n, self.n),
                                  dtype=torch.float32, device=self.device)
        self.W_inV = torch.tensor(0.01 * np.random.randn(self.n, self.n),
                                  dtype=torch.float32, device=self.device)

        # ------------- A->MSI (exc) and V->MSI (exc) --------------
        init_a2msi = torch.tensor(0.005 * np.random.randn(self.n, self.n),
                                  dtype=torch.float32, device=self.device)
        init_v2msi = torch.tensor(0.005 * np.random.randn(self.n, self.n),
                                  dtype=torch.float32, device=self.device)

        self.W_a2msi_AMPA = init_a2msi.clone() * 0.05 * 0.8
        self.W_a2msi_NMDA = init_a2msi.clone() * 0.95 * 0.8
        self.W_v2msi_AMPA = init_v2msi.clone() * 0.05 * 0.8
        self.W_v2msi_NMDA = init_v2msi.clone() * 0.95 * 0.8

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

        self.W_a2msiInh_AMPA = init_a2msi_inh.clone() * 0.05 * 5.0
        self.W_a2msiInh_NMDA = init_a2msi_inh.clone() * 0.95 * 15.0
        self.W_v2msiInh_AMPA = init_v2msi_inh.clone() * 0.05 * 5.0
        self.W_v2msiInh_NMDA = init_v2msi_inh.clone() * 0.95 * 15.0

        # ------------- MSI_inh -> MSI_exc (GABA) --------------
        self.W_msiInh2Exc_GABA = torch.tensor(0.003 * np.random.randn(self.n, self.n_inh),
                                              dtype=torch.float32, device=self.device)

        # ------------- MSI -> Out --------------
        self.W_msi2out = torch.tensor(0.01 * np.random.randn(self.n, self.n),
                                      dtype=torch.float32, device=self.device)

        # ------------- Biases --------------
        self.b_uniA = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        self.b_uniV = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        self.b_msi = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        self.b_msi_inh = torch.zeros(self.n_inh, dtype=torch.float32, device=self.device)
        self.b_out = torch.zeros(self.n, dtype=torch.float32, device=self.device)

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
        self.conduction_delay_msi_inh2exc = 3
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
        self.gNMDA = 0.15
        self.tau_nmda = 220.0
        self.nmda_alpha = 0.05
        self.mg_k = 0.1
        self.Erev_nmda = 10.0
        self.tau_nmdaVolt = 220.0
        self.v_nmda_rest = -65.0
        self.nmda_vrest_offset = 7.0
        self.mg_vhalf = -55.0

        # Two dendrites for mg-block gating in MSI excit:
        self.dend_coupling_alpha = 0.1
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

        # ------------- surround inhibition in MSI excit (Mexican-hat) --------------
        self.g_GABA = 0.4
        self.W_MSI_inh = torch.zeros((self.n, self.n), device=self.device)
        for i in range(self.n):
            for j in range(self.n):
                d = min(abs(i - j), self.n - abs(i - j))
                self.W_MSI_inh[i, j] = (
                        np.exp(-(d / 3.0) ** 2)
                        - 0.8 * np.exp(-(d / 12.0) ** 2)
                )
        self.W_MSI_inh.clamp_(min=0)

        # self.g_GABA *= 15
        #
        # # 2) Recurrent MSI_inh -> MSI_exc weights
        # self.W_msiInh2Exc_GABA.mul_(20)
        #
        # # 3) Feed‑forward inhibition from A and V layers
        # self.W_inA_inh.mul_(20)
        # self.W_inV_inh.mul_(20)

        self.auto_calibrate_input_gain()

        # ------------------------------------------------------------------
        #                       calibration helpers
        # ------------------------------------------------------------------

    def _probe_spike_sum(self):
        """
        Return MSI spike sum for a 3step Gaussian pulse delivered to the
        centre neuron. Uses the current `input_scaling`.
        """
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
        Binarysearch `self.input_scaling` until the singlepulse probe
        produces `target � tol` MSI spikes.
        """
        print("Calibrating input gain &")
        low, high = 0.0, high_bound
        best_gain = self.input_scaling
        best_err = float('inf')

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
        print(f" input_scaling final = {self.input_scaling:4.2f}  "
              f"(spike sum {int(self._probe_spike_sum())})")

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

            # Anti-Hebbian STDP for W_MSI_inh
            dw_inh = -0.0005 * torch.bmm(
                self._latest_sMSI.unsqueeze(2),
                self._latest_sMSI.unsqueeze(1)
            ).mean(dim=0)
            self.W_MSI_inh.add_(dw_inh)
            self.W_MSI_inh.clamp_(min=0)

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

            # MSI inh
            dVMi = (0.04 * self.v_msi_inh.pow(2) + 5.0 * self.v_msi_inh + 140.0
                    - self.u_msi_inh + self.I_M_inh)
            self.v_msi_inh += self.dt * dVMi
            self.u_msi_inh += self.dt * (self.aMi * (self.bMi * self.v_msi_inh - self.u_msi_inh))
            new_sMi = (self.v_msi_inh >= spike_threshold).float()
            self.v_msi_inh[self.v_msi_inh >= spike_threshold] = self.cMi
            self.u_msi_inh[self.v_msi_inh == self.cMi] += self.dMi

            # Out
            dVO = (0.04 * self.v_out.pow(2) + 5.0 * self.v_out + 140.0 - self.u_out + self.I_O)
            self.v_out += self.dt * dVO
            self.u_out += self.dt * (self.aO * (self.bO * self.v_out - self.u_out))
            new_sO = (self.v_out >= spike_threshold).float()
            self.v_out[self.v_out >= spike_threshold] = self.cO
            self.u_out[self.v_out == self.cO] += self.dO

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

        if record_voltages:
            sM_substeps_tensor = torch.stack(sM_substeps, dim=0)
            volt_dict = {'sM_substeps': sM_substeps_tensor}
            return sA, sV, sM, sO, volt_dict
        else:
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
                self.W_v2msi_AMPA = 0.2 * W_v2msi_total
                self.W_v2msi_NMDA = 0.8 * W_v2msi_total

                # STDP: Uni(A)->MSI (AMPA+NMDA)
                W_a2msi_total = self.W_a2msi_AMPA + self.W_a2msi_NMDA
                self.pre_trace_a2msi, self.post_trace_a2msi, _ = self.stdp_update_batch(
                    W_a2msi_total, sMSI, sA,
                    self.post_trace_a2msi, self.pre_trace_a2msi, self.lr_msi
                )
                self.normalize_rows_gpu(W_a2msi_total)
                self.W_a2msi_AMPA = 0.2 * W_a2msi_total
                self.W_a2msi_NMDA = 0.8 * W_a2msi_total

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
    loc_seq = [999] * T
    mod_seq = ["X"] * T

    # Audio event from t=0..(D-1)
    for t in range(D):
        loc_seq[t] = loc
        mod_seq[t] = 'A'

    # Visual event from t=offset..offset+(D-1)
    start_v = offset
    end_v = offset + D
    if end_v > T:
        end_v = T
    for t in range(start_v, end_v):
        if loc_seq[t] == 999:
            loc_seq[t] = loc
            mod_seq[t] = 'V'
        elif mod_seq[t] == 'A':
            mod_seq[t] = 'B'

    return loc_seq, mod_seq


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

    # # Plot NMDA ON
    # plt.figure(figsize=(12, 10))
    # plt.subplot(2, 1, 1)
    # plt.plot(log_intensities, audio_rates_nmda_on * 100.0, 'r-o', label='Audio')
    # plt.plot(log_intensities, visual_rates_nmda_on * 100.0, 'b-o', label='Visual')
    # plt.plot(log_intensities, bimodal_rates_nmda_on * 100.0, 'k-o', label='Bimodal')
    # plt.plot(log_intensities, (audio_rates_nmda_on + visual_rates_nmda_on) * 100.0,
    #          'k--', label='Sum(A+V)')
    # plt.xlabel('log10(intensity * 1000)')
    # plt.ylabel('Firing rate (arbitrary scale)')
    # plt.title('Inverse Effectiveness - NMDA ON')
    # plt.legend()
    # plt.grid(True)

    # plt.subplot(2, 1, 2)
    # plt.plot(log_intensities, additivity_index_nmda_on, 'r-o')
    # plt.axhline(1.0, color='k', linestyle='--')
    # plt.xlabel('log10(intensity * 1000)')
    # plt.ylabel('Additivity index')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # # Plot NMDA OFF
    # plt.figure(figsize=(12, 10))
    # plt.subplot(2, 1, 1)
    # plt.plot(log_intensities, audio_rates_nmda_off * 100.0, 'r-o', label='Audio')
    # plt.plot(log_intensities, visual_rates_nmda_off * 100.0, 'b-o', label='Visual')
    # plt.plot(log_intensities, bimodal_rates_nmda_off * 100.0, 'k-o', label='Bimodal')
    # plt.plot(log_intensities, (audio_rates_nmda_off + visual_rates_nmda_off) * 100.0,
    #          'k--', label='Sum(A+V)')
    # plt.xlabel('log10(intensity * 1000)')
    # plt.ylabel('Firing rate (arbitrary scale)')
    # plt.title('Inverse Effectiveness - NMDA OFF')
    # plt.legend()
    # plt.grid(True)

    # plt.subplot(2, 1, 2)
    # plt.plot(log_intensities, additivity_index_nmda_off, 'g-o')
    # plt.axhline(1.0, color='k', linestyle='--')
    # plt.xlabel('log10(intensity * 1000)')
    # plt.ylabel('Additivity index')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # # Compare NMDA ON vs OFF
    # plt.figure(figsize=(12, 10))
    # plt.subplot(2, 1, 1)
    # plt.plot(log_intensities, bimodal_rates_nmda_on * 100.0, 'r-o', label='Bimodal (NMDA ON)')
    # plt.plot(log_intensities, bimodal_rates_nmda_off * 100.0, 'g-o', label='Bimodal (NMDA OFF)')
    # plt.plot(log_intensities, (audio_rates_nmda_on + visual_rates_nmda_on) * 100.0,
    #          'r--', label='Sum(A+V) ON')
    # plt.plot(log_intensities, (audio_rates_nmda_off + visual_rates_nmda_off) * 100.0,
    #          'g--', label='Sum(A+V) OFF')
    # plt.xlabel('log10(intensity * 1000)')
    # plt.ylabel('Firing rate (arbitrary scale)')
    # plt.title('Inverse Effectiveness - NMDA ON vs OFF (Bimodal)')
    # plt.legend()
    # plt.grid(True)

    # plt.subplot(2, 1, 2)
    # plt.plot(log_intensities, additivity_index_nmda_on, 'r-o', label='NMDA ON')
    # plt.plot(log_intensities, additivity_index_nmda_off, 'g-o', label='NMDA OFF')
    # plt.axhline(1.0, color='k', linestyle='--')
    # plt.xlabel('log10(intensity * 1000)')
    # plt.ylabel('Additivity index')
    # plt.legend()
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()

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
      rerun.  After *max_retries* attempts, returns ``np.nan``.
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


# ---------------------------------------------------------------------------
# 3.  AUTO‑CALIBRATION LOOP WITH NAN HANDLING
# ---------------------------------------------------------------------------

import math
import numpy as np


def pretrain_autocalibrate(
        net,
        target_ei: float = 1.0,
        ei_tol: float = 0.2,
        max_passes: int = 12,
        bg_lambda_hz: float = 800.0,
        bg_amp: float = 0.25,
        probe_steps: int = 120,
        excite_gain_target: int = 100,
) -> None:
    """
    Calibrate global inhibition so the network’s mean E/I ratio – measured with
    a quick Poisson-drive probe – lands at `target_ei ± ei_tol`.

    Improvements over the original version
    --------------------------------------
    1. **Median ratio** is used instead of the mean and zeros/NaNs are skipped,
       making the estimate less sensitive to a few silent steps.
    2. Whenever the network falls completely silent
       (ratio ≤ 1e-3 → over-inhibited), we *halve* every inhibitory knob *and*
       immediately re-run ``auto_calibrate_input_gain`` to boost excitation so
       the next probe has a real chance to spike.
    3. A short docstring and type hints for clarity.
    """
    import math
    import numpy as np

    # -- 1. Start with a strong inhibition “safety” boost
    strong_boost = 100.0
    net.g_GABA *= strong_boost
    net.W_msiInh2Exc_GABA *= strong_boost
    net.W_inA_inh *= strong_boost
    net.W_inV_inh *= strong_boost

    # -- 2. Make sure excitation is in the right ball-park *before* we start
    net.auto_calibrate_input_gain(target=excite_gain_target, max_iter=12, high_bound=400)

    def quick_ei_probe(n_macro_steps: int = probe_steps) -> float:
        """Return the *median* E/I ratio over `n_macro_steps` Poisson-drive steps."""
        instrument_network(net)
        net._ei_ratio_log.clear()

        net.reset_state(batch_size=1)
        dt_macro = net.dt * net.n_substeps
        p_event = bg_lambda_hz * dt_macro * 1e-3  # Poisson prob per neuron per macro-step
        for _ in range(n_macro_steps):
            mask = (torch.rand(1, net.n, device=net.device) < p_event)
            drive = bg_amp * mask.float()
            net.update_all_layers_batch(drive, drive)

        vals = [v for v in net._ei_ratio_log if v > 0 and math.isfinite(v)]
        return float(np.median(vals)) if vals else 0.0

    for p in range(1, max_passes + 1):
        ei_ratio = quick_ei_probe()
        print(f"[EI pass {p}]  median ratio = {ei_ratio:.2f}")

        if ei_ratio <= 1e-3:
            print("   • Network silent → halving inhibition and re-boosting excitation")
            for w in (net.W_msiInh2Exc_GABA,
                      net.W_inA_inh,
                      net.W_inV_inh):
                w *= 0.5
            net.g_GABA *= 0.5
            net.auto_calibrate_input_gain(target=excite_gain_target, max_iter=4)
            continue

        delta = target_ei / ei_ratio
        delta = min(max(delta, 0.5), 8.0)
        net.g_GABA *= delta
        net.W_msiInh2Exc_GABA *= delta
        net.W_inA_inh *= delta
        net.W_inV_inh *= delta

        if abs(ei_ratio - target_ei) <= ei_tol:
            print(f"✓ EI-calibration converged in {p} passes "
                  f"(ratio ≈ {ei_ratio:.2f})")
            return

    print("⚠ EI-calibration hit max_passes without full convergence")


def inverse_effectiveness_check(net,
                                intensities=(0.002, 0.0022, 0.0025, 0.0028, 0.003, 0.0032,
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
    """
    Convenience function that calls analyze_inverse_effectiveness with a set of intensities,
    and then plots the additivity index B/(A+V).
    """
    global IMG_SAVE_DIR
    instrument_network(net)
    res = analyze_inverse_effectiveness(net, list(intensities))
    # Quick display of mean EI ratio during the test:
    mean_ei = np.mean(net._ei_ratio_log) if len(net._ei_ratio_log) > 0 else 0
    print(f"InverseEffect: Mean EI ratio during run: {mean_ei:.2f}")

    ai_on = res['additivity_index_on']
    xvals = list(range(len(ai_on)))

    plt.figure(figsize=(5, 3))
    plt.plot(xvals, ai_on, 'o-', label='NMDA ON')
    plt.axhline(1, ls='--', c='k')
    plt.xlabel('Intensity index')
    plt.ylabel('B/(A+V)')
    plt.title("Inverse Effectiveness (NMDA ON)\n(Parameter set: see figure title)")
    plt.tight_layout()
    plt.show()
    # CHANGED: save into subfolder
    return res


def instrument_network(net):
    """Attach EI‑ratio & irregularity logging to *net* if not already patched."""
    if getattr(net, "_ei_patched", False):  # already instrumented
        return net

    # containers for logs
    net._ei_ratio_log = []  # one entry per macro time‑step
    net._pop_cv_log = []

    # keep original method
    original = net.update_all_layers_batch

    def wrapped(self, *args, **kwargs):  # NOTE: *self* now explicit
        """Proxy that calls the original update then records statistics."""
        out = original(*args, **kwargs)  # same signature

        # --------------------------------------------------------------
        # Compute excitation & inhibition totals (batch‑summed).
        # --------------------------------------------------------------
        I_exc = self.I_A + self.I_V + torch.clamp(self.I_M, min=0)  # (B,n)
        I_inh = torch.clamp(-self.I_M, min=0)  # (B,n)
        ei_ratio = (I_inh.sum() / (I_exc.sum() + 1e-6)).item()
        self._ei_ratio_log.append(ei_ratio)

        # --------------------------------------------------------------
        # Population irregularity — CV of spike counts across neurons.
        # --------------------------------------------------------------
        pop_counts = self._latest_sMSI.sum(1)  # (B,)
        mean = pop_counts.mean().item() + 1e-6
        std = pop_counts.std(unbiased=False).item()
        self._pop_cv_log.append(std / mean)

        return out

    # monkey‑patch
    net.update_all_layers_batch = wrapped.__get__(net, net.__class__)
    net._ei_patched = True
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


########################################################
#                MAIN TRAINING SCRIPT
########################################################

def run_training(
        batch_size=256,
        n_unsup_epochs=15,
        sequences_per_unsup_epoch=1000,
        n_readout_epochs=30,
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
        noise_std=0.02,
        single_modality_prob=0.5,
        v_thresh=0.25,
        dt=0.1,
        tau_m=20.0,
        n_substeps=100,
        loc_jitter_std=0,
        space_size=180,
        conduction_delay_a2msi=5,
        conduction_delay_v2msi=5,
        conduction_delay_msi2out=5
    )

    init_W_inA = net.W_inA.clone().cpu().numpy()
    init_W_inV = net.W_inV.clone().cpu().numpy()

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

    pretrain_autocalibrate(net)

    # net.gNMDA *= 20

    offsets = list(range(21))
    visualize_flash_sound_msi(net, offsets=offsets, T=50, D=10, loc=90)

    inverse_effectiveness_check(net)

    instrument_network(net)  # oneliner patch
    ei_balance_probe(net)

    # Single-Pulse Probe
    net.reset_state()
    with torch.no_grad():
        print("\n=== Single-Pulse Probe Test (multiple steps) ===")
        net.reset_state()

        x_test = torch.zeros((net.batch_size, net.n), device=net.device)
        idx = net.n // 2
        x_test[:, idx] = 2.0

        for step in range(3):
            sA, sV, sM, sO, vdict = net.update_all_layers_batch(
                x_test, x_test, record_voltages=True
            )

        dend_mean = net.v_dend_A.mean().item()
        soma_mean = net.v_msi.mean().item()
        print("Probe => dend V (A):", dend_mean,
              "soma V:", soma_mean,
              "MSI spike sum:", sM.sum().item())

    # Compute spike rates on random sequences
    rates0 = calculate_spike_rates_gpu(net, n_seq=50, stimulus_intensity=0.8, batch_size=batch_size)
    print(f"Pre-train spike rates  |  A:{rates0['audio']:.3f}  "
          f"V:{rates0['visual']:.3f}  MSI:{rates0['msi']:.3f}  "
          f"OUT:{rates0['output']:.3f}\n")
    # ----------------------------------------------------------
    # NEW: EI‑balance / irregularity probe
    # ----------------------------------------------------------
    with torch.no_grad():
        quick_ei_probe(net, n_macro_steps=100)
        cv_val = single_cell_cv(net, macro_steps=300)
        print(f"MSI neuron CV(ISI): {cv_val:.2f}")

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

    # Flash-Sound test
    offsets = list(range(21))
    visualize_flash_sound_msi(net, offsets=offsets, T=50, D=10, loc=90)


