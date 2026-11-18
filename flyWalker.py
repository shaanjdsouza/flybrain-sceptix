"""
fly_walker.py
-------------
Connectome slice (BANC) -> rate-based neural network -> 6-legged MuJoCo robot.

  smell (ORN input) -> AL -> KC/DAN -> MBON -> relay -> DN -> VNC premotor -> leg MNs
                                                                               |
                          leg MN activity  --modulates-->  tripod gait generator
                                                                               |
                                                                      MuJoCo hexapod

HOW THE NETWORK CONTROLS THE BODY (v1)
  * The connectome slice alone does not produce a walking rhythm, so a simple
    alternating-tripod oscillator provides the timing.
  * The network decides WHETHER and HOW HARD each leg moves: mean leg-MN
    activity sets the walking drive (frequency + amplitude); the MN activity
    of each leg (split in 3 chunks -> coxa / femur / tibia) scales that leg's
    joint amplitudes. No odor -> network quiet -> robot stands. Odor -> walks.
  * The MN -> joint assignment is a PLACEHOLDER (MNs are split into 3 chunks
    in ID order). Replace it with real muscle/target labels when you have them.

USAGE
  # 1) test the body + gait alone (no neurons):
  python fly_walker.py --cpg-only --duration 8

  # 2) full pipeline with your extracted subset:
  python fly_walker.py --subset banc_subset/test_subset.csv \
                       --edges banc_subset/test_edges.csv \
                       --neurons neurons.csv --duration 12

  # 3) watch it (macOS needs mjpython instead of python):
  mjpython fly_walker.py --subset ... --edges ... --neurons ... --viewer

Outputs: hexapod.xml (the model) and sim_log.csv (trajectory + drive over time).
"""
"""RUN: mjpython flyWalker.py --subset banc_subset/test_subset.csv --edges banc_subset/test_edges.csv --neurons neurons.csv --match-gains --stimulate DN 
        mjpython flyWalker.py --subset banc_subset/test_subset.csv --edges banc_subset/test_edges.csv --neurons neurons.csv --match-gains --stimulate MBON --viewer"""
import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import mujoco

LEGS = ["T1_L", "T1_R", "T2_L", "T2_R", "T3_L", "T3_R"]
JOINTS = ["coxa", "femur", "tibia"]
TRIPOD_A = {"T1_L", "T3_L", "T2_R"}          # these swing together; the other 3 are anti-phase get what iam telling bruh ;left front and backleg and right middle leg
LEG_X = {"T1": 0.09, "T2": 0.0, "T3": -0.09}  # attachment point along the body (front = +x) its in meters
LEG_Y = 0.05
# pathway order of signals  travel in feed-forward synapses (no feedback loops since its not learning. like u)
STAGE = {"ORN": 0, "AL_LN": 1, "AL_PN": 2, "KC": 3, "DAN": 3, "MBON": 4,
         "relay": 5, "DN": 6, "VNC_premotor": 7, "MN_leg": 8}


# ==========================================================================
# 1. MuJoCo hexapod model(for geeks, could be skipped) 
# ==========================================================================
def build_xml():
    legs_xml, act_xml = [], []
    for leg in LEGS:
        seg, side = leg.split("_")
        s = 1 if side == "L" else -1      # left legs extend toward +y, right toward -y
        x, y = LEG_X[seg], s * LEG_Y
        # axes chosen so that  +coxa = swing forward,  +femur = lift,  +tibia = extend outward
        legs_xml.append(f"""
      <body name="{leg}_coxa" pos="{x} {y} 0">
        <joint name="{leg}_coxa" type="hinge" axis="0 0 {-s}" range="-0.7 0.7" damping="0.05" armature="0.002"/>
        <geom type="capsule" fromto="0 0 0 0 {s*0.03:.3f} 0" size="0.008" mass="0.010"/>
        <body name="{leg}_femur" pos="0 {s*0.03:.3f} 0">
          <joint name="{leg}_femur" type="hinge" axis="{s} 0 0" range="-0.9 0.9" damping="0.05" armature="0.002"/>
          <geom type="capsule" fromto="0 0 0 0 {s*0.06:.3f} 0" size="0.008" mass="0.015"/>
          <body name="{leg}_tibia" pos="0 {s*0.06:.3f} 0">
            <joint name="{leg}_tibia" type="hinge" axis="{s} 0 0" range="-1.3 1.3" damping="0.05" armature="0.002"/>
            <geom type="capsule" fromto="0 0 0 0 {s*0.02:.3f} -0.09" size="0.007" mass="0.015"
                  friction="1.5 0.02 0.002"/>
          </body>
        </body>
      </body>""")
        for j, rng in zip(JOINTS, ["-0.7 0.7", "-0.9 0.9", "-1.3 1.3"]):
            act_xml.append(f'    <position name="{leg}_{j}" joint="{leg}_{j}" kp="6" kv="0.15" '
                           f'ctrlrange="{rng}" forcerange="-3 3"/>')
    return f"""<mujoco model="fly_hexapod">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <!-- legs/torso collide with the floor only (not with each other) -->
    <geom contype="1" conaffinity="0" condim="3" friction="1.0 0.02 0.002"/>
  </default>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.35 0.38 0.40 1" contype="1" conaffinity="1"/>
    <body name="torso" pos="0 0 0.115">
      <freejoint name="root"/>
      <geom name="torso" type="box" size="0.13 0.05 0.02" mass="0.30" rgba="0.8 0.6 0.2 1"/>
      <geom name="head" type="sphere" pos="0.15 0 0" size="0.03" mass="0.02" rgba="0.8 0.3 0.2 1"/>
{''.join(legs_xml)}
    </body>
  </worldbody>
  <actuator>
{chr(10).join(act_xml)}
  </actuator>
</mujoco>
"""

# ==========================================================================
# 2. Rate-based network built from the connectome slice
# ==========================================================================
def _norm_cols(df):       #removes alpha and other characters from the header and replaces with _ for uniformity and dot-attribute access XD
    df = df.copy()
    df.columns = [re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_") for c in df.columns]
    return df


def _nt_sign(nt): #in flybrain each neuron sends signals to other this done which chemicals, in this case gaba  and glut (glutamate) which means stop sending signals to next neuron but in maths we take -1 (not zero cz multiplcation with give 0 XD) and oppsite to this is acetylcholine ACh or other +1 means pass the signals
    """+1 excitatory (ACh / unknown), -1 inhibitory (GABA, glutamate)."""#this line is doc string its like comment (for those who dont know XD)
    nt = str(nt).lower()
    if "gaba" in nt or "glut" in nt:
        return -1.0
    return 1.0


class FlyNet:
    def __init__(self, subset_csv, edges_csv, neurons_csv=None, gain=1.5, tau=0.03, norm_floor=None, eta=0.0, bias=0.0, seed=0, verbose=True, all_exc=False, feedforward=True, stim="ORN"):
        #gain is to applifies the signal since i chose fraction of the brain, why : signals fades with each layer
        #tau is speed of signals 
        #bias baseline or idle activity
        #seed locked to zero for repeated experiment random stayes locked  

        sub = _norm_cols(pd.read_csv(subset_csv))
        id_col = "root_id" if "root_id" in sub.columns else sub.columns[0]
        edges = _norm_cols(pd.read_csv(edges_csv))

        self.ids = sub[id_col].to_numpy()
        self.n = len(self.ids)
        idx = {i: k for k, i in enumerate(self.ids)}
        self.groups = {g: np.array([idx[i] for i in d[id_col]]) for g, d in sub.groupby("group")}
        self.tau, self.eta, self.bias = tau, eta, bias
        self.stim = stim
        rng = np.random.default_rng(seed)

        # neurotransmitter sign of each presynaptic neuron
        sign = np.ones(self.n)
        if neurons_csv: 
            nm = _norm_cols(pd.read_csv(neurons_csv, low_memory=False))
            nid = "root_id" if "root_id" in nm.columns else nm.columns[0]
            nt = nm.get("verified_nt_type", pd.Series(index=nm.index, dtype=object)).copy()
            if "predicted_nt_type" in nm.columns:
                nt = nt.fillna(nm["predicted_nt_type"])
            lut = dict(zip(nm[nid], nt))
            sign = np.array([_nt_sign(lut.get(i, "")) for i in self.ids])
        else:
            print("(!) No --neurons file: treating every synapse as excitatory.")
        if all_exc:
            sign = np.ones(self.n)      # diagnostic: ignore GABA/glutamate inhibition
        self.sign = sign

        # raw synapse matrix  W[post, pre]
        e = edges[edges.pre.isin(idx) & edges.post.isin(idx)]
        pre = e.pre.map(idx).to_numpy()
        post = e.post.map(idx).to_numpy()
        raw = np.zeros((self.n, self.n))
        np.add.at(raw, (post, pre), e.syn.to_numpy(dtype=float))

        # keep only forward synapses along ORN -> ... -> MN. Feedback / same-stage loops make the
        # rescaled network self-excite (activity that no longer depends on the odor).
        n_before = int((raw > 0).sum())
        if feedforward:
            stg = np.zeros(self.n, dtype=int)
            for g, v in self.groups.items():
                stg[v] = STAGE.get(g, 0)
            raw = raw * (stg[:, None] > stg[None, :])
            if verbose:
                print(f"Feed-forward only: kept {int((raw > 0).sum())} of {n_before} connections "
                      f"(use --allow-recurrent to keep loops).")

        # dopamine neurons modulate (plasticity) rather than drive
        dan = self.groups.get("DAN", np.array([], dtype=int))
        raw_dan = np.zeros_like(raw)
        raw_dan[:, dan] = raw[:, dan]
        raw[:, dan] = 0.0
        self.Dn = raw_dan / np.maximum(raw_dan.sum(1, keepdims=True), 1.0)

        # normalise inputs per postsynaptic neuron (slice only holds a fraction of real inputs)
        in_sum = raw.sum(1)
        floor = norm_floor if norm_floor else (np.median(in_sum[in_sum > 0]) * 0.5 if (in_sum > 0).any() else 1.0)
        self.W = sign[None, :] * raw / np.maximum(in_sum, floor)[:, None] * gain

        # KC -> MBON weights are the plastic ones
        kc = self.groups.get("KC", np.array([], dtype=int))
        mbon = self.groups.get("MBON", np.array([], dtype=int))
        self.plastic = np.zeros_like(self.W, dtype=bool)
        if len(kc) and len(mbon):
            self.plastic[np.ix_(mbon, kc)] = self.W[np.ix_(mbon, kc)] > 0

        # odor input: each ORN has its own sensitivity; small spontaneous rate
        self.orn = self.groups.get(stim, np.array([], dtype=int))   # where the "odor" input is injected
        self.orn_sens = rng.uniform(0.3, 1.0, len(self.orn))
        self.dan = dan
        self.I = np.zeros(self.n)
        self.r = np.zeros(self.n)
        self.odor, self.punish = 0.0, 0.0

        # leg motor neurons, 3 chunks per leg (placeholder joint assignment)
        mn = sub.reset_index(drop=True)
        self.mn_all = self.groups.get("MN_leg", np.array([], dtype=int))
        self.leg_mn = {}
        for leg in LEGS:
            ii = mn.index[(mn.group == "MN_leg") & (mn.get("leg") == leg)].to_numpy()
            self.leg_mn[leg] = np.array_split(ii, 3) if len(ii) else [np.array([], dtype=int)] * 3
        if verbose:
            print(f"Network: {self.n} neurons, {int((self.W != 0).sum())} connections, "
                  f"{int((sign < 0).sum())} inhibitory | groups: "
                  + ", ".join(f"{g}={len(v)}" for g, v in self.groups.items()))

    def set_odor(self, level):
        self.odor = level

    def set_punish(self, level):
        self.punish = level

    def step(self, dt):
        ext = np.full(self.n, self.bias)      # tonic baseline drive (lets inhibition / weak paths act)
        if len(self.orn):
            ext[self.orn] = 0.05 + self.odor * self.orn_sens
        if len(self.dan):
            ext[self.dan] = self.bias + self.punish
        target = np.clip(self.W @ self.r + ext, 0.0, 1.0)
        self.r += dt / self.tau * (target - self.r)
        if self.eta > 0 and self.punish > 0:      # dopamine-gated KC->MBON depression
            dop = self.Dn @ self.r
            dW = -self.eta * dt * dop[:, None] * self.r[None, :] * self.W
            self.W[self.plastic] = np.maximum(0.0, self.W[self.plastic] + dW[self.plastic])

    def match_layer_gains(self, dt, target=0.4, max_scale=500.0):
        """Homeostatic gain matching: walk down the pathway and rescale each stage's incoming
        synapses (all of them, keeping their relative strengths and signs) until the stage
        reaches a moderate rate (~target) during odor. This fixes 'the slice holds only a small
        fraction of the real inputs, so every layer attenuates the signal'. It cannot create a
        path: a stage with no net-excitatory input from the stage before stays silent."""
        order = ["AL_PN", "AL_LN", "KC", "MBON", "relay", "DN", "VNC_premotor", "MN_leg"]
        scales, stuck = {}, []
        self.set_odor(1.0)
        for g in order:
            idx = self.groups.get(g)
            if idx is None or len(idx) == 0 or STAGE[g] <= STAGE[self.stim]:
                continue
            total = 1.0
            for _ in range(25):
                self.r[:] = 0
                for _ in range(int(0.4 / dt)):
                    self.step(dt)
                rate = float(self.r[idx].mean())
                if 0.8 * target <= rate <= 1.25 * target:
                    break
                f = np.clip(target / max(rate, 0.01), 0.25, 4.0)
                if total * f > max_scale:
                    break
                self.W[idx, :] *= f
                total *= f
            self.r[:] = 0
            for _ in range(int(0.4 / dt)):
                self.step(dt)
            if float(self.r[idx].mean()) < 0.5 * target:
                stuck.append(g)
            scales[g] = float(total)
        self.set_odor(0.0)
        self.r[:] = 0
        return scales, stuck

    def mn_rate(self):
        return float(self.r[self.mn_all].mean()) if len(self.mn_all) else 0.0

    def calibrate(self, dt):
        """Measure leg-MN activity without / with odor so drive can be scaled to 0..1."""
        self.r[:] = 0
        self.set_odor(0)
        for _ in range(int(1.0 / dt)):
            self.step(dt)
        base = self.mn_rate()
        r_base = self.r.copy()
        grp_base = {g: float(self.r[v].mean()) for g, v in self.groups.items()}
        self.set_odor(1.0)
        for _ in range(int(2.0 / dt)):
            self.step(dt)
        odor = self.mn_rate()
        grp_odor = {g: float(self.r[v].mean()) for g, v in self.groups.items()}
        self.set_odor(0)
        self.r[:] = r_base
        return base, odor, grp_base, grp_odor


# ==========================================================================
# 3. Gait generator driven by network activity
# ==========================================================================
class Gait:
    def __init__(self, freq_min=0.8, freq_max=2.2, a_coxa=0.40, a_femur=0.55, a_tibia=0.55):
        self.fmin, self.fmax = freq_min, freq_max
        self.amp = np.array([a_coxa, a_femur, a_tibia])
        self.phase = {l: (0.0 if l in TRIPOD_A else np.pi) for l in LEGS}

    def targets(self, dt, drive, gains):
        """gains: dict leg -> array(3) in 0..1. Returns dict (leg, joint_idx) -> target angle."""
        f = self.fmin + (self.fmax - self.fmin) * drive
        out = {}
        for leg in LEGS:
            self.phase[leg] = (self.phase[leg] + 2 * np.pi * f * dt) % (2 * np.pi)
            ph = self.phase[leg]
            g = gains[leg] * (drive > 0.02)
            swing = max(0.0, np.cos(ph))                     # >0 while the leg is in the air
            out[(leg, 0)] = g[0] * self.amp[0] * np.sin(ph)           # fore/aft swing
            out[(leg, 1)] = g[1] * self.amp[1] * swing                # lift
            out[(leg, 2)] = -g[2] * self.amp[2] * swing               # flex in during swing
        return out


# ==========================================================================
# 3b. Auto-tune the network so the odor actually reaches the leg motor neurons
# ==========================================================================
def autotune(args, dt):
    rows = []
    for floor in [1.0, 3.0, None]:
        for gain in [1.5, 3.0, 5.0, 8.0]:
            for bias in [0.0, 0.05, 0.1, 0.2]:
                net = FlyNet(args.subset, args.edges, args.neurons, gain=gain, tau=args.tau,
                             norm_floor=floor, bias=bias, verbose=False)
                base, odor, g_base, g_odor = net.calibrate(dt)
                span = odor - base
                # prefer a big MN response that is not saturated already at rest
                score = span - 0.5 * max(0.0, base - 0.3) - 0.5 * max(0.0, 0.05 - span)
                rows.append(dict(norm_floor=floor, gain=gain, bias=bias, mn_rest=base,
                                 mn_odor=odor, change=span, score=score))
    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    print("\nAuto-tune (top 5 of %d settings):" % len(df))
    print(df.head(5).round(3).to_string(index=False))
    best = df.iloc[0]
    args.norm_floor = None if pd.isna(best.norm_floor) else float(best.norm_floor)
    args.gain, args.bias = float(best.gain), float(best.bias)
    print(f"\nUsing norm_floor={args.norm_floor}, gain={args.gain}, bias={args.bias}")
    if best.change < args.min_span:
        print("(!) No setting gives a clear odor response at the motor neurons. The slice itself "
              "probably has no connected path from the odor input to the legs: re-extract with a larger "
              "--n-per-group (e.g. 60) and/or --min-syn 2.")


# ==========================================================================
# 4. Run
# ==========================================================================
def run(args):
    xml = build_xml()
    Path(args.xml).write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    act = {(leg, j): model.actuator(f"{leg}_{JOINTS[j]}").id for leg in LEGS for j in range(3)}

    net = None
    base = span = 1.0
    if not args.cpg_only and args.autotune:
        autotune(args, dt)
    if not args.cpg_only:
        net = FlyNet(args.subset, args.edges, args.neurons, gain=args.gain, tau=args.tau,
                     norm_floor=args.norm_floor, eta=args.eta, bias=args.bias,
                     all_exc=args.all_excitatory, feedforward=not args.allow_recurrent,
                     stim=args.stimulate)
        if args.match_gains:
            scales, stuck = net.match_layer_gains(dt)
            print("\nPer-stage synaptic gain applied (x):",
                  {g: round(v, 1) for g, v in scales.items()})
            if stuck:
                print(f"(!) These stages stay silent even after rescaling: {stuck}. That means they get no "
                      "net-excitatory input from the stage before them in this slice (try --all-excitatory "
                      "to test whether inhibition is the blocker).")
        base, odor_resp, g_base, g_odor = net.calibrate(dt)
        span = odor_resp - base
        order = ["ORN", "AL_PN", "AL_LN", "KC", "DAN", "MBON", "relay", "DN", "VNC_premotor", "MN_leg"]
        print("\nOdor response per stage (mean firing rate 0-1):")
        print(f"  {'group':14s}{'no odor':>9s}{'odor':>9s}{'change':>9s}")
        for g in order:
            if g in g_base:
                print(f"  {g:14s}{g_base[g]:9.3f}{g_odor[g]:9.3f}{g_odor[g] - g_base[g]:+9.3f}")
        if span < args.min_span:
            print(f"\n(!) The odor changes the leg motor neurons by only {span:+.3f} (need >= {args.min_span}). "
                  "The signal dies somewhere along the chain (see the table: the first stage whose "
                  "'change' collapses). The robot will NOT walk, because the walking drive comes from "
                  "this response. Try --norm-floor 1, a higher --gain (3), or --bias 0.1.")

    gait = Gait()
    log, torso0 = [], None
    n_steps = int(args.duration / dt)

    def one_step(k):
        t = data.time
        on = args.odor_start <= t < args.odor_end
        if net is not None:
            net.set_odor(1.0 if on else 0.0)
            net.set_punish(1.0 if args.punish_start <= t < args.punish_end else 0.0)
            net.step(dt)
            drive = float(np.clip((net.mn_rate() - base) / (span + 1e-9), 0, 1)) if span >= args.min_span else 0.0
            m = max(net.mn_rate(), 1e-6)
            gains = {}
            for leg in LEGS:
                rel = [np.clip(net.r[c].mean() / m, 0.6, 1.4) if len(c) else 1.0 for c in net.leg_mn[leg]]
                gains[leg] = np.clip(drive * np.array(rel), 0, 1)
        else:
            drive = 1.0 if on else 0.0
            gains = {leg: np.ones(3) for leg in LEGS}
        tg = gait.targets(dt, drive, gains)
        for key, a in act.items():
            data.ctrl[a] = tg[key]
        mujoco.mj_step(model, data)
        if k % 25 == 0:
            q = data.qpos
            log.append(dict(t=data.time, x=q[0], y=q[1], z=q[2], drive=drive, odor=float(on),
                            **{f"gain_{l}": float(gains[l].mean()) for l in LEGS}))

    # let the robot settle onto the ground before anything starts
    for _ in range(int(0.5 / dt)):
        mujoco.mj_step(model, data)
    data.time = 0.0
    torso0 = data.qpos[:3].copy()

    if args.viewer:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as v:
            for k in range(n_steps):
                if not v.is_running():
                    break
                t0 = time.time()
                one_step(k)
                if k % 5 == 0:
                    v.sync()
                    time.sleep(max(0, 5 * dt - (time.time() - t0)))
    else:
        for k in range(n_steps):
            one_step(k)

    df = pd.DataFrame(log)
    df.to_csv(args.out, index=False)

    def pos_at(tq):
        r = df.iloc[(df.t - tq).abs().argmin()]
        return np.array([r.x, r.y])

    p0, p1 = pos_at(args.odor_start), pos_at(min(args.odor_end, df.t.max()))
    d = p1 - p0
    print(f"\nDuring odor ({args.odor_start}-{args.odor_end}s): moved dx={d[0]:+.3f} m, dy={d[1]:+.3f} m "
          f"(distance {np.linalg.norm(d):.3f} m)")
    pre = pos_at(0.0), pos_at(args.odor_start)
    print(f"Before odor: moved {np.linalg.norm(pre[1] - pre[0]):.3f} m   | torso height now {data.qpos[2]:.3f} m")
    print(f"Log written to {args.out}   model written to {args.xml}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="banc_subset/test_subset.csv")
    ap.add_argument("--edges", default="banc_subset/test_edges.csv")
    ap.add_argument("--neurons", default=None, help="full neuron table (for neurotransmitter signs)")
    ap.add_argument("--cpg-only", action="store_true", help="skip the network; test body + gait only")
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--odor-start", type=float, default=2.0)
    ap.add_argument("--odor-end", type=float, default=10.0)
    ap.add_argument("--gain", type=float, default=1.5, help="synaptic gain of the rate network")
    ap.add_argument("--bias", type=float, default=0.0, help="tonic baseline drive to every neuron (spontaneous activity)")
    ap.add_argument("--norm-floor", type=float, default=None,
                    help="min total input used to normalise a neuron's synapses (lower = stronger weak-input neurons; try 1)")
    ap.add_argument("--min-span", type=float, default=0.05,
                    help="smallest odor response at the leg motor neurons that counts as a real signal")
    ap.add_argument("--tau", type=float, default=0.03, help="neuron time constant (s)")
    ap.add_argument("--eta", type=float, default=0.0, help="dopamine-gated KC->MBON plasticity rate (0 = off)")
    ap.add_argument("--punish-start", type=float, default=-1, help="DAN (punishment) pulse window start (s); needs --eta > 0")
    ap.add_argument("--punish-end", type=float, default=-1)
    ap.add_argument("--match-gains", action="store_true",
                    help="rescale each stage's synapses so the odor signal reaches every stage (recommended)")
    ap.add_argument("--stimulate", default="ORN", choices=["ORN", "KC", "MBON", "DN", "VNC_premotor"],
                    help="inject the 'odor' input at this stage instead of the ORNs, to test one part of "
                         "the chain on its own (e.g. DN = does the descending->VNC part drive the legs?)")
    ap.add_argument("--allow-recurrent", action="store_true",
                    help="keep feedback / same-stage synapses (default: feed-forward only)")
    ap.add_argument("--all-excitatory", action="store_true",
                    help="diagnostic: treat every synapse as excitatory (ignores GABA/glutamate signs)")
    ap.add_argument("--autotune", action="store_true",
                    help="search gain / norm-floor / bias for the setting that gives the strongest odor response")
    ap.add_argument("--viewer", action="store_true", help="open the interactive viewer (use mjpython on macOS)")
    ap.add_argument("--xml", default="hexapod.xml")
    ap.add_argument("--out", default="sim_log.csv")
    run(ap.parse_args())