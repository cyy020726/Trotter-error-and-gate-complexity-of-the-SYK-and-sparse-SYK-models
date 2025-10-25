#!/usr/bin/env python3
"""
Trotter-error multiprocess GUI demo — sweep over time t (n fixed)
- Repeats per worker
- Two progress bars per worker (Overall + Step-2)
- Overall bar ticks 3 times per repeat: [sample J] + [exact U] + [norm]
- t sweeps from 0 to 10 via linspace with 1000 points
"""

import os
import time
import math
import random
import queue
import datetime
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm, norm as spnorm

import tkinter as tk
from tkinter import ttk

# ------------------------- Configuration -------------------------

PROC_COUNT = max(2, (os.cpu_count() - 4 or 2))  # leave ~4 cores free
N_WORKERS = PROC_COUNT - 1                      # 1 overseer + N_WORKERS workers

# FIXED n (choose an even n; dim = 2^(n//2))
N_FIXED = 10

# Sweep t over [0, 10], 1000 points
T_START, T_END, T_POINTS = 0.0, 1000.0, 100
T_VALUES = np.linspace(T_START, T_END, T_POINTS).tolist()

# Physics / algorithm parameters
J_E = 1.0
k = 2
l = 1                 # 1 or 2 (Strang)
r = 100000            # WARNING: large r is heavy
p = 2                 # Schatten p; fast path for p==2

# Repeats per worker per t:
REPEATS_PER_WORKER = 4

# Optional sleeps to visualize progress (set to 0 for speed)
MIN_SLEEP, MAX_SLEEP = 0.0, 0.0

# Database file
os.makedirs("DATABASE", exist_ok=True)
DATABASE = f"DATABASE/DATA_DENSE_t_sweep(n={N_FIXED},k={k},l={l},r={r},J={round(J_E,3)}).txt"


# ------------------------- Math helpers -------------------------

# 2x2 Pauli as sparse csc (complex128)
X = sp.csc_matrix([[0, 1], [1, 0]], dtype=np.complex128)
Y = sp.csc_matrix([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = sp.csc_matrix([[1, 0], [0, -1]], dtype=np.complex128)
I2 = sp.identity(2, format="csc", dtype=np.complex128)

def kron_all(ops: List[sp.csc_matrix]) -> sp.csc_matrix:
    out = ops[0]
    for op in ops[1:]:
        out = sp.kron(out, op, format="csc")
    out.sort_indices()
    return out

def maj_to_pauli(n: int) -> List[sp.csc_matrix]:
    out, m = [], n // 2
    for i in range(1, m + 1):
        odd  = [Z]*(i-1) + [X] + [I2]*(m-i)
        even = [Z]*(i-1) + [Y] + [I2]*(m-i)
        out.append((1/np.sqrt(2)) * kron_all(odd))
        out.append((1/np.sqrt(2)) * kron_all(even))
    return out

def generate_K(n: int, k_: int) -> List[sp.csc_matrix]:
    # generate the deterministic part of the local terms
    from itertools import combinations
    P = maj_to_pauli(n)
    Ks: List[sp.csc_matrix] = []
    for subset in combinations(range(n), k_):
        M = P[subset[0]]
        for idx in subset[1:]:
            M = (M @ P[idx]).tocsc()
        # Make the Hamiltonian Hermitian
        # -1 comes from absorbing the minus sign in time evolution
        M = -1 * (1j)**(k * (k-1) / 2) * M 
        M.sort_indices()
        Ks.append(M)
    return Ks


def pack_csc_list(Ks: List[sp.csc_matrix]):
    packed = []
    for A in Ks:
        A = A.tocsc().astype(np.complex128, copy=False)
        A.sort_indices()
        packed.append((A.data, A.indices, A.indptr, A.shape))
    return packed

def unpack_csc_list(packed) -> List[sp.csc_matrix]:
    out = []
    for data, indices, indptr, shape in packed:
        out.append(sp.csc_matrix((data, indices, indptr), shape=shape))
    return out


# ------------------------- Worker-side Trotter (with progress) -------------------------

def S_with_progress(
    order_l: int,
    J: np.ndarray,
    K: List[sp.csc_matrix],
    t_step: float,
    outq: mp.Queue,
    wid: int,
    iter_id: int,
    rep_id: int,
) -> sp.csc_matrix:
    """
    Build one Trotter step S_l(t_step); emit Step-2 progress after each local exponential multiply.
    IMPORTANT: We do NOT update the Overall bar here.
    """
    G = len(K)
    dim = K[0].shape[0]
    S = sp.identity(dim, format="csc", dtype=np.complex128)

    if order_l == 1:
        for i in range(G):
            E = expm(1j * t_step * J[i] * K[i])
            S = (S @ E).tocsc()
            outq.put(("progress_step2", wid, iter_id, rep_id, 1))
        return S
    elif order_l == 2:
        half = 0.5 * t_step
        for i in range(G):
            E = expm(1j * half * J[i] * K[i])
            S = (S @ E).tocsc()
            outq.put(("progress_step2", wid, iter_id, rep_id, 1))
        for i in range(G - 1, -1, -1):
            E = expm(1j * half * J[i] * K[i])
            S = (S @ E).tocsc()
            outq.put(("progress_step2", wid, iter_id, rep_id, 1))
        return S
    else:
        raise ValueError("order l must be 1 or 2")


def worker_loop(
    wid: int,
    inbox: mp.Queue,      # overseer -> this worker
    outq: mp.Queue        # worker -> overseer
) -> None:
    """
    For each ("iter", iter_id, payload) with fixed n and varying t:
      Repeat rep_id = 0..REPEATS_PER_WORKER-1:
        1) sample Gaussian J (Overall +1)
        2) build S_l with per-multiply Step-2 progress (Overall unchanged here)
        3) exact U = expm(i t H) (Overall +1)
        4) Schatten p-norm of difference (Overall +1)
        5) send ("done_rep", wid, iter_id, rep_id, msg, rand, norm)
    """
    outq.put(("hello", wid))
    while True:
        cmd = inbox.get()
        if cmd is None:
            break
        kind, iter_id, payload = cmd
        assert kind == "iter", f"Worker {wid} got unexpected cmd: {cmd!r}"

        msg = payload["msg"]
        n, k_, t_, l_, r_, p_ = payload["params"]
        K_gamma = unpack_csc_list(payload["K_pack"])
        G = len(K_gamma)

        # Gaussian sigma
        sigma = math.sqrt((math.factorial(k_ - 1) * (J_E ** 2)) / (k_ * (n ** (k_ - 1))))

        # Overall per repeat: 3 ticks (J, U, norm)
        overall_total = 3

        for rep_id in range(REPEATS_PER_WORKER):
            outq.put(("rep_start", wid, iter_id, rep_id, REPEATS_PER_WORKER,
                      overall_total, (G if l_ == 1 else 2*G), msg))

            # 1) sample J (Overall +1)
            rng = np.random.default_rng()
            J = rng.normal(0.0, sigma, size=G)
            if MAX_SLEEP > 0: time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))
            outq.put(("progress_overall", wid, iter_id, rep_id, 1))

            # 2) Trotter S_l with Step-2 progress only
            outq.put(("step2_start", wid, iter_id, rep_id, (G if l_ == 1 else 2*G)))
            S_l = S_with_progress(l_, J, K_gamma, t_ / r_, outq, wid, iter_id, rep_id)
            outq.put(("step2_end", wid, iter_id, rep_id))

            # raise to r
            S_r = S_l ** r_

            # 3) exact U = expm(i t H) (Overall +1)
            H = sp.csc_matrix(K_gamma[0].shape, dtype=np.complex128)
            for j, Ki in zip(J, K_gamma):
                H = H + (j * Ki)
            U = expm(1j * t_ * H)
            if MAX_SLEEP > 0: time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))
            outq.put(("progress_overall", wid, iter_id, rep_id, 1))

            # 4) Schatten p-norm of difference (Overall +1)
            diff = U - S_r
            if p_ == 2:
                norm_val = float(spnorm(diff, "fro"))
            else:
                norm_val = float(np.linalg.norm(diff.toarray(), ord="fro"))

            if MAX_SLEEP > 0: time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))
            outq.put(("progress_overall", wid, iter_id, rep_id, 1))

            outq.put(("done_rep", wid, iter_id, rep_id, msg, random.random(), norm_val))


# ------------------------- GUI / Overseer -------------------------

@dataclass
class WorkerRow:
    frame: tk.Frame
    title_lbl: ttk.Label
    rep_lbl: ttk.Label
    overall_bar: ttk.Progressbar
    overall_val: ttk.Label
    step2_bar: ttk.Progressbar
    step2_val: ttk.Label

class OverseerGUI:
    def __init__(self, n_workers: int) -> None:
        self.n_workers = n_workers
        self.ctx = mp.get_context("spawn")

        # Queues
        self.outq: mp.Queue = self.ctx.Queue()                # workers -> overseer
        self.inboxes: List[mp.Queue] = [self.ctx.Queue() for _ in range(self.n_workers)]  # overseer -> each worker

        # Precompute K once (n is fixed)
        K_gamma = generate_K(N_FIXED, k)
        self.K_pack = pack_csc_list(K_gamma)

        # Spawn workers
        self.procs: List[mp.Process] = []
        for wid in range(self.n_workers):
            p = self.ctx.Process(target=worker_loop, args=(wid, self.inboxes[wid], self.outq),
                                 name=f"worker-{wid}", daemon=False)
            p.start()
            self.procs.append(p)

        # GUI
        self.root = tk.Tk()
        self.root.title("Overseer – Trotter Error (t sweep, Repeats + Two-Bar Workers)")
        self.root.geometry("1200x900")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # State
        self.rows: Dict[int, WorkerRow] = {}
        self.cur_iter_index = -1  # index into T_VALUES
        self.done_rep_count = 0
        self.active_rep: Dict[int, int] = {wid: -1 for wid in range(self.n_workers)}
        # store by (iter_idx, wid, rep) -> tuple
        self.results: Dict[Tuple[int,int,int], Tuple[str,float,float]] = {}

        self._build_gui()
        self._ensure_db_header()
        self._poll_outq()

        self.root.after(200, self._start_next_iteration)

    # ----- GUI layout -----
    def _build_gui(self) -> None:
        style = ttk.Style(self.root)
        for theme in ("clam", "vista", "xpnative", "alt", "default"):
            try: style.theme_use(theme); break
            except tk.TclError: pass

        header = ttk.Frame(self.root, padding=10)
        header.pack(fill="x")
        ttk.Label(
            header,
            text=(f"Processes: {PROC_COUNT} (= 1 overseer + {self.n_workers} workers) "
                  f"| Repeats/worker: {REPEATS_PER_WORKER} | n fixed: {N_FIXED}"),
            font=("Segoe UI", 11, "bold")
        ).pack(anchor="w")

        it = ttk.Frame(self.root, padding=(10, 6))
        it.pack(fill="x")
        ttk.Label(it, text="t points:", width=12).grid(row=0, column=0, sticky="w")
        self.iter_bar = ttk.Progressbar(it, orient="horizontal", mode="determinate",
                                        length=800, maximum=len(T_VALUES), value=0)
        self.iter_bar.grid(row=0, column=1, padx=8, sticky="we")
        self.iter_val = ttk.Label(it, text=f"0 / {len(T_VALUES)}", width=14, anchor="e")
        self.iter_val.grid(row=0, column=2, sticky="e")

        cont = ttk.Frame(self.root, padding=(10, 6))
        cont.pack(fill="x")
        for wid in range(self.n_workers):
            row = ttk.Frame(cont, padding=6)
            row.pack(fill="x", pady=4)

            title_lbl = ttk.Label(row, text=f"Worker {wid}", width=28)
            title_lbl.grid(row=0, column=0, sticky="w")

            rep_lbl = ttk.Label(row, text="(rep –/–)", width=18, foreground="#555")
            rep_lbl.grid(row=0, column=1, sticky="w")

            ttk.Label(row, text="Overall:", width=10).grid(row=0, column=2, sticky="e")
            overall_bar = ttk.Progressbar(row, orient="horizontal", mode="determinate",
                                          length=700, value=0)
            overall_bar.grid(row=0, column=3, padx=6, sticky="we")
            overall_val = ttk.Label(row, text="0 / ?", width=16, anchor="e")
            overall_val.grid(row=0, column=4, sticky="e")

            ttk.Label(row, text="Step 2:", width=10).grid(row=1, column=2, sticky="e")
            step2_bar = ttk.Progressbar(row, orient="horizontal", mode="determinate",
                                        length=700, value=0)
            step2_bar.grid(row=1, column=3, padx=6, sticky="we")
            step2_val = ttk.Label(row, text="0 / ?", width=16, anchor="e")
            step2_val.grid(row=1, column=4, sticky="e")

            self.rows[wid] = WorkerRow(row, title_lbl, rep_lbl, overall_bar, overall_val, step2_bar, step2_val)

        ttk.Separator(self.root).pack(fill="x", pady=(6, 4))

        # Results panel
        self.results_box = tk.Text(self.root, height=18, wrap="word")
        self.results_box.pack(fill="both", expand=True, padx=10, pady=(0, 6))
        self.results_box.insert("end", "Per-t results will appear here (with closed parentheses).\n")
        self.results_box.config(state="disabled")

        btns = ttk.Frame(self.root, padding=(10, 6))
        btns.pack(fill="x")
        self.quit_btn = ttk.Button(btns, text="Quit", command=self.on_close, state="disabled")
        self.quit_btn.pack(side="right")

    # ----- DB -----
    def _ensure_db_header(self) -> None:
        if not os.path.isfile(DATABASE):
            with open(DATABASE, "w") as f:
                f.write("n,k,t,l,r,p,rep,norm,timestamp,worker\n")

    def _append_db_line(self, n: int, t_cur: float, rep: int, norm: float, wid: int) -> None:
        now = datetime.datetime.now().isoformat(timespec="seconds")
        with open(DATABASE, "a") as f:
            f.write(f"{n},{k},{t_cur},{l},{r},{p},{rep},{norm},{now},{wid}\n")

    # ----- Iteration driver -----
    def _start_next_iteration(self) -> None:
        """Broadcast fixed-n workload with the next t to workers."""
        self.cur_iter_index += 1
        if self.cur_iter_index >= len(T_VALUES):
            self.quit_btn["state"] = "normal"
            return

        t_cur = float(T_VALUES[self.cur_iter_index])
        msg = f"K_broadcast(n={N_FIXED}, k={k}, t={t_cur:.6g})"

        self.done_rep_count = 0
        for wid, row in self.rows.items():
            row.title_lbl.configure(text=f"Worker {wid} (idx {self.cur_iter_index} | t={t_cur:.6g})")
            row.rep_lbl.configure(text=f"(rep 0/{REPEATS_PER_WORKER})")
            for bar, lab in [(row.overall_bar, row.overall_val), (row.step2_bar, row.step2_val)]:
                bar["value"] = 0
                lab.configure(text="0 / ?")
            self.active_rep[wid] = -1

        # Payload now varies only in 't'
        payload = {"msg": msg, "params": (N_FIXED, k, t_cur, l, r, p), "K_pack": self.K_pack}
        for inbox in self.inboxes:
            inbox.put(("iter", self.cur_iter_index, payload))

    # ----- Queue polling -----
    def _poll_outq(self) -> None:
        try:
            while True:
                msg = self.outq.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_outq)

    def _handle_message(self, msg: Tuple) -> None:
        kind = msg[0]

        if kind == "rep_start":
            _, wid, iter_id, rep_id, total_reps, overall_total, step2_total, echoed_msg = msg
            if iter_id != self.cur_iter_index: return
            self.active_rep[wid] = rep_id
            row = self.rows[wid]
            row.rep_lbl.configure(text=f"(rep {rep_id+1}/{total_reps})")
            row.overall_bar["maximum"] = int(overall_total)
            row.overall_bar["value"] = 0
            row.overall_val.configure(text=f"0 / {int(overall_total)}")
            row.step2_bar["maximum"] = int(step2_total)
            row.step2_bar["value"] = 0
            row.step2_val.configure(text=f"0 / {int(step2_total)}")

        elif kind == "progress_overall":
            _, wid, iter_id, rep_id, inc = msg
            if iter_id != self.cur_iter_index or rep_id != self.active_rep.get(wid, -2): return
            row = self.rows[wid]
            newv = min(row.overall_bar["value"] + inc, row.overall_bar["maximum"])
            row.overall_bar["value"] = newv
            row.overall_val.configure(text=f"{int(newv)} / {int(row.overall_bar['maximum'])}")

        elif kind == "step2_start":
            _, wid, iter_id, rep_id, step2_total = msg
            if iter_id != self.cur_iter_index or rep_id != self.active_rep.get(wid, -2): return
            row = self.rows[wid]
            row.step2_bar["maximum"] = int(step2_total)
            row.step2_bar["value"] = 0
            row.step2_val.configure(text=f"0 / {int(step2_total)}")

        elif kind == "progress_step2":
            _, wid, iter_id, rep_id, inc = msg
            if iter_id != self.cur_iter_index or rep_id != self.active_rep.get(wid, -2): return
            row = self.rows[wid]
            newv = min(row.step2_bar["value"] + inc, row.step2_bar["maximum"])
            row.step2_bar["value"] = newv
            row.step2_val.configure(text=f"{int(newv)} / {int(row.step2_bar['maximum'])}")

        elif kind == "step2_end":
            _, wid, iter_id, rep_id = msg
            if iter_id != self.cur_iter_index or rep_id != self.active_rep.get(wid, -2): return
            row = self.rows[wid]
            row.step2_bar["value"] = row.step2_bar["maximum"]
            row.step2_val.configure(text=f"{int(row.step2_bar['maximum'])} / {int(row.step2_bar['maximum'])}")

        elif kind == "done_rep":
            _, wid, iter_id, rep_id, echoed_msg, randval, normval = msg
            if iter_id != self.cur_iter_index: return
            self.results[(iter_id, wid, rep_id)] = (str(echoed_msg), float(randval), float(normval))

            # Append to DB with the actual t for this iteration
            t_cur = float(T_VALUES[self.cur_iter_index])
            self._append_db_line(N_FIXED, t_cur, rep_id, float(normval), wid)

            self.done_rep_count += 1
            if self.done_rep_count == self.n_workers * REPEATS_PER_WORKER:
                new_iter = self.iter_bar["value"] + 1
                self.iter_bar["value"] = new_iter
                self.iter_val.configure(text=f"{int(new_iter)} / {len(T_VALUES)}")
                self._append_iteration_summary(self.cur_iter_index)
                self._start_next_iteration()

    def _append_iteration_summary(self, iter_idx: int) -> None:
        t_cur = float(T_VALUES[iter_idx])
        lines = [f"t = {t_cur:.6g} complete (rep results per worker):"]
        for wid in range(self.n_workers):
            for rep_id in range(REPEATS_PER_WORKER):
                msg, rnum, norm = self.results.get((iter_idx, wid, rep_id), ("<missing>", float("nan"), float("nan")))
                lines.append(f"  worker {wid} (rep {rep_id+1}/{REPEATS_PER_WORKER}): "
                             f"msg='{msg}', rand={rnum:.6f}, norm={norm:.6e}")
        lines.append("")
        self.results_box.config(state="normal")
        self.results_box.insert("end", "\n".join(lines) + "\n")
        self.results_box.see("end")
        self.results_box.config(state="disabled")
        if iter_idx + 1 >= len(T_VALUES):
            self.quit_btn["state"] = "normal"

    # ----- Shutdown -----
    def on_close(self) -> None:
        try:
            deadline = time.time() + 1.5
            for p in self.procs:
                if p.is_alive():
                    p.join(timeout=max(0.0, deadline - time.time()))
            for p in self.procs:
                if p.is_alive():
                    p.terminate()
            for p in self.procs:
                p.join(timeout=0.5)
        finally:
            self.root.destroy()


# ------------------------- Entrypoint -------------------------

def main() -> None:
    print(f"PROC_COUNT={PROC_COUNT} → 1 overseer + {N_WORKERS} workers")
    print(f"n fixed: {N_FIXED} | t points: {len(T_VALUES)} | repeats/worker: {REPEATS_PER_WORKER}")
    print(f"params: k={k}, l={l}, r={r}, p={p}")
    if not os.path.isfile(DATABASE):
        print(f"Database created: {DATABASE}")
    else:
        print(f"Appending to: {DATABASE}")

    gui = OverseerGUI(n_workers=N_WORKERS)
    gui.root.mainloop()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
