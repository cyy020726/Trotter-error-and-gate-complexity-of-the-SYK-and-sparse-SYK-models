#!/usr/bin/env python3
"""
Trotter-error multiprocess GUI demo with:
- Bernoulli mask B_i (Pr[B_i=1]=p_B=n/G, where G=len(K_gamma))
- Updated Gaussian sigma: sqrt( (1/p_B) * ((k-1)! * J_E**2) / (k * n**(k-1)) )
- Efficient skipping of inactive local terms (B_i==0)
- Two progress bars per worker: Overall (3 ticks per repeat) and Step-2 (active multiplies)
- Wider window + larger results text area
- NEW: store active_count = len(active_idx) in DATABASE
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

PROC_COUNT = max(2, (os.cpu_count()-4 or 2))   # total processes >= 2, always leave 4 cores free
N_WORKERS  = PROC_COUNT - 1                  # 1 overseer + N_WORKERS workers

# Iterate n over even values 6..18 (inclusive)
N_START, N_END, N_STEP = 6, 18, 2
N_VALUES = [n for n in range(N_START, N_END + 1, N_STEP)]

# Physics / algorithm parameters
J_E = 1.0
k   = 6
t   = 10.0
l   = 2                 # 1 or 2 (Strang)
r   = 100000             # keep modest for demo; larger can be heavy
p   = 2                 # Schatten p; fast path for p==2

# Repeats per worker per iteration:
REPEATS_PER_WORKER = 10

# Optional sleeps to visualize progress (set to 0 for speed)
MIN_SLEEP, MAX_SLEEP = 0.0, 0.0

# Database file
os.makedirs("DATABASE", exist_ok=True)
DATABASE = f"DATABASE/DATA_SPARSE_n_sweep(k={k},t={t},l={l},r={r},p={p},J={round(J_E,3)}).txt"


# ------------------------- Math helpers -------------------------

# 2x2 Pauli as sparse csc (complex128)
X  = sp.csc_matrix([[0, 1],[1, 0]], dtype=np.complex128)
Y  = sp.csc_matrix([[0,-1j],[1j, 0]], dtype=np.complex128)
Z  = sp.csc_matrix([[1, 0],[0,-1 ]], dtype=np.complex128)
I2 = sp.identity(2, format="csc", dtype=np.complex128)

def kron_all(ops: List[sp.csc_matrix]) -> sp.csc_matrix:
    """Left-to-right Kronecker of csc ops."""
    out = ops[0]
    for op in ops[1:]:
        out = sp.kron(out, op, format="csc")
    out.sort_indices()
    return out

def maj_to_pauli(n: int) -> List[sp.csc_matrix]:
    """Jordan–Wigner: n Majoranas → m=n//2 qubits (csc list length n)."""
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
    """Serialize csc list to tuples so we can send via Queue."""
    packed = []
    for A in Ks:
        A = A.tocsc().astype(np.complex128, copy=False)
        A.sort_indices()
        packed.append((A.data, A.indices, A.indptr, A.shape))
    return packed

def unpack_csc_list(packed) -> List[sp.csc_matrix]:
    """Recreate csc list from packed tuples."""
    out = []
    for data, indices, indptr, shape in packed:
        out.append(sp.csc_matrix((data, indices, indptr), shape=shape))
    return out


# ------------------------- Worker-side Trotter (with progress) -------------------------

def S_with_progress(
    order_l: int,
    active_idx: np.ndarray,   # indices where B_i == 1
    J: np.ndarray,            # Gaussian samples (length G)
    K: List[sp.csc_matrix],
    t_step: float,
    outq: mp.Queue,
    wid: int,
    iter_id: int,
    rep_id: int,
) -> sp.csc_matrix:
    """
    Build one Trotter step S_l(t_step) using ONLY active terms (B_i==1).
    Step-2 bar advances exactly once per local exponential multiply that is executed.
    Overall bar is NOT updated here.
    """
    dim = K[0].shape[0]
    S = sp.identity(dim, format="csc", dtype=np.complex128)

    if order_l == 1:
        # forward over active indices
        for i in active_idx:
            E = expm(1j * t_step * J[i] * K[i])
            S = (S @ E).tocsc()
            outq.put(("progress_step2", wid, iter_id, rep_id, 1))
        return S

    elif order_l == 2:
        half = 0.5 * t_step
        # forward half over active indices
        for i in active_idx:
            E = expm(1j * half * J[i] * K[i])
            S = (S @ E).tocsc()
            outq.put(("progress_step2", wid, iter_id, rep_id, 1))
        # backward half over active indices in reverse order
        for i in active_idx[::-1]:
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
    Worker main loop:
      - Wait for ("iter", iter_id, payload) where payload packs:
          { "msg": str, "params": (n,k,t,l,r,p), "K_pack": packed csc list }
      - For rep_id in 0..REPEATS_PER_WORKER-1:
          1) sample Bernoulli mask B (Pr= n/G) AND Gaussian J with new sigma
          2) build S_l with per-multiply step-2 bar using ONLY active terms (B_i==1)
          3) exact U = expm(i t H) with H = sum_{i: B_i=1} (J_i K_i)
          4) Schatten p-norm of difference
          5) send ("done_rep", wid, iter_id, rep_id, msg, rand, norm, active_count)
    Overall bar ticks ONLY 3 times per repeat: after (J,B) sampling, after U, after norm.
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

        # Bernoulli prob p_B = n / G
        p_B = n / float(G)

        # Updated Gaussian sigma:
        sigma = math.sqrt((1.0 / p_B) * (math.factorial(k_ - 1) * (J_E ** 2)) / (k_ * (n ** (k_ - 1))))

        # Overall bar (per repeat) has exactly 3 ticks
        overall_total = 3

        for rep_id in range(REPEATS_PER_WORKER):
            outq.put(("rep_start", wid, iter_id, rep_id, REPEATS_PER_WORKER,
                      overall_total, 0, msg))

            # Task 1: sample Bernoulli mask B and Gaussian J
            rng = np.random.default_rng()
            J = rng.normal(0.0, sigma, size=G)               # Gaussian
            B = (rng.random(G) < p_B)                        # Bernoulli mask (boolean)
            active_idx = np.nonzero(B)[0].astype(np.int64)   # indices with B_i==1
            active_count = int(active_idx.size)              # <-- NEW: track size
            step2_total = active_count if l_ == 1 else 2 * active_count
            outq.put(("step2_start", wid, iter_id, rep_id, int(step2_total)))

            if MAX_SLEEP > 0: time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))
            outq.put(("progress_overall", wid, iter_id, rep_id, 1))

            # Task 2: build S_l using ONLY active terms
            if active_count == 0:
                dim = K_gamma[0].shape[0]
                S_l = sp.identity(dim, format="csc", dtype=np.complex128)
                outq.put(("step2_end", wid, iter_id, rep_id))
            else:
                S_l = S_with_progress(l_, active_idx, J, K_gamma, t_ / r_,
                                      outq, wid, iter_id, rep_id)
                outq.put(("step2_end", wid, iter_id, rep_id))

            # Raise to r
            S_r = S_l ** r

            # Task 3: exact U for active terms
            H = sp.csc_matrix(K_gamma[0].shape, dtype=np.complex128)
            for i in active_idx:
                H = H + (J[i] * K_gamma[i])
            U = expm(1j * t_ * H)
            if MAX_SLEEP > 0: time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))
            outq.put(("progress_overall", wid, iter_id, rep_id, 1))

            # Task 4: Schatten p-norm of difference
            diff = U - S_r
            if p_ == 2:
                norm_val = float(spnorm(diff, "fro"))
            else:
                norm_val = float(np.linalg.norm(diff.toarray(), ord="fro"))
            if MAX_SLEEP > 0: time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))
            outq.put(("progress_overall", wid, iter_id, rep_id, 1))

            # Report done for this repetition (include active_count)  <-- NEW
            outq.put(("done_rep", wid, iter_id, rep_id, msg, random.random(), norm_val, active_count))


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

        # Spawn workers
        self.procs: List[mp.Process] = []
        for wid in range(self.n_workers):
            p = self.ctx.Process(target=worker_loop, args=(wid, self.inboxes[wid], self.outq),
                                 name=f"worker-{wid}", daemon=False)
            p.start()
            self.procs.append(p)

        # GUI
        self.root = tk.Tk()
        self.root.title("Overseer – Trotter Error (Bernoulli + Repeats + Two-Bar Workers)")
        # Wider window + taller results area (so repetition text is fully visible)
        self.root.geometry("1400x950")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # State
        self.rows: Dict[int, WorkerRow] = {}
        self.cur_iter_index = -1      # index into N_VALUES
        self.done_rep_count = 0       # #done_rep received for current n
        self.active_rep: Dict[int, int] = {wid: -1 for wid in range(self.n_workers)}  # current rep per worker
        # Store results: (n, wid, rep) -> (msg, rand, norm, active_count)  <-- NEW shape
        self.results: Dict[Tuple[int,int,int], Tuple[str,float,float,int]] = {}

        self._build_gui()
        self._ensure_db_header()
        self._poll_outq()

        # Kick off first iteration shortly after GUI shows
        self.root.after(200, self._start_next_iteration)

    # ----- GUI layout -----
    def _build_gui(self) -> None:
        style = ttk.Style(self.root)
        for theme in ("clam", "vista", "xpnative", "alt", "default"):
            try: style.theme_use(theme); break
            except tk.TclError: pass

        # Header
        header = ttk.Frame(self.root, padding=10)
        header.pack(fill="x")
        ttk.Label(
            header,
            text=(f"Processes: {PROC_COUNT} (= 1 overseer + {self.n_workers} workers) "
                  f"| Repeats/worker: {REPEATS_PER_WORKER}"),
            font=("Segoe UI", 11, "bold")
        ).pack(anchor="w")

        # Overseer iteration bar
        it = ttk.Frame(self.root, padding=(10, 6))
        it.pack(fill="x")
        ttk.Label(it, text="Iterations:", width=12).grid(row=0, column=0, sticky="w")
        self.iter_bar = ttk.Progressbar(it, orient="horizontal", mode="determinate",
                                        length=1000, maximum=len(N_VALUES), value=0)  # wider bar
        self.iter_bar.grid(row=0, column=1, padx=8, sticky="we")
        self.iter_val = ttk.Label(it, text=f"0 / {len(N_VALUES)}", width=14, anchor="e")
        self.iter_val.grid(row=0, column=2, sticky="e")

        # Workers container
        cont = ttk.Frame(self.root, padding=(10, 6))
        cont.pack(fill="x")
        for wid in range(self.n_workers):
            row = ttk.Frame(cont, padding=6)
            row.pack(fill="x", pady=4)

            title_lbl = ttk.Label(row, text=f"Worker {wid}", width=24)
            title_lbl.grid(row=0, column=0, sticky="w")

            # Wider repetition label next to worker ID (balanced parentheses)
            rep_lbl = ttk.Label(row, text="(rep –/–)", width=18, foreground="#555")
            rep_lbl.grid(row=0, column=1, sticky="w")

            ttk.Label(row, text="Overall:", width=10).grid(row=0, column=2, sticky="e")
            overall_bar = ttk.Progressbar(row, orient="horizontal", mode="determinate",
                                          length=900, value=0)  # wider bar
            overall_bar.grid(row=0, column=3, padx=6, sticky="we")
            overall_val = ttk.Label(row, text="0 / ?", width=16, anchor="e")
            overall_val.grid(row=0, column=4, sticky="e")

            ttk.Label(row, text="Step 2:", width=10).grid(row=1, column=2, sticky="e")
            step2_bar = ttk.Progressbar(row, orient="horizontal", mode="determinate",
                                        length=900, value=0)  # wider bar
            step2_bar.grid(row=1, column=3, padx=6, sticky="we")
            step2_val = ttk.Label(row, text="0 / ?", width=16, anchor="e")
            step2_val.grid(row=1, column=4, sticky="e")

            self.rows[wid] = WorkerRow(row, title_lbl, rep_lbl, overall_bar, overall_val, step2_bar, step2_val)

        ttk.Separator(self.root).pack(fill="x", pady=(6, 4))

        # Larger results panel (taller text box)
        self.results_box = tk.Text(self.root, height=18, wrap="word")
        self.results_box.pack(fill="both", expand=True, padx=10, pady=(0, 6))
        self.results_box.insert("end", "Per-iteration results will appear here (with closed parentheses).\n")
        self.results_box.config(state="disabled")

        btns = ttk.Frame(self.root, padding=(10, 6))
        btns.pack(fill="x")
        self.quit_btn = ttk.Button(btns, text="Quit", command=self.on_close, state="disabled")
        self.quit_btn.pack(side="right")

    # ----- DB -----
    def _ensure_db_header(self) -> None:
        if not os.path.isfile(DATABASE):
            with open(DATABASE, "w") as f:
                # NEW column 'active_count'
                f.write("n,k,t,l,r,p,rep,norm,active_count,timestamp,worker\n")

    def _append_db_line(self, n: int, rep: int, norm: float, active_count: int, wid: int) -> None:
        now = datetime.datetime.now().isoformat(timespec="seconds")
        with open(DATABASE, "a") as f:
            f.write(f"{n},{k},{t},{l},{r},{p},{rep},{norm},{active_count},{now},{wid}\n")

    # ----- Iteration driver -----
    def _start_next_iteration(self) -> None:
        """Build K for next n and broadcast to workers."""
        self.cur_iter_index += 1
        if self.cur_iter_index >= len(N_VALUES):
            self.quit_btn["state"] = "normal"
            return

        n = N_VALUES[self.cur_iter_index]
        # Build K_gamma once for this n
        K_gamma = generate_K(n, k)
        K_pack = pack_csc_list(K_gamma)
        msg = f"K_broadcast(n={n}, k={k})"

        # Reset repetition state and texts
        self.done_rep_count = 0
        for wid, row in self.rows.items():
            row.title_lbl.configure(text=f"Worker {wid} (iter {self.cur_iter_index} | n={n})")
            row.rep_lbl.configure(text=f"(rep 0/{REPEATS_PER_WORKER})")
            for bar, lab in [(row.overall_bar, row.overall_val), (row.step2_bar, row.step2_val)]:
                bar["value"] = 0
                lab.configure(text="0 / ?")
            self.active_rep[wid] = -1

        # Broadcast to all workers
        payload = {"msg": msg, "params": (n, k, t, l, r, p), "K_pack": K_pack}
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
            # Set bars and repetition badge for this worker/repetition
            _, wid, iter_id, rep_id, total_reps, overall_total, step2_total, echoed_msg = msg
            if iter_id != self.cur_iter_index: return
            self.active_rep[wid] = rep_id
            row = self.rows[wid]
            row.rep_lbl.configure(text=f"(rep {rep_id+1}/{total_reps})")
            row.overall_bar["maximum"] = int(overall_total)
            row.overall_bar["value"] = 0
            row.overall_val.configure(text=f"0 / {int(overall_total)}")
            # step2_total will be set by step2_start for this rep
            row.step2_bar["maximum"] = int(step2_total) if step2_total else 1
            row.step2_bar["value"] = 0
            row.step2_val.configure(text=f"0 / {int(step2_total) if step2_total else 1}")

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
            row.step2_bar["maximum"] = max(1, int(step2_total))
            row.step2_bar["value"] = 0
            row.step2_val.configure(text=f"0 / {int(row.step2_bar['maximum'])}")

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
            # Store result for this (n, wid, rep), log it, and advance counters
            # NEW tuple includes active_count at the end
            _, wid, iter_id, rep_id, echoed_msg, randval, normval, active_count = msg
            if iter_id != self.cur_iter_index: return
            n = N_VALUES[self.cur_iter_index]
            self.results[(n, wid, rep_id)] = (str(echoed_msg), float(randval), float(normval), int(active_count))
            self._append_db_line(n, rep_id, float(normval), int(active_count), wid)

            self.done_rep_count += 1
            if rep_id + 1 == REPEATS_PER_WORKER:
                row = self.rows[wid]
                row.rep_lbl.configure(text=f"(rep {REPEATS_PER_WORKER}/{REPEATS_PER_WORKER})")

            if self.done_rep_count == self.n_workers * REPEATS_PER_WORKER:
                new_iter = self.iter_bar["value"] + 1
                self.iter_bar["value"] = new_iter
                self.iter_val.configure(text=f"{int(new_iter)} / {len(N_VALUES)}")
                self._append_iteration_summary(n)
                self._start_next_iteration()

    def _append_iteration_summary(self, n: int) -> None:
        """Pretty-print all (rep) results for this n."""
        lines = [f"n = {n} complete (rep results per worker):"]
        for wid in range(self.n_workers):
            for rep_id in range(REPEATS_PER_WORKER):
                msg, rnum, norm, act = self.results.get(
                    (n, wid, rep_id), ("<missing>", float("nan"), float("nan"), -1)
                )
                lines.append(
                    f"  worker {wid} (rep {rep_id+1}/{REPEATS_PER_WORKER}): "
                    f"msg='{msg}', rand={rnum:.6f}, norm={norm:.6e}, active_count={act}"
                )
        lines.append("")
        self.results_box.config(state="normal")
        self.results_box.insert("end", "\n".join(lines) + "\n")
        self.results_box.see("end")
        self.results_box.config(state="disabled")
        if self.cur_iter_index + 1 >= len(N_VALUES):
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
    print(f"n values: {N_VALUES} | repeats/worker: {REPEATS_PER_WORKER}")
    print(f"params: k={k}, t={t}, l={l}, r={r}, p={p}")
    if not os.path.isfile(DATABASE):
        print(f"Database created: {DATABASE}")
    else:
        print(f"Appending to: {DATABASE}")

    gui = OverseerGUI(n_workers=N_WORKERS)
    gui.root.mainloop()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
