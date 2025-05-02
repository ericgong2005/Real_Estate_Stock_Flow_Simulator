import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from collections import deque

# Model Parameters with default values
q1, q2, q3, q4 = 1.2, 0.8, 0.5, 1.0
house_cost = 5.0
max_rent = 10.0

convergence_value = 0.001
max_cycles = 200
delay = 0.1

# update linear relation endpoints
def update_ends():
    global Q1_ends, Q2_ends, Q3_ends, Q4_ends
    Q1_ends = [np.array([0.0, max_rent]), np.array([max_rent / q1, 0.0])]
    Q2_ends = [np.array([0.0, 0.0])]
    Q3_ends = [np.array([-house_cost, 0.0])]
    Q4_ends = [np.array([0.0, 0.0])]

update_ends()

# lines for the linear relations between variables
def y_Q1(x): return max_rent - q1 * x
def y_Q2(x): return -q2 * x
def y_Q3(x): return  q3 * (x + house_cost)
def y_Q4(x): return -q4 * x

def snap(pt, ends):
    d = [np.linalg.norm(pt - e) for e in ends]
    return ends[int(np.argmin(d))]

def go_up(pt):
    x,y = pt; yi = y_Q1(x)
    return np.array([x, yi]) if yi >= 0 else snap(pt, Q1_ends)

def go_left(pt):
    x,y = pt; xi = -y / q2
    return np.array([xi, y]) if y >= 0 else snap(pt, Q2_ends)

def go_down(pt):
    x,y = pt; yi = y_Q3(x)
    return np.array([x, yi]) if x <= -house_cost else snap(pt, Q3_ends)

def go_right(pt):
    x,y = pt; xi = -y / q4
    return np.array([xi, y]) if y <= 0 else snap(pt, Q4_ends)

steps = [go_up, go_left, go_down, go_right]

# Initial State
current_point = None
history       = deque(maxlen=9)

pos_x_steps, pos_x_vals = [], []
pos_y_steps, pos_y_vals = [], []
neg_x_steps, neg_x_vals = [], []
neg_y_steps, neg_y_vals = [], []

step_counter = 0

# Drawing simulation movements
def draw_static(ax, x_max):
    x1 = np.linspace(0, max_rent/q1, 200)
    ax.plot(x1, y_Q1(x1), 'C0', lw=2)
    x2 = np.linspace(-max_rent/q2, 0, 200)
    ax.plot(x2, y_Q2(x2), 'C1', lw=2)
    x3 = np.linspace(-house_cost - max_rent/q2, -house_cost, 200)
    ax.plot(x3, y_Q3(x3), 'C2', lw=2)
    x4 = np.linspace(0, x_max, 200)
    ax.plot(x4, y_Q4(x4), 'C3', lw=2)
    ax.axhline(0, color='k', lw=1)
    ax.axvline(0, color='k', lw=1)

def redraw():
    # left plot
    ax_sim.clear()
    xmax = slider_start_x.val * 1.5 + house_cost + 1
    draw_static(ax_sim, xmax)
    if len(history) > 1:
        H = np.array(history)
        ax_sim.plot(H[:,0], H[:,1], '-o', color='k', ms=4)
        ax_sim.plot(H[-1,0], H[-1,1], 'ro', ms=8)
    ax_sim.set_title('Simulation')

    # right plot
    ax_ts.clear()
    ax_ts.plot(pos_x_steps, np.abs(pos_x_vals), '-o', ms=4, label='Housing Stock')
    ax_ts.plot(pos_y_steps, np.abs(pos_y_vals), '-o', ms=4, label='Rent')
    ax_ts.plot(neg_x_steps, np.abs(neg_x_vals), '-o', ms=4, label='Purchasing Cost')
    ax_ts.plot(neg_y_steps, np.abs(neg_y_vals), '-o', ms=4, label='Construction')
    ax_ts.set_xlabel('Step')
    ax_ts.set_ylabel('Variable Value')
    ax_ts.set_title('Value of each Variable over time')
    ax_ts.legend(loc='best')
    ax_ts.grid(True)

    fig.canvas.draw_idle()

def init_state():
    update_ends()
    global current_point, history, step_counter
    x0 = slider_start_x.val
    current_point = np.array([x0, 0.0])
    history.clear(); history.append(current_point.copy())
    for lst in (pos_x_steps, pos_x_vals,
                pos_y_steps, pos_y_vals,
                neg_x_steps, neg_x_vals,
                neg_y_steps, neg_y_vals):
        lst.clear()
    step_counter = 0
    redraw()

def _step(fn):
    global current_point, step_counter
    current_point = fn(current_point)
    history.append(current_point.copy())
    step_counter += 1
    if fn is go_up:
        pos_x_steps.append(step_counter)
        pos_x_vals.append(current_point[0])
    elif fn is go_left:
        pos_y_steps.append(step_counter)
        pos_y_vals.append(current_point[1])
    elif fn is go_down:
        neg_x_steps.append(step_counter)
        neg_x_vals.append(current_point[0])
    else:
        neg_y_steps.append(step_counter)
        neg_y_vals.append(current_point[1])

def on_start(event):
    init_state(); plt.ion()
    for cycle in range(max_cycles):
        start_cycle = current_point.copy()
        for fn in steps:
            _step(fn); redraw(); plt.pause(delay)
        if np.linalg.norm(current_point - start_cycle) < convergence_value:
            print(f"Converged after {cycle+1} cycles at {current_point}")
            return
    print("Max cycles reached without convergence.")

def on_fast(event):
    init_state()
    for cycle in range(max_cycles):
        start_cycle = current_point.copy()
        for fn in steps:
            _step(fn)
        if np.linalg.norm(current_point - start_cycle) < convergence_value:
            print(f"Converged after {cycle+1} cycles at {current_point}")
            break
    redraw()

def on_reset(event):
    global current_point
    current_point = None
    history.clear()
    for lst in (pos_x_steps, pos_x_vals,
                pos_y_steps, pos_y_vals,
                neg_x_steps, neg_x_vals,
                neg_y_steps, neg_y_vals):
        lst.clear()
    redraw()

# Main loop 
if __name__ == '__main__':
    fig = plt.figure(figsize=(12,6))

    gs = fig.add_gridspec(
        nrows=7, ncols=5,
        left=0.05, right=0.95, bottom=0.05, top=0.90,
        width_ratios=[1,1,0.2,1,1],
        height_ratios=[0.08, 0.5, 0.05, 0.1, 0.1, 0.1, 0.1],
        wspace=0.3, hspace=0.4
    )

    ax_sim = fig.add_subplot(gs[1, 0:2])
    ax_ts  = fig.add_subplot(gs[1:, 3:5])

    # Buttons
    ax_start = fig.add_axes([0.05, 0.92, 0.10, 0.04])
    ax_fast  = fig.add_axes([0.17, 0.92, 0.12, 0.04])
    ax_reset = fig.add_axes([0.31, 0.92, 0.10, 0.04])
    start_btn = Button(ax_start, 'Start',        color='lightgreen', hovercolor='0.9')
    fast_btn  = Button(ax_fast,  'Fast-Forward', color='lightblue',  hovercolor='0.9')
    reset_btn = Button(ax_reset, 'Reset',        color='lightcoral', hovercolor='0.9')
    start_btn.on_clicked(on_start)
    fast_btn.on_clicked(on_fast)
    reset_btn.on_clicked(on_reset)

    # Sliders for simulation
    slider_w, slider_h = 0.24, 0.035
    x1, x2 = 0.05, 0.31
    base_y = 0.25
    dy = 0.06

    ax_q1 = fig.add_subplot(gs[3, 0])
    ax_q1.set_title('Q1 Elasticity', pad=6)
    slider_q1 = Slider(ax_q1, '', 0.5, 5.0, valinit=q1, valstep=0.5)

    ax_q2 = fig.add_subplot(gs[3, 1])
    ax_q2.set_title('Q2 Elasticity', pad=6)
    slider_q2 = Slider(ax_q2, '', 0.5, 5.0, valinit=q2, valstep=0.5)

    ax_q3 = fig.add_subplot(gs[4, 0])
    ax_q3.set_title('Q3 Elasticity', pad=6)
    slider_q3 = Slider(ax_q3, '', 0.5, 5.0, valinit=q3, valstep=0.5)

    ax_q4 = fig.add_subplot(gs[4, 1])
    ax_q4.set_title('Q4 Elasticity', pad=6)
    slider_q4 = Slider(ax_q4, '', 0.5, 5.0, valinit=q4, valstep=0.5)

    ax_hc = fig.add_subplot(gs[5, 0])
    ax_hc.set_title('Minimum House Cost', pad=6)
    slider_hc = Slider(ax_hc, '', 1.0, 10.0, valinit=house_cost, valstep=1.0)

    ax_mr = fig.add_subplot(gs[5, 1])
    ax_mr.set_title('Maximum Rent', pad=6)
    slider_mr = Slider(ax_mr, '', 0.0, 20.0, valinit=max_rent, valstep=0.5)

    ax_sx = fig.add_subplot(gs[6, 0])
    ax_sx.set_title('Start x', pad=6)
    slider_start_x = Slider(ax_sx, '', 0.0, 20.0, valinit=5.0, valstep=0.1)

    # Link sliders with action
    slider_q1.on_changed(lambda v: (globals().update(q1=v), init_state()))
    slider_q2.on_changed(lambda v: (globals().update(q2=v), init_state()))
    slider_q3.on_changed(lambda v: (globals().update(q3=v), init_state()))
    slider_q4.on_changed(lambda v: (globals().update(q4=v), init_state()))
    slider_hc.on_changed(lambda v: (globals().update(house_cost=v), init_state()))
    slider_mr.on_changed(lambda v: (globals().update(max_rent=v), init_state()))
    slider_start_x.on_changed(lambda v: init_state())

    redraw()
    plt.show()
