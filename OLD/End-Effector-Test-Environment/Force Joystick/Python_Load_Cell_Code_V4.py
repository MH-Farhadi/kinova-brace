import tkinter as tk
from tkinter import ttk
import serial
import math
import threading
import time
import random
from collections import deque

## Calibration: Z_FORCE_CAL and XY_FORCE_CAL are set so that raw vertical reading * calibration ≈ 0.52 kg.
Z_SMOOTH_ALPHA = 0.2
STOP_THRESHOLD = 0.01
Z_WEIGHT4_MIN = 0.02
Z_WEIGHT4_MAX_POS = 1.0
Z_WEIGHT4_MAX_NEG = 0.7
THRESHOLD_Z = 0.02
Z_FORCE_CAL = 4.45
XY_FORCE_CAL = 1.33
MIN_MAGNITUDE_THRESHOLD = 0.03
THRESHOLD_ANGLE = 3.0
THRESHOLD_MAGNITUDE = 0.02
FORCE_THRESHOLD = 0.05
graph_time_window = 10.0
Z_GRAPH_SCALE = 2.5
FX_GRAPH_SCALE = 2.5
FY_GRAPH_SCALE = 2.5

z_history = deque(maxlen=1000)
fx_history = deque(maxlen=1000)
fy_history = deque(maxlen=1000)

sample_prev = None
sample_curr = None
sample_lock = threading.Lock()

try:
    ser = serial.Serial('COM8', 115200, timeout=0.01)
except Exception as e:
    print("Error: Could not open serial port:", e)
    exit()

def serial_reader():
    global sample_prev, sample_curr
    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            if not line: continue
            tokens = line.split(',')
            if len(tokens) != 8: continue
            try:
                values = list(map(float, tokens))
            except Exception:
                continue
            sample = {
                'weights': values[0:4],
                'fx': values[4],
                'fy': values[5],
                'magnitude': values[6],
                'angle': values[7],
                'timestamp': time.time()
            }
            with sample_lock:
                if sample_curr is None:
                    sample_curr = sample
                    sample_prev = sample
                else:
                    sample_prev = sample_curr
                    sample_curr = sample
        except Exception as e:
            print("Serial reader error:", e)
threading.Thread(target=serial_reader, daemon=True).start()

window = tk.Tk()
window.title("Load Cell Readings")
window.geometry("1200x800")

style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", font=("Roboto", 16))
style.configure("Value.TLabel", font=("Roboto", 20), foreground="#1976D2")
style.configure("TButton", font=("Roboto", 16))

## Left Frame
left_frame = ttk.Frame(window)
left_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ns")
weight_labels = []
for i in range(4):
    lbl = ttk.Label(left_frame, text=f"Weight {i+1}: 0.00 kg", style="Value.TLabel", width=20)
    lbl.pack(anchor="w", pady=5)
    weight_labels.append(lbl)
xy_frame = ttk.Frame(left_frame)
xy_frame.pack(anchor="w", pady=10)
magnitude_label = ttk.Label(xy_frame, text="Magnitude: 0.00 kg", style="Value.TLabel", width=20)
magnitude_label.pack(anchor="w", pady=5)
angle_label = ttk.Label(xy_frame, text="Angle: N/A", style="Value.TLabel", width=20)
angle_label.pack(anchor="w", pady=5)
z_frame = ttk.Frame(left_frame)
z_frame.pack(anchor="w", pady=10)
vertical_label = ttk.Label(z_frame, text="Vertical Force:   0.00 kg", style="Value.TLabel", width=25)
vertical_label.pack(anchor="w", pady=5)
xy_frame.lift()
display_mode = tk.StringVar(value="XY")
def toggle_mode():
    if display_mode.get() == "XY":
        display_mode.set("Z")
        z_frame.lift()
        toggle_button.config(text="Switch to XY Mode")
        graph_canvas_frame.lift()
        fx_frame.grid_remove()
        fy_frame.grid_remove()
    else:
        display_mode.set("XY")
        xy_frame.lift()
        toggle_button.config(text="Switch to Z Mode")
        arrow_canvas_frame.lift()
        fx_frame.grid(row=1, column=0, sticky="nsew")
        fy_frame.grid(row=2, column=0, sticky="nsew")
toggle_button = ttk.Button(left_frame, text="Switch to Z Mode", command=toggle_mode)
toggle_button.pack(anchor="w", pady=10)
filter_slider = tk.Scale(left_frame, from_=-100, to=100, orient=tk.HORIZONTAL, label="Filter", length=400)
filter_slider.set(0)
filter_slider.pack(anchor="w", pady=5)

## Right Frame
display_canvas_frame = ttk.Frame(window)
display_canvas_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
window.columnconfigure(1, weight=1)
window.rowconfigure(0, weight=1)
display_canvas_frame.rowconfigure(0, weight=3)
display_canvas_frame.rowconfigure(1, weight=1)
display_canvas_frame.rowconfigure(2, weight=1)
arrow_canvas_frame = ttk.Frame(display_canvas_frame)
arrow_canvas_frame.grid(row=0, column=0, sticky="nsew")
canvas_size = 500
arrow_canvas = tk.Canvas(arrow_canvas_frame, width=canvas_size, height=canvas_size, bg="white", highlightthickness=0)
arrow_canvas.pack(expand=True, fill="both")
center_x = canvas_size // 2
center_y = canvas_size // 2
arrow = arrow_canvas.create_line(center_x, center_y, center_x, center_y - 50, arrow=tk.LAST, width=6, fill="#1976D2")
graph_canvas_frame = ttk.Frame(display_canvas_frame)
graph_canvas_frame.grid(row=0, column=0, sticky="nsew")
graph_canvas = tk.Canvas(graph_canvas_frame, width=canvas_size, height=canvas_size, bg="white", highlightthickness=0)
graph_canvas.pack(expand=True, fill="both")
fx_frame = ttk.Frame(display_canvas_frame)
fx_frame.grid(row=1, column=0, sticky="nsew")
fx_label = ttk.Label(fx_frame, text="Fx", font=("Roboto", 14))
fx_label.pack(anchor="w")
fx_canvas = tk.Canvas(fx_frame, width=canvas_size, height=150, bg="white", highlightthickness=0)
fx_canvas.pack(expand=True, fill="both")
fy_frame = ttk.Frame(display_canvas_frame)
fy_frame.grid(row=2, column=0, sticky="nsew")
fy_label = ttk.Label(fy_frame, text="Fy", font=("Roboto", 14))
fy_label.pack(anchor="w")
fy_canvas = tk.Canvas(fy_frame, width=canvas_size, height=150, bg="white", highlightthickness=0)
fy_canvas.pack(expand=True, fill="both")
arrow_canvas_frame.lift()

def interpolate_linear(v0, v1, fraction):
    return v0 * (1 - fraction) + v1 * fraction
def interpolate_angle(a0, a1, fraction):
    diff = (a1 - a0 + 180) % 360 - 180
    return (a0 + fraction * diff) % 360
def compute_arrow_length(magnitude):
    if magnitude <= 1.0:
        return 50 + 150 * magnitude
    else:
        return 50 + 150 + 50 * (magnitude - 1.0)

last_gui_angle = None
last_gui_magnitude = None
last_z_smooth = None
last_fx_smooth = None
last_fy_smooth = None

def update_graph():
    if display_mode.get() != "Z":
        graph_canvas.delete("all")
        return
    graph_canvas.delete("all")
    now = time.time()
    t_min = now - graph_time_window
    pts = []
    for t, z in list(z_history):
        if t >= t_min:
            x = (t - t_min) / graph_time_window * canvas_size
            y = canvas_size/2 - (z / Z_GRAPH_SCALE) * (canvas_size/2)
            pts.extend([x, y])
    if len(pts) >= 4:
        graph_canvas.create_line(pts, fill="#1976D2", width=2)
    graph_canvas.create_line(0, canvas_size/2, canvas_size, canvas_size/2, fill="gray", dash=(2,2))

def update_fx_graph():
    fx_canvas.delete("all")
    now = time.time()
    t_min = now - graph_time_window
    pts = []
    for t, val in list(fy_history):  # Swap: Fx graph uses fy_history
        if t >= t_min:
            x = (t - t_min) / graph_time_window * canvas_size
            y = 75 - (val / FX_GRAPH_SCALE) * 75
            pts.extend([x, y])
    if len(pts) >= 4:
        fx_canvas.create_line(pts, fill="blue", width=2)
    fx_canvas.create_line(0, 75, canvas_size, 75, fill="gray", dash=(2,2))

def update_fy_graph():
    fy_canvas.delete("all")
    now = time.time()
    t_min = now - graph_time_window
    pts = []
    for t, val in list(fx_history):  # Swap: Fy graph uses fx_history
        if t >= t_min:
            x = (t - t_min) / graph_time_window * canvas_size
            y = 75 - (val / FY_GRAPH_SCALE) * 75
            pts.extend([x, y])
    if len(pts) >= 4:
        fy_canvas.create_line(pts, fill="red", width=2)
    fy_canvas.create_line(0, 75, canvas_size, 75, fill="gray", dash=(2,2))

def apply_filter(raw, smooth, f_val):
    if f_val >= 0:
        return raw * (1 - f_val/50.0) + smooth * (f_val/50.0)
    else:
        noise = random.gauss(0, 0.05)
        return raw + (abs(f_val)/50.0)*noise

def update_gui_loop():
    global sample_prev, sample_curr, last_gui_angle, last_gui_magnitude, last_z_smooth, last_fx_smooth, last_fy_smooth
    now = time.time()
    with sample_lock:
        sp = sample_prev
        sc = sample_curr
    if sp is None or sc is None:
        window.after(5, update_gui_loop)
        return
    t0 = sp['timestamp']
    t1 = sc['timestamp']
    fraction = 1.0 if t1 == t0 else (now - t0) / (t1 - t0)
    fraction = max(0.0, min(1.0, fraction))
    weights_interp = [interpolate_linear(sp['weights'][i], sc['weights'][i], fraction) for i in range(4)]
    mag_interp = interpolate_linear(sp['magnitude'], sc['magnitude'], fraction)
    ang_interp = interpolate_angle(sp['angle'], sc['angle'], fraction)
    fx_interp = interpolate_linear(sp['fx'], sc['fx'], fraction)
    fy_interp = interpolate_linear(sp['fy'], sc['fy'], fraction)
    z_blue = weights_interp[0] - weights_interp[2]
    z_red = weights_interp[3]
    if z_red >= 0:
        if z_red <= Z_WEIGHT4_MIN:
            blend_factor = 0.0
        elif z_red >= Z_WEIGHT4_MAX_POS:
            blend_factor = 1.0
        else:
            blend_factor = (z_red - Z_WEIGHT4_MIN) / (Z_WEIGHT4_MAX_POS - Z_WEIGHT4_MIN)
    else:
        abs_z_red = abs(z_red)
        if abs_z_red <= Z_WEIGHT4_MIN:
            blend_factor = 0.0
        elif abs_z_red >= Z_WEIGHT4_MAX_NEG:
            blend_factor = 1.0
        else:
            blend_factor = (abs_z_red - Z_WEIGHT4_MIN) / (Z_WEIGHT4_MAX_NEG - Z_WEIGHT4_MIN)
    z_estimated = (1 - blend_factor) * z_blue + blend_factor * z_red
    z_estimated *= Z_FORCE_CAL
    if last_z_smooth is None:
        z_smooth = z_estimated
    else:
        z_smooth = last_z_smooth + Z_SMOOTH_ALPHA * (z_estimated - last_z_smooth)
    last_z_smooth = z_smooth
    if abs(z_smooth) < STOP_THRESHOLD:
        z_smooth = 0.0
        last_z_smooth = 0.0
    if last_fx_smooth is None:
        fx_smooth = fx_interp
    else:
        fx_smooth = last_fx_smooth + Z_SMOOTH_ALPHA * (fx_interp - last_fx_smooth)
    last_fx_smooth = fx_smooth
    if last_fy_smooth is None:
        fy_smooth = fy_interp
    else:
        fy_smooth = last_fy_smooth + Z_SMOOTH_ALPHA * (fy_interp - last_fy_smooth)
    last_fy_smooth = fy_smooth
    f_val = filter_slider.get()
    final_z = apply_filter(z_estimated, z_smooth, f_val)
    final_mag = apply_filter(mag_interp * XY_FORCE_CAL, mag_interp * XY_FORCE_CAL, f_val)
    final_fx = apply_filter(fx_interp, fx_smooth, f_val)
    final_fy = apply_filter(fy_interp, fy_smooth, f_val)
    for i in range(4):
        weight_labels[i].config(text=f"Weight {i+1}: {weights_interp[i]:.2f} kg")
    if display_mode.get() == "XY":
        if final_mag < MIN_MAGNITUDE_THRESHOLD:
            final_mag = 0.0
        magnitude_label.config(text=f"Magnitude: {final_mag:.2f} kg")
        angle_label.config(text=f"Angle: {ang_interp:.2f}°")
        vertical_label.config(text="Vertical Force: N/A")
    else:
        vertical_label.config(text=f"Vertical Force: {final_z:+06.2f} kg")
    if display_mode.get() == "Z":
        z_history.append((now, final_z))
    else:
        graph_canvas.delete("all")
    update_graph()
    if display_mode.get() == "XY":
        fx_history.append((now, final_fx))
        fy_history.append((now, final_fy))
        update_fx_graph()
        update_fy_graph()
    global last_gui_angle, last_gui_magnitude
    if last_gui_angle is None:
        stable_angle = ang_interp
    else:
        if abs((ang_interp - last_gui_angle + 180) % 360 - 180) < THRESHOLD_ANGLE:
            stable_angle = last_gui_angle
        else:
            stable_angle = ang_interp
    if last_gui_magnitude is None:
        stable_magnitude = final_mag
    else:
        if abs(final_mag - last_gui_magnitude) < THRESHOLD_MAGNITUDE:
            stable_magnitude = last_gui_magnitude
        else:
            stable_magnitude = final_mag
    last_gui_angle = stable_angle
    last_gui_magnitude = stable_magnitude
    if display_mode.get() == "XY":
        if stable_magnitude < FORCE_THRESHOLD:
            angle_label.config(text="Angle: N/A")
            arrow_canvas.itemconfigure(arrow, state="hidden")
        else:
            angle_label.config(text=f"Angle: {stable_angle:.2f}°")
            arrow_canvas.itemconfigure(arrow, state="normal")
            arrow_length = compute_arrow_length(stable_magnitude)
            rad = math.radians(stable_angle)
            end_x = center_x + arrow_length * math.sin(rad)
            end_y = center_y - arrow_length * math.cos(rad)
            arrow_canvas.coords(arrow, center_x, center_y, end_x, end_y)
    window.after(5, update_gui_loop)

def update_fx_graph():
    fx_canvas.delete("all")
    now = time.time()
    t_min = now - graph_time_window
    pts = []
    for t, val in list(fy_history):  # Swap: Fx graph uses fy_history
        if t >= t_min:
            x = (t - t_min) / graph_time_window * canvas_size
            y = 75 - (val / FX_GRAPH_SCALE) * 75
            pts.extend([x, y])
    if len(pts) >= 4:
        fx_canvas.create_line(pts, fill="blue", width=2)
    fx_canvas.create_line(0, 75, canvas_size, 75, fill="gray", dash=(2,2))

def update_fy_graph():
    fy_canvas.delete("all")
    now = time.time()
    t_min = now - graph_time_window
    pts = []
    for t, val in list(fx_history):  # Swap: Fy graph uses fx_history
        if t >= t_min:
            x = (t - t_min) / graph_time_window * canvas_size
            y = 75 - (val / FY_GRAPH_SCALE) * 75
            pts.extend([x, y])
    if len(pts) >= 4:
        fy_canvas.create_line(pts, fill="red", width=2)
    fy_canvas.create_line(0, 75, canvas_size, 75, fill="gray", dash=(2,2))

update_gui_loop()
window.mainloop()
