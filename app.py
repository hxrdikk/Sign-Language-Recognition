# app.py — ASL UI with Dark Theme, Live/Snapshot, FPS, smoothing, tips,
# banner (bright white, larger), and an embedded background image.

import os, time, base64
import cv2, numpy as np, tensorflow as tf, gradio as gr

MODEL_PATH = "SLR_final.h5"
assert os.path.exists(MODEL_PATH), "SLR_final.h5 not found next to app.py"
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE", "DELETE", "NOTHING"]
INPUT_SIZE = (64, 64)

# ---------- utils ----------
def b64_data_url(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode('utf-8')}"

BG = b64_data_url("background.jpg")   # site background
BANNER = b64_data_url("banner.png")   # top logo/banner

def center_crop_resize(img, size=(64, 64)):
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s)//2, (w - s)//2
    return cv2.resize(img[y0:y0+s, x0:x0+s], size, interpolation=cv2.INTER_AREA)

def preprocess_from_frame_rgb(frame_rgb):
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    img = center_crop_resize(bgr)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img.astype(np.float32) / 255.0)[None, ...]

def softmax_filter(probs, thresh):
    p = probs.copy()
    p[p < thresh] = 0.0
    return p

def ema(prev, current, alpha):
    return current if prev is None else (1 - alpha) * prev + alpha * current

# ---------- inference ----------
def predict_stream(frame_rgb, mirror, conf_thresh, smooth_alpha, state_prev, state_last_t):
    if frame_rgb is None:
        return {}, "", "FPS: –", state_prev, time.time()
    if mirror:
        frame_rgb = np.ascontiguousarray(frame_rgb[:, ::-1, :])

    x = preprocess_from_frame_rgb(frame_rgb)
    probs = model.predict(x, verbose=0)[0]
    smoothed = ema(state_prev, probs, smooth_alpha)
    disp = softmax_filter(smoothed, conf_thresh)

    top_idx = int(np.argmax(smoothed))
    bars = {CLASS_NAMES[i]: float(disp[i]) for i in range(len(CLASS_NAMES))}

    now = time.time()
    fps = 1.0 / max(1e-6, (now - state_last_t)) if state_last_t else 0.0
    return bars, CLASS_NAMES[top_idx], f"FPS: {fps:.1f}", smoothed, now

def predict_snapshot(frame_rgb, mirror, conf_thresh):
    if frame_rgb is None:
        return {}, "No image"
    if mirror:
        frame_rgb = np.ascontiguousarray(frame_rgb[:, ::-1, :])
    x = preprocess_from_frame_rgb(frame_rgb)
    probs = model.predict(x, verbose=0)[0]
    disp = softmax_filter(probs, conf_thresh)
    top_idx = int(np.argmax(probs))
    return {CLASS_NAMES[i]: float(disp[i]) for i in range(len(CLASS_NAMES))}, CLASS_NAMES[top_idx]

# ---------- CSS ----------
bg_css = f'url("{BG}") center / cover no-repeat fixed, #000000' if BG else "#000000"
custom_css = f"""
/* App background */
body, .gradio-container {{
  background: {bg_css} !important;
  color: #ffffff !important;
}}

/* Dark overlay */
.gradio-container::after {{
  content: "";
  position: fixed; inset: 0;
  background: rgba(0,0,0,0.35);
  pointer-events: none;
  z-index: 0;
}}

/* Panels */
.gr-block, .gr-box {{
  background-color: rgba(17,17,17,0.85) !important;
  border: 1px solid #333 !important;
  border-radius: 10px !important;
  color: #fff !important;
  backdrop-filter: blur(2px);
  z-index: 1;
}}

/* Hide image toolbars */
[data-testid="image-toolbar"],
button[aria-label="Download"],
button[aria-label="Fullscreen"] {{ display: none !important; }}

/* Footer */
footer {{ visibility: hidden; }}

/* Banner container */
#banner-box {{
  text-align: center;
  margin-bottom: 18px;
}}
#banner-box img {{
  width: auto; height: auto; max-height: 140px;
  object-fit: contain; display: block; margin: 0 auto 10px auto;
}}
/* Bright white heading & tagline */
#banner-box h1 {{
  color: #ffffff !important;
  font-size: 40px;        /* bigger */
  font-weight: 800;
  margin: 0;
}}
#banner-box p {{
  color: #ffffff !important;
  font-size: 20px;        /* bigger tagline */
  font-style: italic;
  margin: 6px 0 0 0;
  opacity: 1.0 !important;
}}

/* FPS */
#fpsbox {{ font-weight:600; }}
"""

# ---------- UI ----------
with gr.Blocks(title="ASL Sign Recognition", css=custom_css) as demo:
    with gr.Column(elem_id="banner-box"):
        if BANNER:
            gr.HTML(f'<img src="{BANNER}" alt="Banner" />')
        else:
            gr.HTML("""
            <h1>Sign Language Recognition</h1>
            <p>"Connecting worlds through instant sign language recognition technology"</p>
            """)

    mode = gr.Radio(choices=["Live (Chrome)", "Snapshot"], value="Live (Chrome)", label="Mode")

    with gr.Row():
        with gr.Column(scale=3):
            cam_live = gr.Image(sources=["webcam"], streaming=True, label="Webcam (live)", visible=True, height=420)
            cam_snap = gr.Image(sources=["webcam"], streaming=False, label="Webcam (click Capture)", visible=False, height=420)
            mirror = gr.Checkbox(value=True, label="Mirror preview")

            with gr.Accordion("Advanced", open=False):
                conf_thresh = gr.Slider(0.0, 0.2, value=0.02, step=0.01, label="Display threshold (hide small bars)")
                smooth_alpha = gr.Slider(0.0, 1.0, value=0.35, step=0.05, label="Smoothing (EMA α, Live mode)")

            gr.Markdown(
                "Tips:\n"
                "- Good lighting & neutral background.\n"
                "- Hold the sign near the center.\n"
                "- Static letters work best (J/Z may be weaker).\n"
            )

        with gr.Column(scale=2):
            label = gr.Label(num_top_classes=5, label="Top classes")
            top1 = gr.Textbox(label="Top-1", interactive=False)
            fps_box = gr.Textbox(label="Status", interactive=False, value="FPS: –", elem_id="fpsbox")

    state_prev = gr.State(None)
    state_last_t = gr.State(0.0)

    def on_mode_change(m):
        live = (m == "Live (Chrome)")
        return gr.update(visible=live), gr.update(visible=not live)
    mode.change(on_mode_change, inputs=mode, outputs=[cam_live, cam_snap])

    cam_live.stream(
        fn=predict_stream,
        inputs=[cam_live, mirror, conf_thresh, smooth_alpha, state_prev, state_last_t],
        outputs=[label, top1, fps_box, state_prev, state_last_t]
    )

    cam_snap.change(
        fn=predict_snapshot,
        inputs=[cam_snap, mirror, conf_thresh],
        outputs=[label, top1]
    )

if __name__ == "__main__":
    demo.launch()

