import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit.components.v1 as components
import os
from scipy.ndimage import gaussian_filter, shift

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Magnetovault V2.86 - Gibbs Phantom Fix")

# Imports Médicaux
try:
    import nibabel as nib
    from nilearn import datasets
    HAS_NILEARN = True
except ImportError:
    HAS_NILEARN = False

# --- 2. FONCTION RERUN ---
def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# --- 3. CONSTANTES PHYSIQUES ---
T_FAT = {'T1': 260.0, 'T2': 60.0, 'PD': 1.0, 'Label': 'Graisse'}
T_LCR = {'T1': 3607.0, 'T2': 2000.0, 'PD': 1.0, 'Label': 'Eau (LCR)'}
T_GM  = {'T1': 1300.0, 'T2': 140.0, 'PD': 0.95, 'Label': 'Subst. Grise'}
T_WM  = {'T1': 600.0,  'T2': 90.0,  'PD': 0.70, 'Label': 'Subst. Blanche'}

# --- 4. INITIALISATION ---
if 'init' not in st.session_state:
    st.session_state.seq = 'T1 Standard'
    st.session_state.reset_count = 0
    st.session_state.init = True

# --- 5. LOGIQUE DE RESET ---
def reset_all():
    st.session_state.reset_count += 1
    st.session_state.seq = "T1 Standard"
    safe_rerun()

# --- 6. LOGO ---
dossier_actuel = os.path.dirname(os.path.abspath(__file__))
chemin_logo = os.path.join(dossier_actuel, "logo_mia.png")
if os.path.exists(chemin_logo):
    st.sidebar.image(chemin_logo, width=280)
elif os.path.exists("logo_mia.png"):
    st.sidebar.image("logo_mia.png", width=280)

st.sidebar.title("Réglages")
st.sidebar.button("Reset Complet", on_click=reset_all)

# --- 7. SÉLECTION SÉQUENCE ---
options_seq = [
    "T1 Standard", 
    "T2 Standard", 
    "DP (Densité Protons)", 
    "FLAIR (Eau -)", 
    "Séquence STIR (Graisse)" 
]

seq_key = f"seq_select_{st.session_state.reset_count}"
try: idx_default = options_seq.index(st.session_state.seq)
except: idx_default = 0
seq_choix = st.sidebar.selectbox("Séquence", options_seq, index=idx_default, key=seq_key)
st.session_state.seq = seq_choix

# --- 8. VALEURS PAR DÉFAUT ---
defaults = {'tr': 500.0, 'te': 10.0, 'ti': 0.0}
if seq_choix == "T2 Standard": defaults = {'tr': 4000.0, 'te': 100.0, 'ti': 0.0}
elif seq_choix == "DP (Densité Protons)": defaults = {'tr': 2200.0, 'te': 30.0, 'ti': 0.0}
elif "FLAIR" in seq_choix: defaults = {'tr': 9000.0, 'te': 110.0, 'ti': 2500.0}
elif "STIR" in seq_choix: defaults = {'tr': 3500.0, 'te': 50.0, 'ti': 150.0}

# --- 9. SLIDERS ---
current_reset_id = st.session_state.reset_count
seq_id = seq_choix.replace(" ", "_") 

st.sidebar.header("1. Chrono (ms)")
is_ir = "FLAIR" in seq_choix or "STIR" in seq_choix

if is_ir:
    ti = st.sidebar.slider("TI", 0.0, 3500.0, float(defaults['ti']), step=10.0, key=f"ti_{seq_id}_{current_reset_id}")
else:
    ti = 0.0

tr = st.sidebar.slider("TR", 100.0, 10000.0, float(defaults['tr']), step=50.0, key=f"tr_{seq_id}_{current_reset_id}")
te = st.sidebar.slider("TE Effectif", 5.0, 300.0, float(defaults['te']), step=5.0, key=f"te_{seq_id}_{current_reset_id}")

st.sidebar.header("2. Géométrie")
fov = st.sidebar.slider("FOV (mm)", 100.0, 500.0, 240.0, step=10.0, key=f"fov_{current_reset_id}")
mat = st.sidebar.select_slider("Matrice", options=[64, 128, 256, 512], value=256, key=f"mat_{current_reset_id}")
ep = st.sidebar.slider("Epaisseur", 1.0, 10.0, 5.0, step=0.5, key=f"ep_{current_reset_id}")

st.sidebar.header("3. Options")
nex = st.sidebar.slider("NEX", 1, 8, 1, key=f"nex_{current_reset_id}")
turbo = st.sidebar.slider("Facteur Turbo", 1, 32, 1, key=f"turbo_{current_reset_id}")
bw = st.sidebar.slider("Bande Passante (Hz/px)", 50, 500, 220, step=10, key=f"bw_{current_reset_id}", help="220 Hz = Standard 1.5T")
es = st.sidebar.slider("TE Mini (Espacement)", 2.5, 20.0, 10.0, step=2.5, key=f"es_{current_reset_id}")

# --- 10. CALCULS GLOBAUX ---
def get_signal_val(t1, t2, pd):
    e2 = np.exp(-te / t2)
    val = 0.0
    if ti > 10: 
        val = pd * (1 - 2 * np.exp(-ti / t1) + np.exp(-tr / t1)) * e2
    else: 
        val = pd * (1 - np.exp(-tr / t1)) * e2
    return np.abs(val)

v_lcr = get_signal_val(T_LCR['T1'], T_LCR['T2'], T_LCR['PD'])
v_wm  = get_signal_val(T_WM['T1'], T_WM['T2'], T_WM['PD']) 
v_gm  = get_signal_val(T_GM['T1'], T_GM['T2'], T_GM['PD']) 
v_fat = get_signal_val(T_FAT['T1'], T_FAT['T2'], T_FAT['PD'])

# Temps
raw_ms = (tr * mat * nex) / turbo
final_seconds = raw_ms / 1000.0
mins = int(final_seconds // 60)
secs = int(final_seconds % 60)
str_duree = f"{mins} min {secs} s"

# SNR
vol_factor = (fov/float(mat))**2 * ep
acq_factor = np.sqrt(float(mat)*float(nex)/float(turbo))
bw_factor = np.sqrt(220.0 / float(bw))

ref_wm_signal = 0.7 * (1 - np.exp(-500.0/600.0)) * np.exp(-10.0/90.0)
s_ref = ref_wm_signal 
if s_ref < 0.0001: s_ref = 0.0001
r_sig = v_wm / s_ref

base_vol = (240.0/256.0)**2 * 5.0
base_acq = np.sqrt(256.0)
r_vol = vol_factor / base_vol
r_acq = acq_factor / base_acq

snr_val = r_vol * r_acq * bw_factor * r_sig * 100.0
str_snr = f"{snr_val:.1f} %"

# --- 11. GENERATION FANTOME ---
S = mat
x = np.linspace(-1, 1, S); y = np.linspace(-1, 1, S)
X, Y = np.meshgrid(x, y); D = np.sqrt(X**2 + Y**2)

img_water = np.zeros((S, S))
img_fat = np.zeros((S, S))

img_water[D < 0.20] = v_lcr
img_water[(D >= 0.20) & (D < 0.50)] = v_wm
img_water[(D >= 0.50) & (D < 0.80)] = v_gm
img_fat[(D >= 0.80) & (D < 0.95)] = v_fat

if bw == 220: shift_pixels = 0.0 
else: shift_pixels = 220.0 / float(bw)

img_fat_shifted = shift(img_fat, [0, shift_pixels], mode='constant', cval=0.0)
final = np.clip(img_water + img_fat_shifted, 0, 1.3)

noise_scale = 5.0 / (snr_val + 0.1)
if noise_scale < 0: noise_scale = 0.1
final += np.random.normal(0, noise_scale, (S,S))
final = np.clip(final, 0, 1.3)

f = np.fft.fftshift(np.fft.fft2(final))
kspace = 20 * np.log(np.abs(f) + 1)

# --- 12. MOTEUR ANATOMIQUE ---
class AdvancedMRIProcessor:
    def __init__(self):
        self.ready = False
        if HAS_NILEARN:
            try: self._load_data(); self.ready = True
            except: pass

    @st.cache_resource
    def _load_data(_self):
        dataset = datasets.fetch_icbm152_2009()
        gm = nib.load(dataset['gm']).get_fdata()
        wm = nib.load(dataset['wm']).get_fdata()
        csf = nib.load(dataset['csf']).get_fdata()
        rest = np.clip(1.0 - (gm + wm + csf), 0, 1)
        return gm, wm, csf, rest

    def get_slice(self, axis, index, w_vals):
        if not self.ready: return None
        gm, wm, csf, rest = self._load_data()
        ax_idx = {'x': 0, 'y': 1, 'z': 2}.get(axis, 2)
        s_gm = np.take(gm, index, axis=ax_idx).T
        s_wm = np.take(wm, index, axis=ax_idx).T
        s_csf = np.take(csf, index, axis=ax_idx).T
        s_rest = np.take(rest, index, axis=ax_idx).T
        sim = (s_csf * w_vals['csf']) + (s_gm * w_vals['gm']) + \
              (s_wm * w_vals['wm']) + (s_rest * (w_vals['fat']*0.2))
        if axis == 'z': sim = np.flipud(sim)
        elif axis == 'x': sim = np.flipud(np.fliplr(sim))
        elif axis == 'y': sim = np.flipud(sim)
        return sim

    def get_dims(self): return (197, 233, 189) if self.ready else (100, 100, 100)

processor = AdvancedMRIProcessor()

def apply_window_level(image, window, level):
    win = max(0.001, window)
    vmin, vmax = level - win/2, level + win/2
    return np.clip((image - vmin)/(vmax - vmin), 0, 1)

# --- 13. AFFICHAGE FINAL ---
st.title("Simulateur MagnétoVault V2.86")
t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs(["Fantôme", "Espace K 🌀", "Signaux", "Codage", "🧠 Anatomie", "📈 Physique", "⚡ Chronogramme", "☣️ Artefacts"])

# TAB 1
with t1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("⏱️ Temps d'Acquisition")
        st.latex(r"TA = \frac{TR \times Mat \times NEX}{FacteurTurbo}")
        st.code(f"({tr:.0f} * {mat} * {nex}) / ({turbo} * 1000) = {final_seconds:.1f} s")
        st.success(f"👉 **DURÉE FINALE : {str_duree}**")
        st.divider()
        st.subheader("📉 Rapport Signal/Bruit (SNR)")
        
        # FORMULE CORRIGEE : Utilisation de \mathrm{} pour éviter la traduction
        st.latex(r"SNR \propto V_{vox} \times \sqrt{Acq} \times \frac{1}{\sqrt{\mathrm{BW}}}")
        st.markdown("""
        **Légende :**
        * **$V_{vox}$** : Volume du Voxel.
        * **$\sqrt{Acq}$** : Facteur d'acquisition (NEX).
        * **$BW$** : **Bande Passante** (Bandwidth). Une bande passante étroite augmente le SNR.
        """)
        
        st.table(pd.DataFrame({"Facteur": ["Volume Voxel", "Acquisition", "Bande Passante"], "Impact": [f"x {r_vol:.2f}", f"x {r_acq:.2f}", f"x {bw_factor:.2f}"]}))
        st.info(f"👉 **SNR RELATIF : {str_snr}**")
    with c2:
        fig_anot, ax_anot = plt.subplots(figsize=(5,5))
        ax_anot.imshow(final, cmap='gray', vmin=0, vmax=1.3)
        ax_anot.axis('off')
        c = S // 2; r = S/2.0
        ax_anot.text(c, c, "WATER", color='cyan', ha='center', weight='bold')
        ax_anot.text(c, c - r*0.35, "WM", color='lightgray', ha='center', weight='bold')
        ax_anot.text(c, c - r*0.65, "GM", color='white', ha='center', weight='bold')
        ax_anot.text(c, c - r*0.88, "FAT", color='orange', ha='center', weight='bold')
        st.pyplot(fig_anot)
        plt.close(fig_anot)

# TAB 2
with t2:
    st.markdown("### 🌀 Remplissage de l'Espace K")
    col_k1, col_k2 = st.columns([1, 1])
    with col_k1:
        fill_mode = st.radio("Ordre de Remplissage", ["Linéaire (Haut -> Bas)", "Centrique (Centre -> Bords)"], key=f"k_mode_{current_reset_id}")
        acq_pct = st.slider("Progression (%)", 0, 100, 10, step=1, key=f"k_pct_{current_reset_id}")
        mask_k = np.zeros((S, S))
        lines_to_fill = int(S * (acq_pct / 100.0))
        if "Linéaire" in fill_mode: mask_k[0:lines_to_fill, :] = 1
        else: 
            center_line = S // 2; half = lines_to_fill // 2
            mask_k[center_line-half:center_line+half, :] = 1
    with col_k2:
        kspace_masked = f * mask_k
        img_rec = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_masked)))
        fig_k, ax_k = plt.subplots(figsize=(4, 4))
        ax_k.imshow(20 * np.log(np.abs(kspace_masked) + 1), cmap='inferno')
        ax_k.axis('off')
        st.pyplot(fig_k); plt.close(fig_k)
        st.image(img_rec, clamp=True, width=300)

with t3:
    fig, ax = plt.subplots(figsize=(8, 5))
    vals_bar = [v_lcr, v_gm, v_wm, v_fat] 
    noms = ["WATER", "GM", "WM", "FAT"]
    cols = ['cyan', 'dimgray', 'lightgray', 'orange']
    bars = ax.bar(noms, vals_bar, color=cols, edgecolor='black')
    ax.set_ylim(0, 1.3); ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    for r in bars: ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.02, f"{r.get_height():.2f}", ha='center')
    st.pyplot(fig); plt.close(fig)

with t4:
    h1="<!DOCTYPE html><html><head><style>body{margin:0;padding:5px;font-family:sans-serif;} .box{display:flex;gap:15px;} .ctrl{width:220px;padding:10px;background:#f9f9f9;border:1px solid #ccc;border-radius:8px;} canvas{border:1px solid #ccc;background:#f8f9fa;border-radius:8px;} input{width:100%;} label{font-size:11px;font-weight:bold;display:block;} button{width:100%;padding:8px;background:#4f46e5;color:white;border:none;border-radius:4px;cursor:pointer;}</style></head><body><div class='box'><div class='ctrl'><h4>Codage</h4><label>Freq</label><input type='range' id='f' min='-100' max='100' value='0'><br><label>Phase</label><input type='range' id='p' min='-100' max='100' value='0'><br><label>Coupe</label><input type='range' id='z' min='-100' max='100' value='0'><br><label>Matrice</label><input type='range' id='g' min='5' max='20' value='12'><br><button onclick='rst()'>Reset</button></div><div><canvas id='c1' width='350' height='350'></canvas><canvas id='c2' width='80' height='350'></canvas></div></div>"
    h2="<script>const c1=document.getElementById('c1');const x=c1.getContext('2d');const c2=document.getElementById('c2');const z=c2.getContext('2d');const sf=document.getElementById('f');const sp=document.getElementById('p');const sz=document.getElementById('z');const sg=document.getElementById('g');const pd=30;function arrow(ctx,x,y,a,s){const l=s*0.35;ctx.save();ctx.translate(x,y);ctx.rotate(a);ctx.beginPath();ctx.moveTo(-l,0);ctx.lineTo(l,0);ctx.lineTo(l-6,-6);ctx.moveTo(l,0);ctx.lineTo(l-6,6);ctx.strokeStyle='white';ctx.lineWidth=1.5;ctx.stroke();ctx.restore();} function draw(){x.clearRect(0,0,350,350);z.clearRect(0,0,80,350);const fv=parseFloat(sf.value);const pv=parseFloat(sp.value);const zv=parseFloat(sz.value);const gs=parseInt(sg.value);const st=(350-2*pd)/gs;"
    h3="const h=(pd*0.8)*(fv/100);x.fillStyle='rgba(255,0,0,0.3)';if(fv!=0){x.beginPath();x.moveTo(pd,pd/2);x.lineTo(pd,pd/2-h);x.lineTo(350-pd,pd/2+h);x.lineTo(350-pd,pd/2);x.fill();}const w=(pd*0.8)*(pv/100);x.fillStyle='rgba(0,255,0,0.3)';if(pv!=0){x.beginPath();x.moveTo(350-pd/2,pd);x.lineTo(350-pd/2-w,pd);x.lineTo(350-pd/2+w,350-pd);x.lineTo(350-pd/2,350-pd);x.fill();} for(let i=0;i<gs;i++){for(let j=0;j<gs;j++){const cx=pd+i*st+st/2;const cy=pd+j*st+st/2;const ph=(i-gs/2)*(fv/100)*3+(j-gs/2)*(pv/100)*3;const cph=(j-gs/2)*(pv/100);x.strokeStyle='black';x.beginPath();x.arc(cx,cy,st*0.4,0,6.28);x.fillStyle='#94a3b8';x.fill();if(cph>0.01)x.fillStyle='rgba(255,255,0,0.5)';if(cph<-0.01)x.fillStyle='rgba(0,0,255,0.5)';x.fill();arrow(x,cx,cy,ph,st*0.6);}}"
    h4="const yz=175-(zv/100)*150;const gr=z.createLinearGradient(0,0,0,350);gr.addColorStop(0,'red');gr.addColorStop(1,'blue');z.fillStyle=gr;z.fillRect(10,10,20,330);z.strokeStyle='black';z.lineWidth=3;z.beginPath();z.moveTo(10,yz);z.lineTo(70,yz);z.stroke();z.fillStyle='black';z.fillText('Z',35,yz-5);} [sf,sp,sz,sg].forEach(s=>s.addEventListener('input',draw));function rst(){sf.value=0;sp.value=0;sz.value=0;sg.value=12;draw();}draw();</script></body></html>"
    components.html(h1+h2+h3+h4, height=450)

with t5:
    st.header("Exploration Anatomique")
    if HAS_NILEARN and processor.ready:
        c1, c2 = st.columns([1, 3])
        dims = processor.get_dims()
        with c1:
            plane = st.radio("Plan de Coupe", ["Plan Axial", "Plan Sagittal", "Plan Coronal"], key="or_286")
            if "Axial" in plane: idx = st.slider("Z", 0, dims[2]-1, 90, key=f"sl_{current_reset_id}"); ax='z'
            elif "Sagittal" in plane: idx = st.slider("X", 0, dims[0]-1, 90, key=f"sl_{current_reset_id}"); ax='x'
            else: idx = st.slider("Y", 0, dims[1]-1, 100, key=f"sl_{current_reset_id}"); ax='y'
            st.divider()
            window = st.slider("Fenêtre", 0.01, 2.0, 1.0, 0.005, key=f"wn_{current_reset_id}")
            level = st.slider("Niveau", 0.0, 1.0, 0.5, 0.005, key=f"lv_{current_reset_id}")
        with c2:
            w_vals = {'csf':v_lcr, 'gm':v_gm, 'wm':v_wm, 'fat':v_fat}
            img_raw = processor.get_slice(ax, idx, w_vals)
            if img_raw is not None:
                img_wl = apply_window_level(img_raw, window, level)
                st.image(img_wl, clamp=True, width=600)
            else: st.error("Erreur image.")
    else: st.warning("Module 'nilearn' manquant.")

with t6:
    st.header("📈 Physique")
    tists = [T_FAT, T_WM, T_GM, T_LCR]
    cols = ['orange', 'lightgray', 'dimgray', 'cyan'] 
    fig_t1 = plt.figure(figsize=(10, 3))
    gs = fig_t1.add_gridspec(1, 2, width_ratios=[30, 1], wspace=0.05)
    ax_t1 = fig_t1.add_subplot(gs[0]); ax_bar = fig_t1.add_subplot(gs[1])
    x_t = np.linspace(0, 4000, 500)
    if is_ir:
        ax_t1.set_ylim(-1.1, 1.1); ax_t1.axhline(0, color='black')
        for t, col in zip(tists, cols):
            mz = 1 - 2 * np.exp(-x_t / t['T1'])
            ax_t1.plot(x_t, mz, color=col, label=t['Label'])
        ax_t1.axvline(x=ti, color='green', linestyle='--', label='TI')
        gradient = np.abs(np.linspace(1, -1, 256)).reshape(-1, 1)
        ax_bar.imshow(gradient, aspect='auto', cmap='gray', extent=[0, 1, -1.1, 1.1])
    else:
        ax_t1.set_ylim(0, 1.1)
        for t, col in zip(tists, cols):
            mz = 1 - np.exp(-x_t / t['T1'])
            ax_t1.plot(x_t, mz, color=col, label=t['Label'])
        ax_t1.axvline(x=tr, color='red', linestyle='--', label='TR')
        gradient = np.linspace(1, 0, 256).reshape(-1, 1)
        ax_bar.imshow(gradient, aspect='auto', cmap='gray', extent=[0, 1, 0, 1.1])
    ax_t1.legend(); ax_bar.axis('off'); ax_bar.set_title("Signal")
    st.pyplot(fig_t1); plt.close(fig_t1)
    
    fig_t2 = plt.figure(figsize=(10, 3))
    gs2 = fig_t2.add_gridspec(1, 2, width_ratios=[30, 1], wspace=0.05)
    ax_t2 = fig_t2.add_subplot(gs2[0]); ax_bar2 = fig_t2.add_subplot(gs2[1])
    x_te = np.linspace(0, 500, 300)
    for t, col in zip(tists, cols):
        mxy = np.exp(-x_te / t['T2'])
        ax_t2.plot(x_te, mxy, color=col, label=t['Label'])
    ax_t2.axvline(x=te, color='red', linestyle='--', label='TE Eff')
    gradient_t2 = np.linspace(1, 0, 256).reshape(-1, 1)
    ax_bar2.imshow(gradient_t2, aspect='auto', cmap='gray', extent=[0, 1, 0, 1.0])
    ax_bar2.axis('off'); ax_bar2.set_title("Signal")
    st.pyplot(fig_t2); plt.close(fig_t2)

with t7:
    is_turbo = turbo > 1
    t = np.linspace(0, 200, 1000)
    t_90 = 10; echo_times = [t_90 + (i+1)*es for i in range(turbo)]
    target_te_graph = te + 10 
    closest_idx = np.argmin(np.abs(np.array(echo_times) - target_te_graph))
    max_dist = max(closest_idx, (turbo-1) - closest_idx) if turbo > 1 else 1
    if max_dist == 0: max_dist = 1
    rf_sigma = 0.5; grad_width = max(1.5, es * 0.2)
    st.header(f"⚡ Séquence : {'Turbo ' if is_turbo else ''}Spin Écho (Facteur {turbo})")
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(10, 8), gridspec_kw={'hspace': 0.1})
    rf = np.zeros_like(t)
    def add_rf_pulse(center, amp, w): return amp * np.exp(-0.5 * ((t - center)**2) / (w**2))
    rf += add_rf_pulse(t_90, 1.0, rf_sigma) 
    t_180s = []
    for i in range(turbo):
        t_p = t_90 + (i * es) + (es/2)
        if t_p < 200:
            t_180s.append(t_p); rf += add_rf_pulse(t_p, 1.6, rf_sigma)
    axs[0].plot(t, rf, color='black', linewidth=1.5)
    axs[0].fill_between(t, 0, rf, color='green', alpha=0.4)
    axs[0].set_ylabel("RF"); axs[0].set_yticks([0, 1, 1.6], ["", "90", "180"])
    axs[0].spines['top'].set_visible(False); axs[0].spines['right'].set_visible(False); axs[0].set_xlim(0, 200)
    gsc = np.zeros_like(t)
    def add_trap(center, amp, w): mask = (t > center - w) & (t < center + w); gsc[mask] = amp
    add_trap(t_90, 1.0, grad_width); t_rephase = t_90 + grad_width + 1.5
    add_trap(t_rephase, -0.8, grad_width*0.6)
    for t_p in t_180s: add_trap(t_p, 1.0, grad_width)
    axs[1].fill_between(t, 0, gsc, color='green', alpha=0.6); axs[1].plot(t, gsc, color='green', linewidth=1)
    axs[1].set_ylabel("Gsc"); axs[1].set_yticks([]); axs[1].spines['top'].set_visible(False); axs[1].spines['right'].set_visible(False)
    gcp = np.zeros_like(t)
    for i in range(turbo):
        t_180 = t_90 + (i * es) + (es/2); t_echo = t_90 + ((i+1) * es)
        t_code = (t_180 + t_echo)/2 - (es*0.1); t_rewind = t_echo + (es*0.15)
        if t_rewind < 200:
            if i == closest_idx: height = 0.2; label = "BF"; col_lbl = "red"
            else: dist = abs(i - closest_idx); height = 0.2 + (0.8 * (dist / max_dist)); label = ""; col_lbl = "gray"
            w_ph = grad_width * 0.7; mask_c = (t > t_code - w_ph) & (t < t_code + w_ph); gcp[mask_c] = height
            if label == "BF": axs[2].text(t_code, height+0.1, label, color=col_lbl, ha='center', fontsize=9, weight='bold')
            mask_r = (t > t_rewind - w_ph) & (t < t_rewind + w_ph); gcp[mask_r] = -height
    axs[2].fill_between(t, 0, gcp, color='darkorange', alpha=0.7); axs[2].set_ylabel("Gcp"); axs[2].set_yticks([])
    axs[2].spines['top'].set_visible(False); axs[2].spines['right'].set_visible(False)
    gcf = np.zeros_like(t); t_pre = (t_90 + (t_90 + es/2))/2
    add_trap_gcf = lambda c, w: ((t > c - w) & (t < c + w)); gcf[add_trap_gcf(t_pre, grad_width)] = 1.0 
    for i in range(turbo):
        t_e = echo_times[i]
        if t_e < 200: w_read = grad_width * 1.2; gcf[add_trap_gcf(t_e, w_read)] = 1.0
    axs[3].fill_between(t, 0, gcf, color='dodgerblue', alpha=0.5); axs[3].set_ylabel("Gcf"); axs[3].set_yticks([])
    axs[3].spines['top'].set_visible(False); axs[3].spines['right'].set_visible(False)
    sig = np.zeros_like(t)
    for i in range(turbo):
        t_e = echo_times[i]
        if t_e < 195:
            w_sig = grad_width * 1.2; idx_start = np.argmin(np.abs(t - (t_e - w_sig))); idx_end = np.argmin(np.abs(t - (t_e + w_sig)))
            if idx_end > idx_start:
                grid = np.linspace(-3, 3, idx_end - idx_start); amp = np.exp(-(i*es)/T_GM['T2']) 
                sig[idx_start:idx_end] = np.sinc(grid) * amp
            if i == closest_idx:
                 axs[4].text(t_e, amp+0.2, "TE eff", ha='center', color='red', fontweight='bold', fontsize=10)
                 axs[4].axvline(x=t_e, color='red', linestyle='--', alpha=0.5)
    axs[4].plot(t, sig, color='navy', linewidth=1.5); axs[4].set_ylabel("Signal"); axs[4].set_xlabel("Temps (ms)")
    axs[4].set_yticks([]); axs[4].spines['top'].set_visible(False); axs[4].spines['right'].set_visible(False)
    st.pyplot(fig); plt.close(fig)

# --- TAB 8 : ARTEFACTS ---
with t8:
    st.header("☣️ Laboratoire d'Artefacts")
    
    col_ctrl, col_visu = st.columns([1, 2])
    
    with col_ctrl:
        st.markdown("#### Choix de l'Artefact")
        artefact_type = st.radio("Sélectionnez :", 
                                ["Aliasing (Repliement)", "Décalage Chimique",
                                 "Troncature (Gibbs)", "Mouvement (Ghosting)", "Zipper (Spike)"], 
                                key="art_main_radio")
        st.divider()

    # 1. ALIASING
    if "Aliasing" in artefact_type:
        with col_ctrl:
            with st.expander("ℹ️ Comprendre (L'Effet Pac-Man)", expanded=True):
                st.write("Si le cadre (FOV) est plus petit que le patient, ce qui dépasse d'un côté réapparaît de l'autre.")
            st.info(f"FOV Actuel : **{fov} mm**\n\n(Objet simulé : 230 mm)")
            if fov < 230: st.error("⚠️ Aliasing Actif !"); msg="Aliasing"
            else: st.success("✅ Image Correcte"); msg=""
            
        with col_visu:
            img_art = final.copy()
            if fov < 230:
                ratio = fov / 230.0
                shift_w = int(S * (1 - ratio) / 2)
                top = img_art[0:shift_w, :]; bot = img_art[S-shift_w:S, :]
                img_art = img_art.copy() 
                img_art[S-shift_w:S, :] += top; img_art[0:shift_w, :] += bot
            
            fig_a = plt.figure(figsize=(5,5))
            ax_a = fig_a.add_subplot(111)
            ax_a.imshow(img_art, cmap='gray', vmin=0, vmax=1.3)
            
            if fov < 230:
                ax_a.axhline(0, color='red', linestyle='--', linewidth=2)
                ax_a.axhline(S, color='red', linestyle='--', linewidth=2)
                style = "Simple, tail_width=0.5, head_width=4, head_length=8"
                kw = dict(arrowstyle=style, color="orange")
                ax_a.add_patch(patches.FancyArrowPatch((S/2, -20), (S/2, 20), connectionstyle="arc3,rad=.5", **kw))
                ax_a.add_patch(patches.FancyArrowPatch((S/2, S+20), (S/2, S-20), connectionstyle="arc3,rad=.5", **kw))
                ax_a.text(S/2, 40, "Aliasing", color='orange', ha='center', fontsize=9, fontweight='bold')
                ax_a.text(S/2, S-40, "Aliasing", color='orange', ha='center', fontsize=9, fontweight='bold')
            
            ax_a.axis('off')
            st.pyplot(fig_a, use_container_width=True)

    # 2. DECALAGE CHIMIQUE
    elif "Décalage Chimique" in artefact_type:
        with col_ctrl:
            with st.expander("ℹ️ Comprendre (Le Contour Noir/Blanc)", expanded=True):
                st.write("La graisse et l'eau ne résonnent pas à la même fréquence. Si la Bande Passante est faible, la machine 'décale' la graisse par erreur.")
            st.info(f"Bande Passante (BW) : **{bw} Hz/px**")
            
            if bw < 100: 
                st.error("⚠️ Décalage Visible !"); shift_v=True
            else: 
                st.success("✅ Décalage négligeable"); shift_v=False
            st.caption("Correction : Augmentez la Bande Passante dans le menu 'Réglages'.")

        with col_visu:
            img_w_local = img_water.copy()
            img_f_local = img_fat.copy()
            if bw == 220: px_shift = 0.0
            else: px_shift = 220.0 / float(bw)
            img_f_shifted = shift(img_f_local, [0, px_shift], mode='constant', cval=0.0)
            img_art = img_w_local + img_f_shifted
            fig_cs = plt.figure(figsize=(5,5))
            ax_cs = fig_cs.add_subplot(111)
            ax_cs.imshow(img_art, cmap='gray', vmin=0, vmax=1.3)
            if px_shift > 1.0:
                ax_cs.text(S/2, 10, f"Décalage : {px_shift:.1f} pixels", color='orange', ha='center')
            ax_cs.axis('off')
            st.pyplot(fig_cs, use_container_width=True)

    # 3. TRONCATURE (GIBBS) - PHANTOME COMPLEXE RESTAURÉ
    elif "Troncature" in artefact_type:
        with col_ctrl:
            with st.expander("ℹ️ Comprendre (Le Dessin Pixelisé)", expanded=True):
                st.write("Une matrice trop faible ne peut pas dessiner les bords nets. Cela crée des 'rebonds' (vagues) autour des contrastes forts.")
            
            sim_matrix = st.select_slider("Matrice Simulée (Locale)", options=[32, 64, 128, 256], value=64)
            
            if sim_matrix <= 64: st.warning("⚠️ Effet Gibbs visible")
            else: st.success("✅ Résolution suffisante")
            
        with col_visu:
            # Génération locale d'un fantôme "COMPLET" haute résolution pour la démo
            # On reprend les 4 zones (LCR, WM, GM, FAT) pour bien voir les stries aux interfaces
            S_hr = 512
            x_hr = np.linspace(-1, 1, S_hr); y_hr = np.linspace(-1, 1, S_hr)
            X_hr, Y_hr = np.meshgrid(x_hr, y_hr); D_hr = np.sqrt(X_hr**2 + Y_hr**2)
            
            img_hr = np.zeros((S_hr, S_hr))
            # Remplissage identique au fantôme principal mais en 512x512
            img_hr[D_hr < 0.20] = v_lcr
            img_hr[(D_hr >= 0.20) & (D_hr < 0.50)] = v_wm
            img_hr[(D_hr >= 0.50) & (D_hr < 0.80)] = v_gm
            img_hr[(D_hr >= 0.80) & (D_hr < 0.95)] = v_fat
            
            k_space_hr = np.fft.fftshift(np.fft.fft2(img_hr))
            
            # Troncature selon le slider local
            center = S_hr // 2
            keep = sim_matrix // 2 
            mask = np.zeros_like(k_space_hr)
            mask[center-keep:center+keep, center-keep:center+keep] = 1
            k_space_truncated = k_space_hr * mask
            
            img_art = np.abs(np.fft.ifft2(np.fft.ifftshift(k_space_truncated)))
            
            fig_g = plt.figure(figsize=(5,5))
            ax_g = fig_g.add_subplot(111)
            ax_g.imshow(img_art, cmap='gray', vmin=0, vmax=1.3)
            ax_g.axis('off')
            st.pyplot(fig_g, use_container_width=True)

    # 4. MOUVEMENT
    elif "Mouvement" in artefact_type:
        with col_ctrl:
            with st.expander("ℹ️ Comprendre (La Photo Floue)", expanded=True):
                st.write("Si le patient bouge pendant l'acquisition, l'image bave dans le sens de la lecture ligne par ligne (Phase).")
            intensite = st.slider("Intensité Mouvement", 0.0, 5.0, 0.0, 0.5)
            
        with col_visu:
            k_mv = np.fft.fftshift(np.fft.fft2(final))
            if intensite > 0:
                phase_err = np.random.normal(0, intensite, S)
                for i in range(S): k_mv[i, :] *= np.exp(1j * phase_err[i])
            img_art = np.abs(np.fft.ifft2(np.fft.ifftshift(k_mv)))
            
            fig_m = plt.figure(figsize=(5,5))
            ax_m = fig_m.add_subplot(111)
            ax_m.imshow(img_art, cmap='gray', vmin=0, vmax=1.3)
            ax_m.arrow(10, S/2, 0, S/3, head_width=10, fc='red', ec='red')
            ax_m.text(20, S/2, "Sens Phase", color='red', rotation=90, fontweight='bold')
            ax_m.axis('off')
            st.pyplot(fig_m, use_container_width=True)

    # 5. ZIPPER
    elif "Zipper" in artefact_type:
        with col_ctrl:
            with st.expander("ℹ️ Comprendre (La Radio qui Grésille)", expanded=True):
                st.write("Une onde radio parasite externe crée une ligne pointillée sur l'image. Sa position dépend de la fréquence du parasite.")
            freq_p = st.slider("Fréquence Parasite (Position X)", 0, S-1, S//2, 1)
            int_rf = st.slider("Volume du Bruit", 0, 100, 0, 10)
            
        with col_visu:
            k_zp = np.fft.fftshift(np.fft.fft2(final))
            if int_rf > 0:
                noise = np.random.normal(0, int_rf, S) + (int_rf * 5)
                alt = np.array([1 if i%2==0 else -1 for i in range(S)])
                k_zp[:, freq_p] += noise * alt * 50
            img_art = np.abs(np.fft.ifft2(np.fft.ifftshift(k_zp)))
            
            fig_z = plt.figure(figsize=(5,5))
            ax_z = fig_z.add_subplot(111)
            ax_z.imshow(img_art, cmap='gray', vmin=0, vmax=1.3)
            if int_rf > 0:
                ax_z.text(freq_p, 10, "Zipper", color='red', ha='center', fontweight='bold')
            ax_z.axis('off')
            st.pyplot(fig_z, use_container_width=True)