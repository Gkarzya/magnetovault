import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit.components.v1 as components
import os
from scipy.ndimage import gaussian_filter

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Magnetovault V2.64 - Final Fusion")

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

# --- 3. CONSTANTES PHYSIQUES (V243 EXACT) ---
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
defaults = {'tr': 500.0, 'te': 15.0, 'ti': 0.0}
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
fov = st.sidebar.slider("FOV", 100.0, 500.0, 240.0, step=10.0, key=f"fov_{current_reset_id}")
mat = st.sidebar.select_slider("Matrice", options=[64, 128, 256, 512], value=256, key=f"mat_{current_reset_id}")
ep = st.sidebar.slider("Epaisseur", 1.0, 10.0, 5.0, step=0.5, key=f"ep_{current_reset_id}")

st.sidebar.header("3. Options")
nex = st.sidebar.slider("NEX", 1, 8, 1, key=f"nex_{current_reset_id}")

# --- MISE A JOUR DES SLIDERS POUR LE NOUVEAU CHRONOGRAMME ---
# On garde les noms de variables internes (turbo, es) pour compatibilité
turbo = st.sidebar.slider("Facteur Turbo", 1, 32, 1, key=f"turbo_{current_reset_id}")
es = st.sidebar.slider("TE Mini (Espacement)", 2.5, 20.0, 10.0, step=2.5, key=f"es_{current_reset_id}")

# --- 10. CALCULS GLOBAUX (V243 EXACT) ---
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
s_ref = 0.85 * (1-np.exp(-500/600)) * np.exp(-15/80)
if s_ref < 0.0001: s_ref = 0.0001
r_sig = v_wm / s_ref

base_vol = (240.0/256.0)**2 * 5.0
base_acq = np.sqrt(256.0)
r_vol = vol_factor / base_vol
r_acq = acq_factor / base_acq

snr_val = r_vol * r_acq * r_sig * 100.0
str_snr = f"{snr_val:.1f} %"

# --- 11. GENERATION FANTOME (V243 EXACT) ---
S = mat
x = np.linspace(-1, 1, S); y = np.linspace(-1, 1, S)
X, Y = np.meshgrid(x, y); D = np.sqrt(X**2 + Y**2)
img = np.zeros((S, S))

img[D < 0.20] = v_lcr
img[(D >= 0.20) & (D < 0.50)] = v_wm
img[(D >= 0.50) & (D < 0.80)] = v_gm
img[(D >= 0.80) & (D < 0.95)] = v_fat

noise_scale = 5.0 / (snr_val + 0.1)
if noise_scale < 0: noise_scale = 0.1
noise = np.random.normal(0, noise_scale, (S,S))
final = np.clip(img + noise, 0, 1.3)

f = np.fft.fftshift(np.fft.fft2(final))
kspace = 20 * np.log(np.abs(f) + 1)

# --- 12. MOTEUR ANATOMIQUE (V243 EXACT) ---
class AdvancedMRIProcessor:
    def __init__(self):
        self.ready = False
        if HAS_NILEARN:
            try:
                self._load_data()
                self.ready = True
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
st.title("Simulateur MagnétoVault V2.64")
t1, t2, t3, t4, t5, t6, t7 = st.tabs(["Fantôme", "Espace K 🌀", "Signaux", "Codage", "🧠 Anatomie", "📈 Physique", "⚡ Chronogramme"])

# TAB 1 (V243)
with t1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("⏱️ Temps d'Acquisition")
        st.latex(r"TA = \frac{TR \times Mat \times NEX}{FacteurTurbo}")
        st.code(f"({tr:.0f} * {mat} * {nex}) / ({turbo} * 1000) = {final_seconds:.1f} s")
        st.success(f"👉 **DURÉE FINALE : {str_duree}**")
        st.divider()
        st.subheader("📉 Rapport Signal/Bruit (SNR)")
        st.latex(r"SNR \propto V_{vox} \times \sqrt{Acq} \times Signal")
        st.table(pd.DataFrame({"Facteur": ["Volume", "Acquis.", "Signal"], "Impact": [f"x {r_vol:.2f}", f"x {r_acq:.2f}", f"x {r_sig:.2f}"]}))
        st.info(f"👉 **SNR RELATIF : {str_snr}**")
    with c2:
        fig_anot, ax_anot = plt.subplots(figsize=(5,5))
        ax_anot.imshow(final, cmap='gray', vmin=0, vmax=1.3)
        ax_anot.axis('off')
        c = S // 2
        radius = S / 2.0
        off_wm = int(radius * 0.35)
        off_gm = int(radius * 0.65)
        off_fat = int(radius * 0.88)
        ax_anot.text(c, c, "WATER", color='cyan', ha='center', weight='bold')
        ax_anot.text(c, c - off_wm, "WM", color='lightgray', ha='center', weight='bold')
        ax_anot.text(c, c - off_gm, "GM", color='white', ha='center', weight='bold')
        ax_anot.text(c, c - off_fat, "FAT", color='orange', ha='center', weight='bold')
        st.pyplot(fig_anot)

# TAB 2 (V243)
with t2:
    st.markdown("### 🌀 Remplissage de l'Espace K : L'origine du Contraste")
    col_k1, col_k2 = st.columns([1, 1])
    with col_k1:
        st.markdown("#### 1. Paramètres de Remplissage")
        fill_mode = st.radio("Ordre de Remplissage", ["Linéaire (Haut -> Bas)", "Centrique (Centre -> Bords)"], key=f"k_mode_{current_reset_id}")
        acq_pct = st.slider("Progression (%)", 0, 100, 10, step=1, key=f"k_pct_{current_reset_id}")
        
        mask_k = np.zeros((S, S))
        lines_to_fill = int(S * (acq_pct / 100.0))
        status_msg = ""
        
        if "Linéaire" in fill_mode:
            mask_k[0:lines_to_fill, :] = 1
            center_line = S // 2
            if lines_to_fill < center_line - 20:
                status_msg = "⏳ Remplissage périphérie HAUTE (Détails seulement)."
                color_msg = "red"
            elif lines_to_fill >= center_line - 20 and lines_to_fill <= center_line + 20:
                status_msg = "⚡ PASSAGE AU CENTRE ! Le contraste arrive."
                color_msg = "green"
            else:
                status_msg = "✅ Finition des détails."
                color_msg = "blue"
        else: 
            center_line = S // 2
            half_width = lines_to_fill // 2
            start = max(0, center_line - half_width)
            end = min(S, center_line + half_width)
            mask_k[start:end, :] = 1
            if acq_pct < 10:
                status_msg = "⚡ DÉBUT : Centre acquis immédiatement !"
                color_msg = "green"
            else:
                status_msg = "✅ Ajout des détails fins."
                color_msg = "blue"

        if status_msg: st.markdown(f":{color_msg}[**{status_msg}**]")

    with col_k2:
        kspace_masked = f * mask_k
        img_reconstructed = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_masked)))
        fig_k, ax_k = plt.subplots(figsize=(4, 4))
        ax_k.imshow(20 * np.log(np.abs(kspace_masked) + 1), cmap='inferno')
        ax_k.axis('off')
        st.pyplot(fig_k)
        st.image(img_reconstructed, clamp=True, width=300)

# TAB 3 (V243)
with t3:
    fig, ax = plt.subplots(figsize=(8, 5))
    vals_bar = [v_lcr, v_gm, v_wm, v_fat] 
    noms = ["WATER", "GM", "WM", "FAT"]
    cols = ['cyan', 'dimgray', 'lightgray', 'orange']
    bars = ax.bar(noms, vals_bar, color=cols, edgecolor='black')
    ax.set_ylim(0, 1.3); ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    for r in bars: ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.02, f"{r.get_height():.2f}", ha='center')
    st.pyplot(fig)

# TAB 4 (V243)
with t4:
    h1="<!DOCTYPE html><html><head><style>body{margin:0;padding:5px;font-family:sans-serif;} .box{display:flex;gap:15px;} .ctrl{width:220px;padding:10px;background:#f9f9f9;border:1px solid #ccc;border-radius:8px;} canvas{border:1px solid #ccc;background:#f8f9fa;border-radius:8px;} input{width:100%;} label{font-size:11px;font-weight:bold;display:block;} button{width:100%;padding:8px;background:#4f46e5;color:white;border:none;border-radius:4px;cursor:pointer;}</style></head><body><div class='box'><div class='ctrl'><h4>Codage</h4><label>Freq</label><input type='range' id='f' min='-100' max='100' value='0'><br><label>Phase</label><input type='range' id='p' min='-100' max='100' value='0'><br><label>Coupe</label><input type='range' id='z' min='-100' max='100' value='0'><br><label>Matrice</label><input type='range' id='g' min='5' max='20' value='12'><br><button onclick='rst()'>Reset</button></div><div><canvas id='c1' width='350' height='350'></canvas><canvas id='c2' width='80' height='350'></canvas></div></div>"
    h2="<script>const c1=document.getElementById('c1');const x=c1.getContext('2d');const c2=document.getElementById('c2');const z=c2.getContext('2d');const sf=document.getElementById('f');const sp=document.getElementById('p');const sz=document.getElementById('z');const sg=document.getElementById('g');const pd=30;function arrow(ctx,x,y,a,s){const l=s*0.35;ctx.save();ctx.translate(x,y);ctx.rotate(a);ctx.beginPath();ctx.moveTo(-l,0);ctx.lineTo(l,0);ctx.lineTo(l-6,-6);ctx.moveTo(l,0);ctx.lineTo(l-6,6);ctx.strokeStyle='white';ctx.lineWidth=1.5;ctx.stroke();ctx.restore();} function draw(){x.clearRect(0,0,350,350);z.clearRect(0,0,80,350);const fv=parseFloat(sf.value);const pv=parseFloat(sp.value);const zv=parseFloat(sz.value);const gs=parseInt(sg.value);const st=(350-2*pd)/gs;"
    h3="const h=(pd*0.8)*(fv/100);x.fillStyle='rgba(255,0,0,0.3)';if(fv!=0){x.beginPath();x.moveTo(pd,pd/2);x.lineTo(pd,pd/2-h);x.lineTo(350-pd,pd/2+h);x.lineTo(350-pd,pd/2);x.fill();}const w=(pd*0.8)*(pv/100);x.fillStyle='rgba(0,255,0,0.3)';if(pv!=0){x.beginPath();x.moveTo(350-pd/2,pd);x.lineTo(350-pd/2-w,pd);x.lineTo(350-pd/2+w,350-pd);x.lineTo(350-pd/2,350-pd);x.fill();} for(let i=0;i<gs;i++){for(let j=0;j<gs;j++){const cx=pd+i*st+st/2;const cy=pd+j*st+st/2;const ph=(i-gs/2)*(fv/100)*3+(j-gs/2)*(pv/100)*3;const cph=(j-gs/2)*(pv/100);x.strokeStyle='black';x.beginPath();x.arc(cx,cy,st*0.4,0,6.28);x.fillStyle='#94a3b8';x.fill();if(cph>0.01)x.fillStyle='rgba(255,255,0,0.5)';if(cph<-0.01)x.fillStyle='rgba(0,0,255,0.5)';x.fill();arrow(x,cx,cy,ph,st*0.6);}}"
    h4="const yz=175-(zv/100)*150;const gr=z.createLinearGradient(0,0,0,350);gr.addColorStop(0,'red');gr.addColorStop(1,'blue');z.fillStyle=gr;z.fillRect(10,10,20,330);z.strokeStyle='black';z.lineWidth=3;z.beginPath();z.moveTo(10,yz);z.lineTo(70,yz);z.stroke();z.fillStyle='black';z.fillText('Z',35,yz-5);} [sf,sp,sz,sg].forEach(s=>s.addEventListener('input',draw));function rst(){sf.value=0;sp.value=0;sz.value=0;sg.value=12;draw();}draw();</script></body></html>"
    components.html(h1+h2+h3+h4, height=450)

# TAB 5 (V243)
with t5:
    st.header("Exploration Anatomique")
    if HAS_NILEARN and processor.ready:
        c1, c2 = st.columns([1, 3])
        dims = processor.get_dims()
        with c1:
            plane = st.radio("Plan de Coupe", ["Plan Axial", "Plan Sagittal", "Plan Coronal"], key="orient_v214")
            if "Axial" in plane: idx = st.slider("Z", 0, dims[2]-1, 90, key=f"slice_{current_reset_id}"); ax='z'
            elif "Sagittal" in plane: idx = st.slider("X", 0, dims[0]-1, 90, key=f"slice_{current_reset_id}"); ax='x'
            else: idx = st.slider("Y", 0, dims[1]-1, 100, key=f"slice_{current_reset_id}"); ax='y'
            st.divider()
            window = st.slider("Fenêtre", 0.01, 2.0, 1.0, 0.005, key=f"win_{current_reset_id}")
            level = st.slider("Niveau", 0.0, 1.0, 0.5, 0.005, key=f"lev_{current_reset_id}")
        with c2:
            w_vals = {'csf':v_lcr, 'gm':v_gm, 'wm':v_wm, 'fat':v_fat}
            img_raw = processor.get_slice(ax, idx, w_vals)
            if img_raw is not None:
                img_wl = apply_window_level(img_raw, window, level)
                st.image(img_wl, clamp=True, width=600, caption=f"MNI152 - {plane}")
            else: st.error("Erreur image.")
    else:
        st.warning("Module 'nilearn' manquant.")

# TAB 6 (V243)
with t6:
    st.header("📈 Physique : Courbes de Relaxation")
    tists = [T_FAT, T_WM, T_GM, T_LCR]
    cols = ['orange', 'lightgray', 'dimgray', 'cyan'] 
    
    st.subheader(f"1. Relaxation Longitudinale (Mz)")
    fig_t1 = plt.figure(figsize=(10, 4))
    gs = fig_t1.add_gridspec(1, 2, width_ratios=[30, 1], wspace=0.05)
    ax_t1 = fig_t1.add_subplot(gs[0])
    ax_bar = fig_t1.add_subplot(gs[1])
    
    if is_ir:
        x_t = np.linspace(0, 4000, 500)
        ax_t1.set_ylim(-1.1, 1.1)
        ax_t1.axhline(y=0, color='black', linewidth=1)
        for t, col in zip(tists, cols):
            mz = 1 - 2 * np.exp(-x_t / t['T1'])
            ax_t1.plot(x_t, mz, label=t['Label'], color=col, linewidth=2)
        ax_t1.axvline(x=ti, color='green', linestyle='--', linewidth=2, label=f'TI')
        st.write(f"**Séquence Inversion-Récupération** : TI = {ti:.0f} ms")
        gradient = np.abs(np.linspace(1, -1, 256)).reshape(-1, 1)
        ax_bar.imshow(gradient, aspect='auto', cmap='gray', extent=[0, 1, -1.1, 1.1])
    else:
        x_t = np.linspace(0, 4000, 500)
        ax_t1.set_ylim(0, 1.1)
        for t, col in zip(tists, cols):
            mz = 1 - np.exp(-x_t / t['T1'])
            ax_t1.plot(x_t, mz, label=t['Label'], color=col, linewidth=2)
        ax_t1.axvline(x=tr, color='red', linestyle='--', linewidth=2, label=f'TR')
        gradient = np.linspace(1, 0, 256).reshape(-1, 1)
        ax_bar.imshow(gradient, aspect='auto', cmap='gray', extent=[0, 1, 0, 1.1])

    ax_t1.set_xlabel("Temps (ms)")
    ax_t1.legend(loc='lower right')
    ax_t1.grid(True, alpha=0.3)
    ax_bar.set_axis_off(); ax_bar.set_title("Signal")
    st.pyplot(fig_t1)
    
    st.subheader(f"2. Relaxation Transversale (Mxy) - T2")
    fig_t2 = plt.figure(figsize=(10, 4))
    gs2 = fig_t2.add_gridspec(1, 2, width_ratios=[30, 1], wspace=0.05)
    ax_t2 = fig_t2.add_subplot(gs2[0])
    ax_bar2 = fig_t2.add_subplot(gs2[1])
    x_te = np.linspace(0, 500, 300)
    
    for t, col in zip(tists, cols):
        mxy = np.exp(-x_te / t['T2'])
        ax_t2.plot(x_te, mxy, label=t['Label'], color=col, linewidth=2)
    ax_t2.axvline(x=te, color='red', linestyle='--', linewidth=2, label=f'TE Effectif')
    ax_t2.set_xlabel("Temps (ms)")
    ax_t2.legend()
    ax_t2.grid(True, alpha=0.3)
    gradient_t2 = np.linspace(1, 0, 256).reshape(-1, 1)
    ax_bar2.imshow(gradient_t2, aspect='auto', cmap='gray', extent=[0, 1, 0, 1.0])
    ax_bar2.set_axis_off(); ax_bar2.set_title("Signal")
    st.pyplot(fig_t2)

# --- TAB 7 : CHRONOGRAMME ICONOGRAPHIQUE (NOUVEAU) ---
with t7:
    is_turbo = turbo > 1
    t = np.linspace(0, 200, 1000)
    
    # Timing
    t_90 = 10
    echo_times = [t_90 + (i+1)*es for i in range(turbo)]
    target_te_graph = te + 10 
    closest_idx = np.argmin(np.abs(np.array(echo_times) - target_te_graph))
    max_dist = max(closest_idx, (turbo-1) - closest_idx) if turbo > 1 else 1
    if max_dist == 0: max_dist = 1

    # Largeurs Visuelles Fixes et Nettes (Style Image Médicale)
    rf_sigma = 0.5 
    grad_width = max(1.5, es * 0.2)
    
    st.header(f"⚡ Séquence : {'Turbo ' if is_turbo else ''}Spin Écho (Facteur {turbo})")

    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(10, 8), gridspec_kw={'hspace': 0.1})
    
    # 1. RF (Ondes Gaussiennes Nettes)
    rf = np.zeros_like(t)
    def add_rf_pulse(center, amp, w): return amp * np.exp(-0.5 * ((t - center)**2) / (w**2))
    
    rf += add_rf_pulse(t_90, 1.0, rf_sigma) # 90
    t_180s = []
    for i in range(turbo):
        t_p = t_90 + (i * es) + (es/2)
        if t_p < 200:
            t_180s.append(t_p)
            rf += add_rf_pulse(t_p, 1.6, rf_sigma)
            
    axs[0].plot(t, rf, color='black', linewidth=1.5)
    axs[0].fill_between(t, 0, rf, color='green', alpha=0.4)
    axs[0].set_ylabel("RF"); axs[0].set_yticks([0, 1, 1.6], ["", "90", "180"])
    axs[0].spines['top'].set_visible(False); axs[0].spines['right'].set_visible(False)
    axs[0].set_xlim(0, 200)

    # 2. Gsc (Slice) - Trapèzes
    gsc = np.zeros_like(t)
    def add_trap(center, amp, w):
        mask = (t > center - w) & (t < center + w)
        gsc[mask] = amp
    
    add_trap(t_90, 1.0, grad_width)
    t_rephase = t_90 + grad_width + 1.5
    add_trap(t_rephase, -0.8, grad_width*0.6) # Rephasage
    for t_p in t_180s: add_trap(t_p, 1.0, grad_width)
    
    axs[1].fill_between(t, 0, gsc, color='green', alpha=0.6)
    axs[1].plot(t, gsc, color='green', linewidth=1)
    axs[1].set_ylabel("Gsc"); axs[1].set_yticks([])
    axs[1].spines['top'].set_visible(False); axs[1].spines['right'].set_visible(False)

    # 3. Gcp (Phase) - V Shape
    gcp = np.zeros_like(t)
    for i in range(turbo):
        t_180 = t_90 + (i * es) + (es/2)
        t_echo = t_90 + ((i+1) * es)
        t_code = (t_180 + t_echo)/2 - (es*0.1)
        t_rewind = t_echo + (es*0.15)
        
        if t_rewind < 200:
            if i == closest_idx: height = 0.2; label = "BF"; col_lbl = "red"
            else: 
                dist = abs(i - closest_idx)
                height = 0.2 + (0.8 * (dist / max_dist))
                label = ""; col_lbl = "gray"
            
            w_ph = grad_width * 0.7
            mask_c = (t > t_code - w_ph) & (t < t_code + w_ph)
            gcp[mask_c] = height
            if label == "BF": axs[2].text(t_code, height+0.1, label, color=col_lbl, ha='center', fontsize=9, weight='bold')
            
            mask_r = (t > t_rewind - w_ph) & (t < t_rewind + w_ph)
            gcp[mask_r] = -height

    axs[2].fill_between(t, 0, gcp, color='darkorange', alpha=0.7)
    axs[2].set_ylabel("Gcp"); axs[2].set_yticks([])
    axs[2].spines['top'].set_visible(False); axs[2].spines['right'].set_visible(False)

    # 4. Gcf (Freq)
    gcf = np.zeros_like(t)
    t_pre = (t_90 + (t_90 + es/2))/2
    add_trap_gcf = lambda c, w: ((t > c - w) & (t < c + w))
    gcf[add_trap_gcf(t_pre, grad_width)] = 1.0 
    
    for i in range(turbo):
        t_e = echo_times[i]
        if t_e < 200:
            w_read = grad_width * 1.2
            gcf[add_trap_gcf(t_e, w_read)] = 1.0
            
    axs[3].fill_between(t, 0, gcf, color='dodgerblue', alpha=0.5)
    axs[3].set_ylabel("Gcf"); axs[3].set_yticks([])
    axs[3].spines['top'].set_visible(False); axs[3].spines['right'].set_visible(False)

    # 5. Signal
    sig = np.zeros_like(t)
    for i in range(turbo):
        t_e = echo_times[i]
        if t_e < 195:
            w_sig = grad_width * 1.2
            idx_start = np.argmin(np.abs(t - (t_e - w_sig)))
            idx_end = np.argmin(np.abs(t - (t_e + w_sig)))
            if idx_end > idx_start:
                grid = np.linspace(-3, 3, idx_end - idx_start)
                amp = np.exp(-(i*es)/T_GM['T2']) 
                sig[idx_start:idx_end] = np.sinc(grid) * amp
            if i == closest_idx:
                 axs[4].text(t_e, amp+0.2, "TE eff", ha='center', color='red', fontweight='bold', fontsize=10)
                 axs[4].axvline(x=t_e, color='red', linestyle='--', alpha=0.5)

    axs[4].plot(t, sig, color='navy', linewidth=1.5)
    axs[4].set_ylabel("Signal"); axs[4].set_xlabel("Temps (ms)")
    axs[4].set_yticks([]); axs[4].spines['top'].set_visible(False); axs[4].spines['right'].set_visible(False)
    
    st.pyplot(fig)