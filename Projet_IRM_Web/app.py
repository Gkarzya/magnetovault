import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit.components.v1 as components
import os
from scipy.ndimage import gaussian_filter, shift, sobel, zoom, binary_erosion, binary_dilation

# --- 1. CONFIGURATION ET CSS ---
# VERSION 7.42 - CORRECTION T2 (TR 4000 / TE 85) + FIX PERSISTANCE TE
st.set_page_config(layout="wide", page_title="Magnetovault V7.42 - Fixed T2")

st.markdown("""
    <style>
        /* CSS RENFORCÉ POUR LE SLIDER HORIZONTAL DES ONGLETS */
        div[data-baseweb="tab-list"] {
            display: flex !important;
            flex-wrap: nowrap !important;
            overflow-x: auto !important;
            white-space: nowrap !important;
            gap: 8px;
            padding-bottom: 8px;
            width: 100%;
            scrollbar-width: thin;
            scrollbar-color: #4f46e5 #e0e7ff;
        }
        div[data-baseweb="tab"] {
            flex: 0 0 auto !important;
            min-width: fit-content !important;
        }
        div[data-baseweb="tab-list"]::-webkit-scrollbar { height: 10px !important; display: block !important; }
        div[data-baseweb="tab-list"]::-webkit-scrollbar-track { background: #e0e7ff !important; border-radius: 4px; }
        div[data-baseweb="tab-list"]::-webkit-scrollbar-thumb { background-color: #4f46e5 !important; border-radius: 10px; border: 2px solid #e0e7ff; }
        div[data-baseweb="tab-list"]::-webkit-scrollbar-thumb:hover { background-color: #4338ca !important; }

        /* ALERTE ROUGE (TR FORCE) */
        .tr-alert-box {
            background-color: #fee2e2;
            border-left: 5px solid #ef4444;
            padding: 15px;
            border-radius: 5px;
            color: #7f1d1d;
            font-weight: bold;
            margin-bottom: 15px;
            margin-top: 5px;
        }
        /* ENCART BLEU (OPTIMISATION) */
        .opt-box {
            background-color: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 5px;
            padding: 10px;
            text-align: center;
            margin-bottom: 5px;
            color: #1e3a8a;
        }
    </style>
""", unsafe_allow_html=True)

# Imports Médicaux
try:
    import nibabel as nib
    from nilearn import datasets
    HAS_NILEARN = True
except ImportError:
    HAS_NILEARN = False

# --- 2. FONCTIONS UTILITAIRES ---
def safe_rerun():
    try: st.rerun()
    except AttributeError: st.experimental_rerun()

def apply_window_level(image, window, level):
    win = max(0.001, window)
    vmin, vmax = level - win/2, level + win/2
    return np.clip((image - vmin)/(vmax - vmin), 0, 1)

# CALLBACKS
def set_tr_value(val):
    st.session_state.tr_force = val
    st.session_state.widget_tr = val

def set_optimized_tr(val):
    st.session_state.tr_force = val
    st.session_state.widget_tr = val

def update_tr_from_slider():
    st.session_state.tr_force = st.session_state.widget_tr

# --- 3. CONSTANTES PHYSIQUES ---
T_FAT = {'T1': 260.0, 'T2': 80.0, 'T2s': 30.0, 'ADC': 0.0, 'PD': 1.0, 'Label': 'Graisse'}
T_LCR = {'T1': 3607.0, 'T2': 2000.0, 'T2s': 400.0, 'ADC': 3.0, 'PD': 1.0, 'Label': 'Eau (LCR)'}
T_GM  = {'T1': 1300.0, 'T2': 140.0, 'T2s': 50.0, 'ADC': 0.8, 'PD': 0.95, 'Label': 'Subst. Grise'}
T_WM  = {'T1': 600.0,  'T2': 100.0, 'T2s': 40.0, 'ADC': 0.7, 'PD': 0.70, 'Label': 'Subst. Blanche'}
T_STROKE = {'T1': 900.0, 'T2': 200.0, 'T2s': 80.0, 'ADC': 0.4, 'PD': 0.90, 'Label': 'Ischémie (AVC)'}
T_BLOOD = 1650.0 

# --- 4. STATE MANAGEMENT ---
if 'init' not in st.session_state:
    st.session_state.seq = 'Pondération T1'
    st.session_state.reset_count = 0
    st.session_state.atrophy_active = False 
    st.session_state.tr_force = 500.0
    st.session_state.init = True

# --- 5. BARRE LATÉRALE ---
# Calcul des chemins absolus pour GitHub
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "logo_mia.png")
image_asl_path = os.path.join(current_dir, "image_028fa1.jpg")

# Utilisation du chemin absolu pour le logo
if os.path.exists(logo_path): 
    st.sidebar.image(logo_path, width=280)

st.sidebar.title("Réglages Console")

# --- BOUTON RESET (HARD RELOAD JS) ---
if st.sidebar.button("⚠️ Reset Complet (Rafraîchir)"):
    components.html("<script>window.parent.location.reload();</script>", height=0)

# --- SÉLECTION SÉQUENCE ---
options_seq = [
    "Pondération T1", "Pondération T2", "Écho de Gradient (T2*)", "Diffusion (DWI)", 
    "DP (Densité Protons)", "FLAIR (Eau -)", "Séquence STIR (Graisse)", "SWI (Susceptibilité)",
    "3D T1 (MP-RAGE)", "Perfusion ASL"
]
# La clé contient reset_count, donc le dropdown se reset aussi
seq_key = f"seq_select_{st.session_state.reset_count}"
try: idx_def = options_seq.index(st.session_state.seq)
except: idx_def = 0
seq_choix = st.sidebar.selectbox("Séquence", options_seq, index=idx_def, key=seq_key)

# VALEURS PAR DÉFAUT [MODIFICATION V7.42]
std_params = {
    "Pondération T1": {'tr': 500.0, 'te': 10.0, 'ti': 0.0},
    # ICI : TR passé à 4000.0 pour T2 standard
    "Pondération T2": {'tr': 4000.0, 'te': 85.0, 'ti': 0.0},
    "Écho de Gradient (T2*)": {'tr': 150.0, 'te': 20.0, 'ti': 0.0}, 
    "Diffusion (DWI)": {'tr': 6000.0, 'te': 90.0, 'ti': 0.0},
    "DP (Densité Protons)": {'tr': 2200.0, 'te': 30.0, 'ti': 0.0},
    "FLAIR (Eau -)": {'tr': 9000.0, 'te': 110.0, 'ti': 2500.0},
    "Séquence STIR (Graisse)": {'tr': 3500.0, 'te': 50.0, 'ti': 150.0},
    "SWI (Susceptibilité)": {'tr': 50.0, 'te': 40.0, 'ti': 0.0},
    "3D T1 (MP-RAGE)": {'tr': 2000.0, 'te': 3.0, 'ti': 900.0},
    "Perfusion ASL": {'tr': 4000.0, 'te': 15.0, 'ti': 0.0}
}
defaults = std_params.get(seq_choix, std_params["Pondération T1"])

# --- LOGIQUE DE CHANGEMENT DE SEQUENCE (CORRECTIF PERSISTANCE TE) ---
if seq_choix != st.session_state.seq:
    st.session_state.seq = seq_choix
    
    # 1. Mise à jour du TR
    new_tr = float(defaults['tr'])
    st.session_state.tr_force = new_tr
    if 'widget_tr' in st.session_state: 
        st.session_state.widget_tr = new_tr
    
    # 2. Mise à jour du TE (Force l'écriture dans le session_state)
    new_te = float(defaults['te'])
    te_key_current = f"te_main_{st.session_state.reset_count}"
    st.session_state[te_key_current] = new_te
        
    safe_rerun()

is_gre = "Gradient" in seq_choix
is_dwi = "Diffusion" in seq_choix
is_ir = "FLAIR" in seq_choix or "STIR" in seq_choix
is_swi = "SWI" in seq_choix
is_mprage = "MP-RAGE" in seq_choix
is_asl = "ASL" in seq_choix

# Initialisation
def_tr = defaults['tr']
def_te = defaults['te']
def_ti = defaults['ti']
current_reset_id = st.session_state.reset_count
ti = 0.0
te = float(defaults['te'])
flip_angle = 90

# ==============================================================================
# 1. GÉOMÉTRIE (PRIORITAIRE)
# ==============================================================================
st.sidebar.header("1. Géométrie")
col_ep, col_slice = st.sidebar.columns(2)
with col_ep: ep = st.number_input("Epaisseur (mm)", 1.0, 10.0, 5.0, 0.5, key=f"ep_{current_reset_id}")
with col_slice: n_slices = st.number_input("Nb Coupes", 1, 60, 20, key=f"ns_{current_reset_id}")

if not is_dwi and not is_mprage:
    n_concats = st.sidebar.select_slider("📚 Concaténations", options=[1, 2, 3, 4], value=1, key=f"concat_{current_reset_id}")
else: n_concats = 1

fov = st.sidebar.slider("FOV (mm)", 100.0, 500.0, 240.0, step=10.0, key=f"fov_{current_reset_id}")
mat = st.sidebar.select_slider("Matrice", options=[64, 128, 256, 512], value=256, key=f"mat_{current_reset_id}")

st.sidebar.subheader("Réglage Echo")
if not (is_dwi or is_asl):
    # Le slider TE utilise la clé dynamique pour être réinitialisable
    te = st.sidebar.slider("TE (ms)", 1.0, 300.0, float(defaults['te']), step=1.0, key=f"te_main_{current_reset_id}")
else:
    te = 90.0 if is_dwi else 15.0

# --- CALCUL DU TR AUTOMATIQUE (PUSH) ---
time_per_slice = te + 15.0 
min_tr_required = (n_slices * time_per_slice) / n_concats
current_tr_val = st.session_state.get('widget_tr', st.session_state.tr_force)

auto_adjusted = False
if current_tr_val < min_tr_required:
    st.session_state.tr_force = min_tr_required
    st.session_state.widget_tr = min_tr_required
    auto_adjusted = True
    safe_rerun()

# ==============================================================================
# 2. CHRONO (TR)
# ==============================================================================
st.sidebar.header("2. Chrono (ms)")

b_value = 0; show_stroke = False; show_atrophy = False; show_adc_map = False; show_microbleeds = False; pld = 1500 

if is_dwi:
    b_value = st.sidebar.select_slider("Facteur b", options=[0, 500, 1000], value=0, key=f"bval_{current_reset_id}")
    tr = 6000.0; te = 90.0; ti = 0.0; flip_angle = 90
    st.sidebar.info("TR fixé : 6000ms | TE fixé : 90ms")
    show_stroke = st.sidebar.checkbox("Simuler AVC", False, key=f"avc_{current_reset_id}")
    show_adc_map = st.sidebar.checkbox("Carte ADC", False, key=f"adc_{current_reset_id}")
elif is_asl:
    pld = st.sidebar.slider("PLD", 500, 3000, 1800, step=100, key=f"pld_{current_reset_id}")
    tr = st.sidebar.slider("TR", 3000.0, 8000.0, 4500.0, step=100.0, key=f"tr_asl_{current_reset_id}")
    te = 15.0; ti = 0.0; flip_angle = 90
    show_stroke = st.sidebar.checkbox("AVC", False, key=f"asl_avc_{current_reset_id}")
    st.session_state.atrophy_active = st.sidebar.checkbox("Atrophie", st.session_state.atrophy_active, key=f"asl_atr_{current_reset_id}")
    show_atrophy = st.session_state.atrophy_active
else:
    # --- SLIDER TR ---
    tr = st.sidebar.slider(
        "TR (ms)", 
        min_value=10.0, 
        max_value=12000.0, 
        value=float(st.session_state.tr_force), 
        step=10.0, 
        key="widget_tr", 
        on_change=update_tr_from_slider
    )
    if tr != st.session_state.tr_force:
        st.session_state.tr_force = tr

    if auto_adjusted:
        st.sidebar.markdown(f"""<div class="tr-alert-box">⚠️ TR ajusté auto<br>({int(min_tr_required)}ms) pour {n_slices} coupes.</div>""", unsafe_allow_html=True)
    elif ('T1' in seq_choix and tr > 700 and not is_mprage):
        st.sidebar.markdown("""<div class="tr-alert-box">⚠️ Attention Dépassement T1</div>""", unsafe_allow_html=True)

    if n_concats > 1:
        tr_opti = np.ceil(min_tr_required / 10) * 10
        if tr > (tr_opti + 100):
            st.sidebar.markdown(f"""<div class="opt-box"><b>Optimisation {n_concats} Concats</b><br>TR Min : <b>{int(tr_opti)} ms</b></div>""", unsafe_allow_html=True)
            st.sidebar.button(f"📉 Appliquer TR {int(tr_opti)} ms", on_click=set_optimized_tr, args=(tr_opti,))

    if is_ir or is_mprage: ti = st.sidebar.slider("TI", 0.0, 3500.0, float(defaults['ti']), step=10.0, key=f"ti_{current_reset_id}")
    else: ti = 0.0
    
    if is_gre: flip_angle = st.sidebar.slider("Angle (°)", 5, 90, 15, key=f"fa_{current_reset_id}")
    elif is_swi: flip_angle = st.sidebar.slider("Angle (°)", 5, 40, 15, key=f"fa_{current_reset_id}"); show_microbleeds = st.sidebar.checkbox("Micro-saignements", False, key=f"cmb_{current_reset_id}")
    else: flip_angle = 90 if not is_mprage else 10

st.sidebar.header("3. Options")
nex = st.sidebar.slider("NEX", 1, 8, 1, key=f"nex_{current_reset_id}")
turbo = 1
if not (is_gre or is_dwi or is_swi or is_mprage or is_asl):
    turbo = st.sidebar.slider("Facteur Turbo", 1, 32, 1, key=f"turbo_{current_reset_id}")
bw = st.sidebar.slider("Bande Passante", 50, 500, 220, 10, key=f"bw_{current_reset_id}")
es = st.sidebar.slider("Espace Inter-Echo (ES)", 2.5, 20.0, 10.0, step=2.5, key=f"es_{current_reset_id}")

st.sidebar.header("4. Imagerie Parallèle (iPAT)")
ipat_on = st.sidebar.checkbox("Activer Accélération", value=False, key=f"ipat_on_{current_reset_id}")
ipat_factor = st.sidebar.slider("Facteur R", 2, 4, 2, key=f"ipat_r_{current_reset_id}") if ipat_on else 1

st.sidebar.markdown("---")
with st.sidebar.expander("🛡️ Mentions Légales & Droits"):
    st.markdown("""
    **MagnétoVault Simulator © 2025**
    *Développement & Conception :* **MagnétoVault**
    *📧 Contact :* magnetovault@gmail.com
    **Propriété Intellectuelle :**
    Ce logiciel, incluant ses algorithmes de simulation, son interface et sa logique pédagogique, est protégé par le droit d'auteur.
    ⛔ **Toute reproduction, redistribution ou ingénierie inverse est strictement interdite sans autorisation écrite.**
    *Usage strictement réservé au cadre pédagogique défini par l'auteur.*
    """)

# ==============================================================================
# MOTEUR PHYSIQUE & CALCULS (BASE V700)
# ==============================================================================
t_slice_occupancy = te + 15.0 
if turbo > 1: t_slice_occupancy = (turbo * es) + 15.0

tr_effective = tr 

if is_dwi: raw_ms = tr * nex * 15 
else: raw_ms = (tr_effective * mat * nex * n_concats) / (turbo * ipat_factor)

final_seconds = raw_ms / 1000.0; mins = int(final_seconds // 60); secs = int(final_seconds % 60); str_duree = f"{mins} min {secs} s"

def calculate_signal(tr_val, te_val, ti_val, t1, t2, t2s, adc, pd, fa_deg, gre_mode, dwi_mode, b_val):
    val = 0.0
    if gre_mode:
        rad = np.radians(fa_deg); e1 = np.exp(-tr_val / t1)
        if fa_deg < 40 and t1 > 2000: e1 = e1 * 0.1 
        e2s = np.exp(-te_val / t2s); num = np.sin(rad) * (1 - e1); den = 1 - np.cos(rad) * e1; val = pd * (num / den) * e2s
    else:
        e2 = np.exp(-te_val / t2)
        if ti_val > 10: val = pd * (1 - 2 * np.exp(-ti_val / t1) + np.exp(-tr_val / t1)) * e2
        else: val = pd * (1 - np.exp(-tr_val / t1)) * e2
    if dwi_mode and b_val > 0: diff_decay = np.exp(-b_val * adc * 0.001); val = val * diff_decay
    return np.abs(val)

# CALCUL SIGNAUX
v_lcr = calculate_signal(tr_effective, te, ti, T_LCR['T1'], T_LCR['T2'], T_LCR['T2s'], T_LCR['ADC'], T_LCR['PD'], flip_angle, is_gre, is_dwi, b_value if is_dwi else 0)
v_wm  = calculate_signal(tr_effective, te, ti, T_WM['T1'], T_WM['T2'], T_WM['T2s'], T_WM['ADC'], T_WM['PD'], flip_angle, is_gre, is_dwi, b_value if is_dwi else 0)
v_gm  = calculate_signal(tr_effective, te, ti, T_GM['T1'], T_GM['T2'], T_GM['T2s'], T_GM['ADC'], T_GM['PD'], flip_angle, is_gre, is_dwi, b_value if is_dwi else 0)
v_stroke = calculate_signal(tr_effective, te, ti, T_STROKE['T1'], T_STROKE['T2'], T_STROKE['T2s'], T_STROKE['ADC'], T_STROKE['PD'], flip_angle, is_gre, is_dwi, b_value if is_dwi else 0)

if is_dwi and b_value >= 1000 and show_stroke: v_stroke = 2.0 
v_fat = calculate_signal(tr_effective, te, ti, T_FAT['T1'], T_FAT['T2'], T_FAT['T2s'], T_FAT['ADC'], T_FAT['PD'], flip_angle, is_gre, is_dwi, 0) if not is_dwi else 0.0

# SNR
ref_wm_signal = calculate_signal(def_tr, def_te, def_ti, T_WM['T1'], T_WM['T2'], T_WM['T2s'], T_WM['ADC'], T_WM['PD'], 90, False, False, 0)
if ref_wm_signal < 0.0001: ref_wm_signal = 0.0001 
vol_factor = (fov/float(mat))**2 * ep; turbo_penalty = float(turbo)**0.25 
ipat_penalty = (float(ipat_factor)**0.25) * (1.2 + (0.1 * (ipat_factor - 2))) if ipat_factor > 1 else 1.0
acq_factor = np.sqrt(float(mat)*float(nex)) / (turbo_penalty * ipat_penalty)
bw_factor = np.sqrt(220.0 / float(bw)); r_vol = vol_factor / ((240.0/256.0)**2 * 5.0); r_acq = acq_factor / np.sqrt(256.0); r_sig = v_wm / ref_wm_signal 
snr_val = r_vol * r_acq * bw_factor * r_sig * 100.0; str_snr = f"{snr_val:.1f} %"

# GENERATION FANTOME
S = mat; x = np.linspace(-1, 1, S); y = np.linspace(-1, 1, S); X, Y = np.meshgrid(x, y); D = np.sqrt(X**2 + Y**2)
img_water = np.zeros((S, S)); img_fat = np.zeros((S, S))

val_lcr_phantom = v_lcr
val_wm_phantom = v_wm
val_gm_phantom = v_gm
val_stroke_phantom = v_stroke
val_fat_phantom = v_fat

if is_dwi and show_adc_map:
    val_lcr_phantom = 1.0; val_wm_phantom = 0.3; val_gm_phantom = 0.35; val_stroke_phantom = 0.15; val_fat_phantom = 0.0

img_water[D < 0.20] = val_lcr_phantom
img_water[(D >= 0.20) & (D < 0.50)] = val_wm_phantom
img_water[(D >= 0.50) & (D < 0.80)] = val_gm_phantom
img_fat[(D >= 0.80) & (D < 0.95)] = val_fat_phantom

if show_stroke: 
    lesion_mask = (np.sqrt((X-0.3)**2 + (Y-0.1)**2) < 0.12)
    mask_valid = lesion_mask & (D >= 0.20)
    img_water[mask_valid] = val_stroke_phantom

shift_pixels = 0.0 if bw == 220 else 220.0 / float(bw); img_fat_shifted = shift(img_fat, [0, shift_pixels], mode='constant', cval=0.0)
final = np.clip(img_water + img_fat_shifted, 0, 1.3); noise_level = 5.0 / (snr_val + 20.0) 
final += np.random.normal(0, noise_level, (S,S)); final = np.clip(final, 0, 1.3)
f = np.fft.fftshift(np.fft.fft2(final)); kspace = 20 * np.log(np.abs(f) + 1)

# ANATOMIE (PROCESSOR)
class AdvancedMRIProcessor:
    def __init__(self):
        self.ready = False
        self.static_vein_mask = None 
        if HAS_NILEARN:
            try: self._load_data(); self.ready = True
            except: pass
        self._generate_organic_veins()
    
    @st.cache_resource
    def _load_data(_self):
        dataset = datasets.fetch_icbm152_2009(); gm = nib.load(dataset['gm']).get_fdata(); wm = nib.load(dataset['wm']).get_fdata(); csf = nib.load(dataset['csf']).get_fdata(); rest = np.clip(1.0 - (gm + wm + csf), 0, 1); return gm, wm, csf, rest

    def _generate_organic_veins(self):
        shape = (200, 200); np.random.seed(888)
        noise = np.random.normal(0, 1, shape)
        smooth = gaussian_filter(noise, sigma=2.5)
        sx = sobel(smooth, axis=0); sy = sobel(smooth, axis=1); magnitude = np.hypot(sx, sy)
        threshold = np.percentile(magnitude, 94)
        mask = (magnitude > threshold).astype(float)
        self.static_vein_mask = gaussian_filter(mask, sigma=0.5)

    def get_vein_mask_slice(self, shape, z_index):
        if self.static_vein_mask is None: return np.zeros(shape)
        zoom_factor = (shape[0] / self.static_vein_mask.shape[0], shape[1] / self.static_vein_mask.shape[1])
        final_mask = zoom(self.static_vein_mask, zoom_factor, order=1) 
        return final_mask

    def get_adc_map(self, s_gm, s_wm, s_csf): return (s_csf * 3.0) + (s_gm * 0.8) + (s_wm * 0.7)
    def get_t2s_map(self, s_gm, s_wm, s_csf): return (s_csf * 400.0) + (s_gm * 50.0) + (s_wm * 40.0)
    def get_t1_map(self, s_gm, s_wm, s_csf): return (s_csf * 3500.0) + (s_gm * 1200.0) + (s_wm * 700.0)

    def create_lesion_mask(self, shape, center, radius):
        Y, X = np.ogrid[:shape[0], :shape[1]]
        dist = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
        return dist <= radius

    def inject_dipole_artifact(self, image, axis, contrast_sign):
        cy, cx = image.shape[0] // 2, image.shape[1] // 2
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        r = image.shape[0] * 0.03 
        if axis == 'z':
            mask_center = ((x - cx)**2 + (y - cy)**2) < r**2
            mask_halo = (((x - cx)**2 + (y - cy)**2) < (r*2)**2) & ~mask_center
            image[mask_center] = 0.5 + (0.4 * contrast_sign); image[mask_halo] = 0.5 - (0.1 * contrast_sign)
        else: 
            width = r * 3.0; height = r * 0.5
            mask_rect = (np.abs(x - cx) < width) & (np.abs(y - cy) < height)
            offset_y = r * 2.0; radius_x = r * 1.5; radius_y = r * 1.0
            mask_lobe_up = (((x - cx)**2) / radius_x**2 + ((y - (cy - offset_y))**2) / radius_y**2) <= 1; mask_lobe_down = (((x - cx)**2) / radius_x**2 + ((y - (cy + offset_y))**2) / radius_y**2) <= 1
            image[mask_rect] = 0.5 + (0.4 * contrast_sign)
            image[mask_lobe_up] = 0.5 - (0.35 * contrast_sign); image[mask_lobe_down] = 0.5 - (0.35 * contrast_sign)
        return np.clip(image, 0, 1)

    def get_phase_map(self, s_gm, s_wm, s_csf, vein_mask, contrast_sign, with_bleeds=False, inject_dipole=False, axis='z'):
        np.random.seed(42); noise_raw = np.random.normal(0, 1, s_gm.shape)
        inhomogeneity = gaussian_filter(noise_raw, sigma=25) * 0.12
        base_struct = (s_gm * 0.03) - (s_wm * 0.01)
        phase = 0.5 + base_struct + inhomogeneity
        vein_intensity = 0.35 * contrast_sign
        phase = phase + (vein_mask * vein_intensity)
        if with_bleeds:
            np.random.seed(100); bleeds = np.random.choice([0, 1], size=s_gm.shape, p=[0.998, 0.002])
            bleeds = gaussian_filter(bleeds.astype(float), sigma=0.5)
            phase = phase + (bleeds * vein_intensity * 2.0)
        if inject_dipole: phase = self.inject_dipole_artifact(phase, axis, contrast_sign)
        return np.clip(phase, 0, 1)

    def apply_focal_atrophy(self, g, w, c):
        cy, cx = g.shape[0]//2, g.shape[1]//2
        Y, X = np.ogrid[:g.shape[0], :g.shape[1]]
        angle = np.arctan2(Y - cy, X - cx)
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        angle_mask = (angle > 0.2 * np.pi) & (angle < 0.55 * np.pi)
        dist_mask = (dist > 20) & (dist < 85)
        base_wedge = angle_mask & dist_mask
        smooth_wedge = gaussian_filter(base_wedge.astype(float), sigma=5.0)
        curvilinear_mask = smooth_wedge > 0.4 
        brain_tissue = (g > 0.1) | (w > 0.1)
        final_roi_mask = curvilinear_mask & brain_tissue
        np.random.seed(999); noise = np.random.normal(0, 1, g.shape)
        large_blobs = gaussian_filter(noise, sigma=4.5) 
        atrophy_holes = (large_blobs > 0.9) & final_roi_mask
        g[atrophy_holes] = 0.0; w[atrophy_holes] = 0.0; c[atrophy_holes] = 1.0
        return g, w, c, final_roi_mask

    def get_asl_maps(self, axis, index, pld, label_dur, with_stroke=False, with_atrophy=False):
        if not self.ready: return None, None, None
        gm, wm, csf, rest = self._load_data()
        ax_idx = {'x': 0, 'y': 1, 'z': 2}.get(axis, 2)
        idx_safe = max(0, min(index, gm.shape[ax_idx]-1))
        g = np.take(gm, idx_safe, axis=ax_idx).T; w = np.take(wm, idx_safe, axis=ax_idx).T; c = np.take(csf, idx_safe, axis=ax_idx).T
        if axis == 'z': g = np.flipud(g); w = np.flipud(w); c = np.flipud(c)
        elif axis == 'x': g = np.flipud(np.fliplr(g)); w = np.flipud(np.fliplr(w)); c = np.flipud(np.fliplr(c))
        elif axis == 'y': g = np.flipud(g); w = np.flipud(w); c = np.flipud(c)
        mask_atrophy = None
        if with_atrophy: g, w, c, mask_atrophy = self.apply_focal_atrophy(g, w, c)
        cbf = (g * 60.0) + (w * 20.0) + np.random.normal(0, 2, g.shape)
        if with_stroke:
            cy, cx = g.shape[0]//2, g.shape[1]//2; Y, X = np.ogrid[:g.shape[0], :g.shape[1]]
            angle = np.arctan2(Y - cy, X - cx); dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            stroke_mask = (dist > 15) & (dist < 85) & (np.abs(angle) > 2.2) 
            brain_core = binary_erosion((g + w + c) > 0.1, iterations=5); stroke_mask = stroke_mask & brain_core
            stroke_mask = gaussian_filter(stroke_mask.astype(float), sigma=1.5) > 0.5; cbf[stroke_mask] = 5.0
        if with_atrophy and mask_atrophy is not None: cbf[mask_atrophy] = cbf[mask_atrophy] * 0.3 
        att_map = (g * 1200.0) + (w * 1500.0) + (c * 2000.0) + 100.0
        decay = np.exp(-pld / T_BLOOD); arrival = np.clip((pld - att_map) / 500.0, 0, 1)
        asl_signal_visible = cbf * decay * arrival * 0.015 
        noise = np.random.normal(0, 0.01, g.shape)
        control_img = (g * 0.8) + (w * 0.6) + (c * 0.2) + noise
        label_img = control_img - asl_signal_visible
        perf_map_calc = control_img - label_img
        return control_img, label_img, perf_map_calc
    def get_slice(self, axis, index, w_vals, seq_type=None, te=0, tr=500, fa=90, b_val=0, adc_mode=False, with_stroke=False, swi_mode=None, with_bleeds=False, swi_sys="RHS", swi_sub="Hématome", with_dipole=False):
        if not self.ready: return None
        gm, wm, csf, rest = self._load_data(); ax_idx = {'x': 0, 'y': 1, 'z': 2}.get(axis, 2)
        idx_safe = max(0, min(index, gm.shape[ax_idx]-1)); g = np.take(gm, idx_safe, axis=ax_idx).T; w = np.take(wm, idx_safe, axis=ax_idx).T; c = np.take(csf, idx_safe, axis=ax_idx).T; r = np.take(rest, idx_safe, axis=ax_idx).T
        if axis == 'z': g = np.flipud(g); w = np.flipud(w); c = np.flipud(c); r = np.flipud(r)
        elif axis == 'x': g = np.flipud(np.fliplr(g)); w = np.flipud(np.fliplr(w)); c = np.flipud(np.fliplr(c)); r = np.flipud(np.fliplr(r))
        elif axis == 'y': g = np.flipud(g); w = np.flipud(w); c = np.flipud(c); r = np.flipud(r)
        if st.session_state.atrophy_active and axis == 'z': g, w, c, _ = self.apply_focal_atrophy(g, w, c)
        stroke_mask = None
        if with_stroke and axis == 'z' and (85 < index < 95): stroke_mask = self.create_lesion_mask(g.shape, center=(110, 70), radius=9)
        if swi_mode:
            vein_mask = self.get_vein_mask_slice(g.shape, index) if axis == 'z' else self.get_vein_mask_slice((g.shape[0], g.shape[1]), 0) 
            vein_mask = vein_mask * ((g + w + c) > 0.1); is_rhs = "RHS" in swi_sys; is_para = "Hématome" in swi_sub; contrast_sign = (1 if is_para else -1) * (1 if is_rhs else -1) 
            if swi_mode == 'phase': sim = self.get_phase_map(g, w, c, vein_mask, contrast_sign, with_bleeds, inject_dipole=with_dipole, axis=axis)
            elif swi_mode == 'mag':
                t2s = self.get_t2s_map(g, w, c); mag = (c + g*0.8 + w*0.7) * np.exp(-te / (t2s + 1e-6)); mag = mag * (1 - vein_mask * 0.5) 
                if with_bleeds: bleeds = gaussian_filter(np.random.choice([0, 1], size=g.shape, p=[0.999, 0.001]).astype(float), sigma=0.8); mag = mag * (1 - bleeds * 8)
                sim = mag
            elif swi_mode == 'minip': sim = (c + g*0.8 + w*0.7) * (1 - vein_mask * 0.9) 
        elif adc_mode: 
            raw_adc = self.get_adc_map(g, w, c); 
            if stroke_mask is not None: raw_adc[stroke_mask] = 0.4
            sim = (raw_adc + np.random.normal(0, 0.1, raw_adc.shape)) / 3.2 
        elif seq_type == 'dwi':
            adc_map = self.get_adc_map(g, w, c); s0_map = (c * 3.0) + (g * 1.5) + (w * 1.0) + (r * 0.1)
            if stroke_mask is not None: adc_map[stroke_mask] = 0.4; s0_map[stroke_mask] = 3.0
            sim = s0_map * np.exp(-b_val * adc_map * 0.001) / 3.0
        elif seq_type == 'gre':
             t1_map = self.get_t1_map(g, w, c); t2s_map = self.get_t2s_map(g, w, c); pd_map = (c * 1.0) + (g * 0.9) + (w * 0.7) + (r * 0.2)
             rad = np.radians(fa); e1 = np.exp(-tr / (t1_map + 1e-6)); sim = pd_map * ((np.sin(rad) * (1 - e1)) / (1 - np.cos(rad) * e1 + 1e-6)) * np.exp(-te / (t2s_map + 1e-6)) * 5.0
        else:
            if stroke_mask is not None and ('T2' in st.session_state.seq or 'FLAIR' in st.session_state.seq): w[stroke_mask] = 0; c[stroke_mask] = 1.0
            sim = (c * w_vals['csf']) + (g * w_vals['gm']) + (w * w_vals['wm']) + (r * (w_vals['fat']*0.2))
        return sim 
    def get_dims(self): return (197, 233, 189) if self.ready else (100, 100, 100)
processor = AdvancedMRIProcessor()

# --- 13. AFFICHAGE FINAL ---
st.title("Simulateur MagnétoVault V7.42")

# Démarrage des Onglets
t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14 = st.tabs([
    "Fantôme", "Espace K 🌀", "Signaux", "Codage", "🧠 Anatomie", 
    "📈 Physique", "⚡ Chronogramme", "☣️ Artefacts", "🚀 iPAT", "🧬 Théorie Diffusion", "🎓 Cours", "🩸 SWI & Dipôle", "3D T1 (MP-RAGE)", "ASL (Perfusion)"
])

# [TAB 1 : FANTOME]
with t1:
    c1, c2 = st.columns([1, 1])
    with c1:
        k1, k2 = st.columns(2); k1.metric("⏱️ Durée", str_duree); k2.metric("📉 SNR Relatif", str_snr); st.divider()
        st.subheader("1. Formules & Légendes")
        
        if is_dwi: 
            st.markdown("##### 🧬 Signal de Diffusion"); st.latex(r"S_b = S_0 \times e^{-b \times ADC}")
            st.markdown("##### 🧮 Calcul ADC"); st.latex(r"ADC = \frac{1}{b} \times \ln\left(\frac{S_0}{S_b}\right)")
            st.markdown("""
            **Légende Détaillée :**
            * **$S_b$** : Signal mesuré avec pondération de diffusion (b > 0).
            * **$S_0$** : Signal de référence sans diffusion (b = 0, équivalent T2).
            * **$b$** : Facteur de diffusion ($s/mm^2$). Dépend de l'amplitude et durée des gradients.
            * **$ADC$** : Coefficient de Diffusion Apparent ($mm^2/s$). Reflète la mobilité des molécules d'eau.
            """)
            
        elif is_asl: 
            st.markdown("##### 🩸 Perfusion ASL"); st.latex(r"\Delta M = \frac{2 \cdot M_0 \cdot \alpha \cdot f}{\lambda} \cdot e^{-\frac{PLD}{T_{1b}}}")
            st.markdown("""
            **Légende Détaillée :**
            * **$\Delta M$** : Signal de perfusion (Différence entre image Contrôle et Marquée).
            * **$M_0$** : Magnétisation d'équilibre du tissu.
            * **$\alpha$** : Efficacité de l'inversion (marquage).
            * **$f$ (CBF)** : Flux Sanguin Cérébral ($ml/100g/min$). C'est ce qu'on mesure.
            * **$\lambda$** : Coefficient de partage sang/tissu.
            * **$PLD$** : Post Labeling Delay.
            * **$T_{1b}$** : T1 du sang artériel (~1650ms).
            """)
            
        else:
            st.markdown("##### ⏱️ Temps d'Acquisition"); st.latex(r"TA = \frac{TR \times N_{Ph} \times NEX}{TF \times R}")
            st.markdown("##### 📉 Rapport Signal/Bruit (Simplifié)"); st.latex(r"SNR \propto V_{vox} \times \sqrt{\frac{N_{Ph} \times NEX}{BW}} \times \frac{1}{g \sqrt{R}}")
            st.markdown("""
            **Légende Détaillée :**
            * **$TR$** : Temps de Répétition (ms). Intervalle entre deux impulsions d'excitation.
            * **$TE$** : Temps d'Écho (ms). Temps entre l'excitation et la mesure du signal.
            * **$N_{Ph}$** : Nombre de lignes de phase (Résolution Y).
            * **$NEX$** : Nombre d'excitations (Moyennages). Augmente le SNR.
            * **$TF$** : Facteur Turbo (Train d'échos). Accélère l'acquisition en remplissant plusieurs lignes k par TR.
            * **$R$** : Facteur d'Accélération (Imagerie Parallèle iPAT).
            * **$V_{vox}$** : Volume du Voxel ($dx \times dy \times dz$).
            * **$BW$** : Bande Passante (Hz/px). Plus elle est élevée, plus le bruit augmente.
            * **$g$** : Facteur g (Geometry Factor). Bruit lié à la géométrie des antennes en iPAT.
            """)
        
        st.divider(); st.subheader("Paramètres Actuels"); st.markdown(f"* **$TR$ :** {tr:.0f} ms | **$TE$ :** {te:.0f} ms | **$N_{{Ph}}$ :** {mat}")
        
        if show_stroke: st.error("⚠️ **PATHOLOGIE : AVC Ischémique**")
        if show_atrophy: st.warning("🧠 **PATHOLOGIE : Atrophie (Alzheimer)**")

    with c2:
        fig_anot, ax_anot = plt.subplots(figsize=(5,5)); ax_anot.imshow(final, cmap='gray', vmin=0, vmax=1.3); ax_anot.axis('off')
        ax_anot.text(S/2, S/2, "Eau\n(LCR)", color='white', ha='center', va='center', fontsize=9, fontweight='bold')
        ax_anot.text(S/2, S/2 + (S*0.35/2), "SB", color='black', ha='center', va='center', fontsize=9, fontweight='bold')
        ax_anot.text(S/2, S/2 + (S*0.65/2), "SG", color='white', ha='center', va='center', fontsize=9, fontweight='bold')
        ax_anot.text(S/2, S*0.93, "FAT", color='orange', ha='center', va='center', fontsize=10, fontweight='bold')
        st.pyplot(fig_anot); plt.close(fig_anot)

with t2:
    st.markdown("### 🌀 Remplissage de l'Espace K")
    col_k1, col_k2 = st.columns([1, 1])
    with col_k1:
        fill_mode = st.radio("Ordre de Remplissage", ["Linéaire (Haut -> Bas)", "Centrique (Centre -> Bords)"], key=f"k_mode_{current_reset_id}")
        acq_pct = st.slider("Progression (%)", 0, 100, 10, step=1, key=f"k_pct_{current_reset_id}")
        if turbo > 1 and not is_gre and not is_dwi:
            st.divider(); st.markdown("#### 🚅 Stratégie Turbo")
            echo_vals = [(i+1)*es for i in range(turbo)]; closest_idx = np.argmin(np.abs(np.array(echo_vals) - te)) 
            fig_seg, ax_seg = plt.subplots(figsize=(6, 5)); k_size = 100; half = k_size // 2
            ax_seg.add_patch(patches.Rectangle((-half, -half), k_size, k_size, color='black', alpha=0.1)); ax_seg.add_patch(patches.Rectangle((-half, -half), k_size, k_size, linewidth=2, edgecolor='black', facecolor='none'))
            ax_seg.text(0, -half - 10, "Axe Fréquence (kx)", ha='center', fontsize=8); ax_seg.text(half + 5, 0, "Axe Phase (ky)", va='center', rotation=90, fontsize=8)
            band_height = k_size / turbo; current_dist = 1; direction = 1; echo_data = []
            for i, val in enumerate(echo_vals): echo_data.append({'index_orig': i, 'time': val, 'diff': abs(val - te)})
            echo_data.sort(key=lambda x: (x['diff'], x['time'])); eff_item = echo_data[0]; pos_map = {}; pos_map[eff_item['index_orig']] = 0
            for k in range(1, len(echo_data)):
                item = echo_data[k]; offset = current_dist * direction; pos_map[item['index_orig']] = offset
                if direction == 1: direction = -1
                else: direction = 1; current_dist += 1
            for i in range(turbo):
                y_pos = (pos_map[i] * band_height) - (band_height/2); is_eff = (i == eff_item['index_orig']); col = 'red' if is_eff else 'dodgerblue'
                ax_seg.add_patch(patches.Rectangle((-half, y_pos), k_size, band_height, facecolor=col, edgecolor='white', alpha=0.6))
                label_txt = f"Écho {i+1} ({echo_vals[i]:.0f}ms)"; font_c = 'red' if is_eff else 'black'
                ax_seg.text(-130, y_pos + band_height/2, label_txt, color=font_c, fontsize=9, ha='right', va='center')
            ax_seg.set_xlim(-180, 70); ax_seg.set_ylim(-70, 70); ax_seg.axis('off'); st.pyplot(fig_seg)
    with col_k2:
        mask_k = np.zeros((S, S)); lines_to_fill = int(S * (acq_pct / 100.0))
        if "Linéaire" in fill_mode: mask_k[0:lines_to_fill, :] = 1
        else: center_line = S // 2; half = lines_to_fill // 2; mask_k[center_line-half:center_line+half, :] = 1
        kspace_masked = f * mask_k; img_rec = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_masked)))
        fig_k, ax_k = plt.subplots(figsize=(4, 4)); ax_k.imshow(20 * np.log(np.abs(kspace_masked) + 1), cmap='inferno'); ax_k.axis('off'); st.pyplot(fig_k); plt.close(fig_k); st.image(img_rec, clamp=True, width=300)

with t3:
    st.markdown("### 📊 Comparaison des Signaux")
    c_sig_left, c_sig_center, c_sig_right = st.columns([1, 2, 1])
    with c_sig_center:
        fig_sig, ax_sig = plt.subplots(figsize=(4, 2.5))
        vals_bar = [v_lcr, v_gm, v_wm, v_fat]; noms = ["WATER", "GM", "WM", "FAT"]
        if show_stroke: vals_bar.append(v_stroke); noms.append("AVC")
        cols = ['cyan', 'dimgray', 'lightgray', 'orange', 'red'] if show_stroke else ['cyan', 'dimgray', 'lightgray', 'orange']
        bars = ax_sig.bar(noms, vals_bar, color=cols, edgecolor='black'); ax_sig.set_ylim(0, 1.3); ax_sig.grid(True, axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig_sig); plt.close(fig_sig)

with t4:
    components.html("""<!DOCTYPE html><html><head><style>body{margin:0;padding:5px;font-family:sans-serif;} .box{display:flex;gap:15px;} .ctrl{width:220px;padding:10px;background:#f9f9f9;border:1px solid #ccc;border-radius:8px;} canvas{border:1px solid #ccc;background:#f8f9fa;border-radius:8px;} input{width:100%;} label{font-size:11px;font-weight:bold;display:block;} button{width:100%;padding:8px;background:#4f46e5;color:white;border:none;border-radius:4px;cursor:pointer;}</style></head><body><div class='box'><div class='ctrl'><h4>Codage</h4><label>Freq</label><input type='range' id='f' min='-100' max='100' value='0'><br><label>Phase</label><input type='range' id='p' min='-100' max='100' value='0'><br><label>Coupe</label><input type='range' id='z' min='-100' max='100' value='0'><br><label>Matrice</label><input type='range' id='g' min='5' max='20' value='12'><br><button onclick='rst()'>Reset</button></div><div><canvas id='c1' width='350' height='350'></canvas><canvas id='c2' width='80' height='350'></canvas></div></div><script>const c1=document.getElementById('c1');const x=c1.getContext('2d');const c2=document.getElementById('c2');const z=c2.getContext('2d');const sf=document.getElementById('f');const sp=document.getElementById('p');const sz=document.getElementById('z');const sg=document.getElementById('g');const pd=30;function arrow(ctx,x,y,a,s){const l=s*0.35;ctx.save();ctx.translate(x,y);ctx.rotate(a);ctx.beginPath();ctx.moveTo(-l,0);ctx.lineTo(l,0);ctx.lineTo(l-6,-6);ctx.moveTo(l,0);ctx.lineTo(l-6,6);ctx.strokeStyle='white';ctx.lineWidth=1.5;ctx.stroke();ctx.restore();} function draw(){x.clearRect(0,0,350,350);z.clearRect(0,0,80,350);const fv=parseFloat(sf.value);const pv=parseFloat(sp.value);const zv=parseFloat(sz.value);const gs=parseInt(sg.value);const st=(350-2*pd)/gs;const h=(pd*0.8)*(fv/100);x.fillStyle='rgba(255,0,0,0.3)';if(fv!=0){x.beginPath();x.moveTo(pd,pd/2);x.lineTo(pd,pd/2-h);x.lineTo(350-pd,pd/2+h);x.lineTo(350-pd,pd/2);x.fill();}const w=(pd*0.8)*(pv/100);x.fillStyle='rgba(0,255,0,0.3)';if(pv!=0){x.beginPath();x.moveTo(350-pd/2,pd);x.lineTo(350-pd/2-w,pd);x.lineTo(350-pd/2+w,350-pd);x.lineTo(350-pd/2,350-pd);x.fill();} for(let i=0;i<gs;i++){for(let j=0;j<gs;j++){const cx=pd+i*st+st/2;const cy=pd+j*st+st/2;const ph=(i-gs/2)*(fv/100)*3+(j-gs/2)*(pv/100)*3;const cph=(j-gs/2)*(pv/100);x.strokeStyle='black';x.beginPath();x.arc(cx,cy,st*0.4,0,6.28);x.fillStyle='#94a3b8';x.fill();if(cph>0.01)x.fillStyle='rgba(255,255,0,0.5)';if(cph<-0.01)x.fillStyle='rgba(0,0,255,0.5)';x.fill();arrow(x,cx,cy,ph,st*0.6);}}const yz=175-(zv/100)*150;const gr=z.createLinearGradient(0,0,0,350);gr.addColorStop(0,'red');gr.addColorStop(1,'blue');z.fillStyle=gr;z.fillRect(10,10,20,330);z.strokeStyle='black';z.lineWidth=3;z.beginPath();z.moveTo(10,yz);z.lineTo(70,yz);z.stroke();z.fillStyle='black';z.fillText('Z',35,yz-5);} [sf,sp,sz,sg].forEach(s=>s.addEventListener('input',draw));function rst(){sf.value=0;sp.value=0;sz.value=0;sg.value=12;draw();}draw();</script></body></html>""", height=450)

with t5:
    st.header("Exploration Anatomique (Physique Avancée)")
    if HAS_NILEARN and processor.ready:
        c1, c2 = st.columns([1, 3]); dims = processor.get_dims()
        with c1:
            plane = st.radio("Plan de Coupe", ["Plan Axial", "Plan Sagittal", "Plan Coronal"], key="or_298")
            if "Axial" in plane: idx = st.slider("Z", 0, dims[2]-1, 90, key=f"sl_{current_reset_id}"); ax='z'
            elif "Sagittal" in plane: idx = st.slider("X", 0, dims[0]-1, 90, key=f"sl_{current_reset_id}"); ax='x'
            else: idx = st.slider("Y", 0, dims[1]-1, 100, key=f"sl_{current_reset_id}"); ax='y'
            st.divider(); window = st.slider("Fenêtre", 0.01, 2.0, 0.74, 0.005, key=f"wn_{current_reset_id}"); level = st.slider("Niveau", 0.0, 1.0, 0.55, 0.005, key=f"lv_{current_reset_id}")
            if is_dwi: 
                if show_adc_map: st.info("🗺️ **Mode Carte ADC** (LCR Blanc)")
                else: st.success(f"🧬 **Mode Diffusion** (b={b_value})")
            elif is_gre: st.warning(f"⚡ **Mode Gradient** (TE={te}ms)")
            else: st.info("📸 **Mode Pondéré Standard**")
            if show_stroke and ax == 'z': st.error("⚠️ **AVC Visible**")
            if st.session_state.atrophy_active and ax == 'z': st.warning("🧠 **Atrophie GCP Active**")
        with c2:
            w_vals = {'csf':v_lcr, 'gm':v_gm, 'wm':v_wm, 'fat':v_fat}
            if show_stroke: w_vals['wm'] = w_vals['wm'] * 0.9 + v_stroke * 0.1
            seq_type_arg = 'dwi' if is_dwi else ('gre' if is_gre else None)
            img_raw = processor.get_slice(ax, idx, w_vals, seq_type=seq_type_arg, te=te, tr=tr, fa=flip_angle, b_val=b_value, adc_mode=show_adc_map, with_stroke=show_stroke)
            if img_raw is not None: st.image(apply_window_level(img_raw, window, level), clamp=True, width=600)
    else: st.warning("Module 'nilearn' manquant.")

with t6:
    st.header("📈 Physique")
    tists = [T_FAT, T_WM, T_GM, T_LCR]; cols = ['orange', 'lightgray', 'dimgray', 'cyan'] 
    if show_stroke: tists.append(T_STROKE); cols.append('red') 
    fig_t1 = plt.figure(figsize=(10, 3)); gs = fig_t1.add_gridspec(1, 2, width_ratios=[30, 1], wspace=0.05)
    ax_t1 = fig_t1.add_subplot(gs[0]); ax_bar = fig_t1.add_subplot(gs[1]); x_t = np.linspace(0, 4000, 500)
    ax_t1.set_title("Relaxation Longitudinale (T1)")
    if is_gre:
        start_mz = np.cos(np.radians(flip_angle)); ax_t1.set_ylim(-0.1, 1.1)
        for t, col in zip(tists, cols): mz = 1 - (1 - start_mz) * np.exp(-x_t / t['T1']); ax_t1.plot(x_t, mz, color=col, label=t['Label']); ax_t1.axhline(start_mz, color='gray', linestyle=':', label=f"Mz(0)")
    elif is_ir:
        ax_t1.set_ylim(-1.1, 1.1); ax_t1.axhline(0, color='black')
        for t, col in zip(tists, cols): mz = 1 - 2 * np.exp(-x_t / t['T1']); ax_t1.plot(x_t, mz, color=col, label=t['Label']); ax_t1.axvline(x=ti, color='green', linestyle='--', label='TI')
    else:
        ax_t1.set_ylim(0, 1.1)
        for t, col in zip(tists, cols): mz = 1 - np.exp(-x_t / t['T1']); ax_t1.plot(x_t, mz, color=col, label=t['Label'])
    ax_t1.axvline(x=tr_effective, color='red', linestyle='--', label='TR Réel'); gradient = np.linspace(1, 0, 256).reshape(-1, 1)
    if is_ir: gradient = np.abs(np.linspace(1, -1, 256)).reshape(-1, 1)
    ax_bar.imshow(gradient, aspect='auto', cmap='gray', extent=[0, 1, ax_t1.get_ylim()[0], ax_t1.get_ylim()[1]])
    ax_t1.legend(); ax_bar.axis('off'); st.pyplot(fig_t1); plt.close(fig_t1)
    
    fig_t2 = plt.figure(figsize=(10, 3)); gs2 = fig_t2.add_gridspec(1, 2, width_ratios=[30, 1], wspace=0.05)
    ax_t2 = fig_t2.add_subplot(gs2[0]); ax_bar2 = fig_t2.add_subplot(gs2[1]); x_te = np.linspace(0, 500, 300)
    ax_t2.set_title("Relaxation Transversale (T2/T2*)")
    for t, col in zip(tists, cols): 
        val_t2 = t['T2s'] if is_gre else t['T2']; mxy = np.exp(-x_te / val_t2); label_cur = f"{t['Label']} (T2*)" if is_gre else t['Label']
        ax_t2.plot(x_te, mxy, color=col, label=label_cur)
    ax_t2.axvline(x=te, color='red', linestyle='--', label='TE Eff'); gradient_t2 = np.linspace(1, 0, 256).reshape(-1, 1)
    ax_bar2.imshow(gradient_t2, aspect='auto', cmap='gray', extent=[0, 1, 0, 1.0]); ax_bar2.axis('off'); st.pyplot(fig_t2); plt.close(fig_t2)

with t7:
    st.header("⚡ Chronogramme")
    t_90 = 10
    if is_gre:
        st.subheader(f"Séquence : Écho de Gradient (Angle {flip_angle}°)")
        t_max = max(tr + 40, te + 50); t = np.linspace(0, t_max, 2000)
        rf_sigma = 0.5; grad_width = 3.0
        fig, axs = plt.subplots(5, 1, sharex=True, figsize=(10, 8), gridspec_kw={'hspace': 0.1})
        rf = np.zeros_like(t); amp_rf = flip_angle / 90.0
        rf += amp_rf * np.exp(-0.5 * ((t - t_90)**2) / (rf_sigma**2))
        t_90_next = t_90 + tr; rf += amp_rf * np.exp(-0.5 * ((t - t_90_next)**2) / (rf_sigma**2))
        axs[0].plot(t, rf, color='black'); axs[0].fill_between(t, 0, rf, color='green', alpha=0.4)
        axs[0].set_ylabel("RF"); axs[0].set_yticks([0, 1], ["", f"{flip_angle}°"]); axs[0].spines['top'].set_visible(False)
        gsc = np.zeros_like(t); mask_sel = (t > t_90 - grad_width) & (t < t_90 + grad_width); gsc[mask_sel] = 1.0
        mask_reph = (t > t_90 + grad_width + 1) & (t < t_90 + 2*grad_width + 1); gsc[mask_reph] = -0.8
        mask_sel2 = (t > t_90_next - grad_width) & (t < t_90_next + grad_width); gsc[mask_sel2] = 1.0
        axs[1].plot(t, gsc, color='green'); axs[1].fill_between(t, 0, gsc, color='green', alpha=0.6); axs[1].set_ylabel("Gsc")
        gcp = np.zeros_like(t); t_code = t_90 + 15; mask_c = (t > t_code - grad_width) & (t < t_code + grad_width); gcp[mask_c] = 0.5
        axs[2].plot(t, gcp, color='orange'); axs[2].fill_between(t, 0, gcp, color='orange', alpha=0.6); axs[2].set_ylabel("Gcp")
        gcf = np.zeros_like(t); t_read = t_90 + te; mask_read = (t > t_read - grad_width) & (t < t_read + grad_width); gcf[mask_read] = 1.0
        t_pre = t_read - (2 * grad_width) - 2; 
        if t_pre > t_90 + grad_width: mask_pre = (t > t_pre - grad_width) & (t < t_pre + grad_width); gcf[mask_pre] = -1.0
        axs[3].plot(t, gcf, color='dodgerblue'); axs[3].fill_between(t, 0, gcf, color='dodgerblue', alpha=0.6); axs[3].set_ylabel("Gcf")
        sig = np.zeros_like(t); idx_s = np.argmin(np.abs(t - (t_read - 3))); idx_e = np.argmin(np.abs(t - (t_read + 3)))
        if idx_e > idx_s: grid = np.linspace(-3, 3, idx_e - idx_s); sig[idx_s:idx_e] = np.sinc(grid)
        axs[4].plot(t, sig, color='navy'); axs[4].set_ylabel("Echo"); axs[4].axvline(x=t_read, color='red', linestyle='--'); axs[4].text(t_read, 1.1, f"TE={te:.0f}ms", color='red', ha='center')
        axs[0].annotate("", xy=(t_90_next, 1.2), xytext=(t_90, 1.2), arrowprops=dict(arrowstyle="<->", color='red')); axs[0].text((t_90+t_90_next)/2, 1.3, f"TR={tr:.0f}ms", color='red', ha='center')
        st.pyplot(fig); plt.close(fig)
    else:
        is_turbo = turbo > 1; t_90 = 10
        if is_dwi: st.subheader("Séquence : Diffusion (DWI - SE EPI)")
        elif is_turbo: st.subheader(f"Séquence : Turbo Spin Écho (TSE) - Facteur {turbo}")
        else: st.subheader("Séquence : Spin Écho (SE)")
        if not is_turbo: echo_times = [t_90 + te]; t_180s = [t_90 + (te/2)]; es_disp = te; t_max = max(200, t_90 + te + 50)
        else: echo_times = [t_90 + (i+1)*es for i in range(turbo)]; t_180s = []; es_disp = es; t_max = max(200, echo_times[-1] + 50)
        t = np.linspace(0, t_max, 2000); rf_sigma = 0.5; grad_width = max(1.5, es_disp * 0.2)
        t_180s = [];
        for i in range(turbo): t_p = t_90 + (i * es) + (es/2); t_180s.append(t_p)
        fig, axs = plt.subplots(5, 1, sharex=True, figsize=(10, 8), gridspec_kw={'hspace': 0.1})
        rf = np.zeros_like(t)
        def add_rf_pulse(center, amp, w): return amp * np.exp(-0.5 * ((t - center)**2) / (w**2))
        rf += add_rf_pulse(t_90, 1.0, rf_sigma) 
        for t_p in t_180s:
            if t_p < t_max: rf += add_rf_pulse(t_p, 1.6, rf_sigma)
        axs[0].plot(t, rf, color='black', linewidth=1.5); axs[0].fill_between(t, 0, rf, color='green', alpha=0.4)
        axs[0].set_ylabel("RF"); axs[0].set_yticks([0, 1, 1.6], ["", "90", "180"]); axs[0].spines['top'].set_visible(False); axs[0].spines['right'].set_visible(False); axs[0].set_xlim(0, t_max)
        gsc = np.zeros_like(t)
        def add_trap(center, amp, w): mask = (t > center - w) & (t < center + w); gsc[mask] = amp
        add_trap(t_90, 1.0, grad_width); t_rephase = t_90 + grad_width + 1.5; add_trap(t_rephase, -0.8, grad_width*0.6)
        for t_p in t_180s: add_trap(t_p, 1.0, grad_width)
        axs[1].fill_between(t, 0, gsc, color='green', alpha=0.6); axs[1].plot(t, gsc, color='green', linewidth=1); axs[1].set_ylabel("Gsc"); axs[1].set_yticks([]); axs[1].spines['top'].set_visible(False); axs[1].spines['right'].set_visible(False)
        gcp = np.zeros_like(t); target_te_graph = te; closest_idx = np.argmin(np.abs(np.array(echo_times) - target_te_graph))
        max_dist = max(closest_idx, (len(echo_times)-1) - closest_idx) if len(echo_times) > 1 else 1; 
        if max_dist == 0: max_dist = 1
        for i, t_e in enumerate(echo_times):
            if not is_turbo: t_180_curr = t_180s[0]
            else: t_180_curr = t_180s[i]
            t_code = (t_180_curr + t_e)/2 - (es_disp*0.1); t_rewind = t_e + (es_disp*0.15)
            if t_rewind < t_max:
                if i == closest_idx: height = 0.2; label = "BF"; col_lbl = "red"
                else: dist = abs(i - closest_idx); height = 0.2 + (0.8 * (dist / max_dist)); label = ""; col_lbl = "gray"
                w_ph = grad_width * 0.7; mask_c = (t > t_code - w_ph) & (t < t_code + w_ph); gcp[mask_c] = height
                if label == "BF": axs[2].text(t_code, height+0.1, label, color=col_lbl, ha='center', fontsize=9, weight='bold')
                mask_r = (t > t_rewind - w_ph) & (t < t_rewind + w_ph); gcp[mask_r] = -height
        axs[2].fill_between(t, 0, gcp, color='darkorange', alpha=0.7); axs[2].set_ylabel("Gcp"); axs[2].set_yticks([]); axs[2].spines['top'].set_visible(False); axs[2].spines['right'].set_visible(False)
        gcf = np.zeros_like(t); t_pre = (t_90 + t_180s[0])/2; add_trap_gcf = lambda c, w: ((t > c - w) & (t < c + w)); gcf[add_trap_gcf(t_pre, grad_width)] = 1.0 
        for t_e in echo_times:
            if t_e < t_max: w_read = grad_width * 1.2; gcf[add_trap_gcf(t_e, w_read)] = 1.0
        axs[3].fill_between(t, 0, gcf, color='dodgerblue', alpha=0.5); axs[3].set_ylabel("Gcf"); axs[3].set_yticks([]); axs[3].spines['top'].set_visible(False); axs[3].spines['right'].set_visible(False)
        sig = np.zeros_like(t)
        for i, t_e in enumerate(echo_times):
            if t_e < t_max - 5:
                w_sig = grad_width * 1.2; idx_start = np.argmin(np.abs(t - (t_e - w_sig))); idx_end = np.argmin(np.abs(t - (t_e + w_sig)))
                if idx_end > idx_start:
                    grid = np.linspace(-3, 3, idx_end - idx_start); amp = np.exp(-t_e / T_GM['T2']) 
                    sig[idx_start:idx_end] = np.sinc(grid) * amp
                if i == closest_idx:
                     axs[4].text(t_e, amp+0.3, "TE eff", ha='center', color='red', fontweight='bold', fontsize=10)
                     axs[4].axvline(x=t_e, color='red', linestyle='--', alpha=0.5)
        axs[4].plot(t, sig, color='navy', linewidth=1.5); axs[4].set_ylabel("Signal"); axs[4].set_xlabel("Temps (ms)"); axs[4].set_yticks([]); axs[4].spines['top'].set_visible(False); axs[4].spines['right'].set_visible(False)
        st.pyplot(fig); plt.close(fig)

with t8:
    st.header("☣️ Laboratoire d'Artefacts")
    col_ctrl, col_visu = st.columns([1, 2])
    with col_ctrl:
        st.markdown("#### Choix de l'Artefact")
        artefact_type = st.radio("Sélectionnez :", ["Aliasing", "Décalage Chimique", "Troncature", "Mouvement", "Zipper"], key="art_main_radio")
        st.divider()
    if "Aliasing" in artefact_type:
        with col_ctrl:
            st.info(f"FOV Actuel : **{fov} mm** (Objet : 230 mm)")
            if fov < 230: st.error("⚠️ Aliasing Actif !")
        with col_visu:
            img_art = final.copy()
            if fov < 230:
                ratio = fov / 230.0; shift_w = int(S * (1 - ratio) / 2)
                top = img_art[0:shift_w, :]; bot = img_art[S-shift_w:S, :]
                img_art = img_art.copy(); img_art[S-shift_w:S, :] += top; img_art[0:shift_w, :] += bot
            fig_a = plt.figure(figsize=(5,5)); ax_a = fig_a.add_subplot(111); ax_a.imshow(img_art, cmap='gray', vmin=0, vmax=1.3); ax_a.axis('off'); st.pyplot(fig_a)
    elif "Décalage" in artefact_type:
        with col_ctrl: st.info(f"BW : **{bw} Hz/px**")
        with col_visu:
            if bw == 220: px = 0.0 
            else: px = 220.0/float(bw)
            sh = shift(img_fat, [0, px]); res = img_water + sh
            fig_cs = plt.figure(figsize=(5,5)); ax_cs = fig_cs.add_subplot(111); ax_cs.imshow(res, cmap='gray', vmin=0, vmax=1.3); ax_cs.axis('off'); st.pyplot(fig_cs)
    elif "Troncature" in artefact_type:
        with col_ctrl: 
            sm = st.select_slider("Matrice Sim", [32, 64, 128, 256], 64)
            if sm <= 64: st.warning("Visible")
        with col_visu:
            ft = np.fft.fftshift(np.fft.fft2(final)); c=S//2; k=sm//2; m=np.zeros_like(ft); m[c-k:c+k, c-k:c+k]=1
            res = np.abs(np.fft.ifft2(np.fft.ifftshift(ft*m)))
            fig_g = plt.figure(figsize=(5,5)); ax_g = fig_g.add_subplot(111); ax_g.imshow(res, cmap='gray', vmin=0, vmax=1.3); ax_g.axis('off'); st.pyplot(fig_g)
    elif "Mouvement" in artefact_type:
        with col_ctrl: it = st.slider("Intensité", 0.0, 5.0, 0.5)
        with col_visu:
            ft = np.fft.fftshift(np.fft.fft2(final))
            if it > 0:
                ph = np.random.normal(0, it, S)
                for i in range(S): ft[i, :] *= np.exp(1j * ph[i])
            res = np.abs(np.fft.ifft2(np.fft.ifftshift(ft)))
            fig_m = plt.figure(figsize=(5,5)); ax_m = fig_m.add_subplot(111); ax_m.imshow(res, cmap='gray', vmin=0, vmax=1.3); ax_m.axis('off'); st.pyplot(fig_m)
    elif "Zipper" in artefact_type:
        with col_ctrl: fr = st.slider("Freq", 0, S-1, S//2); vol = st.slider("Vol", 0, 100, 10)
        with col_visu:
            ft = np.fft.fftshift(np.fft.fft2(final))
            if vol > 0:
                ns = np.random.normal(0, vol, S) + (vol*5); alt = np.array([1 if i%2==0 else -1 for i in range(S)])
                ft[:, fr] += ns * alt * 50
            res = np.abs(np.fft.ifft2(np.fft.ifftshift(ft)))
            fig_z = plt.figure(figsize=(5,5)); ax_z = fig_z.add_subplot(111); ax_z.imshow(res, cmap='gray', vmin=0, vmax=1.3); ax_z.axis('off'); st.pyplot(fig_z)

with t9:
    st.header("🚀 Imagerie Parallèle (iPAT) : Le Point de Vue")
    if ipat_factor == 1:
        st.warning("⚠️ L'accélération iPAT est désactivée. Activez-la dans la barre latérale (Section 4).")
    else:
        st.success(f"✅ Accélération Active : **R = {ipat_factor}**")
        st.markdown("### 1. Analogie de la Fenêtre (Interactive)")
        st.info("Déplacez-vous ou choisissez 'Vue Simultanée' pour voir la reconstruction totale.")
        pos_obs = st.select_slider("📍 Votre Position devant la fenêtre :", options=["Gauche", "Centre", "Droite", "👁️ Vue Simultanée (Tous)"], value="Centre", key=f"pos_fenetre_{current_reset_id}")
        wall_g_rect = patches.Rectangle((-10, 8), 40, 1, color='lightgray'); wall_d_rect = patches.Rectangle((70, 8), 40, 1, color='lightgray'); window_frame_x = [30, 70]
        if pos_obs == "👁️ Vue Simultanée (Tous)":
            c_simu1, c_simu2 = st.columns([2, 1])
            with c_simu1:
                fig_all, ax_all = plt.subplots(figsize=(8, 4))
                ax_all.add_patch(wall_g_rect); ax_all.text(10, 8.5, "Mur G", color='black', ha='center', va='center', fontsize=8)
                ax_all.add_patch(wall_d_rect); ax_all.text(90, 8.5, "Mur D", color='black', ha='center', va='center', fontsize=8)
                ax_all.add_patch(patches.Circle((50, 8.5), 3, color='purple')); ax_all.text(50, 6.5, "Machine", color='purple', ha='center', va='top', fontweight='bold')
                ax_all.plot(window_frame_x, [4, 4], color='black', linewidth=3); ax_all.text(25, 4, "Fenêtre", ha='right', va='center')
                ax_all.plot(30, 0, 'o', color='blue', markersize=10); ax_all.add_patch(plt.Polygon([[30, 0], [35, 9], [100, 9]], color='blue', alpha=0.1))
                ax_all.plot(70, 0, 'o', color='orange', markersize=10); ax_all.add_patch(plt.Polygon([[70, 0], [0, 9], [65, 9]], color='orange', alpha=0.1))
                ax_all.plot(50, 0, 'o', color='green', markersize=10); ax_all.add_patch(plt.Polygon([[50, 0], [35, 9], [65, 9]], color='green', alpha=0.1))
                ax_all.set_xlim(-10, 110); ax_all.set_ylim(-3, 10); ax_all.axis('off'); ax_all.set_title("Les 3 observateurs regardent (Fenêtre étroite)", fontsize=10); st.pyplot(fig_all)
            with c_simu2:
                st.markdown("**👀 Résultat Reconstitué :**"); st.markdown("_Somme des 3 vues = Image Totale_")
                fig_full, ax_f = plt.subplots(figsize=(4, 4)); ax_f.set_xlim(0, 100); ax_f.set_ylim(0, 100); ax_f.axis('off'); ax_f.add_patch(patches.Rectangle((0,0), 100, 100, color='whitesmoke'))
                ax_f.add_patch(patches.Rectangle((0, 0), 30, 100, color='gray')); ax_f.text(15, 50, "MUR G", color='white', ha='center', va='center', rotation=90, fontweight='bold')
                ax_f.add_patch(patches.Circle((50, 50), 15, color='purple')); ax_f.add_patch(patches.Rectangle((70, 0), 30, 100, color='gray'))
                ax_f.text(85, 50, "MUR D", color='white', ha='center', va='center', rotation=90, fontweight='bold'); ax_f.set_title("Votre Rétine (Synthèse)", fontsize=9); st.pyplot(fig_full)
        else:
            c_simu1, c_simu2 = st.columns([2, 1])
            with c_simu1:
                fig_analog, ax_an = plt.subplots(figsize=(8, 4))
                ax_an.add_patch(wall_g_rect); ax_an.text(10, 8.5, "Mur G", color='black', ha='center', va='center', fontsize=8)
                ax_an.add_patch(wall_d_rect); ax_an.text(90, 8.5, "Mur D", color='black', ha='center', va='center', fontsize=8)
                ax_an.add_patch(patches.Circle((50, 8.5), 3, color='purple')); ax_an.text(50, 6.5, "Machine", color='purple', ha='center', va='top', fontweight='bold')
                ax_an.plot(window_frame_x, [4, 4], color='black', linewidth=3); ax_an.text(25, 4, "Fenêtre", ha='right', va='center')
                ax_an.plot(30, 0, 'o', color='lightgray', alpha=0.5); ax_an.plot(70, 0, 'o', color='lightgray', alpha=0.5); ax_an.plot(50, 0, 'o', color='lightgray', alpha=0.5) 
                if pos_obs == "Gauche": user_x = 30; col_u = "blue"; poly_pts = [[30, 0], [35, 9], [100, 9]]; msg_view = "Je vois surtout le Mur de Droite"
                elif pos_obs == "Droite": user_x = 70; col_u = "orange"; poly_pts = [[70, 0], [0, 9], [65, 9]]; msg_view = "Je vois surtout le Mur de Gauche"
                else: user_x = 50; col_u = "green"; poly_pts = [[50, 0], [35, 9], [65, 9]]; msg_view = "Je vois uniquement la Machine (Centre)"
                ax_an.plot(user_x, 0, 'o', color=col_u, markersize=12); ax_an.text(user_x, -1.5, "VOUS", color=col_u, ha='center', va='top', fontweight='bold')
                ax_an.add_patch(plt.Polygon(poly_pts, color=col_u, alpha=0.2))
                ax_an.set_xlim(-10, 110); ax_an.set_ylim(-3, 10); ax_an.axis('off'); ax_an.set_title(f"Vue de dessus : {pos_obs}", fontsize=10); st.pyplot(fig_analog)
            with c_simu2:
                st.markdown(f"**👀 Ce que vous voyez :**"); st.markdown(f"_{msg_view}_")
                fig_view, ax_v = plt.subplots(figsize=(3, 3)); ax_v.set_xlim(0, 100); ax_v.set_ylim(0, 100); ax_v.axis('off'); ax_v.add_patch(patches.Rectangle((0,0), 100, 100, color='whitesmoke'))
                if pos_obs == "Centre": ax_v.add_patch(patches.Circle((50, 50), 25, color='purple'))
                elif pos_obs == "Gauche": ax_v.add_patch(patches.Rectangle((50, 0), 50, 100, color='gray')); ax_v.text(75, 50, "MUR D", color='white', ha='center', va='center', rotation=90, fontweight='bold'); ax_v.add_patch(patches.Circle((20, 50), 15, color='purple'))
                elif pos_obs == "Droite": ax_v.add_patch(patches.Rectangle((0, 0), 50, 100, color='gray')); ax_v.text(25, 50, "MUR G", color='white', ha='center', va='center', rotation=90, fontweight='bold'); ax_v.add_patch(patches.Circle((80, 50), 15, color='purple'))
                ax_v.set_title("Votre Rétine", fontsize=9); st.pyplot(fig_view)
        
        st.divider(); st.markdown("### 2. Application Technique : Les Antennes")
        st.write("En réalité, nous utilisons des antennes (bobines) disposées autour du patient. Voici leur 'sensibilité' (ce qu'elles voient le mieux, en rouge).")
        fig_coils, axes_coils = plt.subplots(2, 2, figsize=(6, 6))
        mask_tl = np.zeros((S, S)); mask_tl[:S//2, :S//2] = 1; mask_tl = gaussian_filter(mask_tl, sigma=S/8); axes_coils[0,0].imshow(mask_tl, cmap='hot'); axes_coils[0,0].axis('off'); axes_coils[0,0].set_title("Antenne H-G (Haut-Gauche)")
        mask_tr = np.zeros((S, S)); mask_tr[:S//2, S//2:] = 1; mask_tr = gaussian_filter(mask_tr, sigma=S/8); axes_coils[0,1].imshow(mask_tr, cmap='hot'); axes_coils[0,1].axis('off'); axes_coils[0,1].set_title("Antenne H-D (Haut-Droite)")
        mask_bl = np.zeros((S, S)); mask_bl[S//2:, :S//2] = 1; mask_bl = gaussian_filter(mask_bl, sigma=S/8); axes_coils[1,0].imshow(mask_bl, cmap='hot'); axes_coils[1,0].axis('off'); axes_coils[1,0].set_title("Antenne B-G (Bas-Gauche)")
        mask_br = np.zeros((S, S)); mask_br[S//2:, S//2:] = 1; mask_br = gaussian_filter(mask_br, sigma=S/8); axes_coils[1,1].imshow(mask_br, cmap='hot'); axes_coils[1,1].axis('off'); axes_coils[1,1].set_title("Antenne B-D (Bas-Droite)")
        st.pyplot(fig_coils); st.caption("Ces cartes de sensibilité sont l'équivalent des 'points de vue' différents dans l'analogie de la fenêtre.")
        
        st.divider(); st.markdown("### 3. Le Secret : Démêler les pixels")
        st.info("""**Le Secret de la Reconstruction :**\n1. **Le Problème :** À cause des lignes sautées, l'image est repliée. Un pixel de l'image finale contient en réalité la somme de plusieurs points du patient (ex: le nez superposé sur l'oreille).\n2. **La Solution :** L'ordinateur regarde ce pixel avec l'Antenne 1, puis avec l'Antenne 2, etc.\n3. **Le Calcul :** Comme l'Antenne 1 "entend" fort le nez mais pas l'oreille, et que l'Antenne 2 fait l'inverse, on peut résoudre une équation pour séparer le nez de l'oreille !""")
        st.markdown("### 4. Résultat Final (Reconstruction SENSE)")
        k_full = np.fft.fftshift(np.fft.fft2(final)); mask_undersample = np.zeros((S, S)); mask_undersample[::ipat_factor, :] = 1; k_under = k_full * mask_undersample; img_aliased = np.abs(np.fft.ifft2(np.fft.ifftshift(k_under))); noise_sense = np.random.normal(0, noise_level * np.sqrt(ipat_factor) * 1.5, (S,S)); img_sense = np.clip(final + noise_sense, 0, 1.3)
        c_vis1, c_vis2, c_vis3 = st.columns(3)
        with c_vis1: st.caption("Espace K (Lignes sautées)"); fig_ksub, ax_ksub = plt.subplots(); ax_ksub.imshow(mask_undersample, cmap='gray', aspect='auto'); ax_ksub.axis('off'); st.pyplot(fig_ksub)
        with c_vis2: st.caption(f"Image Brute (Repliement x{ipat_factor})"); fig_alias, ax_alias = plt.subplots(); ax_alias.imshow(img_aliased, cmap='gray'); ax_alias.axis('off'); st.pyplot(fig_alias)
        with c_vis3: st.caption("Image Reconstruite (SENSE)"); fig_sense, ax_sense = plt.subplots(); ax_sense.imshow(img_sense, cmap='gray', vmin=0, vmax=1.3); ax_sense.axis('off'); st.pyplot(fig_sense)

with t10:
    st.header("🧬 Théorie de la Diffusion (DWI)")
    st.markdown("""L'imagerie de diffusion est unique car elle sonde le **mouvement microscopique** des molécules d'eau (Mouvement Brownien).""")
    st.divider()
    
    st.subheader("1. Isotropie vs Anisotropie")
    st.write("La capacité de l'eau à se déplacer dépend des obstacles qu'elle rencontre (membranes, fibres).")
    fig_iso, ax_iso = plt.subplots(1, 2, figsize=(6, 2))
    ax_iso[0].set_title("Isotrope (LCR)"); ax_iso[0].add_patch(patches.Circle((0.5, 0.5), 0.3, color='lightblue', alpha=0.3)); ax_iso[0].text(0.5, 0.5, "H2O", ha='center', va='center', fontweight='bold')
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        rad = np.radians(angle); dx, dy = np.cos(rad)*0.25, np.sin(rad)*0.25; ax_iso[0].arrow(0.5, 0.5, dx, dy, head_width=0.05, color='blue')
    ax_iso[0].axis('off')
    ax_iso[1].set_title("Anisotrope (Fibre)"); ax_iso[1].add_patch(patches.Rectangle((0.1, 0.3), 0.8, 0.05, color='orange', alpha=0.5)); ax_iso[1].add_patch(patches.Rectangle((0.1, 0.65), 0.8, 0.05, color='orange', alpha=0.5)); ax_iso[1].text(0.5, 0.8, "Fibre Nerveuse", ha='center', color='orange')
    ax_iso[1].text(0.5, 0.5, "H2O", ha='center', va='center', fontweight='bold')
    ax_iso[1].arrow(0.5, 0.5, 0.3, 0, head_width=0.05, color='blue'); ax_iso[1].arrow(0.5, 0.5, -0.3, 0, head_width=0.05, color='blue')
    ax_iso[1].arrow(0.5, 0.5, 0, 0.1, head_width=0.03, color='red', alpha=0.5); ax_iso[1].arrow(0.5, 0.5, 0, -0.1, head_width=0.03, color='red', alpha=0.5)
    ax_iso[1].axis('off')
    st.pyplot(fig_iso)
    
    st.divider()
    
    st.subheader("2. La Construction de l'Image Trace")
    fig_trace, axes_trace = plt.subplots(1, 4, figsize=(10, 1.5))
    titles = ["Axe X (G-D)", "Axe Y (A-P)", "Axe Z (H-B)", "IMAGE TRACE"]
    for i, ax in enumerate(axes_trace):
        ax.set_facecolor('black'); ax.set_title(titles[i], color='white', fontsize=9); ax.axis('off')
        ax.add_patch(patches.Ellipse((0.5, 0.5), 0.8, 0.9, color='#333333')) 
        ax.add_patch(patches.Circle((0.5, 0.5), 0.15, color='white')) 
        ax.text(0.5, 0.5, "AVC", color='red', ha='center', va='center', fontweight='bold', fontsize=8)
        if i == 0:
            ax.arrow(0.1, 0.5, 0.8, 0, head_width=0.05, color='yellow', linewidth=2, alpha=0.7) 
            ax.add_patch(patches.Rectangle((0.2, 0.7), 0.2, 0.1, color='white')); ax.text(0.3, 0.65, "Rect", color='cyan', ha='center', fontsize=8)
        elif i == 1:
            ax.arrow(0.5, 0.1, 0, 0.8, head_width=0.05, color='yellow', linewidth=2, alpha=0.7) 
            ax.add_patch(patches.Rectangle((0.7, 0.3), 0.15, 0.15, color='white')); ax.text(0.77, 0.25, "Carré", color='cyan', ha='center', fontsize=8)
        elif i == 2:
            ax.add_patch(patches.Circle((0.5, 0.5), 0.3, fill=False, edgecolor='yellow', linewidth=3)) 
            ax.add_patch(patches.Circle((0.5, 0.5), 0.05, color='yellow')) 
            ax.add_patch(patches.Polygon([[0.2, 0.3], [0.3, 0.5], [0.4, 0.3]], color='white')); ax.text(0.3, 0.25, "Tri", color='cyan', ha='center', fontsize=8)
        elif i == 3:
            ax.text(0.5, 0.85, "TRACE (Somme)", color='lime', ha='center', fontsize=9, fontweight='bold')
    st.pyplot(fig_trace)
    st.info("""**Processus d'acquisition :**\n1.  **Image b=0 :** Référence T2 (sans diffusion).\n2.  **Image X, Y, Z :** Sensibles au mouvement dans chaque axe.\n👉 **Image Trace (Finale) = Combinaison des 3 axes.**""")
    
    st.divider()
    
    st.subheader("3. La Dynamique du Facteur b (Constante b)")
    st.markdown("La valeur de **b** ($s/mm^2$) détermine la sensibilité au mouvement. Plus **b** est grand, plus on \"écrase\" le signal de l'eau qui bouge.")
    col_b_graph, col_b_text = st.columns([2, 1])
    with col_b_graph:
        fig_decay, ax_decay = plt.subplots(figsize=(6, 3))
        b_vals_sim = np.linspace(0, 3000, 100)
        sig_csf = 0.9 - (b_vals_sim * 0.00028); sig_csf = np.maximum(sig_csf, 0)
        sig_tissue = 0.5 - (b_vals_sim * 0.0001); sig_tissue = np.maximum(sig_tissue, 0)
        sig_lesion = 0.6 - (b_vals_sim * 0.00005); sig_lesion = np.maximum(sig_lesion, 0)
        ax_decay.plot(b_vals_sim, sig_csf, color='deepskyblue', label='LCR (Eau Libre)', linewidth=2)
        ax_decay.plot(b_vals_sim, sig_tissue, color='orange', label='Tissu Sain', linewidth=2)
        ax_decay.plot(b_vals_sim, sig_lesion, color='hotpink', label='Lésion (Restriction)', linewidth=2)
        ax_decay.set_xlabel("Facteur b (s/mm²)")
        ax_decay.set_ylabel("Signal (Log)")
        ax_decay.set_title("Décroissance du Signal (Modèle Simplifié)")
        ax_decay.legend(); ax_decay.grid(True, linestyle='--', alpha=0.3); ax_decay.set_facecolor('#f0f0f0')
        st.pyplot(fig_decay)
    with col_b_text:
        st.info("""**Analyse :**\n* 🟦 **LCR :** Pente raide.\n* 🟧 **Tissu :** Pente moyenne.\n* 🟥 **Lésion :** Pente douce.""")
    st.markdown("**Visualisation : Évolution de l'image quand b augmente**")
    fig_b_series, axes_b = plt.subplots(1, 4, figsize=(10, 2))
    b_steps = [0, 500, 1000, 2000]
    for i, b_step in enumerate(b_steps):
        ax = axes_b[i]; ax.set_facecolor('black'); ax.axis('off')
        ax.set_title(f"b = {b_step}", color='white', fontsize=10)
        tissue_decay = np.exp(-b_step * 0.0007); col_tissue = str(0.4 * tissue_decay)
        ax.add_patch(patches.Ellipse((0.5, 0.5), 0.8, 0.9, color=col_tissue))
        csf_decay = np.exp(-b_step * 0.0025); col_csf = str(0.9 * csf_decay)
        ax.add_patch(patches.Polygon([[0.45, 0.5], [0.55, 0.5], [0.5, 0.4]], color=col_csf)) 
        ax.add_patch(patches.Polygon([[0.45, 0.5], [0.55, 0.5], [0.5, 0.6]], color=col_csf))
        lesion_decay = np.exp(-b_step * 0.0003); col_lesion = str(0.8 * lesion_decay)
        ax.add_patch(patches.Circle((0.3, 0.4), 0.1, color=col_lesion))
        if i == 0: ax.text(0.3, 0.25, "AVC", color='hotpink', ha='center', fontsize=8)
    st.pyplot(fig_b_series)

    st.divider()
    st.subheader("4. Coefficient de Diffusion Apparent (ADC)")
    st.markdown("**ADC = Coefficient de Diffusion Apparent**")
    st.markdown("""### 💡 La Métaphore du "Compteur de Vitesse"\nOubliez les mathématiques complexes. Voyez l'ADC comme une **carte de vitesse** des molécules d'eau :\n* **Eau "Libre" (Rapide) 🐇 :** LCR, Œdème. Elle bouge vite -> **ADC ÉLEVÉ (Blanc)**.\n* **Eau "Coincée" (Lente) 🐢 :** AVC. Elle bouge peu -> **ADC FAIBLE (Noir)**.""")
    fig_adc, ax = plt.subplots(1, 2, figsize=(8, 1.5))
    ax[0].set_facecolor('black'); ax[0].axis('off'); ax[0].set_title("SCÉNARIO 1 : AVC (Restriction)", color='lime', weight='bold', fontsize=9)
    ax[0].text(0.3, 0.8, "b=1000", color='black', ha='center', fontsize=8, fontweight='bold')
    ax[0].text(0.7, 0.8, "Map ADC", color='black', ha='center', fontsize=8, fontweight='bold')
    ax[0].add_patch(patches.Circle((0.3, 0.5), 0.15, edgecolor='red', facecolor='white', linewidth=4)) 
    ax[0].text(0.3, 0.25, "DWI", color='white', ha='center', fontweight='bold', fontsize=7)
    ax[0].text(0.5, 0.5, "➔", color='white', fontsize=12, ha='center', va='center')
    ax[0].add_patch(patches.Circle((0.7, 0.5), 0.15, edgecolor='red', facecolor='black', linewidth=4)) 
    ax[0].text(0.7, 0.25, "ADC (Noir)", color='white', ha='center', fontweight='bold', fontsize=7)
    ax[1].set_facecolor('black'); ax[1].axis('off'); 
    # MODIF V734 - "Shine-through" -> "Liquide"
    ax[1].set_title("SCÉNARIO 2 : LCR (Liquide)", color='red', weight='bold', fontsize=9)
    ax[1].text(0.3, 0.8, "b=1000", color='black', ha='center', fontsize=8, fontweight='bold')
    ax[1].text(0.7, 0.8, "Map ADC", color='black', ha='center', fontsize=8, fontweight='bold')
    ax[1].add_patch(patches.Circle((0.3, 0.5), 0.15, edgecolor='red', facecolor='white', linewidth=4)) 
    ax[1].text(0.3, 0.25, "DWI", color='white', ha='center', fontweight='bold', fontsize=7)
    ax[1].text(0.5, 0.5, "➔", color='white', fontsize=12, ha='center', va='center')
    ax[1].add_patch(patches.Circle((0.7, 0.5), 0.15, edgecolor='red', facecolor='white', linewidth=4)) 
    ax[1].text(0.7, 0.25, "ADC (Blanc)", color='white', ha='center', fontweight='bold', fontsize=7)
    st.pyplot(fig_adc)

# [NOUVEL ONGLET 11 : COURS]
with t11:
    st.header("🎓 Cours Théorique (Support PPT)")
    slides = [f"Slide {i+1}" for i in range(5)]
    if 'slide_index' not in st.session_state: st.session_state.slide_index = 0
    st.session_state.slide_index = st.select_slider("Diapositive", options=range(len(slides)), value=st.session_state.slide_index, format_func=lambda x: slides[x])
    
    current_slide = slides[st.session_state.slide_index]
    st.markdown(f"### 📄 {current_slide}")
    fig_ppt, ax_ppt = plt.subplots(figsize=(10, 6)); ax_ppt.text(0.5, 0.5, f"CONTENU DU COURS\n\n(Diapositive: {current_slide})", ha='center', va='center', fontsize=20, color='gray'); ax_ppt.set_facecolor('#f0f0f5'); ax_ppt.axis('off'); st.pyplot(fig_ppt)

# [TAB 12 : SWI & DIPOLE - RESTRUCTURÉ V5.31]
with t12:
    st.header("🩸 Séquence SWI (Susceptibility Weighted Imaging)")
    
    # 1. Principe ASL (MODIF V738 : Image Externe Path Robuste)
    st.subheader("1. 🧲 Le Laboratoire du Dipôle")
    
    col_dip_ctrl, col_dip_visu = st.columns([1, 3])
    with col_dip_ctrl:
        dipole_substance = st.radio("Substance :", ["Hématome (Paramagnétique)", "Calcium (Diamagnétique)"])
        dipole_system = st.radio("Convention Phase :", ["RHS (GE/Philips/Canon)", "LHS (Siemens)"])
        st.divider()
        st.markdown("#### ↕️ Position Coupe Axiale")
        z_pos = st.slider("Coupe Axiale (Z)", -1.5, 1.5, 0.0, 0.1, help="0 = Equateur du dipôle. +/- = Lobes (Pôles).")
        
    with col_dip_visu:
        fig_dip, axes_dip = plt.subplots(1, 2, figsize=(10, 4))
        fig_dip.patch.set_facecolor('#404040')
        
        is_rhs = "RHS" in dipole_system
        is_para = "Hématome" in dipole_substance
        
        # Logique V5.31 (V4.991) : Inversion pour correspondre à la photo utilisateur
        sign_sub = 1 if is_para else -1
        sign_sys = 1 if is_rhs else -1
        combo = sign_sub * sign_sys
        
        if combo > 0:
            col_equator_center = 'white'; col_equator_halo = 'black'; col_poles = 'black'
        else:
            col_equator_center = 'black'; col_equator_halo = 'white'; col_poles = 'white'
            
        axes_dip[0].set_title("Vue Coronale (Référence)", fontsize=10, color='white')
        axes_dip[0].set_facecolor('#404040'); axes_dip[0].axis('off')
        axes_dip[0].add_patch(patches.Ellipse((0.5, 0.7), 0.25, 0.35, color=col_poles, alpha=0.9))
        axes_dip[0].add_patch(patches.Ellipse((0.5, 0.3), 0.25, 0.35, color=col_poles, alpha=0.9))
        axes_dip[0].add_patch(patches.Rectangle((0.35, 0.48), 0.3, 0.04, color=col_equator_center))
        y_line = 0.5 - (z_pos * 0.2)
        axes_dip[0].axhline(y=y_line, color='yellow', linewidth=2, linestyle='--')
        axes_dip[0].text(0.1, y_line, "Coupe", color='yellow', va='bottom', fontsize=8)

        axes_dip[1].set_title(f"Vue Axiale (à Z={z_pos})", fontsize=10, color='white')
        axes_dip[1].set_facecolor('#404040'); axes_dip[1].axis('off')
        
        if abs(z_pos) < 0.2:
            axes_dip[1].add_patch(patches.Circle((0.5, 0.5), 0.35, color=col_equator_halo, alpha=0.5))
            axes_dip[1].add_patch(patches.Circle((0.5, 0.5), 0.15, color=col_equator_center))
            axes_dip[1].text(0.5, 0.1, "EQUATEUR", color='white', ha='center', fontsize=9)
        elif 0.2 <= abs(z_pos) < 1.0:
            size = 0.25 * (1.2 - abs(z_pos))
            axes_dip[1].add_patch(patches.Circle((0.5, 0.5), size, color=col_poles))
            axes_dip[1].text(0.5, 0.1, "LOBE (Pôle)", color='white', ha='center', fontsize=9)
        else:
            axes_dip[1].text(0.5, 0.5, "Hors Champ", color='gray', ha='center', fontsize=12)

        st.pyplot(fig_dip)
    
    st.divider()
    
    # 2. IMAGERIE CLINIQUE
    st.subheader("2. Imagerie SWI Clinique")
    
    if HAS_NILEARN and processor.ready:
        dims = processor.get_dims() 
        c1_swi, c2_swi = st.columns([1, 4])
        with c1_swi:
             st.markdown("#### 🔄 Navigation")
             swi_view = st.radio("Plan de Coupe :", ["Axiale", "Coronale", "Sagittale"], key="swi_view_mode")
             st.markdown("---")
             if swi_view == "Axiale":
                 swi_slice = st.slider("Position Z", 0, dims[2]-1, 90, key="swi_z"); axis_code = 'z'
             elif swi_view == "Coronale":
                 swi_slice = st.slider("Position Y", 0, dims[1]-1, 100, key="swi_y"); axis_code = 'y'
             else: 
                 swi_slice = st.slider("Position X", 0, dims[0]-1, 90, key="swi_x"); axis_code = 'x'
             
             show_microbleeds_swi = st.checkbox("Simuler Micro-saignements", value=False)
             st.markdown("---")
             show_dipole_test = st.checkbox("🧪 Simuler Dipôle (Test)", value=False)
             st.success(f"Mode : {swi_view}")

        with c2_swi:
            sys_arg = "RHS" if "RHS" in dipole_system else "LHS"
            sub_arg = dipole_substance 
            
            img_mag = processor.get_slice(axis_code, swi_slice, {}, swi_mode='mag', te=te, with_bleeds=show_microbleeds_swi)
            img_phase = processor.get_slice(axis_code, swi_slice, {}, swi_mode='phase', with_bleeds=show_microbleeds_swi, swi_sys=sys_arg, swi_sub=sub_arg, with_dipole=show_dipole_test)
            img_minip = processor.get_slice(axis_code, swi_slice, {}, swi_mode='minip', te=te, with_bleeds=show_microbleeds_swi)
            
            c_mag, c_pha, c_min = st.columns(3)
            with c_mag: st.caption(f"1. Magnitude ({swi_view})"); st.image(apply_window_level(img_mag, 1.0, 0.5), clamp=True, use_container_width=True)
            with c_pha: st.caption(f"2. Phase ({sub_arg} - {sys_arg})"); st.image(apply_window_level(img_phase, 1.0, 0.5), clamp=True, use_container_width=True)
            with c_min: st.caption(f"3. MinIP Veineux ({swi_view})"); st.image(apply_window_level(img_minip, 1.0, 0.5), clamp=True, use_container_width=True)
            
            if show_dipole_test:
                if swi_view == "Axiale": st.info("ℹ️ **Dipôle (Test) :** Coupe Équatoriale (Rond). La couleur dépend du réglage ci-dessus.")
                else: st.info("ℹ️ **Dipôle (Test) :** Coupe Longitudinale (Papillon). La couleur dépend du réglage ci-dessus.")
    else: st.warning("Module Anatomique requis.")

# [NOUVEL ONGLET 13 : MP-RAGE (3D T1) - MISE A JOUR V5.31]
with t13:
    st.header("🧠 Séquence 3D T1 (MP-RAGE)")
    col_mp_ctrl, col_mp_plot = st.columns([1, 2])
    with col_mp_ctrl:
        constructeur_mp = st.radio("Logiciel Constructeur :", ["SIEMENS", "GE", "PHILIPS", "CANON"])
        st.info("Regardez comment la notion de 'TR' change sur le schéma.")
    with col_mp_plot:
        fig_mp, ax_mp = plt.subplots(figsize=(10, 3))
        ax_mp.set_title(f"Structure MP-RAGE - Définition {constructeur_mp}")
        ti_mp = 900; train_len = 500; recovery = 800; tr_global = ti_mp + train_len + recovery; tr_echo = 8 
        end_train = ti_mp + train_len
        
        # 1. Barre Inversion
        ax_mp.bar(0, 1, width=50, color='red', label='Inversion 180°')
        ax_mp.text(ti_mp/2, 0.6, f'TI = {ti_mp}ms', ha='center', color='black')
        ax_mp.annotate('', xy=(ti_mp, 0.5), xytext=(0, 0.5), arrowprops=dict(arrowstyle='<->', color='black'))

        # 2. Train d'échos (Lecture)
        for k in range(0, train_len, 50): ax_mp.bar(ti_mp + k, 0.6, width=10, color='blue', alpha=0.5)
        ax_mp.text(ti_mp + train_len/2, 0.8, 'Train d\'échos (Lecture)', ha='center', color='blue')
        
        # [MODIF V5.31] Flèches Strictes
        if constructeur_mp == "SIEMENS":
            # Flèche Horizontale Bidirectionnelle : De Inversion (0) à Fin du Train (end_train)
            # Positionnée en bas
            y_arrow = -0.5
            ax_mp.annotate('', xy=(end_train, y_arrow), xytext=(0, y_arrow), arrowprops=dict(arrowstyle='<->', color='green', lw=2))
            ax_mp.text(end_train/2, y_arrow - 0.3, "TR = TI + Train (Définition spécifique)", color='green', weight='bold', ha='center')
            
        else:
            # Flèche entre deux échos (Inter-echo spacing)
            # Positionnée SOUS les barres bleues
            echo_1 = ti_mp
            echo_2 = ti_mp + 50 
            y_arrow = 0.0 # Juste sous les barres (qui vont de 0 à 0.6)
            
            ax_mp.annotate('', xy=(echo_2, y_arrow), xytext=(echo_1, y_arrow), arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
            ax_mp.text(echo_2 + 40, y_arrow, f'TR (~{tr_echo}ms)', color='orange', weight='bold', va='center')
            
            # Note grise en bas (Optionnel, mais utile)
            # ax_mp.text(tr_global/2, -0.6, f'Cycle complet nommé "Prep Time"', color='gray', ha='center')

        ax_mp.set_ylim(-1.0, 1.5); ax_mp.set_xlabel("Temps (ms)"); ax_mp.get_yaxis().set_visible(False); ax_mp.legend(loc='upper right')
        st.pyplot(fig_mp)

# [NOUVEL ONGLET 14 : ASL - MODIFIE V738 (PATH ROBUSTE)]
with t14:
    st.header("🩸 Perfusion ASL (Arterial Spin Labeling)")
    
    # 1. Principe ASL (MODIF V738 : Image Externe Path Robuste)
    st.subheader("1. Principe de la Technique")
    c_principe, c_texte = st.columns([1, 1])
    with c_principe:
        # Code "Golden Fix" pour charger l'image utilisateur
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_asl_path = os.path.join(current_dir, "image_028fa1.jpg")
        
        if os.path.exists(image_asl_path):
            st.image(image_asl_path, caption="Principe ASL : Marquage et Acquisition", use_container_width=True)
        else:
            # Fallback en cas d'oubli de l'image (Affiche une erreur utile)
            st.error(f"Image introuvable au chemin : {image_asl_path}")
            st.markdown(f"*Vérifiez que le fichier 'image_028fa1.jpg' est bien dans le dossier : {current_dir}*")

    with c_texte:
        st.markdown("""
        ### Comment ça marche ?
        1.  **Marquage (Tag) :** Une impulsion radiofréquence est appliquée au niveau du cou (rectangle jaune). Elle "retourne" l'aimantation des protons du sang artériel qui monte vers le cerveau.
        2.  **Délai (PLD) :** On attend un temps précis (Post Labeling Delay) pour laisser le sang marqué arriver dans les capillaires cérébraux.
        3.  **Acquisition :** On prend une image du cerveau.
        4.  **Soustraction :** On soustrait cette image d'une image "Contrôle" (sans marquage). La différence ne montre que le sang qui est arrivé : c'est la perfusion !
        """)

    st.divider()

    # 2. Simulation Clinique
    st.subheader("2. Simulation Clinique & Pathologies")
    
    if HAS_NILEARN and processor.ready:
        c1_asl, c2_asl = st.columns([1, 4])
        with c1_asl:
            st.markdown("#### ⚙️ Paramètres")
            asl_slice = st.slider("Coupe Axiale (Z)", 0, dims[2]-1, 90, key="asl_z")
            st.info(f"PLD Actuel : {pld} ms")
            st.markdown("---")
            st.markdown("#### 🚑 Pathologies")
            if show_stroke: st.error("⚠️ **AVC Ischémique**")
            if show_atrophy: st.warning("🧠 **Atrophie (Alzheimer)**")
        
        with c2_asl:
            ctrl_img, label_img, perf_map = processor.get_asl_maps('z', asl_slice, pld, 1600, with_stroke=show_stroke, with_atrophy=show_atrophy)
            
            if ctrl_img is not None:
                col_ctrl, col_label, col_perf = st.columns(3)
                with col_ctrl:
                    st.image(apply_window_level(ctrl_img, 1.0, 0.5), caption="1. Image Contrôle (Anatomie)", clamp=True, use_container_width=True)
                with col_label:
                    st.image(apply_window_level(label_img, 1.0, 0.5), caption="2. Image Marquée (Sang 'Noir')", clamp=True, use_container_width=True)
                with col_perf:
                    fig_perf, ax_perf = plt.subplots()
                    im = ax_perf.imshow(perf_map, cmap='jet', vmin=0, vmax=np.max(perf_map)*0.8)
                    ax_perf.axis('off')
                    st.pyplot(fig_perf)
                    st.caption("3. Carte de Perfusion (CBF)")
            else:
                st.warning("Erreur de calcul ASL.")
    else: st.warning("Module Anatomique requis.")