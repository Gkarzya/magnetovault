import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import streamlit.components.v1 as components
import os
from scipy.ndimage import shift, gaussian_filter
import plotly.express as px
import plotly.graph_objects as go

# IMPORTS DES MODULES LOCAUX
import constantes as cst
import utils
import physique as phy
from anatomie import AdvancedMRIProcessor, HAS_NILEARN

# CONFIG & CSS
st.set_page_config(layout="wide", page_title="Magnetovault V8.38")
utils.inject_css()
plt.style.use('seaborn-v0_8-whitegrid')

# --- FONCTIONS UTILITAIRES ---
def gaussian(x: np.ndarray, mu: float, sigma: float, amp: float) -> np.ndarray:
    """Génère un profil spectral gaussien (Fat Sat)."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

def make_phantom_subtraction(offset_x: float) -> tuple[np.ndarray, np.ndarray]:
    """Génère un fantôme simple pour tester la soustraction."""
    size = 100
    y, x = np.ogrid[:size, :size]
    center = size // 2
    fat_mask = np.sqrt((x - (center + offset_x))**2 + (y - center)**2) < 35
    lesion_mask = np.sqrt((x - (center + offset_x))**2 + (y - center)**2) < 8
    img = np.zeros((size, size))
    img[fat_mask] = 1.0
    return img, lesion_mask

def generate_sensitivity_map(shape, center_x, center_y, sigma):
    """Génère une carte de sensibilité d'antenne."""
    y, x = np.ogrid[:shape[0], :shape[1]]
    mask = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
    return mask

# --- STATE MANAGEMENT ---
if 'init' not in st.session_state:
    st.session_state.seq = 'Pondération T1'
    st.session_state.reset_count = 0
    st.session_state.atrophy_active = False 
    st.session_state.tr_force = 500.0
    st.session_state.widget_tr = 500.0
    st.session_state.mem_turbo = 1 
    st.session_state.init = True

# INITIALISATION PROCESSEUR
processor = AdvancedMRIProcessor()

# --- BARRE LATÉRALE ---
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "logo_mia.png")

if os.path.exists(logo_path): 
    st.sidebar.image(logo_path, width=280)

st.sidebar.title("Réglages Console")
if st.sidebar.button("⚠️ Reset Complet (Rafraîchir)"):
    components.html("<script>window.parent.location.reload();</script>", height=0)

# SÉLECTION SÉQUENCE
seq_key = f"seq_select_{st.session_state.reset_count}"
try: idx_def = cst.OPTIONS_SEQ.index(st.session_state.seq)
except: idx_def = 0
seq_choix = st.sidebar.selectbox("Séquence", cst.OPTIONS_SEQ, index=idx_def, key=seq_key)

defaults = cst.STD_PARAMS.get(seq_choix, cst.STD_PARAMS["Pondération T1"])
current_reset_id = st.session_state.reset_count

# LOGIQUE DE CHANGEMENT DE SÉQUENCE
if seq_choix != st.session_state.seq:
    st.session_state.seq = seq_choix
    st.session_state.tr_force = float(defaults['tr'])
    if 'widget_tr' in st.session_state: 
        st.session_state.widget_tr = float(defaults['tr'])
    te_key_current = f"te_{current_reset_id}" 
    st.session_state[te_key_current] = float(defaults['te'])
    utils.safe_rerun()

is_gre = "Gradient" in seq_choix
is_dwi = "Diffusion" in seq_choix
is_ir = "FLAIR" in seq_choix or "STIR" in seq_choix
is_swi = "SWI" in seq_choix
is_mprage = "MP-RAGE" in seq_choix
is_asl = "ASL" in seq_choix

# Paramètres
ti = 0.0
te = float(defaults['te'])
flip_angle = 90

# ==============================================================================
# 1. GÉOMÉTRIE
# ==============================================================================
st.sidebar.header("1. Géométrie")
col_ep, col_slice = st.sidebar.columns(2)

ep = col_ep.number_input("Epaisseur (mm)", min_value=1.0, max_value=10.0, value=5.0, step=0.5, key=f"ep_{current_reset_id}")
n_slices = col_slice.slider("Nb Coupes", 1, 100, 20, step=1, key=f"ns_{current_reset_id}")

if not is_dwi and not is_mprage:
    n_concats = st.sidebar.select_slider("📚 Concaténations", options=[1, 2, 3, 4], value=1, key=f"concat_{current_reset_id}")
else: 
    n_concats = 1

fov = st.sidebar.slider("FOV (mm)", 100.0, 500.0, 240.0, step=10.0, key=f"fov_{current_reset_id}")
mat = st.sidebar.select_slider("Matrice", options=[64, 128, 256, 512], value=256, key=f"mat_{current_reset_id}")

st.sidebar.subheader("Réglage Echo")
if not (is_dwi or is_asl):
    te = st.sidebar.slider("TE (ms)", 1.0, 300.0, float(defaults['te']), step=1.0, key=f"te_{current_reset_id}")
else:
    te = 90.0 if is_dwi else 15.0

# --- CALCUL DU TR AUTOMATIQUE ---
time_per_slice = te + 15.0 
min_tr_required = (n_slices * time_per_slice) / n_concats
current_tr_val = st.session_state.get('widget_tr', st.session_state.tr_force)

auto_adjusted = False 

if current_tr_val < min_tr_required and not is_asl and not is_dwi: 
    st.session_state.tr_force = min_tr_required
    st.session_state.widget_tr = min_tr_required
    auto_adjusted = True
    utils.safe_rerun()

# ==============================================================================
# 2. CHRONO (TR)
# ==============================================================================
st.sidebar.header("2. Chrono (ms)")
b_value = 0; show_stroke = False; show_atrophy = False; show_adc_map = False; show_microbleeds = False; pld = 1500 

def update_tr_from_slider():
    st.session_state.tr_force = st.session_state.widget_tr

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
    tr = st.sidebar.slider(
        "TR (ms)", 
        min_value=10.0, 
        max_value=12000.0, 
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
            def set_optimized_tr(val):
                st.session_state.tr_force = val
                st.session_state.widget_tr = val
            st.sidebar.markdown(f"""<div class="opt-box"><b>Optimisation {n_concats} Concats</b><br>TR Min : <b>{int(tr_opti)} ms</b></div>""", unsafe_allow_html=True)
            st.sidebar.button(f"📉 Appliquer TR {int(tr_opti)} ms", on_click=set_optimized_tr, args=(tr_opti,))

    if is_ir or is_mprage: ti = st.sidebar.slider("TI", 0.0, 3500.0, float(defaults['ti']), step=10.0, key=f"ti_{current_reset_id}")
    else: ti = 0.0
    
    if is_gre: flip_angle = st.sidebar.slider("Angle (°)", 5, 90, 15, key=f"fa_{current_reset_id}")
    elif is_swi: flip_angle = st.sidebar.slider("Angle (°)", 5, 40, 15, key=f"fa_{current_reset_id}"); show_microbleeds = st.sidebar.checkbox("Micro-saignements", False, key=f"cmb_{current_reset_id}")
    else: flip_angle = 90 if not is_mprage else 10

st.sidebar.header("3. Options")
nex = st.sidebar.slider("NEX", 1, 8, 1, key=f"nex_{current_reset_id}")

# --- MÉMOIRE TURBO ---
turbo = 1
if not (is_gre or is_dwi or is_swi or is_mprage or is_asl):
    def_turbo = st.session_state.mem_turbo
    turbo = st.sidebar.slider("Facteur Turbo", 1, 32, def_turbo, key=f"turbo_{current_reset_id}")
    st.session_state.mem_turbo = turbo

bw = st.sidebar.slider("Bande Passante", 50, 500, 220, 10, key=f"bw_{current_reset_id}")
es = st.sidebar.slider("Espace Inter-Echo (ES)", 2.5, 20.0, 10.0, step=2.5, key=f"es_{current_reset_id}")

st.sidebar.header("4. Imagerie Parallèle (iPAT)")
ipat_on = st.sidebar.checkbox("Activer Accélération", value=False, key=f"ipat_on_{current_reset_id}")
ipat_factor = st.sidebar.slider("Facteur R", 2, 4, 2, key=f"ipat_r_{current_reset_id}") if ipat_on else 1

st.sidebar.markdown("---")

# ==============================================================================
# MENTIONS LÉGALES & BIBLIOGRAPHIE
# ==============================================================================
with st.sidebar.expander("🛡️ Mentions Légales & Droits"):
    st.markdown("""
    **MagnétoVault Simulator © 2025**
    
    **1. Usage Pédagogique :** Ce simulateur est un outil exclusivement éducatif. Il ne doit **en aucun cas** être utilisé pour du diagnostic médical ou de la recherche clinique sur des patients.
    
    **2. Propriété Intellectuelle :** Le code source et la conception sont protégés. Toute reproduction sans accord est interdite.
    
    **3. Responsabilité :** L'auteur décline toute responsabilité quant à l'interprétation des données simulées.
    
    📧 **Contact :** [magnetovault@gmail.com](mailto:magnetovault@gmail.com)
    """)

with st.sidebar.expander("📚 Bibliographie & Crédits"):
    st.markdown("""
    L'onglet **Anatomie** repose sur des outils scientifiques open-source reconnus :
    
    * **Moteur Python :** [Nilearn](https://nilearn.github.io/) (Machine learning for Neuro-Imaging in Python).
    * **Template Géométrique :** **MNI152** (ICBM 2009c Nonlinear Asymmetric).
    * **Atlas Cortical & Sous-cortical :** [Harvard-Oxford Structural Atlases](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases) (FMRIB Centre, University of Oxford).
    * **Visualisation :** Plotly & Matplotlib.
    """)

# ==============================================================================
# CALCULS PHYSIQUES
# ==============================================================================
tr_effective = tr 

try:
    raw_ms = phy.calculate_acquisition_time(tr, mat, nex, turbo, ipat_factor, n_concats, n_slices, is_mprage)
except AttributeError:
    base_time = (tr * mat * nex) / (turbo * ipat_factor)
    if is_mprage: raw_ms = base_time * n_slices
    else: raw_ms = base_time * n_concats

final_seconds = raw_ms / 1000.0; mins = int(final_seconds // 60); secs = int(final_seconds % 60); str_duree = f"{mins} min {secs} s"

v_lcr = phy.calculate_signal(tr_effective, te, ti, cst.T_LCR['T1'], cst.T_LCR['T2'], cst.T_LCR['T2s'], cst.T_LCR['ADC'], cst.T_LCR['PD'], flip_angle, is_gre, is_dwi, b_value if is_dwi else 0)
v_wm  = phy.calculate_signal(tr_effective, te, ti, cst.T_WM['T1'], cst.T_WM['T2'], cst.T_WM['T2s'], cst.T_WM['ADC'], cst.T_WM['PD'], flip_angle, is_gre, is_dwi, b_value if is_dwi else 0)
v_gm  = phy.calculate_signal(tr_effective, te, ti, cst.T_GM['T1'], cst.T_GM['T2'], cst.T_GM['T2s'], cst.T_GM['ADC'], cst.T_GM['PD'], flip_angle, is_gre, is_dwi, b_value if is_dwi else 0)
v_stroke = phy.calculate_signal(tr_effective, te, ti, cst.T_STROKE['T1'], cst.T_STROKE['T2'], cst.T_STROKE['T2s'], cst.T_STROKE['ADC'], cst.T_STROKE['PD'], flip_angle, is_gre, is_dwi, b_value if is_dwi else 0)

if is_dwi and b_value >= 1000 and show_stroke: v_stroke = 2.0 
v_fat = phy.calculate_signal(tr_effective, te, ti, cst.T_FAT['T1'], cst.T_FAT['T2'], cst.T_FAT['T2s'], cst.T_FAT['ADC'], cst.T_FAT['PD'], flip_angle, is_gre, is_dwi, 0) if not is_dwi else 0.0

snr_tr_ref = float(defaults['tr']); snr_te_ref = float(defaults['te'])
v_wm_snr = phy.calculate_signal(snr_tr_ref, snr_te_ref, ti, cst.T_WM['T1'], cst.T_WM['T2'], cst.T_WM['T2s'], cst.T_WM['ADC'], cst.T_WM['PD'], 90, False, False, 0)
ref_wm_signal = phy.calculate_signal(snr_tr_ref, snr_te_ref, ti, cst.T_WM['T1'], cst.T_WM['T2'], cst.T_WM['T2s'], cst.T_WM['ADC'], cst.T_WM['PD'], 90, False, False, 0)
snr_val = phy.calculate_snr_relative(mat, nex, turbo, ipat_factor, bw, fov, ep, v_wm_snr, ref_wm_signal)
str_snr = f"{snr_val:.1f} %"

# GENERATION FANTOME
S = mat; x = np.linspace(-1, 1, S); y = np.linspace(-1, 1, S); X, Y = np.meshgrid(x, y); D = np.sqrt(X**2 + Y**2)
img_water = np.zeros((S, S)); img_fat = np.zeros((S, S))

val_lcr_phantom = v_lcr; val_wm_phantom = v_wm; val_gm_phantom = v_gm; val_stroke_phantom = v_stroke; val_fat_phantom = v_fat

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

shift_pixels = 0.0 if bw == 220 else 220.0 / float(bw)
img_fat_shifted = shift(img_fat, [0, shift_pixels], mode='constant', cval=0.0)
final = np.clip(img_water + img_fat_shifted, 0, 1.3)
noise_level = 5.0 / (snr_val + 20.0) 
final += np.random.normal(0, noise_level, (S,S)); final = np.clip(final, 0, 1.3)
f = np.fft.fftshift(np.fft.fft2(final)); kspace = 20 * np.log(np.abs(f) + 1)

# --- 13. AFFICHAGE FINAL ---
st.title("Simulateur MagnétoVault V8.38")

# ONGLETS (16 onglets : V8.38 + Sécurité)
# Note : t4 est sauté comme dans votre demande (Fusion Espace K & Codage)
t_home, t1, t2, t3, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16 = st.tabs([
    "🏠 Accueil", 
    "Fantôme", 
    "🌀 Espace K & Codage", 
    "Signaux", 
    "🧠 Anatomie", 
    "📈 Physique", 
    "⚡ Chronogramme", 
    "☣️ Artefacts", 
    "🚀 Imagerie Parallèle", 
    "🧬 Diffusion", 
    "🎓 Cours", 
    "🩸 SWI & Dipôle", 
    "3D T1 (MP-RAGE)", 
    "ASL (Perfusion)", 
    "🍔 Fat Sat",
    "🔥 Sécurité (SAR/B1+RMS)"
])

# [TAB 0 : ACCUEIL]
with t_home:
    st.markdown("""
    <div style="background-color:#1e293b; padding:20px; border-radius:10px; margin-bottom:20px;">
        <h1 style="color:white; margin:0;">🧲 MagnétoVault Simulator</h1>
        <h3 style="color:#a5b4fc; margin-top:5px;">La "Boîte Blanche" de l'IRM</h3>
        <p style="color:#cbd5e1;"><i>"Ne vous contentez pas de voir l'image. Comprenez la mécanique de sa création."</i></p>
    </div>
    """, unsafe_allow_html=True)

    c_intro1, c_intro2 = st.columns([1, 1])
    with c_intro1:
        st.markdown("### 🔍 Pourquoi ce simulateur est unique ?")
        st.markdown("""
        La plupart des simulateurs sont des "boîtes noires" : vous rentrez des paramètres, une image sort, mais vous ne savez pas pourquoi.
        
        **MagnétoVault est un laboratoire transparent.** Ici, nous ouvrons le capot de la machine pour vous montrer les mathématiques et la physique en action.
        """)
    with c_intro2:
        st.info("""
        **Objectif :** Faire le lien entre la **Physique** (Spin, Vecteurs), l'**Espace K** (Fourier) et l'**Image Clinique** (Contraste).
        """)

    st.divider()

    st.markdown("### 🧪 Ce que vous pouvez explorer")
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        st.markdown("#### 1. Mécanique de l'Espace K")
        st.markdown("""
        Visualisez l'invisible. Comment la machine remplit-elle les lignes ?
        * **Facteur Turbo (TSE) :** Voyez comment les trains d'échos sont rangés. Lequel porte le contraste ? Lequel donne les détails ?
        * **TE Effectif :** Comprenez pourquoi il est placé au centre de l'espace K.
        """)
    with col_p2:
        st.markdown("#### 2. Physique Temps Réel")
        st.markdown("""
        Pas d'images pré-calculées. Tout est généré par les équations de Bloch.
        * **TR & TE :** Modifiez-les et voyez les courbes de relaxation changer.
        * **iPAT (Imagerie Parallèle) :** Activez le facteur d'accélération et observez la perte de SNR.
        * **Artefacts :** Créez du Repliement (Aliasing) ou du Décalage Chimique.
        """)
    with col_p3:
        st.markdown("#### 3. Clinique Avancée")
        st.markdown("""
        Au-delà du T1/T2 classique. Simulez des séquences complexes :
        * **Diffusion (DWI) :** Jouez avec le *Facteur b* et la carte *ADC*.
        * **Perfusion (ASL) :** Comprenez le marquage des spins artériels.
        * **SWI :** Visualisez la Phase et la Magnitude (Effet dipôle).
        """)

    st.divider()
    st.markdown("### 🚀 Guide de Démarrage")
    st.markdown("""
    1.  **🎛️ Console (Gauche) :** C'est votre poste de pilotage. Choisissez la **Séquence**, réglez le **FOV**, la **Matrice**, le **TR/TE** et le **Facteur Turbo**.
    2.  **🌀 Espace K (Onglet 2) :** Regardez comment votre séquence remplit les données brutes.
    3.  **🧠 Anatomie (Onglet 5) :** Explorez un cerveau humain réel (Atlas *Harvard-Oxford*) et simulez des pathologies (**AVC**, **Atrophie**).
    """)
    # ... (Votre code actuel de l'accueil reste au-dessus inchangé) ...

    st.divider()
    
    # --- GLOSSAIRE DÉPLOYABLE (A RAJOUTER À LA FIN) ---
    with st.expander("📖 Glossaire Complet (Variables & Formules)", expanded=False):
        
        # 1. PHYSIQUE FONDAMENTALE
        st.markdown("### 🧲 1. Physique Fondamentale")
        col_phy1, col_phy2 = st.columns(2)
        with col_phy1:
            st.markdown("""
            * **$B_0$ (Tesla)** : Champ magnétique statique principal.
            * **$\gamma$ (Gamma)** : Rapport gyromagnétique (42.58 MHz/T).
            * **$\omega_0$ (Hz)** : Fréquence de Larmor ($\omega_0 = \gamma B_0$).
            """)
        with col_phy2:
            st.markdown("""
            * **$M_0$** : Aimantation nette à l'équilibre.
            * **$M_z$** : Aimantation longitudinale (T1).
            * **$M_{xy}$** : Aimantation transversale (T2).
            """)

        st.markdown("---")

        # 2. PROPRIÉTÉS TISSULAIRES
        st.markdown("### 🧠 2. Propriétés Tissulaires")
        col_tis1, col_tis2 = st.columns(2)
        with col_tis1:
            st.markdown("""
            * **$T1$ (ms)** : Relaxation longitudinale (Spin-Réseau).
            * **$T2$ (ms)** : Relaxation transversale (Spin-Spin).
            """)
        with col_tis2:
            st.markdown("""
            * **$T2^*$ (ms)** : T2 réel + Inhomogénéités de champ.
            * **$\rRho$ (DP)** : Densité de Protons (quantité d'H+).
            """)

        st.markdown("---")

        # 3. PARAMÈTRES SÉQUENCE
        st.markdown("### ⏱️ 3. Paramètres Séquence")
        col_seq1, col_seq2 = st.columns(2)
        with col_seq1:
            st.markdown("""
            * **$TR$ (ms)** : Temps de Répétition.
            * **$TE$ (ms)** : Temps d'Écho.
            * **$TI$ (ms)** : Temps d'Inversion.
            """)
        with col_seq2:
            st.markdown("""
            * **$alpha$ (Flip Angle)** : Angle de bascule RF.
            * **$ETL$** : Echo Train Length (Facteur Turbo).
            * **$BW$ (Hz/Px)** : Bande Passante.
            """)

        st.markdown("---")

        # 4. SÉCURITÉ
        st.markdown("### 🔥 4. Sécurité")
        col_sar1, col_sar2 = st.columns(2)
        with col_sar1:
            st.markdown("""
            * **$B_1^{+RMS}$ ($\mu T$)** : Moyenne champ RF (Risque Implants).
            * **$B_{1,peak}$** : Amplitude max instantanée.
            """)
        with col_sar2:
            st.markdown("""
            * **$SAR$ (W/kg)** : Énergie absorbée par le patient (Chauffe).
            * **$DC$ (%)** : Duty Cycle (Rapport Cyclique).
            """)

# [TAB 1 : FANTOME]
with t1:
    c1, c2 = st.columns([1, 1])
    with c1:
        # Métriques (Durée et SNR)
        k1, k2 = st.columns(2); k1.metric("⏱️ Durée", str_duree); k2.metric("📉 SNR Relatif", str_snr); st.divider()
        
        st.subheader("1. Formules & Glossaire")
        
        # Affichage conditionnel des formules
        if is_dwi: 
            st.markdown("**Formule Diffusion :**")
            st.latex(r"S = S_0 \cdot e^{-b \cdot ADC}")
        elif is_mprage: 
            st.markdown("**Temps Acquisition 3D :**")
            st.latex(r"TA = TR \times N_{Ph} \times N_{Slices} \times NEX")
        else: 
            st.markdown("**Temps Acquisition 2D :**")
            st.latex(r"TA = \frac{TR \times N_{Ph} \times NEX}{TF \times R} \times Concats")
        
        st.markdown("**Rapport Signal/Bruit (SNR) :**")
        st.latex(r"SNR \propto V_{vox} \times \sqrt{\frac{N_{Ph} \times NEX}{BW}} \times \frac{1}{g \sqrt{R}}")
        
        # GLOSSAIRE COMPLÉTÉ (C'est ici que j'ai ajouté les termes)
        with st.expander("📖 Glossaire Complet", expanded=False):
            st.markdown("""
            | Symbole | Terme Complet | Signification / Impact |
            | :--- | :--- | :--- |
            | **TA** | Temps d'Acquisition | Durée totale de la séquence. |
            | **TR** | Temps de Répétition | Temps entre deux excitations RF. |
            | **TE** | Temps d'Écho | Temps jusqu'à la lecture du signal. |
            | **N_Ph** | Lignes de Phase | Nombre de lignes à acquérir dans l'espace K (Résolution). |
            | **NEX** | Nombre d'Excitations | Moyennages. Augmente SNR ($\sqrt{N}$) et TA ($N$). |
            | **TF** | Facteur Turbo | Nombre d'échos par TR (Accélère le temps). |
            | **R** | Facteur d'Accélération | Accélération parallèle (iPAT). Divise le temps par $R$. |
            | **V_vox**| Volume du Voxel | Taille du pixel $\times$ Épaisseur. Impact massif sur le SNR. |
            | **BW** | Bande Passante | Vitesse de lecture. Haut BW = Moins de distorsion mais moins de SNR. |
            | **g** | Facteur g | Bruit géométrique lié à l'imagerie parallèle. |
            | **Concats**| Concaténations | Divisions des coupes en paquets (pour gérer le TR/SAR). |
            | **b** | Valeur b | Sensibilisation à la diffusion ($s/mm^2$). |
            | **ADC** | Coeff. Diffusion | Mobilité des molécules d'eau (Apparent Diffusion Coefficient). |
            """)

        # Alertes Pathologies
        if show_stroke: st.error("⚠️ **PATHOLOGIE : AVC Ischémique**")
        if show_atrophy: st.warning("🧠 **PATHOLOGIE : Atrophie (Alzheimer)**")

    with c2:
        fig_anot, ax_anot = plt.subplots(figsize=(5,5))
        ax_anot.imshow(final, cmap='gray', vmin=0, vmax=1.3)
        ax_anot.axis('off')
        ax_anot.text(S/2, S/2, "CSF/H2O", color='cyan', ha='center', va='center', fontsize=10, fontweight='bold')
        ax_anot.text(S/2, S/2 + (S*0.35/2), "WM", color='black', ha='center', va='center', fontsize=9, fontweight='bold')
        ax_anot.text(S/2, S/2 + (S*0.65/2), "GM", color='white', ha='center', va='center', fontsize=9, fontweight='bold')
        ax_anot.text(S/2, S*0.93, "FAT", color='orange', ha='center', va='center', fontsize=10, fontweight='bold')
        st.pyplot(fig_anot)
        plt.close(fig_anot)
# [TAB 2 : ESPACE K - TERMINOLOGIE CORRIGÉE (DÉPHASAGE)]
with t2:
    # 1. TITRE PRINCIPAL
    st.markdown("""
    <div style="background-color: #1e293b; padding: 20px; border-radius: 10px; margin-bottom: 25px; text-align: center; border-bottom: 4px solid #3b82f6;">
        <h1 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">🌀 Espace K : La Bibliothèque de l'Image</h1>
        <p style="color: #94a3b8; margin-top: 5px; font-size: 16px;">De la Fréquence au Pixel : Le voyage du signal</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. EN-TÊTE PÉDAGOGIQUE
    with st.expander("🎶 Comprendre le Codage : De la Chorale à la Physique", expanded=True):
        c_txt1, c_txt2, c_txt3 = st.columns(3)
        
        with c_txt1:
            # LE PROBLÈME
            st.markdown("""
            <div style="background-color: #eff6ff; padding: 15px; border-radius: 8px; border-left: 5px solid #3b82f6; height: 100%;">
                <h3 style="color: #1e40af; margin: 0 0 10px 0; font-size: 20px;">1. Le Problème</h3>
                <p style="font-size: 14px; color: #334155; margin: 0;"><b>Le Chaos :</b> Imaginez une <b>foule</b> où tout le monde crie "A" en même temps. Impossible de savoir qui est où. Sans codage spatial, l'IRM ne reçoit qu'un bruit global.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with c_txt2:
            # LA SOLUTION
            st.markdown("""
            <div style="background-color: #fff7ed; padding: 15px; border-radius: 8px; border-left: 5px solid #f97316; height: 100%;">
                <h3 style="color: #9a3412; margin: 0 0 10px 0; font-size: 20px;">2. La Solution</h3>
                <p style="font-size: 14px; color: #334155; margin: 0;"><b>Le Tri :</b> On applique des gradients pour "trier" les signaux :</p>
                <ul style="font-size: 13px; color: #334155; padding-left: 20px; margin-top: 5px;">
                    <li style="margin-bottom: 5px;"><b>Fréquence :</b> Trie de Gauche à Droite (Grave ↔ Aigu).</li>
                    <li><b>Phase :</b> Trie de Haut en Bas (En Avance ↔ En Retard).</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with c_txt3:
            # LE RÉSULTAT
            st.markdown("""
            <div style="background-color: #f0fdf4; padding: 15px; border-radius: 8px; border-left: 5px solid #22c55e; height: 100%;">
                <h3 style="color: #166534; margin: 0 0 10px 0; font-size: 20px;">3. La Réalité Physique</h3>
                <p style="font-size: 14px; color: #334155; margin: 0;">Pour créer l'image, la machine combine 3 axes :</p>
                <ul style="font-size: 13px; color: #334155; padding-left: 20px; margin-top: 5px;">
                    <li style="margin-bottom: 5px;"><b>Axe Z (Sélection) :</b> Isole la <b>Coupe</b> (L'épaisseur).</li>
                    <li style="margin-bottom: 5px;"><b>Axe Y (Phase) :</b> Encode les <b>Lignes</b>.</li>
                    <li><b>Axe X (Fréquence) :</b> Encode les <b>Colonnes</b>.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    st.write("") 
    
    # RÉSUMÉ
    st.markdown("""
    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <span style="font-size: 24px; vertical-align: middle;">📍</span> 
        <span style="font-size: 16px; font-weight: bold; color: #0f172a; vertical-align: middle;">
            En résumé : L'IRM est une grille 3D. Z choisit la tranche de pain, Y choisit la rangée, X choisit la colonne.
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Création des deux sous-onglets
    sub_tabs = st.tabs(["👁️ Cycle de Codage (Visualisation)", "🎨 Espace K (Remplissage)"])
    
    # SOUS-ONGLET 1 : CODAGE (HTML du main1.py)
    with sub_tabs[0]:
        st.markdown("<h3 style='color: #4f46e5; border-bottom: 2px solid #4f46e5; padding-bottom: 5px;'>🎛️ Simulateur de Codage</h3>", unsafe_allow_html=True)
        components.html("""<!DOCTYPE html><html><head><style>body{margin:0;padding:5px;font-family:sans-serif;} .box{display:flex;gap:15px;} .ctrl{width:220px;padding:10px;background:#f9f9f9;border:1px solid #ccc;border-radius:8px;} canvas{border:1px solid #ccc;background:#f8f9fa;border-radius:8px;} input{width:100%;} label{font-size:11px;font-weight:bold;display:block;} button{width:100%;padding:8px;background:#4f46e5;color:white;border:none;border-radius:4px;cursor:pointer;}</style></head><body><div class='box'><div class='ctrl'><h4>Codage</h4><label>Freq</label><input type='range' id='f' min='-100' max='100' value='0'><br><label>Phase</label><input type='range' id='p' min='-100' max='100' value='0'><br><label>Coupe</label><input type='range' id='z' min='-100' max='100' value='0'><br><label>Matrice</label><input type='range' id='g' min='5' max='20' value='12'><br><button onclick='rst()'>Reset</button></div><div><canvas id='c1' width='350' height='350'></canvas><canvas id='c2' width='80' height='350'></canvas></div></div><script>const c1=document.getElementById('c1');const x=c1.getContext('2d');const c2=document.getElementById('c2');const z=c2.getContext('2d');const sf=document.getElementById('f');const sp=document.getElementById('p');const sz=document.getElementById('z');const sg=document.getElementById('g');const pd=30;function arrow(ctx,x,y,a,s){const l=s*0.35;ctx.save();ctx.translate(x,y);ctx.rotate(a);ctx.beginPath();ctx.moveTo(-l,0);ctx.lineTo(l,0);ctx.lineTo(l-6,-6);ctx.moveTo(l,0);ctx.lineTo(l-6,6);ctx.strokeStyle='white';ctx.lineWidth=1.5;ctx.stroke();ctx.restore();} function draw(){x.clearRect(0,0,350,350);z.clearRect(0,0,80,350);const fv=parseFloat(sf.value);const pv=parseFloat(sp.value);const zv=parseFloat(sz.value);const gs=parseInt(sg.value);const st=(350-2*pd)/gs;const h=(pd*0.8)*(fv/100);x.fillStyle='rgba(255,0,0,0.3)';if(fv!=0){x.beginPath();x.moveTo(pd,pd/2);x.lineTo(pd,pd/2-h);x.lineTo(350-pd,pd/2+h);x.lineTo(350-pd,pd/2);x.fill();}const w=(pd*0.8)*(pv/100);x.fillStyle='rgba(0,255,0,0.3)';if(pv!=0){x.beginPath();x.moveTo(350-pd/2,pd);x.lineTo(350-pd/2-w,pd);x.lineTo(350-pd/2+w,350-pd);x.lineTo(350-pd/2,350-pd);x.fill();} for(let i=0;i<gs;i++){for(let j=0;j<gs;j++){const cx=pd+i*st+st/2;const cy=pd+j*st+st/2;const ph=(i-gs/2)*(fv/100)*3+(j-gs/2)*(pv/100)*3;const cph=(j-gs/2)*(pv/100);x.strokeStyle='black';x.beginPath();x.arc(cx,cy,st*0.4,0,6.28);x.fillStyle='#94a3b8';x.fill();if(cph>0.01)x.fillStyle='rgba(255,255,0,0.5)';if(cph<-0.01)x.fillStyle='rgba(0,0,255,0.5)';x.fill();arrow(x,cx,cy,ph,st*0.6);}}const yz=175-(zv/100)*150;const gr=z.createLinearGradient(0,0,0,350);gr.addColorStop(0,'red');gr.addColorStop(1,'blue');z.fillStyle=gr;z.fillRect(10,10,20,330);z.strokeStyle='black';z.lineWidth=3;z.beginPath();z.moveTo(10,yz);z.lineTo(70,yz);z.stroke();z.fillStyle='black';z.fillText('Z',35,yz-5);} [sf,sp,sz,sg].forEach(s=>s.addEventListener('input',draw));function rst(){sf.value=0;sp.value=0;sz.value=0;sg.value=12;draw();}draw();</script></body></html>""", height=450)
        
        st.divider()
        st.markdown("<h3 style='background-color: #e0e7ff; padding: 10px; border-radius: 5px; color: #3730a3;'>🧠 Synthèse : Gradient & Espace K</h3>", unsafe_allow_html=True)
        col_c1, col_c2 = st.columns(2)
        
        # --- MODIFICATION TERMINOLOGIQUE ICI ---
        with col_c1:
            st.info("**1. Gradient Faible (Lignes Centrales)**\n* Faible Déphasage = Signal Fort.\n* Contraste de l'image.")
        with col_c2:
            st.error("**2. Gradient Fort (Lignes Périphériques)**\n* Fort Déphasage = Détails fins.\n* Résolution spatiale.")
        # ---------------------------------------

    # SOUS-ONGLET 2 : ESPACE K
    with sub_tabs[1]:
        st.markdown("<h3 style='color: #db2777; border-bottom: 2px solid #db2777; padding-bottom: 5px;'>🖼️ Remplissage & Reconstruction</h3>", unsafe_allow_html=True)
        
        col_k1, col_k2 = st.columns([1, 1])
        with col_k1:
            fill_mode = st.radio("Ordre de Remplissage", ["Linéaire (Haut -> Bas)", "Centrique (Centre -> Bords)"], key=f"k_mode_{current_reset_id}")
            acq_pct = st.slider("Progression (%)", 0, 100, 10, step=1, key=f"k_pct_{current_reset_id}")
            st.divider()
            
            if turbo > 1:
                # TITRE TSE STYLISÉ
                st.markdown(f"""
                <div style="background-color: #fce7f3; padding: 10px; border-radius: 5px; border-left: 5px solid #db2777; margin-bottom: 10px;">
                    <h4 style="margin:0; color: #831843;">🚅 Rangement des {turbo} Échos (Ky)</h4>
                </div>
                """, unsafe_allow_html=True)
                st.info(f"TE Cible : **{int(te)} ms** | Facteur Turbo : **{turbo}**")
                
                # [CODE LOGIQUE IDENTIQUE - NON MODIFIÉ]
                echo_data = []
                for i in range(turbo):
                    te_real = (i + 1) * es; delta = abs(te_real - te)
                    echo_data.append({"id": i + 1, "te": te_real, "delta": delta})
                effective_echo = min(echo_data, key=lambda x: x['delta'])
                sorted_by_relevance = sorted(echo_data, key=lambda x: x['delta'])
                k_space_slots = [None] * turbo; center_idx = turbo // 2
                for i, echo in enumerate(sorted_by_relevance):
                    if i % 2 == 0: offset = i // 2
                    else: offset = -((i // 2) + 1)
                    target_slot = center_idx + offset
                    if 0 <= target_slot < turbo: k_space_slots[target_slot] = echo
                    else:
                        for k in range(turbo):
                            if k_space_slots[k] is None: k_space_slots[k] = echo; break
                fig_tse, ax = plt.subplots(figsize=(5, 4))
                y_height = 1.0 / turbo
                for idx, echo in enumerate(k_space_slots):
                    if echo is None: continue
                    color_val = (echo['id'] - 1) / max(1, (turbo - 1)); color = plt.cm.jet(color_val)
                    is_eff = (echo['id'] == effective_echo['id'])
                    rect = patches.Rectangle((0, 1.0 - (idx + 1) * y_height), 1, y_height, linewidth=3 if is_eff else 0.5, edgecolor='black' if is_eff else 'white', facecolor=color)
                    ax.add_patch(rect)
                    label = f"Echo {echo['id']} (TE={int(echo['te'])}ms)"; 
                    if is_eff: label += " ★"
                    ax.text(0.5, 1.0 - (idx + 0.5) * y_height, label, ha='center', va='center', color='white', fontweight='bold', path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
                ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
                ax.text(-0.05, 0.5, "CENTRE (K=0)", ha='right', va='center', fontweight='bold')
                ax.annotate("", xy=(-0.02, 0.4), xytext=(-0.02, 0.6), arrowprops=dict(arrowstyle="-", color="black", lw=2))
                st.pyplot(fig_tse)
                plt.close(fig_tse)
            else:
                st.markdown(f"#### 🐢 Acquisition Standard (1 Écho/TR)")
                st.info(f"TE Unique : **{int(te)} ms**")
                fig_tse, ax = plt.subplots(figsize=(5, 4))
                n_disp_lines = 24; y_h = 1.0 / n_disp_lines; color = plt.cm.jet(0)
                for i in range(n_disp_lines):
                    rect = patches.Rectangle((0, 1.0 - (i + 1) * y_h), 1, y_h, linewidth=0.5, edgecolor='white', facecolor=color)
                    ax.add_patch(rect)
                ax.text(0.5, 0.5, f"ECHO 1 (TE={int(te)}ms)\nAppliqué à chaque ligne", ha='center', va='center', color='white', fontweight='bold', fontsize=12, path_effects=[path_effects.withStroke(linewidth=3, foreground='black')])
                ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
                st.pyplot(fig_tse)
                plt.close(fig_tse)
        with col_k2:
            mask_k = np.zeros((S, S)); lines_to_fill = int(S * (acq_pct / 100.0))
            if "Linéaire" in fill_mode: mask_k[0:lines_to_fill, :] = 1
            else: center_line = S // 2; half = lines_to_fill // 2; mask_k[center_line-half:center_line+half, :] = 1
            kspace_masked = f * mask_k; img_rec = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_masked)))
            fig_k, ax_k = plt.subplots(figsize=(4, 4))
            ax_k.imshow(20 * np.log(np.abs(kspace_masked) + 1), cmap='inferno'); ax_k.axis('off')
            st.pyplot(fig_k)
            plt.close(fig_k)
            st.image(img_rec, clamp=True, width=300, caption="Reconstruction")

# [TAB 3 : SIGNAUX]
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

# [TAB 4 : ANATOMIE]
with t5:
    st.header("Exploration Anatomique (Physique Avancée)")
    if HAS_NILEARN and processor.ready:
        c1, c2 = st.columns([1, 3])
        dims = processor.get_dims()
        with c1:
            plane = st.radio("Plan de Coupe", ["Plan Axial", "Plan Sagittal", "Plan Coronal"], key="or_298")
            if "Axial" in plane: 
                idx = st.slider("Z", 0, dims[2]-1, 90, key=f"sl_{current_reset_id}"); ax='z'
            elif "Sagittal" in plane: 
                idx = st.slider("X", 0, dims[0]-1, 90, key=f"sl_{current_reset_id}"); ax='x'
            else: 
                idx = st.slider("Y", 0, dims[1]-1, 100, key=f"sl_{current_reset_id}"); ax='y'
            st.divider()
            window = st.slider("Fenêtre", 0.01, 2.0, 0.74, 0.005, key=f"wn_{current_reset_id}")
            level = st.slider("Niveau", 0.0, 1.0, 0.55, 0.005, key=f"lv_{current_reset_id}")
            st.divider()
            show_interactive_legends = st.checkbox("🔍 Activer Légendes (Atlas Harvard-Oxford)", value=False, help="Identifie les structures (Gyrus, Noyaux, Tronc, Cervelet) au survol.")
            if is_dwi: 
                if show_adc_map: st.info("🗺️ **Mode Carte ADC** (LCR Blanc)")
                else: st.success(f"🧬 **Mode Diffusion** (b={b_value})")
            if show_stroke and ax == 'z': st.error("⚠️ **AVC Visible**")
        with c2:
            w_vals = {'csf':v_lcr, 'gm':v_gm, 'wm':v_wm, 'fat':v_fat}
            if show_stroke: w_vals['wm'] = w_vals['wm'] * 0.9 + v_stroke * 0.1
            seq_type_arg = 'dwi' if is_dwi else ('gre' if is_gre else None)
            img_raw = processor.get_slice(ax, idx, w_vals, seq_type=seq_type_arg, te=te, tr=tr, fa=flip_angle, b_val=b_value, adc_mode=show_adc_map, with_stroke=show_stroke)
            if img_raw is not None:
                img_display = utils.apply_window_level(img_raw, window, level)
                if show_interactive_legends:
                    with st.spinner("Génération de la carte anatomique..."):
                        labels_map = processor.get_anatomical_labels(ax, idx)
                        fig = px.imshow(img_display, color_continuous_scale='gray', zmin=0, zmax=1, binary_string=False)
                        fig.update_traces(customdata=labels_map, hovertemplate="<b>%{customdata}</b><br>Signal: %{z:.2f}<extra></extra>")
                        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False, width=600, height=600, xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
                        st.plotly_chart(fig, config={'displayModeBar': False})
                        st.caption("ℹ️ Passez la souris sur l'image pour voir les structures.")
                else:
                    st.image(img_display, clamp=True, width=600)
    else: st.warning("Module 'nilearn' manquant ou données non chargées.")

with t6:
    st.header("📈 Physique")
    tists = [cst.T_FAT, cst.T_WM, cst.T_GM, cst.T_LCR]; cols = ['orange', 'lightgray', 'dimgray', 'cyan'] 
    if show_stroke: tists.append(cst.T_STROKE); cols.append('red') 
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
        t_max = max(tr + 40, te + 50); t = np.linspace(0, t_max, 2000); rf_sigma = 0.5; grad_width = 3.0
        fig, axs = plt.subplots(5, 1, sharex=True, figsize=(10, 8), gridspec_kw={'hspace': 0.3})
        rf = np.zeros_like(t); amp_rf = flip_angle / 90.0
        rf += amp_rf * np.exp(-0.5 * ((t - t_90)**2) / (rf_sigma**2)); t_90_next = t_90 + tr; rf += amp_rf * np.exp(-0.5 * ((t - t_90_next)**2) / (rf_sigma**2))
        axs[0].plot(t, rf, color='black'); axs[0].fill_between(t, 0, rf, color='green', alpha=0.4); axs[0].set_ylabel("RF"); axs[0].set_yticks([0, 1], ["", f"{flip_angle}°"])
        gsc = np.zeros_like(t); mask_sel = (t > t_90 - grad_width) & (t < t_90 + grad_width); gsc[mask_sel] = 1.0; mask_reph = (t > t_90 + grad_width + 1) & (t < t_90 + 2*grad_width + 1); gsc[mask_reph] = -0.8
        axs[1].plot(t, gsc, color='green'); axs[1].fill_between(t, 0, gsc, color='green', alpha=0.6); axs[1].set_ylabel("Gss")
        gcp = np.zeros_like(t); t_code = t_90 + 15; mask_c = (t > t_code - grad_width) & (t < t_code + grad_width); gcp[mask_c] = 0.5
        axs[2].plot(t, gcp, color='orange'); axs[2].fill_between(t, 0, gcp, color='orange', alpha=0.6); axs[2].set_ylabel("Gpe")
        gcf = np.zeros_like(t); t_read = t_90 + te; mask_read = (t > t_read - grad_width) & (t < t_read + grad_width); gcf[mask_read] = 1.0; t_pre = t_read - (2 * grad_width) - 2; 
        if t_pre > t_90 + grad_width: mask_pre = (t > t_pre - grad_width) & (t < t_pre + grad_width); gcf[mask_pre] = -1.0
        axs[3].plot(t, gcf, color='dodgerblue'); axs[3].fill_between(t, 0, gcf, color='dodgerblue', alpha=0.6); axs[3].set_ylabel("Gro")
        sig = np.zeros_like(t); idx_s = np.argmin(np.abs(t - (t_read - 3))); idx_e = np.argmin(np.abs(t - (t_read + 3)))
        if idx_e > idx_s: grid = np.linspace(-3, 3, idx_e - idx_s); sig[idx_s:idx_e] = np.sinc(grid)
        axs[4].plot(t, sig, color='navy'); axs[4].set_ylabel("Signal"); axs[4].axvline(x=t_read, color='red', linestyle='--'); axs[4].text(t_read, 1.1, f"TE={te:.0f}ms", color='red', ha='center')
        st.pyplot(fig); plt.close(fig)
    else:
        is_turbo = turbo > 1; t_90 = 10
        if is_dwi: st.subheader("Séquence : Diffusion (DWI - SE EPI)")
        elif is_turbo: st.subheader(f"Séquence : Turbo Spin Écho (TSE) - Facteur {turbo}")
        else: st.subheader("Séquence : Spin Écho (SE)")
        if not is_turbo: echo_times = [t_90 + te]; t_180s = [t_90 + (te/2)]; es_disp = te; t_max = max(200, t_90 + te + 50)
        else: echo_times = [t_90 + (i+1)*es for i in range(turbo)]; t_180s = []; es_disp = es; t_max = max(200, echo_times[-1] + 50)
        t = np.linspace(0, t_max, 2000); rf_sigma = 0.5; grad_width = max(1.5, es_disp * 0.2); t_180s = []; 
        for i in range(turbo): t_p = t_90 + (i * es) + (es/2); t_180s.append(t_p)
        fig, axs = plt.subplots(5, 1, sharex=True, figsize=(10, 8), gridspec_kw={'hspace': 0.3})
        rf = np.zeros_like(t)
        def add_rf_pulse(center, amp, w): return amp * np.exp(-0.5 * ((t - center)**2) / (w**2))
        rf += add_rf_pulse(t_90, 1.0, rf_sigma) 
        for t_p in t_180s:
            if t_p < t_max: rf += add_rf_pulse(t_p, 1.6, rf_sigma)
        axs[0].plot(t, rf, color='black', linewidth=1.5); axs[0].fill_between(t, 0, rf, color='green', alpha=0.4); axs[0].set_ylabel("RF"); axs[0].set_yticks([0, 1, 1.6], ["", "90", "180"])
        gsc = np.zeros_like(t)
        def add_trap(center, amp, w): mask = (t > center - w) & (t < center + w); gsc[mask] = amp
        add_trap(t_90, 1.0, grad_width); t_rephase = t_90 + grad_width + 1.5; add_trap(t_rephase, -0.8, grad_width*0.6)
        for t_p in t_180s: add_trap(t_p, 1.0, grad_width)
        axs[1].fill_between(t, 0, gsc, color='green', alpha=0.6); axs[1].plot(t, gsc, color='green', linewidth=1); axs[1].set_ylabel("Gss")
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
        axs[2].fill_between(t, 0, gcp, color='darkorange', alpha=0.7); axs[2].set_ylabel("Gpe")
        gcf = np.zeros_like(t); t_pre = (t_90 + t_180s[0])/2; add_trap_gcf = lambda c, w: ((t > c - w) & (t < c + w)); gcf[add_trap_gcf(t_pre, grad_width)] = 1.0 
        for t_e in echo_times:
            if t_e < t_max: w_read = grad_width * 1.2; gcf[add_trap_gcf(t_e, w_read)] = 1.0
        axs[3].fill_between(t, 0, gcf, color='dodgerblue', alpha=0.5); axs[3].set_ylabel("Gro")
        sig = np.zeros_like(t)
        for i, t_e in enumerate(echo_times):
            if t_e < t_max - 5:
                w_sig = grad_width * 1.2; idx_start = np.argmin(np.abs(t - (t_e - w_sig))); idx_end = np.argmin(np.abs(t - (t_e + w_sig)))
                if idx_end > idx_start:
                    grid = np.linspace(-3, 3, idx_end - idx_start); amp = np.exp(-t_e / cst.T_GM['T2']) 
                    sig[idx_start:idx_end] = np.sinc(grid) * amp
                if i == closest_idx:
                     axs[4].text(t_e, amp+0.3, "TE eff", ha='center', color='red', fontweight='bold', fontsize=10)
                     axs[4].axvline(x=t_e, color='red', linestyle='--', alpha=0.5)
        axs[4].plot(t, sig, color='navy', linewidth=1.5); axs[4].set_ylabel("Signal")
        st.pyplot(fig); plt.close(fig)

with t8:
    st.header("☣️ Laboratoire d'Artefacts")
    col_ctrl, col_visu = st.columns([1, 2])
    with col_ctrl:
        st.markdown("#### Choix de l'Artefact")
        artefact_type = st.radio("Sélectionnez :", ["Aliasing", "Décalage Chimique", "Troncature", "Mouvement", "Zipper"], key="art_main_radio")
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

# [TAB 9 : IMAGERIE PARALLÈLE - REMPLACEMENT CHIRURGICAL]
with t9:
    st.header("🚀 Imagerie Parallèle (PI)")
    
    # 0. BOUTON CACHE ET METAPHORE (Réintégré)
    show_meta = st.checkbox("👁️ Afficher le Concept (Analogie de la Fenêtre)", value=False)
    if show_meta:
        st.markdown("### 1. Analogie de la Fenêtre (Interactive)")
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
        st.divider()

    # 1. Principe & Lignes
    st.markdown("#### 1. Principe & Sous-échantillonnage")
    col_pi_info, col_pi_ctrl = st.columns([2, 1])
    with col_pi_info:
        st.info(f"**Gain de Temps :** L'acquisition est accélérée par un facteur **R = {ipat_factor}**.")
        st.warning(r"**Coût (Pénalité SNR) :** Le signal diminue de $\sqrt{R}$.")
        
        # Visualisation des Lignes (NEW : Graphique R)
        st.markdown(f"**Visualisation de l'acquisition des lignes (R={ipat_factor}) :**")
        fig_lines, ax_lines = plt.subplots(figsize=(10, 1.5))
        for i in range(25): 
            if i % ipat_factor == 0:
                ax_lines.vlines(i, 0, 1, colors='green', linewidth=3)
            else:
                ax_lines.vlines(i, 0, 1, colors='red', linestyles='dotted', linewidth=1.5)
        ax_lines.set_xlim(-1, 26); ax_lines.set_ylim(0, 1); ax_lines.axis('off')
        ax_lines.text(26, 0.5, "Vert = Acquise\nRouge = Sautée", va='center', fontsize=9)
        st.pyplot(fig_lines); plt.close(fig_lines)

    with col_pi_ctrl:
        if ipat_factor == 1: st.error("⚠️ Accélération désactivée (R=1).")
        else: st.success(f"✅ Accélération Active (R={ipat_factor})")

    st.divider()

    # 2. Antennes (Profils Couleur)
    st.markdown("#### 2. Les \"Yeux\" de la Machine (Profils de Sensibilité)")
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    h, w = final.shape; sigma_coil = h / 2.5
    centers = [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)]
    titles = ["Antenne 1 (HG)", "Antenne 2 (HD)", "Antenne 3 (BG)", "Antenne 4 (BD)"]
    cols = [col_c1, col_c2, col_c3, col_c4]
    
    # Stockage des images partielles pour reconstruction RSS
    part_imgs = []
    
    for i, (cy, cx) in enumerate(centers):
        sens = generate_sensitivity_map((h,w), h*cy, w*cx, sigma_coil)
        part_img = final * sens
        part_imgs.append(part_img)
        cols[i].image(part_img, caption=titles[i], clamp=True, use_container_width=True)
        # Profil Couleur (NEW)
        fig_s, ax_s = plt.subplots(figsize=(2, 2))
        ax_s.imshow(sens, cmap='jet', vmin=0, vmax=1); ax_s.axis('off')
        cols[i].pyplot(fig_s); plt.close(fig_s)

    # 3. La Reconstruction
    st.divider()
    st.markdown(f"#### 3. Résultat : Rempliement vs Reconstruction (R={ipat_factor})")
    c_res1, c_res2 = st.columns(2)
    
    # Calcul image RSS (Racine somme carrés) pour "Image Combinée"
    rss_img = np.sqrt(sum(img**2 for img in part_imgs))
    
    if ipat_factor > 1:
        shift_amount = int(h / ipat_factor)
        img_aliased = (final + np.roll(final, shift_amount, axis=0)) / 2.0
        # Simulation Reconstruction (Ajout bruit)
        noise_factor = np.sqrt(ipat_factor) * 1.5
        added_noise = np.random.normal(0, (5.0/(snr_val+20.0)) * noise_factor, (h, w))
        img_reconstructed = np.clip(rss_img + added_noise, 0, 1.3)
        
        c_res1.image(img_aliased, caption="Image Brute (Repliée/Aliasing)", clamp=True, use_container_width=True)
        c_res2.image(img_reconstructed, caption="Image Reconstruite (Dépliée via SENSE/GRAPPA)", clamp=True, use_container_width=True)
        # TEXTE PERTINENT AJOUTÉ
        c_res2.caption(f"⚠️ Notez l'augmentation du bruit (Grain) due au facteur R={ipat_factor} (SNR divisé par √{ipat_factor}).")
    else:
        c_res1.image(final, caption="Image de Référence (R=1)", clamp=True, use_container_width=True)
        c_res2.image(rss_img, caption="Combinaison des 4 signaux (Somme Quadratique)", clamp=True, use_container_width=True)

# [TAB 10 : DIFFUSION - VERSION FINALE VALIDÉE]
with t10:
    st.header("🧬 Théorie de la Diffusion (DWI)")
    st.markdown("""L'imagerie de diffusion est unique car elle sonde le **mouvement microscopique** des molécules d'eau.""")
    st.divider()
    
    # --- 1. CODE RESTAURÉ (ISOTROPIE & ADC) ---
    st.subheader("1. Isotropie vs Anisotropie")
    fig_iso, ax_iso = plt.subplots(1, 2, figsize=(6, 2))
    
    # Isotropie
    ax_iso[0].set_title("Isotrope (LCR)")
    ax_iso[0].add_patch(patches.Circle((0.5, 0.5), 0.3, color='lightblue', alpha=0.3))
    ax_iso[0].text(0.5, 0.5, "H2O", ha='center', va='center', fontweight='bold')
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        rad = np.radians(angle); dx, dy = np.cos(rad)*0.25, np.sin(rad)*0.25
        ax_iso[0].arrow(0.5, 0.5, dx, dy, head_width=0.05, color='blue')
    ax_iso[0].axis('off')
    
    # Anisotropie
    ax_iso[1].set_title("Anisotrope (Fibre)")
    ax_iso[1].add_patch(patches.Rectangle((0.1, 0.3), 0.8, 0.05, color='orange', alpha=0.5))
    ax_iso[1].add_patch(patches.Rectangle((0.1, 0.65), 0.8, 0.05, color='orange', alpha=0.5))
    ax_iso[1].text(0.5, 0.8, "Fibre Nerveuse", ha='center', color='orange')
    ax_iso[1].text(0.5, 0.5, "H2O", ha='center', va='center', fontweight='bold')
    ax_iso[1].arrow(0.5, 0.5, 0.3, 0, head_width=0.05, color='blue')
    ax_iso[1].arrow(0.5, 0.5, -0.3, 0, head_width=0.05, color='blue')
    ax_iso[1].arrow(0.5, 0.5, 0, 0.1, head_width=0.03, color='red', alpha=0.5)
    ax_iso[1].arrow(0.5, 0.5, 0, -0.1, head_width=0.03, color='red', alpha=0.5)
    ax_iso[1].axis('off')
    st.pyplot(fig_iso); plt.close(fig_iso)
    
    st.divider()
    
    st.subheader("2. Coefficient de Diffusion Apparent (ADC)")
    fig_adc, ax = plt.subplots(1, 2, figsize=(8, 1.5))
    
    # Scenario 1 : AVC
    ax[0].set_facecolor('black'); ax[0].axis('off')
    ax[0].set_title("SCÉNARIO 1 : AVC (Restriction)", color='lime', weight='bold', fontsize=9)
    ax[0].text(0.3, 0.8, "b=1000", color='black', ha='center', fontsize=8, fontweight='bold')
    ax[0].text(0.7, 0.8, "Map ADC", color='black', ha='center', fontsize=8, fontweight='bold')
    ax[0].add_patch(patches.Circle((0.3, 0.5), 0.15, edgecolor='red', facecolor='white', linewidth=4)) 
    ax[0].text(0.3, 0.25, "DWI", color='white', ha='center', fontweight='bold', fontsize=7)
    ax[0].text(0.5, 0.5, "➔", color='white', fontsize=12, ha='center', va='center')
    ax[0].add_patch(patches.Circle((0.7, 0.5), 0.15, edgecolor='red', facecolor='black', linewidth=4)) 
    ax[0].text(0.7, 0.25, "ADC (Noir)", color='white', ha='center', fontweight='bold', fontsize=7)
    
    # Scenario 2 : LCR
    ax[1].set_facecolor('black'); ax[1].axis('off')
    ax[1].set_title("SCÉNARIO 2 : LCR (Liquide)", color='red', weight='bold', fontsize=9)
    ax[1].text(0.3, 0.8, "b=1000", color='black', ha='center', fontsize=8, fontweight='bold')
    ax[1].text(0.7, 0.8, "Map ADC", color='black', ha='center', fontsize=8, fontweight='bold')
    ax[1].add_patch(patches.Circle((0.3, 0.5), 0.15, edgecolor='red', facecolor='white', linewidth=4)) 
    ax[1].text(0.3, 0.25, "DWI", color='white', ha='center', fontweight='bold', fontsize=7)
    ax[1].text(0.5, 0.5, "➔", color='white', fontsize=12, ha='center', va='center')
    ax[1].add_patch(patches.Circle((0.7, 0.5), 0.15, edgecolor='red', facecolor='white', linewidth=4)) 
    ax[1].text(0.7, 0.25, "ADC (Blanc)", color='white', ha='center', fontweight='bold', fontsize=7)
    st.pyplot(fig_adc); plt.close(fig_adc)
    
    st.divider()

    # --- 2. FORMULE & GRAPHIQUE (IVIM / KURTOSIS) ---
    st.subheader("3. Comprendre la Décroissance (Avancé)")
    
    # Formule unique + Légende
    st.markdown("##### La Formule de Base")
    st.latex(r"S = S_0 \cdot e^{-b \cdot ADC}")
    
    with st.expander("📖 Légende de la formule (Cliquez pour ouvrir)"):
        st.markdown("""
        * **S** : Signal mesuré (ce qu'on voit sur l'image).
        * **S₀** : Signal de base sans diffusion (b=0, image T2 pure).
        * **e** : Exponentielle (la décroissance est rapide).
        * **b** : Facteur b (puissance du gradient de diffusion).
        * **ADC** : Coefficient de Diffusion (la mobilité de l'eau).
        """)

    # Graphique Semi-Log (Reproduction Image)
    col_plot, col_expl = st.columns([2, 1])
    
    with col_plot:
        b = np.linspace(0, 3000, 300)
        adc_pure = 0.8e-3
        
        # Courbes théoriques
        ln_S_adc = -b * adc_pure # Droite rouge
        ivim_effect = 0.4 * np.exp(-b * 0.02)
        ln_S_ivim = np.log(np.exp(ln_S_adc) + ivim_effect) # Zone violette
        kurtosis_term = (1.0/6.0) * (b**2) * (adc_pure**2) * 1.5
        ln_S_kurt = ln_S_adc + kurtosis_term # Zone verte

        fig_decay, ax_d = plt.subplots(figsize=(8, 5))
        
        # Zones
        ax_d.fill_between(b, ln_S_adc, ln_S_ivim, where=(b < 800), color='#9b59b6', alpha=0.3, label='Effet IVIM')
        ax_d.fill_between(b, ln_S_adc, ln_S_kurt, where=(b > 1000), color='#2ecc71', alpha=0.4, label='Effet Kurtosis')
        
        # Droite ADC
        ax_d.plot(b, ln_S_adc, color='red', linewidth=3, label='ADC (Modèle Gaussien)')
        
        # Points simulés
        b_pts = np.arange(0, 3100, 200)
        y_pts = -b_pts * adc_pure
        y_pts[b_pts < 500] += np.log(1 + 0.4*np.exp(-b_pts[b_pts < 500]*0.02))
        y_pts[b_pts > 1500] += (1.0/6.0) * (b_pts[b_pts > 1500]**2) * (adc_pure**2) * 1.5
        ax_d.scatter(b_pts, y_pts, color='black', zorder=5, label='Données')

        # Annotations
        ax_d.text(300, -0.2, "IVIM (Sang)", color='purple', fontweight='bold')
        ax_d.text(2200, -2.5, "Kurtosis (Cellules)", color='green', fontweight='bold')
        ax_d.text(1200, -1.2, "Pente = -ADC", color='red', rotation=-30, fontweight='bold')

        ax_d.set_xlabel("Facteur b"); ax_d.set_ylabel("ln(Signal)")
        ax_d.set_xlim(0, 3000); ax_d.set_ylim(-4, 0.2)
        ax_d.legend(); ax_d.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_decay); plt.close(fig_decay)

    with col_expl:
        st.info("### 🟣 Zone IVIM (b < 200)")
        st.markdown("""
        **\"La Fausse Diffusion\"**
        Au début, le signal chute vite. Ce n'est pas de la diffusion, c'est le **sang** qui circule (Pseudo-diffusion).
        * *Utile pour voir la perfusion sans produit de contraste.*
        """)
        
        st.success("### 🟢 Zone Kurtosis (b > 1000)")
        st.markdown("""
        **\"L'Obstacle\"**
        À la fin, la courbe remonte. L'eau tape dans les murs des cellules (Membranes).
        * *Utile pour grader les tumeurs complexes.*
        """)
with t11:
    st.header("🎓 Cours Théorique")
    slides = [f"Slide {i+1}" for i in range(5)]
    if 'slide_index' not in st.session_state: st.session_state.slide_index = 0
    st.session_state.slide_index = st.select_slider("Diapositive", options=range(len(slides)), value=st.session_state.slide_index, format_func=lambda x: slides[x])
    current_slide = slides[st.session_state.slide_index]
    st.markdown(f"### 📄 {current_slide}")
    fig_ppt, ax_ppt = plt.subplots(figsize=(10, 6)); ax_ppt.text(0.5, 0.5, f"CONTENU DU COURS\n\n(Diapositive: {current_slide})", ha='center', va='center', fontsize=20, color='gray'); ax_ppt.set_facecolor('#f0f0f5'); ax_ppt.axis('off'); st.pyplot(fig_ppt)

with t12:
    st.header("🩸 Imagerie de Susceptibilité Magnétique (SWI)")
    
    swi_tab1, swi_tab2, swi_tab3 = st.tabs([
        "1. Physique (Phase & Vecteurs)", 
        "2. Le Dipôle (Simulation)", 
        "3. Imagerie Clinique"
    ])

   # --- SOUS-ONGLET 1 : PHYSIQUE & EXPLICATIONS (Layout Optimisé) ---
    with swi_tab1:
        st.subheader("1. Physique : L'Analogie de la Boussole")
        col_ctrl, col_graph = st.columns([1, 2], gap="medium")
        with col_ctrl:
            st.markdown("#### 🎛️ Contrôles")
            st.caption("_Modifiez les valeurs pour faire tourner l'aiguille._")
            te_simu = st.slider("Temps d'Écho (TE)", 0, 80, 20, step=1, key="swi_te_p1_pedago")
            fa_simu = st.slider("Angle Bascule (°)", 5, 90, 30, key="swi_fa_p1_pedago")
            t2_star = 50.0; df = 8.0 
            mag = np.sin(np.radians(fa_simu)) * np.exp(-te_simu / t2_star)
            phase_visu = np.radians(max(10, min(80, 60 - (te_simu/2))))
            vec_visu = mag * np.exp(1j * phase_visu)
            st.divider()
            c_met1, c_met2 = st.columns(2)
            c_met1.metric("Réel (Ombre Sol)", f"{vec_visu.real:.2f}")
            c_met2.metric("Imag (Ombre Mur)", f"{vec_visu.imag:.2f}")

        with col_graph:
            fig_v, ax_v = plt.subplots(figsize=(5, 5)) # Taille optimisée
            fig_v.patch.set_alpha(0) 
            lim = 1.1; ax_v.set_xlim(-0.1, lim); ax_v.set_ylim(-0.1, lim)
            ax_v.axhline(0, color='white', lw=1); ax_v.axvline(0, color='white', lw=1)
            ax_v.arrow(0, 0, vec_visu.real, vec_visu.imag, head_width=0.03, lw=4, fc='#3498db', ec='#3498db', length_includes_head=True, zorder=5)
            ax_v.text(vec_visu.real/2, vec_visu.imag/2 + 0.1, "SIGNAL", color='#3498db', fontweight='bold', ha='center', fontsize=12)
            ax_v.plot([vec_visu.real, vec_visu.real], [0, vec_visu.imag], color='gray', ls=':', lw=1)
            ax_v.arrow(0, 0, vec_visu.real, 0, head_width=0.02, lw=3, fc='#e74c3c', ec='#e74c3c', length_includes_head=True, zorder=4)
            ax_v.text(vec_visu.real/2, -0.08, "Réel (X)", color='#e74c3c', ha='center', fontsize=10, fontweight='bold')
            ax_v.plot([0, vec_visu.real], [vec_visu.imag, vec_visu.imag], color='gray', ls=':', lw=1)
            ax_v.arrow(0, 0, 0, vec_visu.imag, head_width=0.02, lw=3, fc='#2ecc71', ec='#2ecc71', length_includes_head=True, zorder=4)
            ax_v.text(-0.02, vec_visu.imag/2, "Imag (Y)", color='#2ecc71', ha='right', va='center', fontsize=10, fontweight='bold')
            arc = patches.Arc((0,0), 0.4, 0.4, theta1=0, theta2=np.degrees(phase_visu), color='yellow', lw=2); ax_v.add_patch(arc)
            ax_v.text(0.25, 0.1, "Phase", color='yellow', fontsize=11, fontweight='bold')
            ax_v.set_title("Visualisation Vectorielle", color='white', fontsize=12); ax_v.set_aspect('equal'); ax_v.axis('off')
            st.pyplot(fig_v); plt.close(fig_v)

        st.markdown("---")
        with st.expander("📖 Comprendre l'Analogie (Cliquez pour dérouler)", expanded=True):
            c_txt1, c_txt2 = st.columns(2)
            with c_txt1:
                st.info("""**🧭 L'Analogie de la Boussole** \n * **L'Aiguille (Bleue)** : C'est le Signal IRM total. \n * **Sa Longueur** : La force du signal (Magnitude). \n * **Sa Direction** : La nature du tissu (Phase).""")
            with c_txt2:
                st.warning("""**💡 Pourquoi Réel & Imaginaire ?** \n L'ordinateur ne stocke pas une flèche. Il stocke ses ombres : \n * **Partie Réelle :** L'ombre au sol (Axe X). \n * **Partie Imaginaire :** L'ombre au mur (Axe Y).""")
    
    with swi_tab2:
        st.subheader("2. 🧲 Le Laboratoire du Dipôle")
        col_dip_ctrl, col_dip_visu = st.columns([1, 3])
        with col_dip_ctrl:
            dipole_substance = st.radio("Substance :", ["Hématome (Paramagnétique)", "Calcium (Diamagnétique)"], key="dip_sub_key")
            dipole_system = st.radio("Convention Phase :", ["RHS (GE/Philips/Canon)", "LHS (Siemens)"], key="dip_sys_key")
            st.divider()
            z_pos = st.slider("Coupe Axiale (Z)", -1.5, 1.5, 0.0, 0.1, key="dip_z_key")
        with col_dip_visu:
            fig_dip, axes_dip = plt.subplots(1, 2, figsize=(10, 4)); fig_dip.patch.set_facecolor('#404040')
            is_rhs = "RHS" in dipole_system; is_para = "Hématome" in dipole_substance
            combo = (1 if is_para else -1) * (1 if is_rhs else -1)
            col_eq_cen, col_eq_halo, col_poles = ('white', 'black', 'black') if combo > 0 else ('black', 'white', 'white')
            axes_dip[0].set_facecolor('#404040'); axes_dip[0].axis('off')
            axes_dip[0].add_patch(patches.Ellipse((0.5, 0.7), 0.25, 0.35, color=col_poles, alpha=0.9))
            axes_dip[0].add_patch(patches.Ellipse((0.5, 0.3), 0.25, 0.35, color=col_poles, alpha=0.9))
            axes_dip[0].add_patch(patches.Rectangle((0.35, 0.48), 0.3, 0.04, color=col_eq_cen))
            axes_dip[0].axhline(y=0.5 - (z_pos * 0.2), color='yellow', linewidth=2, linestyle='--')
            axes_dip[1].set_facecolor('#404040'); axes_dip[1].axis('off')
            if abs(z_pos) < 0.2:
                axes_dip[1].add_patch(patches.Circle((0.5, 0.5), 0.35, color=col_eq_halo, alpha=0.5))
                axes_dip[1].add_patch(patches.Circle((0.5, 0.5), 0.15, color=col_eq_cen))
            elif 0.2 <= abs(z_pos) < 1.0:
                axes_dip[1].add_patch(patches.Circle((0.5, 0.5), 0.25 * (1.2 - abs(z_pos)), color=col_poles))
            st.pyplot(fig_dip); plt.close(fig_dip)

    with swi_tab3:
        st.subheader("3. Imagerie SWI Clinique")
        path_minip_fixe = os.path.join(current_dir, "minip_static.png") 
        if HAS_NILEARN and processor.ready:
            dims = processor.get_dims() 
            c1_swi, c2_swi = st.columns([1, 4])
            with c1_swi:
                 st.markdown("##### 🩻 Navigation")
                 swi_view = st.radio("Plan de Coupe :", ["Axiale", "Coronale", "Sagittale"], key="swi_view_mode")
                 if swi_view == "Axiale": swi_slice = st.slider("Position Z", 0, dims[2]-1, 90, key="swi_z"); axis_code = 'z'
                 elif swi_view == "Coronale": swi_slice = st.slider("Position Y", 0, dims[1]-1, 100, key="swi_y"); axis_code = 'y'
                 else: swi_slice = st.slider("Position X", 0, dims[0]-1, 90, key="swi_x"); axis_code = 'x'
                 st.divider()
                 show_microbleeds_swi = st.checkbox("Simuler Micro-saignements", False, key="swi_bleed_check")
                 show_dipole_test = st.checkbox("🧪 Dipôle (Test)", False, key="swi_dip_test_check")
            with c2_swi:
                sys_arg = "RHS" if "RHS" in dipole_system else "LHS"; sub_arg = dipole_substance 
                img_mag = processor.get_slice(axis_code, swi_slice, {}, swi_mode='mag', te=te_simu, with_bleeds=show_microbleeds_swi)
                img_phase = processor.get_slice(axis_code, swi_slice, {}, swi_mode='phase', with_bleeds=show_microbleeds_swi, swi_sys=sys_arg, swi_sub=sub_arg, with_dipole=show_dipole_test)
                c_mag, c_pha, c_min = st.columns(3)
                with c_mag: st.image(utils.apply_window_level(img_mag, 1.0, 0.5), caption=f"1. Magnitude ({swi_view})", use_container_width=True)
                with c_pha: st.image(utils.apply_window_level(img_phase, 1.0, 0.5), caption=f"2. Phase ({swi_view})", use_container_width=True)
                with c_min: 
                    if os.path.exists(path_minip_fixe): st.image(path_minip_fixe, caption="3. MinIP (Référence Axiale)", use_container_width=True)
                    else: st.image(np.zeros((200,200)), caption="Image manquante", clamp=True)

with t13:
    st.header("🧠 Séquence 3D T1 Ultra-Rapide (MP-RAGE)")
    st.markdown("""
    <style>
    .table-style {width: 100%; border-collapse: collapse; font-size: 14px;}
    .table-style th {background-color: #f0f2f6; padding: 8px; text-align: left; border-bottom: 2px solid #ddd;}
    .table-style td {padding: 8px; border-bottom: 1px solid #ddd;}
    .brand-col {font-weight: bold; color: #31333F;} .name-col {font-weight: bold; color: #d63031;}
    </style>
    <table class="table-style">
        <tr><th>Constructeur</th><th>Nom Commercial</th><th>Signification Technique</th></tr>
        <tr><td class="brand-col">SIEMENS</td><td class="name-col">MP-RAGE</td><td>Magnetization Prepared - Rapid Gradient Echo</td></tr>
        <tr><td class="brand-col">GE</td><td class="name-col">3D IR-FSPGR (BRAVO)</td><td>Inversion Recovery Fast SPGR</td></tr>
        <tr><td class="brand-col">PHILIPS</td><td class="name-col">3D T1-TFE</td><td>Turbo Field Echo (avec Pré-impulsion)</td></tr>
        <tr><td class="brand-col">CANON</td><td class="name-col">3D Fast FE</td><td>Fast Field Echo (avec Inversion)</td></tr>
    </table><br>
    """, unsafe_allow_html=True)
    st.divider()
    col_mp_ctrl, col_mp_plot = st.columns([1, 2])
    with col_mp_ctrl:
        constructeur_mp = st.radio("Sélecteur Constructeur :", ["SIEMENS", "GE", "PHILIPS", "CANON"], key="mp_const_select_final")
        st.markdown("**Pourquoi le TR diffère ?** Le TR affiché sur la console ne représente pas la même chose selon le constructeur.")
    with col_mp_plot:
        fig_mp, ax_mp = plt.subplots(figsize=(10, 4))
        ti_mp = 900; train_len = 600; tr_echo_val = 8 
        ax_mp.bar(0, 1.2, width=40, color='#e74c3c', label='Inversion 180°', zorder=3)
        ax_mp.text(0, 1.35, "180°", color='#e74c3c', ha='center', fontweight='bold')
        echo_step = 60 
        for k in range(0, train_len, echo_step): ax_mp.bar(ti_mp + k, 0.7, width=25, color='#3498db', alpha=0.7)
        ax_mp.add_patch(patches.Rectangle((ti_mp - 20, 0), train_len + 10, 0.8, color='#3498db', alpha=0.1))
        if constructeur_mp == "SIEMENS":
            ax_mp.annotate('', xy=(ti_mp + train_len + 100, -1.0), xytext=(0, -1.0), arrowprops=dict(arrowstyle='<->', color='green', lw=2.5))
            ax_mp.text((ti_mp + train_len)/2, -1.35, "TR Siemens : Temps du Cycle Complet (~2300ms)", color='green', weight='bold', ha='center', fontsize=10)
        else:
            start_x = ti_mp + echo_step; end_x = ti_mp + 2 * echo_step
            ax_mp.annotate('', xy=(end_x, 0.35), xytext=(start_x, 0.35), arrowprops=dict(arrowstyle='<->', color='#f39c12', lw=3))
            ax_mp.text((start_x + end_x)/2, 0.15, f"TR {constructeur_mp} = {tr_echo_val}ms", color='#f39c12', weight='bold', ha='center', fontsize=11)
        ax_mp.set_ylim(-1.6, 1.6); ax_mp.set_xlim(-100, ti_mp + train_len + 200); ax_mp.axis('off'); ax_mp.axhline(0, color='black', linewidth=0.5)
        st.pyplot(fig_mp); plt.close(fig_mp)
    st.divider()
    st.subheader("3. Pourquoi cette séquence est-elle unique ?")
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown("#### 1. Préparation (180°)"); st.write("Double la dynamique du signal T1.")
    with c2: st.markdown("#### 2. Le TI (Null Point)"); st.write("Annule le LCR pour un contraste parfait.")
    with c3: st.markdown("#### 3. Acquisition Ultra-Rapide"); st.write("Angles faibles (~8°) pour imager vite.")

# [TAB 14 : PERFUSION ASL]
with t14:
    st.header("🩸 Perfusion ASL (Arterial Spin Labeling)")
    c_principe, c_texte = st.columns([1, 1])
    with c_principe:
        image_asl_path = os.path.join(current_dir, "image_028fa1.jpg")
        if os.path.exists(image_asl_path): st.image(image_asl_path, caption="Principe ASL", use_container_width=True)
    with c_texte:
        st.markdown("""
        ### Comment ça marche ?
        1.  **Marquage (Tag) :** Une impulsion "retourne" le sang au niveau du cou.
        2.  **Délai (PLD) :** On attend que le sang monte.
        3.  **Acquisition :** On prend une image.
        4.  **Soustraction :** Image Contrôle - Image Marquée = Perfusion.
        """)
    with st.expander("⏱️ Focus Physique : Pourquoi TR > 4000ms ?"):
        st.markdown("**Cycle ASL :** Marquage (~2s) + Attente (~2s) + Acquisition (~0.5s) = TR ~4.5s")
    st.divider()
    st.subheader("2. Simulation Clinique & Pathologies")
    if HAS_NILEARN and processor.ready:
        c1_asl, c2_asl = st.columns([1, 4])
        with c1_asl:
            asl_slice = st.slider("Coupe Axiale (Z)", 0, dims[2]-1, 90, key="asl_z")
            st.info(f"⏱️ **PLD Actuel : {pld} ms**")
            if show_stroke: st.error("⚠️ **AVC Ischémique**")
            if show_atrophy: st.warning("🧠 **Atrophie (Alzheimer)**")
        with c2_asl:
            ctrl_img, label_img, perf_map = processor.get_asl_maps('z', asl_slice, pld, 1600, with_stroke=show_stroke, with_atrophy=show_atrophy)
            if ctrl_img is not None:
                col_ctrl, col_label, col_perf = st.columns(3)
                with col_ctrl: st.image(utils.apply_window_level(ctrl_img, 1.0, 0.5), caption="1. Image Contrôle", clamp=True, use_container_width=True)
                with col_label: st.image(utils.apply_window_level(label_img, 1.0, 0.5), caption="2. Image Marquée", clamp=True, use_container_width=True)
                with col_perf:
                    fig_perf, ax_perf = plt.subplots()
                    im = ax_perf.imshow(perf_map, cmap='jet', vmin=0, vmax=np.max(perf_map)*0.8); ax_perf.axis('off'); st.pyplot(fig_perf); st.caption("3. Carte de Perfusion")
    else: st.warning("Module Anatomique requis.")

# [TAB 15 : SUPPRESSION DE GRAISSE - VERSION UNIQUE ET NETTOYÉE]
with t15:
    st.header("🍔 Suppression de Graisse (Fat Sat)")
    
    # Menu des techniques
    fs_tabs = st.tabs(["1. Saturation Fréquentielle", "2. Séquence SPAIR", "3. Séquence Dixon", "4. Excitation Eau", "5. Soustraction", "6. Séquence STIR"])
    
    # --- 1. FAT SAT (SATURATION FRÉQUENTIELLE) ---
    with fs_tabs[0]:
        st.subheader("1. Saturation Fréquentielle (Fat Sat)")
        c_fs1, c_fs2 = st.columns([1, 2])
        with c_fs1:
            st.markdown("##### 🎛️ Paramètres Aimant")
            b0_fs = st.select_slider("Champ Magnétique B0 (Tesla)", options=[0.5, 1.5, 3.0], value=1.5, key="fs_b0_clean")
            ppm_val = 3.5; larmor_h = 42.58; shift_hz = int(larmor_h * b0_fs * ppm_val)
            st.info(f"**Écart Eau-Graisse :** {shift_hz} Hz (à {b0_fs}T)")
            
            st.markdown("**Perturbation (Métal / Shimming)**")
            drift = st.slider("Décalage Fréquentiel (Hz)", -300, 300, 0, 10, key="fs_drift_clean")
            
            if abs(drift) > 80: st.error("🚨 **ECHEC FATSAT**")
            elif drift != 0: st.warning(f"⚠️ Dérive de {drift} Hz")

        with c_fs2:
            fig_fs, ax_fs = plt.subplots(figsize=(8, 4))
            x_min, x_max = -1200, 400; freq_range = np.linspace(x_min, x_max, 1000)
            
            rf_target = -shift_hz
            real_fat = -shift_hz + drift
            real_water = 0 + drift
            
            water_curve = gaussian(freq_range, real_water, 30, 1.0)
            fat_curve = gaussian(freq_range, real_fat, 40, 0.8)
            
            ax_fs.fill_between(freq_range, water_curve, color='#3498db', alpha=0.6, label='Eau')
            ax_fs.fill_between(freq_range, fat_curve, color='#ff7f0e', alpha=0.6, label='Graisse')
            
            rf_bw = 100
            ax_fs.axvspan(rf_target - rf_bw/2, rf_target + rf_bw/2, color='#2ecc71', alpha=0.4, label='RF Machine')
            
            ax_fs.plot([real_fat, real_fat], [0, 0.8], color='#e67e22', linestyle='--', linewidth=1.5)
            ax_fs.plot([rf_target, rf_target], [0, 0.8], color='green', linestyle=':', linewidth=1.5)
            
            # Annotation 3.5 ppm
            arrow_y = 0.65
            ax_fs.annotate("", xy=(real_fat, arrow_y), xytext=(real_water, arrow_y), 
                           arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
            ax_fs.text((real_fat + real_water)/2, arrow_y + 0.05, "δ = 3.5 ppm", 
                       ha='center', va='bottom', fontweight='bold', fontsize=10, color='black')

            if drift != 0:
                ax_fs.annotate("", xy=(real_fat, 0.4), xytext=(rf_target, 0.4), arrowprops=dict(arrowstyle="<->", color="red", lw=2))
                ax_fs.text((real_fat + rf_target)/2, 0.45, f"Décalage {abs(drift)} Hz", color='red', ha='center', fontweight='bold', fontsize=9)
            
            dist = abs(real_fat - rf_target)
            title = "✅ SATURATION RÉUSSIE" if dist < 40 else "❌ SATURATION RATÉE (Inhomogénéité)"
            ax_fs.set_title(title, color='green' if dist < 40 else 'red', fontweight='bold')
            ax_fs.set_xlim(x_min, x_max); ax_fs.set_yticks([]); ax_fs.set_xlabel("Fréquence (Hz)"); ax_fs.legend(loc='upper left'); ax_fs.grid(True, alpha=0.3)
            st.pyplot(fig_fs); plt.close(fig_fs)

    # --- 2. SPAIR (ANIMATION COMPLETE & TI ORIGINAL) ---
    with fs_tabs[1]:
        import time
        st.subheader("2. SPAIR (Spectral Adiabatic Inversion Recovery)")
        col_bilan1, col_bilan2 = st.columns([1, 1])
        with col_bilan1: st.success("✅ **Points Clés :** Adiabatique (Sweep), Homogène, SAR faible.")
        with col_bilan2: st.info("🛡️ **Compatible Gado :** L'eau n'est PAS inversée.")
        st.divider()
        
        # --- A. DYNAMIQUE TEMPORELLE (TI) ---
        st.markdown("#### 📉 A. Dynamique Temporelle (TI)")
        c_sp1, c_sp2 = st.columns([1, 2])
        with c_sp1:
            ti_spair = st.slider("Temps d'Inversion (TI)", 50, 400, 180, 5, key="spair_ti_clean")
            mz_fat = 1 - 2 * np.exp(-ti_spair/260.0)
            if abs(mz_fat) < 0.1: st.success(f"✅ Graisse Annulée\n({mz_fat:.2f})")
            else: st.warning(f"Graisse Visible\n({mz_fat:.2f})")
        with c_sp2:
            fig_sp, ax_sp = plt.subplots(figsize=(8, 4))
            t = np.linspace(0, 800, 500)
            ax_sp.plot(t, 1 - 2 * np.exp(-t/260.0), color='#e67e22', linewidth=3, label='Graisse')
            ax_sp.plot(t, 1 - 0.1 * np.exp(-t/1000.0), color='#3498db', linewidth=2, linestyle='--', label='Eau')
            ax_sp.axhline(0, color='black', linewidth=1); ax_sp.axvline(ti_spair, color='green', linewidth=2, label=f'TI ({ti_spair}ms)')
            ax_sp.annotate('Départ inversé (-1)', xy=(10, -0.9), xytext=(150, -0.8), arrowprops=dict(facecolor='#e67e22', arrowstyle='->'), color='#e67e22')
            ax_sp.set_ylim(-1.1, 1.1); ax_sp.legend(loc='lower right'); ax_sp.grid(True, alpha=0.3)
            st.pyplot(fig_sp); plt.close(fig_sp)
        
        st.divider()
        
        # --- B. PRINCIPE ADIABATIQUE (ANIMATION) ---
        st.markdown("#### 🎯 B. Principe Adiabatique (Le Sweep)")
        c_anim_ctrl, c_anim_plot = st.columns([1, 2])
        
        # Paramètres fixes
        f_water = 0; f_fat = -220; bw_pulse = 60 
        
        def draw_spair_spectrum(pulse_center, label_pulse):
            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.linspace(-800, 200, 1000)
            water = gaussian(x, f_water, 30, 1.0)
            fat = gaussian(x, f_fat, 40, 0.8)
            ax.fill_between(x, water, color='#3498db', alpha=0.6, label='Eau')
            ax.fill_between(x, fat, color='#ff7f0e', alpha=0.6, label='Graisse')
            
            # Flèche 3.5 ppm
            arrow_y = 0.65
            ax.annotate("", xy=(f_fat, arrow_y), xytext=(f_water, arrow_y), arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
            ax.text((f_fat + f_water)/2, arrow_y + 0.05, "δ = 3.5 ppm", ha='center', va='bottom', fontweight='bold', fontsize=10, color='black')
            
            # Bande verte
            ax.axvspan(pulse_center - bw_pulse/2, pulse_center + bw_pulse/2, color='#2ecc71', alpha=0.5, label='Impulsion Adiabatique')
            ax.text(pulse_center, 1.1, label_pulse, ha='center', color='#27ae60', fontweight='bold')
            ax.annotate("", xy=(pulse_center, 1.0), xytext=(pulse_center, 1.08), arrowprops=dict(arrowstyle="->", color="#27ae60", lw=2))

            ax.set_xlim(-600, 200); ax.set_ylim(0, 1.3); ax.set_yticks([]); ax.set_xlabel("Fréquence (Hz)")
            ax.legend(loc='upper left'); ax.set_title("Spectre SPAIR : Balayage en Fréquence", fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('white')
            return fig

        with c_anim_ctrl:
            st.info("**Le Balayage :** L'impulsion traverse **tout** le pic de graisse (de gauche à droite).")
            start_anim = st.button("▶️ Lancer le Balayage", type="primary")
            if not start_anim:
                manual_pos = st.select_slider("Position", options=["Base Gauche", "Sommet", "Base Droite"], value="Sommet")

        with c_anim_plot:
            graph_placeholder = st.empty()
            if start_anim:
                # Traversée complète : Base Gauche -> Base Droite
                start_freq = f_fat - 100; end_freq = f_fat + 100
                steps = np.linspace(start_freq, end_freq, 30)
                for freq in steps:
                    fig = draw_spair_spectrum(freq, "Balayage...")
                    graph_placeholder.pyplot(fig); plt.close(fig)
                    time.sleep(0.04)
                fig = draw_spair_spectrum(end_freq, "Fin du Pulse")
                graph_placeholder.pyplot(fig); plt.close(fig)
            else:
                if manual_pos == "Base Gauche": center = f_fat - 100
                elif manual_pos == "Base Droite": center = f_fat + 100
                else: center = f_fat
                fig = draw_spair_spectrum(center, "Impulsion Adiabatique")
                graph_placeholder.pyplot(fig); plt.close(fig)

    # --- 3. DIXON ---
    with fs_tabs[2]:
        st.subheader("3. Séquence Dixon (Chemical Shift Imaging)")
        st.markdown("#### 📡 A. L'Acquisition (2 Échos)")
        c_dx1, c_dx2 = st.columns([1, 2])
        with c_dx1:
            te_dixon = st.select_slider("Choisir le Temps d'Echo (TE)", options=[2.2, 4.5], key="dx_te_clean")
            if te_dixon == 2.2:
                st.error("📉 **OUT OF PHASE (Opposition)**"); st.write("Eau et Graisse s'opposent."); st.latex(r"S = E - G")
            else:
                st.success("📈 **IN PHASE (Somme)**"); st.write("Eau et Graisse s'additionnent."); st.latex(r"S = E + G")
        with c_dx2:
            fig_dx, ax_dx = plt.subplots(figsize=(8, 3)); t_ms = np.linspace(0, 10, 500)
            ax_dx.plot(t_ms, np.ones_like(t_ms), color='#3498db', label='Eau')
            ax_dx.plot(t_ms, np.cos(2 * np.pi * 220 * t_ms / 1000.0), color='#e67e22', label='Graisse')
            ax_dx.plot(te_dixon, np.cos(2 * np.pi * 220 * te_dixon / 1000.0), 'ro', markersize=12, label='Acquisition')
            ax_dx.axvline(te_dixon, color='gray', linestyle='--')
            ax_dx.set_xlabel("TE (ms)"); ax_dx.set_yticks([-1, 0, 1]); ax_dx.set_yticklabels(["Opp", "Quad", "Phase"]); ax_dx.legend(); ax_dx.grid(True, alpha=0.3)
            st.pyplot(fig_dx); plt.close(fig_dx)
        st.divider()
        st.markdown("#### 🧮 B. Le Calcul")
        c_calc1, c_calc2 = st.columns(2)
        with c_calc1: st.markdown("##### 💧 Image EAU"); st.latex(r"W = \frac{IP + OOP}{2}")
        with c_calc2: st.markdown("##### 🥓 Image GRAISSE"); st.latex(r"F = \frac{IP - OOP}{2}")

    # --- 4. EXCITATION EAU ---
    with fs_tabs[3]:
        import pandas as pd 
        st.subheader("4. Excitation de l'Eau (Water Excitation / WE)")
        st.markdown("#### 🌊 Principe : Sélection sans Saturation")
        c_we_txt, c_we_acro = st.columns([2, 1])
        with c_we_txt:
            st.info("""
            **Différence avec la Fat-Sat :**
            * **Fat-Sat :** Excite la graisse puis la tue (Gradient de déphasage).
            * **WE (Water Excitation) :** N'utilise **pas de gradient de déphasage**. Elle stimule sélectivement l'eau en laissant la graisse tranquille.
            """)
            st.markdown("""
            **La Séquence Binomiale (1-1) :**
            1. **Pulse 45° :** Tout le monde bascule.
            2. **Délai :** On attend l'opposition de phase (180°).
            3. **Pulse 45° :** L'Eau s'additionne (90°), la Graisse se soustrait (0°).
            """)
        with c_we_acro:
            st.markdown("#### 🏷️ Noms Commerciaux")
            df_names = pd.DataFrame({"Constructeur": ["Siemens / Fuji", "GE", "Philips", "Canon"], "Acronyme": ["WE", "SSRF", "ProSET", "WET / PASTA"]})
            st.table(df_names.set_index("Constructeur"))
        st.divider()
        st.markdown("#### 🕹️ Visualisation Dynamique (Impulsion 1-1)")
        step = st.select_slider("Étapes", options=["1. Équilibre (M0)", "2. Premier Pulse (45°)", "3. Délai (Opposition 180°)", "4. Second Pulse (45°)"], value="1. Équilibre (M0)")
        w_vec = np.array([0.0, 0.0, 1.0]); f_vec = np.array([0.0, 0.0, 1.0]); desc = "Aimantation longitudinale (z)."
        if "2." in step:
            val = np.sin(np.pi/4); w_vec = np.array([0.0, val, val]); f_vec = np.array([0.0, val, val]); desc = "Pulse 45°. Tout bascule."
        elif "3." in step:
            val = np.sin(np.pi/4); w_vec = np.array([0.0, val, val]); f_vec = np.array([0.0, -val, val]); desc = "Délai : Opposition de phase."
        elif "4." in step:
            w_vec = np.array([0.0, 1.0, 0.0]); f_vec = np.array([0.0, 0.0, 1.0]); desc = "Pulse 45°. Eau à 90°, Graisse à 0°."
        
        c_visu1, c_visu2 = st.columns([1, 2])
        with c_visu1: st.info(f"**État :** {desc}")
        with c_visu2:
            fig = plt.figure(figsize=(6, 5)); ax = fig.add_subplot(111, projection='3d')
            ax.plot([0, 0], [0, 0], [-0.2, 1.2], 'k--', linewidth=1)
            ax.quiver(0, 0, 0, w_vec[0], w_vec[1], w_vec[2], color='#3498db', linewidth=4, arrow_length_ratio=0.1, label='Eau')
            offset = 0.05 if "1." in step or "2." in step else 0.0
            ax.quiver(offset, 0, 0, f_vec[0], f_vec[1], f_vec[2], color='#e67e22', linewidth=3, arrow_length_ratio=0.1, label='Graisse')
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, 1.2); ax.view_init(elev=20, azim=20); ax.legend()
            st.pyplot(fig); plt.close(fig)

    # --- 5. SOUSTRACTION ---
    with fs_tabs[4]:
        st.subheader("5. Soustraction (Post - Pré)")
        c_sub1, c_sub2 = st.columns([1, 2])
        with c_sub1:
            move_x = st.slider("Mouvement Patient (px)", -10, 10, 0, 1, key="sub_move_clean")
            st.info("Le moindre mouvement crée des artefacts.")
        with c_sub2:
            size = 100; y, x = np.ogrid[:size, :size]; center = size // 2
            mask_body = np.sqrt((x - center)**2 + (y - center)**2) < 30
            img_pre = np.zeros((size, size)); img_pre[mask_body] = 0.5
            mask_body_mv = np.sqrt((x - (center+move_x))**2 + (y - center)**2) < 30
            mask_lesion = np.sqrt((x - (center+move_x) - 10)**2 + (y - center - 10)**2) < 5
            img_post = np.zeros((size, size)); img_post[mask_body_mv] = 0.5; img_post[mask_lesion] = 1.0
            c1, c2, c3 = st.columns(3)
            c1.image(img_pre, caption="Pré", clamp=True); c2.image(img_post, caption="Post", clamp=True); c3.image(np.clip(img_post - img_pre, 0, 1), caption="Sub", clamp=True)

    # --- 6. STIR ---
    with fs_tabs[5]:
        st.subheader("6. STIR (Short Tau Inversion Recovery)")
        st.markdown("#### 📡 1. Pourquoi \"Non-Sélectif\" ? (Bande Large)")
        col_ex1, col_ex2 = st.columns([2, 1])
        with col_ex1: st.info("Le STIR utilise une impulsion courte qui tape **tout le spectre** (Eau, Graisse, Gado...).")
        with col_ex2:
            fig_bw, ax_bw = plt.subplots(figsize=(4, 2.5))
            ax_bw.fill_between(np.linspace(-500, 500, 100), 0, 1, color='purple', alpha=0.4)
            ax_bw.text(0, 0.5, "Bande Large", ha='center', color='purple'); ax_bw.set_yticks([]); ax_bw.set_xlim(-500, 500)
            st.pyplot(fig_bw); plt.close(fig_bw)
        st.divider()
        st.markdown("#### 📉 3. Visualisation (Signal en Module)")
        c_st1, c_st2 = st.columns([1, 2])
        with c_st1:
            ti_stir = st.slider("Choisir le moment du 'CLIC' (TI)", 50, 800, 170, 10, key="st_ti_clean")
            mz_fat = 1 - 2 * np.exp(-ti_stir/260.0); mz_gado = 1 - 2 * np.exp(-ti_stir/280.0)
            st.metric("Signal Graisse", f"{abs(mz_fat):.2f}")
            if abs(mz_fat) < 0.1: st.success("✅ **GRAISSE NOIRE**")
            else: st.warning("Graisse visible")
            if abs(mz_gado) < 0.2: st.error("🚨 **GADO ANNULÉ**")
        with c_st2:
            fig_st, (ax_st, ax_bar) = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={'width_ratios': [30, 1]})
            t_rng = np.linspace(0, 5000, 500)
            tissues = {'Graisse (260ms)': (260, '#ff7f0e'), 'Gado (280ms)': (280, 'red'), 'SB (790ms)': (790, '#bdc3c7'), 'LCR (4000ms)': (4000, 'cyan')}
            for name, (t1_val, col) in tissues.items():
                ax_st.plot(t_rng, 1 - 2 * np.exp(-t_rng / t1_val), label=name, color=col)
            ax_st.axhline(0, color='black'); ax_st.axvline(ti_stir, color='green', linewidth=2, label=f'TI ({ti_stir}ms)')
            ax_st.set_xlim(0, 5000); ax_st.set_ylim(-1.1, 1.1); ax_st.legend(loc='lower right', fontsize=8); ax_st.grid(True, alpha=0.3)
            y_grad = np.linspace(1.1, -1.1, 200).reshape(-1, 1)
            ax_bar.imshow(np.abs(y_grad), aspect='auto', cmap='gray', vmin=0, vmax=1, extent=[0, 1, -1.1, 1.1])
            ax_bar.set_xticks([]); ax_bar.set_yticks([]); ax_bar.plot(0.5, 1 - 2 * np.exp(-ti_stir/260.0), 'o', color='orange', markeredgecolor='white')
            st.pyplot(fig_st); plt.close(fig_st)
# [TAB 16 : SÉCURITÉ RF - VERSION INTÉGRALE (TOUS TABLEAUX)]
with t16:
    st.header("🔥 Sécurité RF : Console de Contrôle")
    
    # --- 0. AVERTISSEMENT ---
    st.warning("⚠️ **Module Autonome :** Utilisez les réglages ci-dessous. Les valeurs sont simulées pour illustrer l'impact des paramètres sur la chauffe.")

    # --- 1. CONFIGURATION ---
    GAMMA = 267.513 
    SAR_CALIB_K = 1.5 
    
    # Bibliothèque des Formes
    PULSE_LIBRARY = {
        "Sinc (3 lobes)":       {"amp": 0.67, "power": 0.55, "desc": "Standard"},
        "Rectangulaire (Hard)": {"amp": 1.0, "power": 1.0, "desc": "Rapide"},
        "Gaussienne":           {"amp": 0.41, "power": 0.29, "desc": "Sélectif"}
    }
    
    # Bibliothèque des Intensités (Mode RF)
    RF_MODES = {
        "Faible (Low SAR)": 0.8,
        "Moyenne (Standard)": 1.0,
        "Forte (High BW)": 1.3
    }

    # --- 2. ENTRÉES UTILISATEUR ---
    c_pat, c_seq, c_scan = st.columns(3)
    
    with c_pat:
        st.markdown("#### 👤 Patient")
        weight = st.number_input("Poids (kg)", 40.0, 150.0, 75.0, 5.0, key="sar_w_full")
        height = st.number_input("Taille (m)", 1.0, 2.2, 1.75, 0.05, key="sar_h_full")
        vol = weight / 1010.0 
        radius_m = np.sqrt(vol / (np.pi * height))

    with c_seq:
        st.markdown("#### 📡 Séquence & RF")
        seq_type = st.selectbox("Type Séquence", ["Spin Echo (SE)", "Turbo Spin Echo (TSE)", "Gradient Echo (GRE)"], key="sar_type_full")
        
        # 1. FORME
        pulse_shape = st.selectbox("Forme Onde", list(PULSE_LIBRARY.keys()), index=0, key="sar_shape_full")
        
        # 2. INTENSITÉ (Mode RF)
        rf_mode_name = st.select_slider("Mode RF / Intensité", options=list(RF_MODES.keys()), value="Moyenne (Standard)", key="sar_mode_full")
        rf_intensity = RF_MODES[rf_mode_name]

        # 3. Angle & ETL
        if "TSE" in seq_type: def_etl, def_ang = 10, 150 
        elif "GRE" in seq_type: def_etl, def_ang = 0, 20   
        else: def_etl, def_ang = 0, 90   

        angle = st.slider("Flip Angle (°)", 5, 180, def_ang, key="sar_angle_full")
        
        if "TSE" in seq_type:
            etl = st.slider("Train d'Échos (ETL)", 2, 64, def_etl, key="sar_etl_full")
        else:
            etl = 0
            st.slider("Train d'Échos", 0, 1, 0, disabled=True, key="sar_etl_dis_full")

    with c_scan:
        st.markdown("#### ⚙️ Paramètres Scan")
        tr = st.number_input("TR (ms)", 20, 10000, 600, 50, key="sar_tr_full")
        nb_slices = st.slider("Nombre de Coupes", 1, 50, 20, key="sar_slices_full")
        nex = st.slider("NEX (Moyennages)", 1, 8, 1, key="sar_nex_full")
        matrix = st.select_slider("Matrice Phase", options=[128, 192, 256, 512], value=256, key="sar_mat_full")
        
        scan_time_sec = (tr * matrix * nex) / 1000
        st.caption(f"⏱️ Scan : {scan_time_sec/60:.1f} min")
        duration = 3.0 

    st.divider()

    # --- 3. MOTEUR DE CALCUL ---
    shape_data = PULSE_LIBRARY[pulse_shape]
    
    # Pulses par TR par Coupe
    if "GRE" in seq_type: nb_pulses_per_slice = 1
    elif "TSE" in seq_type: nb_pulses_per_slice = 1 + etl 
    else: nb_pulses_per_slice = 2 
    
    # B1 Peak (Ajusté par Intensité RF)
    angle_rad = np.radians(angle if angle > 90 else 90) 
    duration_s = duration / 1000.0
    
    b1_base = (angle_rad / (GAMMA * duration_s * shape_data["amp"]))
    b1_peak_ut = b1_base * rf_intensity
    
    # Duty Cycle (Total Coupes)
    total_rf_time_ms = nb_pulses_per_slice * duration * nb_slices
    if total_rf_time_ms > tr:
        st.error(f"⚠️ **TR TROP COURT** ({tr}ms) pour {nb_slices} coupes !")
        duty_cycle = 1.0
    else:
        duty_cycle = total_rf_time_ms / tr
    
    # B1+rms & SAR
    correction_factor = 0.5 if "TSE" in seq_type else 1.0
    b1_rms_ut = b1_peak_ut * np.sqrt(duty_cycle * shape_data["power"]) * correction_factor
    sar_val = SAR_CALIB_K * (radius_m**2) * (b1_rms_ut**2)

    # --- 4. VISUALISATION ---
    st.subheader("📊 Moniteurs de Sécurité")
    
    def draw_gauge_cursor(value, label, limit_norm, limit_first, max_scale=6.0):
        fig, ax = plt.subplots(figsize=(6, 1.8))
        
        # Zones
        ax.add_patch(plt.Rectangle((0, 0), limit_norm, 1, color='#2ecc71', alpha=0.9))
        ax.text(limit_norm/2, 0.5, "NORMAL", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        
        ax.add_patch(plt.Rectangle((limit_norm, 0), limit_first-limit_norm, 1, color='#f1c40f', alpha=0.9))
        ax.text((limit_norm+limit_first)/2, 0.5, "1st LEVEL", ha='center', va='center', color='white', fontweight='bold', fontsize=8)
        
        ax.add_patch(plt.Rectangle((limit_first, 0), max_scale-limit_first, 1, color='#e74c3c', alpha=0.9))
        ax.text((limit_first+max_scale)/2, 0.5, "DANGER", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        
        # Curseur
        cursor_pos = min(value, max_scale - 0.1)
        ax.plot([cursor_pos, cursor_pos], [-0.2, 1.2], color='black', linewidth=4, solid_capstyle='round')
        ax.text(cursor_pos, 1.35, f"{value:.2f}", ha='center', fontweight='bold', fontsize=12, color='black')
        
        ax.set_xlim(0, max_scale); ax.set_ylim(0, 1.6); ax.set_yticks([]); ax.set_xticks([0, limit_norm, limit_first, max_scale])
        ax.set_title(label, loc='left', fontweight='bold')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False); ax.spines['bottom'].set_visible(True)
        return fig

    c_g1, c_g2 = st.columns(2)
    with c_g1:
        st.pyplot(draw_gauge_cursor(sar_val, "SAR (W/kg)", 2.0, 4.0))
        if sar_val > 4.0: st.error("🚨 SAR CRITIQUE ! Augmentez le TR ou baissez l'intensité RF.")
    with c_g2:
        st.pyplot(draw_gauge_cursor(b1_rms_ut, "B1+rms (µT)", 2.8, 4.0))
        if b1_rms_ut > 2.8: st.error("⚡ RISQUE IMPLANT !")

    st.divider()
    
    # --- 5. EXPLICATIONS & GLOSSAIRES (3 VOLETS RESTAURÉS) ---
    
    # 1. GLOSSAIRE PARAMÈTRES (Restauré)
    with st.expander("📝 Détails des Paramètres & Seuils", expanded=False):
        st.markdown("""
        * **ETL (Facteur Turbo)** : Multiplie le nombre d'impulsions RF. Paramètre critique pour le SAR en TSE.
        * **Matrice** : Augmente le temps mais change peu le SAR instantané.
        * **Nombre de Coupes** : Multiplie directement l'énergie envoyée par TR.
        * **Seuils SAR (IEC)** : 
            * 🟩 **< 2 W/kg** (Mode Normal).
            * 🟨 **2-4 W/kg** (Premier Niveau - Contrôle Médical).
            * 🟥 **> 4 W/kg** (Interdit).
        """)

    # 2. INTENSITÉ RF (Nouveau)
    with st.expander("⚡ Comprendre l'Intensité RF", expanded=False):
        st.markdown("""
        * **Mode Faible (Low SAR)** : La machine optimise l'impulsion (plus longue) pour minimiser le pic d'énergie.
        * **Mode Moyenne (Standard)** : Le compromis habituel.
        * **Mode Forte (High BW)** : La machine utilise plus de puissance (bande passante élevée) pour réduire les durées.
        """)

    # 3. TABLEAU CLINIQUE (Restauré)
    with st.expander("🏥 Clinique : Formes d'Impulsions & Séquences", expanded=True):
        st.markdown("""
        | Forme | Usage Principal | Avantage | Risque / Inconvénient |
        | :--- | :--- | :--- | :--- |
        | **Sinc** | **TSE, SE** (Coupes 2D) | Profil de coupe rectangulaire (Pas de croisement). | **SAR Élevé** (Impulsions longues & nombreuses). |
        | **Rectangulaire** | **MP-RAGE** (Volume 3D) | Ultra-rapide (TR court). | Coupe "sale" (bords flous) - corrigé par encodage 3D. |
        | **Gaussienne** | **Fat Sat** | Très sélectif en fréquence. | **Pic B1 Élevé** (Stress sur l'ampli RF). |
        """)