# main.py - VERSION 7.98 (LÉGENDES COMPLÈTES & BIBLIOGRAPHIE)
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
st.set_page_config(layout="wide", page_title="Magnetovault V7.98 - Final")
utils.inject_css()

# --- STATE MANAGEMENT ---
if 'init' not in st.session_state:
    st.session_state.seq = 'Pondération T1'
    st.session_state.reset_count = 0
    st.session_state.atrophy_active = False 
    st.session_state.tr_force = 500.0
    st.session_state.widget_tr = 500.0
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

# LOGIQUE DE CHANGEMENT DE SÉQUENCE
if seq_choix != st.session_state.seq:
    st.session_state.seq = seq_choix
    st.session_state.tr_force = float(defaults['tr'])
    if 'widget_tr' in st.session_state: 
        st.session_state.widget_tr = float(defaults['tr'])
    te_key_current = f"te_main_{st.session_state.reset_count}"
    st.session_state[te_key_current] = float(defaults['te'])
    utils.safe_rerun()

is_gre = "Gradient" in seq_choix
is_dwi = "Diffusion" in seq_choix
is_ir = "FLAIR" in seq_choix or "STIR" in seq_choix
is_swi = "SWI" in seq_choix
is_mprage = "MP-RAGE" in seq_choix
is_asl = "ASL" in seq_choix

# Initialisation variables
current_reset_id = st.session_state.reset_count
ti = 0.0
te = float(defaults['te'])
flip_angle = 90

# ==============================================================================
# 1. GÉOMÉTRIE
# ==============================================================================
st.sidebar.header("1. Géométrie")
col_ep, col_slice = st.sidebar.columns(2)

# Epaisseur
ep = col_ep.number_input("Epaisseur (mm)", min_value=1.0, max_value=10.0, value=5.0, step=0.5, key=f"ep_{current_reset_id}")
# Nb Coupes (Slider limité à 100 pour plus de finesse)
n_slices = col_slice.slider("Nb Coupes", 1, 100, 20, step=1, key=f"ns_{current_reset_id}")
# Concaténations
if not is_dwi and not is_mprage:
    n_concats = st.sidebar.select_slider("📚 Concaténations", options=[1, 2, 3, 4], value=1, key=f"concat_{current_reset_id}")
else: 
    n_concats = 1

fov = st.sidebar.slider("FOV (mm)", 100.0, 500.0, 240.0, step=10.0, key=f"fov_{current_reset_id}")
mat = st.sidebar.select_slider("Matrice", options=[64, 128, 256, 512], value=256, key=f"mat_{current_reset_id}")

st.sidebar.subheader("Réglage Echo")
if not (is_dwi or is_asl):
    te = st.sidebar.slider("TE (ms)", 1.0, 300.0, float(defaults['te']), step=1.0, key=f"te_main_{current_reset_id}")
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
turbo = 1
if not (is_gre or is_dwi or is_swi or is_mprage or is_asl):
    turbo = st.sidebar.slider("Facteur Turbo", 1, 32, 1, key=f"turbo_{current_reset_id}")
bw = st.sidebar.slider("Bande Passante", 50, 500, 220, 10, key=f"bw_{current_reset_id}")
es = st.sidebar.slider("Espace Inter-Echo (ES)", 2.5, 20.0, 10.0, step=2.5, key=f"es_{current_reset_id}")

st.sidebar.header("4. Imagerie Parallèle (iPAT)")
ipat_on = st.sidebar.checkbox("Activer Accélération", value=False, key=f"ipat_on_{current_reset_id}")
ipat_factor = st.sidebar.slider("Facteur R", 2, 4, 2, key=f"ipat_r_{current_reset_id}") if ipat_on else 1

st.sidebar.markdown("---")

# ==============================================================================
# MENTIONS LÉGALES & BIBLIOGRAPHIE (MIS À JOUR)
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
# CALCULS PHYSIQUES (CORRECTION BUG CONCATS)
# ==============================================================================
tr_effective = tr 

try:
    # Appel du module physique mis à jour
    raw_ms = phy.calculate_acquisition_time(tr, mat, nex, turbo, ipat_factor, n_concats, n_slices, is_mprage)
except AttributeError:
    # FALLBACK DE SÉCURITÉ
    base_time = (tr * mat * nex) / (turbo * ipat_factor)
    if is_mprage: raw_ms = base_time * n_slices
    else: raw_ms = base_time * n_concats

final_seconds = raw_ms / 1000.0; mins = int(final_seconds // 60); secs = int(final_seconds % 60); str_duree = f"{mins} min {secs} s"

# 2. Calcul des SIGNAUX
v_lcr = phy.calculate_signal(tr_effective, te, ti, cst.T_LCR['T1'], cst.T_LCR['T2'], cst.T_LCR['T2s'], cst.T_LCR['ADC'], cst.T_LCR['PD'], flip_angle, is_gre, is_dwi, b_value if is_dwi else 0)
v_wm  = phy.calculate_signal(tr_effective, te, ti, cst.T_WM['T1'], cst.T_WM['T2'], cst.T_WM['T2s'], cst.T_WM['ADC'], cst.T_WM['PD'], flip_angle, is_gre, is_dwi, b_value if is_dwi else 0)
v_gm  = phy.calculate_signal(tr_effective, te, ti, cst.T_GM['T1'], cst.T_GM['T2'], cst.T_GM['T2s'], cst.T_GM['ADC'], cst.T_GM['PD'], flip_angle, is_gre, is_dwi, b_value if is_dwi else 0)
v_stroke = phy.calculate_signal(tr_effective, te, ti, cst.T_STROKE['T1'], cst.T_STROKE['T2'], cst.T_STROKE['T2s'], cst.T_STROKE['ADC'], cst.T_STROKE['PD'], flip_angle, is_gre, is_dwi, b_value if is_dwi else 0)

if is_dwi and b_value >= 1000 and show_stroke: v_stroke = 2.0 
v_fat = phy.calculate_signal(tr_effective, te, ti, cst.T_FAT['T1'], cst.T_FAT['T2'], cst.T_FAT['T2s'], cst.T_FAT['ADC'], cst.T_FAT['PD'], flip_angle, is_gre, is_dwi, 0) if not is_dwi else 0.0

# 3. Calcul du SNR
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
st.title("Simulateur MagnétoVault V7.98")

t_home, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14 = st.tabs([
    "🏠 Accueil", "Fantôme", "Espace K 🌀", "Signaux", "Codage", "🧠 Anatomie", 
    "📈 Physique", "⚡ Chronogramme", "☣️ Artefacts", "🚀 iPAT", "🧬 Théorie Diffusion", "🎓 Cours", "🩸 SWI & Dipôle", "3D T1 (MP-RAGE)", "ASL (Perfusion)"
])

# [TAB 0 : ACCUEIL]
with t_home:
    st.header("Bienvenue dans le Simulateur MagnétoVault")
    st.markdown("""
    **MagnétoVault** est un outil pédagogique interactif conçu pour comprendre la physique et la technique de l'IRM.
    
    ---
    ### 📘 Comment utiliser le simulateur ?
    #### 1️⃣ La Console de Commande (Barre de Gauche)
    * **Séquence :** Choisissez le type d'image.
    * **Géométrie :** Réglez FOV, Matrice, Coupes.
    * **Chrono :** Ajustez TR, TE.
    * **Options :** iPAT, BW, **Facteur Turbo**.
    
    #### 2️⃣ Les Onglets de Visualisation
    * **Fantôme :** Votre résultat principal.
    * **Espace K :** Données brutes.
    * **🧠 Anatomie :** (Module *nilearn*) Cerveau humain simulé.
    """)
    st.info("💡 Sélectionnez l'onglet **'Fantôme'** ci-dessus pour commencer.")

# [TAB 1 : FANTOME (LÉGENDES ET FORMULES COMPLÈTES)]
with t1:
    c1, c2 = st.columns([1, 1])
    with c1:
        k1, k2 = st.columns(2); k1.metric("⏱️ Durée", str_duree); k2.metric("📉 SNR Relatif", str_snr); st.divider()
        st.subheader("1. Formules & Glossaire")
        
        # --- FORMULES MATHÉMATIQUES COMPLETES ---
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
        
        # --- LÉGENDE EXHAUSTIVE DE TOUS LES TERMES ---
        with st.expander("📖 Glossaire Complet des Termes Affichés", expanded=False):
            st.markdown("""
            | Terme | Signification | Contexte |
            | :--- | :--- | :--- |
            | **TR** | Temps de Répétition | Temps entre deux excitations (Contrôle le T1 et la durée). |
            | **TE** | Temps d'Écho | Temps jusqu'à la lecture du signal (Contrôle le T2). |
            | **TF / Turbo** | Facteur Turbo | Nombre d'échos acquis par TR (Accélère la séquence). |
            | **$N_{Ph}$** | Lignes de Phase | Nombre de lignes à acquérir (Matrice). |
            | **$N_{Slices}$** | Nombre de Coupes | En 3D, agit comme une 2ème dimension de phase. |
            | **NEX** | Nombre d'Excitations | Moyennage pour augmenter le SNR (mais allonge le temps). |
            | **Concats** | Concaténations | Découpage des coupes en plusieurs paquets (pour TR long). |
            | **R (iPAT)** | Facteur d'Accélération | Imagerie Parallèle (réduit le temps, réduit le SNR). |
            | **$V_{vox}$** | Volume du Voxel | Taille du pixel × épaisseur (Le SNR dépend du volume !). |
            | **BW** | Bande Passante | Vitesse de lecture (BW élevé = Acquisition rapide mais moins de SNR). |
            | **b** | Facteur b (DWI) | Force du gradient de diffusion ($s/mm^2$). |
            | **ADC** | Coeff. Diffusion | Capacité de l'eau à bouger (Restriction = Signal élevé en DWI). |
            | **$S_0$** | Signal de base | Signal T2 sans pondération de diffusion. |
            """)
        
        if show_stroke: st.error("⚠️ **PATHOLOGIE : AVC Ischémique (Zone Rouge)**")
        if show_atrophy: st.warning("🧠 **PATHOLOGIE : Atrophie (Alzheimer)**")

    with c2:
        # VISUEL FANTÔME
        fig_anot, ax_anot = plt.subplots(figsize=(5,5))
        ax_anot.imshow(final, cmap='gray', vmin=0, vmax=1.3)
        ax_anot.axis('off')
        ax_anot.text(S/2, S/2, "CSF/H2O", color='cyan', ha='center', va='center', fontsize=10, fontweight='bold')
        ax_anot.text(S/2, S/2 + (S*0.35/2), "WM", color='black', ha='center', va='center', fontsize=9, fontweight='bold')
        ax_anot.text(S/2, S/2 + (S*0.65/2), "GM", color='white', ha='center', va='center', fontsize=9, fontweight='bold')
        ax_anot.text(S/2, S*0.93, "FAT", color='orange', ha='center', va='center', fontsize=10, fontweight='bold')
        st.pyplot(fig_anot)
        plt.close(fig_anot)

# [TAB 2 : ESPACE K (VISUALISATION AVEC TE RÉEL)]
with t2:
    st.markdown("### 🌀 Espace K et Remplissage")
    col_k1, col_k2 = st.columns([1, 1])
    with col_k1:
        fill_mode = st.radio("Ordre de Remplissage", ["Linéaire (Haut -> Bas)", "Centrique (Centre -> Bords)"], key=f"k_mode_{current_reset_id}")
        acq_pct = st.slider("Progression (%)", 0, 100, 10, step=1, key=f"k_pct_{current_reset_id}")
        
        # --- VISUALISATION DU RANGEMENT TSE AVEC TE RÉEL ---
        if turbo > 1:
            st.divider()
            st.markdown(f"#### 🚅 Train d'Échos (Facteur {turbo})")
            st.caption(f"Espace Inter-Echo (ES) : {es} ms")
            
            fig_tse, ax_tse = plt.subplots(figsize=(5, 3))
            n_lines = 32 
            colors = plt.cm.jet(np.linspace(0, 1, turbo))
            
            for i in range(n_lines):
                dist_from_center = abs(i - n_lines/2)
                norm_dist = dist_from_center / (n_lines/2)
                echo_idx = int(norm_dist * (turbo-1))
                if echo_idx >= turbo: echo_idx = turbo - 1
                ax_tse.hlines(i, 0, 1, colors=colors[echo_idx], linewidth=2)
            
            # --- MODIFICATION ICI : AFFICHER LE TE RÉEL ---
            for e in range(turbo):
                # Calcul du TE de l'écho e : (Numéro d'écho) * ES
                te_val_echo = (e + 1) * es
                ax_tse.text(1.05, e * (n_lines/turbo), f"TE={int(te_val_echo)}ms", color=colors[e], fontweight='bold', fontsize=9)
                
            ax_tse.set_yticks([]); ax_tse.set_xticks([])
            ax_tse.set_title(f"Contribution des échos au Contraste (Ky)", fontsize=9)
            ax_tse.spines['top'].set_visible(False); ax_tse.spines['right'].set_visible(False)
            ax_tse.spines['bottom'].set_visible(False); ax_tse.spines['left'].set_visible(False)
            ax_tse.set_ylabel("K-Space (Ky)")
            ax_tse.axhline(n_lines/2, color='black', linestyle='--', linewidth=1)
            ax_tse.text(0.5, n_lines/2 + 1, "Centre (Contraste)", ha='center', fontsize=8)
            
            st.pyplot(fig_tse)
            plt.close(fig_tse)

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
            # --- LE NOUVEAU BOUTON ---
            show_interactive_legends = st.checkbox("🔍 Activer Légendes (Atlas Harvard-Oxford)", value=False, help="Identifie les structures (Gyrus, Noyaux, Tronc, Cervelet) au survol.")

            if is_dwi: 
                if show_adc_map: st.info("🗺️ **Mode Carte ADC** (LCR Blanc)")
                else: st.success(f"🧬 **Mode Diffusion** (b={b_value})")
            if show_stroke and ax == 'z': st.error("⚠️ **AVC Visible**")

        with c2:
            # 1. Calcul de l'image IRM
            w_vals = {'csf':v_lcr, 'gm':v_gm, 'wm':v_wm, 'fat':v_fat}
            if show_stroke: w_vals['wm'] = w_vals['wm'] * 0.9 + v_stroke * 0.1
            seq_type_arg = 'dwi' if is_dwi else ('gre' if is_gre else None)
            
            img_raw = processor.get_slice(ax, idx, w_vals, seq_type=seq_type_arg, te=te, tr=tr, fa=flip_angle, b_val=b_value, adc_mode=show_adc_map, with_stroke=show_stroke)
            
            if img_raw is not None:
                img_display = utils.apply_window_level(img_raw, window, level)

                if show_interactive_legends:
                    # MODE INTERACTIF (PLOTLY)
                    with st.spinner("Chargement de l'Atlas Anatomique..."):
                        labels_map = processor.get_anatomical_labels(ax, idx)
                        fig = px.imshow(
                            img_display, 
                            color_continuous_scale='gray', 
                            zmin=0, zmax=1,
                            binary_string=False
                        )
                        fig.update_traces(
                            customdata=labels_map,
                            hovertemplate="<b>%{customdata}</b><br>Intensité: %{z:.2f}<extra></extra>"
                        )
                        fig.update_layout(
                            margin=dict(l=0, r=0, t=0, b=0),
                            coloraxis_showscale=False,
                            width=600, height=600,
                            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
                        )
                        st.plotly_chart(fig, config={'displayModeBar': False})
                        st.caption("ℹ️ Source: Atlas Harvard-Oxford (Cortical & Sous-cortical).")
                else:
                    # MODE RAPIDE (STANDARD)
                    st.image(img_display, clamp=True, width=600)
    else: 
        st.warning("Module 'nilearn' manquant ou données non chargées.")

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
        t_max = max(tr + 40, te + 50); t = np.linspace(0, t_max, 2000)
        rf_sigma = 0.5; grad_width = 3.0
        fig, axs = plt.subplots(5, 1, sharex=True, figsize=(10, 8), gridspec_kw={'hspace': 0.3})
        rf = np.zeros_like(t); amp_rf = flip_angle / 90.0
        rf += amp_rf * np.exp(-0.5 * ((t - t_90)**2) / (rf_sigma**2))
        t_90_next = t_90 + tr; rf += amp_rf * np.exp(-0.5 * ((t - t_90_next)**2) / (rf_sigma**2))
        axs[0].plot(t, rf, color='black'); axs[0].fill_between(t, 0, rf, color='green', alpha=0.4)
        axs[0].set_ylabel("RF"); axs[0].set_yticks([0, 1], ["", f"{flip_angle}°"])
        gsc = np.zeros_like(t); mask_sel = (t > t_90 - grad_width) & (t < t_90 + grad_width); gsc[mask_sel] = 1.0
        mask_reph = (t > t_90 + grad_width + 1) & (t < t_90 + 2*grad_width + 1); gsc[mask_reph] = -0.8
        axs[1].plot(t, gsc, color='green'); axs[1].fill_between(t, 0, gsc, color='green', alpha=0.6); axs[1].set_ylabel("Gss")
        gcp = np.zeros_like(t); t_code = t_90 + 15; mask_c = (t > t_code - grad_width) & (t < t_code + grad_width); gcp[mask_c] = 0.5
        axs[2].plot(t, gcp, color='orange'); axs[2].fill_between(t, 0, gcp, color='orange', alpha=0.6); axs[2].set_ylabel("Gpe")
        gcf = np.zeros_like(t); t_read = t_90 + te; mask_read = (t > t_read - grad_width) & (t < t_read + grad_width); gcf[mask_read] = 1.0
        t_pre = t_read - (2 * grad_width) - 2; 
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
        t = np.linspace(0, t_max, 2000); rf_sigma = 0.5; grad_width = max(1.5, es_disp * 0.2)
        t_180s = [];
        for i in range(turbo): t_p = t_90 + (i * es) + (es/2); t_180s.append(t_p)
        fig, axs = plt.subplots(5, 1, sharex=True, figsize=(10, 8), gridspec_kw={'hspace': 0.3})
        rf = np.zeros_like(t)
        def add_rf_pulse(center, amp, w): return amp * np.exp(-0.5 * ((t - center)**2) / (w**2))
        rf += add_rf_pulse(t_90, 1.0, rf_sigma) 
        for t_p in t_180s:
            if t_p < t_max: rf += add_rf_pulse(t_p, 1.6, rf_sigma)
        axs[0].plot(t, rf, color='black', linewidth=1.5); axs[0].fill_between(t, 0, rf, color='green', alpha=0.4)
        axs[0].set_ylabel("RF"); axs[0].set_yticks([0, 1, 1.6], ["", "90", "180"])
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

with t9:
    st.header("🚀 Imagerie Parallèle (iPAT)")
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

with t10:
    st.header("🧬 Théorie de la Diffusion (DWI)")
    st.markdown("""L'imagerie de diffusion est unique car elle sonde le **mouvement microscopique** des molécules d'eau.""")
    st.divider()
    st.subheader("1. Isotropie vs Anisotropie")
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
    st.subheader("4. Coefficient de Diffusion Apparent (ADC)")
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
    ax[1].set_title("SCÉNARIO 2 : LCR (Liquide)", color='red', weight='bold', fontsize=9)
    ax[1].text(0.3, 0.8, "b=1000", color='black', ha='center', fontsize=8, fontweight='bold')
    ax[1].text(0.7, 0.8, "Map ADC", color='black', ha='center', fontsize=8, fontweight='bold')
    ax[1].add_patch(patches.Circle((0.3, 0.5), 0.15, edgecolor='red', facecolor='white', linewidth=4)) 
    ax[1].text(0.3, 0.25, "DWI", color='white', ha='center', fontweight='bold', fontsize=7)
    ax[1].text(0.5, 0.5, "➔", color='white', fontsize=12, ha='center', va='center')
    ax[1].add_patch(patches.Circle((0.7, 0.5), 0.15, edgecolor='red', facecolor='white', linewidth=4)) 
    ax[1].text(0.7, 0.25, "ADC (Blanc)", color='white', ha='center', fontweight='bold', fontsize=7)
    st.pyplot(fig_adc)

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