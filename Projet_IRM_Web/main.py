import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import streamlit.components.v1 as components
import os
import base64
from PIL import Image
from scipy.ndimage import shift, gaussian_filter
import plotly.express as px
import plotly.graph_objects as go
from google import genai
import re
from streamlit_paste_button import paste_image_button

# IMPORTS DES MODULES LOCAUX
import constantes as cst
import utils
import physique as phy
from anatomie import AdvancedMRIProcessor, HAS_NILEARN
# Initialisation de l'IA avec la clé cachée
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# CONFIG & CSS (Toujours en premier !)
st.set_page_config(layout="wide", page_title="Magnetovault V8.38")
utils.inject_css()
plt.style.use('seaborn-v0_8-whitegrid')

# FONCTION CACHE
@st.cache_data
def get_k_space(image_matrix):
    f = np.fft.fftshift(np.fft.fft2(image_matrix))
    return 20 * np.log(np.abs(f) + 1)
@st.cache_data
def process_vps_fourier_reconstruction(complex_matrix, mask_matrix):
    """Calcul mathématique de reconstruction par TFI (Isolé pour mise en cache serveur)"""
    f_local = np.fft.fftshift(np.fft.fft2(complex_matrix))
    kspace_masked = f_local * mask_matrix
    img_rec = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_masked)))
    
    max_val = np.max(img_rec)
    if max_val > 0:
        img_rec = img_rec / max_val
    return img_rec
# --- 🌐 GESTION LANGUE (MOTEUR) ---
if 'lang' not in st.session_state:
    st.session_state.lang = 'fr'

def T(fr, en, de=None):
    """
    Renvoie le texte français, anglais ou allemand selon l'état.
    Si l'allemand n'est pas fourni, on utilise l'anglais par défaut (Fallback).
    """
    if st.session_state.lang == 'fr':
        return fr
    elif st.session_state.lang == 'de':
        return de if de is not None else en
    else:
        return en

# --- FONCTIONS UTILITAIRES ---
def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def show_centered_image(file_path, width=23):
    """Affiche une image centrée via HTML pour l'alignement drapeau/bouton"""
    if os.path.exists(file_path):
        img_b64 = get_img_as_base64(file_path)
        st.markdown(
            f"<div style='text-align: center; margin-bottom: 2px;'>"
            f"<img src='data:image/png;base64,{img_b64}' width='{width}'>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown("<div style='text-align: center;'>🏴</div>", unsafe_allow_html=True)
def render_tof_auto_segmentation(img_base64, view_type, show_aca, show_acm, show_acp, show_bas, show_vert_car, sens_ax, sens_cor):
    """
    Détection auto avec zones polygonales précises et affichage pleine hauteur.
    """
    import cv2
    import numpy as np
    from PIL import Image
    import io

    # 1. Chargement
    img_data = base64.b64decode(img_base64)
    pil_image = Image.open(io.BytesIO(img_data)).convert("RGBA")
    img = np.array(pil_image)
    h, w = img.shape[:2]
    
    # 2. SEUILLAGE (Threshold)
    # On choisit le seuil selon la vue active
    sensitivity = sens_ax if view_type == "AXIAL" else sens_cor
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    _, mask_vessels = cv2.threshold(gray, sensitivity, 255, cv2.THRESH_BINARY)
    
    # Calque de couleur
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    
    # --- FONCTION UTILITAIRE : CRÉATION DE MASQUE POLYGONAL ---
    def make_poly_mask(points_rel):
        """Crée un masque binaire à partir de points relatifs (0.0-1.0)"""
        mask = np.zeros((h, w), dtype=np.uint8)
        # Conversion relatif -> absolu
        pts = np.array([[int(p[0]*w), int(p[1]*h)] for p in points_rel], np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask > 0 # Retourne un booléen

    # --- DÉFINITION DES ZONES ANATOMIQUES (POLYGONES) ---
    
    if view_type == "AXIAL":
        # Centre approximatif du Polygone de Willis (ajustable)
        cx, cy = 0.5, 0.52 
        
        # ACA (Triangle Haut)
        poly_aca = [[cx, cy], [0.2, 0.0], [0.8, 0.0]]
        mask_aca = make_poly_mask(poly_aca)
        
        # ACP / Basilaire (Triangle Bas)
        poly_acp = [[cx, cy], [0.2, 1.0], [0.8, 1.0]]
        mask_acp = make_poly_mask(poly_acp)
        
        # ACM (Les ailes restantes sur les cotés)
        # On définit tout ce qui n'est NI ACA, NI ACP
        mask_acm = ~(mask_aca | mask_acp)
        
        # Siphons Carotidiens (Deux cercles/zones proches du centre mais latéraux)
        # On affine la zone ACP pour ne pas manger les carotides si elles sont hautes
        # (Ici simplifié : on considère les siphons dans la zone ACM ou ACP selon le niveau)

    elif view_type == "CORONAL":
        # ACA (Tout en haut, central)
        mask_aca = make_poly_mask([[0.35, 0.0], [0.65, 0.0], [0.65, 0.35], [0.35, 0.35]])
        
        # ACM (Haut Latéral)
        mask_acm = make_poly_mask([[0.0, 0.0], [0.35, 0.0], [0.35, 0.45], [0.0, 0.45]]) | \
                   make_poly_mask([[0.65, 0.0], [1.0, 0.0], [1.0, 0.45], [0.65, 0.45]])
        
        # Tronc Basilaire (Tube Central)
        mask_bas = make_poly_mask([[0.42, 0.45], [0.58, 0.45], [0.58, 0.65], [0.42, 0.65]])
        
        # ACP (Juste au dessus du Basilaire, sous les ACA)
        mask_acp = make_poly_mask([[0.30, 0.35], [0.70, 0.35], [0.70, 0.45], [0.30, 0.45]])
        
        # Vertébrales (V inversé en bas au centre)
        mask_vert = make_poly_mask([[0.42, 0.65], [0.58, 0.65], [0.58, 1.0], [0.42, 1.0]])
        
        # Carotides (Tubes latéraux en bas)
        mask_car = make_poly_mask([[0.20, 0.45], [0.40, 0.45], [0.40, 1.0], [0.20, 1.0]]) | \
                   make_poly_mask([[0.60, 0.45], [0.80, 0.45], [0.80, 1.0], [0.60, 1.0]])

    # 4. COLORISATION INTELLIGENTE (Intersection Pixel Blanc + Zone Polygone)
    def paint(mask_zone, color):
        # Pixel doit être: 1. Dans la zone, 2. Brillant (Vaisseau)
        target = mask_zone & (mask_vessels == 255)
        overlay[target] = color

    if view_type == "AXIAL":
        if show_aca: paint(mask_aca, [0, 255, 255, 140])   # Cyan
        if show_acp: paint(mask_acp, [255, 0, 0, 140])     # Rouge
        if show_bas: paint(mask_acp, [255, 140, 0, 140])   # Orange (Inclus dans zone ACP bas)
        if show_acm: paint(mask_acm, [160, 32, 240, 140])  # Violet
        
    elif view_type == "CORONAL":
        if show_vert_car:
            paint(mask_car, [0, 255, 127, 140])  # Carotides (Vert)
            paint(mask_vert, [255, 215, 0, 140]) # Vertébrales (Jaune)
        if show_bas: paint(mask_bas, [255, 140, 0, 140])
        if show_acp: paint(mask_acp, [255, 0, 0, 140])
        if show_acm: paint(mask_acm, [160, 32, 240, 140])
        if show_aca: paint(mask_aca, [0, 255, 255, 140])

    # 5. RENDU FINAL (Gestion taille CSS améliorée)
    overlay_img = Image.fromarray(overlay, 'RGBA')
    bg_dark = Image.fromarray((img[:,:,:3] * 0.9).astype(np.uint8)).convert("RGBA")
    combined = Image.alpha_composite(bg_dark, overlay_img)
    
    buf = io.BytesIO()
    combined.save(buf, format="PNG")
    new_b64 = base64.b64encode(buf.getvalue()).decode()
    
    # CSS CORRIGÉ : width 100% et height AUTO pour ne jamais tronquer
    html = f"""
    <div style="width: 100%; display: flex; justify-content: center; background-color: black; border-radius: 8px; overflow: hidden;">
        <img src="data:image/png;base64,{new_b64}" style="width: 100%; height: auto; object-fit: contain;">
    </div>
    """
    return html

def gaussian(x: np.ndarray, mu: float, sigma: float, amp: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

def make_phantom_subtraction(offset_x: float) -> tuple[np.ndarray, np.ndarray]:
    size = 100
    y, x = np.ogrid[:size, :size]
    center = size // 2
    fat_mask = np.sqrt((x - (center + offset_x))**2 + (y - center)**2) < 35
    lesion_mask = np.sqrt((x - (center + offset_x))**2 + (y - center)**2) < 8
    img = np.zeros((size, size))
    img[fat_mask] = 1.0
    return img, lesion_mask

def generate_sensitivity_map(shape, center_x, center_y, sigma):
    y, x = np.ogrid[:shape[0], :shape[1]]
    mask = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
    return mask
def render_tof_dual_view(img_base64, view_type, show_aca, show_acm, show_acp, show_bas, show_vert, show_car):
    """
    Génère l'overlay vectoriel adapté aux images tof_ax.png et tof_coro.png
    """
    # Couleurs "Néon" semi-transparentes
    c_aca  = "rgba(0, 255, 255, 0.5)" if show_aca else "rgba(0,0,0,0)"   # Cyan
    c_acm  = "rgba(160, 32, 240, 0.5)" if show_acm else "rgba(0,0,0,0)"  # Violet
    c_acp  = "rgba(255, 0, 0, 0.5)"    if show_acp else "rgba(0,0,0,0)"  # Rouge
    c_bas  = "rgba(255, 140, 0, 0.5)"  if show_bas else "rgba(0,0,0,0)"  # Orange
    c_vert = "rgba(255, 215, 0, 0.5)"  if show_vert else "rgba(0,0,0,0)" # Jaune
    c_car  = "rgba(0, 255, 127, 0.5)"  if show_car else "rgba(0,0,0,0)"  # Vert

    # CSS pour l'effet de surbrillance au survol
    style = """
    <style>
        .vessel { fill: none; stroke-linecap: round; stroke-width: 9; transition: all 0.3s ease; mix-blend-mode: screen; filter: blur(2px); }
        .vessel:hover { stroke-width: 15; filter: blur(0px); cursor: pointer; stroke-opacity: 1 !important; }
    </style>
    """
    
    svg_paths = ""
    
    # --- VUE AXIALE (Recalée sur votre image) ---
    if view_type == "AXIAL":
        svg_paths = f"""
        <path d="M 250 340 L 250 320" stroke="{c_bas}" class="vessel" stroke-width="12" />
        
        <path d="M 250 320 Q 220 330 190 350 Q 160 370 140 400" stroke="{c_acp}" class="vessel" />
        <path d="M 250 320 Q 280 330 310 350 Q 340 370 360 400" stroke="{c_acp}" class="vessel" />

        <circle cx="185" cy="280" r="14" fill="{c_car}" style="filter:blur(6px);" />
        <circle cx="315" cy="280" r="14" fill="{c_car}" style="filter:blur(6px);" />

        <path d="M 185 280 Q 120 280 40 270" stroke="{c_acm}" class="vessel" /> <path d="M 315 280 Q 380 280 460 270" stroke="{c_acm}" class="vessel" /> <path d="M 185 280 Q 240 260 245 220 L 245 120" stroke="{c_aca}" class="vessel" /> <path d="M 315 280 Q 260 260 255 220 L 255 120" stroke="{c_aca}" class="vessel" /> <line x1="245" y1="220" x2="255" y2="220" stroke="{c_aca}" stroke-width="6" />
        """
        
    # --- VUE CORONALE (Recalée sur votre image cou/tête) ---
    elif view_type == "CORONAL":
        svg_paths = f"""
        <path d="M 200 580 Q 210 500 245 420" stroke="{c_vert}" class="vessel" /> 
        <path d="M 300 580 Q 290 500 255 420" stroke="{c_vert}" class="vessel" /> 

        <path d="M 250 420 L 250 250" stroke="{c_bas}" class="vessel" />

        <path d="M 250 250 Q 220 240 190 250" stroke="{c_acp}" class="vessel" />
        <path d="M 250 250 Q 280 240 310 250" stroke="{c_acp}" class="vessel" />

        <path d="M 160 580 Q 155 400 170 300" stroke="{c_car}" class="vessel" /> 
        <path d="M 340 580 Q 345 400 330 300" stroke="{c_car}" class="vessel" /> 

        <path d="M 170 300 Q 130 250 100 200" stroke="{c_acm}" class="vessel" />
        <path d="M 330 300 Q 370 250 400 200" stroke="{c_acm}" class="vessel" />
        
        <path d="M 170 300 Q 200 280 240 180" stroke="{c_aca}" class="vessel" />
        <path d="M 330 300 Q 300 280 260 180" stroke="{c_aca}" class="vessel" />
        """

    html = f"""
    {style}
    <div style="position: relative; width: 500px; height: 600px; margin: auto; border: 2px solid #333; border-radius: 8px; overflow: hidden; background-color: black;">
        <img src="data:image/png;base64,{img_base64}" 
             style="width: 100%; height: 100%; object-fit: cover; filter: contrast(1.2);">
        
        <svg width="100%" height="100%" viewBox="0 0 500 600" style="position: absolute; top: 0; left: 0;">
            {svg_paths}
        </svg>
    </div>
    """
    return html

# --- 📝 TRADUCTION INTELLIGENTE DES SÉQUENCES ---
def translate_seq(name):
    """Traduit le nom de la séquence en détectant des mots-clés."""
    if st.session_state.lang == 'fr':
        return name
        
    n = name.lower()
    
    if st.session_state.lang == 'en':
        if "t1" in n: return "T1 Weighting"
        if "t2" in n: return "T2 Weighting"
        if "densit" in n or "proton" in n: return "Proton Density"
        if "flair" in n: return "FLAIR (Fluid Suppressed)"
        if "stir" in n: return "STIR (Fat Suppressed)"
        if "diffusion" in n or "dwi" in n: return "Diffusion (DWI)"
        if "gradient" in n: return "Gradient Echo"
        if "swi" in n: return "SWI"
        if "asl" in n: return "ASL"
        if "fat" in n and "sat" in n: return "Fat Sat"
        if "mp" in n and "rage" in n: return "MP-RAGE"
        
    elif st.session_state.lang == 'de':
        if "t1" in n: return "T1-Wichtung"
        if "t2" in n: return "T2-Wichtung"
        if "densit" in n or "proton" in n: return "Protonendichte (PD)"
        if "flair" in n: return "FLAIR"
        if "stir" in n: return "STIR (Fettunterdrückung)"
        if "diffusion" in n or "dwi" in n: return "Diffusion (DWI)"
        if "gradient" in n: return "Gradientenecho (GRE)"
        if "swi" in n: return "SWI"
        if "asl" in n: return "ASL"
        if "fat" in n and "sat" in n: return "Fett-Sat"
        if "mp" in n and "rage" in n: return "MP-RAGE"
        
    return name
# --- NOUVEAU : MOTEUR DE CALCUL CENTRALISÉ ---
@st.cache_data
def get_computed_physics(tr, te, ti, flip_angle, is_gre, is_dwi, b_value, seq_choix):
    # On récupère les paramètres par défaut pour le SNR
    defaults = cst.STD_PARAMS.get(seq_choix, cst.STD_PARAMS["Pondération T1"])
    
    # Calcul des signaux (Extraits de ton code original)
    v_lcr = phy.calculate_signal(tr, te, ti, cst.T_LCR['T1'], cst.T_LCR['T2'], cst.T_LCR['T2s'], cst.T_LCR['ADC'], cst.T_LCR['PD'], flip_angle, is_gre, is_dwi, b_value if is_dwi else 0)
    v_wm  = phy.calculate_signal(tr, te, ti, cst.T_WM['T1'], cst.T_WM['T2'], cst.T_WM['T2s'], cst.T_WM['ADC'], cst.T_WM['PD'], flip_angle, is_gre, is_dwi, b_value if is_dwi else 0)
    v_gm  = phy.calculate_signal(tr, te, ti, cst.T_GM['T1'], cst.T_GM['T2'], cst.T_GM['T2s'], cst.T_GM['ADC'], cst.T_GM['PD'], flip_angle, is_gre, is_dwi, b_value if is_dwi else 0)
    v_stroke = phy.calculate_signal(tr, te, ti, cst.T_STROKE['T1'], cst.T_STROKE['T2'], cst.T_STROKE['T2s'], cst.T_STROKE['ADC'], cst.T_STROKE['PD'], flip_angle, is_gre, is_dwi, b_value if is_dwi else 0)
    
    # SNR Relatif
    ref_wm_signal = phy.calculate_signal(float(defaults['tr']), float(defaults['te']), 0, cst.T_WM['T1'], cst.T_WM['T2'], cst.T_WM['T2s'], cst.T_WM['ADC'], cst.T_WM['PD'], 90, False, False, 0)
    # Protection simple contre division par zéro
    ref_wm_signal = ref_wm_signal if ref_wm_signal != 0 else 0.001
    
    return {
        "v_lcr": v_lcr, "v_wm": v_wm, "v_gm": v_gm, "v_stroke": v_stroke,
        "ref_wm_signal": ref_wm_signal
    }
# --- STATE MANAGEMENT ---
if 'init' not in st.session_state:
    st.session_state.seq = 'Pondération T1'
    st.session_state.reset_count = 0
    st.session_state.atrophy_active = False 
    st.session_state.tr_force = 500.0
    st.session_state.widget_tr = 500.0
    st.session_state.mem_turbo = 1 
    st.session_state.init = True

# INITIALISATION SÉCURISÉE ET PARTAGÉE DU PROCESSEUR (OPTIMISATION VPS ENTRÉE TP)
@st.cache_resource
def load_vps_shared_processor():
    return AdvancedMRIProcessor()

processor = load_vps_shared_processor()

# ==============================================================================
# 🎛️ BARRE LATÉRALE (SIDEBAR)
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "logo_mia.png")

# Images Drapeaux
flag_fr_path = os.path.join(current_dir, "image_15c952.png")
flag_uk_path = os.path.join(current_dir, "image_15c972.png")
flag_de_path = os.path.join(current_dir, "flag_de.png") # Nouveau drapeau

# Fallback noms standards
if not os.path.exists(flag_fr_path): flag_fr_path = os.path.join(current_dir, "flag_fr.png")
if not os.path.exists(flag_uk_path): flag_uk_path = os.path.join(current_dir, "flag_uk.png")
if not os.path.exists(flag_de_path): flag_de_path = os.path.join(current_dir, "flag_de.png") 

if os.path.exists(logo_path): 
    st.sidebar.image(logo_path, width=280)

st.sidebar.title(T("Réglages Console", "Console Settings", "Konsoleneinstellungen"))

# --- ZONE ACTIONS COMPACTE (Reset + Langues) ---
# On modifie les proportions pour inclure la 5ème colonne (c_de)
c_reset, c_space, c_fr, c_uk, c_de = st.sidebar.columns([1.3, 0.1, 0.7, 0.7, 0.7])

with c_reset:
    st.write("") 
    st.write("")
    if st.button(T("⚠️ Reset", "⚠️ Reset", "⚠️ Reset"), use_container_width=True):
        components.html("<script>window.parent.location.reload();</script>", height=0)

with c_fr:
    show_centered_image(flag_fr_path, width=23)
    type_fr = "primary" if st.session_state.lang == 'fr' else "secondary"
    if st.button("FR", key="lang_fr", type=type_fr, use_container_width=True):
        st.session_state.lang = 'fr'
        st.rerun()

with c_uk:
    show_centered_image(flag_uk_path, width=23)
    type_en = "primary" if st.session_state.lang == 'en' else "secondary"
    if st.button("EN", key="lang_en", type=type_en, use_container_width=True):
        st.session_state.lang = 'en'
        st.rerun()

with c_de:
    show_centered_image(flag_de_path, width=23)
    type_de = "primary" if st.session_state.lang == 'de' else "secondary"
    if st.button("DE", key="lang_de", type=type_de, use_container_width=True):
        st.session_state.lang = 'de'
        st.rerun()

st.sidebar.markdown("---")
# --- SÉLECTEUR D'OBJET (AUTOMATIQUE & INVISIBLE) ---
# On garde les définitions de clés pour la logique interne
opt_brain  = "Cerveau"
opt_dipole = "Dipôle"
opt_bottle = "Bouteille"

# AU LIEU D'AFFICHER UN MENU, ON DÉCIDE AUTOMATIQUEMENT :
# Si l'utilisateur est sur la séquence SWI -> On charge le Dipôle
# Sinon -> On charge le Cerveau
if "SWI" in st.session_state.seq:
    phantom_choice_raw = opt_dipole
else:
    phantom_choice_raw = opt_brain

# Note : phantom_choice_raw existe maintenant, mais rien ne s'affiche dans la barre latérale !
# --- FIN DE LA SUPPRESSION ---
# --- SÉLECTION SÉQUENCE (AVEC FILTRE & TRADUCTION INTELLIGENTE) ---
allowed_seqs = [s for s in cst.OPTIONS_SEQ if "MP-RAGE" not in s]
seq_key = f"seq_select_{st.session_state.reset_count}"

try: 
    if st.session_state.seq not in allowed_seqs:
        st.session_state.seq = 'Pondération T1'
    idx_def = allowed_seqs.index(st.session_state.seq)
except: 
    idx_def = 0

seq_choix = st.sidebar.selectbox(
    T("Séquence", "Sequence"), 
    allowed_seqs, 
    index=idx_def, 
    format_func=translate_seq, # Utilise la nouvelle fonction robuste
    key=seq_key
)

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
is_mprage = False 
is_asl = "ASL" in seq_choix

# Paramètres
ti = 0.0
te = float(defaults['te'])
flip_angle = 90

# --- 1. GÉOMÉTRIE (CORRECTION MATRICE FORCÉE) ---
st.sidebar.header(T("1. Géométrie", "1. Geometry", "1. Geometrie"))
col_ep, col_slice = st.sidebar.columns(2)

ep = col_ep.number_input(T("Epaisseur (mm)", "Slice Thick. (mm)", "Schichtdicke (mm)"), min_value=1.0, max_value=10.0, value=4.0, step=0.5, key=f"ep_{current_reset_id}")
n_slices = col_slice.slider(T("Nb Coupes", "Slices", "Schichten"), 1, 100, 20, step=1, key=f"ns_{current_reset_id}")

if not is_dwi:
    n_concats = st.sidebar.select_slider(T("📚 Concaténations", "📚 Concatenations", "📚 Konkatenationen"), options=[1, 2, 3, 4], value=1, key=f"concat_{current_reset_id}")
else: 
    n_concats = 1

fov = st.sidebar.slider("FOV (mm)", 100.0, 500.0, 240.0, step=10.0, key=f"fov_{current_reset_id}")

# --- CORRECTION MATRICE ---
# On définit l'index par défaut : 1 (pour 128) si DWI, sinon 2 (pour 256)
idx_mat_def = 1 if is_dwi else 2
options_mat = [64, 128, 256, 512]

mat = st.sidebar.select_slider(
    T("Matrice", "Matrix", "Matrix"), 
    options=options_mat, 
    value=options_mat[idx_mat_def], 
    key=f"mat_{current_reset_id}_{is_dwi}" 
)

st.sidebar.subheader(T("Réglage Echo", "Echo Settings"))
# ... (Le reste des sliders TE/TR reste identique jusqu'au calcul du temps) ...
# ...
# ... (Descendez jusqu'à la section CALCUL DURÉE ACQUISITION) ...

# --- CORRECTION CALCUL TEMPS (SINGLE SHOT EPI) ---
# Bloc situé juste avant l'affichage "str_duree"
try:
    if is_dwi:
        # EPI Single Shot : Le temps ne dépend PAS de la matrice (tout est acquis en 1 TR)
        # Formule : TR * NEX * Concat
        raw_ms = tr * nex * n_concats
    elif is_mprage:
        # 3D : Dépend du nombre de coupes encodées
        raw_ms = (tr * mat * nex * n_slices) / (turbo * ipat_factor)
    else:
        # 2D Standard : Dépend des lignes de phase (mat)
        base_time = (tr * mat * nex) / (turbo * ipat_factor)
        raw_ms = base_time * n_concats
except:
    raw_ms = 0

final_seconds = raw_ms / 1000.0
mins = int(final_seconds // 60)
secs = int(final_seconds % 60)
str_duree = f"{mins} min {secs} s"

# --- 2. CHRONO (TR) ---
# --- CALCUL DU TR AUTOMATIQUE (BLOC MANQUANT A REINSERER) ---
time_per_slice = te + 15.0 
min_tr_required = (n_slices * time_per_slice) / n_concats
current_tr_val = st.session_state.get('widget_tr', st.session_state.tr_force)

auto_adjusted = False 

if current_tr_val < min_tr_required and not is_asl and not is_dwi: 
    st.session_state.tr_force = min_tr_required
    st.session_state.widget_tr = min_tr_required
    auto_adjusted = True
    utils.safe_rerun()
st.sidebar.header(T("2. Chrono (ms)", "2. Timing (ms)", "2. Timing (ms)"))
b_value = 0; show_stroke = False; show_atrophy = False; show_adc_map = False; show_microbleeds = False; pld = 1500 

def update_tr_from_slider():
    st.session_state.tr_force = st.session_state.widget_tr

if is_dwi:
    b_value = st.sidebar.select_slider(T("Facteur b", "b-Value", "b-Wert"), options=[0, 500, 1000], value=0, key=f"bval_{current_reset_id}")
    tr = 6000.0; te = 90.0; ti = 0.0; flip_angle = 90
    st.sidebar.info(T("TR fixé : 6000ms | TE fixé : 90ms", "Fixed TR: 6000ms | Fixed TE: 90ms", "Fester TR: 6000ms | Fester TE: 90ms"))
    show_stroke = st.sidebar.checkbox(T("Simuler AVC", "Simulate Stroke", "Schlaganfall simulieren"), False, key=f"avc_{current_reset_id}")
    show_adc_map = st.sidebar.checkbox(T("Carte ADC", "ADC Map", "ADC-Karte"), False, key=f"adc_{current_reset_id}")
elif is_asl:
    pld = st.sidebar.slider("PLD", 500, 3000, 1800, step=100, key=f"pld_{current_reset_id}")
    tr = st.sidebar.slider("TR", 3000.0, 8000.0, 4500.0, step=100.0, key=f"tr_asl_{current_reset_id}")
    te = 15.0; ti = 0.0; flip_angle = 90
    show_stroke = st.sidebar.checkbox(T("AVC", "Stroke", "Schlaganfall"), False, key=f"asl_avc_{current_reset_id}")
    st.session_state.atrophy_active = st.sidebar.checkbox(T("Atrophie", "Atrophy", "Atrophie"), st.session_state.atrophy_active, key=f"asl_atr_{current_reset_id}")
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
        msg = T(f"⚠️ TR ajusté auto<br>({int(min_tr_required)}ms) pour {n_slices} coupes.", 
                f"⚠️ Auto Adjusted TR<br>({int(min_tr_required)}ms) for {n_slices} slices.",
                f"⚠️ Automatisch angepasster TR<br>({int(min_tr_required)}ms) für {n_slices} Schichten.")
        st.sidebar.markdown(f"""<div class="tr-alert-box">{msg}</div>""", unsafe_allow_html=True)
    elif ('T1' in seq_choix and tr > 700):
        st.sidebar.markdown(f"""<div class="tr-alert-box">{T("⚠️ Attention Dépassement T1", "⚠️ T1 Limit Exceeded", "⚠️ T1-Limit überschritten")}</div>""", unsafe_allow_html=True)

    if n_concats > 1:
        tr_opti = np.ceil(min_tr_required / 10) * 10
        if tr > (tr_opti + 100):
            def set_optimized_tr(val):
                st.session_state.tr_force = val
                st.session_state.widget_tr = val
            msg_opt = T("Optimisation", "Optimize", "Optimieren")
            st.sidebar.markdown(f"""<div class="opt-box"><b>{msg_opt} {n_concats} Concats</b><br>TR Min : <b>{int(tr_opti)} ms</b></div>""", unsafe_allow_html=True)
            st.sidebar.button(f"📉 {T('Appliquer', 'Apply', 'Anwenden')} TR {int(tr_opti)} ms", on_click=set_optimized_tr, args=(tr_opti,))

# --- RÉTABLISSEMENT DU RÉGLAGE ECHO (TE) ---
st.sidebar.subheader(T("Réglage Echo", "Echo Settings", "Echo-Einstellungen"))

te = st.sidebar.slider(
    T("TE effectif (ms)", "Effective TE (ms)", "Effektive TE (ms)"), 
    min_value=1.0, 
    max_value=300.0, 
    value=te, 
    step=0.5, 
    key=f"te_{current_reset_id}"
)
if is_ir: ti = st.sidebar.slider("TI", 0.0, 3500.0, float(defaults['ti']), step=10.0, key=f"ti_{current_reset_id}")
else: ti = 0.0

if is_gre: flip_angle = st.sidebar.slider(T("Angle (°)", "Flip Angle (°)", "Flipwinkel (°)"), 5, 90, 15, key=f"fa_{current_reset_id}")
elif is_swi: 
    flip_angle = st.sidebar.slider(T("Angle (°)", "Flip Angle (°)", "Flipwinkel (°)"), 5, 40, 15, key=f"fa_{current_reset_id}")
    show_microbleeds = st.sidebar.checkbox(T("Micro-saignements", "Microbleeds", "Mikroblutungen"), False, key=f"cmb_{current_reset_id}")
else: flip_angle = 90

# ====================================================================
# ⚠️ ATTENTION À L'INDENTATION ICI : ON REVIENT COLLÉ AU BORD GAUCHE
# ====================================================================

# --- 3. OPTIONS ---
st.sidebar.header(T("3. Options", "3. Options", "3. Optionen"))
nex = st.sidebar.slider("NEX", 1, 8, 1, key=f"nex_{current_reset_id}")

# Mémoire Turbo
turbo = 1
if not (is_gre or is_dwi or is_swi or is_asl):
    def_turbo = st.session_state.mem_turbo
    turbo = st.sidebar.slider(T("Facteur Turbo", "Turbo Factor", "Turbofaktor"), 1, 32, def_turbo, key=f"turbo_{current_reset_id}")
    st.session_state.mem_turbo = turbo

bw = st.sidebar.slider(T("Bande Passante (Hz/Pixel)", "Bandwidth (Hz/Pixel)", "Bandbreite (Hz/Pixel)"), 50, 500, 220, 10, key=f"bw_{current_reset_id}")
es = st.sidebar.slider(T("Espace Inter-Echo (ES)", "Echo Spacing (ES)", "Echoabstand (ES)"), 2.5, 20.0, 10.0, step=2.5, key=f"es_{current_reset_id}")

# --- 4. IMAGERIE PARALLÈLE ---
st.sidebar.header(T("4. Imagerie Parallèle (iPAT)", "4. Parallel Imaging (iPAT)", "4. Parallele Bildgebung (iPAT)"))
ipat_on = st.sidebar.checkbox(T("Activer Accélération", "Enable Acceleration", "Beschleunigung aktivieren"), value=False, key=f"ipat_on_{current_reset_id}")
ipat_factor = st.sidebar.slider(T("Facteur R", "R Factor", "R-Faktor"), 2, 4, 2, key=f"ipat_r_{current_reset_id}") if ipat_on else 1

st.sidebar.markdown("---")

# MENTIONS LÉGALES
with st.sidebar.expander(T("🛡️ Mentions Légales", "🛡️ Legal Notice", "🛡️ Rechtlicher Hinweis")):
    st.markdown(T("""
    **MagnétoVault Simulator © 2025**
    
    **1. Usage Pédagogique :** Ce simulateur est un outil exclusivement éducatif. Il ne doit **en aucun cas** être utilisé pour du diagnostic médical ou de la recherche clinique.
    """, """
    **MagnétoVault Simulator © 2025**
    
    **1. Educational Use:** This simulator is an educational tool. It must **NOT** be used for medical diagnosis or clinical research.
    """, """
    **MagnétoVault Simulator © 2025**
    
    **1. Pädagogische Nutzung:** Dieser Simulator ist ein rein pädagogisches Werkzeug. Er darf **unter keinen Umständen** für medizinische Diagnosen oder klinische Forschung verwendet werden.
    """))

# MENTIONS LÉGALES
with st.sidebar.expander(T("🛡️ Mentions Légales", "🛡️ Legal Notice")):
    st.markdown(T("""
    **MagnétoVault Simulator © 2025**
    
    **1. Usage Pédagogique :** Ce simulateur est un outil exclusivement éducatif. Il ne doit **en aucun cas** être utilisé pour du diagnostic médical ou de la recherche clinique sur des patients.
    
    **2. Propriété Intellectuelle :** Le code source et la conception sont protégés. Toute reproduction sans accord est interdite.
    
    **3. Responsabilité :** L'auteur décline toute responsabilité quant à l'interprétation des données simulées.
    
    📧 **Contact :** [magnetovault@gmail.com](mailto:magnetovault@gmail.com)
    """, """
    **MagnétoVault Simulator © 2025**
    
    **1. Educational Use:** This simulator is an educational tool. It must **NOT** be used for medical diagnosis or clinical research on patients.
    
    **2. Intellectual Property:** The source code and design are protected. Unauthorized reproduction is prohibited.
    
    **3. Liability:** The author declines all responsibility for the interpretation of the simulated data.
    
    📧 **Contact:** [magnetovault@gmail.com](mailto:magnetovault@gmail.com)
    """))

# BIBLIOGRAPHIE
with st.sidebar.expander(T("📚 Bibliographie & Crédits", "📚 Bibliography & Credits")):
    st.markdown(T("""
    L'onglet **Anatomie** repose sur des outils scientifiques open-source reconnus :
    
    * **Moteur Python :** [Nilearn](https://nilearn.github.io/) (Machine learning for Neuro-Imaging in Python).
    * **Template Géométrique :** **MNI152** (ICBM 2009c Nonlinear Asymmetric).
    * **Atlas Cortical & Sous-cortical :** [Harvard-Oxford Structural Atlases](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases) (FMRIB Centre, University of Oxford).
    * **Visualisation :** Plotly & Matplotlib.
    """, """
    The **Anatomy** tab relies on recognized open-source scientific tools:
    
    * **Python Engine:** [Nilearn](https://nilearn.github.io/) (Machine learning for Neuro-Imaging in Python).
    * **Geometric Template:** **MNI152** (ICBM 2009c Nonlinear Asymmetric).
    * **Cortical & Subcortical Atlas:** [Harvard-Oxford Structural Atlases](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases) (FMRIB Centre, University of Oxford).
    * **Visualization:** Plotly & Matplotlib.
    """))

# ==============================================================================
# FIN DE L'ÉTAPE 1 (Séquences Intelligentes & Design Compact)
# ==============================================================================
# ==============================================================================
# 🛠️ ZONE DE RÉPARATION : SIGNAUX & FANTÔMES & SWI
# ==============================================================================

# INITIALISATION CRITIQUE
tr_effective = tr 

# --- 1. PATCH DE SÉCURITÉ TR (CORRIGÉ SANS CRASH) ---
if tr < 20:
    # On force la valeur locale pour les calculs (Image réparée)
    tr = 500.0
    tr_effective = 500.0 
    
    # IMPORTANT : On ne touche PAS à st.session_state ici pour éviter l'erreur API
    # Le slider restera visuellement bas, mais l'image sera correcte.
    st.toast("⚠️ TR corrigé pour le calcul (Sécurité)", icon="🔧")

# --- CALCUL DE LA DURÉE D'ACQUISITION ---
try:
    raw_ms = phy.calculate_acquisition_time(tr, mat, nex, turbo, ipat_factor, n_concats, n_slices, is_mprage)
except AttributeError:
    base_time = (tr * mat * nex) / (turbo * ipat_factor)
    if is_mprage: raw_ms = base_time * n_slices
    else: raw_ms = base_time * n_concats

final_seconds = raw_ms / 1000.0
mins = int(final_seconds // 60)
secs = int(final_seconds % 60)
str_duree = f"{mins} min {secs} s"

# --- APPEL DU MOTEUR ---
res = get_computed_physics(tr, te, ti, flip_angle, is_gre, is_dwi, b_value if is_dwi else 0, seq_choix)

v_lcr = res["v_lcr"]
v_wm = res["v_wm"]
v_gm = res["v_gm"]
v_stroke = res["v_stroke"]

# --- LA LIGNE MANQUANTE ICI ---
# On calcule directement le signal de la graisse pour qu'elle s'affiche toujours
v_fat = phy.calculate_signal(tr, te, ti, 250, 60, 40, 0, 0.9, flip_angle, is_gre, is_dwi, 0) if not is_dwi else 0.0

# --- 3. CALCUL DU SNR (SOLUTION ULTIME STABLE) ---
snr_tr_ref = float(defaults['tr'])
snr_te_ref = float(defaults['te'])

# 1. Signal de référence
ref_wm_signal = phy.calculate_signal(snr_tr_ref, snr_te_ref, ti, cst.T_WM['T1'], cst.T_WM['T2'], cst.T_WM['T2s'], cst.T_WM['ADC'], cst.T_WM['PD'], 90, False, False, 0)
if ref_wm_signal == 0: ref_wm_signal = 0.001 

# 2. CALCUL DU SNR DE RÉFÉRENCE POUR 1 COUPE (On fixe n_slices à 1 ici)
# On crée une variable locale 'snr_pour_une_coupe' en ignorant la variable n_slices du slider
snr_pour_une_coupe = phy.calculate_snr_relative(mat, nex, turbo, ipat_factor, bw, fov, ep, v_wm, ref_wm_signal)

if is_mprage:
    # En 3D, le signal s'additionne réellement avec les coupes (partitions)
    # On multiplie manuellement par la racine du curseur n_slices
    snr_val = snr_pour_une_coupe * np.sqrt(max(1, n_slices))
else:
    # EN 2D, ON FORCE L'AFFICHAGE DU SNR D'UNE SEULE COUPE
    # Peu importe la valeur du curseur n_slices, le résultat sera le même.
    snr_val = snr_pour_une_coupe

# 3. Calibration finale
snr_val = snr_val * 1.25 
str_snr = f"{snr_val:.1f} %"
# ==============================================================================
# GENERATION FANTOME ROBUSTE (VERSION FINALE)
# ==============================================================================

# D. Initialisation des Matrices (INDISPENSABLE POUR SWI)
S = mat
x = np.linspace(-1, 1, S)
y = np.linspace(-1, 1, S)
X, Y = np.meshgrid(x, y)
D = np.sqrt(X**2 + Y**2)

img_water = np.zeros((S, S))
img_fat = np.zeros((S, S))

# Variables GLOBALES pour l'onglet SWI (évite le crash)
phase_map = np.zeros((S, S))    
dipole_field = np.zeros((S, S)) 

# --- LOGIQUE DE DESSIN ---

# CAS 1 : DIPÔLE (SWI)
if phantom_choice_raw == opt_dipole:
    # Sphère de fond
    mask_sphere = D < 0.6
    img_water[mask_sphere] = 0.8
    
    # Calcul Mathématique du Dipôle
    center = S // 2
    y_idx, x_idx = np.ogrid[:S, :S]
    dist_px = np.sqrt((x_idx - center)**2 + (y_idx - center)**2) + 1e-6
    
    cos_theta = (y_idx - center) / dist_px
    raw_dipole = (3 * cos_theta**2 - 1) / (dist_px**3)
    
    # On sauvegarde le champ pour l'onglet SWI (Important!)
    dipole_field = np.clip(raw_dipole * 1000, -np.pi, np.pi) 
    
    # Le "caillot" central (Signal noir)
    core_mask = dist_px < (S * 0.05)
    img_water[core_mask] = 0.0
    
    # On remplit la carte de phase (C'est ça qui fait l'effet SWI)
    phase_map = dipole_field * 5.0
    
    # Alerte pédagogique
    if not (is_swi or is_gre):
        st.sidebar.warning(T("⚠️ Passez en SWI ou Gradient Echo !", "⚠️ Switch to SWI or GRE!"))

# CAS 2 : BOUTEILLE
elif phantom_choice_raw == opt_bottle:
    mask_rect = (np.abs(X) < 0.4) & (np.abs(Y) < 0.7)
    img_water[mask_rect] = 0.9
    for bx, by, br in [(0.1, 0.2, 0.05), (-0.1, -0.3, 0.08), (0.2, -0.5, 0.04)]:
        mask_bubble = np.sqrt((X-bx)**2 + (Y-by)**2) < br
        img_water[mask_bubble] = 0.0

# CAS 3 : CERVEAU (Défaut)
else:
    # On vérifie si les variables existent, sinon on met des valeurs de secours
    # Cela évite le crash si une séquence (comme DWI) ne calcule pas certains tissus
    val_lcr    = v_lcr if 'v_lcr' in locals() and v_lcr > 0 else 1.0
    val_wm     = v_wm  if 'v_wm' in locals() and v_wm  > 0 else 0.6
    val_gm     = v_gm  if 'v_gm' in locals() and v_gm  > 0 else 0.8
    val_fat    = v_fat if 'v_fat' in locals() else 0.0
    val_stroke = v_stroke if 'v_stroke' in locals() else 0.0

    # Mode Carte ADC (Diffusion)
    if is_dwi and show_adc_map:
        val_lcr = 1.0; val_wm = 0.3; val_gm = 0.35; val_stroke = 0.15; val_fat = 0.0

    # Application sur le fantôme
    img_water[D < 0.20] = val_lcr
    img_water[(D >= 0.20) & (D < 0.50)] = val_wm
    img_water[(D >= 0.50) & (D < 0.80)] = val_gm
    
    # Sécurité pour la graisse (évite d'afficher de la graisse en DWI par exemple)
    img_fat[(D >= 0.80) & (D < 0.95)] = val_fat

    if show_stroke: 
        lesion_mask = (np.sqrt((X-0.3)**2 + (Y-0.1)**2) < 0.12)
        mask_valid = lesion_mask & (D >= 0.20)
        img_water[mask_valid] = val_stroke

# --- ASSEMBLAGE FINAL DE L'IMAGE ---

# 1. Chemical Shift (Fat Sat)
shift_pixels = 0.0 if bw == 220 else 220.0 / float(bw)
img_fat_shifted = shift(img_fat, [0, shift_pixels], mode='constant', cval=0.0)

# 2. Magnitude de base
magn_image = np.clip(img_water + img_fat_shifted, 0, 1.3)

# 3. Application de la Phase (Vital pour SWI)
complex_image = magn_image * np.exp(1j * phase_map)

# 4. Ajout du Bruit Réaliste (Réel + Imaginaire)
noise_level = 5.0 / (snr_val + 20.0)
n_real = np.random.normal(0, noise_level, (S, S))
n_imag = np.random.normal(0, noise_level, (S, S))
final_complex = complex_image + (n_real + 1j * n_imag)

# 5. Image Finale affichée (Module)
final = np.abs(final_complex)
final = np.clip(final, 0, 1.3)

# 6. Espace K (FFT mis en cache ultra-rapide)
kspace = get_k_space(final_complex)
                     
# ==============================================================================
# FONCTION ONGLET SÉCURITÉ (DMI & PHYSIQUE)
# ==============================================================================
@st.fragment
def render_safety_tab():
    st.header(T("🔥 Sécurité RF & Compatibilité DMI", "🔥 RF Safety & DMI Compatibility", "🔥 HF-Sicherheit & MRT-Kompatibilität"))
    
    # --- 📢 MESSAGE D'ACCUEIL & MENTIONS LÉGALES ---
    st.info(T(
        "**🎯 Objectif du module :** Ce logiciel a été conçu spécifiquement pour assister les manipulateurs en électroradiologie médicale (MERM) et les médecins radiologues dans la gestion quotidienne de la sécurité IRM. Il permet de centraliser la recherche de compatibilité des Dispositifs Médicaux Implantables (DMI) grâce à l'intelligence artificielle, et d'estimer théoriquement les contraintes physiques (SAR, B1+rms) liées aux séquences.",
        "**🎯 Module Objective:** This software was specifically designed to assist radiographers (MRI Techs) and radiologists in the daily management of MRI safety. It centralizes compatibility research for Active Implantable Medical Devices (AIMD) using artificial intelligence, and theoretically estimates physical constraints (SAR, B1+rms) related to sequences.",
        "**🎯 Ziel des Moduls:** Diese Software wurde speziell entwickelt, um MTRA und Radiologen beim täglichen Management der MRT-Sicherheit zu unterstützen. Sie zentralisiert die Kompatibilitätsprüfung für aktive implantierbare medizinische Geräte (AIMD) mithilfe künstlicher Intelligenz und schätzt theoretisch die physikalischen Belastungen (SAR, B1+rms) der Sequenzen ab."
    ))
    
    with st.expander(T("⚖️ Cadre Législatif & Responsabilité (⚠️ À LIRE)", "⚖️ Legislative Framework & Liability (⚠️ READ ME)", "⚖️ Rechtlicher Rahmen & Haftung (⚠️ BITTE LESEN)"), expanded=False):
        st.warning(T(
            "**Ce logiciel n'est PAS un Dispositif Médical certifié (marquage CE).**\n\n* **Aide à la décision uniquement :** Les résultats générés par le module IA et les estimations physiques sont donnés à titre purement indicatif.\n* **Responsabilité humaine :** L'utilisateur final reste seul responsable de la validation des données, de la consultation des manuels officiels des constructeurs, et de la décision finale.\n* **Aucune garantie :** L'auteur décline toute responsabilité en cas d'incident résultant de l'utilisation de cet outil.",
            "**This software is NOT a certified Medical Device (CE marking).**\n\n* **Decision support only:** Results generated by the AI module and physical estimations are for informational purposes only.\n* **Human responsibility:** The end user remains solely responsible for validating data, consulting official manufacturer manuals, and making the final decision.\n* **No warranty:** The author declines all responsibility for incidents resulting from the use of this tool.",
            "**Diese Software ist KEIN zertifiziertes Medizinprodukt (CE-Kennzeichnung).**\n\n* **Nur Entscheidungshilfe:** Die vom KI-Modul generierten Ergebnisse und physikalischen Schätzungen dienen nur zu Informationszwecken.\n* **Menschliche Verantwortung:** Der Endbenutzer trägt die alleinige Verantwortung für die Validierung der Daten, die Konsultation der offiziellen Herstellerhandbücher und die endgültige Entscheidung.\n* **Keine Garantie:** Der Autor lehnt jede Haftung für Vorfälle ab, die aus der Nutzung dieses Tools resultieren."
        ))
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 22px !important;
        font-weight: bold !important;
        padding: 5px 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    tab_rf, tab_dmi = st.tabs([
        T("📊 1. Moniteurs Physiques (SAR & B1+rms)", "📊 1. Physical Monitors (SAR & B1+rms)", "📊 1. Physikalische Monitore (SAR & B1+rms)"),
        T("🦿 2. Assistant IA DMI & Rapports", "🦿 2. AI DMI Assistant & Reports", "🦿 2. KI-Implantat-Assistent & Berichte")
    ])
     
    # =========================================================
    # ONGLET 1 : SÉCURITÉ PHYSIQUE
    # =========================================================
    with tab_rf:
        st.warning(T(
            "⚠️ **Simulateur Théorique :** Module reproduisant les contraintes de la norme (IEC 60601-2-33). **Ce module est éducatif et ne remplace pas les moniteurs réels du constructeur.**",
            "⚠️ **Theoretical Simulator:** Module reproducing the norm constraints (IEC 60601-2-33). **This module is educational and does not replace the manufacturer's real monitors.**",
            "⚠️ **Theoretischer Simulator:** Modul, das die normativen Einschränkungen (IEC 60601-2-33) nachbildet. **Dieses Modul ist pädagogisch und ersetzt nicht die echten Monitore des Herstellers.**"
        ))

        # --- 1. CONFIGURATION ---
        SAR_CALIB_FACTOR = 0.005
        k_sinc = T("Sinc (Standard 2D)", "Sinc (Standard 2D)", "Sinc (Standard 2D)")
        k_rect = T("Rect (Hard Pulse 3D)", "Rect (Hard Pulse 3D)", "Rect (Hard Pulse 3D)")
        k_gauss = T("Gauss (Sélectif)", "Gauss (Selective)", "Gauss (Selektiv)")

        PULSE_LIBRARY = {
            k_sinc:  {"factor": 1.0, "desc": T("Pour coupes 2D nettes", "For sharp 2D slices", "Für scharfe 2D-Schichten")},
            k_rect:  {"factor": 1.4, "desc": T("Pour volumes 3D rapides", "For fast 3D volumes", "Für schnelle 3D-Volumina")},
            k_gauss: {"factor": 0.7, "desc": T("Pour Saturation ou Inversion", "For Saturation or Inversion", "Für Sättigung oder Inversion")}
        }
        
        RF_MODES = {"Low SAR": 0.8, "Normal": 1.0, "High Power": 1.2}

        # --- 2. ENTRÉES UTILISATEUR ---
        c_pat, c_seq, c_scan = st.columns(3)
        
        with c_pat:
            st.markdown(f"#### {T('👤 Patient', '👤 Patient', '👤 Patient')}")
            weight = st.number_input(T("Poids (kg)", "Weight (kg)", "Gewicht (kg)"), 30.0, 150.0, 75.0, 5.0, key="saf_weight")
            height = st.number_input(T("Taille (m)", "Height (m)", "Größe (m)"), 1.0, 2.2, 1.75, 0.05, key="saf_height")

        with c_seq:
            st.markdown(f"#### {T('📡 Séquence', '📡 Sequence', '📡 Sequenz')}")
            seq_type = st.selectbox(T("Type Séquence", "Sequence Type", "Sequenztyp"), ["Spin Echo (SE)", "Turbo Spin Echo (TSE)", "Echo de Gradient (GRE)"], key="saf_seq_type")
            b0_val = st.radio(T("Champ Magnétique (B0)", "Magnetic Field (B0)", "Magnetfeld (B0)"), [1.5, 3.0], horizontal=True, key="saf_b0")
            pulse_shape = st.selectbox(T("Forme Onde", "Waveform", "Wellenform"), list(PULSE_LIBRARY.keys()), index=0, key="saf_pulse_shape")
            
            if "GRE" in seq_type:
                def_etl, def_ang = 0, 20
                label_angle = T("Angle d'Excitation (α)", "Excitation Angle (α)", "Anregungswinkel (α)")
            elif "TSE" in seq_type: 
                def_etl, def_ang = 3, 180
                label_angle = T("Angle de Refoc (°)", "Refoc Angle (°)", "Refokussierungswinkel (°)")
            else: 
                def_etl, def_ang = 0, 180
                label_angle = T("Angle de Refoc (°)", "Refoc Angle (°)", "Refokussierungswinkel (°)")

            angle = st.slider(label_angle, 5, 180, def_ang, key="saf_angle")
            
            if "TSE" in seq_type:
                etl = st.slider(T("ETL (Facteur Turbo)", "ETL (Turbo Factor)", "ETL (Turbofaktor)"), 2, 64, def_etl, key="saf_etl")
            else:
                etl = 0
                st.slider("ETL", 0, 1, 0, disabled=True, key="saf_etl_disabled")

        with c_scan:
            st.markdown(f"#### {T('⚙️ Paramètres Scan', '⚙️ Scan Settings', '⚙️ Scan-Parameter')}")
            tr_saf = st.number_input("TR (ms)", 20, 10000, 600, 50, key="saf_tr")
            nb_slices = st.slider(T("Nombre de Coupes", "Number of Slices", "Anzahl der Schichten"), 1, 60, 20, key="saf_slices")
            rf_mode_name = st.select_slider("Mode RF", options=list(RF_MODES.keys()), value="Normal", key="saf_rf_mode")
            rf_intensity = RF_MODES[rf_mode_name]

        st.divider()

        # --- 3. MOTEUR PHYSIQUE ---
        factor_b0 = (b0_val / 1.5) ** 2 
        energy_90 = 1.0 
        energy_angle_slider = (angle / 90.0) ** 2 
        
        if "GRE" in seq_type: total_energy_per_slice = energy_angle_slider 
        elif "TSE" in seq_type: total_energy_per_slice = energy_90 + (etl * energy_angle_slider) 
        else: total_energy_per_slice = energy_90 + (1 * energy_angle_slider) 
        
        total_energy_per_tr = total_energy_per_slice * nb_slices
        power_factor = total_energy_per_tr / (tr_saf / 1000.0)
        factor_weight = 75.0 / weight
        factor_shape = PULSE_LIBRARY[pulse_shape]["factor"]
        
        sar_val = SAR_CALIB_FACTOR * factor_b0 * power_factor * factor_weight * rf_intensity * factor_shape
        
        peak_angle = angle 
        b1_peak_est = (peak_angle / 90.0) * 4.0 * rf_intensity 
        
        if "GRE" in seq_type: p_count = 1
        elif "TSE" in seq_type: p_count = 1 + etl
        else: p_count = 2
        
        duty_cycle = (p_count * nb_slices * 2.5) / tr_saf 
        duty_cycle = min(duty_cycle, 1.0)
        b1_rms_ut = b1_peak_est * np.sqrt(duty_cycle) * factor_shape

        # --- 4. VISUALISATION ---
        st.subheader(T("📊 Moniteurs de Sécurité", "📊 Safety Monitors", "📊 Sicherheitsmonitore"))
        
        c_visu_g, c_visu_d = st.columns([1, 1])
        
        with c_visu_g:
            st.markdown(f"##### {T('📉 Profil RF & Charge', '📉 RF Profile & Load', '📉 HF-Profil & Belastung')}")
            fig_p = Figure(figsize=(5, 2.5)); ax_p = fig_p.subplots()
            t_axis = np.linspace(-1, 1, 200)
            if "Rect" in pulse_shape: y_pulse = np.where(np.abs(t_axis)<0.5, 1, 0)
            elif "Sinc" in pulse_shape: y_pulse = np.sinc(t_axis * 3)
            else: y_pulse = np.exp(-t_axis**2 * 5)
                
            y_pulse = y_pulse * b1_peak_est
            ax_p.plot(t_axis, y_pulse, color='#8e44ad', lw=2)
            ax_p.fill_between(t_axis, y_pulse, color='#8e44ad', alpha=0.2)
            ax_p.set_ylim(0, max(10, b1_peak_est * 1.3))
            ax_p.set_yticks([]); ax_p.set_xticks([])
            ax_p.set_ylabel("B1 (µT)")
            ax_p.set_title(T(f"Pic B1: {b1_peak_est:.1f} µT (x{factor_b0:.0f} énergie à {b0_val}T)", f"B1 Peak: {b1_peak_est:.1f} µT", f"B1 Peak: {b1_peak_est:.1f} µT (x{factor_b0:.0f} Energie bei {b0_val}T)"), fontsize=9, color='gray')
            st.pyplot(fig_p); 
            
            if b0_val == 3.0:
                st.error(T("⚠️ **ATTENTION 3T** : Énergie x4 par rapport à 1.5T.", "⚠️ **WARNING 3T**: Energy x4.", "⚠️ **ACHTUNG 3T**: Energie x4 im Vergleich zu 1.5T."))

        with c_visu_d:
            def draw_gauge_cursor(value, label, limit_norm, limit_first, max_scale=6.0):
                fig = Figure(figsize=(6, 2)); ax = fig.subplots()
                ax.add_patch(plt.Rectangle((0, 0), limit_norm, 1, color='#2ecc71', alpha=0.9))
                ax.text(limit_norm/2, 0.5, "NORMAL", ha='center', va='center', color='white', fontweight='bold', fontsize=8)
                ax.add_patch(plt.Rectangle((limit_norm, 0), limit_first-limit_norm, 1, color='#f1c40f', alpha=0.9))
                ax.text((limit_norm+limit_first)/2, 0.5, "LEVEL 1", ha='center', va='center', color='white', fontweight='bold', fontsize=8)
                ax.add_patch(plt.Rectangle((limit_first, 0), max_scale-limit_first, 1, color='#e74c3c', alpha=0.9))
                ax.text((limit_first+max_scale)/2, 0.5, "STOP", ha='center', va='center', color='white', fontweight='bold', fontsize=8)
                
                cursor_pos = min(value, max_scale - 0.1)
                ax.plot([cursor_pos, cursor_pos], [-0.2, 1.2], color='black', linewidth=4)
                ax.text(cursor_pos, 1.35, f"{value:.2f}", ha='center', fontweight='bold', fontsize=12, color='black')
                ax.set_xlim(0, max_scale); ax.set_ylim(0, 1.6); ax.axis('off')
                ax.set_title(label, loc='left', fontweight='bold')
                return fig

            st.pyplot(draw_gauge_cursor(sar_val, T("SAR Global (W/kg)", "Global SAR (W/kg)", "Globales SAR (W/kg)"), 2.0, 4.0))
            st.pyplot(draw_gauge_cursor(b1_rms_ut, "B1+rms (µT)", 2.8, 4.0))

        st.divider()
        
        # --- 5. FORMULES & GLOSSAIRES ---
        c_f1, c_f2 = st.columns(2)
        with c_f1:
            st.markdown(f"#### {T('🌡️ Calcul du SAR', '🌡️ SAR Calculation', '🌡️ SAR-Berechnung')}")
            st.latex(r"SAR \propto B_0^2 \times E_{totale} \times \frac{1}{TR \cdot Poids}")
        with c_f2:
            st.markdown(f"#### {T('⚡ Calcul du B1+rms', '⚡ B1+rms Calculation', '⚡ B1+rms-Berechnung')}")
            st.latex(r"B_{1}^{+rms} \propto B_{1,peak} \times \sqrt{DC}")

        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            with st.expander(T("📖 Facteurs SAR (Détails)", "📖 SAR Factors (Details)", "📖 SAR-Faktoren (Details)")):
                 st.info(T(
                     "**Pourquoi le SAR est CALCULÉ ?**\nLe SAR (Débit d'Absorption Spécifique) mesure l'échauffement des tissus. Il est mathématiquement **calculé** car il est impossible à mesurer directement in vivo. Ce calcul dépend obligatoirement de la morphologie du patient (poids), du champ statique (B0) et du modèle thermique du constructeur.",
                     "**Why is SAR CALCULATED?**\nSAR measures tissue heating. It is mathematically **calculated** because it's impossible to measure directly in vivo. This calculation strictly depends on patient morphology (weight), static field (B0), and the manufacturer's thermal model.",
                     "**Warum wird die SAR BERECHNET?**\nDie SAR (Spezifische Absorptionsrate) misst die Erwärmung des Gewebes. Sie wird mathematisch **berechnet**, da sie in vivo nicht direkt messbar ist. Diese Berechnung hängt zwingend von der Morphologie des Patienten (Gewicht), dem statischen Feld (B0) und dem thermischen Modell des Herstellers ab."
                 ))
                 if "GRE" in seq_type:
                    st.markdown(T(
                    "* **Type** : Écho de Gradient (GRE).\n* **Énergie** : Une seule impulsion d'excitation (Angle variable de **5° à 90°**).\n* **Analyse** : Le SAR est réduit car il n'y a **pas de train d'impulsions de refocalisation**.", 
                    "* **Type**: Gradient Echo (GRE).\n* **Energy**: Single excitation pulse (Variable angle **5° to 90°**).\n* **Analysis**: SAR is low because there is **no refocusing pulse train**.",
                    "* **Typ**: Gradientenecho (GRE).\n* **Energie**: Ein einziger Anregungspuls (Variabler Winkel **5° bis 90°**).\n* **Analyse**: Die SAR ist gering, da es **keine Refokussierungspulse** gibt."
                    ))
                 else:
                    st.markdown(T(
                    f"* **Type** : Spin Echo / TSE.\n* **Énergie** : Excitation 90° (Fixe) + Refocalisations {angle}° (Variable).\n* **Poids du 180°** : Un pulse 180° chauffe **4x** plus qu'un 90°.", 
                    f"* **Type**: Spin Echo / TSE.\n* **Energy**: Excitation 90° (Fixed) + Refocusing {angle}° (Variable).\n* **Weight of 180°**: A 180° pulse heats **4x** more than a 90°.",
                    f"* **Typ**: Spin-Echo / TSE.\n* **Energie**: 90°-Anregung (Fest) + Refokussierung {angle}° (Variabel).\n* **Gewicht des 180°-Pulses**: Ein 180°-Puls erwärmt **4x** mehr als ein 90°-Puls."
                    ))
                    
        with c_exp2:
            with st.expander(T("📖 Facteurs B1+rms (Détails)", "📖 B1+rms Factors (Details)", "📖 B1+rms-Faktoren (Details)")):
                 st.info(T(
                     "**Pourquoi le B1+rms est ESTIMÉ ?**\nLe B1+rms représente l'amplitude moyenne du champ magnétique RF. Il est **estimé** car c'est une grandeur physique pure (en µT) générée par l'antenne. **Il est indépendant du poids ou de l'anatomie du patient**.",
                     "**Why is B1+rms ESTIMATED?**\nB1+rms represents the average amplitude of the RF magnetic field. It is **estimated** because it's a pure physical quantity (in µT) generated by the coil. **It is independent of patient weight or anatomy**.",
                     "**Warum wird B1+rms GESCHÄTZT?**\nB1+rms stellt die durchschnittliche Amplitude des HF-Magnetfelds dar. Es wird **geschätzt**, da es eine rein physikalische Größe (in µT) ist, die von der Spule erzeugt wird. **Es ist unabhängig von Gewicht oder Anatomie des Patienten**."
                 ))
                 if "GRE" in seq_type:
                    st.markdown(T(
                    "* **B1 Peak** : Dépend de l'angle d'excitation.\n* **Duty Cycle** : Très faible (1 pulse par TR).", 
                    "* **B1 Peak**: Depends on excitation angle.\n* **Duty Cycle**: Very low (1 pulse per TR).",
                    "* **B1 Peak**: Hängt vom Anregungswinkel ab.\n* **Duty Cycle**: Sehr niedrig (1 Puls pro TR)."
                    ))
                 else:
                    st.markdown(T(
                    "* **B1 Peak** : Intensité des pulses de refocalisation.\n* **Duty Cycle** : Élevé en TSE (mitraillage).", 
                    "* **B1 Peak**: Refocusing pulse intensity.\n* **Duty Cycle**: High in TSE (rapid firing).",
                    "* **B1 Peak**: Intensität der Refokussierungspulse.\n* **Duty Cycle**: Hoch bei TSE (schnelle Abfolge)."
                    ))
        
        st.divider()
        
        with st.expander(T("📝 Seuils & Paramètres IEC", "📝 IEC Thresholds & Parameters", "📝 IEC-Grenzwerte & Parameter"), expanded=False):
            st.markdown(T(
                "* 🟢 :green[**Mode Normal**] : **< 2.0 W/kg** (Routine Clinique, aucun risque).\n* 🟠 :orange[**Mode Contrôlé (Niveau 1)**] : **2.0 - 4.0 W/kg** (Surveillance médicale requise).\n* 🔴 :red[**Mode Restreint (Niveau 2)**] : **> 4.0 W/kg** (Blocage logiciel, risque d'échauffement > 1°C).", 
                "* 🟢 :green[**Normal Mode**]: **< 2.0 W/kg** (Clinical Routine, no risk).\n* 🟠 :orange[**First Level Mode**]: **2.0 - 4.0 W/kg** (Medical supervision required).\n* 🔴 :red[**Second Level Mode**]: **> 4.0 W/kg** (Software lockout, heating risk > 1°C).",
                "* 🟢 :green[**Normaler Modus**]: **< 2.0 W/kg** (Klinische Routine, kein Risiko).\n* 🟠 :orange[**Kontrollierter Modus (Stufe 1)**]: **2.0 - 4.0 W/kg** (Medizinische Überwachung erforderlich).\n* 🔴 :red[**Eingeschränkter Modus (Stufe 2)**]: **> 4.0 W/kg** (Software-Sperre, Erwärmungsrisiko > 1°C)."
            ))

        with st.expander(T("🏥 Clinique : Formes d'Impulsions & Séquences", "🏥 Clinical: Pulse Shapes & Sequences", "🏥 Klinik: Pulsformen & Sequenzen"), expanded=False):
            table_formes_fr = "\n".join([
                "| Forme | Usage Principal | Avantage | Risque / Inconvénient |",
                "| :--- | :--- | :--- | :--- |",
                "| **Sinc** | **TSE, SE (2D)** | Profil de coupe rectangulaire (Pas de croisement). | **SAR Élevé** (Impulsions longues & nombreuses). |",
                "| **Rectangulaire** | **MP-RAGE (3D)** | Ultra-rapide (TR court). | Coupe \"sale\" (bords flous) - corrigé par encodage 3D. |",
                "| **Gaussienne** | **Fat Sat** | Très sélectif en fréquence. | **Pic B1 Élevé** (Stress sur l'ampli RF). |"
            ])
            table_formes_en = "\n".join([
                "| Shape | Main Usage | Advantage | Risk / Drawback |",
                "| :--- | :--- | :--- | :--- |",
                "| **Sinc** | **TSE, SE (2D)** | Rectangular slice profile (No crosstalk). | **High SAR** (Long & numerous pulses). |",
                "| **Rectangular** | **MP-RAGE (3D)** | Ultra-fast (Short TR). | \"Dirty\" slice (blurred edges) - corrected by 3D encoding. |",
                "| **Gaussian** | **Fat Sat** | Highly frequency selective. | **High B1 Peak** (RF Amp Stress). |"
            ])
            table_formes_de = "\n".join([
                "| Form | Hauptnutzung | Vorteil | Risiko / Nachteil |",
                "| :--- | :--- | :--- | :--- |",
                "| **Sinc** | **TSE, SE (2D)** | Rechteckiges Schichtprofil (Kein Crosstalk). | **Hohe SAR** (Lange & viele Pulse). |",
                "| **Rechteckig** | **MP-RAGE (3D)** | Ultraschnell (Kurze TR). | \"Unsaubere\" Schicht (unscharfe Ränder) - korrigiert durch 3D-Kodierung. |",
                "| **Gauß** | **Fat Sat** | Sehr frequenzselektiv. | **Hoher B1 Peak** (HF-Verstärker-Stress). |"
            ])
            st.markdown(T(table_formes_fr, table_formes_en, table_formes_de))

        with st.expander(T("🎯 Guide Pratique : Impact des Paramètres", "🎯 Quick Guide: Parameters Impact", "🎯 Kurzanleitung: Auswirkung der Parameter"), expanded=True):
            st.markdown(T("Ce tableau résume le comportement des paramètres d'acquisition sur les deux moniteurs de sécurité.", 
                          "This table summarizes the behavior of acquisition parameters on both safety monitors.",
                          "Diese Tabelle fasst das Verhalten der Akquisitionsparameter auf die beiden Sicherheitsmonitore zusammen."))
            
            table_impact_fr = "\n".join([
                "| Paramètre | Action | Impact sur le SAR (Chaleur) | Impact sur le B1+rms (Implant) |",
                "| :--- | :--- | :--- | :--- |",
                "| ⚖️ **Poids (Weight)** | ⬆️ Augmentation | ⬇️ **Baisse** (Énergie diluée dans la masse) | ➖ **Aucun effet** (Grandeur matérielle) |",
                "| 📏 **Taille (Height)** | ⬆️ Augmentation | ➖ **Aucun effet direct** | ➖ **Aucun effet direct** |",
                "| 📡 **Type Séquence** | GRE ➔ SE ➔ TSE | ⬆️ **Forte Augmentation** (Nb d'impulsions ↗) | ⬆️ **Augmentation** (Duty Cycle ↗) |",
                "| 🧲 **Champ B0** | 1.5T ➔ 3.0T | ⬆️ **Quadruple (x4)** | ➖ **Aucun effet direct** (Le seuil reste en µT) |",
                "| 〰️ **Forme Onde** | Rect ➔ Sinc ➔ Gauss | ⬆️ **Augmentation** (Énergie de l'impulsion ↗) | ⬆️ **Augmentation** (Pic B1 ↗) |",
                "| 📐 **Angle (α)** | ⬆️ Augmentation | ⬆️ **Forte Augmentation (x²)** | ⬆️ **Augmentation directe** |",
                "| 🚀 **ETL (Turbo)** | ⬆️ Augmentation | ⬆️ **Augmentation** (Mitraillage RF) | ⬆️ **Augmentation** (Duty Cycle ↗) |",
                "| ⏱️ **TR** | ⬆️ Augmentation | ⬇️ **Baisse** (Plus de temps de refroidissement)| ⬇️ **Baisse** (Duty Cycle ↘) |",
                "| 🍕 **Nb Coupes** | ⬆️ Augmentation | ⬆️ **Augmentation** (Plus d'impulsions par TR) | ⬆️ **Augmentation** (Duty Cycle ↗) |",
                "| 🔋 **Mode RF** | Low ➔ Normal ➔ High | ⬆️ **Augmentation** | ⬆️ **Augmentation** |"
            ])
            table_impact_en = "\n".join([
                "| Parameter | Action | Impact on SAR (Heating) | Impact on B1+rms (Implant) |",
                "| :--- | :--- | :--- | :--- |",
                "| ⚖️ **Weight** | ⬆️ Increase | ⬇️ **Decreases** (Energy diluted in mass) | ➖ **No effect** (Hardware quantity) |",
                "| 📏 **Height** | ⬆️ Increase | ➖ **No effect directly** | ➖ **No effect directly** |",
                "| 📡 **Sequence Type**| GRE ➔ SE ➔ TSE | ⬆️ **Strong Increase** (Nb of pulses ↗) | ⬆️ **Increases** (Duty Cycle ↗) |",
                "| 🧲 **B0 Field** | 1.5T ➔ 3.0T | ⬆️ **Quadruples (x4)** | ➖ **No effect directly** (Threshold stays in µT) |",
                "| 〰️ **Waveform** | Rect ➔ Sinc ➔ Gauss | ⬆️ **Increases** (Pulse energy ↗) | ⬆️ **Increases** (B1 Peak ↗) |",
                "| 📐 **Angle (α)** | ⬆️ Increase | ⬆️ **Strong Increase (x²)** | ⬆️ **Direct Increase** |",
                "| 🚀 **ETL (Turbo)** | ⬆️ Increase | ⬆️ **Increases** (RF rapid firing) | ⬆️ **Increases** (Duty Cycle ↗) |",
                "| ⏱️ **TR** | ⬆️ Increase | ⬇️ **Decreases** (More cooling time) | ⬇️ **Decreases** (Duty Cycle ↘) |",
                "| 🍕 **Nb Slices** | ⬆️ Increase | ⬆️ **Increases** (More pulses per TR) | ⬆️ **Increases** (Duty Cycle ↗) |",
                "| 🔋 **RF Mode** | Low ➔ Normal ➔ High | ⬆️ **Increases** | ⬆️ **Increases** |"
            ])
            table_impact_de = "\n".join([
                "| Parameter | Aktion | Auswirkung auf SAR (Wärme) | Auswirkung auf B1+rms (Implantat) |",
                "| :--- | :--- | :--- | :--- |",
                "| ⚖️ **Gewicht** | ⬆️ Erhöhen | ⬇️ **Sinkt** (Energie verteilt sich auf mehr Masse) | ➖ **Kein Effekt** (Hardware-Größe) |",
                "| 📏 **Größe** | ⬆️ Erhöhen | ➖ **Kein direkter Effekt** | ➖ **Kein direkter Effekt** |",
                "| 📡 **Sequenztyp**| GRE ➔ SE ➔ TSE | ⬆️ **Starker Anstieg** (Anzahl der Pulse ↗) | ⬆️ **Steigt** (Duty Cycle ↗) |",
                "| 🧲 **B0-Feld** | 1.5T ➔ 3.0T | ⬆️ **Vervierfacht sich (x4)** | ➖ **Kein direkter Effekt** (Grenzwert bleibt in µT) |",
                "| 〰️ **Wellenform** | Rect ➔ Sinc ➔ Gauss | ⬆️ **Steigt** (Pulsenergie ↗) | ⬆️ **Steigt** (B1 Peak ↗) |",
                "| 📐 **Winkel (α)** | ⬆️ Erhöhen | ⬆️ **Starker Anstieg (x²)** | ⬆️ **Direkter Anstieg** |",
                "| 🚀 **ETL (Turbo)** | ⬆️ Erhöhen | ⬆️ **Steigt** (Schnelle HF-Abfolge) | ⬆️ **Steigt** (Duty Cycle ↗) |",
                "| ⏱️ **TR** | ⬆️ Erhöhen | ⬇️ **Sinkt** (Mehr Abkühlzeit) | ⬇️ **Sinkt** (Duty Cycle ↘) |",
                "| 🍕 **Schichten** | ⬆️ Erhöhen | ⬆️ **Steigt** (Mehr Pulse pro TR) | ⬆️ **Steigt** (Duty Cycle ↗) |",
                "| 🔋 **HF-Modus** | Low ➔ Normal ➔ High | ⬆️ **Steigt** | ⬆️ **Steigt** |"
            ])
            st.markdown(T(table_impact_fr, table_impact_en, table_impact_de))
            st.markdown(T("*💡 **L'astuce du Manipulateur :** Pour faire baisser le **B1+rms** d'une séquence TSE récalcitrante, le moyen le plus efficace est d'augmenter le **TR** ou de baisser l'angle de refocalisation (ex: 120° au lieu de 180°).* ", 
                          "*💡 **Tech Tip:** To lower the **B1+rms** of a stubborn TSE sequence, the most effective way is to increase the **TR** or lower the refocusing angle (e.g., 120° instead of 180°).* ",
                          "*💡 **Tipp für MTRA:** Um die **B1+rms** einer hartnäckigen TSE-Sequenz zu senken, ist der effektivste Weg, die **TR** zu erhöhen oder den Refokussierungswinkel zu senken (z. B. 120° statt 180°).* "))

    # =========================================================
    # ONGLET 2 : ASSISTANT IA & DMI
    # =========================================================
    with tab_dmi:
        st.error(T(
            "⚠️ **AVERTISSEMENT CLINIQUE :** Ce module exige votre validation humaine. L'IA facilite la recherche, mais **VOUS** êtes responsable de la vérification finale et de la saisie des données.",
            "⚠️ **CLINICAL WARNING:** This module requires human validation. The AI facilitates research, but **YOU** are responsible for final verification and data entry.",
            "⚠️ **KLINISCHE WARNUNG:** Dieses Modul erfordert Ihre menschliche Validierung. Die KI erleichtert die Suche, aber **SIE** sind für die endgültige Überprüfung und Dateneingabe verantwortlich."
        ))
        
        # Définition de la langue cible pour l'IA
        target_lang = "FRANÇAIS"
        if st.session_state.lang == 'en': target_lang = "ENGLISH"
        elif st.session_state.lang == 'de': target_lang = "DEUTSCH"
        
        if 'widget_key' not in st.session_state: st.session_state.widget_key = 0
        if "etape_dmi" not in st.session_state: st.session_state.etape_dmi = 0
        if "nom_dmi_memoire" not in st.session_state: st.session_state.nom_dmi_memoire = ""
        if "sources_ia" not in st.session_state: st.session_state.sources_ia = ""
        if "fiche_ia" not in st.session_state: st.session_state.fiche_ia = ""
        if "rapport_final_html" not in st.session_state: st.session_state.rapport_final_html = ""
        
        # --- ÉTAPE 1 : IDENTIFICATION ---
        st.markdown(T("### 1️⃣ Identification du dispositif", "### 1️⃣ Device Identification", "### 1️⃣ Geräteidentifikation"))
        
        nom_dmi = st.text_input(T("Saisissez un Nom, Modèle ou Réf/Lot :", "Enter a Name, Model, or Ref/Lot:", "Geben Sie einen Namen, ein Modell oder eine Ref/Lot ein:"), placeholder="Ex: Medtronic Advisa, Nucleus 7...", key=f"input_nom_dmi_{st.session_state.widget_key}")
        
        st.markdown(T("**📄 Preuve visuelle de la carte (Optionnel) :**", "**📄 Visual proof of the card (Optional):**", "**📄 Visueller Nachweis der Karte (Optional):**"))
        
        onglets_img1, onglets_img2, onglets_img3 = st.tabs([T("📁 Parcourir", "📁 Browse", "📁 Durchsuchen"), T("📸 Appareil Photo", "📸 Camera", "📸 Kamera"), T("📋 Coller (Presse-papier)", "📋 Paste (Clipboard)", "📋 Einfügen (Zwischenablage)")])
        
        with onglets_img1:
            fichier_preuve = st.file_uploader(T("Glissez ou sélectionnez une image :", "Drag or select an image:", "Bild hierher ziehen oder auswählen:"), type=['png', 'jpg', 'jpeg'], key=f"input_file_dmi_{st.session_state.widget_key}")
        
        with onglets_img2:
            st.info(T("💡 Astuce : Si votre tablette ne bascule pas sur la caméra arrière, prenez la photo avec l'application native de votre tablette et utilisez l'onglet 'Parcourir' ci-dessus.", 
                      "💡 Tip: If your tablet cannot switch to the back camera, take the photo with your native app and use the 'Browse' tab above.",
                      "💡 Tipp: Wenn Ihr Tablet nicht auf die Rückkamera umschaltet, machen Sie das Foto mit Ihrer nativen App und verwenden Sie oben die Registerkarte 'Durchsuchen'."))
            activer_camera = st.toggle(T("🎥 Activer l'appareil photo", "🎥 Enable camera", "🎥 Kamera aktivieren"), key=f"toggle_cam_{st.session_state.widget_key}")
            photo_camera = None
            if activer_camera:
                photo_camera = st.camera_input(T("Prendre une photo de la carte du patient :", "Take a picture of the patient's card:", "Machen Sie ein Foto vom Ausweis des Patienten:"), key=f"camera_dmi_{st.session_state.widget_key}")

        with onglets_img3:
            st.info(T("Copiez une image (Ctrl+C) puis cliquez sur le bouton ci-dessous :", "Copy an image (Ctrl+C) then click the button below:", "Kopieren Sie ein Bild (Strg+C) und klicken Sie dann auf die Schaltfläche unten:"))
            label_text = "📋 Coller l'image" if st.session_state.lang == 'fr' else ("📋 Paste image" if st.session_state.lang == 'en' else "📋 Bild einfügen")
            image_collee = paste_image_button(
                label=label_text,
                background_color="#4A86e8",
                hover_background_color="#205b9f",
                text_color="#ffffff"
            )
            if image_collee.image_data is not None:
                st.success(T("✅ Image collée avec succès !", "✅ Image pasted successfully!", "✅ Bild erfolgreich eingefügt!"))
                st.image(image_collee.image_data, width=250)

        image_fournie = None
        if fichier_preuve: image_fournie = fichier_preuve
        elif photo_camera: image_fournie = photo_camera
        elif image_collee.image_data is not None: image_fournie = image_collee.image_data

        if nom_dmi != st.session_state.nom_dmi_memoire:
            st.session_state.nom_dmi_memoire = nom_dmi
            st.session_state.etape_dmi = 0

        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            btn_rechercher = st.button(T("🔍 1. Rechercher le constructeur et les accès", "🔍 1. Search manufacturer & access", "🔍 1. Hersteller & Zugriff suchen"), use_container_width=True)
        with col_btn2:
            btn_reset = st.button(T("🔄 Nouvelle recherche", "🔄 New search", "🔄 Neue Suche"), use_container_width=True)

        if btn_reset:
            st.session_state.widget_key += 1 
            st.session_state.etape_dmi = 0
            st.session_state.nom_dmi_memoire = ""
            st.session_state.sources_ia = ""
            st.session_state.fiche_ia = ""
            st.session_state.rapport_final_html = ""
            st.rerun()

        texte_manuel = ""

        if btn_rechercher:
            if nom_dmi or image_fournie:
                st.session_state.etape_dmi = 1
                with st.spinner(T("Analyse par Gemini en cours...", "Analysis by Gemini in progress...", "Analyse durch Gemini läuft...")):
                    
                    titre_id = T("### 🆔 Identification du Système Complet (Boîtier + Sondes)", "### 🆔 Complete System Identification (Device + Leads)", "### 🆔 Komplette Systemidentifikation (Gerät + Sonden)")
                    titre_pre = T("### 🧠 Pré-Analyse Clinique (Mémoire IA)", "### 🧠 Clinical Pre-Analysis (AI Memory)", "### 🧠 Klinische Voranalyse (KI-Speicher)")
                    titre_val = T("### 🎯 Validation Médico-Légale (Liens de Recherche)", "### 🎯 Medico-Legal Validation (Search Links)", "### 🎯 Medizolegale Validierung (Suchlinks)")
                    
                    prompt_sources = f"""
                    INSTRUCTION SYSTÈME : Tu es un script d'extraction de données cliniques.
                    RÈGLE ABSOLUE 0 : Tu DOIS générer l'intégralité de ta réponse en {target_lang}.
                    
                    Dispositif identifié par texte ou image : "{nom_dmi if nom_dmi else "IMAGE CI-JOINTE."}"
                    
                    RÈGLE ABSOLUE 1 : Zéro bavardage. Ne dis AUCUN mot avant le titre "{titre_id}".
                    RÈGLE ABSOLUE 2 : Si une image est fournie, tu DOIS faire un inventaire exhaustif. Tu dois lister LE BOÎTIER ET TOUTES LES SONDES PRÉSENTES.
                    
                    ANNUAIRE DES PORTAILS OFFICIELS :
                    - Medtronic : https://manuals.medtronic.com/manuals/mri/
                    - Abbott / St. Jude : https://www.cardiovascular.abbott/us/en/hcp/resources/mri-ready.html
                    - Boston Scientific : https://www.bostonscientific.com/en-US/mri-safety.html
                    - Biotronik : https://www.biotronik.com/en-de/products/manuals
                    - Cochlear : https://www.cochlear.com/global/en/professionals/resources/mri-guidelines
                    - LivaNova : https://www.livanova.com/en-us/mri
                    - MicroPort : https://www.microportmanuals.com/
                    
                    MODÈLE DE RÉPONSE OBLIGATOIRE (À TRADUIRE EN {target_lang}) :
                    
                    {titre_id}
                    * **Boîtier / Générateur :** [Préciser la Marque et le Modèle exact]
                    * **Sonde(s) implantée(s) :** [Lister ICI obligatoirement toutes les sondes visibles. Avertissement "Mix-and-Match" si marques différentes].
                    
                    {titre_pre}
                    *(Ces données sont issues de la base IA pour vous orienter).*
                    * 🧲 **Champ Magnétique (B0) :** [Préciser les Tesla]
                    * 🌡️ **Limites d'exposition (SAR / B1+rms) :** [Préciser W/kg ou µT]
                    * ⚠️ **Restrictions Cliniques Majeures :** [Sois extrêmement exhaustif]
                    
                    {titre_val}
                    **🏛️ Portails Officiels :**
                    * **[Marque] :** [Lien de l'annuaire]
                    
                    **🔍 Recherches Google Optimisées :**
                    * **Recherche pour Boîtier [Marque + Modèle] :** https://www.google.com/search?q=%22[Marque]%22+%22[Modèle]%22+%28%22MRI+Safety%22+OR+%22Conditions+IRM%22+OR+%22Manuel%22%29
                    """
                    
                    contenu_etape1 = [prompt_sources]
                    
                    if image_fournie:
                        try:
                            if isinstance(image_fournie, Image.Image):
                                img = image_fournie
                            else:
                                image_fournie.seek(0)
                                img = Image.open(image_fournie)
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            contenu_etape1.append(img)
                        except Exception as e:
                            st.error(f"Erreur image : {e}")

                    try:
                        if "client" in globals():
                            reponse_src = client.models.generate_content(model='gemini-2.5-flash', contents=contenu_etape1)
                            st.session_state.sources_ia = reponse_src.text
                        else:
                            st.error("API Gemini non initialisée.")
                    except Exception as e:
                        st.error(f"Erreur IA : {e}")
            else:
                st.warning(T("Veuillez saisir un nom ou fournir une photo.", "Please enter a name or provide a photo.", "Bitte geben Sie einen Namen ein oder stellen Sie ein Foto zur Verfügung."))
                
        if st.session_state.etape_dmi >= 1:
            st.markdown(st.session_state.sources_ia)
            st.divider()

            # --- ÉTAPE 2 : VALIDATION ---
            st.markdown(T("### 2️⃣ Validation et Conditions IRM", "### 2️⃣ Validation and MRI Conditions", "### 2️⃣ Validierung und MRT-Bedingungen"))
            st.info(T("Allez sur le portail du fabricant, copiez le texte des conditions de sécurité IRM et collez-le ci-dessous.", 
                      "Go to the manufacturer's portal, copy the MRI safety conditions text, and paste it below.",
                      "Gehen Sie zum Herstellerportal, kopieren Sie den Text der MRT-Sicherheitsbedingungen und fügen Sie ihn unten ein."))
            
            texte_manuel = st.text_area(T("Collez les données brutes du fabricant ici :", "Paste raw manufacturer data here:", "Fügen Sie hier die Rohdaten des Herstellers ein:"), height=150)

            if st.button(T("⚙️ 2. Analyser les conditions de sécurité", "⚙️ 2. Analyze safety conditions", "⚙️ 2. Sicherheitsbedingungen analysieren"), use_container_width=True):
                if texte_manuel:
                    st.session_state.etape_dmi = 2
                    with st.spinner(T("Extraction ultra-rapide...", "Ultra-fast extraction...", "Ultraschnelle Extraktion...")):
                        
                        prompt_analyse = f"""
                        INSTRUCTION SYSTÈME : Extraction stricte. AUCUNE phrase d'introduction ni de conclusion.
                        RÈGLE ABSOLUE 0 : Tu DOIS générer l'intégralité de ta réponse en {target_lang}. Traduis les titres de la liste ci-dessous.
                        
                        Analyse ce texte officiel du fabricant et extrais UNIQUEMENT les conditions de sécurité pour l'IRM.
                        Texte du fabricant : "{texte_manuel}"

                        Génère UNIQUEMENT ce format (à traduire) :
                        * 🧲 **Champ Magnétique Autorisé :** [Ex: 1.5T, 3T]
                        * 🌡️ **SAR Maximum :** [Ex: 2.0 W/kg]
                        * ⚡ **B1+rms Maximum :** [Ex: 2.8 µT]
                        * 📐 **Gradient Spatial Max :** [Ex: 3000 Gauss/cm]
                        * 🚫 **Contraintes cliniques / Exclusions :** [Ex: Scan de la tête uniquement]
                        """
                        try:
                            if "client" in globals():
                                reponse_analyse = client.models.generate_content(model='gemini-2.5-flash', contents=[prompt_analyse])
                                st.session_state.fiche_ia = reponse_analyse.text
                        except Exception as e:
                            st.error(f"Erreur IA : {e}")
                else:
                    st.warning(T("Veuillez coller le texte du manuel.", "Please paste the manual text.", "Bitte fügen Sie den Handbuchtext ein."))

        if st.session_state.etape_dmi >= 2:
            st.success(T("✅ Analyse terminée :", "✅ Analysis complete:", "✅ Analyse abgeschlossen:"))
            st.markdown(st.session_state.fiche_ia)
            st.divider()

            # --- ÉTAPE 3 : RAPPORT HTML DYNAMIQUE TRILINGUE ---
            st.markdown(T("### 3️⃣ Génération de la Fiche Clinique", "### 3️⃣ Clinical Form Generation", "### 3️⃣ Generierung des klinischen Formulars"))

            if st.button(T("📝 3. Remplir et Imprimer la Fiche Clinique", "📝 3. Fill and Print Clinical Form", "📝 3. Klinisches Formular ausfüllen und drucken"), use_container_width=True):
                st.session_state.etape_dmi = 3
                with st.spinner(T("Création de la mise en page...", "Creating layout...", "Erstellen des Layouts...")):
                    
                    # Dictionnaire HTML traduit dynamiquement
                    t_title = T("Fiche de compatibilité IRM pour patient porteur de DMI", "MRI Compatibility Form for AIMD Patient", "MRT-Kompatibilitätsformular für AIMD-Patienten")
                    t_pat = T("Patient", "Patient", "Patient")
                    t_name = T("Nom :", "Last Name:", "Nachname:")
                    t_fname = T("Prénom :", "First Name:", "Vorname:")
                    t_dob = T("Né(e) le :", "DOB:", "Geb. am:")
                    t_mri_dt = T("Examen IRM le :", "MRI date:", "MRT-Datum:")
                    t_type = T("Type :", "Type:", "Typ:")
                    t_tel = T("Tél :", "Phone:", "Tel:")
                    t_comp_sec = T("Compatibilité IRM", "MRI Compatibility", "MRT-Kompatibilität")
                    t_warn = T("Toute association de dispositifs non testés ensemble est considérée comme Unsafe", "Any combination of untested medical devices is considered Unsafe", "Jede Kombination ungetesteter Medizinprodukte gilt als Unsafe")
                    t_dev = T("Dispositifs Médicaux", "Medical Devices", "Medizinprodukte")
                    t_th_type = T("Type", "Type", "Typ")
                    t_th_brand = T("Marque", "Brand", "Marke")
                    t_th_ref = T("Référence", "Reference", "Referenz")
                    t_th_dt = T("Date pose", "Implant Date", "Datum")
                    t_th_comp = T("Compatibilité", "Compatibility", "Kompatibilität")
                    t_aimd_type = T("Type de DMI :", "AIMD Type:", "AIMD Typ:")
                    t_act = T("Actif", "Active", "Aktiv")
                    t_pass = T("Passif", "Passive", "Passiv")
                    t_risk1 = T("-> Risque de dysfonctionnement", "-> Malfunction risk", "-> Fehlfunktionsrisiko")
                    t_ferro = T("Matériau ferromagnétique :", "Ferromagnetic material:", "Ferromagnetisches Material:")
                    t_yes = T("Oui", "Yes", "Ja")
                    t_no = T("Non", "No", "Nein")
                    t_risk2 = T("-> Risque d'attraction, torsion", "-> Attraction, torsion risk", "-> Anziehungs-/Torsionsrisiko")
                    t_cond = T("Matériau Conducteur :", "Conductive Material:", "Leitfähiges Material:")
                    t_risk3 = T("-> Risque d'échauffement", "-> Heating risk", "-> Erwärmungsrisiko")
                    t_cond_sec = T("Conditions préconisées par le constructeur", "Conditions recommended by manufacturer", "Vom Hersteller empfohlene Bedingungen")
                    t_b0 = T("Champ Magnétique max (B0) :", "Max static magnetic field (B0):", "Max. statisches Magnetfeld (B0):")
                    t_spat = T("Gradients spatial max :", "Max spatial gradient :", "Max. räumlicher Gradient :")
                    t_slew = T("Vitesse de montée (Slew Rate) :", "Slew rate :", "Slew-Rate :")
                    t_amp = T("Amplitude max des gradients :", "Max gradient amplitude :", "Max. Gradientenamplitude :")
                    t_lvl1 = T("Niveau 1", "Level 1", "Stufe 1")
                    t_lvl2 = T("Niveau 2", "Level 2", "Stufe 2")
                    t_oth = T("Autre :", "Other:", "Sonstiges:")
                    t_time = T("Temps d'examen max :", "Max scan time:", "Maximale Scanzeit:")
                    t_coil = T("Antennes :", "Coils:", "Spulen:")
                    t_excl = T("Zone d'exclusion :", "Exclusion zone:", "Ausschlusszone:")
                    t_pos = T("Positionnement patient :", "Patient positioning:", "Patientenpositionierung:")
                    t_loc = T("Localisation DMI autorisé :", "Allowed AIMD location:", "Zulässige AIMD-Position:")
                    t_spec = T("Contrôle par un spécialiste :", "Check by specialist:", "Prüfung durch Spezialisten:")
                    t_surv = T("Surveillance pendant examen :", "Monitoring during exam:", "Überwachung während der Untersuchung:")
                    t_done = T("Fait le :", "Date:", "Datum:")
                    t_tech = T("Manipulateur :", "Radiographer:", "MTRA:")
                    t_dr = T("Médecin ok :", "Physician OK:", "Arzt OK:")

                    html_template = f"""
                    <div style="font-family: Arial, sans-serif; border: 1px solid #ccc; padding: 20px; border-radius: 5px; background-color: white; color: black;"><h2 style='text-align: center; color: #1f497d; text-decoration: underline;'>{t_title}</h2><p><strong>{t_pat}</strong><br><strong>{t_name}</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>{t_fname}</strong> <br><strong>{t_dob}</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>{t_mri_dt}</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>{t_type}</strong> <br><strong>{t_tel}</strong> </p><div style="border: 2px solid black; padding: 10px; margin-bottom: 10px;"><h3 style="color: #4472c4; margin-top: 0;">{t_comp_sec}</h3><p style="font-size: 18px; text-align: center; font-weight: bold;">[ ] MR Safe &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [ ] MR Conditional &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [ ] MR Unsafe</p><p style="color: red; font-weight: bold; text-align: center; font-size: 18px;">{t_warn}</p></div><h3 style="color: #4472c4; text-decoration: underline;">{t_dev}</h3><table style="width: 100%; border-collapse: collapse; text-align: center;" border="1"><tr style="background-color: #f2f2f2;"><th style="padding: 5px;">{t_th_type}</th><th style="padding: 5px;">{t_th_brand}</th><th style="padding: 5px;">{t_th_ref}</th><th style="padding: 5px;">{t_th_dt}</th><th style="padding: 5px;">{t_th_comp}</th></tr><tr><td style="padding: 5px;">&nbsp;</td><td style="padding: 5px;">&nbsp;</td><td style="padding: 5px;">&nbsp;</td><td style="padding: 5px;">&nbsp;</td><td style="padding: 5px;">&nbsp;</td></tr><tr><td style="padding: 5px;">&nbsp;</td><td style="padding: 5px;">&nbsp;</td><td style="padding: 5px;">&nbsp;</td><td style="padding: 5px;">&nbsp;</td><td style="padding: 5px;">&nbsp;</td></tr></table><p style="margin-top: 15px;"><strong>{t_aimd_type}</strong> [ ] {t_act} <span style="color: red; font-weight: bold;">{t_risk1}</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [ ] {t_pass} <br><strong>{t_ferro}</strong> [ ] {t_yes} <span style="color: red; font-weight: bold;">{t_risk2}</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [ ] {t_no} <br><strong>{t_cond}</strong> [ ] {t_yes} <span style="color: red; font-weight: bold;">{t_risk3}</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [ ] {t_no}</p><div style="border: 2px solid black; padding: 15px;"><h3 style="color: #4472c4; text-decoration: underline; margin-top: 0;">{t_cond_sec}</h3><strong>{t_b0}</strong> [ ] 1.5 T &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [ ] 3T <br><strong>{t_spat}</strong> <br><strong>{t_slew}</strong> <br><strong>{t_amp}</strong> <br><strong>SAR :</strong> [ ] {t_lvl1} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [ ] {t_lvl2} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [ ] {t_oth} <br><strong>B1+RMS :</strong> [ ] ≤ 2,8 µT &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [ ] {t_oth} <br><strong>{t_time}</strong> <br><br><strong>{t_coil}</strong> <br><strong>{t_excl}</strong> [ ] {t_yes} &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [ ] {t_no} <br><strong>{t_pos}</strong> <br><strong>{t_loc}</strong> <br><strong>{t_spec}</strong> <br><strong>{t_surv}</strong> <br><strong>{t_oth}</strong> </div><p style="margin-top: 15px;"><strong>{t_done}</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>{t_tech}</strong> .......................... &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>{t_dr}</strong> ..........................</p></div>
                    """

                    prompt_rapport = f"""
                    INSTRUCTION SYSTÈME : Tu es un processeur de code HTML muet.
                    RÈGLE ABSOLUE 0 : Tu DOIS remplir le texte de la fiche en {target_lang}.
                    
                    Tu dois remplir la fiche ci-dessous en croisant précisément ces données :
                    1. INVENTAIRE DU PATIENT : {st.session_state.sources_ia}
                    2. NOTICE DU CONSTRUCTEUR : {texte_manuel}
                    3. ANALYSE DE SÉCURITÉ : {st.session_state.fiche_ia}

                    CONSIGNES STRICTES :
                    - Reproduis EXACTEMENT le code HTML fourni ci-dessous.
                    - Remplace les "[ ]" par "[X]" si la condition est validée.
                    - Complète les informations manquantes DANS LA LANGUE DEMANDÉE ({target_lang}).
                    - Ne génère AUCUN texte en dehors du bloc HTML.
                    
                    MODÈLE HTML À REPRODUIRE ET REMPLIR :
                    {html_template}
                    """
                    
                    try:
                        if "client" in globals():
                            reponse_rapport = client.models.generate_content(model='gemini-2.5-flash', contents=[prompt_rapport])
                            html_brut = reponse_rapport.text.replace("```html", "").replace("```", "").strip()
                            
                            btn_print_text = T("🖨️ CLIQUEZ ICI POUR IMPRIMER", "🖨️ CLICK HERE TO PRINT", "🖨️ HIER KLICKEN ZUM DRUCKEN")
                            annexe_title = T("Annexe - Données Constructeur", "Appendix - Manufacturer Data", "Anhang - Herstellerdaten")
                            
                            html_iframe = f"""
                            <!DOCTYPE html>
                            <html>
                            <head>
                                <meta charset="utf-8">
                                <style>
                                    body {{ font-family: Arial, sans-serif; padding: 20px; background-color: white; color: black; }}
                                    @media print {{
                                        .btn-print {{ display: none !important; }}
                                        .page-break {{ page-break-before: always !important; display: block; height: 1px; }}
                                        body {{ padding: 0; margin: 0; }}
                                    }}
                                    .btn-print {{ background-color: #4A86e8; color: white; padding: 15px 20px; border: none; border-radius: 5px; font-size: 18px; cursor: pointer; margin-bottom: 20px; width: 100%; font-weight: bold; text-align: center; }}
                                    .raw-data {{ border: 2px dashed #4472c4; padding: 20px; background-color: #f8f9fa; font-family: monospace; white-space: pre-wrap; }}
                                </style>
                            </head>
                            <body>
                                <button class="btn-print" onclick="window.print()">{btn_print_text}</button>
                                {html_brut}
                                <div class="page-break"></div>
                                <div style="margin-top: 40px;">
                                    <h2 style='color: #1f497d; text-align: center;'>{annexe_title}</h2>
                                    <div class="raw-data">{texte_manuel}</div>
                                </div>
                            </body>
                            </html>
                            """
                            st.session_state.rapport_final_html = html_iframe
                    except Exception as e:
                        st.error(f"Erreur IA : {e}")

            if st.session_state.etape_dmi >= 3 and st.session_state.rapport_final_html:
                st.success(T("✅ Fiche générée avec succès !", "✅ Form generated successfully!", "✅ Formular erfolgreich erstellt!"))
                components.html(st.session_state.rapport_final_html, height=1000, scrolling=True)

@st.fragment
def render_architecture_tab():
    st.header(T("🏗️ Architecture : Structure et Composants IRM", "🏗️ Architecture: MRI Structure and Components", "🏗️ Architektur: MRT-Struktur und Komponenten"))
    
    # --- SÉLECTEUR DE TECHNOLOGIE ET ACTIONS ---
    c_tech, c_action = st.columns([2, 1])
    with c_tech:
        tech_mode = st.radio(
            T("🔧 Technologie de Refroidissement :", "🔧 Cooling Technology:", "🔧 Kühltechnologie:"),
            [
                T("Classique (Bain d'hélium ~1500L, Quench)", "Classic (Helium bath ~1500L, Quench)", "Klassisch (Heliumbad ~1500L, Quench)"),
                T("Micro-refroidissement (Low-Helium scellé)", "Micro-cooling (Sealed Low-Helium)", "Mikrokühlung (Versiegeltes Low-Helium)")
            ],
            horizontal=True,
            key="arch_tech_mode"
        )
        
    is_classic = "Classique" in tech_mode or "Classic" in tech_mode or "Klassisch" in tech_mode
    
    with c_action:
        st.write("") # Espace d'alignement
        simuler_coupure = st.toggle(
            T("⚡ Simuler Coupure / Quench", "⚡ Simulate Power Loss / Quench", "⚡ Stromausfall / Quench simulieren"), 
            value=False, 
            key="arch_vidange_toggle"
        )

    st.divider()

    # --- DISPOSITION CLASSIQUE (MENU VERTICAL) ---
    col_view, col_desc = st.columns([2.8, 1.2])
    
    with col_desc:
        options_view = [
            T("1. Machine (Coque & Tunnel)", "1. Machine (Shell & Bore)", "1. Maschine (Gehäuse & Tunnel)"),
            T("2. Cryostat & Refroidissement", "2. Cryostat & Cooling", "2. Kryostat & Kühlung"),
            T("3. Aimant B0 & 5 Gauss", "3. B0 Magnet & 5 Gauss", "3. B0-Magnet & 5 Gauss"), 
            T("4. Shim Passif", "4. Passive Shim", "4. Passiver Shim"), 
            T("5. Shim Actif & Blindage", "5. Active Shim & Shielding", "5. Aktiver Shim & Abschirmung"), 
            T("6. GZ (Vert)", "6. GZ (Green)", "6. GZ (Grün)"), 
            T("7. GY (Jaune)", "7. GY (Yellow)", "7. GY (Gelb)"), 
            T("8. GX (Bleu)", "8. GX (Blue)", "8. GX (Blau)"), 
            T("9. Antenne RF (Rouge)", "9. RF Coil (Red)", "9. HF-Spule (Rot)"),
            T("10. Tout visualiser", "10. Show All", "10. Alles anzeigen")
        ]
        
        view_mode = st.radio(
            T("Progression pédagogique :", "Pedagogical progression:", "Pädagogischer Fortschritt:"),
            options_view,
            index=0, key="arch_final_clean"
        )
        
        st.divider()
        if simuler_coupure:
            if is_classic:
                st.error(T("🚨 **QUENCH !**\nLa tête froide s'arrête. L'hélium bout massivement, s'échappe par la cheminée, et l'aimantation B0 s'effondre.", 
                           "🚨 **QUENCH!**\nThe cold head stops. Helium boils massively, escapes through the chimney, and B0 magnetization drops.",
                           "🚨 **QUENCH!**\nDer Kaltkopf stoppt. Helium kocht massiv, entweicht durch den Kamin und die B0-Magnetisierung bricht zusammen."))
            else:
                st.error(T("🚨 **Coupure Électrique !**\nLa tête froide s'arrête. L'hélium est évacué vers la cuve et l'aimantation B0 disparaît.", 
                           "🚨 **Power Outage!**\nThe cold head stops. Helium is evacuated into the tank and B0 magnetization drops.",
                           "🚨 **Stromausfall!**\nDer Kaltkopf stoppt. Helium wird in den Tank evakuiert und die B0-Magnetisierung verschwindet."))
        else:
            st.write(T("🔬 **Analyse** : Visualisation des couches internes de l'aimant.", 
                       "🔬 **Analysis**: Visualizing the magnet's internal layers.",
                       "🔬 **Analyse**: Visualisierung der inneren Schichten des Magneten."))

    with col_view:
        fig = Figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black'); fig.patch.set_facecolor('black')

        # --- LE SECRET POUR MANGER TOUT LE VIDE NOIR ---
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.dist = 5.5  # ZOOM HYPER PUISSANT (Défaut = 10)

        mode_idx = options_view.index(view_mode)
        show_all = (mode_idx == 9)

        lim = 5.5 
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect([1, 1, 1])

        def get_poids(target_idx):
            if show_all:
                if target_idx >= 5: return 1.0  
                return 0.2                      
            if target_idx == mode_idx: return 1.0
            if target_idx == 1 and mode_idx == 2: return 0.4 
            if target_idx < mode_idx: return 0.15
            return 0.0

        def draw_perfect_edges(r, l, alpha):
            for y in [-r, r]:
                for z in [-r, r]:
                    ax.plot([-l, l], [y, y], [z, z], color='white', lw=1.5, alpha=alpha)
            for x in [-l, l]:
                for side in [-r, r]:
                    ax.plot([x, x], [-r, r], [side, side], color='white', lw=1, alpha=alpha*0.6)
                    ax.plot([x, x], [side, side], [-r, r], color='white', lw=1, alpha=alpha*0.6)

        def draw_bipolar_ramp(color, mode, alpha):
            if not show_all:
                pts = np.linspace(-1.1, 1.1, 40)
                for p in pts:
                    if abs(p) < 0.05: continue
                    amp = p * 0.4
                    if mode == "GZ": ax.plot([p, p], [0, 0], [0, amp], color=color, lw=3, alpha=alpha)
                    elif mode == "GX": ax.plot([0, 0], [p, p], [0, amp], color=color, lw=3, alpha=alpha)
                    elif mode == "GY": ax.plot([0, 0], [0, amp], [p, p], color=color, lw=3, alpha=alpha)

        def draw_coil(z_r, t_r, color, alpha):
            t_g = np.deg2rad(np.linspace(t_r[0], t_r[1], 30)); r = 1.05
            ax.plot([z_r[0], z_r[1]], [r*np.cos(t_g[0])]*2, [r*np.sin(t_g[0])]*2, color=color, lw=5, alpha=alpha)
            ax.plot([z_r[0], z_r[1]], [r*np.cos(t_g[-1])]*2, [r*np.sin(t_g[-1])]*2, color=color, lw=5, alpha=alpha)
            ax.plot([z_r[0]]*30, r*np.cos(t_g), r*np.sin(t_g), color=color, lw=5, alpha=alpha)
            ax.plot([z_r[1]]*30, r*np.cos(t_g), r*np.sin(t_g), color=color, lw=5, alpha=alpha)

        # --- 1. MACHINE, TUNNEL & TABLE ---
        p_m = get_poids(0)
        if p_m > 0 or show_all:
            draw_perfect_edges(1.9, 2.2, p_m)
            a_tun = 0.6 if show_all else 0.2
            z_t = np.linspace(-2.2, 2.2, 40); t_t = np.linspace(0, 2*np.pi, 40)
            if show_all: t_t = np.linspace(np.pi/2, 2*np.pi, 40)
            Z_t, T_t = np.meshgrid(z_t, t_t)
            ax.plot_surface(Z_t, 0.85*np.cos(T_t), 0.85*np.sin(T_t), color='whitesmoke', alpha=a_tun)
            
            # TABLE D'EXAMEN
            a_tab = 0.5 if show_all else p_m * 0.9
            z_tab = np.linspace(-5.5, -1.0, 10) 
            y_tab = np.linspace(-0.28, 0.28, 10) 
            Z_tab, Y_tab = np.meshgrid(z_tab, y_tab)
            X_tab = np.full_like(Z_tab, -0.35) 
            ax.plot_surface(Z_tab, Y_tab, X_tab, color='#ecf0f1', alpha=a_tab)
            
            ax.plot([-5.5, -5.5], [-0.28, 0.28], [-0.35, -0.35], color='white', lw=2, alpha=a_tab)
            ax.plot([-5.5, -1.0], [-0.28, -0.28], [-0.35, -0.35], color='white', lw=2, alpha=a_tab)
            ax.plot([-5.5, -1.0], [0.28, 0.28], [-0.35, -0.35], color='white', lw=2, alpha=a_tab)

            z_leg = np.linspace(-4.5, -3.2, 10)
            x_leg = np.linspace(-2.2, -0.35, 10) 
            Z_leg, X_leg = np.meshgrid(z_leg, x_leg)
            Y_leg_L = np.full_like(Z_leg, -0.15)
            Y_leg_R = np.full_like(Z_leg, 0.15)
            ax.plot_surface(Z_leg, Y_leg_L, X_leg, color='#bdc3c7', alpha=a_tab*0.8)
            ax.plot_surface(Z_leg, Y_leg_R, X_leg, color='#bdc3c7', alpha=a_tab*0.8)
            
            y_front = np.linspace(-0.15, 0.15, 10)
            Y_front, X_front = np.meshgrid(y_front, x_leg)
            Z_front = np.full_like(Y_front, -4.5)
            ax.plot_surface(Z_front, Y_front, X_front, color='#95a5a6', alpha=a_tab*0.9)

        # --- 2. CRYOSTAT & TÊTE FROIDE ---
        p_c = get_poids(1)
        if p_c > 0:
            if is_classic:
                z_c = np.linspace(-2.0, 2.0, 40); t_c_s = np.linspace(0, 2*np.pi, 40)
                if show_all: t_c_s = np.linspace(np.pi/2, 2*np.pi, 40)
                Z_c, T_c = np.meshgrid(z_c, t_c_s)
                color_cryo = '#7f8c8d' if simuler_coupure else '#0097e6'
                alpha_cryo = p_c * 0.05 if simuler_coupure else p_c * 0.3
                ax.plot_surface(Z_c, 1.75*np.cos(T_c), 1.75*np.sin(T_c), color=color_cryo, alpha=alpha_cryo)
                
                z_q = np.linspace(1.8, 3.8, 10) ; theta_q = np.linspace(0, 2*np.pi, 20)
                Z_q, T_q = np.meshgrid(z_q, theta_q)
                X_q = 0.8 + 0.4 * np.cos(T_q); Y_q = 0.0 + 0.4 * np.sin(T_q)
                color_quench = '#00d2ff' if simuler_coupure else '#bdc3c7'
                ax.plot_surface(X_q, Y_q, Z_q, color=color_quench, alpha=p_c*0.8)

                if simuler_coupure:
                    z_flux = np.linspace(3.8, 5.5, 15); Z_f, T_f = np.meshgrid(z_flux, theta_q)
                    R_f = 0.4 + 0.4 * (Z_f - 3.8); X_f = 0.8 + R_f * np.cos(T_f); Y_f = 0.0 + R_f * np.sin(T_f)
                    ax.plot_surface(X_f, Y_f, Z_f, color='white', alpha=p_c*0.7)
            else:
                z_serp = np.linspace(-1.8, 1.8, 400)
                y_serp = 1.68 * np.cos(10 * np.pi * z_serp); z_vert_serp = 1.68 * np.sin(10 * np.pi * z_serp)
                color_serp = '#7f8c8d' if simuler_coupure else '#0097e6'
                ax.plot(z_serp, y_serp, z_vert_serp, color=color_serp, alpha=p_c*0.8, lw=1.5)
                
                x_cv = np.linspace(-1.8, -1.2, 10); theta_cv = np.linspace(0, 2*np.pi, 20)
                X_cv, T_cv = np.meshgrid(x_cv, theta_cv)
                Y_cv = -1.2 + 0.25 * np.cos(T_cv); Z_cv = -1.8 + 0.25 * np.sin(T_cv) 
                color_cuve = '#00d2ff' if simuler_coupure else '#2c3e50'
                ax.plot_surface(X_cv, Y_cv, Z_cv, color=color_cuve, alpha=p_c*0.8)

            z_ch = np.linspace(1.8, 2.2, 10); theta_ch = np.linspace(0, 2*np.pi, 20)
            Z_ch, T_ch = np.meshgrid(z_ch, theta_ch)
            X_ch = -1.0 + 0.25 * np.cos(T_ch); Y_ch = 0.0 + 0.25 * np.sin(T_ch)
            color_head = '#7f8c8d' if simuler_coupure else '#e67e22'
            ax.plot_surface(X_ch, Y_ch, Z_ch, color=color_head, alpha=p_c*0.9)

        # --- 3, 4, 5. AIMANT B0, LIGNES DE CHAMP ET 5 GAUSS ---
        p_b0 = 1.0 if mode_idx in [2, 3, 4] else get_poids(2) 
            
        if p_b0 > 0:
            z_h = np.linspace(-1.9, 1.9, 800)
            ax.plot(z_h, 1.6*np.cos(25*np.pi*z_h), 1.6*np.sin(25*np.pi*z_h), color='#8e44ad', alpha=p_b0*0.8, lw=2.0)
            
            if not simuler_coupure:
                ax.quiver(-4.5, 0, 0, 9.0, 0, 0, color='#00d2ff', lw=5, alpha=p_b0, arrow_length_ratio=0.03)
                ax.text(4.7, 0, 0, "B0", color='#00d2ff', fontsize=18, weight='bold', ha='center', va='center', alpha=p_b0)

                if mode_idx <= 2:
                    r_5g_z = 8.5; r_5g_xy = 8.5  
                elif mode_idx == 3:
                    r_5g_z = 6.6; r_5g_xy = 6.6  
                else:
                    r_5g_z = 4.8; r_5g_xy = 4.8  
                
                t_5g = np.linspace(0, 2 * np.pi, 100)
                ax.plot(r_5g_z * np.cos(t_5g), r_5g_xy * np.sin(t_5g), np.zeros_like(t_5g), color='red', lw=2.5, ls=':', alpha=p_b0 * 0.8)
                ax.plot(r_5g_z * np.cos(t_5g), np.zeros_like(t_5g), r_5g_xy * np.sin(t_5g), color='red', lw=2.5, ls=':', alpha=p_b0 * 0.8)
                ax.plot(np.zeros_like(t_5g), r_5g_xy * np.cos(t_5g), r_5g_xy * np.sin(t_5g), color='red', lw=2.5, ls=':', alpha=p_b0 * 0.8)
                
                if mode_idx >= 2:
                    text_pos = min(r_5g_xy, 4.5)
                    ax.text(0, text_pos, 0.5, "5 Gauss", color='red', fontsize=14, fontweight='bold', alpha=p_b0)

                z_l = np.linspace(-4.5, 4.5, 200) 
                num_lines = 10  
                angles = np.linspace(0, 2 * np.pi, num_lines, endpoint=False)
                base_radius = 0.55
                
                if mode_idx <= 3:
                    flare = np.where(np.abs(z_l) > 1.9, (np.abs(z_l) - 1.9)**1.8 * 0.4, 0)
                else:
                    flare = np.where(np.abs(z_l) > 1.9, (np.abs(z_l) - 1.9)**1.2 * 0.1, 0)

                if mode_idx <= 2: 
                    radial_dev = 0.25 * np.sin(4 * z_l) + flare
                elif mode_idx == 3: 
                    radial_dev = 0.08 * np.sin(5 * z_l) + flare
                else: 
                    radial_dev = np.zeros_like(z_l) + flare
                
                mid_idx = len(z_l) // 2
                for theta in angles:
                    x_line = (base_radius + radial_dev) * np.cos(theta)
                    y_line = (base_radius + radial_dev) * np.sin(theta)
                    ax.plot(z_l, y_line, x_line, color='#00d2ff', lw=1.5, alpha=p_b0*0.6)
                    ax.quiver(0, y_line[mid_idx], x_line[mid_idx], 0.5, 0, 0, color='#00d2ff', lw=1.5, alpha=p_b0*0.9, arrow_length_ratio=0.3)

        # --- 4, 5. SHIMS ---
        p_passif = get_poids(3)
        if p_passif > 0:
            theta_plates = np.linspace(0, 2*np.pi, 6, endpoint=False)
            z_plates = [-1.0, 0.0, 1.0]
            for z_p in z_plates:
                for t_p in theta_plates:
                    ax.scatter(z_p, 1.48 * np.cos(t_p), 1.48 * np.sin(t_p), color='#bdc3c7', s=80, marker='s', alpha=p_passif)

        p_sh = get_poids(4)
        t_circ = np.linspace(0, 2*np.pi, 100)
        if p_sh > 0:
            for z_p in [-1.8, 1.8]:
                ax.plot([z_p]*100, 1.45*np.cos(t_circ), 1.45*np.sin(t_circ), color='orange', lw=6, alpha=p_sh)

        # --- 6, 7, 8. GRADIENTS ---
        p_gz = get_poids(5)
        if p_gz > 0:
            ax.plot([0.8]*100, np.cos(t_circ), np.sin(t_circ), color='#27ae60', lw=7, alpha=p_gz)
            ax.plot([-0.8]*100, np.cos(t_circ), np.sin(t_circ), color='#27ae60', lw=7, alpha=p_gz)
            if mode_idx == 5: 
                draw_bipolar_ramp('#27ae60', "GZ", 1.0)
                ax.text(0, 0, 2.2, T("Bobines de Maxwell (GZ)", "Maxwell Coils (GZ)", "Maxwell-Spulen (GZ)"), color='#27ae60', fontsize=14, weight='bold', ha='center', alpha=1.0)
            
        p_gy = get_poids(6)
        if p_gy > 0:
            for z in [[0.1, 0.75], [-0.75, -0.1]]:
                for t in [[65, 115], [245, 295]]: draw_coil(z, t, '#f1c40f', p_gy)
            if mode_idx == 6: 
                draw_bipolar_ramp('#f1c40f', "GY", 1.0)
                ax.text(0, 0, 2.2, T("Bobines de Golay (GY)", "Golay Coils (GY)", "Golay-Spulen (GY)"), color='#f1c40f', fontsize=14, weight='bold', ha='center', alpha=1.0)
            
        p_gx = get_poids(7)
        if p_gx > 0:
            for z in [[0.1, 0.75], [-0.75, -0.1]]:
                for t in [[-25, 25], [155, 205]]: draw_coil(z, t, '#2980b9', p_gx)
            if mode_idx == 7: 
                draw_bipolar_ramp('#2980b9', "GX", 1.0)
                ax.text(0, 0, 2.2, T("Bobines de Golay (GX)", "Golay Coils (GX)", "Golay-Spulen (GX)"), color='#2980b9', fontsize=14, weight='bold', ha='center', alpha=1.0)

        # --- 9. ANTENNE RF ---
        p_rf = get_poids(8)
        if p_rf > 0:
            r_rf = 0.75
            z_rf_ends = [-1.0, 1.0]
            for z_ring in z_rf_ends:
                ax.plot([z_ring]*100, r_rf*np.cos(t_circ), r_rf*np.sin(t_circ), color='#ff0000', lw=4, alpha=p_rf)
            for i in range(16):
                angle = i * (2 * np.pi / 16)
                ax.plot(z_rf_ends, [r_rf*np.cos(angle)]*2, [r_rf*np.sin(angle)]*2, color='#ff0000', lw=2, alpha=p_rf)
            if mode_idx == 8:
                ax.text(0, 0, 1.2, "Body Coil (B1)", color='#ff0000', fontsize=16, weight='bold', ha='center', alpha=1.0)

        lim_cam = 4.5 
        ax.set_xlim(-lim_cam, lim_cam)
        ax.set_ylim(-lim_cam, lim_cam)
        ax.set_zlim(-lim_cam, lim_cam)
        
        ax.view_init(elev=20, azim=-115)
        ax.set_axis_off()
        st.pyplot(fig, use_container_width=True)

    # --- EXPLICATIONS DÉTAILLÉES ---
    st.divider()
    cols = st.columns(3)
    with cols[0]:
        st.subheader(T("🧊 Cryogénie & B0", "🧊 Cryogenics & B0", "🧊 Kryotechnik & B0"))
        st.write(T("**Coque, Tunnel & Table** : Structure mécanique et accueil du patient.", "**Shell, Bore & Table** : Mechanical structure and patient accommodation.", "**Gehäuse, Tunnel & Tisch** : Mechanische Struktur und Patientenaufnahme."))
        
        if is_classic:
            st.write(T("**Cryostat (Bain)** : Enceinte thermique remplie de ~1500L d'hélium liquide (-269°C).", "**Cryostat (Bath)** : Thermal enclosure filled with ~1500L of liquid helium (-269°C).", "**Kryostat (Bad)** : Wärmekammer gefüllt mit ~1500L flüssigem Helium (-269°C)."))
        else:
            st.write(T("**Micro-refroidissement** : Un fin serpentin d'hélium liquide refroidit directement l'aimant.", "**Micro-cooling** : A thin coil of liquid helium directly cools the magnet.", "**Mikrokühlung** : Eine dünne Spule aus flüssigem Helium kühlt den Magneten direkt."))

    with cols[1]:
        st.subheader(T("🎯 Homogénéité & Sécurité", "🎯 Homogeneity & Safety", "🎯 Homogenität & Sicherheit"))
        st.write(T("**Ligne 5 Gauss** : Délimite la zone de danger (attraction des métaux). Rétractée par le Blindage Actif.", "**5 Gauss Line** : Defines the danger zone (metal attraction). Retracted by Active Shielding.", "**5-Gauss-Linie** : Definiert die Gefahrenzone (Metallanziehung). Zurückgezogen durch aktive Abschirmung."))
        st.write(T("**Shim Actif & Passif** : Bobines et plaques qui lissent les lignes du champ magnétique B0 à l'intérieur du tunnel.", "**Active & Passive Shim** : Coils and plates that smooth the B0 magnetic field lines inside the bore.", "**Aktiver & passiver Shim** : Spulen und Platten, die die B0-Magnetfeldlinien im Tunnel glätten."))
        
    with cols[2]:
        st.subheader(T("📡 Codage Spatial", "📡 Spatial Encoding", "📡 Räumliche Kodierung"))
        st.write(T("**GZ (Vert)** : Sélection de la coupe transversale.", "**GZ (Green)** : Cross-sectional slice selection.", "**GZ (Grün)** : Auswahl der transversalen Schicht."))
        st.write(T("**GY/GX (Jaune/Bleu)** : Codage de phase et de fréquence.", "**GY/GX (Yellow/Blue)** : Phase and frequency encoding.", "**GY/GX (Gelb/Blau)** : Phasen- und Frequenzkodierung."))
        st.write(T("**Antenne (Rouge)** : Émet la radiofréquence (90°/180°).", "**Coil (Red)** : Emits the radiofrequency (90°/180°).", "**Spule (Rot)** : Sendet die Hochfrequenz (90°/180°)."))
@st.fragment
def render_fatsat_tab():
    st.header(T("🍔 Suppression de Graisse (Fat Sat)", "🍔 Fat Suppression (Fat Sat)", "🍔 Fettunterdrückung (Fat Sat)"))
    
    # --- DÉFINITION DES NOMS DES ONGLETS ---
    fs_tabs_names = [
        T("1. Saturation Fréquentielle", "1. Frequency Saturation", "1. Frequenzselektive Sättigung"),
        T("2. Séquence SPAIR", "2. SPAIR Sequence", "2. SPAIR-Sequenz"),
        T("3. Séquence Dixon", "3. Dixon Sequence", "3. Dixon-Sequenz"),
        T("4. Excitation Eau", "4. Water Excitation", "4. Wasseranregung (WE)"),
        T("5. Soustraction", "5. Subtraction", "5. Subtraktion"),
        T("6. Séquence STIR", "6. STIR Sequence", "6. STIR-Sequenz"),
        T("7. Séquence PSIR", "7. PSIR Sequence", "7. PSIR-Sequenz")
    ]
    
    fs_tabs = st.tabs(fs_tabs_names)
    
    # --- 1. SATURATION FRÉQUENTIELLE (FAT SAT CLASSIQUE) ---
    with fs_tabs[0]:
        st.subheader(T("1. Saturation Fréquentielle (Fat Sat)", "1. Frequency Selective Saturation (Fat Sat)", "1. Frequenzselektive Sättigung (Fat Sat)"))

        st.markdown("#### " + T("A. Séquence Temporelle", "A. Timing Sequence", "A. Sequenzdiagramm"))
        
        st.info(T(
            "1. **Au début du cycle :** Un 90° non sélectif (Bande Large). Pas de recueil de signal lors du premier TR.\n"
            "2. **À la fin du premier TR :** On utilise une bande de SAT Étroite sélective sur le pic de la Graisse.\n"
            "3. **Nouveau cycle :** Nouveau 90° (Bande Large), puis 180° (Bande Large) et Recueil du signal.",
            
            "1. **Cycle Start:** Non-selective 90° (Broad Band). No signal recording during first TR.\n"
            "2. **End of first TR:** Use of a Narrow SAT band selective on the Fat peak.\n"
            "3. **New Cycle:** New 90° (Broad Band), then 180° (Broad Band) and Signal recording.",
            
            "1. **Zyklusbeginn:** Nicht-selektiver 90°-Puls (Breitband). Keine Signalaufzeichnung während der ersten TR.\n"
            "2. **Ende der ersten TR:** Verwendung eines schmalen SAT-Bandes, das selektiv auf den Fett-Peak wirkt.\n"
            "3. **Neuer Zyklus:** Neuer 90°-Puls (Breitband), dann 180° (Breitband) und Signalaufzeichnung."
        ))

        fig_time = Figure(figsize=(10, 6)); ax_time = fig_time.subplots()
        
        TR = 800
        t_sat_delay = 100
        t_sat = TR - t_sat_delay 
        t_exc = TR 
        t_end_plot = TR + 350
        time = np.linspace(0, t_end_plot, 1000)
        
        T1_water = 600
        T1_fat = 150
        T_decay = 100 
        
        mz_water = np.zeros_like(time)
        mz_fat = np.zeros_like(time)
        
        for i, t in enumerate(time):
            if t < t_sat:
                mz_water[i] = 1 - np.exp(-t / T1_water)
                mz_fat[i] = 1 - np.exp(-t / T1_fat)
            elif t >= t_sat and t < t_exc:
                mz_water[i] = 1 - np.exp(-t / T1_water)
                dt = t - t_sat
                mz_fat[i] = 0 + (1 - np.exp(-dt / T1_fat)) * 0.15 
            elif t >= t_exc:
                val_water_start = 1 - np.exp(-t_exc / T1_water)
                val_fat_start = (1 - np.exp(-(t_exc - t_sat) / T1_fat)) * 0.15
                dt_decay = t - t_exc
                mz_water[i] = val_water_start * np.exp(-dt_decay / T_decay)
                mz_fat[i] = val_fat_start * np.exp(-dt_decay / T_decay)

        ax_time.plot(time, mz_water, color='#3498db', lw=3, label=T("Eau", "Water", "Wasser"))
        ax_time.plot(time, mz_fat, color='#e67e22', lw=3, linestyle='--', label=T("Graisse", "Fat", "Fett"))
        
        val_fat_at_sat = 1 - np.exp(-t_sat / T1_fat)
        ax_time.plot([t_sat, t_sat], [val_fat_at_sat, 0], color='#e67e22', lw=2, linestyle=':')

        y_pulse = -0.3
        h_pulse = 0.2
        
        ax_time.add_patch(patches.Rectangle((-20, y_pulse), 40, h_pulse, facecolor='#e74c3c', edgecolor='red'))
        ax_time.text(0, y_pulse + h_pulse + 0.05, "90°", ha='center', color='red', fontweight='bold')
        
        ax_time.add_patch(patches.Rectangle((t_sat-20, y_pulse), 40, h_pulse, facecolor='#2ecc71', edgecolor='green'))
        ax_time.text(t_sat, y_pulse + h_pulse + 0.05, "SAT", ha='center', color='green', fontweight='bold')
        ax_time.text(t_sat, y_pulse - 0.15, T("BP Étroite", "Narrow BW", "Schmale BB"), ha='center', color='green', fontsize=8)
        
        ax_time.add_patch(patches.Rectangle((t_exc-20, y_pulse), 40, h_pulse, facecolor='#e74c3c', edgecolor='red'))
        ax_time.text(t_exc, y_pulse + h_pulse + 0.05, "90°", ha='center', color='red', fontweight='bold')
        
        t_180 = t_exc + 120
        ax_time.add_patch(patches.Rectangle((t_180-20, y_pulse), 40, h_pulse*1.2, facecolor='#e74c3c', edgecolor='red', alpha=0.6))
        ax_time.text(t_180, y_pulse + h_pulse*1.2 + 0.05, "180°", ha='center', color='#c0392b', fontsize=9, fontweight='bold')
        ax_time.text(t_180, y_pulse - 0.15, T("BP Large", "Broad BW", "Breite BB"), ha='center', color='red', fontsize=8)
        
        t_echo = t_180 + 120
        ts = np.linspace(t_echo-30, t_echo+30, 100)
        wave = np.exp(-0.005*(ts-t_echo)**2) * np.cos(0.3*(ts-t_echo)) * 0.3
        ax_time.plot(ts, wave + y_pulse + 0.1, color='black')
        ax_time.text(t_echo, y_pulse + 0.45, "ECHO", ha='center', fontweight='bold')

        y_tr_line = -0.6
        ax_time.annotate('', xy=(0, y_tr_line), xytext=(t_exc, y_tr_line), arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax_time.text(t_exc/2, y_tr_line - 0.1, T("TR 1", "TR 1", "TR 1"), ha='center', fontweight='bold')
        ax_time.annotate('', xy=(t_exc, y_tr_line), xytext=(t_end_plot, y_tr_line), arrowprops=dict(arrowstyle='->', color='black', lw=1.5)) 
        ax_time.text(t_exc + 150, y_tr_line - 0.1, T("TR 2", "TR 2", "TR 2"), ha='center', fontweight='bold')
        ax_time.axvline(TR, color='black', linestyle='--', alpha=0.4, ymin=0, ymax=1)

        ax_time.set_ylim(-0.8, 1.3)
        ax_time.set_xlim(-50, t_end_plot)
        ax_time.set_yticks([0, 1])
        ax_time.set_ylabel("Signal / Mz")
        ax_time.spines['top'].set_visible(False)
        ax_time.spines['right'].set_visible(False)
        ax_time.spines['bottom'].set_visible(False)
        ax_time.get_xaxis().set_ticks([])
        ax_time.legend(loc='upper left', bbox_to_anchor=(0, 1.05))
        
        st.pyplot(fig_time)
        

        st.divider()

        st.markdown("#### " + T("B. Sélectivité Spectrale & Inhomogénéité", "B. Spectral Selectivity & Inhomogeneity", "B. Spektrale Selektivität & Inhomogenität"))

        col_ctrl, col_spec = st.columns([1, 2])
        
        with col_ctrl:
            st.write(T(
                "La Fat Sat est calibrée pour taper exactement à la fréquence de la graisse. Si le champ magnétique ($B_0$) est hétérogène, les pics se décalent vers la droite.",
                "Fat Sat is calibrated to hit exactly the fat frequency. If the magnetic field ($B_0$) is inhomogeneous, peaks shift to the right.",
                "Fat Sat ist genau auf die Fettfrequenz kalibriert. Wenn das Magnetfeld ($B_0$) inhomogen ist, verschieben sich die Peaks nach rechts."
            ))
            st.write("")
            
            is_inhomogeneous = st.toggle(
                T("⚠️ Simuler Inhomogénéité B0", "⚠️ Simulate B0 Inhomogeneity", "⚠️ B0-Inhomogenität simulieren"),
                value=False,
                key="fs_b0_toggle_simple"
            )
            
            if is_inhomogeneous:
                b0_shift = 80 
                st.error(T(
                    "❌ **ÉCHEC FAT SAT**\nL'impulsion BP étroite (verte) tire dans le vide. La graisse reste en hyper signal.",
                    "❌ **FAT SAT FAILURE**\nThe narrow BW pulse (green) misses the target. Fat remains hyperintense.",
                    "❌ **FAT SAT FEHLGESCHLAGEN**\nDer schmalbandige Puls (grün) verfehlt das Ziel. Das Fett bleibt im Hypersignal."
                ))
            else:
                b0_shift = 0
                st.success(T(
                    "✅ **SUCCÈS**\nChamp homogène. L'impulsion sature parfaitement la graisse.",
                    "✅ **SUCCESS**\nHomogeneous field. Pulse perfectly saturates fat.",
                    "✅ **ERFOLG**\nHomogenes Feld. Der Puls sättigt das Fett perfekt."
                ))

        with col_spec:
            fig_s = Figure(figsize=(7, 4)); ax_s = fig_s.subplots()
            
            freqs = np.linspace(-500, 200, 500)
            
            center_water = 0 + b0_shift
            center_fat = -220 + b0_shift
            
            sigma_water = 20
            sigma_fat = 25 
            
            peak_water = np.exp(-0.5 * ((freqs - center_water) / sigma_water)**2)
            peak_fat = np.exp(-0.5 * ((freqs - center_fat) / sigma_fat)**2)
            
            ax_s.fill_between(freqs, peak_water, color='#3498db', alpha=0.6, label=T('Eau', 'Water', 'Wasser'))
            ax_s.fill_between(freqs, peak_fat, color='#e67e22', alpha=0.6, label=T('Graisse', 'Fat', 'Fett'))
            
            fixed_bw = 120 
            fixed_center = -220
            
            ax_s.axvspan(fixed_center - fixed_bw/2, fixed_center + fixed_bw/2, 
                         color='#2ecc71', alpha=0.5, label=T('Impulsion SAT (Fixe)', 'SAT Pulse (Fixed)', 'SAT-Puls (Fest)'))
            
            ax_s.text(fixed_center, 1.05, T("BP Étroite", "Narrow BW", "Schmale BB"), color='green', ha='center', fontweight='bold')
            
            mid_point = (center_water + center_fat) / 2
            ax_s.annotate('', xy=(center_water, 0.5), xytext=(center_fat, 0.5), 
                          arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
            ax_s.text(mid_point, 0.55, "3.5 ppm", ha='center', fontweight='bold')
            
            if b0_shift != 0:
                ax_s.annotate(T("Dérive B0", "B0 Drift", "B0-Drift"), 
                              xy=(center_water, 0.2), xytext=(0, 0.2),
                              arrowprops=dict(arrowstyle='->', color='red', lw=2), color='red', ha='center', fontsize=9, fontweight='bold')

            ax_s.set_xlim(-500, 200)
            ax_s.set_ylim(0, 1.2)
            ax_s.set_yticks([])
            ax_s.set_xlabel("Fréquence (Hz)")
            ax_s.legend(loc='upper left', fontsize='small')
            ax_s.grid(True, alpha=0.1)
            
            st.pyplot(fig_s)
            

    # --- 2. SPAIR ---
    with fs_tabs[1]:
        st.subheader(T("2. SPAIR (Spectral Adiabatic Inversion Recovery)", "2. SPAIR (Spectral Adiabatic Inversion Recovery)", "2. SPAIR (Spectral Adiabatic Inversion Recovery)"))
        
        st.markdown("#### " + T("A. Séquence Temporelle (TI & Décroissance)", "A. Timing Sequence (TI & Decay)", "A. Sequenzdiagramm (TI & Zerfall)"))
        
        st.info(T(
            "1. **Impulsion Adiabatique (Verte) :** Inverse la Graisse (-Mz) mais laisse les autres tissus intacts (+Mz).\n"
            "2. **Attente (TI) :** La Graisse remonte vers 0.\n"
            "3. **Excitation (90°) :** À la fin du TI (ligne pointillée), on bascule. On observe alors la **décroissance exponentielle** du signal (Relaxation T2) des tissus A et B. La Graisse, elle, est éteinte.",
            
            "1. **Adiabatic Pulse (Green):** Inverts Fat (-Mz) but leaves other tissues intact (+Mz).\n"
            "2. **Wait (TI):** Fat recovers towards 0.\n"
            "3. **Excitation (90°):** At the end of TI (dotted line), flip occurs. We observe **exponential decay** (T2 Relaxation) of tissues A and B. Fat is nulled.",
            
            "1. **Adiabatischer Puls (Grün):** Invertiert das Fett (-Mz), lässt aber andere Gewebe intakt (+Mz).\n"
            "2. **Wartezeit (TI):** Fett erholt sich in Richtung 0.\n"
            "3. **Anregung (90°):** Am Ende der TI (gepunktete Linie) erfolgt der Flip. Wir beobachten den **exponentiellen Zerfall** (T2-Relaxation) der Gewebe A und B. Fett ist ausgelöscht."
        ))

        fig_spair = Figure(figsize=(10, 6.5)); ax_spair = fig_spair.subplots() 
        
        TI = 180  
        TE = 60
        t_exc = TI
        t_echo = TI + TE
        t_total = t_echo + 100
        time = np.linspace(0, t_total, 1000)
        
        T1_Fat = 260   
        T2_Tissue_A = 100 
        T2_Tissue_B = 40  
        
        mz_tissue_A = np.zeros_like(time)
        mz_tissue_B = np.zeros_like(time)
        mz_fat = np.zeros_like(time)
        
        for i, t in enumerate(time):
            if t < t_exc:
                mz_tissue_A[i] = 1.0 
                mz_tissue_B[i] = 1.0 
            else:
                dt = t - t_exc
                mz_tissue_A[i] = 1.0 * np.exp(-dt / T2_Tissue_A)
                mz_tissue_B[i] = 1.0 * np.exp(-dt / T2_Tissue_B)
                
            if t < t_exc:
                mz_fat[i] = 1 - 2 * np.exp(-t / T1_Fat)
            else:
                mz_fat[i] = 0
        
        ax_spair.plot(time, mz_tissue_A, color='#3498db', lw=3, label=T("Tissu A (Décroissance lente)", "Tissue A (Slow decay)", "Gewebe A (Langsamer Zerfall)"))
        ax_spair.plot(time, mz_tissue_B, color='#5dade2', lw=2, linestyle=':', label=T("Tissu B (Décroissance rapide)", "Tissue B (Fast decay)", "Gewebe B (Schneller Zerfall)"))
        ax_spair.plot(time, mz_fat, color='#e67e22', lw=3, label=T("Graisse (Inversée)", "Fat (Inverted)", "Fett (Invertiert)"))
        
        ax_spair.axhline(0, color='black', lw=1)

        ax_spair.axvline(TI, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax_spair.text(TI, 1.05, "TI", ha='center', fontweight='bold', fontsize=10)

        y_pulse = -1.3
        h_pulse = 0.3
        
        ax_spair.add_patch(patches.Rectangle((-15, y_pulse), 30, h_pulse, facecolor='#2ecc71', edgecolor='green'))
        ax_spair.text(0, y_pulse + h_pulse + 0.1, "180° Adiabatic", ha='center', color='green', fontweight='bold', fontsize=9)
        ax_spair.text(0, y_pulse - 0.2, T("BP Étroite", "Narrow BW", "Schmale BB"), ha='center', color='green', fontsize=8)
        
        x_90_start = t_exc - 10
        x_90_end = t_exc + 10
        ax_spair.add_patch(patches.Rectangle((x_90_start, y_pulse), 20, h_pulse, facecolor='#e74c3c', edgecolor='red'))
        ax_spair.text(t_exc, y_pulse + h_pulse + 0.1, "90°", ha='center', color='red', fontweight='bold')
        
        t_180 = t_exc + TE/2
        x_180_start = t_180 - 10
        x_180_end = t_180 + 10
        ax_spair.add_patch(patches.Rectangle((x_180_start, y_pulse), 20, h_pulse*1.2, facecolor='#e74c3c', edgecolor='red', alpha=0.6))
        ax_spair.text(t_180, y_pulse + h_pulse*1.2 + 0.1, "180°", ha='center', color='#c0392b', fontsize=8)

        y_bracket_top = y_pulse
        y_bracket_bottom = y_pulse - 0.15
        
        ax_spair.plot([x_90_start, x_90_start, x_180_end, x_180_end], 
                      [y_bracket_top, y_bracket_bottom, y_bracket_bottom, y_bracket_top], 
                      color='#e74c3c', lw=1.5)
        
        mid_bracket = (x_90_start + x_180_end) / 2
        ax_spair.text(mid_bracket, y_bracket_bottom - 0.1, T("Bande Large", "Broad BW", "Breite BB"), 
                      ha='center', va='top', color='#c0392b', fontweight='bold', fontsize=9)

        ts = np.linspace(t_echo-15, t_echo+15, 100)
        wave = np.exp(-0.01*(ts-t_echo)**2) * np.cos(0.4*(ts-t_echo)) * 0.4
        ax_spair.plot(ts, wave + y_pulse + 0.15, color='black')
        ax_spair.text(t_echo, y_pulse + 0.5, "ECHO", ha='center', fontweight='bold')
        
        val_A_at_echo = mz_tissue_A[np.argmin(np.abs(time - t_echo))]
        ax_spair.plot([t_echo, t_echo], [val_A_at_echo, y_pulse + 0.5], color='#3498db', linestyle=':', alpha=0.5)

        ax_spair.annotate('', xy=(0, -1), xytext=(0, 1), arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2, ls='--'))
        ax_spair.annotate('', xy=(0, 0.1), xytext=(t_exc, 0.1), arrowprops=dict(arrowstyle='<->', color='black'))
        
        y_tr = -1.9
        ax_spair.annotate('', xy=(0, y_tr), xytext=(t_total, y_tr), 
                          arrowprops=dict(arrowstyle='<->', color='black', lw=2))
        ax_spair.text(t_total/2, y_tr - 0.2, "TR (Time of Repetition)", ha='center', fontweight='bold')
        
        ax_spair.plot([0, 0], [y_pulse, y_tr], color='black', lw=1)
        ax_spair.plot([t_total, t_total], [y_pulse, y_tr], color='black', lw=1, linestyle=':')

        ax_spair.set_ylim(-2.2, 1.2)
        ax_spair.set_xlim(-40, t_total + 20)
        ax_spair.set_yticks([-1, 0, 1])
        ax_spair.set_ylabel("Signal")
        
        ax_spair.spines['top'].set_visible(False)
        ax_spair.spines['right'].set_visible(False)
        ax_spair.spines['bottom'].set_visible(False)
        ax_spair.get_xaxis().set_ticks([])
        ax_spair.legend(loc='upper right')
        
        st.pyplot(fig_spair)
        

        st.divider()

        st.markdown("#### " + T("B. Spectre : Robustesse au champ B0", "B. Spectrum: B0 Field Robustness", "B. Spektrum: B0-Feld-Robustheit"))

        c_spair_ctrl, c_spair_plot = st.columns([1, 2])
        
        with c_spair_ctrl:
            step_labels = [
                T("1. Flanc Gauche", "1. Left Slope", "1. Linke Flanke"), 
                T("2. Centre du Pic", "2. Peak Center", "2. Peak-Zentrum"), 
                T("3. Flanc Droit", "3. Right Slope", "3. Rechte Flanke")
            ]
            sweep_pos = st.select_slider(
                T("1. Position Impulsion", "1. Pulse Position", "1. Pulsposition"),
                options=step_labels,
                value=step_labels[0],
                key="spair_step_final_restored"
            )
            
            st.write("---")
            
            b0_shift = st.slider(
                T("2. Inhomogénéité B0 (Hz)", "2. B0 Inhomogeneity (Hz)", "2. B0-Inhomogenität (Hz)"),
                min_value=-40, max_value=40, value=0, step=5, 
                key="spair_b0_shift_final_restored"
            )
            
            if b0_shift != 0:
                st.warning(T(f"⚠️ Pics décalés de {b0_shift} Hz.", f"⚠️ Peaks shifted by {b0_shift} Hz.", f"⚠️ Peaks um {b0_shift} Hz verschoben."))
            else:
                st.success(T("✅ Champs homogène.", "✅ Homogeneous field.", "✅ Homogenes Feld."))

        with c_spair_plot:
            fig_sp = Figure(figsize=(8, 4)); ax_sp = fig_sp.subplots()
            
            freqs = np.linspace(-600, 200, 500)
            center_water_real = 0 + b0_shift
            center_fat_real = -220 + b0_shift
            
            water_peak = np.exp(-0.5 * ((freqs - center_water_real) / 20)**2) 
            fat_peak = np.exp(-0.5 * ((freqs - center_fat_real) / 35)**2) 
            
            pulse_width = 87.5 
            shift_step = pulse_width 
            
            center_fat_theo = -220.0 
            
            if sweep_pos == step_labels[0]:
                current_center = center_fat_theo - shift_step
                label_txt = "Zone 1"
            elif sweep_pos == step_labels[1]:
                current_center = center_fat_theo
                label_txt = "Zone 2"
            else: 
                current_center = center_fat_theo + shift_step
                label_txt = "Zone 3"
            
            x_start = current_center - (pulse_width / 2)
            x_end = current_center + (pulse_width / 2)

            ax_sp.fill_between(freqs, water_peak, color='#3498db', alpha=0.6, label=T('Eau', 'Water', 'Wasser'))
            ax_sp.fill_between(freqs, fat_peak, color='#e67e22', alpha=0.6, label=T('Graisse', 'Fat', 'Fett'))
            
            ax_sp.axvspan(x_start, x_end, color='#2ecc71', alpha=0.5, label=T('Impulsion (Fixe)', 'Pulse (Fixed)', 'Puls (Fest)'))
            
            ax_sp.arrow(current_center, 1.05, 0, -0.1, head_width=10, head_length=0.05, fc='green', ec='green')
            ax_sp.text(current_center, 1.1, label_txt, ha='center', color='green', fontsize=9, fontweight='bold')

            x_water_arrow = 0 + b0_shift
            x_fat_arrow = -220 + b0_shift
            x_text = -110 + b0_shift
            
            ax_sp.annotate('', xy=(x_water_arrow, 0.5), xytext=(x_fat_arrow, 0.5), 
                           arrowprops=dict(arrowstyle='<->', lw=1.5, color='black'))
            ax_sp.text(x_text, 0.55, r"$\delta = 3.5$ ppm", ha='center', fontweight='bold')

            if b0_shift != 0:
                ax_sp.annotate(T("Dérive", "Drift", "Drift"), 
                               xy=(center_fat_real, 0.3), xytext=(center_fat_theo, 0.3),
                               arrowprops=dict(arrowstyle='->', color='red', lw=2), color='red', ha='center', fontsize=8)

            ax_sp.set_xlim(-600, 200)
            ax_sp.set_ylim(0, 1.35)
            ax_sp.set_xlabel("Fréquence (Hz)")
            ax_sp.set_yticks([]) 
            
            ax_sp.legend(loc='upper left', fontsize='small')
            ax_sp.grid(True, alpha=0.2)
            
            st.pyplot(fig_sp)
            

    # --- 3. DIXON ---
    with fs_tabs[2]:
        st.subheader(T("3. Séquence Dixon (Chemical Shift Imaging)", "3. Dixon Sequence (Chemical Shift Imaging)", "3. Dixon-Sequenz (Chemical Shift Imaging)"))
        st.markdown("#### " + T("📡 A. L'Acquisition (2 Échos)", "📡 A. Acquisition (2 Echoes)", "📡 A. Die Akquisition (2 Echos)"))
        
        c_dx1, c_dx2 = st.columns([1.2, 2])
        
        with c_dx1:
            te_dixon = st.select_slider(
                T("Choisir le Temps d'Echo (TE)", "Select Echo Time (TE)", "Echozeit wählen (TE)"), 
                options=[2.2, 4.5], 
                key="dx_te_final_S_notation"
            )
            
            st.divider()
            
            lbl_oop = T("📉 Opposition (S_Out)", "📉 Out of Phase (S_Out)", "📉 Gegenphase (S_Out)")
            form_oop = T(r"S_{Out} = E - G", r"S_{Out} = W - F", r"S_{Out} = W - F")
            
            if te_dixon == 2.2:
                with st.container():
                    st.error(f"**{lbl_oop}**")
                    st.latex(form_oop)
                    st.caption(T("Eau et Graisse s'opposent.", "Water and Fat oppose each other.", "Wasser und Fett heben sich auf."))
            else:
                st.markdown(f"**{lbl_oop}**") 
                st.latex(form_oop)

            st.write("") 

            lbl_ip = T("📈 Phase (S_In)", "📈 In Phase (S_In)", "📈 In Phase (S_In)")
            form_ip = T(r"S_{In} = E + G", r"S_{In} = W + F", r"S_{In} = W + F")
            
            if te_dixon == 4.5:
                with st.container():
                    st.success(f"**{lbl_ip}**")
                    st.latex(form_ip)
                    st.caption(T("Eau et Graisse s'additionnent.", "Water and Fat sum up.", "Wasser und Fett addieren sich."))
            else:
                st.markdown(f"**{lbl_ip}**")
                st.latex(form_ip)

        with c_dx2:
            fig_dx = Figure(figsize=(8, 4)); ax_dx = fig_dx.subplots()
            t_ms = np.linspace(0, 10, 500)
            
            ax_dx.plot(t_ms, np.ones_like(t_ms), color='#3498db', label=T('Eau', 'Water', 'Wasser'))
            ax_dx.plot(t_ms, np.cos(2 * np.pi * 220 * t_ms / 1000.0), color='#e67e22', label=T('Graisse', 'Fat', 'Fett'))
            
            ax_dx.plot(te_dixon, np.cos(2 * np.pi * 220 * te_dixon / 1000.0), 'ro', markersize=12, label=T('Acquisition', 'Acquisition', 'Akquisition'))
            
            ax_dx.axvline(te_dixon, color='gray', linestyle='--')
            ax_dx.set_xlabel("TE (ms)")
            ax_dx.set_yticks([-1, 0, 1])
            ax_dx.set_yticklabels([T("Out", "Out", "Out"), "Quad", T("In", "In", "In")])
            ax_dx.legend(loc='upper right'); ax_dx.grid(True, alpha=0.3)
            st.pyplot(fig_dx); 
            
        st.divider()
        st.markdown("#### " + T("🧮 B. Le Calcul", "🧮 B. Calculation", "🧮 B. Die Berechnung"))
        c_calc1, c_calc2 = st.columns(2)
        
        with c_calc1: 
            st.markdown(f"##### {T('💧 Image EAU', '💧 WATER Image', '💧 WASSER-Bild')}")
            st.latex(r"W = \frac{S_{In} + S_{Out}}{2}")
            
        with c_calc2: 
            st.markdown(f"##### {T('🧈 Image GRAISSE', '🧈 FAT Image', '🧈 FETT-Bild')}")
            st.latex(r"F = \frac{S_{In} - S_{Out}}{2}")

    # --- 4. EXCITATION EAU ---
    with fs_tabs[3]:
        import pandas as pd 
        st.subheader(T("4. Excitation de l'Eau (Water Excitation / WE)", "4. Water Excitation (WE)", "4. Wasseranregung (WE)"))
        st.markdown("#### " + T("🌊 Principe : Sélection sans Saturation", "🌊 Principle: Selection without Saturation", "🌊 Prinzip: Selektion ohne Sättigung"))
        
        c_we_txt, c_we_acro = st.columns([2, 1])
        with c_we_txt:
            txt_diff = T("""
            **Différence avec la Fat-Sat :**
            * **Fat-Sat :** Excite la graisse puis la tue (Gradient de déphasage).
            * **WE (Water Excitation) :** N'utilise **pas de gradient de déphasage**. Elle stimule sélectivement l'eau en laissant la graisse tranquille.
            """, """
            **Difference with Fat-Sat:**
            * **Fat-Sat:** Excites fat then kills it (Dephasing gradient).
            * **WE (Water Excitation):** Does **not use a dephasing gradient**. It selectively stimulates water while leaving fat alone.
            """, """
            **Unterschied zu Fat-Sat:**
            * **Fat-Sat:** Regt Fett an und tötet es dann ab (Dephasierungsgradient).
            * **WE (Water Excitation):** Verwendet **keinen Dephasierungsgradienten**. Sie stimuliert selektiv das Wasser und lässt das Fett in Ruhe.
            """)
            st.info(txt_diff)
            
            txt_pulse = T("""
            **La Séquence Binomiale (1-1) :**
            1. **Pulse 45° :** Tout le monde bascule.
            2. **Délai :** On attend l'opposition de phase (180°).
            3. **Pulse 45° :** L'Eau s'additionne (90°), la Graisse se soustrait (0°).
            """, """
            **The Binomial Sequence (1-1):**
            1. **Pulse 45°:** Everyone flips.
            2. **Delay:** Wait for phase opposition (180°).
            3. **Pulse 45°:** Water sums up (90°), Fat subtracts (0°).
            """, """
            **Die Binomialsequenz (1-1):**
            1. **45°-Puls:** Alles kippt.
            2. **Wartezeit:** Warten auf Phasenopposition (180°).
            3. **45°-Puls:** Wasser addiert sich auf (90°), Fett subtrahiert sich (0°).
            """)
            st.markdown(txt_pulse)

        with c_we_acro:
            st.markdown(f"#### {T('🏷️ Noms Commerciaux', '🏷️ Commercial Names', '🏷️ Handelsnamen')}")
            col_brand = T("Constructeur", "Manufacturer", "Hersteller")
            col_acro = T("Acronyme", "Acronym", "Akronym")
            
            df_names = pd.DataFrame({
                col_brand: ["Siemens / Fuji", "GE", "Philips", "Canon"], 
                col_acro: ["WE", "SSRF", "ProSET", "WET / PASTA"]
            })
            st.table(df_names.set_index(col_brand))
            
        st.divider()
        
        st.markdown("#### " + T("🕹️ Visualisation Dynamique (Impulsion 1-1)", "🕹️ Dynamic Visualization (1-1 Pulse)", "🕹️ Dynamische Visualisierung (1-1 Puls)"))
        
        opt_step1 = T("1. Équilibre (M0)", "1. Equilibrium (M0)", "1. Gleichgewicht (M0)")
        opt_step2 = T("2. Premier Pulse (45°)", "2. First Pulse (45°)", "2. Erster Puls (45°)")
        opt_step3 = T("3. Délai (Opposition 180°)", "3. Delay (Opposition 180°)", "3. Wartezeit (Gegenphase 180°)")
        opt_step4 = T("4. Second Pulse (45°)", "4. Second Pulse (45°)", "4. Zweiter Puls (45°)")
        
        step = st.select_slider(T("Étapes", "Steps", "Schritte"), options=[opt_step1, opt_step2, opt_step3, opt_step4], value=opt_step1)
        
        w_vec = np.array([0.0, 0.0, 1.0])
        f_vec = np.array([0.0, 0.0, 1.0])
        desc = ""
        
        if step == opt_step1:
            desc = T("Aimantation longitudinale (z).", "Longitudinal Magnetization (z).", "Longitudinale Magnetisierung (z).")
        elif step == opt_step2:
            val = np.sin(np.pi/4)
            w_vec = np.array([0.0, val, val])
            f_vec = np.array([0.0, val, val])
            desc = T("Pulse 45°. Tout bascule.", "Pulse 45°. Everything flips.", "45°-Puls. Alles kippt.")
        elif step == opt_step3:
            val = np.sin(np.pi/4)
            w_vec = np.array([0.0, val, val])
            f_vec = np.array([0.0, -val, val])
            desc = T("Délai : Opposition de phase.", "Delay: Phase Opposition.", "Wartezeit: Phasenopposition.")
        elif step == opt_step4:
            w_vec = np.array([0.0, 1.0, 0.0])
            f_vec = np.array([0.0, 0.0, 1.0])
            desc = T("Pulse 45°. Eau à 90°, Graisse à 0°.", "Pulse 45°. Water at 90°, Fat at 0°.", "45°-Puls. Wasser auf 90°, Fett auf 0°.")
        
        c_visu1, c_visu2 = st.columns([1, 2])
        with c_visu1: 
            st.info(f"**{T('État', 'State', 'Zustand')} :** {desc}")
            
        with c_visu2:
            fig = Figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot([0, 0], [0, 0], [-0.2, 1.2], 'k--', linewidth=1)
            ax.quiver(0, 0, 0, w_vec[0], w_vec[1], w_vec[2], color='#3498db', linewidth=4, arrow_length_ratio=0.1, label=T('Eau', 'Water', 'Wasser'))
            offset = 0.05 if step in [opt_step1, opt_step2] else 0.0
            ax.quiver(offset, 0, 0, f_vec[0], f_vec[1], f_vec[2], color='#e67e22', linewidth=3, arrow_length_ratio=0.1, label=T('Graisse', 'Fat', 'Fett'))
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, 1.2)
            ax.view_init(elev=20, azim=20); ax.legend()
            st.pyplot(fig); 

    # --- 5. SOUSTRACTION ---
    with fs_tabs[4]:
        st.subheader(T("5. Soustraction (Post - Pré)", "5. Subtraction (Post - Pre)", "5. Subtraktion (Post - Prä)"))
        c_sub1, c_sub2 = st.columns([1, 2])
        with c_sub1:
            move_x = st.slider(T("Mouvement Patient (px)", "Patient Motion (px)", "Patientenbewegung (px)"), -10, 10, 0, 1, key="sub_move_clean")
            st.info(T("Le moindre mouvement crée des artefacts.", "The slightest movement creates artifacts.", "Die kleinste Bewegung erzeugt Artefakte."))
        with c_sub2:
            size = 100
            y, x = np.ogrid[:size, :size]
            center = size // 2
            mask_body = np.sqrt((x - center)**2 + (y - center)**2) < 30
            img_pre = np.zeros((size, size))
            img_pre[mask_body] = 0.5
            
            mask_body_mv = np.sqrt((x - (center+move_x))**2 + (y - center)**2) < 30
            mask_lesion = np.sqrt((x - (center+move_x) - 10)**2 + (y - center - 10)**2) < 5
            img_post = np.zeros((size, size))
            img_post[mask_body_mv] = 0.5
            img_post[mask_lesion] = 1.0
            
            c1, c2, c3 = st.columns(3)
            c1.image(img_pre, caption=T("Pré", "Pre", "Prä"), clamp=True)
            c2.image(img_post, caption="Post", clamp=True)
            c3.image(np.clip(img_post - img_pre, 0, 1), caption="Sub", clamp=True)

    # --- 6. STIR ---
    with fs_tabs[5]:
        st.subheader(T("6. STIR (Short Tau Inversion Recovery)", "6. STIR (Short Tau Inversion Recovery)", "6. STIR (Short Tau Inversion Recovery)"))
        st.markdown("#### " + T("📡 1. Pourquoi \"Non-Sélectif\" ? (Bande Large)", "📡 1. Why \"Non-Selective\"? (Broadband)", "📡 1. Warum \"Nicht-Selektiv\"? (Breitband)"))
        
        col_ex1, col_ex2 = st.columns([2, 1])
        with col_ex1: 
            st.info(T("Le STIR utilise une impulsion courte qui tape **tout le spectre** (Eau, Graisse, Gado...).", 
                      "STIR uses a short pulse that hits **the entire spectrum** (Water, Fat, Gado...).",
                      "STIR verwendet einen kurzen Puls, der **das gesamte Spektrum** trifft (Wasser, Fett, Gado...)."))
        with col_ex2:
            fig_bw, ax_bw = Figure(figsize=(4, 2.5))
            ax_bw.fill_between(np.linspace(-500, 500, 100), 0, 1, color='purple', alpha=0.4)
            ax_bw.text(0, 0.5, T("Bande Large", "Broadband", "Breitband"), ha='center', color='purple')
            ax_bw.set_yticks([]); ax_bw.set_xlim(-500, 500)
            st.pyplot(fig_bw); 
            
        st.divider()
        st.markdown("#### " + T("📉 3. Visualisation (Signal en Module)", "📉 3. Visualization (Magnitude Signal)", "📉 3. Visualisierung (Betragssignal)"))
        
        c_st1, c_st2 = st.columns([1, 2])
        with c_st1:
            ti_stir = st.slider(T("Choisir le moment du 'CLIC' (TI)", "Select timing 'CLICK' (TI)", "Zeitpunkt für 'KLICK' wählen (TI)"), 50, 800, 170, 10, key="st_ti_clean")
            mz_fat = 1 - 2 * np.exp(-ti_stir/260.0)
            mz_gado = 1 - 2 * np.exp(-ti_stir/280.0)
            
            st.metric(T("Signal Graisse (Module)", "Fat Signal (Magnitude)", "Fettsignal (Betrag)"), f"{abs(mz_fat):.2f}")
            
            if abs(mz_fat) < 0.1: 
                st.success(T("✅ **GRAISSE NOIRE**", "✅ **BLACK FAT**", "✅ **SCHWARZES FETT**"))
            else: 
                st.warning(T("Graisse visible", "Fat visible", "Fett sichtbar"))
                
            if abs(mz_gado) < 0.2: 
                st.error(T("🚨 **GADO ANNULÉ**", "🚨 **GADO NULLIFIED**", "🚨 **GADO ANNULLIERT**"))

        with c_st2:
            fig_st = Figure(figsize=(8, 4)); ax_st, ax_bar = fig_st.subplots(1, 2, gridspec_kw={'width_ratios': [30, 1]})
            t_rng = np.linspace(0, 5000, 500)
            
            tissues = {
                T('Graisse (260ms)', 'Fat (260ms)', 'Fett (260ms)'): (260, '#ff7f0e'), 
                T('Gado (280ms)', 'Gado (280ms)', 'Gado (280ms)'): (280, 'red'), 
                T('SB (790ms)', 'WM (790ms)', 'WS (790ms)'): (790, '#bdc3c7'), 
                T('LCR (4000ms)', 'CSF (4000ms)', 'Liquor (4000ms)'): (4000, 'cyan')
            }
            
            for name, (t1_val, col) in tissues.items():
                ax_st.plot(t_rng, 1 - 2 * np.exp(-t_rng / t1_val), label=name, color=col)
                
            ax_st.axhline(0, color='black')
            ax_st.axvline(ti_stir, color='green', linewidth=2, label=f'TI ({ti_stir}ms)')
            ax_st.set_xlim(0, 5000); ax_st.set_ylim(-1.1, 1.1)
            ax_st.legend(loc='lower right', fontsize=8); ax_st.grid(True, alpha=0.3)
            
            y_grad = np.abs(np.linspace(1.1, -1.1, 200)).reshape(-1, 1)
            ax_bar.imshow(y_grad, aspect='auto', cmap='gray', vmin=0, vmax=1, extent=[0, 1, -1.1, 1.1])
            ax_bar.set_xticks([]); ax_bar.set_yticks([])
            ax_bar.set_title("Module", fontsize=8)
            ax_bar.plot(0.5, 1 - 2 * np.exp(-ti_stir/260.0), 'o', color='orange', markeredgecolor='white')
            
            st.pyplot(fig_st); 

    # --- 7. PSIR ---
    with fs_tabs[6]:
        st.subheader(T("7. PSIR : Robustesse vs TI Scout", "7. PSIR: Robustness vs TI Scout", "7. PSIR: Robustheit vs. TI Scout"))
        
        st.info(T(
            "**Démonstration de l'Insensibilité au TI :**\n"
            "👉 En imagerie classique (**Magnitude**), il faut un **TI Scout** précis pour annuler le myocarde (Le rendre noir). Si le TI est mal réglé (ex: Myocarde = -0.25, Fibrose = +0.25), les deux apparaissent gris identiques ($| -0.25 | = | +0.25 |$).\n"
            "👉 En **PSIR**, on garde le signe. -0.25 reste Noir, +0.25 reste Blanc. Le contraste est préservé sans réglage parfait.",
            
            "**Demonstrating TI Insensitivity:**\n"
            "👉 In classic (**Magnitude**) imaging, a precise **TI Scout** is needed to null myocardium. If TI is off (e.g. Myo = -0.25, Fib = +0.25), both appear identical grey ($| -0.25 | = | +0.25 |$).\n"
            "👉 In **PSIR**, sign is kept. -0.25 stays Black, +0.25 stays White. Contrast is preserved without perfect settings.",
            
            "**Demonstration der TI-Unempfindlichkeit:**\n"
            "👉 In der klassischen Bildgebung (**Betrag**) ist ein präziser **TI Scout** erforderlich, um das Myokard auszulöschen (schwarz zu machen). Wenn die TI falsch eingestellt ist (z.B. Myo = -0.25, Fibrose = +0.25), erscheinen beide identisch grau ($| -0.25 | = | +0.25 |$).\n"
            "👉 Bei **PSIR** bleibt das Vorzeichen erhalten. -0.25 bleibt Schwarz, +0.25 bleibt Weiß. Der Kontrast bleibt auch ohne perfekte Einstellungen erhalten."
        ))
        
        col_psir_ctrl, col_psir_graph = st.columns([1, 2])
        
        with col_psir_ctrl:
            st.markdown(f"#### {T('🎛️ Réglage du TI', '🎛️ TI Settings', '🎛️ TI-Einstellungen')}")
            
            ti_psir = st.slider(
                T("Temps d'Inversion (TI)", "Inversion Time (TI)", "Inversionszeit (TI)"), 
                100, 500, 280, step=10, format="%d ms",
                key="psir_ti_slider"
            )
            
            st.divider()
            
            mode_display = st.radio(
                T("Mode de Reconstruction", "Reconstruction Mode", "Rekonstruktionsmodus"),
                [T("A. Module (Nécessite TI Scout)", "A. Magnitude (Needs TI Scout)", "A. Betrag (Benötigt TI Scout)"), 
                 T("B. PSIR (Robuste)", "B. PSIR (Robust)", "B. PSIR (Robust)")],
                index=1,
                key="psir_mode_radio"
            )
            
            t1_myo_c = 596
            t1_fib_c = 286
            
            val_myo_c = 1 - 2 * np.exp(-ti_psir / t1_myo_c)
            val_fib_c = 1 - 2 * np.exp(-ti_psir / t1_fib_c)
            
            con_mag = abs(abs(val_myo_c) - abs(val_fib_c))
            con_psir = abs(val_myo_c - val_fib_c)
            
            st.divider()
            st.markdown(f"#### {T('📊 Score de Contraste', '📊 Contrast Score', '📊 Kontrast-Score')}")
            
            c_score1, c_score2 = st.columns(2)
            c_score1.metric("Magnitude", f"{con_mag:.2f}", delta_color="off")
            c_score2.metric("PSIR", f"{con_psir:.2f}", delta=T("Robuste", "Robust", "Robust") if con_psir > 0.4 else T("Faible", "Low", "Schwach"))
            
            if con_mag < 0.1 and con_psir > 0.4:
                st.error(T(
                    "🚨 **ÉCHEC MAGNITUDE !**\nLe contraste est nul (<0.1) car les signaux sont symétriques.\nC'est ici qu'il aurait fallu un **TI Scout** parfait.", 
                    "🚨 **MAGNITUDE FAIL!**\nContrast is null (<0.1) because signals are symmetric.\nThis is where a perfect **TI Scout** was needed.",
                    "🚨 **BETRAG FEHLGESCHLAGEN!**\nDer Kontrast ist null (<0.1), da die Signale symmetrisch sind.\nHier wäre ein perfekter **TI Scout** nötig gewesen."
                ))
                st.success(T(
                    "✅ **SUCCÈS PSIR !**\nMalgré le 'mauvais' TI, le contraste reste énorme (>0.4).\n👉 **Preuve de l'insensibilité au TI.**",
                    "✅ **PSIR SUCCESS!**\nDespite 'bad' TI, contrast remains huge (>0.4).\n👉 **Proof of TI insensitivity.**",
                    "✅ **PSIR ERFOLGREICH!**\nTrotz 'schlechter' TI bleibt der Kontrast enorm (>0.4).\n👉 **Beweis für die TI-Unempfindlichkeit.**"
                ))
            elif con_mag > 0.3:
                 st.info(T("Bon TI pour le Module (Chance ou TI Scout réussi).", "Good TI for Magnitude (Luck or good TI Scout).", "Gute TI für den Betrag (Glück oder guter TI Scout)."))

        with col_psir_graph:
            fig_psir = Figure(figsize=(10, 6)); ax_psir = fig_psir.subplots()
            
            tr_sim = 1000 
            t = np.linspace(0, tr_sim, 1000)
            
            mz_myo_raw = 1 - 2 * np.exp(-t / t1_myo_c)
            mz_blood_raw = 1 - 2 * np.exp(-t / t1_fib_c)
            
            pt_myo = val_myo_c
            pt_fib = val_fib_c
            
            if mode_display.startswith("A"):
                y_myo = np.abs(mz_myo_raw)
                y_blood = np.abs(mz_blood_raw)
                pt_myo_plot = np.abs(pt_myo)
                pt_fib_plot = np.abs(pt_fib)
                
                y_label = "Signal |Mz| (Rectifié)"
                title_g = T("Reconstruction Module (Tout est Positif)", "Magnitude Reconstruction (All Positive)", "Betragsrekonstruktion (Alles positiv)")
                
                ymin_graph, ymax_graph = -0.2, 1.2 
                grad_min, grad_max = 1, 0
                grad_extent = [0, 1, 0, 1] 
                yticks = [0, 0.5, 1]
                
            else:
                y_myo = mz_myo_raw
                y_blood = mz_blood_raw
                pt_myo_plot = pt_myo
                pt_fib_plot = pt_fib
                
                y_label = "Aimantation Mz (Réelle)"
                title_g = T("Reconstruction PSIR (Signe Conservé)", "PSIR Reconstruction (Sign Preserved)", "PSIR-Rekonstruktion (Vorzeichen erhalten)")
                
                ymin_graph, ymax_graph = -1.8, 1.3
                grad_min, grad_max = 1, -1
                grad_extent = [0, 1, -1, 1]
                yticks = [-1, -0.5, 0, 0.5, 1]

            ax_psir.plot(t, y_myo, label=T("Myocarde Sain", "Healthy Myocardium", "Gesundes Myokard"), color='#3498db', lw=2)
            ax_psir.plot(t, y_blood, label=T("Fibrose (Gado)", "Fibrosis (Gado)", "Fibrose (Gado)"), color='#e74c3c', lw=2)
            
            ax_psir.axhline(0, color='black', lw=1)
            ax_psir.axvline(ti_psir, color='gray', linestyle='--', alpha=0.8)
            ax_psir.text(ti_psir, 1.15, f"TI = {ti_psir}ms", color='gray', ha='center', fontweight='bold')
            
            ax_psir.plot(ti_psir, pt_myo_plot, 'o', color='#3498db', markersize=10, markeredgecolor='black', zorder=10)
            ax_psir.plot(ti_psir, pt_fib_plot, 'o', color='#e74c3c', markersize=10, markeredgecolor='black', zorder=10)

            y_seq_base = -1.5 if not mode_display.startswith("A") else -0.15 
            if not mode_display.startswith("A"):
                h_pulse = 0.25
                rect_180 = patches.Rectangle((0, y_seq_base), 40, h_pulse, facecolor='#c0392b', edgecolor='black', linewidth=1)
                ax_psir.add_patch(rect_180)
                ax_psir.text(20, y_seq_base - 0.15, "180°", ha='center', color='#c0392b', fontsize=8, fontweight='bold')
                
                n_readout = 8
                for k in range(n_readout):
                    pos_x = ti_psir + (k * 15)
                    rect_read = patches.Rectangle((pos_x, y_seq_base), 10, h_pulse*0.7, facecolor='#f1c40f', edgecolor='orange')
                    ax_psir.add_patch(rect_read)
                ax_psir.text(ti_psir + (n_readout*15)/2, y_seq_base - 0.15, "Lecture", ha='center', color='#d35400', fontsize=8)

            ax_psir.set_ylim(ymin_graph, ymax_graph)
            ax_psir.set_xlim(-50, tr_sim + 50)
            ax_psir.set_yticks(yticks)
            ax_psir.set_xlabel("Temps (ms)")
            ax_psir.set_ylabel(y_label)
            ax_psir.set_title(title_g)
            ax_psir.legend(loc='upper right')
            ax_psir.grid(True, alpha=0.3)
            
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax_psir)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cax.set_ylim(ymin_graph, ymax_graph)
            
            grad = np.linspace(grad_min, grad_max, 100).reshape(-1, 1)
            cax.imshow(grad, aspect='auto', cmap='gray', extent=[0, 1, grad_extent[2], grad_extent[3]])
            
            cax.plot([0.5], [pt_myo_plot], 'o', color='#3498db', markeredgecolor='white', markersize=8)
            cax.plot([0.5], [pt_fib_plot], 'o', color='#e74c3c', markeredgecolor='white', markersize=8)

            cax.set_xticks([])
            cax.set_yticks(yticks)
            cax.yaxis.set_ticks_position('right')
            cax.set_ylabel(T("Contraste", "Contrast", "Kontrast"))
            
            st.pyplot(fig_psir)
            
# --- 13. AFFICHAGE FINAL / FINAL DISPLAY ---
st.title(T("Simulateur MagnétoVault", "MagnetoVault Simulator", "MagnétoVault-Simulator"))

# LISTE DES MODULES POUR LE MENU DE NAVIGATION
liste_modules = [
    T("🏠 Accueil", "🏠 Home", "🏠 Startseite"), 
    T("👻 Fantôme", "👻 Phantom", "👻 Phantom"), 
    T("🌀 Espace K & Codage", "🌀 K-Space & Encoding", "🌀 K-Raum & Kodierung"), 
    T("📊 Signaux", "📊 Signals", "📊 Signale"), 
    T("🧠 Anatomie", "🧠 Anatomy", "🧠 Anatomie"), 
    T("📈 Physique", "📈 Physics", "📈 Physik"), 
    T("⚡ Chronogramme", "⚡ Timing Diagram", "⚡ Sequenzdiagramm"), 
    T("☣️ Artefacts", "☣️ Artifacts", "☣️ Artefakte"), 
    T("🚀 Imagerie Parallèle", "🚀 Parallel Imaging", "🚀 Parallele Bildgebung"), 
    T("🧬 Diffusion", "🧬 Diffusion", "🧬 Diffusion"), 
    T("🎓 Cours", "🎓 Course", "🎓 Kurs"), 
    T("🩸 SWI & Dipôle", "🩸 SWI & Dipole", "🩸 SWI & Dipol"), 
    T("🧊 3D T1 (MP-RAGE)", "🧊 3D T1 (MP-RAGE)", "🧊 3D T1 (MP-RAGE)"), 
    T("🏷️ ASL (Perfusion)", "🏷️ ASL (Perfusion)", "🏷️ ASL (Perfusion)"), 
    T("🩸 Angio TOF", "🩸 Angio TOF", "🩸 TOF-Angiographie"), 
    T("🍔 Fat Sat", "🍔 Fat Sat", "🍔 Fettunterdrückung"),
    T("🔥 Sécurité (SAR/B1+RMS)", "🔥 Safety (SAR/B1+RMS)", "🔥 Sicherheit (SAR/B1+RMS)"),
    T("🏗️ Architecture", "🏗️ Architecture", "🏗️ Architektur")
]

# MÉMOIRE DE LA POSITION
if 'nav_index' not in st.session_state:
    st.session_state.nav_index = 0

def update_nav():
    """Sauvegarde la position choisie dans la mémoire persistante."""
    st.session_state.nav_index = liste_modules.index(st.session_state.nav_widget)

# MENU DÉROULANT
module_actif = st.selectbox(
    T("🧭 Navigation", "🧭 Navigation", "🧭 Navigation"), 
    options=liste_modules, 
    index=st.session_state.nav_index,
    key="nav_widget",
    on_change=update_nav
)
st.markdown("---")

# [TAB 0 : ACCUEIL / HOME / STARTSEITE]
if module_actif == liste_modules[0]:
    st.markdown(T("""
    <div style="background-color:#1e293b; padding:20px; border-radius:10px; margin-bottom:20px;">
        <h1 style="color:white; margin:0;">🧲 Simulateur MagnétoVault</h1>
        <h3 style="color:#a5b4fc; margin-top:5px;">La "Boîte Blanche" de l'IRM</h3>
        <p style="color:#cbd5e1;"><i>"Ne vous contentez pas de voir l'image. Comprenez la mécanique de sa création."</i></p>
    </div>
    """, """
    <div style="background-color:#1e293b; padding:20px; border-radius:10px; margin-bottom:20px;">
        <h1 style="color:white; margin:0;">🧲 MagnétoVault Simulator</h1>
        <h3 style="color:#a5b4fc; margin-top:5px;">The MRI "White Box"</h3>
        <p style="color:#cbd5e1;"><i>"Don't just see the image. Understand the mechanics of its creation."</i></p>
    </div>
    """, """
    <div style="background-color:#1e293b; padding:20px; border-radius:10px; margin-bottom:20px;">
        <h1 style="color:white; margin:0;">🧲 MagnétoVault-Simulator</h1>
        <h3 style="color:#a5b4fc; margin-top:5px;">Die „White Box“ der MRT</h3>
        <p style="color:#cbd5e1;"><i>"Sehen Sie sich nicht nur das Bild an. Verstehen Sie die Mechanik seiner Entstehung."</i></p>
    </div>
    """), unsafe_allow_html=True)

    c_intro1, c_intro2 = st.columns([1, 1])
    with c_intro1:
        st.markdown(T("### 🔍 Pourquoi ce simulateur est unique ?", "### 🔍 Why is this simulator unique?", "### 🔍 Warum ist dieser Simulator einzigartig?"))
        st.markdown(T("""
        La plupart des simulateurs sont des "boîtes noires" : vous rentrez des paramètres, une image sort, mais vous ne savez pas pourquoi.
        
        **MagnétoVault est un laboratoire transparent.** Ici, nous ouvrons le capot de la machine pour vous montrer les mathématiques et la physique en action.
        """, """
        Most simulators are "black boxes": you enter parameters, an image comes out, but you don't know why.
        
        **MagnétoVault is a transparent laboratory.** Here, we open the hood of the machine to show you the mathematics and physics in action.
        """, """
        Die meisten Simulatoren sind „Black Boxes“: Sie geben Parameter ein, ein Bild kommt heraus, aber Sie wissen nicht, warum.
        
        **MagnétoVault ist un transparentes Labor.** Hier öffnen wir die Haube der Maschine, um Ihnen die Mathematik und Physik in Aktion zu zeigen.
        """))
    with c_intro2:
        st.info(T("""
        **Objectif :** Faire le lien entre la **Physique** (Spin, Vecteurs), l'**Espace K** (Fourier) et l'**Image Clinique** (Contraste).
        """, """
        **Goal:** Bridge the gap between **Physics** (Spin, Vectors), **K-Space** (Fourier), and **Clinical Image** (Contrast).
        """, """
        **Ziel:** Die Verknüpfung von **Physik** (Spin, Vektoren), **K-Raum** (Fourier) und **klinischem Bild** (Kontrast) verständlich zu machen.
        """))

    st.divider()

    st.markdown(T("### 🧪 Ce que vous pouvez explorer", "### 🧪 What you can explore", "### 🧪 Was Sie untersuchen können"))
    
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        st.markdown(T("#### 1. Mécanique de l'Espace K", "#### 1. K-Space Mechanics", "#### 1. K-Raum-Mechanik"))
        st.markdown(T("""
        Visualisez l'invisible. Comment la machine remplit-elle les lignes ?
        * **Facteur Turbo (TSE) :** Voyez comment les trains d'échos sont rangés. Lequel porte le contraste ? Lequel donne les détails ?
        * **TE Effectif :** Comprenez pourquoi il est placé au centre de l'espace K.
        """, """
        Visualize the invisible. How does the machine fill the lines?
        * **Turbo Factor (TSE):** See how echo trains are ordered. Which one carries contrast? Which one gives details?
        * **Effective TE:** Understand why it is placed at the center of K-space.
        """, """
        Visualisieren Sie das Unsichtbare. Wie füllt die Maschine die Zeilen?
        * **Turbofaktor (TSE):** Sehen Sie, wie Echozüge angeordnet sind. Welcher trägt den Kontrast? Welcher liefert die Details?
        * **Effektive TE:** Verstehen Sie, warum sie im Zentrum des K-Raums platziert ist.
        """))
    with col_p2:
        st.markdown(T("#### 2. Physique Temps Réel", "#### 2. Real-Time Physics", "#### 2. Echtzeit-Physik"))
        st.markdown(T("""
        Pas d'images pré-calculées. Tout est généré par les équations de Bloch.
        * **TR & TE :** Modifiez-les et voyez les courbes de relaxation changer.
        * **iPAT (Imagerie Parallèle) :** Activez le facteur d'accélération et observez la perte de SNR.
        * **Artefacts :** Créez du Repliement (Aliasing) ou du Décalage Chimique.
        """, """
        No pre-calculated images. Everything is generated by Bloch equations.
        * **TR & TE:** Modify them and watch relaxation curves change.
        * **iPAT (Parallel Imaging):** Enable acceleration factor and observe SNR loss.
        * **Artifacts:** Create Aliasing or Chemical Shift.
        """, """
        Keine vorberechneten Bilder. Alles wird durch die Bloch-Gleichungen generiert.
        * **TR & TE:** Ändern Sie sie und sehen Sie, wie sich die Relaxationskurven verändern.
        * **iPAT (Parallele Bildgebung):** Aktivieren Sie den Beschleunigungsfaktor und beobachten Sie den SNR-Verlust.
        * **Artefakte:** Erzeugen Sie Aliasing (Einfaltung) oder chemische Verschiebung (Chemical Shift).
        """))
    with col_p3:
        st.markdown(T("#### 3. Clinique Avancée", "#### 3. Advanced Clinical", "#### 3. Fortgeschrittene Klinik"))
        st.markdown(T("""
        Au-delà du T1/T2 classique. Simulez des séquences complexes :
        * **Diffusion (DWI) :** Jouez avec le *Facteur b* et la carte *ADC*.
        * **Perfusion (ASL) :** Comprenez le marquage des spins artériels.
        * **SWI :** Visualisez la Phase et la Magnitude (Effet dipôle).
        """, """
        Beyond classic T1/T2. Simulate complex sequences:
        * **Diffusion (DWI):** Play with *b-Factor* and *ADC* map.
        * **Perfusion (ASL):** Understand arterial spin labeling.
        * **SWI:** Visualize Phase and Magnitude (Dipole effect).
        """, """
        Über das klassische T1/T2 hinaus. Simulieren Sie komplexe Sequenzen:
        * **Diffusion (DWI):** Spielen Sie mit dem *b-Wert* und der *ADC-Karte*.
        * **Perfusion (ASL):** Verstehen Sie die Markierung arterieller Spins (Arterial Spin Labeling).
        * **SWI:** Visualisieren Sie Phase und Magnitude (Dipoleffekt).
        """))

    st.divider()
    st.markdown(T("### 🚀 Guide de Démarrage", "### 🚀 Quick Start Guide", "### 🚀 Schnellstartanleitung"))
    st.markdown(T("""
    1.  **🎛️ Console (Gauche) :** C'est votre poste de pilotage. Choisissez la **Séquence**, réglez le **FOV**, la **Matrice**, le **TR/TE** et le **Facteur Turbo**.
    2.  **🌀 Espace K (Onglet 2) :** Regardez comment votre séquence remplit les données brutes.
    3.  **🧠 Anatomie (Onglet 5) :** Explorez un cerveau humain réel (Atlas *Harvard-Oxford*) et simulez des pathologies (**AVC**, **Atrophie**).
    """, """
    1.  **🎛️ Console (Left):** This is your cockpit. Choose the **Sequence**, adjust **FOV**, **Matrix**, **TR/TE**, and **Turbo Factor**.
    2.  **🌀 K-Space (Tab 2):** Watch how your sequence fills the raw data.
    3.  **🧠 Anatomy (Tab 5):** Explore a real human brain (*Harvard-Oxford* Atlas) and simulate pathologies (**Stroke**, **Atrophy**).
    """, """
    1.  **🎛️ Konsole (Links):** Das ist Ihr Cockpit. Wählen Sie die **Sequenz**, stellen Sie **FOV**, **Matrix**, **TR/TE** und **Turbofaktor** ein.
    2.  **🌀 K-Raum (Tab 2):** Sehen Sie, wie Ihre Sequenz die Rohdaten füllt.
    3.  **🧠 Anatomie (Tab 5):** Erkunden Sie ein reales menschliches Gehirn (*Harvard-Oxford*-Atlas) und simulieren Sie Pathologien (**Schlaganfall**, **Atrophie**).
    """))

    st.divider()
    
    # --- GLOSSAIRE DÉPLOYABLE / EXPANDABLE GLOSSARY ---
    with st.expander(T("📖 Glossaire Complet (Variables & Formules)", "📖 Complete Glossary (Variables & Formulas)", "📖 Vollständiges Glossar (Variablen & Formeln)"), expanded=False):
        
        # 1. PHYSIQUE FONDAMENTALE
        st.markdown(T("### 🧲 1. Physique Fondamentale", "### 🧲 1. Fundamental Physics", "### 🧲 1. Grundlagen der Physik"))
        col_phy1, col_phy2 = st.columns(2)
        with col_phy1:
            st.markdown(T("""
            * **$B_0$ (Tesla)** : Champ magnétique statique principal.
            * **$\gamma$ (Gamma)** : Rapport gyromagnétique (42.58 MHz/T).
            * **$\omega_0$ (Hz)** : Fréquence de Larmor ($\omega_0 = \gamma B_0$).
            """, """
            * **$B_0$ (Tesla)**: Main static magnetic field.
            * **$\gamma$ (Gamma)**: Gyromagnetic ratio (42.58 MHz/T).
            * **$\omega_0$ (Hz)**: Larmor frequency ($\omega_0 = \gamma B_0$).
            """, """
            * **$B_0$ (Tesla)**: Hauptstatisches Magnetfeld.
            * **$\gamma$ (Gamma)**: Gyromagnetisches Verhältnis (42.58 MHz/T).
            * **$\omega_0$ (Hz)**: Larmor-Frequenz ($\omega_0 = \gamma B_0$).
            """))
        with col_phy2:
            st.markdown(T("""
            * **$M_0$** : Aimantation nette à l'équilibre.
            * **$M_z$** : Aimantation longitudinal (T1).
            * **$M_{xy}$** : Aimantation transversale (T2).
            """, """
            * **$M_0$**: Net magnetization at equilibrium.
            * **$M_z$**: Longitudinal magnetization (T1).
            * **$M_{xy}$**: Transverse magnetization (T2).
            """, """
            * **$M_0$**: Nettomagnetisierung im Gleichgewicht.
            * **$M_z$**: Longitudinale Magnetisierung (T1).
            * **$M_{xy}$**: Transversale Magnetisierung (T2).
            """))

        st.markdown("---")

        # 2. PROPRIÉTÉS TISSULAIRES
        st.markdown(T("### 🧠 2. Propriétés Tissulaires", "### 🧠 2. Tissue Properties", "### 🧠 2. Gewebeeigenschaften"))
        col_tis1, col_tis2 = st.columns(2)
        with col_tis1:
            st.markdown(T("""
            * **$T1$ (ms)** : Relaxation longitudinale (Spin-Réseau).
            * **$T2$ (ms)** : Relaxation transversale (Spin-Spin).
            """, """
            * **$T1$ (ms)**: Longitudinal relaxation (Spin-Lattice).
            * **$T2$ (ms)**: Transverse relaxation (Spin-Spin).
            """, """
            * **$T1$ (ms)**: Longitudinale Relaxation (Spin-Gitter).
            * **$T2$ (ms)**: Transversale Relaxation (Spin-Spin).
            """))
        with col_tis2:
            st.markdown(T(r"""
            * **$T2^*$ (ms)** : T2 réel + Inhomogénéités de champ.
            * **$\rho$ (DP)** : Densité de Protons (quantité d'H+).
            """, r"""
            * **$T2^*$ (ms)**: True T2 + Field inhomogeneities.
            * **$\rho$ (PD)**: Proton Density (amount of H+).
            """, r"""
            * **$T2^*$ (ms)**: Echtes T2 + Feldinhomogenitäten.
            * **$\rho$ (PD)**: Protonendichte (Anzahl der H+).
            """))

        st.markdown("---")

        # 3. PARAMÈTRES SÉQUENCE
        st.markdown(T("### ⏱️ 3. Paramètres Séquence", "### ⏱️ Sequence Parameters", "### ⏱️ Sequenzparameter"))
        col_seq1, col_seq2 = st.columns(2)
        with col_seq1:
            st.markdown(T("""
            * **TR (ms)** : Temps de Répétition.
            * **TE (ms)** : Temps d'Écho.
            * **TI (ms)** : Temps d'Inversion.
            """, """
            * **TR (ms)**: Repetition Time.
            * **TE (ms)**: Echo Time.
            * **TI (ms)**: Inversion Time.
            """, """
            * **TR (ms)**: Repetitionszeit (Repetition Time).
            * **TE (ms)**: Echozeit (Echo Time).
            * **TI (ms)**: Inversionszeit (Inversion Time).
            """))
        with col_seq2:
            st.markdown(T(r"""
            * **$\alpha$ (Flip Angle)** : Angle de bascule RF.
            * **ETL** : Echo Train Length (Facteur Turbo).
            * **BW** (Hz/Px) : Bande Passante.
            """, r"""
            * **$\alpha$ (Flip Angle)**: RF Flip Angle.
            * **ETL**: Echo Train Length (Turbo Factor).
            * **BW** (Hz/Px): Bandwidth.
            """, r"""
            * **$\alpha$ (Flip Angle)**: RF-Ablenkwinkel (Flipwinkel).
            * **ETL**: Echozuglänge (Turbofaktor).
            * **BW** (Hz/Px): Bandbreite.
            """))

        st.markdown("---")

        # 4. SÉCURITÉ
        st.markdown(T("### 🔥 4. Sécurité", "### 🔥 4. Safety", "### 🔥 4. Sicherheit"))
        col_sar1, col_sar2 = st.columns(2)
        with col_sar1:
            st.markdown(T("""
            * **$B_1^{+RMS}$ ($\mu T$)** : Moyenne champ RF (Risque Implants).
            * **$B_{1,peak}$** : Amplitude max instantanée.
            """, """
            * **$B_1^{+RMS}$ ($\mu T$)**: RMS RF field (Implant Risk).
            * **$B_{1,peak}$**: Peak instantaneous amplitude.
            """, """
            * **$B_1^{+RMS}$ ($\mu T$)**: RF-Effektivfeld (Implantatrisiko).
            * **$B_{1,peak}$**: Maximale Momentanamplitude.
            """))
        with col_sar2:
            st.markdown(T("""
            * **$SAR$ (W/kg)** : Énergie absorbée par le patient (Chauffe).
            * **$DC$ (%)** : Duty Cycle (Rapport Cyclique).
            """, """
            * **$SAR$ (W/kg)**: Specific Absorption Rate (Patient heating).
            * **$DC$ (%)**: Duty Cycle.
            """, """
            * **$SAR$ (W/kg)**: Spezifische Absorptionsrate (Patientenerwärmung).
            * **$DC$ (%)**: Tastgrad (Duty Cycle).
            """))

# [TAB 1 : FANTOME / PHANTOM]
elif module_actif == liste_modules[1]:
    # =========================================================
    # 1. SETUP & PARAMÈTRES
    # =========================================================
    def get_p(name, def_val): return getattr(cst, name, def_val)

    T_WM = get_p('T_WM', {'T1': 600, 'T2': 80, 'PD': 0.7, 'ADC': 0.7e-3})
    T_GM = get_p('T_GM', {'T1': 1100, 'T2': 100, 'PD': 0.8, 'ADC': 0.8e-3})
    T_CSF = get_p('T_LCR', {'T1': 4000, 'T2': 2000, 'PD': 1.0, 'ADC': 3.0e-3})
    T_STROKE = get_p('T_STROKE', {'T1': 1100, 'T2': 200, 'PD': 0.9, 'ADC': 0.4e-3})
    
    # On crée une copie locale pour ne pas altérer la constante globale
    T_FAT = get_p('T_FAT', {'T1': 250, 'T2': 60, 'PD': 0.9, 'ADC': 0}).copy()
    
    # --- CORRECTION : Effet J-coupling (Graisse brillante en TSE) ---
    if turbo > 1:
        T_FAT['T2'] = 150  # Allongement artificiel du T2 de la graisse

    # =========================================================
    # 2. PHYSIQUE : TEMPS (TA)
    # =========================================================
    esp = 10.0 
    overhead = 8.0 
    time_per_slice = overhead + (turbo * esp) 
    
    max_slices_per_tr = int(tr / time_per_slice)
    if max_slices_per_tr < 1: max_slices_per_tr = 1
    
    import math
    min_concats = math.ceil(n_slices / max_slices_per_tr)
    
    if not (is_mprage or is_dwi):
        final_concats = max(1, min_concats)
    else:
        final_concats = 1 

    # --- SECTION CORRIGÉE POUR LE TEMPS D'ACQUISITION ---
    if is_dwi:
        # On multiplie par 18.5 pour atteindre ~1m51s (si TR = 6000ms)
        raw_time_ms = tr * nex * 18.5
    else:
        # Séquences classiques (Ligne par ligne)
        raw_time_ms = (tr * mat * nex * final_concats) / (turbo * ipat_factor)
    
    # Le calcul final se fait une seule fois pour les deux cas
    final_seconds = raw_time_ms / 1000.0
    str_duree = f"{int(final_seconds // 60)} min {int(final_seconds % 60)} s"

    # =========================================================
    # 3. PHYSIQUE : SIGNAUX
    # =========================================================
    v_wm = phy.calculate_signal(tr, te, ti, T_WM['T1'], T_WM['T2'], 50, T_WM.get('ADC',0), T_WM['PD'], flip_angle, is_gre, is_dwi, b_value if is_dwi else 0)
    v_gm = phy.calculate_signal(tr, te, ti, T_GM['T1'], T_GM['T2'], 60, T_GM.get('ADC',0), T_GM['PD'], flip_angle, is_gre, is_dwi, b_value if is_dwi else 0)
    v_csf = phy.calculate_signal(tr, te, ti, T_CSF['T1'], T_CSF['T2'], 500, T_CSF.get('ADC',0), T_CSF['PD'], flip_angle, is_gre, is_dwi, b_value if is_dwi else 0)
    v_fat = phy.calculate_signal(tr, te, ti, T_FAT['T1'], T_FAT['T2'], 40, T_FAT.get('ADC',0), T_FAT['PD'], flip_angle, is_gre, is_dwi, 0) if not is_dwi else 0.0

    if show_stroke and is_dwi: v_stroke = 2.0 if b_value >= 800 else v_gm
    else: v_stroke = phy.calculate_signal(tr, te, ti, T_STROKE['T1'], T_STROKE['T2'], 80, T_STROKE.get('ADC',0), T_STROKE['PD'], flip_angle, is_gre, is_dwi, 0)

    # =========================================================
    # 4. PHYSIQUE : SNR & BRUIT (IMPACT iPAT CORRIGÉ)
    # =========================================================
    
    # Références (Calibration 100%)
    ref_ep = 4.0; ref_bw = 220.0; ref_nex = 1.0
    ref_mat = 256.0; ref_fov = 240.0; ref_turbo = 1.0
    
    # Signal de Ref
    def_tr = float(defaults['tr']); def_te = float(defaults['te'])
    ref_sig = phy.calculate_signal(def_tr, def_te, 0, T_WM['T1'], T_WM['T2'], 50, 0, T_WM['PD'], 90, False, False, 0)
    if ref_sig == 0: ref_sig = 0.001

    # --- FACTEURS ---
    # 1. Voxel (Matrice & FOV) - Impact Carré
    pixel_area_cur = (fov / mat) ** 2
    pixel_area_ref = (ref_fov / ref_mat) ** 2
    f_vox = (pixel_area_cur / pixel_area_ref) * (ep / ref_ep)

    # 2. BW & NEX
    f_bw = np.sqrt(ref_bw / float(bw))
    f_nex = np.sqrt(nex / ref_nex)
    f_turbo = 1.0 / (turbo ** 0.15)
    
    # 3. iPAT (CORRECTION MAJEURE : FACTEUR G)
    if ipat_factor > 1:
        g_factor = 1.0 + (0.3 * (ipat_factor - 1))
        f_ipat = 1.0 / (g_factor * np.sqrt(ipat_factor))
    else:
        g_factor = 1.0
        f_ipat = 1.0

    # --- NOUVEAU : BASCULE MODE CLINIQUE / PHYSIQUE ---
    mode_clinique_actif = st.toggle(
        T("🏥 Mode Console Clinique", "🏥 Clinical Console Mode", "🏥 Klinischer Konsolenmodus"), 
        value=True, 
        help=T("Désactivez pour voir l'impact physique réel du TR (lié au nb de coupes/concats) sur le SNR.", 
               "Disable to see the true physical impact of TR (from slices/concats) on SNR.",
               "Deaktivieren Sie diese Option, um die tatsächlichen physikalischen Auswirkungen der TR (durch Anzahl der Schichten/Konkatenationen) auf das SNR zu sehen.")
    )

    # 4. Signal Tissulaire (Impact du TR)
    if mode_clinique_actif:
        f_sig = 1.0  
    else:
        f_sig = (v_wm / ref_sig) if ref_sig > 0 else 0  
    
    # --- CALCUL DU SNR ---
    if is_dwi:
        snr_base = 100.0 
        facteur_matrice = (128 / mat) ** 2
        snr_final = snr_base * facteur_matrice * np.exp(-b_value * 0.001)
    else:
        snr_final = 100.0 * f_vox * f_bw * f_nex * f_turbo * f_sig * f_ipat
        if is_mprage:
            snr_final = snr_final * np.sqrt(max(1, n_slices))

    str_snr = f"{snr_final:.1f} %"

    # =========================================================
    # 5. MOTEUR VISUEL
    # =========================================================
    import scipy.ndimage as ndimage
    
    # Génération Image
    S_render = 256 
    x = np.linspace(-1, 1, S_render); y = np.linspace(-1, 1, S_render)
    X, Y = np.meshgrid(x, y); D = np.sqrt(X**2 + Y**2)

    img_sim = np.zeros((S_render, S_render))
    img_sim[D < 0.20] = v_csf
    img_sim[(D >= 0.20) & (D < 0.50)] = v_wm
    img_sim[(D >= 0.50) & (D < 0.80)] = v_gm
    img_sim[(D >= 0.80) & (D < 0.95)] = v_fat

    if show_stroke:
        mask_stroke = (np.sqrt((X-0.3)**2 + (Y-0.1)**2) < 0.15) & (D >= 0.20)
        img_sim[mask_stroke] = v_stroke

    # Pixelisation Matrice
    zoom_down = mat / S_render
    img_pix = ndimage.zoom(img_sim, zoom_down, order=0)
    img_disp = ndimage.zoom(img_pix, 256.0 / mat, order=0)

    # 0. Initialisation des variables de contrôle
    decentrage_penalite = 0.0
    
    if turbo > 1:
        center_index = (turbo + 1) / 2
        effective_index = max(1, min(turbo, round(te / max(es, 1.0))))
        dist_from_center = abs(effective_index - center_index)
        
        if dist_from_center <= 3:
            sigma_base = 0.01 
            decentrage_penalite = 0.0
        else:
            decentrage_penalite = (dist_from_center - 3) / (turbo / 2)
            sigma_base = 0.01 + (decentrage_penalite * 0.5) 
            
        sigma_val = (turbo - 1) * sigma_base
        img_disp = ndimage.gaussian_filter(img_disp, sigma=sigma_val)

    # --- AFFICHAGE DES MESSAGES PÉDAGOGIQUES (Sidebar) ---
    if turbo > 1:
        if decentrage_penalite > 0.5:
            st.sidebar.error(T("🛑 Flou T2 critique : TE trop éloigné du centre !", "🛑 Critical T2 blur: TE too far from center!", "🛑 Kritische T2-Unschärfe: TE zu weit vom Zentrum entfernt!"))
        elif decentrage_penalite > 0:
            st.sidebar.warning(T("⚠️ Flou T2 visible : TE hors zone optimale.", "⚠️ Visible T2 blur: TE out of optimal zone.", "⚠️ Sichtbare T2-Unschärfe: TE außerhalb der optimalen Zone."))
        else:
            st.sidebar.success(T("✨ Netteté optimale : TE dans la zone de confort (+/- 3).", "✨ Optimal sharpness: TE in comfort zone (+/- 3).", "✨ Optimale Schärfe: TE in der Komfortzone (+/- 3)."))

    # BRUIT VISUEL (Loi en 1/SNR²)
    max_val = np.max(img_disp)
    if max_val > 0: img_disp /= max_val
    
    base_noise = 0.02
    sigma_noise = base_noise * ((100.0 / (snr_final + 0.1)) ** 1.5)
    sigma_noise = min(sigma_noise, 0.8)
    
    noise_map = np.random.normal(0, sigma_noise, (256, 256))
    img_final = np.clip(img_disp + noise_map, 0, 1)

    # =========================================================
    # 6. AFFICHAGE
    # =========================================================
    c1, c2 = st.columns([1, 1])
    
    with c1:
        k1, k2 = st.columns(2)
        k1.metric(T("⏱️ Durée", "⏱️ Duration", "⏱️ Dauer"), str_duree)
        k2.metric(T("📉 SNR Relatif", "📉 Relative SNR", "📉 Relatives SNR"), str_snr)
        
        # Feedback visuel iPAT
        if ipat_factor > 1:
            msg_ipat = T(f"⚠️ iPAT x{ipat_factor} activé : Perte SNR importante (Facteur g={g_factor:.1f})",
                         f"⚠️ iPAT x{ipat_factor} enabled: Significant SNR loss (g-Factor={g_factor:.1f})",
                         f"⚠️ iPAT x{ipat_factor} aktiviert: Erheblicher SNR-Verlust (g-Faktor={g_factor:.1f})")
            st.warning(msg_ipat)

        st.divider()
        st.subheader(T("1. Formules Physiques", "1. Physics Formulas", "1. Physikalische Formeln"))
        
        # --- A. TEMPS D'ACQUISITION (TA) ---
        st.markdown("**A. " + T("Temps d'Acquisition", "Acquisition Time", "Akquisitionszeit") + " (TA) :**")
        st.latex(r"TA = \frac{TR \times N_{PE} \times NEX}{ETL \times R}")
        
        # --- B. SNR ---
        st.markdown("**B. " + T("Rapport Signal / Bruit", "Signal-to-Noise Ratio", "Signal-Rausch-Verhältnis") + " (SNR) :**")
        st.latex(r"SNR \propto V_{vox} \times \sqrt{\frac{NEX}{BW}} \times \frac{1}{g \cdot \sqrt{R}}")

        # --- C. LÉGENDES DÉTAILLÉES (GLOSSAIRE COMPLET) ---
        with st.expander(T("📖 Légende des Variables (Cliquez)", "📖 Variable Legend (Click)", "📖 Variablenlegende (Klicken)"), expanded=False):
            st.markdown(T("""
            * **$TR$** : Temps de Répétition (ms).
            * **$N_{PE}$** : Lignes de Phase (Matrice Y).
            * **$NEX$** : Nombre de Moyennages (Averages).
            * **$ETL$** : Facteur Turbo (Echo Train Length).
            * **$R$** : Facteur d'Accélération (iPAT).
            * **$V_{vox}$** : Volume du Voxel (dépend de FOV, Matrice, Épaisseur).
            * **$FOV$** : Champ de Vue (Field of View).
            * **$Matrice$** : Résolution de l'image (Ex: 256x256).
            * **$Ep$** : Épaisseur de Coupe (Slice Thickness).
            * **$BW$** : Bande Passante (Bandwidth).
            * **$g$** : Facteur de géométrie (Bruit lié à l'imagerie parallèle).
            """, """
            * **$TR$**: Repetition Time (ms).
            * **$N_{PE}$**: Phase Encoding Lines (Matrix Y).
            * **$NEX$**: Number of Excitations (Averages).
            * **$ETL$**: Turbo Factor (Echo Train Length).
            * **$R$**: Acceleration Factor (iPAT).
            * **$V_{vox}$**: Voxel Volume (depends on FOV, Matrix, Thickness).
            * **$FOV$**: Field of View.
            * **$Matrix$**: Image Resolution (e.g., 256x256).
            * **$Thick$**: Slice Thickness.
            * **$BW$**: Bandwidth.
            * **$g$**: Geometry factor (Noise form parallel imaging).
            """, """
            * **$TR$**: Repetitionszeit (ms).
            * **$N_{PE}$**: Phasenkodierzeilen (Matrix Y).
            * **$NEX$**: Anzahl der Anregungen (Averages).
            * **$ETL$**: Turbofaktor (Echo Train Length).
            * **$R$**: Beschleunigungsfaktor (iPAT).
            * **$V_{vox}$**: Voxelvolumen (abhängig von FOV, Matrix, Schichtdicke).
            * **$FOV$**: Sichtfeld (Field of View).
            * **$Matrix$**: Bildauflösung (z.B. 256x256).
            * **$Dicke$**: Schichtdicke.
            * **$BW$**: Bandbreite.
            * **$g$**: Geometriefaktor (Rauschen bei paralleler Bildgebung).
            """))

        # Glossaire Analyse d'Impact
        with st.expander(T("🔍 Analyse d'Impact", "🔍 Impact Analysis", "🔍 Auswirkungsanalyse"), expanded=True):
            st.markdown(T("""
            * **iPAT ($R$)** : 📉 Divise le temps par $R$, mais le SNR chute de plus de $\sqrt{R}$.
            * **Matrice ($N_{PE}$)** : 🕰️ Augmente le temps (proportionnel) et 📉 Diminue le SNR (au carré).
            * **FOV** : 📈 Impact SNR énorme (au carré).
            """, """
            * **iPAT ($R$)**: 📉 Divides time by $R$, but SNR drops more than $\sqrt{R}$.
            * **Matrix ($N_{PE}$)**: 🕰️ Increases time (linear) and 📉 Decreases SNR (squared).
            * **FOV**: 📈 Huge impact on SNR (squared).
            """, """
            * **iPAT ($R$)**: 📉 Teilt die Zeit durch $R$, aber das SNR sinkt um mehr als $\sqrt{R}$.
            * **Matrix ($N_{PE}$)**: 🕰️ Erhöht die Zeit (linear) und 📉 verringert das SNR (quadratisch).
            * **FOV**: 📈 Enorme Auswirkung auf das SNR (quadratisch).
            """))
            
    with c2:
            st.write(T("🖼️ **Rendu Visuel**", "🖼️ **Visual Render**", "🖼️ **Visuelles Rendering**"))
            
            fig_p = Figure(figsize=(5, 5))
            ax_p = fig_p.subplots()
            
            ax_p.imshow(img_final, cmap='gray', vmin=0, vmax=1)
            ax_p.axis('off')
            
            info = f"SNR: {int(snr_final)}% | iPAT: {ipat_factor} (g={g_factor:.1f}) | Mat: {mat}"
            ax_p.set_title(info, fontsize=10, color="gray")
            
            if sigma_noise < 0.4:
                # (Le code à l'intérieur du if est décalé d'un cran supplémentaire vers la droite)
                pass # (exemple)
            
            # st.pyplot DOIT ÊTRE ALIGNÉ AVEC LE "if" ET LE "ax_p" !
            st.pyplot(fig_p, use_container_width=False)

    # =========================================================
    # 7. EXPLICATION PÉDAGOGIQUE DU MODE PHYSIQUE PURE
    # =========================================================
    if not mode_clinique_actif:
        st.divider()
        st.markdown(T(
            "<h3 style='color: #8e44ad; border-bottom: 2px solid #8e44ad; padding-bottom: 5px;'>🔬 Mode Physique Pure : Les Équations de Bloch en Action</h3>", 
            "<h3 style='color: #8e44ad; border-bottom: 2px solid #8e44ad; padding-bottom: 5px;'>🔬 Pure Physics Mode: Bloch Equations in Action</h3>",
            "<h3 style='color: #8e44ad; border-bottom: 2px solid #8e44ad; padding-bottom: 5px;'>🔬 Reiner Physikmodus: Die Bloch-Gleichungen in Aktion</h3>"
        ), unsafe_allow_html=True)
        
        c_bloch1, c_bloch2 = st.columns([1.2, 1])
        with c_bloch1:
            st.info(T(
                "**Pourquoi le SNR varie-t-il avec les coupes et les concaténations ?**\n\n"
                "Dans la réalité physique, le signal dépend intimement du temps laissé aux protons pour repousser selon l'axe longitudinal (Repousse T1). "
                "C'est ce que décrit l'équation de Bloch :\n\n"
                "• 📈 **Plus de coupes :** La machine est obligée d'allonger le Temps de Répétition (TR) pour avoir le temps d'acquérir chaque coupe. "
                "Un TR plus long laisse l'aimantation $M_z$ remonter plus haut. Le signal brut explose, et le SNR avec.\n\n"
                "• 📉 **Plus de concaténations :** La machine divise l'acquisition en plusieurs paquets, ce qui lui permet de relâcher la contrainte sur le TR "
                "(le TR redevient court). L'aimantation n'a pas le temps de repousser, le signal s'effondre.",
                
                "**Why does SNR vary with slices and concatenations?**\n\n"
                "In physical reality, the signal depends intimately on the time given to protons to recover along the longitudinal axis (T1 Recovery). "
                "This is described by the Bloch equations:\n\n"
                "• 📈 **More slices:** The machine is forced to extend the Repetition Time (TR) to have time to acquire each slice. "
                "A longer TR allows the magnetization $M_z$ to recover higher. The raw signal explodes, and the SNR with it.\n\n"
                "• 📉 **More concatenations:** The machine divides the acquisition into several packages, easing the constraint on the TR "
                "(TR becomes short again). Magnetization doesn't have time to recover, the signal collapses.",
                
                "**Warum variiert das SNR mit den Schichten und Konkatenationen?**\n\n"
                "In der physikalischen Realität hängt das Signal eng mit der Zeit zusammen, die den Protonen zur Verfügung steht, um entlang der Längsachse zu relaxieren (T1-Erholung). "
                "Dies wird durch die Bloch-Gleichung beschrieben:\n\n"
                "• 📈 **Mehr Schichten:** Die Maschine muss die Repetitionszeit (TR) verlängern, um Zeit für die Akquisition jeder Schicht zu haben. "
                "Eine längere TR lässt die Magnetisierung $M_z$ höher ansteigen. Das Rohsignal explodiert, und damit auch das SNR.\n\n"
                "• 📉 **Mehr Konkatenationen:** Die Maschine teilt die Akquisition in mehrere Pakete auf, wodurch die Einschränkung der TR gelockert wird "
                "(die TR wird wieder kurz). Die Magnetisierung hat keine Zeit sich zu erholen, das Signal bricht ein."
            ))
            
        with c_bloch2:
            st.markdown(T("#### Les Mathématiques du Signal", "#### The Mathematics of Signal", "#### Die Mathematik des Signals"))
            
            st.markdown(T("**1. Repousse Longitudinale (Contraste T1) :**", "**1. Longitudinal Recovery (T1 Contrast):**", "**1. Longitudinale Erholung (T1-Kontrast):**"))
            st.latex(r"M_z(TR) = M_0 \left(1 - e^{-\frac{TR}{T_1}}\right)")
            
            st.markdown(T("**2. Décroissance Transversale (Contraste T2) :**", "**2. Transverse Decay (T2 Contrast):**", "**2. Transversaler Zerfall (T2-Kontrast):**"))
            st.latex(r"M_{xy}(TE) = M_z \cdot e^{-\frac{TE}{T_2}}")
            
            st.caption(T(
                "💡 *Note : Sur une vraie console clinique, le calculateur de SNR fige arbitrairement l'impact du TR (Mode Clinique) pour ne pas vous induire en erreur sur la qualité visuelle finale de l'image.*",
                "💡 *Note: On a real clinical console, the SNR calculator arbitrarily freezes the impact of TR (Clinical Mode) so as not to mislead you about the final visual quality of the image.*",
                "💡 *Hinweis: Auf einer echten klinischen Konsole friert der SNR-Rechner die Auswirkungen der TR (klinischer Modus) willkürlich ein, um Sie nicht über die endgültige visuelle Qualität des Bildes in die Irre zu führen.*"
            ))

# ==============================================================================
# [TAB 2 : ESPACE K - TERMINOLOGIE CORRIGÉE]
# ==============================================================================
elif module_actif == liste_modules[2]:
    # 1. TITRE PRINCIPAL
    st.markdown(T("""
    <div style="background-color: #1e293b; padding: 20px; border-radius: 10px; margin-bottom: 25px; text-align: center; border-bottom: 4px solid #3b82f6;">
        <h1 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">🌀 Espace K : La Bibliothèque de l'Image</h1>
        <p style="color: #94a3b8; margin-top: 5px; font-size: 16px;">De la Fréquence au Pixel : Le voyage du signal</p>
    </div>
    """, """
    <div style="background-color: #1e293b; padding: 20px; border-radius: 10px; margin-bottom: 25px; text-align: center; border-bottom: 4px solid #3b82f6;">
        <h1 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">🌀 K-Space: The Image Library</h1>
        <p style="color: #94a3b8; margin-top: 5px; font-size: 16px;">From Frequency to Pixel: The signal's journey</p>
    </div>
    """, """
    <div style="background-color: #1e293b; padding: 20px; border-radius: 10px; margin-bottom: 25px; text-align: center; border-bottom: 4px solid #3b82f6;">
        <h1 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">🌀 K-Raum: Die Bildbibliothek</h1>
        <p style="color: #94a3b8; margin-top: 5px; font-size: 16px;">Von der Frequenz zum Pixel: Die Reise des Signals</p>
    </div>
    """), unsafe_allow_html=True)
    
    # 2. EN-TÊTE PÉDAGOGIQUE
    with st.expander(T("🎶 Comprendre le Codage : De la Chorale à la Physique", "🎶 Understanding Encoding: From Choir to Physics", "🎶 Kodierung verstehen: Vom Chor zur Physik"), expanded=True):
        c_txt1, c_txt2, c_txt3 = st.columns(3)
        
        with c_txt1:
            st.markdown(T("""
            <div style="background-color: #eff6ff; padding: 15px; border-radius: 8px; border-left: 5px solid #3b82f6; height: 100%;">
                <h3 style="color: #1e40af; margin: 0 0 10px 0; font-size: 20px;">1. Le Problème</h3>
                <p style="font-size: 14px; color: #334155; margin: 0;"><b>Le Chaos :</b> Imaginez une <b>foule</b> où tout le monde crie "A" en même temps. Impossible de savoir qui est où. Sans codage spatial, l'IRM ne reçoit qu'un bruit global.</p>
            </div>
            """, """
            <div style="background-color: #eff6ff; padding: 15px; border-radius: 8px; border-left: 5px solid #3b82f6; height: 100%;">
                <h3 style="color: #1e40af; margin: 0 0 10px 0; font-size: 20px;">1. The Problem</h3>
                <p style="font-size: 14px; color: #334155; margin: 0;"><b>The Chaos:</b> Imagine a <b>crowd</b> where everyone shouts "A" at the same time. Impossible to locate anyone. Without spatial encoding, MRI receives only global noise.</p>
            </div>
            """, """
            <div style="background-color: #eff6ff; padding: 15px; border-radius: 8px; border-left: 5px solid #3b82f6; height: 100%;">
                <h3 style="color: #1e40af; margin: 0 0 10px 0; font-size: 20px;">1. Das Problem</h3>
                <p style="font-size: 14px; color: #334155; margin: 0;"><b>Das Chaos:</b> Stellen Sie sich eine <b>Menge</b> vor, in der alle gleichzeitig "A" rufen. Unmöglich zu wissen, wer wo ist. Ohne räumliche Kodierung empfängt das MRT nur globales Rauschen.</p>
            </div>
            """), unsafe_allow_html=True)
            
        with c_txt2:
            st.markdown(T("""
            <div style="background-color: #fff7ed; padding: 15px; border-radius: 8px; border-left: 5px solid #f97316; height: 100%;">
                <h3 style="color: #9a3412; margin: 0 0 10px 0; font-size: 20px;">2. La Solution</h3>
                <p style="font-size: 14px; color: #334155; margin: 0;"><b>Le Tri :</b> On applique des gradients pour "trier" les signaux :</p>
                <ul style="font-size: 13px; color: #334155; padding-left: 20px; margin-top: 5px;">
                    <li style="margin-bottom: 5px;"><b>Fréquence :</b> Trie de Gauche à Droite (Grave ↔ Aigu).</li>
                    <li><b>Phase :</b> Trie de Haut en Bas (En Avance ↔ En Retard).</li>
                </ul>
            </div>
            """, """
            <div style="background-color: #fff7ed; padding: 15px; border-radius: 8px; border-left: 5px solid #f97316; height: 100%;">
                <h3 style="color: #9a3412; margin: 0 0 10px 0; font-size: 20px;">2. The Solution</h3>
                <p style="font-size: 14px; color: #334155; margin: 0;"><b>Sorting:</b> We apply gradients to "sort" signals:</p>
                <ul style="font-size: 13px; color: #334155; padding-left: 20px; margin-top: 5px;">
                    <li style="margin-bottom: 5px;"><b>Frequency:</b> Sorts Left to Right (Low ↔ High).</li>
                    <li><b>Phase:</b> Sorts Top to Bottom (Early ↔ Late).</li>
                </ul>
            </div>
            """, """
            <div style="background-color: #fff7ed; padding: 15px; border-radius: 8px; border-left: 5px solid #f97316; height: 100%;">
                <h3 style="color: #9a3412; margin: 0 0 10px 0; font-size: 20px;">2. Die Lösung</h3>
                <p style="font-size: 14px; color: #334155; margin: 0;"><b>Das Sortieren:</b> Wir wenden Gradienten an, um Signale zu "sortieren":</p>
                <ul style="font-size: 13px; color: #334155; padding-left: 20px; margin-top: 5px;">
                    <li style="margin-bottom: 5px;"><b>Frequenz:</b> Sortiert von Links nach Rechts (Tief ↔ Hoch).</li>
                    <li><b>Phase:</b> Sortiert von Oben nach Unten (Früh ↔ Spät).</li>
                </ul>
            </div>
            """), unsafe_allow_html=True)
            
        with c_txt3:
            st.markdown(T("""
            <div style="background-color: #f0fdf4; padding: 15px; border-radius: 8px; border-left: 5px solid #22c55e; height: 100%;">
                <h3 style="color: #166534; margin: 0 0 10px 0; font-size: 20px;">3. La Réalité Physique</h3>
                <p style="font-size: 14px; color: #334155; margin: 0;">Pour créer l'image, la machine combine 3 axes :</p>
                <ul style="font-size: 13px; color: #334155; padding-left: 20px; margin-top: 5px;">
                    <li style="margin-bottom: 5px;"><b>Axe Z (Sélection) :</b> Isole la <b>Coupe</b> (L'épaisseur).</li>
                    <li style="margin-bottom: 5px;"><b>Axe Y (Phase) :</b> Encode les <b>Lignes</b>.</li>
                    <li><b>Axe X (Fréquence) :</b> Encode les <b>Colonnes</b>.</li>
                </ul>
            </div>
            """, """
            <div style="background-color: #f0fdf4; padding: 15px; border-radius: 8px; border-left: 5px solid #22c55e; height: 100%;">
                <h3 style="color: #166534; margin: 0 0 10px 0; font-size: 20px;">3. Physical Reality</h3>
                <p style="font-size: 14px; color: #334155; margin: 0;">To create the image, the machine combines 3 axes:</p>
                <ul style="font-size: 13px; color: #334155; padding-left: 20px; margin-top: 5px;">
                    <li style="margin-bottom: 5px;"><b>Z Axis (Selection):</b> Isolates the <b>Slice</b>.</li>
                    <li style="margin-bottom: 5px;"><b>Y Axis (Phase):</b> Encodes <b>Lines</b>.</li>
                    <li><b>X Axis (Frequency):</b> Encodes <b>Columns</b>.</li>
                </ul>
            </div>
            """, """
            <div style="background-color: #f0fdf4; padding: 15px; border-radius: 8px; border-left: 5px solid #22c55e; height: 100%;">
                <h3 style="color: #166534; margin: 0 0 10px 0; font-size: 20px;">3. Physikalische Realität</h3>
                <p style="font-size: 14px; color: #334155; margin: 0;">Um das Bild zu erzeugen, kombiniert die Maschine 3 Achsen:</p>
                <ul style="font-size: 13px; color: #334155; padding-left: 20px; margin-top: 5px;">
                    <li style="margin-bottom: 5px;"><b>Z-Achse (Schicht):</b> Isoliert die <b>Schicht</b> (Die Dicke).</li>
                    <li style="margin-bottom: 5px;"><b>Y-Achse (Phase):</b> Kodiert die <b>Zeilen</b>.</li>
                    <li><b>X-Achse (Frequenz):</b> Kodiert die <b>Spalten</b>.</li>
                </ul>
            </div>
            """), unsafe_allow_html=True)
    
    st.write("") 
    
    # RÉSUMÉ
    st.markdown(T("""
    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <span style="font-size: 24px; vertical-align: middle;">📍</span> 
        <span style="font-size: 16px; font-weight: bold; color: #0f172a; vertical-align: middle;">
            En résumé : L'IRM est une grille 3D. Z choisit la tranche de pain, Y choisit la rangée, X choisit la colonne.
        </span>
    </div>
    """, """
    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <span style="font-size: 24px; vertical-align: middle;">📍</span> 
        <span style="font-size: 16px; font-weight: bold; color: #0f172a; vertical-align: middle;">
            In summary: MRI is a 3D grid. Z chooses the bread slice, Y chooses the row, X chooses the column.
        </span>
    </div>
    """, """
    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <span style="font-size: 24px; vertical-align: middle;">📍</span> 
        <span style="font-size: 16px; font-weight: bold; color: #0f172a; vertical-align: middle;">
            Zusammenfassung: Die MRT ist ein 3D-Raster. Z wählt die Brotscheibe, Y die Zeile, X die Spalte.
        </span>
    </div>
    """), unsafe_allow_html=True)
    
    # Création des deux sous-onglets
    sub_tabs = st.tabs([T("👁️ Cycle de Codage (Visualisation)", "👁️ Encoding Cycle (Visualization)", "👁️ Kodierungszyklus (Visualisierung)"), 
                        T("🎨 Espace K (Remplissage)", "🎨 K-Space (Filling)", "🎨 K-Raum (Füllung)")])
    
    # SOUS-ONGLET 1 : CODAGE (HTML du main1.py)
    with sub_tabs[0]:
        st.markdown(T("<h3 style='color: #4f46e5; border-bottom: 2px solid #4f46e5; padding-bottom: 5px;'>🎛️ Simulateur de Codage</h3>", 
                      "<h3 style='color: #4f46e5; border-bottom: 2px solid #4f46e5; padding-bottom: 5px;'>🎛️ Encoding Simulator</h3>",
                      "<h3 style='color: #4f46e5; border-bottom: 2px solid #4f46e5; padding-bottom: 5px;'>🎛️ Kodierungs-Simulator</h3>"), unsafe_allow_html=True)
        
        st.caption(T("⚠️ **Note clinique :** Si le déphasage entre deux pixels dépasse 180° (limite physique), la machine ne sait plus d'où vient le signal, ce qui crée un artéfact de repliement (Aliasing).", 
                     "⚠️ **Clinical note:** If the phase shift between two pixels exceeds 180° (physical limit), the machine loses spatial tracking, creating an aliasing artifact.",
                     "⚠️ **Klinischer Hinweis:** Wenn die Phasenverschiebung zwischen zwei Pixeln 180° überschreitet (physikalische Grenze), weiß das Gerät nicht mehr, woher das Signal kommt, was ein Aliasing-Artefakt (Einfaltung) erzeugt."))
        
        # HTML/JS AVEC TRADUCTION INJECTÉE (FR, EN, DE)
        components.html(T("""<!DOCTYPE html><html><head><style>body{margin:0;padding:5px;font-family:sans-serif;} .box{display:flex;gap:15px;} .ctrl{width:220px;padding:10px;background:#f9f9f9;border:1px solid #ccc;border-radius:8px;} canvas{border:1px solid #ccc;background:#f8f9fa;border-radius:8px;} input{width:100%;} label{font-size:11px;font-weight:bold;display:block;} button{width:100%;padding:8px;background:#4f46e5;color:white;border:none;border-radius:4px;cursor:pointer;}</style></head><body><div class='box'><div class='ctrl'><h4>Codage</h4><label>Freq</label><input type='range' id='f' min='-100' max='100' value='0'><br><label>Phase</label><input type='range' id='p' min='-100' max='100' value='0'><br><label>Coupe</label><input type='range' id='z' min='-100' max='100' value='0'><br><label>Matrice</label><input type='range' id='g' min='5' max='20' value='12'><br><button onclick='rst()'>Reset</button></div><div><canvas id='c1' width='350' height='350'></canvas><canvas id='c2' width='80' height='350'></canvas></div></div><script>const c1=document.getElementById('c1');const x=c1.getContext('2d');const c2=document.getElementById('c2');const z=c2.getContext('2d');const sf=document.getElementById('f');const sp=document.getElementById('p');const sz=document.getElementById('z');const sg=document.getElementById('g');const pd=30;function arrow(ctx,x,y,a,s){const l=s*0.35;ctx.save();ctx.translate(x,y);ctx.rotate(a);ctx.beginPath();ctx.moveTo(-l,0);ctx.lineTo(l,0);ctx.lineTo(l-6,-6);ctx.moveTo(l,0);ctx.lineTo(l-6,6);ctx.strokeStyle='white';ctx.lineWidth=1.5;ctx.stroke();ctx.restore();} function draw(){x.clearRect(0,0,350,350);z.clearRect(0,0,80,350);const fv=parseFloat(sf.value);const pv=parseFloat(sp.value);const zv=parseFloat(sz.value);const gs=parseInt(sg.value);const st=(350-2*pd)/gs;const h=(pd*0.8)*(fv/100);x.fillStyle='rgba(255,0,0,0.3)';if(fv!=0){x.beginPath();x.moveTo(pd,pd/2);x.lineTo(pd,pd/2-h);x.lineTo(350-pd,pd/2+h);x.lineTo(350-pd,pd/2);x.fill();}const w=(pd*0.8)*(pv/100);x.fillStyle='rgba(0,255,0,0.3)';if(pv!=0){x.beginPath();x.moveTo(350-pd/2,pd);x.lineTo(350-pd/2-w,pd);x.lineTo(350-pd/2+w,350-pd);x.lineTo(350-pd/2,350-pd);x.fill();} for(let i=0;i<gs;i++){for(let j=0;j<gs;j++){const cx=pd+i*st+st/2;const cy=pd+j*st+st/2;const ph=(i-gs/2)*(fv/100)*3+(j-gs/2)*(pv/100)*3 - Math.PI/2;const cph=(j-gs/2)*(pv/100);x.strokeStyle='black';x.beginPath();x.arc(cx,cy,st*0.4,0,6.28);x.fillStyle='#94a3b8';x.fill();if(cph>0.01)x.fillStyle='rgba(255,255,0,0.5)';if(cph<-0.01)x.fillStyle='rgba(0,0,255,0.5)';x.fill();arrow(x,cx,cy,ph,st*0.6);}}const yz=175-(zv/100)*150;const gr=z.createLinearGradient(0,0,0,350);gr.addColorStop(0,'red');gr.addColorStop(1,'blue');z.fillStyle=gr;z.fillRect(10,10,20,330);z.strokeStyle='black';z.lineWidth=3;z.beginPath();z.moveTo(10,yz);z.lineTo(70,yz);z.stroke();z.fillStyle='black';z.fillText('Z',35,yz-5);} [sf,sp,sz,sg].forEach(s=>s.addEventListener('input',draw));function rst(){sf.value=0;sp.value=0;sz.value=0;sg.value=12;draw();}draw();</script></body></html>""", 
        """<!DOCTYPE html><html><head><style>body{margin:0;padding:5px;font-family:sans-serif;} .box{display:flex;gap:15px;} .ctrl{width:220px;padding:10px;background:#f9f9f9;border:1px solid #ccc;border-radius:8px;} canvas{border:1px solid #ccc;background:#f8f9fa;border-radius:8px;} input{width:100%;} label{font-size:11px;font-weight:bold;display:block;} button{width:100%;padding:8px;background:#4f46e5;color:white;border:none;border-radius:4px;cursor:pointer;}</style></head><body><div class='box'><div class='ctrl'><h4>Encoding</h4><label>Freq</label><input type='range' id='f' min='-100' max='100' value='0'><br><label>Phase</label><input type='range' id='p' min='-100' max='100' value='0'><br><label>Slice</label><input type='range' id='z' min='-100' max='100' value='0'><br><label>Matrix</label><input type='range' id='g' min='5' max='20' value='12'><br><button onclick='rst()'>Reset</button></div><div><canvas id='c1' width='350' height='350'></canvas><canvas id='c2' width='80' height='350'></canvas></div></div><script>const c1=document.getElementById('c1');const x=c1.getContext('2d');const c2=document.getElementById('c2');const z=c2.getContext('2d');const sf=document.getElementById('f');const sp=document.getElementById('p');const sz=document.getElementById('z');const sg=document.getElementById('g');const pd=30;function arrow(ctx,x,y,a,s){const l=s*0.35;ctx.save();ctx.translate(x,y);ctx.rotate(a);ctx.beginPath();ctx.moveTo(-l,0);ctx.lineTo(l,0);ctx.lineTo(l-6,-6);ctx.moveTo(l,0);ctx.lineTo(l-6,6);ctx.strokeStyle='white';ctx.lineWidth=1.5;ctx.stroke();ctx.restore();} function draw(){x.clearRect(0,0,350,350);z.clearRect(0,0,80,350);const fv=parseFloat(sf.value);const pv=parseFloat(sp.value);const zv=parseFloat(sz.value);const gs=parseInt(sg.value);const st=(350-2*pd)/gs;const h=(pd*0.8)*(fv/100);x.fillStyle='rgba(255,0,0,0.3)';if(fv!=0){x.beginPath();x.moveTo(pd,pd/2);x.lineTo(pd,pd/2-h);x.lineTo(350-pd,pd/2+h);x.lineTo(350-pd,pd/2);x.fill();}const w=(pd*0.8)*(pv/100);x.fillStyle='rgba(0,255,0,0.3)';if(pv!=0){x.beginPath();x.moveTo(350-pd/2,pd);x.lineTo(350-pd/2-w,pd);x.lineTo(350-pd/2+w,350-pd);x.lineTo(350-pd/2,350-pd);x.fill();} for(let i=0;i<gs;i++){for(let j=0;j<gs;j++){const cx=pd+i*st+st/2;const cy=pd+j*st+st/2;const ph=(i-gs/2)*(fv/100)*3+(j-gs/2)*(pv/100)*3 - Math.PI/2;const cph=(j-gs/2)*(pv/100);x.strokeStyle='black';x.beginPath();x.arc(cx,cy,st*0.4,0,6.28);x.fillStyle='#94a3b8';x.fill();if(cph>0.01)x.fillStyle='rgba(255,255,0,0.5)';if(cph<-0.01)x.fillStyle='rgba(0,0,255,0.5)';x.fill();arrow(x,cx,cy,ph,st*0.6);}}const yz=175-(zv/100)*150;const gr=z.createLinearGradient(0,0,0,350);gr.addColorStop(0,'red');gr.addColorStop(1,'blue');z.fillStyle=gr;z.fillRect(10,10,20,330);z.strokeStyle='black';z.lineWidth=3;z.beginPath();z.moveTo(10,yz);z.lineTo(70,yz);z.stroke();z.fillStyle='black';z.fillText('Z',35,yz-5);} [sf,sp,sz,sg].forEach(s=>s.addEventListener('input',draw));function rst(){sf.value=0;sp.value=0;sz.value=0;sg.value=12;draw();}draw();</script></body></html>""",
        """<!DOCTYPE html><html><head><style>body{margin:0;padding:5px;font-family:sans-serif;} .box{display:flex;gap:15px;} .ctrl{width:220px;padding:10px;background:#f9f9f9;border:1px solid #ccc;border-radius:8px;} canvas{border:1px solid #ccc;background:#f8f9fa;border-radius:8px;} input{width:100%;} label{font-size:11px;font-weight:bold;display:block;} button{width:100%;padding:8px;background:#4f46e5;color:white;border:none;border-radius:4px;cursor:pointer;}</style></head><body><div class='box'><div class='ctrl'><h4>Kodierung</h4><label>Freq</label><input type='range' id='f' min='-100' max='100' value='0'><br><label>Phase</label><input type='range' id='p' min='-100' max='100' value='0'><br><label>Schicht</label><input type='range' id='z' min='-100' max='100' value='0'><br><label>Matrix</label><input type='range' id='g' min='5' max='20' value='12'><br><button onclick='rst()'>Reset</button></div><div><canvas id='c1' width='350' height='350'></canvas><canvas id='c2' width='80' height='350'></canvas></div></div><script>const c1=document.getElementById('c1');const x=c1.getContext('2d');const c2=document.getElementById('c2');const z=c2.getContext('2d');const sf=document.getElementById('f');const sp=document.getElementById('p');const sz=document.getElementById('z');const sg=document.getElementById('g');const pd=30;function arrow(ctx,x,y,a,s){const l=s*0.35;ctx.save();ctx.translate(x,y);ctx.rotate(a);ctx.beginPath();ctx.moveTo(-l,0);ctx.lineTo(l,0);ctx.lineTo(l-6,-6);ctx.moveTo(l,0);ctx.lineTo(l-6,6);ctx.strokeStyle='white';ctx.lineWidth=1.5;ctx.stroke();ctx.restore();} function draw(){x.clearRect(0,0,350,350);z.clearRect(0,0,80,350);const fv=parseFloat(sf.value);const pv=parseFloat(sp.value);const zv=parseFloat(sz.value);const gs=parseInt(sg.value);const st=(350-2*pd)/gs;const h=(pd*0.8)*(fv/100);x.fillStyle='rgba(255,0,0,0.3)';if(fv!=0){x.beginPath();x.moveTo(pd,pd/2);x.lineTo(pd,pd/2-h);x.lineTo(350-pd,pd/2+h);x.lineTo(350-pd,pd/2);x.fill();}const w=(pd*0.8)*(pv/100);x.fillStyle='rgba(0,255,0,0.3)';if(pv!=0){x.beginPath();x.moveTo(350-pd/2,pd);x.lineTo(350-pd/2-w,pd);x.lineTo(350-pd/2+w,350-pd);x.lineTo(350-pd/2,350-pd);x.fill();} for(let i=0;i<gs;i++){for(let j=0;j<gs;j++){const cx=pd+i*st+st/2;const cy=pd+j*st+st/2;const ph=(i-gs/2)*(fv/100)*3+(j-gs/2)*(pv/100)*3 - Math.PI/2;const cph=(j-gs/2)*(pv/100);x.strokeStyle='black';x.beginPath();x.arc(cx,cy,st*0.4,0,6.28);x.fillStyle='#94a3b8';x.fill();if(cph>0.01)x.fillStyle='rgba(255,255,0,0.5)';if(cph<-0.01)x.fillStyle='rgba(0,0,255,0.5)';x.fill();arrow(x,cx,cy,ph,st*0.6);}}const yz=175-(zv/100)*150;const gr=z.createLinearGradient(0,0,0,350);gr.addColorStop(0,'red');gr.addColorStop(1,'blue');z.fillStyle=gr;z.fillRect(10,10,20,330);z.strokeStyle='black';z.lineWidth=3;z.beginPath();z.moveTo(10,yz);z.lineTo(70,yz);z.stroke();z.fillStyle='black';z.fillText('Z',35,yz-5);} [sf,sp,sz,sg].forEach(s=>s.addEventListener('input',draw));function rst(){sf.value=0;sp.value=0;sz.value=0;sg.value=12;draw();}draw();</script></body></html>"""), height=450)
        
        st.divider()
        st.markdown(T("<h3 style='background-color: #e0e7ff; padding: 10px; border-radius: 5px; color: #3730a3;'>🧠 Synthèse : Gradient & Espace K</h3>", 
                      "<h3 style='background-color: #e0e7ff; padding: 10px; border-radius: 5px; color: #3730a3;'>🧠 Summary: Gradient & K-Space</h3>",
                      "<h3 style='background-color: #e0e7ff; padding: 10px; border-radius: 5px; color: #3730a3;'>🧠 Zusammenfassung: Gradient & K-Raum</h3>"), unsafe_allow_html=True)
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.info(T("**1. Gradient Faible (Lignes Centrales)**\n* Faible Déphasage = Signal Fort.\n* Contraste de l'image.", 
                      "**1. Low Gradient (Center Lines)**\n* Low Dephasing = Strong Signal.\n* Image Contrast.",
                      "**1. Schwacher Gradient (Zentrale Zeilen)**\n* Geringe Dephasierung = Starkes Signal.\n* Bildkontrast."))
        with col_c2:
            st.error(T("**2. Gradient Fort (Lignes Périphériques)**\n* Fort Déphasage = Détails fins.\n* Résolution spatiale.", 
                       "**2. High Gradient (Peripheral Lines)**\n* High Dephasing = Fine Details.\n* Spatial Resolution.",
                       "**2. Starker Gradient (Periphere Zeilen)**\n* Starke Dephasierung = Feine Details.\n* Räumliche Auflösung."))

    # SOUS-ONGLET 2 : ESPACE K
    with sub_tabs[1]:
        st.markdown(T("<h3 style='color: #db2777; border-bottom: 2px solid #db2777; padding-bottom: 5px;'>🖼️ Remplissage & Reconstruction</h3>", 
                      "<h3 style='color: #db2777; border-bottom: 2px solid #db2777; padding-bottom: 5px;'>🖼️ Filling & Reconstruction</h3>",
                      "<h3 style='color: #db2777; border-bottom: 2px solid #db2777; padding-bottom: 5px;'>🖼️ Füllung & Rekonstruktion</h3>"), unsafe_allow_html=True)
        
        col_k1, col_k2 = st.columns([1, 1])
        with col_k1:
            lbl_mode = T("Ordre de Remplissage", "Filling Order", "Füllreihenfolge")
            opt_lin = T("Linéaire cartésien (Haut -> Bas)", "Linear Cartesian (Top -> Bottom)", "Linear kartesisch (Oben -> Unten)")
            opt_rad = T("Radial / Spirale (Proportionnel)", "Radial / Spiral (Proportional)", "Radial / Spiral (Proportional)")
            
            fill_mode = st.radio(lbl_mode, [opt_lin, opt_rad], key=f"k_mode_{current_reset_id}")
            acq_pct = st.slider(T("Progression (%)", "Progress (%)", "Fortschritt (%)"), 0, 100, 10, step=1, key=f"k_pct_{current_reset_id}")
            
            st.divider()
            
            if turbo > 1:
                st.markdown(T(f"""
                <div style="background-color: #fce7f3; padding: 10px; border-radius: 5px; border-left: 5px solid #db2777; margin-bottom: 10px;">
                    <h4 style="margin:0; color: #831843;">🚅 Rangement des {turbo} Échos (Ky)</h4>
                </div>
                """, f"""
                <div style="background-color: #fce7f3; padding: 10px; border-radius: 5px; border-left: 5px solid #db2777; margin-bottom: 10px;">
                    <h4 style="margin:0; color: #831843;">🚅 Ordering of {turbo} Echoes (Ky)</h4>
                </div>
                """, f"""
                <div style="background-color: #fce7f3; padding: 10px; border-radius: 5px; border-left: 5px solid #db2777; margin-bottom: 10px;">
                    <h4 style="margin:0; color: #831843;">🚅 Anordnung der {turbo} Echos (Ky)</h4>
                </div>
                """), unsafe_allow_html=True)

                st.info(T(f"TE effectif : **{int(te)} ms** | Facteur Turbo : **{turbo}**", 
                          f"Effective TE: **{int(te)} ms** | Turbo Factor: **{turbo}**",
                          f"Effektive TE: **{int(te)} ms** | Turbofaktor: **{turbo}**"))
                
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
                fig_tse = Figure(figsize=(5, 4))
                ax = fig_tse.subplots()
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
                ax.text(-0.05, 0.5, T("CENTRE (K=0)", "CENTER (K=0)", "ZENTRUM (K=0)"), ha='right', va='center', fontweight='bold')
                ax.annotate("", xy=(-0.02, 0.4), xytext=(-0.02, 0.6), arrowprops=dict(arrowstyle="-", color="black", lw=2))
                st.pyplot(fig_tse)
                
            else:
                st.markdown(T(f"#### 🐢 Acquisition Standard (1 Écho/TR)", f"#### 🐢 Standard Acquisition (1 Echo/TR)", f"#### 🐢 Standard-Akquisition (1 Echo/TR)"))
                st.info(f"TE : **{int(te)} ms**")
                fig_tse = Figure(figsize=(5, 4)); ax = fig_tse.subplots()
                n_disp_lines = 24; y_h = 1.0 / n_disp_lines; color = plt.cm.jet(0)
                for i in range(n_disp_lines):
                    rect = patches.Rectangle((0, 1.0 - (i + 1) * y_h), 1, y_h, linewidth=0.5, edgecolor='white', facecolor=color)
                    ax.add_patch(rect)
                ax.text(0.5, 0.5, T(f"ECHO 1 (TE={int(te)}ms)\nAppliqué à chaque ligne", f"ECHO 1 (TE={int(te)}ms)\nApplied to each line", f"ECHO 1 (TE={int(te)}ms)\nAuf jede Zeile angewendet"), ha='center', va='center', color='white', fontweight='bold', fontsize=12, path_effects=[path_effects.withStroke(linewidth=3, foreground='black')])
                ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
                st.pyplot(fig_tse)
                
        
        with col_k2:
            mask_k = np.zeros((S, S))
            
            if fill_mode == opt_lin: 
                lines_to_fill = int(S * (acq_pct / 100.0))
                mask_k[0:lines_to_fill, :] = 1
            else: 
                rayon_actuel = (acq_pct / 100.0) * 1.5 
                mask_k[D <= rayon_actuel] = 1
                
            # Calcul local sécurisé pour l'espace K
            f_local = np.fft.fftshift(np.fft.fft2(final_complex))
            kspace_masked = f_local * mask_k
            img_rec = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_masked)))
            
            # --- CORRECTION RÉSOLUTION FANTÔME ---
            max_val = np.max(img_rec)
            if max_val > 0:
                img_rec = img_rec / max_val
            # -------------------------------------
            
            fig_k = Figure(figsize=(4, 4))
            ax_k = fig_k.subplots()
            
            st.image(img_rec, clamp=True, width=300, caption=T("Reconstruction", "Reconstruction", "Rekonstruktion"))

# [TAB 3 : SIGNAUX]
elif module_actif == liste_modules[3]:
    st.markdown(T("### 📊 Comparaison des Signaux", "### 📊 Signal Comparison", "### 📊 Signalvergleich"))
    
    # =========================================================
    # 1. SIGNAL RÉSULTANT (SÉQUENCE EN COURS)
    # =========================================================
    st.markdown(T("#### 🎛️ Signal résultant de la séquence en cours", 
                  "#### 🎛️ Resulting signal of the current sequence", 
                  "#### 🎛️ Resultierendes Signal der aktuellen Sequenz"))
                  
    c_sig_left, c_sig_center, c_sig_right = st.columns([1, 2, 1])
    with c_sig_center:
        fig_sig = Figure(figsize=(5, 3)); ax_sig = fig_sig.subplots()
        
        vals_bar = [v_lcr, v_gm, v_wm, v_fat]
        noms = [T("EAU", "WATER", "WASSER"), T("SG", "GM", "GS"), T("SB", "WM", "WS"), T("GRAISSE", "FAT", "FETT")]
        
        if show_stroke: 
            vals_bar.append(v_stroke)
            noms.append(T("AVC", "STROKE", "SCHLAGANFALL"))
            
        cols_sig = ['cyan', 'dimgray', 'lightgray', 'orange', 'red'] if show_stroke else ['cyan', 'dimgray', 'lightgray', 'orange']
        bars = ax_sig.bar(noms, vals_bar, color=cols_sig, edgecolor='black')
        
        ax_sig.set_ylim(0, 1.3)
        ax_sig.grid(True, axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig_sig); 

    # =========================================================
    # 2. TABLEAU DE PONDÉRATION (CLINIQUE & RIGOUREUX)
    # =========================================================
    st.divider()
    st.markdown(T("### 📚 Règles de Pondération", "### 📚 Weighting Rules", "### 📚 Wichtungsregeln"))
    
    th_pond = T("Pondération", "Weighting", "Wichtung")
    th_se = T("Séquences Spin Écho", "Spin Echo Sequences", "Spin-Echo-Sequenzen")
    th_gre = T("Écho de Gradient", "Gradient Echo", "Gradientenecho")
    th_tr = "TR"
    th_te = "TE"
    th_fa = T("Angle (α)", "Angle (α)", "Winkel (α)")
    
    t1_name = T("T1", "T1", "T1")
    t2_name = T("T2", "T2", "T2")
    t2star_name = T("T2*", "T2*", "T2*")
    pd_name = T("Densité de Protons (DP)", "Proton Density (PD)", "Protonendichte (PD)")
    
    court = T("Court", "Short", "Kurz")
    long_val = T("Long", "Long", "Lang")
    grand = T("Grand", "Large", "Groß")
    petit = T("Petit", "Small", "Klein")
    
    st.markdown(f"""
    <style>
    .table-weighting {{width: 100%; border-collapse: collapse; font-size: 15px; margin-top: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);}}
    .table-weighting th {{border: 1px solid #cbd5e1; padding: 10px; text-align: center; color: white;}}
    .table-weighting td {{padding: 12px; text-align: center; border: 1px solid #cbd5e1; color: #334155;}}
    .table-weighting tr:nth-child(even) td {{background-color: #f8fafc;}}
    .table-weighting tr:nth-child(odd) td {{background-color: #ffffff;}}
    .w-title {{font-weight: bold; color: #1e293b !important; font-size: 16px; background-color: #eff6ff !important;}}
    .val-highlight {{font-weight: bold; color: #0f172a; font-size: 16px;}}
    .bg-main {{background-color: #3b82f6;}}
    .bg-se {{background-color: #1e40af;}}
    .bg-gre {{background-color: #0369a1;}}
    .na-cell {{background-color: #e2e8f0 !important; color: #94a3b8 !important; font-size: 18px; font-weight: bold;}}
    </style>
    
    <table class="table-weighting">
        <tr>
            <th rowspan="2" class="bg-main" style="vertical-align: middle; font-size:16px;">{th_pond}</th>
            <th colspan="2" class="bg-se" style="font-size:16px; letter-spacing: 1px;">{th_se}</th>
            <th colspan="2" class="bg-gre" style="font-size:16px; letter-spacing: 1px;">{th_gre}</th>
        </tr>
        <tr>
            <th class="bg-se" style="opacity: 0.9;">{th_tr}</th>
            <th class="bg-se" style="opacity: 0.9;">{th_te}</th>
            <th class="bg-gre" style="opacity: 0.9;">{th_fa}</th>
            <th class="bg-gre" style="opacity: 0.9;">{th_te}</th>
        </tr>
        <tr>
            <td class="w-title">{t1_name}</td>
            <td><span class="val-highlight">{court}</span><br>(400 - 700 ms)</td>
            <td><span class="val-highlight">{court}</span><br>(10 - 20 ms)</td>
            <td><span class="val-highlight">{grand}</span><br>(&gt; 70°)</td>
            <td><span class="val-highlight">{court}</span><br>(&lt; 5 ms)</td>
        </tr>
        <tr>
            <td class="w-title">{t2_name}</td>
            <td><span class="val-highlight">{long_val}</span><br>(&gt; 4000 ms)</td>
            <td><span class="val-highlight">{long_val}</span><br>(&gt; 50 ms)</td>
            <td class="na-cell">-</td>
            <td class="na-cell">-</td>
        </tr>
        <tr>
            <td class="w-title">{t2star_name}</td>
            <td class="na-cell">-</td>
            <td class="na-cell">-</td>
            <td><span class="val-highlight">{petit}</span><br>(5 - 20°)</td>
            <td><span class="val-highlight">{long_val}</span><br>(&gt; 20 ms)</td>
        </tr>
        <tr>
            <td class="w-title">{pd_name}</td>
            <td><span class="val-highlight">{long_val}</span><br>(&gt; 1800 ms)</td>
            <td><span class="val-highlight">{court}</span><br>(10 - 30 ms)</td>
            <td><span class="val-highlight">{petit}</span><br>(5 - 20°)</td>
            <td><span class="val-highlight">{court}</span><br>(&lt; 5 ms)</td>
        </tr>
    </table>
    """, unsafe_allow_html=True)

    # =========================================================
    # 3. REPRÉSENTATION QUANTITATIVE (DENSITÉ DE PROTONS) COMPLÈTE
    # =========================================================
    st.divider()
    st.markdown(T("#### 💧 Densité de Protons (M0)", 
                  "#### 💧 Proton Density (M0)", 
                  "#### 💧 Protonendichte (M0)"))
    st.caption(T("Quantité d'hydrogène disponible par tissu (Base absolue du signal avant pondération).", 
                 "Available hydrogen quantity per tissue (Absolute signal baseline before weighting).",
                 "Verfügbare Wasserstoffmenge pro Gewebe (Absolute Signalbasis vor Wichtung)."))
    
    # Dessin des "Éprouvettes" via Matplotlib
    fig_dp = Figure(figsize=(14, 4.5)); ax_dp = fig_dp.subplots() 
    
    # Liste EXHAUSTIVE issue de l'image de référence
    dp_labels = [
        T("Urine", "Urine", "Urin"),
        T("LCR", "CSF", "Liquor"),
        T("Subst. Grise", "Gray Matter", "Graue Substanz"),
        T("Rein", "Kidney", "Niere"),
        T("Muscle / Rate", "Muscle / Spleen", "Muskel / Milz"),
        T("Subst. Blanche", "White Matter", "Weiße Substanz"),
        T("Foie", "Liver", "Leber"),
        T("Ligaments\n& Tendons", "Ligaments\n& Tendons", "Bänder\n& Sehnen"),
        T("Os Cortical", "Cortical Bone", "Kortikalis"),
        T("Dent", "Tooth", "Zahn"),
        T("Air", "Air", "Luft")
    ]
    
    dp_values = [99, 97, 84, 81, 79, 72, 71, 64, 12, 6, 0.5] 
    
    # Couleurs adaptées à chaque tissu
    dp_colors = [
        '#f1c40f',  # Urine (Jaune)
        'cyan',     # LCR (Cyan)
        'dimgray',  # SG (Gris foncé)
        '#a0522d',  # Rein (Sienne/Marron clair)
        '#c0392b',  # Muscle/Rate (Rouge foncé)
        'lightgray',# SB (Gris clair)
        '#8e44ad',  # Foie (Violet/Bordeaux)
        '#f5b041',  # Ligaments (Orange/Beige)
        '#bdc3c7',  # Os Cortical (Argent)
        'white',    # Dent (Blanc)
        '#34495e'   # Air (Bleu/Gris très foncé)
    ]
    
    for i, (val, color, label) in enumerate(zip(dp_values, dp_colors, dp_labels)):
        x_center = i * 1.8 
        
        # 1. Le liquide (Rectangle rempli)
        ax_dp.add_patch(patches.Rectangle((x_center - 0.3, 0), 0.6, val, fill=True, color=color, alpha=0.8))
        
        if val > 1:
            # 2. Effet 3D : Ellipses
            ax_dp.add_patch(patches.Ellipse((x_center, val), 0.6, 4, fill=True, color=color, alpha=0.9))
            ax_dp.add_patch(patches.Ellipse((x_center, val), 0.6, 4, fill=False, edgecolor='black', lw=1, alpha=0.5))
        
        # 3. Effet 3D : Reflet (uniquement s'il y a assez de liquide)
        if val > 5:
            ax_dp.add_patch(patches.Rectangle((x_center - 0.25, 3), 0.1, val-6, fill=True, color='white', alpha=0.4))
        
        # 4. Le tube en verre
        ax_dp.plot([x_center - 0.3, x_center - 0.3], [0, 110], color='black', lw=2)
        ax_dp.plot([x_center + 0.3, x_center + 0.3], [0, 110], color='black', lw=2)
        
        # 5. Fond du tube
        ax_dp.add_patch(patches.Arc((x_center, 0), 0.6, 6, theta1=180, theta2=360, edgecolor='black', lw=2))
        
        # 6. Texte pourcentage (Formatage spécial pour l'Air à 0.5%)
        text_col = color if color not in ['lightgray', 'cyan', '#f1c40f', 'white', '#bdc3c7'] else 'black'
        if color == '#34495e': text_col = 'gray'
        
        txt_val = f"{val}%" if val >= 1 else f"{val:.1f}%"
        ax_dp.text(x_center, val + 5, txt_val, ha='center', fontweight='bold', fontsize=11, color=text_col)
        
        # 7. LÉGENDE TRILINGUE (Inclinée pour prendre moins de place)
        ax_dp.text(x_center, -8, label, ha='right', va='top', fontweight='bold', fontsize=11, color='#2c3e50', rotation=35)
        
    ax_dp.set_xlim(-1, len(dp_labels)*1.8 - 0.5)
    ax_dp.set_ylim(-45, 125) # Marge négative plus importante pour les textes inclinés
    ax_dp.axis('off')
    
    fig_dp.tight_layout()
    st.pyplot(fig_dp); 
# [TAB 5 : ANATOMIE] - RÉINITIALISÉ ET MODIFIÉ
elif module_actif == liste_modules[4]:
    st.header(T("🧠 Exploration Anatomique (Physique Avancée)", "🧠 Anatomical Exploration (Advanced Physics)", "🧠 Anatomische Untersuchung (Fortgeschrittene Physik)"))
    
    try:
        if HAS_NILEARN and processor.ready:
            c1, c2 = st.columns([1, 3])
            dims = processor.get_dims()
            
            with c1:
                # SÉLECTEUR DE PLAN
                plane = st.radio(
                    T("Plan de Coupe", "Slice Plane", "Schnittebene"), 
                    [T("Plan Axial", "Axial Plane", "Axialebene"), T("Plan Sagittal", "Sagittal Plane", "Sagittalebene"), T("Plan Coronal", "Coronal Plane", "Koronarebene")], 
                    key="or_298"
                )
                
                # SLIDER POSITION (Z/X/Y)
                if "Axial" in plane: 
                    idx = st.slider("Z", 0, dims[2]-1, 90, key=f"sl_{current_reset_id}"); ax='z'
                elif "Sagittal" in plane: 
                    idx = st.slider("X", 0, dims[0]-1, 90, key=f"sl_{current_reset_id}"); ax='x'
                else: 
                    idx = st.slider("Y", 0, dims[1]-1, 100, key=f"sl_{current_reset_id}"); ax='y'
                
                st.divider()
                
                # --- LOGIQUE DE FENÊTRAGE (NIVEAUX DICOM CLINIQUE) ---
                DICOM_FACTOR = 2000.0
                def_win_f, def_lev_f = (0.74, 0.55)
                
                # Règle pour la diffusion b>=1000
                try:
                    if is_dwi and (b_value >= 1000):
                        def_win_f = 0.90
                except:
                    pass
                
                def_win_dicom = int(def_win_f * DICOM_FACTOR)
                def_lev_dicom = int(def_lev_f * DICOM_FACTOR)
                
                key_suffix = f"{current_reset_id}_{is_dwi}_{b_value if is_dwi else 0}"
                
                window_dicom = st.slider(T("Fenêtre (W)", "Window (W)", "Fenster (W)"), 10, 6000, def_win_dicom, 10, key=f"wn_{key_suffix}")
                level_dicom = st.slider(T("Niveau (L)", "Level (L)", "Niveau (L)"), 0, 3000, def_lev_dicom, 10, key=f"lv_{key_suffix}")
                
                # Reconversion silencieuse pour le moteur d'image
                window = window_dicom / DICOM_FACTOR
                level = level_dicom / DICOM_FACTOR
                
                # Bouton pour afficher les légendes
                show_interactive_legends = st.toggle(T("Afficher les Légendes", "Show Legends", "Legenden anzeigen"), value=False)
                
                if show_interactive_legends:
                    st.info(T(
                        "💡 **Note de l'Atlas Harvard-Oxford :** Le survol interactif est optimisé pour l'identification des structures macroscopiques majeures. Les zones ultra-fines, les vaisseaux fins ou les lisières de pixels de transition peuvent ne pas afficher de nomenclature.",
                        "💡 **Harvard-Oxford Atlas Note:** Interactive hovering is optimized for identifying major macroscopic structures. Ultra-fine zones, small vessels, or transitional pixel edges may not display nomenclature.",
                        "💡 **Hinweis zum Harvard-Oxford-Atlas:** Das interaktive Einblenden ist für la makroskopische Hauptstrukturen optimiert. Sehr feine Bereiche, kleine Gefäße oder Pixelränder zeigen möglicherweise keine Bezeichnungen an."
                    ))
                # ----------------------------------

            with c2:
                # 1. Préparation des poids (Signaux calculés plus haut)
                w_vals = {'csf':v_lcr, 'gm':v_gm, 'wm':v_wm, 'fat':v_fat}
                if show_stroke: 
                    w_vals['wm'] = w_vals['wm'] * 0.9 + v_stroke * 0.1
                
                seq_type_arg = 'dwi' if is_dwi else ('gre' if is_gre else None)
                
                # 2. Appel CORRECT à la méthode du processeur
                try:
                    img_raw = processor.get_slice(ax, idx, w_vals, seq_type=seq_type_arg, te=te, tr=tr, fa=flip_angle, b_val=b_value, adc_mode=show_adc_map, with_stroke=show_stroke)
                except Exception as e:
                    img_raw = None
                    st.error(f"Erreur interne lors du calcul anatomique : {e}")
                
                if img_raw is not None:
                    # Inversion D/G (Convention Radiologique) sauf en Sagittal
                    if ax != 'x':
                        img_raw = np.fliplr(img_raw)
                                                  
                    try:
                        if is_dwi and not show_adc_map and b_value > 0:
                            noise = np.random.normal(0, (b_value/1000.0)*0.05, img_raw.shape)
                            img_noisy = img_raw + noise
                            img_raw = gaussian_filter(img_noisy, sigma=0.8)
                            img_raw = np.clip(img_raw, 0, 2.0)
                    except:
                        pass # Sécurité si is_dwi n'est pas défini

                    # APPLICATION FENÊTRAGE
                    img_display = utils.apply_window_level(img_raw, window, level)
                    
                    if show_interactive_legends:
                        labels_map = processor.get_anatomical_labels(ax, idx)
                        try:
                            # Inversion des légendes
                            if ax != 'x':
                                labels_map = np.fliplr(labels_map)
                        except:
                            pass
                            
                        fig_anat = px.imshow(img_display, color_continuous_scale='gray', zmin=0, zmax=1)
                        fig_anat.update_traces(customdata=labels_map, hovertemplate="<b>%{customdata}</b><extra></extra>")
                        
                        fig_anat.update_layout(
                            margin=dict(l=0, r=0, t=0, b=0), 
                            coloraxis_showscale=False, 
                            width=600, 
                            height=600,
                            xaxis=dict(visible=False), 
                            yaxis=dict(visible=False), 
                            plot_bgcolor='black',
                            paper_bgcolor='black'
                        )
                        st.plotly_chart(fig_anat, config={'displayModeBar': False})
                    else:
                        st.image(img_display, clamp=True, width=600)
                else:
                    st.error("🚨 Le moteur d'anatomie n'a pas pu générer l'image. Vérifiez les paramètres (TR/TE) ou la sélection de séquence.")
        else:
            st.warning("⚠️ Module Anatomie non disponible.")
            
    except Exception as e:
        st.error(f"Erreur générale Anatomie : {e}")
# [TAB 6 : PHYSIQUE]
elif module_actif == liste_modules[5]:
    st.header(T("📈 Physique", "📈 Physics", "📈 Physik"))
    
    # Définition des tissus et couleurs
    tists = [cst.T_FAT, cst.T_WM, cst.T_GM, cst.T_LCR]
    cols = ['orange', 'lightgray', 'dimgray', 'cyan'] 
    labels_tr = [T('Graisse', 'Fat', 'Fett'), T('SB', 'WM', 'WS'), T('SG', 'GM', 'GS'), T('LCR', 'CSF', 'Liquor')]
    
    # Ajout pathologie si active
    if show_stroke: 
        tists.append(cst.T_STROKE)
        cols.append('red') 
        labels_tr.append(T('AVC', 'Stroke', 'Schlaganfall'))
    
    # =========================================================
    # GRAPHIQUE 1 : RELAXATION LONGITUDINALE (T1)
    # =========================================================
    fig_t1 = Figure(figsize=(10, 3))
    gs = fig_t1.add_gridspec(1, 2, width_ratios=[30, 1], wspace=0.05)
    ax_t1 = fig_t1.add_subplot(gs[0])
    ax_bar = fig_t1.add_subplot(gs[1])
    
    x_t = np.linspace(0, 4000, 500)
    
    ax_t1.set_title(T("Relaxation Longitudinale (T1)", "Longitudinal Relaxation (T1)", "Longitudinale Relaxation (T1)"))
    
    # LOGIQUE DE TRACÉ SELON SÉQUENCE
    if is_gre:
        start_mz = np.cos(np.radians(flip_angle))
        ax_t1.set_ylim(-0.1, 1.1)
        for t, col, lbl in zip(tists, cols, labels_tr): 
            mz = 1 - (1 - start_mz) * np.exp(-x_t / t['T1'])
            ax_t1.plot(x_t, mz, color=col, label=lbl)
        ax_t1.axhline(start_mz, color='gray', linestyle=':', label=f"Mz(0)")
        
    elif is_ir:
        ax_t1.set_ylim(-1.1, 1.1)
        ax_t1.axhline(0, color='black')
        for t, col, lbl in zip(tists, cols, labels_tr): 
            mz = 1 - 2 * np.exp(-x_t / t['T1'])
            ax_t1.plot(x_t, mz, color=col, label=lbl)
        ax_t1.axvline(x=ti, color='green', linestyle='--', label='TI')
        
    else:
        ax_t1.set_ylim(0, 1.1)
        for t, col, lbl in zip(tists, cols, labels_tr): 
            mz = 1 - np.exp(-x_t / t['T1'])
            ax_t1.plot(x_t, mz, color=col, label=lbl)
        
    ax_t1.axvline(x=tr_effective, color='red', linestyle='--', label=T('TR Réel', 'Real TR', 'Tatsächliche TR'))
    
    gradient = np.linspace(1, 0, 256).reshape(-1, 1)
    if is_ir: gradient = np.abs(np.linspace(1, -1, 256)).reshape(-1, 1)
    
    ax_bar.imshow(gradient, aspect='auto', cmap='gray', extent=[0, 1, ax_t1.get_ylim()[0], ax_t1.get_ylim()[1]])
    ax_bar.axis('off')
    
    # --- CORRECTION LÉGENDE (Anti-Doublons) ---
    handles, labels_ax = ax_t1.get_legend_handles_labels()
    by_label = dict(zip(labels_ax, handles)) 
    ax_t1.legend(by_label.values(), by_label.keys(), loc='best')

    st.pyplot(fig_t1)
    
    
    # =========================================================
    # GRAPHIQUE 2 : RELAXATION TRANSVERSALE (T2)
    # =========================================================
    fig_t2 = Figure(figsize=(10, 3))
    gs2 = fig_t2.add_gridspec(1, 2, width_ratios=[30, 1], wspace=0.05)
    ax_t2 = fig_t2.add_subplot(gs2[0])
    ax_bar2 = fig_t2.add_subplot(gs2[1])
    
    x_te = np.linspace(0, 500, 300)
    
    ax_t2.set_title(T("Relaxation Transversale (T2/T2*)", "Transverse Relaxation (T2/T2*)", "Transversale Relaxation (T2/T2*)"))
    
    for t, col, lbl in zip(tists, cols, labels_tr): 
        val_t2 = t['T2s'] if is_gre else t['T2']
        label_suffix = " (T2*)" if is_gre else ""
        
        mxy = np.exp(-x_te / val_t2)
        ax_t2.plot(x_te, mxy, color=col, label=f"{lbl}{label_suffix}")
        
    ax_t2.axvline(x=te, color='red', linestyle='--', label=T('TE Eff', 'Eff TE', 'Eff. TE'))
    
    ax_t2.legend()
    
    gradient_t2 = np.linspace(1, 0, 256).reshape(-1, 1)
    ax_bar2.imshow(gradient_t2, aspect='auto', cmap='gray', extent=[0, 1, 0, 1.0])
    ax_bar2.axis('off')
    
    st.pyplot(fig_t2)
    

# [TAB 7 : CHRONOGRAMME]
elif module_actif == liste_modules[6]:
    st.header(T("⚡ Chronogramme", "⚡ Timing Diagram", "⚡ Sequenzdiagramm"))
    t_90 = 10
    
    if is_gre:
        st.subheader(T(f"Séquence : Écho de Gradient (Angle {flip_angle}°)", f"Sequence: Gradient Echo (Angle {flip_angle}°)", f"Sequenz: Gradientenecho (Winkel {flip_angle}°)"))
        t_max = max(tr + 40, te + 50); t = np.linspace(0, t_max, 2000); rf_sigma = 0.5; grad_width = 3.0
        fig = Figure(figsize=(10, 8)); axs = fig.subplots(5, 1, sharex=True, gridspec_kw={'hspace': 0.3})
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
        axs[4].plot(t, sig, color='navy'); axs[4].set_ylabel(T("Signal", "Signal", "Signal")); axs[4].axvline(x=t_read, color='red', linestyle='--'); axs[4].text(t_read, 1.1, f"TE={te:.0f}ms", color='red', ha='center')
        st.pyplot(fig); 
    else:
        is_turbo = turbo > 1; t_90 = 10
        if is_dwi: st.subheader(T("Séquence : Diffusion (DWI - SE EPI)", "Sequence: Diffusion (DWI - SE EPI)", "Sequenz: Diffusion (DWI - SE EPI)"))
        elif is_turbo: st.subheader(T(f"Séquence : Turbo Spin Écho (TSE) - Facteur {turbo}", f"Sequence: Turbo Spin Echo (TSE) - Factor {turbo}", f"Sequenz: Turbo-Spin-Echo (TSE) - Faktor {turbo}"))
        else: st.subheader(T("Séquence : Spin Écho (SE)", "Sequence: Spin Echo (SE)", "Sequenz: Spin-Echo (SE)"))
        
        if not is_turbo: echo_times = [t_90 + te]; t_180s = [t_90 + (te/2)]; es_disp = te; t_max = max(200, t_90 + te + 50)
        else: echo_times = [t_90 + (i+1)*es for i in range(turbo)]; t_180s = []; es_disp = es; t_max = max(200, echo_times[-1] + 50)
        
        t = np.linspace(0, t_max, 2000); rf_sigma = 0.5; grad_width = max(1.5, es_disp * 0.2); t_180s = []; 
        for i in range(turbo): t_p = t_90 + (i * es) + (es/2); t_180s.append(t_p)
        
        fig = Figure(figsize=(10, 8)); axs = fig.subplots(5, 1, sharex=True, gridspec_kw={'hspace': 0.3})
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
                if label == "BF": 
                    txt_bf = T("BF", "Ctr", "Zentrum")
                    axs[2].text(t_code, height+0.1, txt_bf, color=col_lbl, ha='center', fontsize=9, weight='bold')
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
                     axs[4].text(t_e, amp+0.3, T("TE eff", "Eff TE", "Eff. TE"), ha='center', color='red', fontweight='bold', fontsize=10)
                     axs[4].axvline(x=t_e, color='red', linestyle='--', alpha=0.5)
        axs[4].plot(t, sig, color='navy', linewidth=1.5); axs[4].set_ylabel(T("Signal", "Signal", "Signal"))
        st.pyplot(fig); 

# [TAB 7 : ARTEFACTS]
elif module_actif == liste_modules[7]:
    st.header(T("☣️ Laboratoire d'Artefacts", "☣️ Artifact Laboratory", "☣️ Artefakt-Labor"))
    
    st.info(T("💡 Astuce : Utilisez le menu de gauche (Géométrie / Options) pour modifier la console (FOV, Bande Passante, Facteur Turbo, TE) afin de déclencher les artéfacts.", 
              "💡 Tip: Use the left menu (Geometry / Options) to modify the console (FOV, Bandwidth, Turbo Factor, TE) to trigger artifacts.",
              "💡 Tipp: Verwenden Sie das left Menü (Geometrie / Optionen), um die Konsole zu ändern, um Artefakte auszulösen."))
              
    col_ctrl, col_view = st.columns([1, 2])
    
    opt_aliasing = T("Repliement (Aliasing)", "Aliasing", "Einfaltung (Aliasing)")
    opt_shift    = T("Décalage Chimique", "Chemical Shift", "Chemische Verschiebung")
    opt_trunc    = T("Troncature (Gibbs)", "Truncation (Gibbs)", "Abschneidung (Gibbs)")
    opt_motion   = T("Mouvement", "Motion", "Bewegung")
    opt_zipper   = "Zipper"
    opt_blurring = T("Flou T2 (T2 Blurring)", "T2 Blurring", "T2-Unschärfe")

    options_list = [opt_aliasing, opt_shift, opt_trunc, opt_motion, opt_zipper, opt_blurring]

    with col_ctrl:
        st.markdown("#### " + T("Choix de l'Artefact", "Artifact Selection", "Artefaktauswahl"))
        artefact_type = st.radio(T("Sélectionnez :", "Select:", "Auswählen:"), options_list, key="art_main_radio")

    # --- 1. ALIASING ---
    if artefact_type == opt_aliasing:
        with col_ctrl:
            st.info(f"{T('FOV Actuel', 'Current FOV', 'Aktuelles FOV')} : **{fov} mm** ({T('Objet', 'Object', 'Objekt')} : 230 mm)")
            if fov < 230: st.error(T("⚠️ Aliasing Actif !", "⚠️ Aliasing Active!", "⚠️ Aliasing Aktiv!"))
            else: st.success(T("Pas d'aliasing", "No Aliasing", "Kein Aliasing"))

        with col_view:
            img_art = final.copy()
            if fov < 230:
                ratio = fov / 230.0
                shift_w = int(S * (1 - ratio) / 2)
                shift_w = max(0, min(shift_w, S // 2))
                if shift_w > 0:
                    top = img_art[0:shift_w, :]; bot = img_art[S-shift_w:S, :]
                    img_art = img_art.copy()
                    img_art[S-shift_w:S, :] += top; img_art[0:shift_w, :] += bot
            st.image(np.clip(img_art / 1.3, 0, 1), caption=T("Image avec Repliement", "Aliased Image", "Bild mit Einfaltung"), use_container_width=True)

    # --- 2. DÉCALAGE CHIMIQUE ---
    elif artefact_type == opt_shift:
        with col_ctrl: 
            st.info(f"BW : **{bw} Hz/px**")
            st.markdown(T("Le décalage de la graisse est inversement proportionnel à la bande passante (BW).", "Spatial shift of fat is inversely proportional to bandwidth (BW).", "Die Verschiebung von Fett ist umgekehrt proportional zur Bandbreite (BW)."))
            if bw != 220: st.warning(T("⚠️ Décalage visible.", "⚠️ Visible shift.", "⚠️ Sichtbare Verschiebung."))
            else: st.success(T("✅ Bande passante de référence.", "✅ Reference bandwidth.", "✅ Referenzbandbreite."))
            
        with col_view:
            demo_fat = img_fat.copy()
            if np.max(demo_fat) == 0: demo_fat[(D >= 0.80) & (D < 0.95)] = 0.8 

            px_shift = 0.0 if bw == 220 else 220.0 / float(bw) 
            factor_visu = 5.0 
            effective_shift = -px_shift * factor_visu if bw > 220 else px_shift * factor_visu

            sh = shift(demo_fat, [0, effective_shift], mode='constant', cval=0.0)
            res = img_water + sh
            
            cap = T("Pas de décalage", "No shift", "Keine Verschiebung") if effective_shift == 0 else f"Shift : {abs(effective_shift):.1f} px"
            st.image(np.clip(res / 1.3, 0, 1), caption=cap, use_container_width=True)

    # --- 3. TRONCATURE (GIBBS) ---
    elif artefact_type == opt_trunc:
        with col_ctrl: 
            sm = st.select_slider(T("Matrice Simulée", "Simulated Matrix", "Simulierte Matrix"), [32, 64, 128, 256], 64)
            if sm <= 64: st.warning(T("Oscillations visibles (Gibbs)", "Ringing visible (Gibbs)", "Sichtbare Oszillationen (Gibbs)"))
        with col_view:
            ft = np.fft.fftshift(np.fft.fft2(final))
            c = S // 2; k = sm // 2
            m = np.zeros_like(ft); m[c-k:c+k, c-k:c+k] = 1
            res = np.abs(np.fft.ifft2(np.fft.ifftshift(ft * m)))
            st.image(np.clip(res / 1.3, 0, 1), caption=T("Artefact de Troncature", "Truncation Artifact", "Abschneide-Artefakt"), use_container_width=True)

    # --- 4. MOUVEMENT ---
    elif artefact_type == opt_motion:
        with col_ctrl: 
            it = st.slider(T("Intensité Mouvement", "Motion Intensity", "Bewegungsintensität"), 0.0, 5.0, 0.5)
        with col_view:
            ft = np.fft.fftshift(np.fft.fft2(final))
            if it > 0:
                ph = np.random.normal(0, it, S)
                for i in range(S): ft[i, :] *= np.exp(1j * ph[i])
            res = np.abs(np.fft.ifft2(np.fft.ifftshift(ft)))
            st.image(np.clip(res / 1.3, 0, 1), caption=T("Fantômes de Mouvement", "Motion Ghosts", "Bewegungs-Geisterbilder"), use_container_width=True)

    # --- 5. ZIPPER ---
    elif artefact_type == opt_zipper:
        with col_ctrl: 
            fr = st.slider(T("Fréquence (Ligne)", "Frequency (Line)", "Frequenz (Zeile)"), 0, S-1, S//2)
            vol = st.slider("Volume / Amplitude", 0, 100, 10)
        with col_view:
            ft = np.fft.fftshift(np.fft.fft2(final))
            if vol > 0:
                ns = np.random.normal(0, vol, S) + (vol * 5)
                alt = np.array([1 if i%2==0 else -1 for i in range(S)])
                ft[:, fr] += ns * alt * 50
            res = np.abs(np.fft.ifft2(np.fft.ifftshift(ft)))
            st.image(np.clip(res / 1.3, 0, 1), caption="Zipper Artifact", use_container_width=True)

    # --- 6. FLOU T2 (T2 BLURRING) ---
    elif artefact_type == opt_blurring:
        with col_ctrl:
            st.info(T(f"📊 **Statut du Train d'Échos :**\n* Facteur Turbo (ETL) : **{turbo}**\n* TE Effectif : **{te} ms**\n* Echo Spacing (ES) : **{es} ms**",
                      f"📊 **Echo Train Status:**\n* Turbo Factor (ETL): **{turbo}**\n* Effective TE: **{te} ms**\n* Echo Spacing (ES): **{es} ms**"))
            
            # Calcul de la position de l'écho par rapport au centre du train
            effective_index = max(1, min(turbo, round(te / max(es, 1.0))))
            center_index = (turbo + 1) / 2
            dist_from_center = abs(effective_index - center_index)
            
            st.markdown(f"🎯 **{T('Position du TE effectif :', 'Effective TE Position:')}** "
                        f"{T(f'Écho n°**{effective_index}** sur un train de **{turbo}**.', f'Echo n°**{effective_index}** out of a train of **{turbo}**.')}")
            
            if turbo <= 1:
                st.info(T("💡 Le mode Spin Écho conventionnel (Turbo = 1) ne produit pas de flou T2. Augmentez le Facteur Turbo dans le menu de gauche.",
                          "💡 Conventional Spin Echo mode (Turbo = 1) does not produce T2 blurring. Increase Turbo Factor in the left menu."))
                sigma_val = 0.0
            else:
                # Le flou dépend EXCLUSIVEMENT du décentrage du train d'échos
                sigma_val = dist_from_center * 0.4
                
                if dist_from_center == 0:
                    st.success(T("✨ **Centrage Parfait !** Le TE effectif est au centre exact du train d'échos. L'effet de flou est totalement corrigé : netteté maximale.",
                                 "✨ **Perfect Centering!** The effective TE is exactly at the center of the echo train. The blur effect is completely corrected: maximum sharpness."))
                elif dist_from_center <= 2:
                    st.warning(T("⚠️ **Léger décentrage :** Le TE effectif s'éloigne du centre. Un flou directionnel discret apparaît.",
                                 "⚠️ **Slight off-centering:** The effective TE moves away from the center. A discrete directional blur appears."))
                else:
                    st.error(T("🛑 **Décentrage Important :** Le TE effectif est mal positionné dans le train d'échos. Le flou sur l'axe des phases est majeur.",
                               "🛑 **Significant off-centering:** The effective TE is poorly positioned in the echo train. Blur on the phase axis is major."))

        with col_view:
            # 🎯 RECTIFICATION CHIRURGICALE :
            # On combine l'eau et la graisse disponibles globalement pour reconstruire img_sim sans bruit
            img_art = (img_water + img_fat).copy()
            
            if turbo > 1 and sigma_val > 0:
                # Application du filtre gaussien directionnel strict (Axe vertical Y uniquement)
                img_art = gaussian_filter(img_art, sigma=[sigma_val, 0.0])
                
            # Normalisation graphique purement locale pour préserver le contraste
            max_val = np.max(img_art)
            if max_val > 0:
                img_art = img_art / max_val
                
            st.image(np.clip(img_art, 0, 1), 
                     caption=f"{T('Rendu pur de l‘artéfact de Flou T2 — Axe vertical de codage des phases uniquement (sans bruit)', 'Pure T2 Blurring Render — Vertical Phase Encoding Axis Only (Noise-free)')} (Sigma={sigma_val:.2f})", 
                     use_container_width=True)

elif module_actif == liste_modules[8]:
    st.header(T("🚀 Imagerie Parallèle (PI)", "🚀 Parallel Imaging (PI)", "🚀 Parallele Bildgebung (PI)"))
    
    show_meta = st.checkbox(
        T("👁️ Afficher le Concept (Analogie de la Fenêtre)", "👁️ Show Concept (Window Analogy)", "👁️ Konzept anzeigen (Fenster-Analogie)"), 
        value=False
    )

    if show_meta:
        st.markdown("### " + T("1. Analogie de la Fenêtre (Interactive)", "1. The Window Analogy (Interactive)", "1. Die Fenster-Analogie (Interaktiv)"))
        
        opt_left = T("Gauche", "Left", "Links")
        opt_center = T("Centre", "Center", "Mitte")
        opt_right = T("Droite", "Right", "Rechts")
        opt_all = T("👁️ Vue Simultanée (Tous)", "👁️ Simultaneous View (All)", "👁️ Gleichzeitige Ansicht (Alle)")
        
        pos_obs = st.select_slider(
            T("📍 Votre Position devant la fenêtre :", "📍 Your Position in front of the window:", "📍 Ihre Position vor dem Fenster:"), 
            options=[opt_left, opt_center, opt_right, opt_all], 
            value=opt_center, 
            key=f"pos_fenetre_{current_reset_id}"
        )

        wall_g_rect = patches.Rectangle((-10, 8), 40, 1, color='lightgray')
        wall_d_rect = patches.Rectangle((70, 8), 40, 1, color='lightgray')
        window_frame_x = [30, 70]

        txt_wall_g = T("Mur G", "Wall L", "Wand L")
        txt_wall_d = T("Mur D", "Wall R", "Wand R")
        txt_machine = T("Machine", "Scanner", "Scanner")
        txt_window = T("Fenêtre", "Window", "Fenster")
        txt_you = T("VOUS", "YOU", "SIE")

        # --- CAS : VUE SIMULTANÉE ---
        if pos_obs == opt_all:
            c_simu1, c_simu2 = st.columns([2, 1])
            with c_simu1:
                fig_all = Figure(figsize=(8, 4)); ax_all = fig_all.subplots()
                ax_all.add_patch(wall_g_rect)
                ax_all.text(10, 8.5, txt_wall_g, color='black', ha='center', va='center', fontsize=8)
                ax_all.add_patch(wall_d_rect)
                ax_all.text(90, 8.5, txt_wall_d, color='black', ha='center', va='center', fontsize=8)
                
                ax_all.add_patch(patches.Circle((50, 8.5), 3, color='purple'))
                ax_all.text(50, 6.5, txt_machine, color='purple', ha='center', va='top', fontweight='bold')
                
                ax_all.plot(window_frame_x, [4, 4], color='black', linewidth=3)
                ax_all.text(25, 4, txt_window, ha='right', va='center')
                
                ax_all.plot(30, 0, 'o', color='blue', markersize=10)
                ax_all.add_patch(plt.Polygon([[30, 0], [35, 9], [100, 9]], color='blue', alpha=0.1))
                ax_all.plot(70, 0, 'o', color='orange', markersize=10)
                ax_all.add_patch(plt.Polygon([[70, 0], [0, 9], [65, 9]], color='orange', alpha=0.1))
                ax_all.plot(50, 0, 'o', color='green', markersize=10)
                ax_all.add_patch(plt.Polygon([[50, 0], [35, 9], [65, 9]], color='green', alpha=0.1))
                
                ax_all.set_xlim(-10, 110); ax_all.set_ylim(-3, 10); ax_all.axis('off')
                ax_all.set_title(T("Les 3 observateurs regardent (Fenêtre étroite)", "The 3 observers watching (Narrow window)", "Die 3 Beobachter schauen (Schmales Fenster)"), fontsize=10)
                st.pyplot(fig_all)

            with c_simu2:
                st.markdown("**👀 " + T("Résultat Reconstitué :", "Reconstructed Result:", "Rekonstruiertes Ergebnis:") + "**")
                st.markdown("_" + T("Somme des 3 vues = Image Totale", "Sum of 3 views = Total Image", "Summe der 3 Ansichten = Gesamtbild") + "_")
                
                fig_full = Figure(figsize=(4, 4)); ax_f = fig_full.subplots()
                ax_f.set_xlim(0, 100); ax_f.set_ylim(0, 100); ax_f.axis('off')
                ax_f.add_patch(patches.Rectangle((0,0), 100, 100, color='whitesmoke'))
                
                ax_f.add_patch(patches.Rectangle((0, 0), 30, 100, color='gray'))
                ax_f.text(15, 50, txt_wall_g.upper(), color='white', ha='center', va='center', rotation=90, fontweight='bold')
                
                ax_f.add_patch(patches.Circle((50, 50), 15, color='purple'))
                
                ax_f.add_patch(patches.Rectangle((70, 0), 30, 100, color='gray'))
                ax_f.text(85, 50, txt_wall_d.upper(), color='white', ha='center', va='center', rotation=90, fontweight='bold')
                
                ax_f.set_title(T("Votre Rétine (Synthèse)", "Your Retina (Synthesis)", "Ihre Netzhaut (Synthese)"), fontsize=9)
                st.pyplot(fig_full)

        # --- CAS : VUE UNIQUE ---
        else:
            c_simu1, c_simu2 = st.columns([2, 1])
            with c_simu1:
                fig_analog = Figure(figsize=(8, 4)); ax_an = fig_analog.subplots()
                ax_an.add_patch(wall_g_rect)
                ax_an.text(10, 8.5, txt_wall_g, color='black', ha='center', va='center', fontsize=8)
                ax_an.add_patch(wall_d_rect)
                ax_an.text(90, 8.5, txt_wall_d, color='black', ha='center', va='center', fontsize=8)
                
                ax_an.add_patch(patches.Circle((50, 8.5), 3, color='purple'))
                ax_an.text(50, 6.5, txt_machine, color='purple', ha='center', va='top', fontweight='bold')
                
                ax_an.plot(window_frame_x, [4, 4], color='black', linewidth=3)
                ax_an.text(25, 4, txt_window, ha='right', va='center')
                
                ax_an.plot(30, 0, 'o', color='lightgray', alpha=0.5)
                ax_an.plot(70, 0, 'o', color='lightgray', alpha=0.5)
                ax_an.plot(50, 0, 'o', color='lightgray', alpha=0.5) 
                
                if pos_obs == opt_left:
                    user_x = 30; col_u = "blue"
                    poly_pts = [[30, 0], [35, 9], [100, 9]]
                    msg_view = T("Je vois surtout le Mur de Droite", "I mostly see the Right Wall", "Ich sehe hauptsächlich die rechte Wand")
                elif pos_obs == opt_right:
                    user_x = 70; col_u = "orange"
                    poly_pts = [[70, 0], [0, 9], [65, 9]]
                    msg_view = T("Je vois surtout le Mur de Gauche", "I mostly see the Left Wall", "Ich sehe hauptsächlich die linke Wand")
                else: 
                    user_x = 50; col_u = "green"
                    poly_pts = [[50, 0], [35, 9], [65, 9]]
                    msg_view = T("Je vois uniquement la Machine (Centre)", "I only see the Machine (Center)", "Ich sehe nur den Scanner (Mitte)")

                ax_an.plot(user_x, 0, 'o', color=col_u, markersize=12)
                ax_an.text(user_x, -1.5, txt_you, color=col_u, ha='center', va='top', fontweight='bold')
                ax_an.add_patch(plt.Polygon(poly_pts, color=col_u, alpha=0.2))
                
                ax_an.set_xlim(-10, 110); ax_an.set_ylim(-3, 10); ax_an.axis('off')
                ax_an.set_title(T("Vue de dessus : ", "Top View: ", "Draufsicht: ") + pos_obs, fontsize=10)
                st.pyplot(fig_analog)

            with c_simu2:
                st.markdown("**👀 " + T("Ce que vous voyez :", "What you see:", "Was Sie sehen:") + "**")
                st.markdown(f"_{msg_view}_")
                
                fig_view = Figure(figsize=(3, 3)); ax_v = fig_view.subplots()
                ax_v.set_xlim(0, 100); ax_v.set_ylim(0, 100); ax_v.axis('off')
                ax_v.add_patch(patches.Rectangle((0,0), 100, 100, color='whitesmoke'))
                
                if pos_obs == opt_center:
                    ax_v.add_patch(patches.Circle((50, 50), 25, color='purple'))
                elif pos_obs == opt_left:
                    ax_v.add_patch(patches.Rectangle((50, 0), 50, 100, color='gray'))
                    ax_v.text(75, 50, txt_wall_d.upper(), color='white', ha='center', va='center', rotation=90, fontweight='bold')
                    ax_v.add_patch(patches.Circle((20, 50), 15, color='purple'))
                elif pos_obs == opt_right:
                    ax_v.add_patch(patches.Rectangle((0, 0), 50, 100, color='gray'))
                    ax_v.text(25, 50, txt_wall_g.upper(), color='white', ha='center', va='center', rotation=90, fontweight='bold')
                    ax_v.add_patch(patches.Circle((80, 50), 15, color='purple'))
                
                ax_v.set_title(T("Votre Rétine", "Your Retina", "Ihre Netzhaut"), fontsize=9)
                st.pyplot(fig_view)
        
        st.divider()

    # --- 1. PRINCIPE & LIGNES ---
    st.markdown("#### " + T("1. Principe & Sous-échantillonnage", "1. Principle & Undersampling", "1. Prinzip & Unterabtastung"))
    col_pi_info, col_pi_ctrl = st.columns([2, 1])
    
    with col_pi_info:
        st.info(T(
            f"**Gain de Temps :** L'acquisition est accélérée par un facteur **R = {ipat_factor}**.",
            f"**Time Saving:** Acquisition is accelerated by factor **R = {ipat_factor}**.",
            f"**Zeitersparnis:** Die Akquisition wird um den Faktor **R = {ipat_factor}** beschleunigt."
        ))
        st.warning(T(
            r"**Coût (Pénalité SNR) :** Le signal diminue de $\sqrt{R}$.",
            r"**Cost (SNR Penalty):** Signal decreases by $\sqrt{R}$.",
            r"**Kosten (SNR-Strafe):** Das Signal sinkt um $\sqrt{R}$."
        ))
        
        st.markdown("**" + T("Visualisation de l'acquisition des lignes", "Line acquisition visualization", "Visualisierung der Zeilenakquisition") + f" (R={ipat_factor}) :**")
        fig_lines = Figure(figsize=(10, 1.5)); ax_lines = fig_lines.subplots()
        for i in range(25): 
            if i % ipat_factor == 0:
                ax_lines.vlines(i, 0, 1, colors='green', linewidth=3)
            else:
                ax_lines.vlines(i, 0, 1, colors='red', linestyles='dotted', linewidth=1.5)
        
        ax_lines.set_xlim(-1, 26); ax_lines.set_ylim(0, 1); ax_lines.axis('off')
        
        txt_legend = T("Vert = Acquise\nRouge = Sautée", "Green = Acquired\nRed = Skipped", "Grün = Akquiriert\nRot = Übersprungen")
        ax_lines.text(26, 0.5, txt_legend, va='center', fontsize=9)
        st.pyplot(fig_lines)
        

    with col_pi_ctrl:
        if ipat_factor == 1: 
            st.error(T("⚠️ Accélération désactivée (R=1).", "⚠️ Acceleration disabled (R=1).", "⚠️ Beschleunigung deaktiviert (R=1)."))
        else: 
            st.success(T(f"✅ Accélération Active (R={ipat_factor})", f"✅ Acceleration Active (R={ipat_factor})", f"✅ Beschleunigung aktiv (R={ipat_factor})"))

    st.divider()

    # --- 2. ANTENNES (PROFILS) ---
    st.markdown("#### " + T('2. Les "Yeux" de la Machine (Profils de Sensibilité)', '2. Machine "Eyes" (Sensitivity Profiles)', '2. Die "Augen" der Maschine (Spulenprofile)'))
    
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    h, w = final.shape
    sigma_coil = h / 2.5
    centers = [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)]
    
    titles = [
        T("Antenne 1 (HG)", "Coil 1 (TL)", "Spule 1 (OL)"), 
        T("Antenne 2 (HD)", "Coil 2 (TR)", "Spule 2 (OR)"), 
        T("Antenne 3 (BG)", "Coil 3 (BL)", "Spule 3 (UL)"), 
        T("Antenne 4 (BD)", "Coil 4 (BR)", "Spule 4 (UR)")
    ]
    cols = [col_c1, col_c2, col_c3, col_c4]
    
    part_imgs = []
    
    for i, (cy, cx) in enumerate(centers):
        sens = generate_sensitivity_map((h,w), h*cy, w*cx, sigma_coil)
        part_img = final * sens
        part_imgs.append(part_img)
        cols[i].image(part_img, caption=titles[i], clamp=True, use_container_width=True)
        
        fig_s = Figure(figsize=(2, 2)); ax_s = fig_s.subplots()
        ax_s.imshow(sens, cmap='jet', vmin=0, vmax=1); ax_s.axis('off')
        cols[i].pyplot(fig_s); 

    # --- 3. RECONSTRUCTION ---
    st.divider()
    st.markdown("#### " + T("3. Résultat : Rempliement vs Reconstruction", "3. Result: Aliasing vs Reconstruction", "3. Ergebnis: Einfaltung vs. Rekonstruktion") + f" (R={ipat_factor})")
    
    c_res1, c_res2 = st.columns(2)
    
    rss_img = np.sqrt(sum(img**2 for img in part_imgs))
    
    if ipat_factor > 1:
        shift_amount = int(h / ipat_factor)
        img_aliased = (final + np.roll(final, shift_amount, axis=0)) / 2.0
        
        noise_factor = np.sqrt(ipat_factor) * 1.5
        safe_snr = snr_val if 'snr_val' in locals() else 50.0
        
        added_noise = np.random.normal(0, (5.0/(safe_snr+20.0)) * noise_factor, (h, w))
        img_reconstructed = np.clip(rss_img + added_noise, 0, 1.3)
        
        c_res1.image(
            img_aliased, 
            caption=T("Image Brute (Repliée/Aliasing)", "Raw Image (Aliased)", "Rohbild (Eingefaltet/Aliasing)"), 
            clamp=True, 
            use_container_width=True
        )
        c_res2.image(
            img_reconstructed, 
            caption=T("Image Reconstruite (Dépliée via SENSE/GRAPPA)", "Reconstructed Image (Unfolded via SENSE/GRAPPA)", "Rekonstruiertes Bild (Entfaltet via SENSE/GRAPPA)"), 
            clamp=True, 
            use_container_width=True
        )
        
        c_res2.caption(T(
            f"⚠️ Notez l'augmentation du bruit (Grain) due au facteur R={ipat_factor} (SNR divisé par √{ipat_factor}).",
            f"⚠️ Note the noise increase (Grain) due to factor R={ipat_factor} (SNR divided by √{ipat_factor}).",
            f"⚠️ Beachten Sie das erhöhte Rauschen (Körnung) aufgrund des Faktors R={ipat_factor} (SNR geteilt durch √{ipat_factor})."
        ))
    else:
        c_res1.image(
            final, 
            caption=T("Image de Référence (R=1)", "Reference Image (R=1)", "Referenzbild (R=1)"), 
            clamp=True, 
            use_container_width=True
        )
        c_res2.image(
            rss_img, 
            caption=T("Combinaison des 4 signaux (Somme Quadratique)", "Combination of 4 signals (Sum of Squares)", "Kombination der 4 Signale (Quadratsumme)"), 
            clamp=True, 
            use_container_width=True
        )

elif module_actif == liste_modules[8]:
    st.header(T("🚀 Imagerie Parallèle (PI)", "🚀 Parallel Imaging (PI)"))
    
    # --- 0. BOUTON CACHE ET METAPHORE ---
    show_meta = st.checkbox(
        T("👁️ Afficher le Concept (Analogie de la Fenêtre)", "👁️ Show Concept (Window Analogy)"), 
        value=False
    )

    if show_meta:
        st.markdown(f"### {T('1. Analogie de la Fenêtre (Interactive)', '1. The Window Analogy (Interactive)')}")
        
        # Définition des options traduites pour le slider
        opt_left = T("Gauche", "Left")
        opt_center = T("Centre", "Center")
        opt_right = T("Droite", "Right")
        opt_all = T("👁️ Vue Simultanée (Tous)", "👁️ Simultaneous View (All)")
        
        pos_obs = st.select_slider(
            T("📍 Votre Position devant la fenêtre :", "📍 Your Position in front of the window:"), 
            options=[opt_left, opt_center, opt_right, opt_all], 
            value=opt_center, 
            key=f"pos_fenetre_{current_reset_id}"
        )

        # Éléments graphiques communs
        wall_g_rect = patches.Rectangle((-10, 8), 40, 1, color='lightgray')
        wall_d_rect = patches.Rectangle((70, 8), 40, 1, color='lightgray')
        window_frame_x = [30, 70]

        # TEXTES VARIABLES
        txt_wall_g = T("Mur G", "Wall L")
        txt_wall_d = T("Mur D", "Wall R")
        txt_machine = T("Machine", "Scanner")
        txt_window = T("Fenêtre", "Window")
        txt_you = T("VOUS", "YOU")

        # --- CAS : VUE SIMULTANÉE ---
        if pos_obs == opt_all:
            c_simu1, c_simu2 = st.columns([2, 1])
            with c_simu1:
                fig_all = Figure(figsize=(8, 4)); ax_all = fig_all.subplots()
                ax_all.add_patch(wall_g_rect)
                ax_all.text(10, 8.5, txt_wall_g, color='black', ha='center', va='center', fontsize=8)
                ax_all.add_patch(wall_d_rect)
                ax_all.text(90, 8.5, txt_wall_d, color='black', ha='center', va='center', fontsize=8)
                
                ax_all.add_patch(patches.Circle((50, 8.5), 3, color='purple'))
                ax_all.text(50, 6.5, txt_machine, color='purple', ha='center', va='top', fontweight='bold')
                
                ax_all.plot(window_frame_x, [4, 4], color='black', linewidth=3)
                ax_all.text(25, 4, txt_window, ha='right', va='center')
                
                # Les 3 observateurs
                ax_all.plot(30, 0, 'o', color='blue', markersize=10)
                ax_all.add_patch(plt.Polygon([[30, 0], [35, 9], [100, 9]], color='blue', alpha=0.1))
                ax_all.plot(70, 0, 'o', color='orange', markersize=10)
                ax_all.add_patch(plt.Polygon([[70, 0], [0, 9], [65, 9]], color='orange', alpha=0.1))
                ax_all.plot(50, 0, 'o', color='green', markersize=10)
                ax_all.add_patch(plt.Polygon([[50, 0], [35, 9], [65, 9]], color='green', alpha=0.1))
                
                ax_all.set_xlim(-10, 110); ax_all.set_ylim(-3, 10); ax_all.axis('off')
                ax_all.set_title(T("Les 3 observateurs regardent (Fenêtre étroite)", "The 3 observers watching (Narrow window)"), fontsize=10)
                st.pyplot(fig_all)

            with c_simu2:
                st.markdown(f"**👀 {T('Résultat Reconstitué :', 'Reconstructed Result:')}**")
                st.markdown(f"_{T('Somme des 3 vues = Image Totale', 'Sum of 3 views = Total Image')}_")
                
                fig_full = Figure(figsize=(4, 4)); ax_f = fig_full.subplots()
                ax_f.set_xlim(0, 100); ax_f.set_ylim(0, 100); ax_f.axis('off')
                ax_f.add_patch(patches.Rectangle((0,0), 100, 100, color='whitesmoke'))
                
                ax_f.add_patch(patches.Rectangle((0, 0), 30, 100, color='gray'))
                ax_f.text(15, 50, txt_wall_g.upper(), color='white', ha='center', va='center', rotation=90, fontweight='bold')
                
                ax_f.add_patch(patches.Circle((50, 50), 15, color='purple'))
                
                ax_f.add_patch(patches.Rectangle((70, 0), 30, 100, color='gray'))
                ax_f.text(85, 50, txt_wall_d.upper(), color='white', ha='center', va='center', rotation=90, fontweight='bold')
                
                ax_f.set_title(T("Votre Rétine (Synthèse)", "Your Retina (Synthesis)"), fontsize=9)
                st.pyplot(fig_full)

        # --- CAS : VUE UNIQUE ---
        else:
            c_simu1, c_simu2 = st.columns([2, 1])
            with c_simu1:
                fig_analog = Figure(figsize=(8, 4)); ax_an = fig_analog.subplots()
                ax_an.add_patch(wall_g_rect)
                ax_an.text(10, 8.5, txt_wall_g, color='black', ha='center', va='center', fontsize=8)
                ax_an.add_patch(wall_d_rect)
                ax_an.text(90, 8.5, txt_wall_d, color='black', ha='center', va='center', fontsize=8)
                
                ax_an.add_patch(patches.Circle((50, 8.5), 3, color='purple'))
                ax_an.text(50, 6.5, txt_machine, color='purple', ha='center', va='top', fontweight='bold')
                
                ax_an.plot(window_frame_x, [4, 4], color='black', linewidth=3)
                ax_an.text(25, 4, txt_window, ha='right', va='center')
                
                # Fantômes gris
                ax_an.plot(30, 0, 'o', color='lightgray', alpha=0.5)
                ax_an.plot(70, 0, 'o', color='lightgray', alpha=0.5)
                ax_an.plot(50, 0, 'o', color='lightgray', alpha=0.5) 
                
                # Logique position
                if pos_obs == opt_left:
                    user_x = 30; col_u = "blue"
                    poly_pts = [[30, 0], [35, 9], [100, 9]]
                    msg_view = T("Je vois surtout le Mur de Droite", "I mostly see the Right Wall")
                elif pos_obs == opt_right:
                    user_x = 70; col_u = "orange"
                    poly_pts = [[70, 0], [0, 9], [65, 9]]
                    msg_view = T("Je vois surtout le Mur de Gauche", "I mostly see the Left Wall")
                else: # Center
                    user_x = 50; col_u = "green"
                    poly_pts = [[50, 0], [35, 9], [65, 9]]
                    msg_view = T("Je vois uniquement la Machine (Centre)", "I only see the Machine (Center)")

                ax_an.plot(user_x, 0, 'o', color=col_u, markersize=12)
                ax_an.text(user_x, -1.5, txt_you, color=col_u, ha='center', va='top', fontweight='bold')
                ax_an.add_patch(plt.Polygon(poly_pts, color=col_u, alpha=0.2))
                
                ax_an.set_xlim(-10, 110); ax_an.set_ylim(-3, 10); ax_an.axis('off')
                ax_an.set_title(f"{T('Vue de dessus :', 'Top View:')} {pos_obs}", fontsize=10)
                st.pyplot(fig_analog)

            with c_simu2:
                st.markdown(f"**👀 {T('Ce que vous voyez :', 'What you see:')}**")
                st.markdown(f"_{msg_view}_")
                
                fig_view = Figure(figsize=(3, 3)); ax_v = fig_view.subplots()
                ax_v.set_xlim(0, 100); ax_v.set_ylim(0, 100); ax_v.axis('off')
                ax_v.add_patch(patches.Rectangle((0,0), 100, 100, color='whitesmoke'))
                
                if pos_obs == opt_center:
                    ax_v.add_patch(patches.Circle((50, 50), 25, color='purple'))
                elif pos_obs == opt_left:
                    ax_v.add_patch(patches.Rectangle((50, 0), 50, 100, color='gray'))
                    ax_v.text(75, 50, txt_wall_d.upper(), color='white', ha='center', va='center', rotation=90, fontweight='bold')
                    ax_v.add_patch(patches.Circle((20, 50), 15, color='purple'))
                elif pos_obs == opt_right:
                    ax_v.add_patch(patches.Rectangle((0, 0), 50, 100, color='gray'))
                    ax_v.text(25, 50, txt_wall_g.upper(), color='white', ha='center', va='center', rotation=90, fontweight='bold')
                    ax_v.add_patch(patches.Circle((80, 50), 15, color='purple'))
                
                ax_v.set_title(T("Votre Rétine", "Your Retina"), fontsize=9)
                st.pyplot(fig_view)
        
        st.divider()

    # --- 1. PRINCIPE & LIGNES ---
    st.markdown(f"#### {T('1. Principe & Sous-échantillonnage', '1. Principle & Undersampling')}")
    col_pi_info, col_pi_ctrl = st.columns([2, 1])
    
    with col_pi_info:
        st.info(T(
            f"**Gain de Temps :** L'acquisition est accélérée par un facteur **R = {ipat_factor}**.",
            f"**Time Saving:** Acquisition is accelerated by factor **R = {ipat_factor}**."
        ))
        st.warning(T(
            r"**Coût (Pénalité SNR) :** Le signal diminue de $\sqrt{R}$.",
            r"**Cost (SNR Penalty):** Signal decreases by $\sqrt{R}$."
        ))
        
        # Visualisation Lignes
        st.markdown(f"**{T('Visualisation de l\'acquisition des lignes', 'Line acquisition visualization')} (R={ipat_factor}) :**")
        fig_lines = Figure(figsize=(10, 1.5)); ax_lines = fig_lines.subplots()
        for i in range(25): 
            if i % ipat_factor == 0:
                ax_lines.vlines(i, 0, 1, colors='green', linewidth=3)
            else:
                ax_lines.vlines(i, 0, 1, colors='red', linestyles='dotted', linewidth=1.5)
        
        ax_lines.set_xlim(-1, 26); ax_lines.set_ylim(0, 1); ax_lines.axis('off')
        
        txt_legend = T("Vert = Acquise\nRouge = Sautée", "Green = Acquired\nRed = Skipped")
        ax_lines.text(26, 0.5, txt_legend, va='center', fontsize=9)
        st.pyplot(fig_lines)
        

    with col_pi_ctrl:
        if ipat_factor == 1: 
            st.error(T("⚠️ Accélération désactivée (R=1).", "⚠️ Acceleration disabled (R=1)."))
        else: 
            st.success(T(f"✅ Accélération Active (R={ipat_factor})", f"✅ Acceleration Active (R={ipat_factor})"))

    st.divider()

    # --- 2. ANTENNES (PROFILS) ---
    st.markdown(f"#### {T('2. Les \"Yeux\" de la Machine (Profils de Sensibilité)', '2. Machine \"Eyes\" (Sensitivity Profiles)')}")
    
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    h, w = final.shape
    sigma_coil = h / 2.5
    centers = [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)]
    
    # Titres traduits pour chaque antenne
    titles = [
        T("Antenne 1 (HG)", "Coil 1 (TL)"), 
        T("Antenne 2 (HD)", "Coil 2 (TR)"), 
        T("Antenne 3 (BG)", "Coil 3 (BL)"), 
        T("Antenne 4 (BD)", "Coil 4 (BR)")
    ]
    cols = [col_c1, col_c2, col_c3, col_c4]
    
    part_imgs = []
    
    for i, (cy, cx) in enumerate(centers):
        sens = generate_sensitivity_map((h,w), h*cy, w*cx, sigma_coil)
        part_img = final * sens
        part_imgs.append(part_img)
        cols[i].image(part_img, caption=titles[i], clamp=True, use_container_width=True)
        
        fig_s = Figure(figsize=(2, 2)); ax_s = fig_s.subplots()
        ax_s.imshow(sens, cmap='jet', vmin=0, vmax=1); ax_s.axis('off')
        cols[i].pyplot(fig_s); 

    # --- 3. RECONSTRUCTION ---
    st.divider()
    st.markdown(f"#### {T('3. Résultat : Rempliement vs Reconstruction', '3. Result: Aliasing vs Reconstruction')} (R={ipat_factor})")
    
    c_res1, c_res2 = st.columns(2)
    
    # Somme quadratique (RSS)
    rss_img = np.sqrt(sum(img**2 for img in part_imgs))
    
    if ipat_factor > 1:
        shift_amount = int(h / ipat_factor)
        img_aliased = (final + np.roll(final, shift_amount, axis=0)) / 2.0
        
        # Simulation bruit iPAT
        noise_factor = np.sqrt(ipat_factor) * 1.5
        # Note: on utilise snr_val qui doit venir du scope global (tab 5)
        # Si snr_val n'est pas défini, on met une valeur par défaut pour éviter le crash
        safe_snr = snr_val if 'snr_val' in locals() else 50.0
        
        added_noise = np.random.normal(0, (5.0/(safe_snr+20.0)) * noise_factor, (h, w))
        img_reconstructed = np.clip(rss_img + added_noise, 0, 1.3)
        
        c_res1.image(
            img_aliased, 
            caption=T("Image Brute (Repliée/Aliasing)", "Raw Image (Aliased)"), 
            clamp=True, 
            use_container_width=True
        )
        c_res2.image(
            img_reconstructed, 
            caption=T("Image Reconstruite (Dépliée via SENSE/GRAPPA)", "Reconstructed Image (Unfolded via SENSE/GRAPPA)"), 
            clamp=True, 
            use_container_width=True
        )
        
        c_res2.caption(T(
            f"⚠️ Notez l'augmentation du bruit (Grain) due au facteur R={ipat_factor} (SNR divisé par √{ipat_factor}).",
            f"⚠️ Note the noise increase (Grain) due to factor R={ipat_factor} (SNR divided by √{ipat_factor})."
        ))
    else:
        c_res1.image(
            final, 
            caption=T("Image de Référence (R=1)", "Reference Image (R=1)"), 
            clamp=True, 
            use_container_width=True
        )
        c_res2.image(
            rss_img, 
            caption=T("Combinaison des 4 signaux (Somme Quadratique)", "Combination of 4 signals (Sum of Squares)"), 
            clamp=True, 
            use_container_width=True
        )

elif module_actif == liste_modules[9]:
    st.header(T("🧬 Théorie de la Diffusion (DWI)", "🧬 Diffusion Theory (DWI)", "🧬 Diffusionstheorie (DWI)"))
    st.markdown(T(
        "L'imagerie de diffusion est unique car elle sonde le **mouvement microscopique** des molécules d'eau.",
        "Diffusion imaging is unique because it probes the **microscopic movement** of water molecules.",
        "Die Diffusionsbildgebung ist einzigartig, da sie die **mikroskopische Bewegung** von Wassermolekülen untersucht."
    ))
    st.divider()
    
    # --- 1. CODE RESTAURÉ (ISOTROPIE & ADC) ---
    st.subheader(T("1. Isotropie vs Anisotropie", "1. Isotropy vs Anisotropy", "1. Isotropie vs. Anisotropie"))
    
    fig_iso = Figure(figsize=(6, 2)); ax_iso = fig_iso.subplots(1, 2)
    
    # Isotropie
    ax_iso[0].set_title(T("Isotrope (LCR)", "Isotropic (CSF)", "Isotrop (Liquor)"))
    ax_iso[0].add_patch(patches.Circle((0.5, 0.5), 0.3, color='lightblue', alpha=0.3))
    ax_iso[0].text(0.5, 0.5, "H2O", ha='center', va='center', fontweight='bold')
    
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        rad = np.radians(angle)
        dx, dy = np.cos(rad)*0.25, np.sin(rad)*0.25
        ax_iso[0].arrow(0.5, 0.5, dx, dy, head_width=0.05, color='blue')
    ax_iso[0].axis('off')
    
    # Anisotropie
    ax_iso[1].set_title(T("Anisotrope (Fibre)", "Anisotropic (Fiber)", "Anisotrop (Faser)"))
    ax_iso[1].add_patch(patches.Rectangle((0.1, 0.3), 0.8, 0.05, color='orange', alpha=0.5))
    ax_iso[1].add_patch(patches.Rectangle((0.1, 0.65), 0.8, 0.05, color='orange', alpha=0.5))
    
    ax_iso[1].text(0.5, 0.8, T("Fibre Nerveuse", "Nerve Fiber", "Nervenfaser"), ha='center', color='orange')
    ax_iso[1].text(0.5, 0.5, "H2O", ha='center', va='center', fontweight='bold')
    
    ax_iso[1].arrow(0.5, 0.5, 0.3, 0, head_width=0.05, color='blue')
    ax_iso[1].arrow(0.5, 0.5, -0.3, 0, head_width=0.05, color='blue')
    ax_iso[1].arrow(0.5, 0.5, 0, 0.1, head_width=0.03, color='red', alpha=0.5)
    ax_iso[1].arrow(0.5, 0.5, 0, -0.1, head_width=0.03, color='red', alpha=0.5)
    ax_iso[1].axis('off')
    
    st.pyplot(fig_iso)
    
    
    st.divider()
    
    st.subheader(T("2. Coefficient de Diffusion Apparent (ADC)", "2. Apparent Diffusion Coefficient (ADC)", "2. Scheinbarer Diffusionskoeffizient (ADC)"))
    
    fig_adc = Figure(figsize=(8, 1.5)); ax = fig_adc.subplots(1, 2)
    
    txt_b1000 = "b=1000"
    txt_map = T("Map ADC", "ADC Map", "ADC-Karte")
    txt_dwi = "DWI"
    
    # Scenario 1 : AVC
    ax[0].set_facecolor('black')
    ax[0].axis('off')
    ax[0].set_title(T("SCÉNARIO 1 : AVC (Restriction)", "SCENARIO 1: STROKE (Restriction)", "SZENARIO 1: SCHLAGANFALL (Restriktion)"), color='lime', weight='bold', fontsize=9)
    ax[0].text(0.3, 0.8, txt_b1000, color='black', ha='center', fontsize=8, fontweight='bold')
    ax[0].text(0.7, 0.8, txt_map, color='black', ha='center', fontsize=8, fontweight='bold')
    
    ax[0].add_patch(patches.Circle((0.3, 0.5), 0.15, edgecolor='red', facecolor='white', linewidth=4)) 
    ax[0].text(0.3, 0.25, txt_dwi, color='white', ha='center', fontweight='bold', fontsize=7)
    
    ax[0].text(0.5, 0.5, "➔", color='white', fontsize=12, ha='center', va='center')
    
    ax[0].add_patch(patches.Circle((0.7, 0.5), 0.15, edgecolor='red', facecolor='black', linewidth=4)) 
    ax[0].text(0.7, 0.25, T("ADC (Noir)", "ADC (Dark)", "ADC (Dunkel)"), color='white', ha='center', fontweight='bold', fontsize=7)
    
    # Scenario 2 : LCR
    ax[1].set_facecolor('black')
    ax[1].axis('off')
    ax[1].set_title(T("SCÉNARIO 2 : LCR (Liquide)", "SCENARIO 2: CSF (Liquid)", "SZENARIO 2: LIQUOR (Flüssigkeit)"), color='red', weight='bold', fontsize=9)
    ax[1].text(0.3, 0.8, txt_b1000, color='black', ha='center', fontsize=8, fontweight='bold')
    ax[1].text(0.7, 0.8, txt_map, color='black', ha='center', fontsize=8, fontweight='bold')
    
    ax[1].add_patch(patches.Circle((0.3, 0.5), 0.15, edgecolor='red', facecolor='black', linewidth=4)) 
    ax[1].text(0.3, 0.25, txt_dwi, color='white', ha='center', fontweight='bold', fontsize=7)
    
    ax[1].text(0.5, 0.5, "➔", color='white', fontsize=12, ha='center', va='center')
    
    ax[1].add_patch(patches.Circle((0.7, 0.5), 0.15, edgecolor='red', facecolor='white', linewidth=4)) 
    ax[1].text(0.7, 0.25, T("ADC (Blanc)", "ADC (Bright)", "ADC (Hell)"), color='white', ha='center', fontweight='bold', fontsize=7)
    
    st.pyplot(fig_adc)
    
    
    st.divider()

    # --- 2. FORMULE & GRAPHIQUE (IVIM / KURTOSIS) ---
    st.subheader(T("3. Comprendre la Décroissance (Avancé)", "3. Understanding Decay (Advanced)", "3. Signalabfall verstehen (Fortgeschritten)"))
    
    st.markdown("##### " + T('La Formule de Base', 'The Basic Formula', 'Die Grundformel'))
    st.latex(r"S = S_0 \cdot e^{-b \cdot ADC}")
    
    with st.expander(T("📖 Légende de la formule (Cliquez)", "📖 Formula Legend (Click)", "📖 Formellegende (Klicken)")):
        st.markdown(T("""
        * **S** : Signal mesuré (ce qu'on voit sur l'image).
        * **S₀** : Signal de base sans diffusion (b=0, image T2 pure).
        * **e** : Exponentielle (la décroissance est rapide).
        * **b** : Facteur b (puissance du gradient de diffusion).
        * **ADC** : Coefficient de Diffusion (la mobilité de l'eau).
        """, """
        * **S**: Measured signal (what you see on the image).
        * **S₀**: Base signal without diffusion (b=0, pure T2 image).
        * **e**: Exponential (decay is rapid).
        * **b**: b-Factor (strength of the diffusion gradient).
        * **ADC**: Diffusion Coefficient (water mobility).
        """, """
        * **S**: Gemessenes Signal (was auf dem Bild zu sehen ist).
        * **S₀**: Basissignal ohne Diffusion (b=0, reines T2-Bild).
        * **e**: Exponentialfunktion (der Abfall ist schnell).
        * **b**: b-Wert (Stärke des Diffusionsgradienten).
        * **ADC**: Diffusionskoeffizient (Beweglichkeit des Wassers).
        """))

    col_plot, col_expl = st.columns([2, 1])
    
    with col_plot:
        b = np.linspace(0, 3000, 300)
        adc_pure = 0.8e-3
        
        ln_S_adc = -b * adc_pure
        ivim_effect = 0.4 * np.exp(-b * 0.02)
        ln_S_ivim = np.log(np.exp(ln_S_adc) + ivim_effect)
        kurtosis_term = (1.0/6.0) * (b**2) * (adc_pure**2) * 1.5
        ln_S_kurt = ln_S_adc + kurtosis_term

        fig_decay = Figure(figsize=(8, 5)); ax_d = fig_decay.subplots()
        
        ax_d.fill_between(b, ln_S_adc, ln_S_ivim, where=(b < 800), color='#9b59b6', alpha=0.3, label=T('Effet IVIM', 'IVIM Effect', 'IVIM-Effekt'))
        ax_d.fill_between(b, ln_S_adc, ln_S_kurt, where=(b > 1000), color='#2ecc71', alpha=0.4, label=T('Effet Kurtosis', 'Kurtosis Effect', 'Kurtosis-Effekt'))
        
        ax_d.plot(b, ln_S_adc, color='red', linewidth=3, label=T('ADC (Modèle Gaussien)', 'ADC (Gaussian Model)', 'ADC (Gauß-Modell)'))
        
        b_pts = np.arange(0, 3100, 200)
        y_pts = -b_pts * adc_pure
        y_pts[b_pts < 500] += np.log(1 + 0.4*np.exp(-b_pts[b_pts < 500]*0.02))
        y_pts[b_pts > 1500] += (1.0/6.0) * (b_pts[b_pts > 1500]**2) * (adc_pure**2) * 1.5
        ax_d.scatter(b_pts, y_pts, color='black', zorder=5, label=T('Données', 'Data', 'Daten'))

        ax_d.text(300, -0.2, T("IVIM (Sang)", "IVIM (Blood)", "IVIM (Blut)"), color='purple', fontweight='bold')
        ax_d.text(2200, -2.5, T("Kurtosis (Cellules)", "Kurtosis (Cells)", "Kurtosis (Zellen)"), color='green', fontweight='bold')
        
        txt_slope = T("Pente = -ADC", "Slope = -ADC", "Steigung = -ADC")
        ax_d.text(1200, -1.2, txt_slope, color='red', rotation=-30, fontweight='bold')

        ax_d.set_xlabel(T("Facteur b", "b-Factor", "b-Wert"))
        ax_d.set_ylabel("ln(Signal)")
        ax_d.set_xlim(0, 3000)
        ax_d.set_ylim(-4, 0.2)
        ax_d.legend()
        ax_d.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_decay)
        

    with col_expl:
        st.info("### 🟣 " + T('Zone IVIM (b < 200)', 'IVIM Zone (b < 200)', 'IVIM-Zone (b < 200)'))
        st.markdown(T("""
        **"La Fausse Diffusion"**
        Au début, le signal chute vite. Ce n'est pas de la diffusion, c'est le **sang** qui circule (Pseudo-diffusion).
        * *Utile pour voir la perfusion sans produit de contraste.*
        """, """
        **"The Pseudo-Diffusion"**
        At the start, signal drops fast. This isn't diffusion, it's circulating **blood**.
        * *Useful to see perfusion without contrast agent.*
        """, """
        **"Die Pseudo-Diffusion"**
        Zu Beginn fällt das Signal schnell ab. Dies ist keine Diffusion, sondern zirkulierendes **Blut** (Pseudo-Diffusion).
        * *Nützlich, um die Perfusion ohne Kontrastmittel zu sehen.*
        """))
        
        st.success("### 🟢 " + T('Zone Kurtosis (b > 1000)', 'Kurtosis Zone (b > 1000)', 'Kurtosis-Zone (b > 1000)'))
        st.markdown(T("""
        **"L'Obstacle"**
        À la fin, la courbe remonte. L'eau tape dans les murs des cellules (Membranes).
        * *Utile pour grader les tumeurs complexes.*
        """, """
        **"The Obstacle"**
        At the end, the curve rises. Water hits cell walls (Membranes).
        * *Useful for grading complex tumors.*
        """, """
        **"Das Hindernis"**
        Am Ende steigt die Kurve an. Wasser stößt an Zellwände (Membranen).
        * *Nützlich zur Einstufung komplexer Tumore.*
        """))

elif module_actif == liste_modules[10]:
    st.header(T("🎓 Cours Théorique", "🎓 Theoretical Course", "🎓 Theoretischer Kurs"))

    slides_data = [
        {
            "fr": "1. Le Spin Nucléaire", 
            "en": "1. Nuclear Spin",
            "de": "1. Der Kernspin",
            "body_fr": "Le proton H+ agit comme un petit aimant.\nEn l'absence de champ B0, ils sont orientés aléatoirement.\nSous B0, ils s'alignent (Parallèle / Anti-parallèle).",
            "body_en": "The H+ proton acts like a small magnet.\nWithout B0 field, they are randomly oriented.\nUnder B0, they align (Parallel / Anti-parallel).",
            "body_de": "Das H+-Proton verhält sich wie ein kleiner Magnet.\nOhne B0-Feld sind sie zufällig ausgerichtet.\nUnter B0 richten sie sich aus (Parallel / Antiparallel)."
        },
        {
            "fr": "2. Résonance & Excitation", 
            "en": "2. Resonance & Excitation",
            "de": "2. Resonanz & Anregung",
            "body_fr": "Pour basculer l'aimantation, on envoie une onde RF.\nLa fréquence doit être exactement la Fréquence de Larmor.\nF = γ * B0 (42.58 MHz/T pour l'Hydrogène).",
            "body_en": "To flip magnetization, an RF wave is sent.\nThe frequency must match the Larmor Frequency.\nF = γ * B0 (42.58 MHz/T for Hydrogen).",
            "body_de": "Um die Magnetisierung zu kippen, wird eine HF-Welle gesendet.\nDie Frequenz muss genau der Larmorfrequenz entsprechen.\nF = γ * B0 (42,58 MHz/T für Wasserstoff)."
        },
        {
            "fr": "3. Relaxation T1 & T2", 
            "en": "3. T1 & T2 Relaxation",
            "de": "3. T1- & T2-Relaxation",
            "body_fr": "T1 (Longitudinal) : Repousse de l'aimantation (Graisse rapide, Eau lente).\nT2 (Transversal) : Déphasage des spins (Interaction spin-spin).\nC'est la base du contraste de l'image.",
            "body_en": "T1 (Longitudinal): Regrowth of magnetization (Fat fast, Water slow).\nT2 (Transverse): Dephasing of spins (Spin-spin interaction).\nThis is the basis of image contrast.",
            "body_de": "T1 (Longitudinal): Wiederaufbau der Magnetisierung (Fett schnell, Wasser langsam).\nT2 (Transversal): Dephasierung der Spins (Spin-Spin-Wechselwirkung).\nDies ist die Grundlage des Bildkontrasts."
        },
        {
            "fr": "4. Espace K (Fourier)", 
            "en": "4. K-Space (Fourier)",
            "de": "4. K-Raum (Fourier)",
            "body_fr": "L'IRM n'acquiert pas l'image directement.\nElle remplit l'Espace K (fréquences spatiales).\nLe centre = Contraste. La périphérie = Détails.",
            "body_en": "MRI does not acquire the image directly.\nIt fills K-Space (spatial frequencies).\nCenter = Contrast. Periphery = Details.",
            "body_de": "Das MRT akquiriert das Bild nicht direkt.\nEs füllt den K-Raum (Ortsfrequenzen).\nZentrum = Kontrast. Peripherie = Details."
        },
        {
            "fr": "5. Sécurité (SAR)", 
            "en": "5. Safety (SAR)",
            "de": "5. Sicherheit (SAR)",
            "body_fr": "Les ondes RF chauffent les tissus (Effet micro-onde).\nSAR = Taux d'Absorption Spécifique (W/kg).\nAttention aux implants, pacemakers et tatouages.",
            "body_en": "RF waves heat tissues (Microwave effect).\nSAR = Specific Absorption Rate (W/kg).\nBeware of implants, pacemakers, and tattoos.",
            "body_de": "HF-Wellen erwärmen das Gewebe (Mikrowelleneffekt).\nSAR = Spezifische Absorptionsrate (W/kg).\nVorsicht bei Implantaten, Herzschrittmachern und Tätowierungen."
        }
    ]

    if 'slide_index' not in st.session_state: 
        st.session_state.slide_index = 0
    
    num_slides = len(slides_data)
    
    idx = st.select_slider(
        T("Navigation Diapositive", "Slide Navigation", "Folien-Navigation"), 
        options=range(num_slides), 
        value=st.session_state.slide_index, 
        format_func=lambda i: T(slides_data[i]["fr"], slides_data[i]["en"], slides_data[i]["de"])
    )
    
    st.session_state.slide_index = idx
    
    current_data = slides_data[idx]
    title_current = T(current_data["fr"], current_data["en"], current_data["de"])
    body_current = T(current_data["body_fr"], current_data["body_en"], current_data["body_de"])

    st.markdown(f"### 📄 {title_current}")

    fig_ppt = Figure(figsize=(10, 6)); ax_ppt = fig_ppt.subplots()
    
    ax_ppt.set_facecolor('#f0f0f5')
    
    ax_ppt.text(0.5, 0.8, title_current.upper(), 
                ha='center', va='center', fontsize=16, fontweight='bold', color='#2c3e50')
    
    ax_ppt.text(0.5, 0.5, body_current, 
                ha='center', va='center', fontsize=14, color='#34495e', wrap=True)
    
    footer_txt = T(f"Diapositive {idx+1}/{num_slides}", f"Slide {idx+1}/{num_slides}", f"Folie {idx+1}/{num_slides}")
    ax_ppt.text(0.95, 0.05, footer_txt, 
                ha='right', va='bottom', fontsize=10, color='gray')

    ax_ppt.set_xlim(0, 1)
    ax_ppt.set_ylim(0, 1)
    ax_ppt.axis('off')
    
    st.pyplot(fig_ppt)
    

elif module_actif == liste_modules[11]:
    st.header(T("🩸 Imagerie de Susceptibilité Magnétique (SWI)", "🩸 Susceptibility Weighted Imaging (SWI)", "🩸 Suszeptibilitätsgewichtete Bildgebung (SWI)"))
    
    swi_tab_names = [
        T("1. Physique (Phase & Vecteurs)", "1. Physics (Phase & Vectors)", "1. Physik (Phase & Vektoren)"), 
        T("2. Le Dipôle (Simulation)", "2. The Dipole (Simulation)", "2. Der Dipol (Simulation)"), 
        T("3. Imagerie Clinique", "3. Clinical Imaging", "3. Klinische Bildgebung")
    ]
    
    swi_tab1, swi_tab2, swi_tab3 = st.tabs(swi_tab_names)

    with swi_tab1:
        st.subheader(T("1. Physique : L'Analogie de la Boussole", "1. Physics: The Compass Analogy", "1. Physik: Die Kompass-Analogie"))
        
        col_ctrl, col_graph = st.columns([1, 2], gap="medium")
        
        with col_ctrl:
            st.markdown("#### " + T('🎛️ Contrôles', '🎛️ Controls', '🎛️ Steuerung'))
            st.caption(T("_Modifiez les valeurs pour faire tourner l'aiguille._", "_Modify values to rotate the needle._", "_Ändern Sie die Werte, um die Nadel zu drehen._"))
            
            te_simu = st.slider(T("Temps d'Écho (TE)", "Echo Time (TE)", "Echozeit (TE)"), 0, 80, 20, step=1, key="swi_te_p1_pedago")
            fa_simu = st.slider(T("Angle Bascule (°)", "Flip Angle (°)", "Flipwinkel (°)"), 5, 90, 30, key="swi_fa_p1_pedago")
            
            t2_star = 50.0; df = 8.0 
            mag = np.sin(np.radians(fa_simu)) * np.exp(-te_simu / t2_star)
            phase_visu = np.radians(max(10, min(80, 60 - (te_simu/2))))
            vec_visu = mag * np.exp(1j * phase_visu)
            
            st.divider()
            
            c_met1, c_met2 = st.columns(2)
            c_met1.metric(T("Réel (Ombre Sol)", "Real (Floor Shadow)", "Real (Bodenschatten)"), f"{vec_visu.real:.2f}")
            c_met2.metric(T("Imag (Ombre Mur)", "Imag (Wall Shadow)", "Imag (Wandschatten)"), f"{vec_visu.imag:.2f}")

        with col_graph:
            fig_v = Figure(figsize=(5, 5)); ax_v = fig_v.subplots() 
            fig_v.patch.set_alpha(0) 
            lim = 1.1; ax_v.set_xlim(-0.1, lim); ax_v.set_ylim(-0.1, lim)
            
            ax_v.axhline(0, color='white', lw=1); ax_v.axvline(0, color='white', lw=1)
            
            ax_v.arrow(0, 0, vec_visu.real, vec_visu.imag, head_width=0.03, lw=4, fc='#3498db', ec='#3498db', length_includes_head=True, zorder=5)
            ax_v.text(vec_visu.real/2, vec_visu.imag/2 + 0.1, "SIGNAL", color='#3498db', fontweight='bold', ha='center', fontsize=12)
            
            ax_v.plot([vec_visu.real, vec_visu.real], [0, vec_visu.imag], color='gray', ls=':', lw=1)
            ax_v.arrow(0, 0, vec_visu.real, 0, head_width=0.02, lw=3, fc='#e74c3c', ec='#e74c3c', length_includes_head=True, zorder=4)
            ax_v.text(vec_visu.real/2, -0.08, T("Réel (X)", "Real (X)", "Real (X)"), color='#e74c3c', ha='center', fontsize=10, fontweight='bold')
            
            ax_v.plot([0, vec_visu.real], [vec_visu.imag, vec_visu.imag], color='gray', ls=':', lw=1)
            ax_v.arrow(0, 0, 0, vec_visu.imag, head_width=0.02, lw=3, fc='#2ecc71', ec='#2ecc71', length_includes_head=True, zorder=4)
            ax_v.text(-0.02, vec_visu.imag/2, T("Imag (Y)", "Imag (Y)", "Imag (Y)"), color='#2ecc71', ha='right', va='center', fontsize=10, fontweight='bold')
            
            arc = patches.Arc((0,0), 0.4, 0.4, theta1=0, theta2=np.degrees(phase_visu), color='yellow', lw=2)
            ax_v.add_patch(arc)
            ax_v.text(0.25, 0.1, "Phase", color='yellow', fontsize=11, fontweight='bold')
            
            ax_v.set_title(T("Visualisation Vectorielle", "Vector Visualization", "Vektorvisualisierung"), color='white', fontsize=12)
            ax_v.set_aspect('equal'); ax_v.axis('off')
            st.pyplot(fig_v); 

        st.markdown("---")
        with st.expander(T("📖 Comprendre l'Analogie (Cliquez)", "📖 Understand Analogy (Click)", "📖 Analogie verstehen (Klicken)"), expanded=True):
            c_txt1, c_txt2 = st.columns(2)
            with c_txt1:
                st.info(T("""**🧭 L'Analogie de la Boussole** \n * **L'Aiguille (Bleue)** : C'est le Signal IRM total. \n * **Sa Longueur** : La force du signal (Magnitude). \n * **Sa Direction** : La nature du tissu (Phase).""",
                          """**🧭 The Compass Analogy** \n * **The Needle (Blue)**: It is the total MRI Signal. \n * **Its Length**: Signal Strength (Magnitude). \n * **Its Direction**: Tissue Nature (Phase).""",
                          """**🧭 Die Kompass-Analogie** \n * **Die Nadel (Blau)**: Ist das gesamte MRT-Signal. \n * **Ihre Länge**: Die Signalstärke (Magnitude). \n * **Ihre Richtung**: Die Gewebenatur (Phase)."""))
            with c_txt2:
                st.warning(T(
                    "**💡 Pourquoi Réel & Imaginaire ?** \n\nL'ordinateur ne stocke pas une flèche tournante. Il stocke ses deux ombres : \n* **Partie Réelle :** L'ombre au sol (Axe X). \n* **Partie Imaginaire :** L'ombre au mur (Axe Y).\n\n*Techniquement, la machine capte physiquement ces deux 'ombres' en même temps en utilisant deux antennes placées à 90° l'une de l'autre.*",
                    "**💡 Why Real & Imaginary?** \n\nThe computer doesn't store a spinning arrow. It stores its two shadows: \n* **Real Part:** Floor shadow (X Axis). \n* **Imaginary Part:** Wall shadow (Y Axis).\n\n*Technically, the scanner physically captures both 'shadows' at the same time by using two antennas placed 90° apart.*",
                    "**💡 Warum Real & Imaginär?** \n\nDer Computer speichert keinen rotierenden Pfeil. Er speichert seine zwei Schatten: \n* **Realteil:** Bodenschatten (X-Achse). \n* **Imaginärteil:** Wandschatten (Y-Achse).\n\n*Technisch gesehen erfasst das Gerät beide 'Schatten' gleichzeitig durch zwei um 90° versetzte Antennen.*"
                ))
    
    with swi_tab2:
        st.subheader(T("2. 🧲 Le Laboratoire du Dipôle", "2. 🧲 Dipole Laboratory", "2. 🧲 Dipol-Labor"))
        
        col_dip_ctrl, col_dip_visu = st.columns([1, 3])
        
        opt_hema = T("Hématome (Paramagnétique)", "Hematoma (Paramagnetic)", "Hämatom (Paramagnetisch)")
        opt_calc = T("Calcium (Diamagnétique)", "Calcium (Diamagnetic)", "Kalzium (Diamagnetisch)")
        
        opt_rhs = T("RHS (GE/Philips/Canon)", "RHS (GE/Philips/Canon)", "RHS (GE/Philips/Canon)")
        opt_lhs = T("LHS (Siemens)", "LHS (Siemens)", "LHS (Siemens)")

        with col_dip_ctrl:
            dipole_substance = st.radio(T("Substance :", "Substance:", "Substanz:"), [opt_hema, opt_calc], key="dip_sub_key")
            dipole_system = st.radio(T("Convention Phase :", "Phase Convention:", "Phasenkonvention:"), [opt_rhs, opt_lhs], key="dip_sys_key")
            st.divider()
            
            if "dipole_field" in locals() and np.max(np.abs(dipole_field)) > 0:
                st.success(T("✅ Champ Dipolaire Détecté (Fantôme)", "✅ Dipole Field Detected (Phantom)", "✅ Dipolfeld erkannt (Phantom)"))
                st.image(utils.apply_window_level(dipole_field, 1.0, 0.5), caption=T("Carte de Champ B (Simulée)", "B-Field Map (Simulated)", "B-Feld-Karte (Simuliert)"), clamp=True)
            else:
                z_pos = st.slider(T("Coupe Axiale (Z)", "Axial Slice (Z)", "Axiale Schicht (Z)"), -1.5, 1.5, 0.0, 0.1, key="dip_z_key")

        with col_dip_visu:
            fig_dip = Figure(figsize=(10, 4)); axes_dip = fig_dip.subplots(1, 2)
            fig_dip.patch.set_facecolor('#404040')
            
            is_rhs = "RHS" in dipole_system
            is_para = dipole_substance == opt_hema
            combo = (1 if is_para else -1) * (1 if is_rhs else -1)
            
            col_eq_cen, col_eq_halo, col_poles = ('white', 'black', 'black') if combo > 0 else ('black', 'white', 'white')
            
            axes_dip[0].set_facecolor('#404040'); axes_dip[0].axis('off')
            axes_dip[0].add_patch(patches.Ellipse((0.5, 0.7), 0.25, 0.35, color=col_poles, alpha=0.9))
            axes_dip[0].add_patch(patches.Ellipse((0.5, 0.3), 0.25, 0.35, color=col_poles, alpha=0.9))
            axes_dip[0].add_patch(patches.Rectangle((0.35, 0.48), 0.3, 0.04, color=col_eq_cen))
            
            z_val = z_pos if 'z_pos' in locals() else 0.0
            axes_dip[0].axhline(y=0.5 - (z_val * 0.2), color='yellow', linewidth=2, linestyle='--')
            
            axes_dip[1].set_facecolor('#404040'); axes_dip[1].axis('off')
            if abs(z_val) < 0.2:
                axes_dip[1].add_patch(patches.Circle((0.5, 0.5), 0.35, color=col_eq_halo, alpha=0.5))
                axes_dip[1].add_patch(patches.Circle((0.5, 0.5), 0.15, color=col_eq_cen))
            elif 0.2 <= abs(z_val) < 1.0:
                axes_dip[1].add_patch(patches.Circle((0.5, 0.5), 0.25 * (1.2 - abs(z_val)), color=col_poles))
                
            st.pyplot(fig_dip); 

    with swi_tab3:
        st.subheader(T("3. Imagerie SWI Clinique", "3. Clinical SWI Imaging", "3. Klinische SWI-Bildgebung"))
        
        path_minip_fixe = os.path.join(current_dir, "minip_static.png") 
        
        if HAS_NILEARN and processor.ready:
            dims = processor.get_dims() 
            c1_swi, c2_swi = st.columns([1, 4])
            
            with c1_swi:
                 st.markdown("##### " + T('🩻 Navigation', '🩻 Navigation', '🩻 Navigation'))
                 
                 opt_ax = T("Axiale", "Axial", "Axial")
                 opt_cor = T("Coronale", "Coronal", "Koronar")
                 opt_sag = T("Sagittale", "Sagittal", "Sagittal")
                 
                 swi_view = st.radio(T("Plan de Coupe :", "Slice Plane:", "Schnittebene:"), [opt_ax, opt_cor, opt_sag], key="swi_view_mode")
                 
                 if swi_view == opt_ax: 
                     swi_slice = st.slider(T("Position Z", "Position Z", "Position Z"), 0, dims[2]-1, 90, key="swi_z"); axis_code = 'z'
                 elif swi_view == opt_cor: 
                     swi_slice = st.slider(T("Position Y", "Position Y", "Position Y"), 0, dims[1]-1, 100, key="swi_y"); axis_code = 'y'
                 else: 
                     swi_slice = st.slider(T("Position X", "Position X", "Position X"), 0, dims[0]-1, 90, key="swi_x"); axis_code = 'x'
                 
                 st.divider()
                 show_microbleeds_swi = st.checkbox(T("Simuler Micro-saignements", "Simulate Microbleeds", "Mikroblutungen simulieren"), False, key="swi_bleed_check")
                 show_dipole_test = st.checkbox(T("🧪 Dipôle (Test)", "🧪 Dipole (Test)", "🧪 Dipol (Test)"), False, key="swi_dip_test_check")
            
            with c2_swi:
                sys_arg = "RHS" if "RHS" in dipole_system else "LHS"
                sub_arg = dipole_substance 
                
                img_mag = processor.get_slice(axis_code, swi_slice, {}, swi_mode='mag', te=te_simu, with_bleeds=show_microbleeds_swi)
                img_phase = processor.get_slice(axis_code, swi_slice, {}, swi_mode='phase', with_bleeds=show_microbleeds_swi, swi_sys=sys_arg, swi_sub=sub_arg, with_dipole=show_dipole_test)
                
                if img_mag is not None:
                    if axis_code != 'x': 
                        img_mag = np.fliplr(img_mag)
                if img_phase is not None:
                    if axis_code != 'x': 
                        img_phase = np.fliplr(img_phase)
                
                c_mag, c_pha, c_min = st.columns(3)
                
                with c_mag: 
                    st.image(utils.apply_window_level(img_mag, 1.0, 0.5), caption=f"1. Magnitude ({swi_view})", use_container_width=True)
                with c_pha: 
                    st.image(utils.apply_window_level(img_phase, 1.0, 0.5), caption=f"2. Phase ({swi_view})", use_container_width=True)
                with c_min: 
                    if os.path.exists(path_minip_fixe): 
                        st.image(path_minip_fixe, caption=T("3. MinIP (Référence Axiale)", "3. MinIP (Axial Ref)", "3. MinIP (Axiale Ref.)"), use_container_width=True)
                    else: 
                        st.image(np.zeros((200,200)), caption=T("Image manquante", "Missing Image", "Fehlendes Bild"), clamp=True)
        else:
            st.info(T("Module Anatomique non chargé. Utilisez le Fantôme Dipôle (Onglet 2) pour la démonstration.", 
                      "Anatomy Module not loaded. Use Dipole Phantom (Tab 2) for demo.",
                      "Anatomiemodul nicht geladen. Verwenden Sie das Dipol-Phantom (Tab 2) zur Demonstration."))

elif module_actif == liste_modules[12]:
    st.header(T("🧠 Séquence 3D T1 Ultra-Rapide (MP-RAGE)", "🧠 Ultra-Fast 3D T1 Sequence (MP-RAGE)", "🧠 Ultraschnelle 3D T1-Sequenz (MP-RAGE)"))
    
    # --- PRÉPARATION DES TEXTES POUR LE TABLEAU HTML ---
    h_brand = T("Constructeur", "Manufacturer", "Hersteller")
    h_name  = T("Nom Commercial", "Commercial Name", "Handelsname")
    h_tech  = T("Signification Technique", "Technical Meaning", "Technische Bedeutung")
    
    txt_philips = T("avec Pré-impulsion", "with Pre-pulse", "mit Präpuls")
    txt_canon   = T("avec Inversion", "with Inversion", "mit Inversion")
    
    st.markdown(f"""
    <style>
    .table-style {{width: 100%; border-collapse: collapse; font-size: 14px;}}
    .table-style th {{background-color: #f0f2f6; padding: 8px; text-align: left; border-bottom: 2px solid #ddd;}}
    .table-style td {{padding: 8px; border-bottom: 1px solid #ddd;}}
    .brand-col {{font-weight: bold; color: #31333F;}} .name-col {{font-weight: bold; color: #d63031;}}
    </style>
    <table class="table-style">
        <tr><th>{h_brand}</th><th>{h_name}</th><th>{h_tech}</th></tr>
        <tr><td class="brand-col">SIEMENS</td><td class="name-col">MP-RAGE</td><td>Magnetization Prepared - Rapid Gradient Echo</td></tr>
        <tr><td class="brand-col">GE</td><td class="name-col">3D IR-FSPGR (BRAVO)</td><td>Inversion Recovery Fast SPGR</td></tr>
        <tr><td class="brand-col">PHILIPS</td><td class="name-col">3D T1-TFE</td><td>Turbo Field Echo ({txt_philips})</td></tr>
        <tr><td class="brand-col">CANON</td><td class="name-col">3D Fast FE</td><td>Fast Field Echo ({txt_canon})</td></tr>
    </table><br>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    col_mp_ctrl, col_mp_plot = st.columns([1, 2])
    
    with col_mp_ctrl:
        constructeur_mp = st.radio(
            T("Sélecteur Constructeur :", "Manufacturer Selector:", "Hersteller-Auswahl:"), 
            ["SIEMENS", "GE", "PHILIPS", "CANON"], 
            key="mp_const_select_final"
        )
        
        st.markdown(T(
            "**Pourquoi le TR diffère ?** Le TR affiché sur la console ne représente pas la même chose selon le constructeur.",
            "**Why does TR differ?** The TR displayed on the console does not represent the same thing depending on the manufacturer.",
            "**Warum unterscheidet sich die TR?** Die an der Konsole angezeigte TR bedeutet je nach Hersteller nicht dasselbe."
        ))

    with col_mp_plot:
        fig_mp = Figure(figsize=(10, 4)); ax_mp = fig_mp.subplots()
        
        ti_mp = 900
        train_len = 600
        tr_echo_val = 8 
        
        ax_mp.bar(0, 1.2, width=40, color='#e74c3c', label='Inversion 180°', zorder=3)
        ax_mp.text(0, 1.35, "180°", color='#e74c3c', ha='center', fontweight='bold')
        
        echo_step = 60 
        for k in range(0, train_len, echo_step): 
            ax_mp.bar(ti_mp + k, 0.7, width=25, color='#3498db', alpha=0.7)
            
        ax_mp.add_patch(patches.Rectangle((ti_mp - 20, 0), train_len + 10, 0.8, color='#3498db', alpha=0.1))
        
        if constructeur_mp == "SIEMENS":
            ax_mp.annotate('', xy=(ti_mp + train_len + 100, -1.0), xytext=(0, -1.0), 
                           arrowprops=dict(arrowstyle='<->', color='green', lw=2.5))
            
            txt_tr_siemens = T("TR Siemens : Temps du Cycle Complet (~2300ms)", "TR Siemens: Full Cycle Time (~2300ms)", "TR Siemens: Volle Zykluszeit (~2300ms)")
            ax_mp.text((ti_mp + train_len)/2, -1.35, txt_tr_siemens, 
                       color='green', weight='bold', ha='center', fontsize=10)
        else:
            start_x = ti_mp + echo_step
            end_x = ti_mp + 2 * echo_step
            
            ax_mp.annotate('', xy=(end_x, 0.35), xytext=(start_x, 0.35), 
                           arrowprops=dict(arrowstyle='<->', color='#f39c12', lw=3))
            
            ax_mp.text((start_x + end_x)/2, 0.15, f"TR {constructeur_mp} = {tr_echo_val}ms", 
                       color='#f39c12', weight='bold', ha='center', fontsize=11)
            
        ax_mp.set_ylim(-1.6, 1.6)
        ax_mp.set_xlim(-100, ti_mp + train_len + 200)
        ax_mp.axis('off')
        ax_mp.axhline(0, color='black', linewidth=0.5)
        
        st.pyplot(fig_mp)
        
        
    st.divider()
    
    st.markdown("#### " + T("Optimisation du Contraste (Substance Blanche vs Grise)", "Contrast Optimization (White vs Gray Matter)", "Kontrastoptimierung (Weiße vs. Graue Substanz)"))

    col_mp_txt, col_mp_plot = st.columns([1, 2])

    with col_mp_txt:
        st.info(T(
            "**Analyse Comparée (Même repère) :**\n\nRegardez les barres de gris à gauche :\n1. **Haut (Module + Phase) :** L'échelle est linéaire. -1 est **Noir**, +1 est **Blanc**. \nLa SG (négative) apparaît sombre, la SB (positive) apparaît claire. **Contraste Fort.**\n\n2. **Bas (Module Seul) :** L'échelle est en 'V'. 0 est **Noir**, mais -1 et +1 sont tous deux **Blancs (Hyper Signal)**.\nLa SG (négative) apparaît donc brillante, tout comme la SB. **Confusion & Perte de Contraste.**",
            "**Comparative Analysis (Same Frame):**\n\nLook at the grayscale bars on the left:\n1. **Top (Magnitude + Phase):** Linear scale. -1 is **Black**, +1 is **White**. \nGM (negative) appears dark, WM (positive) appears bright. **High Contrast.**\n\n2. **Bottom (Magnitude Only):** 'V' shape scale. 0 is **Black**, but both -1 and +1 are **White (Hyper Signal)**.\nGM (negative) thus appears bright, just like WM. **Confusion & Contrast Loss.**",
            "**Vergleichende Analyse (Gleicher Rahmen):**\n\nBetrachten Sie die Graustufenbalken links:\n1. **Oben (Betrag + Phase):** Lineare Skala. -1 ist **Schwarz**, +1 ist **Weiß**. \nDie GS (negativ) erscheint dunkel, die WS (positiv) hell. **Starker Kontrast.**\n\n2. **Unten (Nur Betrag):** V-förmige Skala. 0 ist **Schwarz**, aber -1 und +1 sind beide **Weiß (Hypersignal)**.\nDie GS (negativ) erscheint somit hell, genau wie die WS. **Verwirrung & Kontrastverlust.**"
        ))
        
        st.success(T(
            "**💡 Pourquoi le TI annule-t-il le LCR ?**\n\n**Sans le 180° initial**, cette séquence rapide produirait une image 'plate' (type Densité de Protons) où le LCR serait gris clair.\n\nL'impulsion 180° force l'aimantation à partir de **-1**. En remontant vers **+1**, elle doit obligatoirement **croiser le Zéro**.\n👉 Si on fixe le **TI** exactement à cet instant, le LCR n'a plus de signal. Il apparaît **NOIR PUR**, ce qui est impossible sans cette préparation.",
            "**💡 Why does TI null the CSF?**\n\n**Without the initial 180°**, this fast sequence would yield a 'flat' image (PD type) where CSF would be light gray.\n\nThe 180° pulse forces magnetization to start at **-1**. As it recovers towards **+1**, it must **cross Zero**.\n👉 If we set the **TI** exactly at this moment, the CSF has no signal. It appears **PURE BLACK**, which is impossible without this preparation.",
            "**💡 Warum annulliert die TI den Liquor?**\n\n**Ohne den anfänglichen 180°-Puls** würde diese schnelle Sequenz ein 'flaches' Bild (PD-Gewichtung) erzeugen, auf dem der Liquor hellgrau wäre.\n\nDer 180°-Puls zwingt die Magnetisierung, bei **-1** zu starten. Während sie sich in Richtung **+1** erholt, muss sie **Null kreuzen**.\n👉 Wenn wir die **TI** genau auf diesen Moment legen, hat der Liquor kein Signal mehr. Er erscheint **REIN SCHWARZ**, was ohne diese Präparation unmöglich wäre."
        ))

    with col_mp_plot:
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig_mp = Figure(figsize=(8, 7)); ax_mp = fig_mp.subplots(2, 1, sharex=False)
        fig_mp.subplots_adjust(hspace=0.4)

        t = np.linspace(0, 2500, 500)
        t1_sb = 260   
        t1_sg = 530   
        TI_mp = 250   

        mz_sb_real = 1 - 2 * np.exp(-t / t1_sb)
        mz_sg_real = 1 - 2 * np.exp(-t / t1_sg)
        val_sb_real = 1 - 2 * np.exp(-TI_mp / t1_sb) 
        val_sg_real = 1 - 2 * np.exp(-TI_mp / t1_sg) 

        ax = ax_mp[0]
        ax.set_title(T("✅ IMAGE EN MODULE ET PHASE", "✅ MAGNITUDE AND PHASE IMAGE", "✅ BETRAGS- UND PHASENBILD"), 
                     loc='center', color='green', fontweight='bold', pad=10)
        
        ax.plot(t, mz_sb_real, color='black', lw=2, linestyle='-', label=T('SB', 'WM', 'WS'))
        ax.plot(t, mz_sg_real, color='gray', lw=2, linestyle='--', label=T('SG', 'GM', 'GS'))
        
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)
        ax.axvline(TI_mp, color='green', linestyle='--', alpha=0.8)
        ax.text(TI_mp, 1.15, "TI", color='green', ha='center', fontweight='bold')

        col_sb_ph = (val_sb_real + 1) / 2
        col_sg_ph = (val_sg_real + 1) / 2
        ax.plot(TI_mp, val_sb_real, marker='s', markersize=14, markeredgecolor='green', color=str(col_sb_ph), zorder=10)
        ax.plot(TI_mp, val_sg_real, marker='s', markersize=14, markeredgecolor='green', color=str(col_sg_ph), zorder=10)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("left", size="6%", pad=0.5)
        grad_phase = np.linspace(1, -1, 100).reshape(-1, 1)
        cax.imshow(grad_phase, aspect='auto', cmap='gray', extent=[0, 1, -1, 1])
        cax.set_xticks([]); cax.set_yticks([-1, 0, 1])
        cax.set_yticklabels(["-1", "0", "+1"])
        cax.yaxis.set_ticks_position('left')

        ax.set_xlabel("Temps (ms)")
        ax.set_ylim(-1.2, 1.2)
        ax.grid(False)
        ax.legend(loc='lower right', fontsize=8)

        ax = ax_mp[1]
        ax.set_title(T("❌ IMAGE EN MODULE", "❌ MAGNITUDE IMAGE", "❌ BETRAGSBILD"), 
                     loc='center', color='red', fontweight='bold', pad=10)

        ax.plot(t, mz_sb_real, color='black', lw=2, linestyle='-', label=T('SB', 'WM', 'WS'))
        ax.plot(t, mz_sg_real, color='gray', lw=2, linestyle='--', label=T('SG', 'GM', 'GS'))
        
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)
        ax.axvline(TI_mp, color='red', linestyle='--', alpha=0.8)

        col_sb_mag = abs(val_sb_real)
        col_sg_mag = abs(val_sg_real)
        ax.plot(TI_mp, val_sb_real, marker='s', markersize=14, markeredgecolor='red', color=str(col_sb_mag), zorder=10)
        ax.plot(TI_mp, val_sg_real, marker='s', markersize=14, markeredgecolor='red', color=str(col_sg_mag), zorder=10)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("left", size="6%", pad=0.5)
        grad_v = np.abs(np.linspace(1, -1, 100)).reshape(-1, 1)
        cax.imshow(grad_v, aspect='auto', cmap='gray', extent=[0, 1, -1, 1])
        cax.set_xticks([]); cax.set_yticks([-1, 0, 1])
        cax.set_yticklabels([T("-1 (Blanc)", "-1 (White)", "-1 (Weiß)"), "0", T("+1 (Blanc)", "+1 (White)", "+1 (Weiß)")])
        cax.yaxis.set_ticks_position('left')

        ax.set_xlabel("Temps (ms)")
        ax.set_ylim(-1.2, 1.2)
        ax.grid(False)
        ax.legend(loc='lower right', fontsize=8)

        st.pyplot(fig_mp)
        

elif module_actif == liste_modules[13]:
    st.header(T("🩸 Perfusion ASL (Arterial Spin Labeling)", "🩸 ASL Perfusion (Arterial Spin Labeling)", "🩸 ASL-Perfusion (Arterial Spin Labeling)"))
    
    c_principe, c_texte = st.columns([1, 1])
    
    with c_principe:
        image_asl_path = os.path.join(current_dir, "image_028fa1.jpg")
        if os.path.exists(image_asl_path): 
            st.image(
                image_asl_path, 
                caption=T("Principe ASL", "ASL Principle", "ASL-Prinzip"), 
                use_container_width=True
            )
        else:
            st.info(T("Image explicative non trouvée.", "Explanatory image not found.", "Erklärendes Bild nicht gefunden."))
            
    with c_texte:
        txt_desc = T(
            "### Comment ça marche ?\n1. **Marquage (Tag) :** Une impulsion RF **inverse l'aimantation (180°)** du sang au niveau du cou.\n2. **Délai (PLD) :** On attend que le sang monte au cerveau.\n3. **Acquisition :** On prend une image 'Marquée'.\n4. **Soustraction :** Image Contrôle - Image Marquée = Perfusion.",
            "### How does it work?\n1. **Labeling (Tag):** An RF pulse **inverts the magnetization (180°)** of the blood at the neck level.\n2. **Delay (PLD):** We wait for the blood to flow up to the brain.\n3. **Acquisition:** We take a 'Labeled' image.\n4. **Subtraction:** Control Image - Labeled Image = Perfusion.",
            "### Wie funktioniert es?\n1. **Markierung (Tag):** Ein HF-Puls **invertiert die Magnetisierung (180°)** des Blutes auf Halshöhe.\n2. **Wartezeit (PLD):** Man wartet, bis das Blut ins Gehirn fließt.\n3. **Akquisition:** Man macht ein 'markiertes' Bild.\n4. **Subtraktion:** Kontrollbild - Markiertes Bild = Perfusion."
        )
        st.markdown(txt_desc)
        
        with st.expander(T("⏱️ Focus Physique : Pourquoi TR > 4000ms ?", "⏱️ Physics Focus: Why TR > 4000ms?", "⏱️ Physik-Fokus: Warum TR > 4000ms?")):
            st.markdown(T(
                "**Cycle ASL :** Marquage (~2s) + Attente (~2s) + Acquisition (~0.5s) = TR ~4.5s",
                "**ASL Cycle:** Labeling (~2s) + Delay (~2s) + Acquisition (~0.5s) = TR ~4.5s",
                "**ASL-Zyklus:** Markierung (~2s) + Wartezeit (~2s) + Akquisition (~0.5s) = TR ~4.5s"
            ))

    st.divider()

    st.subheader(T("⏱️ Séquence Temporelle pCASL", "⏱️ pCASL Timing Diagram", "⏱️ pCASL-Sequenzdiagramm"))

    fig_asl = Figure(figsize=(7, 2.2)); ax_asl = fig_asl.subplots()
    
    ax_asl.set_xlim(0, 12)
    ax_asl.set_ylim(-1.5, 2.5)
    ax_asl.axis('off') 
    ax_asl.plot([0.5, 11.5], [0, 0], color='black', linewidth=1)

    txt_sat_title = T("Saturation\n& Fond", "Saturation\n& Bkg", "Sättigung\n& Hntergr.")
    txt_pcasl = T("Marquage pCASL", "pCASL labeling", "pCASL-Markierung")
    txt_bs = T("Suppression\nFond", "Bkg\nSuppression", "Hntergr.-\nUnterdr.")
    txt_acq = T("Acquisition\n3D", "3D\nAcquisition", "3D-\nAkquisition")
    
    txt_dur_label = T("Durée (TL)", "Duration (TL)", "Dauer (TL)")
    txt_pld = "PLD"
    txt_readout = "Readout"

    ax_asl.arrow(1.5, 0, 0, 1.2, head_width=0.2, head_length=0.15, color='#1565c0', lw=2, length_includes_head=True)
    ax_asl.text(1.5, 1.4, txt_sat_title, ha='center', va='bottom', fontsize=8, color='black')

    rect_label = patches.Rectangle((3, 0), 3, 1, linewidth=1, edgecolor='black', facecolor='#ffcdd2') 
    ax_asl.add_patch(rect_label)
    ax_asl.text(4.5, 0.5, txt_pcasl, ha='center', va='center', fontsize=9, fontweight='bold')

    ax_asl.arrow(7, 0, 0, 1.2, head_width=0.2, head_length=0.15, color='#1565c0', lw=2, length_includes_head=True)
    ax_asl.text(7, 1.4, txt_bs, ha='center', va='bottom', fontsize=8, color='black')

    rect_acq = patches.Rectangle((8.5, 0), 2.5, 1.5, linewidth=1, edgecolor='black', facecolor='#dcedc8')
    ax_asl.add_patch(rect_acq)
    ax_asl.text(9.75, 0.75, txt_acq, ha='center', va='center', fontsize=9, fontweight='bold')

    def draw_double_arrow(x_start, x_end, y_pos, text_label):
        ax_asl.annotate('', xy=(x_start, y_pos), xytext=(x_end, y_pos), 
                        arrowprops=dict(arrowstyle='<->', color='#4472c4', lw=1.5))
        ax_asl.text((x_start + x_end) / 2, y_pos - 0.2, text_label, ha='center', va='top', fontsize=8)

    draw_double_arrow(3, 6, -0.3, txt_dur_label)
    draw_double_arrow(6, 8.5, -0.3, txt_pld)
    draw_double_arrow(8.5, 11, -0.3, txt_readout)

    fig_asl.subplots_adjust(left=0, right=1, top=1, bottom=0)
    st.pyplot(fig_asl, use_container_width=False)
    

    st.divider()
    
    st.subheader(T("2. Simulation Clinique & Pathologies", "2. Clinical Simulation & Pathologies", "2. Klinische Simulation & Pathologien"))
    
    if HAS_NILEARN and processor.ready:
        c1_asl, c2_asl = st.columns([1, 4])
        
        with c1_asl:
            dims = processor.get_dims()
            asl_slice = st.slider(T("Coupe Axiale (Z)", "Axial Slice (Z)", "Axiale Schicht (Z)"), 0, dims[2]-1, 90, key="asl_z")
            
            pld_val = locals().get('pld', 1500) 
            st.info(T(f"⏱️ **PLD Actuel : {pld_val} ms**", f"⏱️ **Current PLD: {pld_val} ms**", f"⏱️ **Aktuelle PLD: {pld_val} ms**"))
            
            if show_stroke: 
                st.error(T("⚠️ **AVC Ischémique**", "⚠️ **Ischemic Stroke**", "⚠️ **Ischämischer Schlaganfall**"))
            if show_atrophy: 
                st.warning(T("🧠 **Atrophie (Alzheimer)**", "🧠 **Atrophy (Alzheimer's)**", "🧠 **Atrophie (Alzheimer)**"))
                
        with c2_asl:
            ctrl_img, label_img, perf_map = processor.get_asl_maps(
                'z', asl_slice, pld_val, 1600, 
                with_stroke=show_stroke, 
                with_atrophy=show_atrophy
            )
            
            if ctrl_img is not None:
                col_ctrl, col_label, col_perf = st.columns(3)
                
                with col_ctrl: 
                    st.image(
                        utils.apply_window_level(ctrl_img, 1.0, 0.5), 
                        caption=T("1. Image Contrôle", "1. Control Image", "1. Kontrollbild"), 
                        clamp=True, use_container_width=True
                    )
                with col_label: 
                    st.image(
                        utils.apply_window_level(label_img, 1.0, 0.5), 
                        caption=T("2. Image Marquée", "2. Labeled Image", "2. Markiertes Bild"), 
                        clamp=True, use_container_width=True
                    )
                with col_perf:
                    fig_perf = Figure(); ax_perf = fig_perf.subplots()
                    im = ax_perf.imshow(perf_map, cmap='jet', vmin=0, vmax=np.max(perf_map)*0.8)
                    ax_perf.axis('off')
                    st.pyplot(fig_perf)
                    st.caption(T("3. Carte de Perfusion", "3. Perfusion Map", "3. Perfusionskarte"))
            else:
                st.error("Erreur de calcul des cartes ASL.")
    else: 
        st.warning(T("Module Anatomique requis pour la simulation clinique.", "Anatomy Module required for clinical simulation.", "Anatomiemodul für klinische Simulation erforderlich."))

elif module_actif == liste_modules[14]:
    st.header(T("🩸 Angiographie TOF (Time of Flight)", "🩸 TOF Angiography", "🩸 TOF-Angiographie"))

    with st.expander(T("📘 Principe Physique : Phénomène d'Entrée de Coupe (Inflow)", "📘 Physics: Inflow Effect", "📘 Physik: Inflow-Effekt"), expanded=True):
        col_sci1, col_sci2 = st.columns([1, 1])
        
        with col_sci1:
            st.markdown("### " + T("🌊 Le Mécanisme", "🌊 The Mechanism", "🌊 Der Mechanismus"))
            st.info(T(
                "Le TOF (Time Of Flight) est une séquence en **Écho de Gradient** qui sature les tissus fixes (Signal Noir) et laisse briller le sang frais (Signal Blanc).",
                "TOF (Time Of Flight) is a **Gradient Echo** sequence that saturates static tissues (Black Signal) and highlights fresh blood (White Signal).",
                "TOF (Time Of Flight) ist eine **Gradientenecho**-Sequenz, die stationäres Gewebe sättigt (schwarzes Signal) und frisches Blut hervorhebt (weißes Signal)."
            ))
            st.markdown(T("""
            **La Recette du Contraste :**
            1.  **Saturation :** Des impulsions rapides saturent l'aimantation des tissus immobiles.
            2.  **Inflow (Entrée) :** Le sang "frais" (non saturé) pénètre dans la coupe.
            3.  **Flash :** Il émet un fort signal avant d'être saturé à son tour.
            """, """
            **The Contrast Recipe:**
            1.  **Saturation:** Rapid pulses saturate the magnetization of static tissues.
            2.  **Inflow:** "Fresh" (unsaturated) blood enters the slice.
            3.  **Flash:** It emits a strong signal before becoming saturated itself.
            """, """
            **Das Kontrastrezept:**
            1.  **Sättigung:** Schnelle Pulse sättigen die Magnetisierung von unbewegtem Gewebe.
            2.  **Inflow (Einstrom):** "Frisches" (ungesättigtes) Blut fließt in die Schicht.
            3.  **Flash:** Es sendet ein starkes Signal aus, bevor es selbst gesättigt wird.
            """))

        with col_sci2:
            st.markdown("### " + T("⚠️ Paramètres & Limites", "⚠️ Parameters & Limits", "⚠️ Parameter & Grenzen"))
            st.warning(T("""
            **Les 3 Ennemis du TOF :**
            1.  **Flux Lent :** Si le sang stagne, il sature $\\to$ Devient Noir (Faux Thrombus).
            2.  **Flux dans le Plan :** Un vaisseau parallèle à la coupe sature $\\to$ Invisible.
            3.  **Thrombus Récent :** La Méthémoglobine (T1 court) brille spontanément $\\to$ Faux Flux.
            """, """
            **TOF's 3 Enemies:**
            1.  **Slow Flow:** Stagnant blood saturates $\\to$ Becomes Black.
            2.  **In-plane Flow:** Vessel parallel to slice saturates $\\to$ Invisible.
            3.  **Recent Thrombus:** Methemoglobin (short T1) shines spontaneously $\\to$ Fake Flow.
            """, """
            **Die 3 Feinde der TOF:**
            1.  **Langsamer Fluss:** Stagnierendes Blut sättigt $\\to$ Wird schwarz (Falscher Thrombus).
            2.  **In-Plane-Fluss:** Ein parallel zur Schicht verlaufendes Gefäß sättigt $\\to$ Unsichtbar.
            3.  **Frischer Thrombus:** Methämoglobin (kurzes T1) leuchtet spontan $\\to$ Falscher Fluss.
            """))
            
            st.markdown(T("""
            <div style="background-color:#1e1e1e; padding:5px; border-radius:5px; color:white; font-family:monospace; text-align:center; font-size: 0.8em;">
                🌊 SANG FRAIS (Mz Max) ===> ⬛ TISSUS SATURÉS <br>
                ⬇ <br>
                ✨ SIGNAL HYPERINTENSE
            </div>
            """, """
            <div style="background-color:#1e1e1e; padding:5px; border-radius:5px; color:white; font-family:monospace; text-align:center; font-size: 0.8em;">
                🌊 FRESH BLOOD (Max Mz) ===> ⬛ SATURATED TISSUES <br>
                ⬇ <br>
                ✨ HYPERINTENSE SIGNAL
            </div>
            """, """
            <div style="background-color:#1e1e1e; padding:5px; border-radius:5px; color:white; font-family:monospace; text-align:center; font-size: 0.8em;">
                🌊 FRISCHES BLUT (Max Mz) ===> ⬛ GESÄTTIGTES GEWEBE <br>
                ⬇ <br>
                ✨ HYPERINTENSES SIGNAL
            </div>
            """), unsafe_allow_html=True)

    st.divider()

    @st.cache_data
    def load_tof_image_turbo(path):
        if os.path.exists(path):
            img = Image.open(path).convert("RGB") 
            img.thumbnail((600, 600)) 
            return img
        return None

    col_ctrl, col_view = st.columns([1.3, 2.7])

    points_axial = {
        "Com. Ant":   [0.525, 0.320, "ACA", "AComA", T("AComA - Communicante Antérieure", "AComA - Anterior Communicating", "AComA - Arteria communicans anterior"), 270],
        "ACA A2 (D)": [0.500, 0.380, "ACA", "ACA (D)", T("ACA - Artère Cérébrale Ant. (D)", "ACA - Anterior Cerebral Art. (R)", "ACA - Arteria cerebri anterior (R)"), 250],
        "ACA A2 (G)": [0.550, 0.380, "ACA", "ACA (G)", T("ACA - Artère Cérébrale Ant. (G)", "ACA - Anterior Cerebral Art. (L)", "ACA - Arteria cerebri anterior (L)"), 290],
        "ACM M1 (D)": [0.380, 0.400, "ACM", "ACM (D)", T("ACM - Artère Cérébrale Moy. (D)", "MCA - Middle Cerebral Art. (R)", "MCA - Arteria cerebri media (R)"), 215],
        "ACM M1 (G)": [0.640, 0.420, "ACM", "ACM (G)", T("ACM - Artère Cérébrale Moy. (G)", "MCA - Middle Cerebral Art. (L)", "MCA - Arteria cerebri media (L)"), 325],
        "Com. Post (D)": [0.470, 0.436, "PCOM", "AComP (D)", T("AComP - Communicante Post. (D)", "PComA - Posterior Communicating (R)", "PComA - Arteria communicans posterior (R)"), 195],
        "Com. Post (G)": [0.544, 0.436, "PCOM", "AComP (G)", T("AComP - Communicante Post. (G)", "PComA - Posterior Communicating (L)", "PComA - Arteria communicans posterior (L)"), 345],
        "Carotide (D)": [0.345, 0.540, "ICA", "ACI (D)", T("ACI - Carotide Interne / Siphon (D)", "ICA - Internal Carotid / Siphon (R)", "ICA - Arteria carotis interna (R)"), 180],
        "Carotide (G)": [0.710, 0.540, "ICA", "ACI (G)", T("ACI - Carotide Interne / Siphon (G)", "ICA - Internal Carotid / Siphon (L)", "ICA - Arteria carotis interna (L)"), 0],
        "Basilaire":  [0.525, 0.530, "BAS", "TB",      T("TB - Tronc Basilaire", "BA - Basilar Artery", "BA - Arteria basilaris"), 90],
        "ACP P1 (D)": [0.400, 0.600, "ACP", "ACP (D)", T("ACP - Artère Cérébrale Post. (D)", "PCA - Posterior Cerebral Art. (R)", "PCA - Arteria cerebri posterior (R)"), 160],
        "ACP P1 (G)": [0.600, 0.600, "ACP", "ACP (G)", T("ACP - Artère Cérébrale Post. (G)", "PCA - Posterior Cerebral Art. (L)", "PCA - Arteria cerebri posterior (L)"), 20],
        "Vertébrale (D)": [0.450, 0.700, "BAS", "AV (D)", T("AV - Artère Vertébrale (D)", "VA - Vertebral Artery (R)", "VA - Arteria vertebralis (R)"), 135],
        "Vertébrale (G)": [0.570, 0.700, "BAS", "AV (G)", T("AV - Artère Vertébrale (G)", "VA - Vertebral Artery (L)", "VA - Arteria vertebralis (L)"), 45],
    }
    
    points_coronal = {
        "ACA (D)": [0.480, 0.250, "ACA", "ACA (D)", T("ACA - Artère Cérébrale Ant. (D)", "ACA - Anterior Cerebral Art. (R)", "ACA - Arteria cerebri anterior (R)"), 260], 
        "ACA (G)": [0.520, 0.250, "ACA", "ACA (G)", T("ACA - Artère Cérébrale Ant. (G)", "ACA - Anterior Cerebral Art. (L)", "ACA - Arteria cerebri anterior (L)"), 280],
        "ACM (D)": [0.350, 0.408, "ACM", "ACM (D)", T("ACM - Artère Cérébrale Moy. (D)", "MCA - Middle Cerebral Art. (R)", "MCA - Arteria cerebri media (R)"), 220], 
        "ACM (G)": [0.650, 0.422, "ACM", "ACM (G)", T("ACM - Artère Cérébrale Moy. (G)", "MCA - Middle Cerebral Art. (L)", "MCA - Arteria cerebri media (L)"), 320],
        "ACP (D)": [0.454, 0.482, "ACP", "ACP (D)", T("ACP - Artère Cérébrale Post. (D)", "PCA - Posterior Cerebral Art. (R)", "PCA - Arteria cerebri posterior (R)"), 200], 
        "ACP (G)": [0.528, 0.482, "ACP", "ACP (G)", T("ACP - Artère Cérébrale Post. (G)", "PCA - Posterior Cerebral Art. (L)", "PCA - Arteria cerebri posterior (L)"), 340],
        "Basilaire": [0.500, 0.626, "BAS", "TB",     T("TB - Tronc Basilaire", "BA - Basilar Artery", "BA - Arteria basilaris"), 90], 
        "Carotide (D)": [0.390, 0.600, "ICA", "ACI (D)", T("ACI - Carotide Interne (D)", "ICA - Internal Carotid (R)", "ICA - Arteria carotis interna (R)"), 180], 
        "Carotide (G)": [0.584, 0.600, "ICA", "ACI (G)", T("ACI - Carotide Interne (G)", "ICA - Internal Carotid (L)", "ICA - Arteria carotis interna (L)"), 0],
        "Vertébrale (D)": [0.450, 0.850, "BAS", "AV (D)", T("AV - Artère Vertébrale (D)", "VA - Vertebral Artery (R)", "VA - Arteria vertebralis (R)"), 150], 
        "Vertébrale (G)": [0.550, 0.850, "BAS", "AV (G)", T("AV - Artère Vertébrale (G)", "VA - Vertebral Artery (L)", "VA - Arteria vertebralis (L)"), 30],
    }

    with col_ctrl:
        st.subheader(T("Paramètres", "Parameters", "Parameter"))
        view_mode = st.radio(T("Plan de Coupe :", "Slice Plane:", "Schnittebene:"), ["AXIAL", "CORONAL"], label_visibility="collapsed")
        
        st.divider()
        
        active_dict = points_axial if view_mode == "AXIAL" else points_coronal
        all_keys = list(active_dict.keys())
        
        ms_key = f"ms_sel_turbo_{view_mode}"
        if ms_key not in st.session_state:
            st.session_state[ms_key] = []

        c1, c2 = st.columns(2)
        if c1.button(T("👁️ Tout Voir", "👁️ Show All", "👁️ Alles anzeigen"), use_container_width=True):
            st.session_state[ms_key] = all_keys
            st.rerun()
            
        if c2.button(T("❌ Cacher", "❌ Hide", "❌ Verbergen"), use_container_width=True):
            st.session_state[ms_key] = []
            st.rerun()

        with st.expander(T("🔍 Sélection Individuelle", "🔍 Individual Selection", "🔍 Individuelle Auswahl"), expanded=True):
            options_map = {k: v[4] for k, v in active_dict.items()}
            selected_keys = st.multiselect(
                T("Cochez les structures :", "Check structures:", "Strukturen ankreuzen:"), 
                options=all_keys,
                format_func=lambda x: options_map[x],
                key=ms_key,
                label_visibility="collapsed"
            )

    with col_view:
        f_name = "tof_ax.png" if view_mode == "AXIAL" else "tof_coro.png"
        img_path = os.path.join(current_dir, f_name)
        
        img_pil = load_tof_image_turbo(img_path)
        
        if img_pil:
            w_img, h_img = img_pil.size
            fig = px.imshow(img_pil, binary_string=True)
            
            if view_mode == "AXIAL":
                lbl_top, lbl_bottom, lbl_left, lbl_right = ("A", "P", "D", "G") if st.session_state.lang == 'fr' else ("A", "P", "R", "L")
            else:
                lbl_top, lbl_bottom, lbl_left, lbl_right = ("H", "B", "D", "G") if st.session_state.lang == 'fr' else ("S", "I", "R", "L")
            
            compass_color = "rgba(255, 255, 0, 0.9)" 
            cx, cy = 0.15, 0.88 
            
            compass_annotations = [
                dict(x=cx, y=cy+0.08, xref="paper", yref="paper", text=lbl_top, showarrow=False, font=dict(color=compass_color, size=14, weight="bold")),
                dict(x=cx, y=cy-0.08, xref="paper", yref="paper", text=lbl_bottom, showarrow=False, font=dict(color=compass_color, size=14, weight="bold")),
                dict(x=cx-0.04, y=cy, xref="paper", yref="paper", text=lbl_left, showarrow=False, font=dict(color=compass_color, size=14, weight="bold")),
                dict(x=cx+0.04, y=cy, xref="paper", yref="paper", text=lbl_right, showarrow=False, font=dict(color=compass_color, size=14, weight="bold")),
                dict(x=cx, y=cy+0.05, xref="paper", yref="paper", ax=0, ay=20, axref="pixel", ayref="pixel", showarrow=True, arrowhead=2, arrowcolor=compass_color, arrowwidth=2),
                dict(x=cx, y=cy-0.05, xref="paper", yref="paper", ax=0, ay=-20, axref="pixel", ayref="pixel", showarrow=True, arrowhead=2, arrowcolor=compass_color, arrowwidth=2),
                dict(x=cx+0.025, y=cy, xref="paper", yref="paper", ax=-20, ay=0, axref="pixel", ayref="pixel", showarrow=True, arrowhead=2, arrowcolor=compass_color, arrowwidth=2),
                dict(x=cx-0.025, y=cy, xref="paper", yref="paper", ax=20, ay=0, axref="pixel", ayref="pixel", showarrow=True, arrowhead=2, arrowcolor=compass_color, arrowwidth=2),
            ]

            cmap = { "ACA": "#00d2d3", "ACM": "#2ecc71", "ACP": "#e74c3c", "BAS": "#e67e22", "ICA": "#9b59b6", "PCOM": "#bdc3c7" }
            import math

            radius_x_std = 0.45 
            radius_y_std = 0.42
            center_x, center_y = 0.5, 0.5

            for name in selected_keys:
                data = active_dict[name]
                x_anat = data[0] * w_img
                y_anat = data[1] * h_img
                group = data[2]
                acronym = data[3]
                full_desc = data[4]
                angle_deg = data[5]
                color = cmap.get(group, "white")
                
                curr_rad_y = radius_y_std
                if name == "Com. Ant":
                    curr_rad_y = radius_y_std * 0.75
                
                angle_rad = math.radians(angle_deg)
                dx = math.cos(angle_rad) * radius_x_std
                dy = math.sin(angle_rad) * curr_rad_y
                
                x_lbl = (center_x + dx) * w_img
                y_lbl = (center_y + dy) * h_img
                
                x_lbl = max(0.05*w_img, min(0.95*w_img, x_lbl))
                y_lbl = max(0.05*h_img, min(0.95*h_img, y_lbl))
                
                fig.add_trace(go.Scatter(x=[x_anat], y=[y_anat], mode='markers', marker=dict(size=9, color=color, line=dict(width=1, color='white')), hoverinfo='text', text=full_desc, showlegend=False))
                fig.add_trace(go.Scatter(x=[x_anat, x_lbl], y=[y_anat, y_lbl], mode='lines', line=dict(color=color, width=1, dash='dot'), hoverinfo='skip', showlegend=False))
                txt_anchor = "left" if (x_lbl/w_img) > 0.5 else "right"
                txt_vert = "middle"
                if (y_lbl/h_img) < 0.1: txt_vert = "bottom"
                if (y_lbl/h_img) > 0.9: txt_vert = "top"
                
                fig.add_trace(go.Scatter(
                    x=[x_lbl], y=[y_lbl], mode='text',
                    text=f"<b>{acronym}</b>",
                    textposition=f"{txt_vert} {txt_anchor}",
                    textfont=dict(color=color, size=15, family="Arial Black", shadow="auto"),
                    hoverinfo='text', hovertext=full_desc,
                    showlegend=False
                ))

            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(visible=False, range=[0, w_img]),
                yaxis=dict(visible=False, range=[h_img, 0]),
                yaxis_scaleanchor="x",
                dragmode='pan',
                hovermode='closest',
                annotations=compass_annotations,
                showlegend=False
            )
            
            st.plotly_chart(fig, config={'displayModeBar': False, 'scrollZoom': True}, use_container_width=True)

        else:
            st.error(f"Image '{f_name}' introuvable.")

elif module_actif == liste_modules[15]:
    render_fatsat_tab()
elif module_actif == liste_modules[16]:
    render_safety_tab()

elif module_actif == liste_modules[17]:
    render_architecture_tab()