import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import streamlit.components.v1 as components
import os
import base64
from PIL import Image
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

# --- 🌐 GESTION LANGUE (MOTEUR) ---
if 'lang' not in st.session_state:
    st.session_state.lang = 'fr'

def T(fr, en):
    """Renvoie le texte français ou anglais selon l'état."""
    return fr if st.session_state.lang == 'fr' else en

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
    # Si on est en français, on renvoie le nom original (source)
    if st.session_state.lang == 'fr':
        return name
    
    # Si on est en anglais, on analyse le nom pour trouver la correspondance
    n = name.lower() # On met tout en minuscule pour comparer sans erreur
    
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
    
    # Si rien n'est trouvé, on renvoie le nom original
    return name

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

# ==============================================================================
# 🎛️ BARRE LATÉRALE (SIDEBAR)
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "logo_mia.png")

# Images Drapeaux (Noms issus de vos fichiers uploadés)
flag_fr_path = os.path.join(current_dir, "image_15c952.png")
flag_uk_path = os.path.join(current_dir, "image_15c972.png")
# Fallback noms standards
if not os.path.exists(flag_fr_path): flag_fr_path = os.path.join(current_dir, "flag_fr.png")
if not os.path.exists(flag_uk_path): flag_uk_path = os.path.join(current_dir, "flag_uk.png")

if os.path.exists(logo_path): 
    st.sidebar.image(logo_path, width=280)

st.sidebar.title(T("Réglages Console", "Console Settings"))

# --- ZONE ACTIONS COMPACTE (Reset + Langues) ---
c_reset, c_space, c_fr, c_uk = st.sidebar.columns([1.3, 0.1, 0.8, 0.8])

with c_reset:
    st.write("") 
    st.write("")
    if st.button(T("⚠️ Reset", "⚠️ Reset"), use_container_width=True):
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
st.sidebar.header(T("1. Géométrie", "1. Geometry"))
col_ep, col_slice = st.sidebar.columns(2)

ep = col_ep.number_input(T("Epaisseur (mm)", "Slice Thick. (mm)"), min_value=1.0, max_value=10.0, value=4.0, step=0.5, key=f"ep_{current_reset_id}")
n_slices = col_slice.slider(T("Nb Coupes", "Slices"), 1, 100, 20, step=1, key=f"ns_{current_reset_id}")

if not is_dwi:
    n_concats = st.sidebar.select_slider(T("📚 Concaténations", "📚 Concatenations"), options=[1, 2, 3, 4], value=1, key=f"concat_{current_reset_id}")
else: 
    n_concats = 1

fov = st.sidebar.slider("FOV (mm)", 100.0, 500.0, 240.0, step=10.0, key=f"fov_{current_reset_id}")

# --- CORRECTION MATRICE ---
# On définit l'index par défaut : 1 (pour 128) si DWI, sinon 2 (pour 256)
idx_mat_def = 1 if is_dwi else 2
options_mat = [64, 128, 256, 512]

# Astuce : on ajoute 'is_dwi' dans la 'key' pour forcer Streamlit à réinitialiser 
# le slider quand on change de mode (sinon il garde 256 en mémoire).
mat = st.sidebar.select_slider(
    T("Matrice", "Matrix"), 
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
st.sidebar.header(T("2. Chrono (ms)", "2. Timing (ms)"))
b_value = 0; show_stroke = False; show_atrophy = False; show_adc_map = False; show_microbleeds = False; pld = 1500 

def update_tr_from_slider():
    st.session_state.tr_force = st.session_state.widget_tr

if is_dwi:
    b_value = st.sidebar.select_slider(T("Facteur b", "b-Value"), options=[0, 500, 1000], value=0, key=f"bval_{current_reset_id}")
    tr = 6000.0; te = 90.0; ti = 0.0; flip_angle = 90
    st.sidebar.info(T("TR fixé : 6000ms | TE fixé : 90ms", "Fixed TR: 6000ms | Fixed TE: 90ms"))
    show_stroke = st.sidebar.checkbox(T("Simuler AVC", "Simulate Stroke"), False, key=f"avc_{current_reset_id}")
    show_adc_map = st.sidebar.checkbox(T("Carte ADC", "ADC Map"), False, key=f"adc_{current_reset_id}")
elif is_asl:
    pld = st.sidebar.slider("PLD", 500, 3000, 1800, step=100, key=f"pld_{current_reset_id}")
    tr = st.sidebar.slider("TR", 3000.0, 8000.0, 4500.0, step=100.0, key=f"tr_asl_{current_reset_id}")
    te = 15.0; ti = 0.0; flip_angle = 90
    show_stroke = st.sidebar.checkbox(T("AVC", "Stroke"), False, key=f"asl_avc_{current_reset_id}")
    st.session_state.atrophy_active = st.sidebar.checkbox(T("Atrophie", "Atrophy"), st.session_state.atrophy_active, key=f"asl_atr_{current_reset_id}")
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
                f"⚠️ Auto Adjusted TR<br>({int(min_tr_required)}ms) for {n_slices} slices.")
        st.sidebar.markdown(f"""<div class="tr-alert-box">{msg}</div>""", unsafe_allow_html=True)
    elif ('T1' in seq_choix and tr > 700):
        st.sidebar.markdown(f"""<div class="tr-alert-box">{T("⚠️ Attention Dépassement T1", "⚠️ T1 Limit Exceeded")}</div>""", unsafe_allow_html=True)

    if n_concats > 1:
        tr_opti = np.ceil(min_tr_required / 10) * 10
        if tr > (tr_opti + 100):
            def set_optimized_tr(val):
                st.session_state.tr_force = val
                st.session_state.widget_tr = val
            msg_opt = T("Optimisation", "Optimize")
            st.sidebar.markdown(f"""<div class="opt-box"><b>{msg_opt} {n_concats} Concats</b><br>TR Min : <b>{int(tr_opti)} ms</b></div>""", unsafe_allow_html=True)
            st.sidebar.button(f"📉 {T('Appliquer', 'Apply')} TR {int(tr_opti)} ms", on_click=set_optimized_tr, args=(tr_opti,))

    if is_ir: ti = st.sidebar.slider("TI", 0.0, 3500.0, float(defaults['ti']), step=10.0, key=f"ti_{current_reset_id}")
    else: ti = 0.0
    
    if is_gre: flip_angle = st.sidebar.slider(T("Angle (°)", "Flip Angle (°)"), 5, 90, 15, key=f"fa_{current_reset_id}")
    elif is_swi: flip_angle = st.sidebar.slider(T("Angle (°)", "Flip Angle (°)"), 5, 40, 15, key=f"fa_{current_reset_id}"); show_microbleeds = st.sidebar.checkbox(T("Micro-saignements", "Microbleeds"), False, key=f"cmb_{current_reset_id}")
    else: flip_angle = 90

# --- 3. OPTIONS ---
st.sidebar.header(T("3. Options", "3. Options"))
nex = st.sidebar.slider("NEX", 1, 8, 1, key=f"nex_{current_reset_id}")

# Mémoire Turbo
turbo = 1
if not (is_gre or is_dwi or is_swi or is_asl):
    def_turbo = st.session_state.mem_turbo
    turbo = st.sidebar.slider(T("Facteur Turbo", "Turbo Factor"), 1, 32, def_turbo, key=f"turbo_{current_reset_id}")
    st.session_state.mem_turbo = turbo

bw = st.sidebar.slider(T("Bande Passante", "Bandwidth"), 50, 500, 220, 10, key=f"bw_{current_reset_id}")
es = st.sidebar.slider(T("Espace Inter-Echo (ES)", "Echo Spacing (ES)"), 2.5, 20.0, 10.0, step=2.5, key=f"es_{current_reset_id}")

# --- 4. IMAGERIE PARALLÈLE ---
st.sidebar.header(T("4. Imagerie Parallèle (iPAT)", "4. Parallel Imaging (iPAT)"))
ipat_on = st.sidebar.checkbox(T("Activer Accélération", "Enable Acceleration"), value=False, key=f"ipat_on_{current_reset_id}")
ipat_factor = st.sidebar.slider(T("Facteur R", "R Factor"), 2, 4, 2, key=f"ipat_r_{current_reset_id}") if ipat_on else 1

st.sidebar.markdown("---")

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

# --- 2. CALCUL DES SIGNAUX ---
v_lcr = phy.calculate_signal(tr_effective, te, ti, cst.T_LCR['T1'], cst.T_LCR['T2'], cst.T_LCR['T2s'], cst.T_LCR['ADC'], cst.T_LCR['PD'], flip_angle, is_gre, is_dwi, b_value if is_dwi else 0)
# ... le reste du code des signaux reste identique ...
v_wm  = phy.calculate_signal(tr_effective, te, ti, cst.T_WM['T1'], cst.T_WM['T2'], cst.T_WM['T2s'], cst.T_WM['ADC'], cst.T_WM['PD'], flip_angle, is_gre, is_dwi, b_value if is_dwi else 0)

v_gm  = phy.calculate_signal(tr_effective, te, ti, cst.T_GM['T1'], cst.T_GM['T2'], cst.T_GM['T2s'], cst.T_GM['ADC'], cst.T_GM['PD'], flip_angle, is_gre, is_dwi, b_value if is_dwi else 0)

v_stroke = phy.calculate_signal(tr_effective, te, ti, cst.T_STROKE['T1'], cst.T_STROKE['T2'], cst.T_STROKE['T2s'], cst.T_STROKE['ADC'], cst.T_STROKE['PD'], flip_angle, is_gre, is_dwi, b_value if is_dwi else 0)

# Modification pour DWI High-B
if is_dwi and b_value >= 1000 and show_stroke: v_stroke = 2.0 
v_fat = phy.calculate_signal(tr_effective, te, ti, cst.T_FAT['T1'], cst.T_FAT['T2'], cst.T_FAT['T2s'], cst.T_FAT['ADC'], cst.T_FAT['PD'], flip_angle, is_gre, is_dwi, 0) if not is_dwi else 0.0
# --- 3. CALCUL DU SNR (BLOC MANQUANT) ---
snr_tr_ref = float(defaults['tr'])
snr_te_ref = float(defaults['te'])

# Calcul du signal de référence pour le SNR
ref_wm_signal = phy.calculate_signal(snr_tr_ref, snr_te_ref, ti, cst.T_WM['T1'], cst.T_WM['T2'], cst.T_WM['T2s'], cst.T_WM['ADC'], cst.T_WM['PD'], 90, False, False, 0)

# Protection contre la division par zéro
if ref_wm_signal == 0: ref_wm_signal = 0.001 

# Calcul final de la variable snr_val
snr_val = phy.calculate_snr_relative(mat, nex, turbo, ipat_factor, bw, fov, ep, v_wm, ref_wm_signal)
snr_val = snr_val * 1.25  # <--- AJOUTE CETTE LIGNE (Correction 4mm = 100%)
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
    val_lcr = v_lcr if v_lcr > 0 else 1.0 # Sécurité anti-noir
    val_wm  = v_wm  if v_wm  > 0 else 0.6
    val_gm  = v_gm  if v_gm  > 0 else 0.8
    val_fat = v_fat 
    val_stroke = v_stroke

    if is_dwi and show_adc_map:
        val_lcr = 1.0; val_wm = 0.3; val_gm = 0.35; val_stroke = 0.15; val_fat = 0.0

    img_water[D < 0.20] = val_lcr
    img_water[(D >= 0.20) & (D < 0.50)] = val_wm
    img_water[(D >= 0.50) & (D < 0.80)] = val_gm
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

# 6. Espace K (FFT sur le complexe)
f = np.fft.fftshift(np.fft.fft2(final_complex))
kspace = 20 * np.log(np.abs(f) + 1)

# ==============================================================================
# GENERATION FANTOME / PHANTOM GENERATION (VERSION ROBUSTE & INVISIBLE)
# ==============================================================================

# 1. LOGIQUE AUTOMATIQUE (SANS MENU VISUEL)
# On détermine quel fantôme dessiner en fonction de la séquence choisie en haut
current_seq_name = st.session_state.get('seq', '')

if "SWI" in current_seq_name:
    phantom_choice = "Dipôle"   # Force le dipôle pour le SWI
elif "Gel" in current_seq_name or "Bottle" in current_seq_name:
    phantom_choice = "Bouteille" # Au cas où vous auriez une séquence test
else:
    phantom_choice = "Cerveau"  # Cerveau pour tout le reste (T1, T2, FLAIR, etc.)

# 2. INITIALISATION DES MATRICES
S = mat
x = np.linspace(-1, 1, S)
y = np.linspace(-1, 1, S)
X, Y = np.meshgrid(x, y)
D = np.sqrt(X**2 + Y**2)

img_water = np.zeros((S, S))
img_fat = np.zeros((S, S))
phase_map = np.zeros((S, S)) # Indispensable pour l'onglet SWI
dipole_field = np.zeros((S, S)) # Indispensable pour la visualisation

# --- LOGIQUE DE DESSIN ---

# CAS 1 : DIPÔLE (SWI)
if phantom_choice == "Dipôle":
    mask_sphere = D < 0.6
    img_water[mask_sphere] = 0.8
    
    # Calcul du champ dipolaire
    center = S // 2
    y_idx, x_idx = np.ogrid[:S, :S]
    dist_px = np.sqrt((x_idx - center)**2 + (y_idx - center)**2) + 1e-6
    
    cos_theta = (y_idx - center) / dist_px
    raw_dipole = (3 * cos_theta**2 - 1) / (dist_px**3)
    
    # On limite les valeurs et on stocke pour l'onglet SWI
    dipole_field = np.clip(raw_dipole * 1000, -np.pi, np.pi)
    
    # Le "caillot" central
    core_mask = dist_px < (S * 0.05)
    img_water[core_mask] = 0.0
    
    # On remplit la carte de phase
    phase_map = dipole_field * 5.0

# CAS 2 : BOUTEILLE
elif phantom_choice == "Bouteille":
    mask_rect = (np.abs(X) < 0.4) & (np.abs(Y) < 0.7)
    img_water[mask_rect] = 0.9
    # Bulles
    for bx, by, br in [(0.1, 0.2, 0.05), (-0.1, -0.3, 0.08), (0.2, -0.5, 0.04)]:
        mask_bubble = np.sqrt((X-bx)**2 + (Y-by)**2) < br
        img_water[mask_bubble] = 0.0

# CAS 3 : CERVEAU (DÉFAUT)
else:
    # Récupération des signaux calculés plus haut
    val_lcr = v_lcr
    val_wm = v_wm
    val_gm = v_gm
    val_stroke = v_stroke
    val_fat = v_fat

    # Modification des contrastes pour la carte ADC (DWI)
    if is_dwi and show_adc_map:
        val_lcr = 1.0; val_wm = 0.3; val_gm = 0.35; val_stroke = 0.15; val_fat = 0.0

    # Dessin des cercles concentriques
    img_water[D < 0.20] = val_lcr
    img_water[(D >= 0.20) & (D < 0.50)] = val_wm
    img_water[(D >= 0.50) & (D < 0.80)] = val_gm
    img_fat[(D >= 0.80) & (D < 0.95)] = val_fat

    if show_stroke: 
        lesion_mask = (np.sqrt((X-0.3)**2 + (Y-0.1)**2) < 0.12)
        mask_valid = lesion_mask & (D >= 0.20)
        img_water[mask_valid] = val_stroke

# --- ASSEMBLAGE FINAL ---

# 1. Gestion du Chemical Shift
shift_pixels = 0.0 if bw == 220 else 220.0 / float(bw)
img_fat_shifted = shift(img_fat, [0, shift_pixels], mode='constant', cval=0.0)

# 2. Image Magnitude
magn_image = np.clip(img_water + img_fat_shifted, 0, 1.3)

# 3. Ajout de la phase et passage en complexe
complex_image = magn_image * np.exp(1j * phase_map)

# 4. Ajout du Bruit
noise_level = 5.0 / (snr_val + 20.0)
n_real = np.random.normal(0, noise_level, (S, S))
n_imag = np.random.normal(0, noise_level, (S, S))
final_complex = complex_image + (n_real + 1j * n_imag)

# 5. Image Finale affichée (Module)
final = np.abs(final_complex)
final = np.clip(final, 0, 1.3)

# 6. Espace K (FFT)
f = np.fft.fftshift(np.fft.fft2(final_complex))
kspace = 20 * np.log(np.abs(f) + 1)

# --- 13. AFFICHAGE FINAL / FINAL DISPLAY ---
st.title(T("Simulateur MagnétoVault", "MagnetoVault Simulator"))

# DÉFINITION DES ONGLETS / TABS DEFINITION
t_home, t1, t2, t3, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t_tof, t15, t16, t17 = st.tabs([
    T("🏠 Accueil", "🏠 Home"), T("Fantôme", "Phantom"), T("🌀 Espace K & Codage", "🌀 K-Space & Encoding"), 
    T("Signaux", "Signals"), T("🧠 Anatomie", "🧠 Anatomy"), T("📈 Physique", "📈 Physics"), 
    T("⚡ Chronogramme", "⚡ Timing Diagram"), T("☣️ Artefacts", "☣️ Artifacts"), 
    T("🚀 Imagerie Parallèle", "🚀 Parallel Imaging"), T("🧬 Diffusion", "🧬 Diffusion"), 
    T("🎓 Cours", "🎓 Course"), T("🩸 SWI & Dipôle", "🩸 SWI & Dipole"), 
    T("3D T1 (MP-RAGE)", "3D T1 (MP-RAGE)"), T("ASL (Perfusion)", "ASL (Perfusion)"), 
    T("🩸 Angio TOF", "🩸 Angio TOF"), T("🍔 Fat Sat", "🍔 Fat Sat"),
    T("🔥 Sécurité (SAR/B1+RMS)", "🔥 Safety (SAR/B1+RMS)"),
    T("🏗️ Architecture", "🏗️ Architecture") # <--- Nouvel onglet
])

# [TAB 0 : ACCUEIL / HOME]
with t_home:
    st.markdown(T("""
    <div style="background-color:#1e293b; padding:20px; border-radius:10px; margin-bottom:20px;">
        <h1 style="color:white; margin:0;">🧲 MagnétoVault Simulator</h1>
        <h3 style="color:#a5b4fc; margin-top:5px;">La "Boîte Blanche" de l'IRM</h3>
        <p style="color:#cbd5e1;"><i>"Ne vous contentez pas de voir l'image. Comprenez la mécanique de sa création."</i></p>
    </div>
    """, """
    <div style="background-color:#1e293b; padding:20px; border-radius:10px; margin-bottom:20px;">
        <h1 style="color:white; margin:0;">🧲 MagnétoVault Simulator</h1>
        <h3 style="color:#a5b4fc; margin-top:5px;">The MRI "White Box"</h3>
        <p style="color:#cbd5e1;"><i>"Don't just see the image. Understand the mechanics of its creation."</i></p>
    </div>
    """), unsafe_allow_html=True)

    c_intro1, c_intro2 = st.columns([1, 1])
    with c_intro1:
        st.markdown(T("### 🔍 Pourquoi ce simulateur est unique ?", "### 🔍 Why is this simulator unique?"))
        st.markdown(T("""
        La plupart des simulateurs sont des "boîtes noires" : vous rentrez des paramètres, une image sort, mais vous ne savez pas pourquoi.
        
        **MagnétoVault est un laboratoire transparent.** Ici, nous ouvrons le capot de la machine pour vous montrer les mathématiques et la physique en action.
        """, """
        Most simulators are "black boxes": you enter parameters, an image comes out, but you don't know why.
        
        **MagnétoVault is a transparent laboratory.** Here, we open the hood of the machine to show you the mathematics and physics in action.
        """))
    with c_intro2:
        st.info(T("""
        **Objectif :** Faire le lien entre la **Physique** (Spin, Vecteurs), l'**Espace K** (Fourier) et l'**Image Clinique** (Contraste).
        """, """
        **Goal:** Bridge the gap between **Physics** (Spin, Vectors), **K-Space** (Fourier), and **Clinical Image** (Contrast).
        """))

    st.divider()

    st.markdown(T("### 🧪 Ce que vous pouvez explorer", "### 🧪 What you can explore"))
    
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        st.markdown(T("#### 1. Mécanique de l'Espace K", "#### 1. K-Space Mechanics"))
        st.markdown(T("""
        Visualisez l'invisible. Comment la machine remplit-elle les lignes ?
        * **Facteur Turbo (TSE) :** Voyez comment les trains d'échos sont rangés. Lequel porte le contraste ? Lequel donne les détails ?
        * **TE Effectif :** Comprenez pourquoi il est placé au centre de l'espace K.
        """, """
        Visualize the invisible. How does the machine fill the lines?
        * **Turbo Factor (TSE):** See how echo trains are ordered. Which one carries contrast? Which one gives details?
        * **Effective TE:** Understand why it is placed at the center of K-space.
        """))
    with col_p2:
        st.markdown(T("#### 2. Physique Temps Réel", "#### 2. Real-Time Physics"))
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
        """))
    with col_p3:
        st.markdown(T("#### 3. Clinique Avancée", "#### 3. Advanced Clinical"))
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
        """))

    st.divider()
    st.markdown(T("### 🚀 Guide de Démarrage", "### 🚀 Quick Start Guide"))
    st.markdown(T("""
    1.  **🎛️ Console (Gauche) :** C'est votre poste de pilotage. Choisissez la **Séquence**, réglez le **FOV**, la **Matrice**, le **TR/TE** et le **Facteur Turbo**.
    2.  **🌀 Espace K (Onglet 2) :** Regardez comment votre séquence remplit les données brutes.
    3.  **🧠 Anatomie (Onglet 5) :** Explorez un cerveau humain réel (Atlas *Harvard-Oxford*) et simulez des pathologies (**AVC**, **Atrophie**).
    """, """
    1.  **🎛️ Console (Left):** This is your cockpit. Choose the **Sequence**, adjust **FOV**, **Matrix**, **TR/TE**, and **Turbo Factor**.
    2.  **🌀 K-Space (Tab 2):** Watch how your sequence fills the raw data.
    3.  **🧠 Anatomy (Tab 5):** Explore a real human brain (*Harvard-Oxford* Atlas) and simulate pathologies (**Stroke**, **Atrophy**).
    """))

    st.divider()
    
    # --- GLOSSAIRE DÉPLOYABLE / EXPANDABLE GLOSSARY ---
    with st.expander(T("📖 Glossaire Complet (Variables & Formules)", "📖 Complete Glossary (Variables & Formulas)"), expanded=False):
        
        # 1. PHYSIQUE FONDAMENTALE
        st.markdown(T("### 🧲 1. Physique Fondamentale", "### 🧲 1. Fundamental Physics"))
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
            """))
        with col_phy2:
            st.markdown(T("""
            * **$M_0$** : Aimantation nette à l'équilibre.
            * **$M_z$** : Aimantation longitudinale (T1).
            * **$M_{xy}$** : Aimantation transversale (T2).
            """, """
            * **$M_0$**: Net magnetization at equilibrium.
            * **$M_z$**: Longitudinal magnetization (T1).
            * **$M_{xy}$**: Transverse magnetization (T2).
            """))

        st.markdown("---")

        # 2. PROPRIÉTÉS TISSULAIRES
        st.markdown(T("### 🧠 2. Propriétés Tissulaires", "### 🧠 2. Tissue Properties"))
        col_tis1, col_tis2 = st.columns(2)
        with col_tis1:
            st.markdown(T("""
            * **$T1$ (ms)** : Relaxation longitudinale (Spin-Réseau).
            * **$T2$ (ms)** : Relaxation transversale (Spin-Spin).
            """, """
            * **$T1$ (ms)**: Longitudinal relaxation (Spin-Lattice).
            * **$T2$ (ms)**: Transverse relaxation (Spin-Spin).
            """))
        with col_tis2:
            st.markdown(T("""
            * **$T2^*$ (ms)** : T2 réel + Inhomogénéités de champ.
            * **$\rho$ (DP)** : Densité de Protons (quantité d'H+).
            """, """
            * **$T2^*$ (ms)**: True T2 + Field inhomogeneities.
            * **$\rho$ (PD)**: Proton Density (amount of H+).
            """))

        st.markdown("---")

        # 3. PARAMÈTRES SÉQUENCE
        st.markdown(T("### ⏱️ 3. Paramètres Séquence", "### ⏱️ Sequence Parameters"))
        col_seq1, col_seq2 = st.columns(2)
        with col_seq1:
            st.markdown(T("""
            * **$TR$ (ms)** : Temps de Répétition.
            * **$TE$ (ms)** : Temps d'Écho.
            * **$TI$ (ms)** : Temps d'Inversion.
            """, """
            * **$TR$ (ms)**: Repetition Time.
            * **$TE$ (ms)**: Echo Time.
            * **$TI$ (ms)**: Inversion Time.
            """))
        with col_seq2:
            st.markdown(T("""
            * **$alpha$ (Flip Angle)** : Angle de bascule RF.
            * **$ETL$** : Echo Train Length (Facteur Turbo).
            * **$BW$ (Hz/Px)** : Bande Passante.
            """, """
            * **$alpha$ (Flip Angle)**: RF Flip Angle.
            * **$ETL$**: Echo Train Length (Turbo Factor).
            * **$BW$ (Hz/Px)**: Bandwidth.
            """))

        st.markdown("---")

        # 4. SÉCURITÉ
        st.markdown(T("### 🔥 4. Sécurité", "### 🔥 4. Safety"))
        col_sar1, col_sar2 = st.columns(2)
        with col_sar1:
            st.markdown(T("""
            * **$B_1^{+RMS}$ ($\mu T$)** : Moyenne champ RF (Risque Implants).
            * **$B_{1,peak}$** : Amplitude max instantanée.
            """, """
            * **$B_1^{+RMS}$ ($\mu T$)**: RMS RF field (Implant Risk).
            * **$B_{1,peak}$**: Peak instantaneous amplitude.
            """))
        with col_sar2:
            st.markdown(T("""
            * **$SAR$ (W/kg)** : Énergie absorbée par le patient (Chauffe).
            * **$DC$ (%)** : Duty Cycle (Rapport Cyclique).
            """, """
            * **$SAR$ (W/kg)**: Specific Absorption Rate (Patient heating).
            * **$DC$ (%)**: Duty Cycle.
            """))

# [TAB 1 : FANTOME / PHANTOM]
with t1:
    # =========================================================
    # 1. SETUP & PARAMÈTRES
    # =========================================================
    def get_p(name, def_val): return getattr(cst, name, def_val)

    T_WM = get_p('T_WM', {'T1': 600, 'T2': 80, 'PD': 0.7, 'ADC': 0.7e-3})
    T_GM = get_p('T_GM', {'T1': 1100, 'T2': 100, 'PD': 0.8, 'ADC': 0.8e-3})
    T_CSF = get_p('T_LCR', {'T1': 4000, 'T2': 2000, 'PD': 1.0, 'ADC': 3.0e-3})
    T_FAT = get_p('T_FAT', {'T1': 250, 'T2': 60, 'PD': 0.9, 'ADC': 0})
    T_STROKE = get_p('T_STROKE', {'T1': 1100, 'T2': 200, 'PD': 0.9, 'ADC': 0.4e-3})

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

    # TA prend en compte iPAT (Divise le temps par R)
    raw_time_ms = (tr * mat * nex * final_concats) / (turbo * ipat_factor)
    final_seconds = raw_time_ms / 1000.0
    str_duree = f"{int(final_seconds // 60)} min {int(final_seconds % 60)} s"
    
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
    f_sig = (v_wm / ref_sig) if ref_sig > 0 else 0
    
    # 3. iPAT (CORRECTION MAJEURE : FACTEUR G)
    # Formule SNR iPAT = SNR / (g * sqrt(R))
    # On simule un facteur g qui augmente avec l'accélération
    # R=2 -> g=1.2 | R=3 -> g=1.5 | R=4 -> g=2.0
    if ipat_factor > 1:
        # Simulation empirique du facteur g
        g_factor = 1.0 + (0.3 * (ipat_factor - 1))
        f_ipat = 1.0 / (g_factor * np.sqrt(ipat_factor))
    else:
        g_factor = 1.0
        f_ipat = 1.0

    # ... (Calculs précédents de f_vox, f_bw, f_nex...) ...

    # --- CALCUL DU SNR ---
    if is_dwi:
        # 1. Base 100% pour la matrice 128
        # On ignore les facteurs standards (f_vox, f_bw) pour cette séquence
        snr_base = 100.0 
        
        # 2. On ajuste si l'utilisateur change la matrice manuellement
        # Si Matrix=64 (SNR augmente), si Matrix=256 (SNR diminue)
        facteur_matrice = (128 / mat) ** 2
        
        # 3. Application de l'atténuation physique du b-value
        # À b=0 -> 100%, à b=1000 -> décroissance exponentielle
        snr_final = snr_base * facteur_matrice * np.exp(-b_value * 0.001)
    else:
        # Formule standard pour toutes les autres séquences (T1, T2, FLAIR...)
        snr_final = 100.0 * f_vox * f_bw * f_nex * f_turbo * f_sig * f_ipat

    # Affichage final
    str_snr = f"{snr_final:.1f} %"

    # =========================================================
    # 5. MOTEUR VISUEL
    # =========================================================
    import scipy.ndimage as ndimage
    
    # Génération Image
    S_render = 512 
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
    
    if turbo > 1:
        img_disp = ndimage.gaussian_filter(img_disp, sigma=(turbo-1)*0.2)

    # BRUIT VISUEL (Loi en 1/SNR²)
    # Si iPAT activé, le SNR chute -> Le bruit monte en flèche
    max_val = np.max(img_disp)
    if max_val > 0: img_disp /= max_val
    
    base_noise = 0.02
    sigma_noise = base_noise * ((100.0 / (snr_final + 0.1)) ** 1.5)
    sigma_noise = min(sigma_noise, 0.8)
    
    noise_map = np.random.normal(0, sigma_noise, (256, 256))
    img_final = np.clip(img_disp + noise_map, 0, 1)

   # ... (Ceci est à l'intérieur de with t1:) ...

    # =========================================================
    # 6. AFFICHAGE
    # =========================================================
    c1, c2 = st.columns([1, 1])
    
    with c1:
        k1, k2 = st.columns(2)
        k1.metric(T("⏱️ Durée", "⏱️ Duration"), str_duree)
        k2.metric(T("📉 SNR Relatif", "📉 Relative SNR"), str_snr)
        
        # Feedback visuel iPAT
        if ipat_factor > 1:
            st.warning(f"⚠️ iPAT x{ipat_factor} activé : Perte SNR importante (Facteur g={g_factor:.1f})")

        st.divider()
        st.subheader(T("1. Formules Physiques", "1. Physics Formulas"))
        
        # --- A. TEMPS D'ACQUISITION (TA) ---
        st.markdown(f"**A. {T('Temps d\'Acquisition', 'Acquisition Time')} (TA) :**")
        st.latex(r"TA = \frac{TR \times N_{PE} \times NEX}{ETL \times R}")
        
        # --- B. SNR ---
        st.markdown(f"**B. {T('Rapport Signal / Bruit', 'Signal-to-Noise Ratio')} (SNR) :**")
        st.latex(r"SNR \propto V_{vox} \times \sqrt{\frac{NEX}{BW}} \times \frac{1}{g \cdot \sqrt{R}}")

        # --- C. LÉGENDES DÉTAILLÉES (GLOSSAIRE COMPLET) ---
        with st.expander(T("📖 Légende des Variables (Cliquez)", "📖 Variable Legend (Click)"), expanded=False):
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
            """))

        # Glossaire Analyse d'Impact
        with st.expander(T("🔍 Analyse d'Impact", "🔍 Impact Analysis"), expanded=True):
            st.markdown(T("""
            * **iPAT ($R$)** : 📉 Divise le temps par $R$, mais le SNR chute de plus de $\sqrt{R}$.
            * **Matrice ($N_{PE}$)** : 🕰️ Augmente le temps (proportionnel) et 📉 Diminue le SNR (au carré).
            * **FOV** : 📈 Impact SNR énorme (au carré).
            """, """
            * **iPAT ($R$)**: 📉 Divides time by $R$, but SNR drops more than $\sqrt{R}$.
            * **Matrix ($N_{PE}$)**: 🕰️ Increases time (linear) and 📉 Decreases SNR (squared).
            * **FOV**: 📈 Huge impact on SNR (squared).
            """))
    with c2:
        st.write(T("🖼️ **Rendu Visuel**", "🖼️ **Visual Render**"))
        fig_p, ax_p = plt.subplots(figsize=(5, 5))
        ax_p.imshow(img_final, cmap='gray', vmin=0, vmax=1)
        ax_p.axis('off')
        
        # Info iPAT dans le titre
        info = f"SNR: {int(snr_final)}% | iPAT: {ipat_factor} (g={g_factor:.1f}) | Mat: {mat}"
        ax_p.set_title(info, fontsize=10, color="gray")
        
        # ANNOTATIONS RÉTABLIES (LCR, SB, SG, Graisse)
        if sigma_noise < 0.4:
            # LCR (Centre)
            ax_p.text(128, 128, "LCR", color='cyan', ha='center', va='center', fontweight='bold')
            # Substance Blanche (Rayon ~0.35 -> Y ~172)
            ax_p.text(128, 172, T("SB", "WM"), color='white', ha='center', va='center', fontweight='bold')
            # Substance Grise (Rayon ~0.65 -> Y ~210)
            ax_p.text(128, 210, T("SG", "GM"), color='white', ha='center', va='center', fontweight='bold')
            # Graisse (Périphérie)
            ax_p.text(128, 240, "GRAISSE", color='orange', ha='center', va='center', fontweight='bold')

        st.pyplot(fig_p, use_container_width=False)

# ==============================================================================
# [TAB 2 : ESPACE K - TERMINOLOGIE CORRIGÉE]
# ==============================================================================
with t2:
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
    """), unsafe_allow_html=True)
    
    # 2. EN-TÊTE PÉDAGOGIQUE
    with st.expander(T("🎶 Comprendre le Codage : De la Chorale à la Physique", "🎶 Understanding Encoding: From Choir to Physics"), expanded=True):
        c_txt1, c_txt2, c_txt3 = st.columns(3)
        
        with c_txt1:
            # LE PROBLÈME
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
            """), unsafe_allow_html=True)
            
        with c_txt2:
            # LA SOLUTION
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
            """), unsafe_allow_html=True)
            
        with c_txt3:
            # LE RÉSULTAT
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
    """), unsafe_allow_html=True)
    
    # Création des deux sous-onglets
    sub_tabs = st.tabs([T("👁️ Cycle de Codage (Visualisation)", "👁️ Encoding Cycle (Visualization)"), T("🎨 Espace K (Remplissage)", "🎨 K-Space (Filling)")])
    
    # SOUS-ONGLET 1 : CODAGE (HTML du main1.py)
    with sub_tabs[0]:
        st.markdown(T("<h3 style='color: #4f46e5; border-bottom: 2px solid #4f46e5; padding-bottom: 5px;'>🎛️ Simulateur de Codage</h3>", "<h3 style='color: #4f46e5; border-bottom: 2px solid #4f46e5; padding-bottom: 5px;'>🎛️ Encoding Simulator</h3>"), unsafe_allow_html=True)
        
        # HTML/JS AVEC TRADUCTION INJECTÉE
        components.html(T("""<!DOCTYPE html><html><head><style>body{margin:0;padding:5px;font-family:sans-serif;} .box{display:flex;gap:15px;} .ctrl{width:220px;padding:10px;background:#f9f9f9;border:1px solid #ccc;border-radius:8px;} canvas{border:1px solid #ccc;background:#f8f9fa;border-radius:8px;} input{width:100%;} label{font-size:11px;font-weight:bold;display:block;} button{width:100%;padding:8px;background:#4f46e5;color:white;border:none;border-radius:4px;cursor:pointer;}</style></head><body><div class='box'><div class='ctrl'><h4>Codage</h4><label>Freq</label><input type='range' id='f' min='-100' max='100' value='0'><br><label>Phase</label><input type='range' id='p' min='-100' max='100' value='0'><br><label>Coupe</label><input type='range' id='z' min='-100' max='100' value='0'><br><label>Matrice</label><input type='range' id='g' min='5' max='20' value='12'><br><button onclick='rst()'>Reset</button></div><div><canvas id='c1' width='350' height='350'></canvas><canvas id='c2' width='80' height='350'></canvas></div></div><script>const c1=document.getElementById('c1');const x=c1.getContext('2d');const c2=document.getElementById('c2');const z=c2.getContext('2d');const sf=document.getElementById('f');const sp=document.getElementById('p');const sz=document.getElementById('z');const sg=document.getElementById('g');const pd=30;function arrow(ctx,x,y,a,s){const l=s*0.35;ctx.save();ctx.translate(x,y);ctx.rotate(a);ctx.beginPath();ctx.moveTo(-l,0);ctx.lineTo(l,0);ctx.lineTo(l-6,-6);ctx.moveTo(l,0);ctx.lineTo(l-6,6);ctx.strokeStyle='white';ctx.lineWidth=1.5;ctx.stroke();ctx.restore();} function draw(){x.clearRect(0,0,350,350);z.clearRect(0,0,80,350);const fv=parseFloat(sf.value);const pv=parseFloat(sp.value);const zv=parseFloat(sz.value);const gs=parseInt(sg.value);const st=(350-2*pd)/gs;const h=(pd*0.8)*(fv/100);x.fillStyle='rgba(255,0,0,0.3)';if(fv!=0){x.beginPath();x.moveTo(pd,pd/2);x.lineTo(pd,pd/2-h);x.lineTo(350-pd,pd/2+h);x.lineTo(350-pd,pd/2);x.fill();}const w=(pd*0.8)*(pv/100);x.fillStyle='rgba(0,255,0,0.3)';if(pv!=0){x.beginPath();x.moveTo(350-pd/2,pd);x.lineTo(350-pd/2-w,pd);x.lineTo(350-pd/2+w,350-pd);x.lineTo(350-pd/2,350-pd);x.fill();} for(let i=0;i<gs;i++){for(let j=0;j<gs;j++){const cx=pd+i*st+st/2;const cy=pd+j*st+st/2;const ph=(i-gs/2)*(fv/100)*3+(j-gs/2)*(pv/100)*3;const cph=(j-gs/2)*(pv/100);x.strokeStyle='black';x.beginPath();x.arc(cx,cy,st*0.4,0,6.28);x.fillStyle='#94a3b8';x.fill();if(cph>0.01)x.fillStyle='rgba(255,255,0,0.5)';if(cph<-0.01)x.fillStyle='rgba(0,0,255,0.5)';x.fill();arrow(x,cx,cy,ph,st*0.6);}}const yz=175-(zv/100)*150;const gr=z.createLinearGradient(0,0,0,350);gr.addColorStop(0,'red');gr.addColorStop(1,'blue');z.fillStyle=gr;z.fillRect(10,10,20,330);z.strokeStyle='black';z.lineWidth=3;z.beginPath();z.moveTo(10,yz);z.lineTo(70,yz);z.stroke();z.fillStyle='black';z.fillText('Z',35,yz-5);} [sf,sp,sz,sg].forEach(s=>s.addEventListener('input',draw));function rst(){sf.value=0;sp.value=0;sz.value=0;sg.value=12;draw();}draw();</script></body></html>""", 
        """<!DOCTYPE html><html><head><style>body{margin:0;padding:5px;font-family:sans-serif;} .box{display:flex;gap:15px;} .ctrl{width:220px;padding:10px;background:#f9f9f9;border:1px solid #ccc;border-radius:8px;} canvas{border:1px solid #ccc;background:#f8f9fa;border-radius:8px;} input{width:100%;} label{font-size:11px;font-weight:bold;display:block;} button{width:100%;padding:8px;background:#4f46e5;color:white;border:none;border-radius:4px;cursor:pointer;}</style></head><body><div class='box'><div class='ctrl'><h4>Encoding</h4><label>Freq</label><input type='range' id='f' min='-100' max='100' value='0'><br><label>Phase</label><input type='range' id='p' min='-100' max='100' value='0'><br><label>Slice</label><input type='range' id='z' min='-100' max='100' value='0'><br><label>Matrix</label><input type='range' id='g' min='5' max='20' value='12'><br><button onclick='rst()'>Reset</button></div><div><canvas id='c1' width='350' height='350'></canvas><canvas id='c2' width='80' height='350'></canvas></div></div><script>const c1=document.getElementById('c1');const x=c1.getContext('2d');const c2=document.getElementById('c2');const z=c2.getContext('2d');const sf=document.getElementById('f');const sp=document.getElementById('p');const sz=document.getElementById('z');const sg=document.getElementById('g');const pd=30;function arrow(ctx,x,y,a,s){const l=s*0.35;ctx.save();ctx.translate(x,y);ctx.rotate(a);ctx.beginPath();ctx.moveTo(-l,0);ctx.lineTo(l,0);ctx.lineTo(l-6,-6);ctx.moveTo(l,0);ctx.lineTo(l-6,6);ctx.strokeStyle='white';ctx.lineWidth=1.5;ctx.stroke();ctx.restore();} function draw(){x.clearRect(0,0,350,350);z.clearRect(0,0,80,350);const fv=parseFloat(sf.value);const pv=parseFloat(sp.value);const zv=parseFloat(sz.value);const gs=parseInt(sg.value);const st=(350-2*pd)/gs;const h=(pd*0.8)*(fv/100);x.fillStyle='rgba(255,0,0,0.3)';if(fv!=0){x.beginPath();x.moveTo(pd,pd/2);x.lineTo(pd,pd/2-h);x.lineTo(350-pd,pd/2+h);x.lineTo(350-pd,pd/2);x.fill();}const w=(pd*0.8)*(pv/100);x.fillStyle='rgba(0,255,0,0.3)';if(pv!=0){x.beginPath();x.moveTo(350-pd/2,pd);x.lineTo(350-pd/2-w,pd);x.lineTo(350-pd/2+w,350-pd);x.lineTo(350-pd/2,350-pd);x.fill();} for(let i=0;i<gs;i++){for(let j=0;j<gs;j++){const cx=pd+i*st+st/2;const cy=pd+j*st+st/2;const ph=(i-gs/2)*(fv/100)*3+(j-gs/2)*(pv/100)*3;const cph=(j-gs/2)*(pv/100);x.strokeStyle='black';x.beginPath();x.arc(cx,cy,st*0.4,0,6.28);x.fillStyle='#94a3b8';x.fill();if(cph>0.01)x.fillStyle='rgba(255,255,0,0.5)';if(cph<-0.01)x.fillStyle='rgba(0,0,255,0.5)';x.fill();arrow(x,cx,cy,ph,st*0.6);}}const yz=175-(zv/100)*150;const gr=z.createLinearGradient(0,0,0,350);gr.addColorStop(0,'red');gr.addColorStop(1,'blue');z.fillStyle=gr;z.fillRect(10,10,20,330);z.strokeStyle='black';z.lineWidth=3;z.beginPath();z.moveTo(10,yz);z.lineTo(70,yz);z.stroke();z.fillStyle='black';z.fillText('Z',35,yz-5);} [sf,sp,sz,sg].forEach(s=>s.addEventListener('input',draw));function rst(){sf.value=0;sp.value=0;sz.value=0;sg.value=12;draw();}draw();</script></body></html>"""), height=450)
        
        st.divider()
        st.markdown(T("<h3 style='background-color: #e0e7ff; padding: 10px; border-radius: 5px; color: #3730a3;'>🧠 Synthèse : Gradient & Espace K</h3>", "<h3 style='background-color: #e0e7ff; padding: 10px; border-radius: 5px; color: #3730a3;'>🧠 Summary: Gradient & K-Space</h3>"), unsafe_allow_html=True)
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.info(T("**1. Gradient Faible (Lignes Centrales)**\n* Faible Déphasage = Signal Fort.\n* Contraste de l'image.", "**1. Low Gradient (Center Lines)**\n* Low Dephasing = Strong Signal.\n* Image Contrast."))
        with col_c2:
            st.error(T("**2. Gradient Fort (Lignes Périphériques)**\n* Fort Déphasage = Détails fins.\n* Résolution spatiale.", "**2. High Gradient (Peripheral Lines)**\n* High Dephasing = Fine Details.\n* Spatial Resolution."))

    # SOUS-ONGLET 2 : ESPACE K
    with sub_tabs[1]:
        st.markdown(T("<h3 style='color: #db2777; border-bottom: 2px solid #db2777; padding-bottom: 5px;'>🖼️ Remplissage & Reconstruction</h3>", "<h3 style='color: #db2777; border-bottom: 2px solid #db2777; padding-bottom: 5px;'>🖼️ Filling & Reconstruction</h3>"), unsafe_allow_html=True)
        
        col_k1, col_k2 = st.columns([1, 1])
        with col_k1:
            
            lbl_mode = T("Ordre de Remplissage", "Filling Order")
            opt_lin = T("Linéaire (Haut -> Bas)", "Linear (Top -> Bottom)")
            opt_cen = T("Centrique (Centre -> Bords)", "Centric (Center -> Edges)")
            
            fill_mode = st.radio(lbl_mode, [opt_lin, opt_cen], key=f"k_mode_{current_reset_id}")
            acq_pct = st.slider(T("Progression (%)", "Progress (%)"), 0, 100, 10, step=1, key=f"k_pct_{current_reset_id}")
            
            st.divider()
            
            if turbo > 1:
                # TITRE TSE STYLISÉ
                st.markdown(T(f"""
                <div style="background-color: #fce7f3; padding: 10px; border-radius: 5px; border-left: 5px solid #db2777; margin-bottom: 10px;">
                    <h4 style="margin:0; color: #831843;">🚅 Rangement des {turbo} Échos (Ky)</h4>
                </div>
                """, f"""
                <div style="background-color: #fce7f3; padding: 10px; border-radius: 5px; border-left: 5px solid #db2777; margin-bottom: 10px;">
                    <h4 style="margin:0; color: #831843;">🚅 Ordering of {turbo} Echoes (Ky)</h4>
                </div>
                """), unsafe_allow_html=True)
                st.info(T(f"TE Cible : **{int(te)} ms** | Facteur Turbo : **{turbo}**", f"Target TE: **{int(te)} ms** | Turbo Factor: **{turbo}**"))
                
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
                ax.text(-0.05, 0.5, T("CENTRE (K=0)", "CENTER (K=0)"), ha='right', va='center', fontweight='bold')
                ax.annotate("", xy=(-0.02, 0.4), xytext=(-0.02, 0.6), arrowprops=dict(arrowstyle="-", color="black", lw=2))
                st.pyplot(fig_tse)
                plt.close(fig_tse)
            else:
                st.markdown(T(f"#### 🐢 Acquisition Standard (1 Écho/TR)", f"#### 🐢 Standard Acquisition (1 Echo/TR)"))
                st.info(f"TE : **{int(te)} ms**")
                fig_tse, ax = plt.subplots(figsize=(5, 4))
                n_disp_lines = 24; y_h = 1.0 / n_disp_lines; color = plt.cm.jet(0)
                for i in range(n_disp_lines):
                    rect = patches.Rectangle((0, 1.0 - (i + 1) * y_h), 1, y_h, linewidth=0.5, edgecolor='white', facecolor=color)
                    ax.add_patch(rect)
                ax.text(0.5, 0.5, T(f"ECHO 1 (TE={int(te)}ms)\nAppliqué à chaque ligne", f"ECHO 1 (TE={int(te)}ms)\nApplied to each line"), ha='center', va='center', color='white', fontweight='bold', fontsize=12, path_effects=[path_effects.withStroke(linewidth=3, foreground='black')])
                ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
                st.pyplot(fig_tse)
                plt.close(fig_tse)
        
        with col_k2:
            mask_k = np.zeros((S, S)); lines_to_fill = int(S * (acq_pct / 100.0))
            if "Linéaire" in fill_mode or "Linear" in fill_mode: mask_k[0:lines_to_fill, :] = 1
            else: center_line = S // 2; half = lines_to_fill // 2; mask_k[center_line-half:center_line+half, :] = 1
            kspace_masked = f * mask_k; img_rec = np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_masked)))
            fig_k, ax_k = plt.subplots(figsize=(4, 4))
            ax_k.imshow(20 * np.log(np.abs(kspace_masked) + 1), cmap='inferno'); ax_k.axis('off')
            st.pyplot(fig_k)
            plt.close(fig_k)
            st.image(img_rec, clamp=True, width=300, caption=T("Reconstruction", "Reconstruction"))

# [TAB 3 : SIGNAUX]
with t3:
    st.markdown(T("### 📊 Comparaison des Signaux", "### 📊 Signal Comparison"))
    
    # Centrage du graphique
    c_sig_left, c_sig_center, c_sig_right = st.columns([1, 2, 1])
    
    with c_sig_center:
        fig_sig, ax_sig = plt.subplots(figsize=(4, 2.5))
        
        # Récupération des valeurs
        vals_bar = [v_lcr, v_gm, v_wm, v_fat]
        noms = [T("EAU", "WATER"), T("SG", "GM"), T("SB", "WM"), T("GRAISSE", "FAT")]
        
        if show_stroke: 
            vals_bar.append(v_stroke)
            noms.append(T("AVC", "STROKE"))
            
        cols = ['cyan', 'dimgray', 'lightgray', 'orange', 'red'] if show_stroke else ['cyan', 'dimgray', 'lightgray', 'orange']
        bars = ax_sig.bar(noms, vals_bar, color=cols, edgecolor='black')
        
        ax_sig.set_ylim(0, 1.3)
        ax_sig.grid(True, axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig_sig); plt.close(fig_sig)

# [TAB 5 : ANATOMIE] - RÉINITIALISÉ ET MODIFIÉ
with t5:
    st.header(T("Exploration Anatomique (Physique Avancée)", "Anatomical Exploration (Advanced Physics)"))
    
    if HAS_NILEARN and processor.ready:
        c1, c2 = st.columns([1, 3])
        dims = processor.get_dims()
        
        with c1:
            # SÉLECTEUR DE PLAN
            plane = st.radio(
                T("Plan de Coupe", "Slice Plane"), 
                [T("Plan Axial", "Axial Plane"), T("Plan Sagittal", "Sagittal Plane"), T("Plan Coronal", "Coronal Plane")], 
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
            
            # --- LOGIQUE DE FENÊTRAGE AUTOMATIQUE (MODIFICATION) ---
            # Valeurs standards
            def_win, def_lev = (0.74, 0.55)
            
            # Application de la règle spécifique : Diffusion + b >= 1000
            # Note : b_value est récupéré depuis la session globale du simulateur
            if is_dwi and (b_value >= 1000):
                def_win = 0.90
            
            # Utilisation d'une clé dynamique pour forcer la mise à jour du widget
            key_suffix = f"{current_reset_id}_{is_dwi}_{b_value >= 1000}"
            
            window = st.slider(T("Fenêtre", "Window"), 0.01, 3.0, float(def_win), 0.01, key=f"wn_{key_suffix}")
            level = st.slider(T("Niveau", "Level"), 0.0, 1.5, float(def_lev), 0.01, key=f"lv_{key_suffix}")
            # --- FIN DE LA MODIFICATION ---
            
            st.divider()
            
            # LÉGENDES INTERACTIVES
            show_interactive_legends = st.checkbox(
                T("🔍 Activer Légendes (Atlas Harvard-Oxford)", "🔍 Enable Legends (Harvard-Oxford Atlas)"), 
                value=False, 
                help=T("Identifie les structures au survol.", "Identifies structures on hover.")
            )
            
            # FEEDBACK VISUEL
            if is_dwi: 
                if show_adc_map: st.info(T("🗺️ **Mode Carte ADC**", "🗺️ **ADC Map Mode**"))
                else: st.success(T(f"🧬 **Mode Diffusion** (b={b_value})", f"🧬 **Diffusion Mode** (b={b_value})"))
            
            if show_stroke and ax == 'z': st.error(T("⚠️ **AVC Visible**", "⚠️ **Stroke Visible**"))
            
        with c2:
            # PRÉPARATION DES SIGNAUX
            w_vals = {'csf':v_lcr, 'gm':v_gm, 'wm':v_wm, 'fat':v_fat}
            if show_stroke: w_vals['wm'] = w_vals['wm'] * 0.9 + v_stroke * 0.1
            
            seq_type_arg = 'dwi' if is_dwi else ('gre' if is_gre else None)
            
            # 1. GÉNÉRATION IMAGE BRUTE
            img_raw = processor.get_slice(ax, idx, w_vals, seq_type=seq_type_arg, te=te, tr=tr, fa=flip_angle, b_val=b_value, adc_mode=show_adc_map, with_stroke=show_stroke)
            
            if img_raw is not None:
                # 2. AJOUT BRUIT & FLOU (Si DWI)
                if is_dwi and not show_adc_map and b_value > 0:
                    noise = np.random.normal(0, (b_value/1000.0)*0.05, img_raw.shape)
                    img_noisy = img_raw + noise
                    img_raw = gaussian_filter(img_noisy, sigma=0.8)
                    img_raw = np.clip(img_raw, 0, 2.0)

                # 3. APPLICATION FENÊTRAGE
                img_display = utils.apply_window_level(img_raw, window, level)
                
                # 4. AFFICHAGE
                if show_interactive_legends:
                    labels_map = processor.get_anatomical_labels(ax, idx)
                    fig = px.imshow(img_display, color_continuous_scale='gray', zmin=0, zmax=1)
                    fig.update_traces(customdata=labels_map, hovertemplate="<b>%{customdata}</b><extra></extra>")
                    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False, width=600, height=600)
                    st.plotly_chart(fig, config={'displayModeBar': False})
                else:
                    st.image(img_display, clamp=True, width=600)
    else: 
        st.warning(T("Module 'nilearn' manquant.", "'nilearn' module missing."))
# [TAB 6 : PHYSIQUE]
with t6:
    st.header(T("📈 Physique", "📈 Physics"))
    
    # Définition des tissus et couleurs
    tists = [cst.T_FAT, cst.T_WM, cst.T_GM, cst.T_LCR]
    cols = ['orange', 'lightgray', 'dimgray', 'cyan'] 
    
    # Ajout pathologie si active
    if show_stroke: 
        tists.append(cst.T_STROKE)
        cols.append('red') 
    
    # =========================================================
    # GRAPHIQUE 1 : RELAXATION LONGITUDINALE (T1)
    # =========================================================
    fig_t1 = plt.figure(figsize=(10, 3))
    gs = fig_t1.add_gridspec(1, 2, width_ratios=[30, 1], wspace=0.05)
    ax_t1 = fig_t1.add_subplot(gs[0])
    ax_bar = fig_t1.add_subplot(gs[1])
    
    x_t = np.linspace(0, 4000, 500)
    
    ax_t1.set_title(T("Relaxation Longitudinale (T1)", "Longitudinal Relaxation (T1)"))
    
    # LOGIQUE DE TRACÉ SELON SÉQUENCE
    if is_gre:
        # Écho de Gradient : Départ à cos(alpha)
        start_mz = np.cos(np.radians(flip_angle))
        ax_t1.set_ylim(-0.1, 1.1)
        for t, col in zip(tists, cols): 
            mz = 1 - (1 - start_mz) * np.exp(-x_t / t['T1'])
            ax_t1.plot(x_t, mz, color=col, label=t['Label'])
        ax_t1.axhline(start_mz, color='gray', linestyle=':', label=f"Mz(0)")
        
    elif is_ir:
        # Inversion Récupération : De -1 à +1
        ax_t1.set_ylim(-1.1, 1.1)
        ax_t1.axhline(0, color='black')
        for t, col in zip(tists, cols): 
            mz = 1 - 2 * np.exp(-x_t / t['T1'])
            ax_t1.plot(x_t, mz, color=col, label=t['Label'])
        # Le TI est tracé une seule fois via la correction légende plus bas
        ax_t1.axvline(x=ti, color='green', linestyle='--', label='TI')
        
    else:
        # Spin Echo Standard : De 0 à 1
        ax_t1.set_ylim(0, 1.1)
        for t, col in zip(tists, cols): 
            mz = 1 - np.exp(-x_t / t['T1'])
            ax_t1.plot(x_t, mz, color=col, label=t['Label'])
        
    # Ligne TR
    ax_t1.axvline(x=tr_effective, color='red', linestyle='--', label=T('TR Réel', 'Real TR'))
    
    # Barre de Dégradé Latérale
    gradient = np.linspace(1, 0, 256).reshape(-1, 1)
    if is_ir: gradient = np.abs(np.linspace(1, -1, 256)).reshape(-1, 1)
    
    ax_bar.imshow(gradient, aspect='auto', cmap='gray', extent=[0, 1, ax_t1.get_ylim()[0], ax_t1.get_ylim()[1]])
    ax_bar.axis('off')
    
    # --- CORRECTION LÉGENDE (Anti-Doublons) ---
    # Permet d'éviter d'avoir 5 fois "TI" ou "TR" dans la légende
    handles, labels = ax_t1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) 
    ax_t1.legend(by_label.values(), by_label.keys(), loc='best')
    # ------------------------------------------

    st.pyplot(fig_t1)
    plt.close(fig_t1)
    
    # =========================================================
    # GRAPHIQUE 2 : RELAXATION TRANSVERSALE (T2)
    # =========================================================
    fig_t2 = plt.figure(figsize=(10, 3))
    gs2 = fig_t2.add_gridspec(1, 2, width_ratios=[30, 1], wspace=0.05)
    ax_t2 = fig_t2.add_subplot(gs2[0])
    ax_bar2 = fig_t2.add_subplot(gs2[1])
    
    x_te = np.linspace(0, 500, 300)
    
    ax_t2.set_title(T("Relaxation Transversale (T2/T2*)", "Transverse Relaxation (T2/T2*)"))
    
    for t, col in zip(tists, cols): 
        # Choix T2 ou T2*
        val_t2 = t['T2s'] if is_gre else t['T2']
        label_suffix = " (T2*)" if is_gre else ""
        
        mxy = np.exp(-x_te / val_t2)
        ax_t2.plot(x_te, mxy, color=col, label=f"{t['Label']}{label_suffix}")
        
    ax_t2.axvline(x=te, color='red', linestyle='--', label=T('TE Eff', 'Eff TE'))
    
    # Légende simple pour T2
    ax_t2.legend()
    
    # Barre Latérale T2
    gradient_t2 = np.linspace(1, 0, 256).reshape(-1, 1)
    ax_bar2.imshow(gradient_t2, aspect='auto', cmap='gray', extent=[0, 1, 0, 1.0])
    ax_bar2.axis('off')
    
    st.pyplot(fig_t2)
    plt.close(fig_t2)

# [TAB 7 : CHRONOGRAMME]
with t7:
    st.header(T("⚡ Chronogramme", "⚡ Timing Diagram"))
    t_90 = 10
    
    if is_gre:
        st.subheader(T(f"Séquence : Écho de Gradient (Angle {flip_angle}°)", f"Sequence: Gradient Echo (Angle {flip_angle}°)"))
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
        if is_dwi: st.subheader(T("Séquence : Diffusion (DWI - SE EPI)", "Sequence: Diffusion (DWI - SE EPI)"))
        elif is_turbo: st.subheader(T(f"Séquence : Turbo Spin Écho (TSE) - Facteur {turbo}", f"Sequence: Turbo Spin Echo (TSE) - Factor {turbo}"))
        else: st.subheader(T("Séquence : Spin Écho (SE)", "Sequence: Spin Echo (SE)"))
        
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
                if label == "BF": 
                    # Traduction de l'annotation "BF" (Basse Fréquence -> Center/LF)
                    txt_bf = T("BF", "Ctr")
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
                     axs[4].text(t_e, amp+0.3, T("TE eff", "Eff TE"), ha='center', color='red', fontweight='bold', fontsize=10)
                     axs[4].axvline(x=t_e, color='red', linestyle='--', alpha=0.5)
        axs[4].plot(t, sig, color='navy', linewidth=1.5); axs[4].set_ylabel("Signal")
        st.pyplot(fig); plt.close(fig)

with t8:
    st.header(T("☣️ Laboratoire d'Artefacts", "☣️ Artifact Laboratory"))
    
    col_ctrl, col_visu = st.columns([1, 2])
    
    # --- DÉFINITION DES NOMS TRADUITS POUR LE MENU ---
    opt_aliasing = T("Repliement (Aliasing)", "Aliasing")
    opt_shift    = T("Décalage Chimique", "Chemical Shift")
    opt_trunc    = T("Troncature (Gibbs)", "Truncation (Gibbs)")
    opt_motion   = T("Mouvement", "Motion")
    opt_zipper   = "Zipper" # Identique dans les deux langues

    options_list = [opt_aliasing, opt_shift, opt_trunc, opt_motion, opt_zipper]

    with col_ctrl:
        st.markdown(f"#### {T('Choix de l\'Artefact', 'Artifact Selection')}")
        artefact_type = st.radio(
            T("Sélectionnez :", "Select:"), 
            options_list, 
            key="art_main_radio"
        )

    # --- 1. ALIASING ---
    if artefact_type == opt_aliasing:
        with col_ctrl:
            # On utilise fov et S qui viennent du contexte global
            st.info(f"{T('FOV Actuel', 'Current FOV')} : **{fov} mm** ({T('Objet', 'Object')} : 230 mm)")
            if fov < 230: 
                st.error(T("⚠️ Aliasing Actif !", "⚠️ Aliasing Active!"))
            else:
                st.success(T("Pas d'aliasing", "No Aliasing"))

        with col_visu:
            img_art = final.copy()
            # Simulation simplifiée du repliement
            if fov < 230:
                ratio = fov / 230.0
                # Calcul du décalage en pixels
                shift_w = int(S * (1 - ratio) / 2)
                # On évite les index négatifs ou hors bornes si shift_w est trop grand
                shift_w = max(0, min(shift_w, S // 2))
                
                if shift_w > 0:
                    top = img_art[0:shift_w, :]
                    bot = img_art[S-shift_w:S, :]
                    img_art = img_art.copy()
                    img_art[S-shift_w:S, :] += top
                    img_art[0:shift_w, :] += bot
            
            fig_a = plt.figure(figsize=(5,5))
            ax_a = fig_a.add_subplot(111)
            ax_a.imshow(img_art, cmap='gray', vmin=0, vmax=1.3)
            ax_a.set_title(T("Image avec Repliement", "Aliased Image"))
            ax_a.axis('off')
            st.pyplot(fig_a)

    # --- 2. DÉCALAGE CHIMIQUE ---
    elif artefact_type == opt_shift:
        with col_ctrl: 
            st.info(f"BW : **{bw} Hz/px**")
            st.markdown(T(
                "Le décalage est inversement proportionnel à la bande passante.",
                "Shift is inversely proportional to bandwidth."
            ))
        with col_visu:
            # Simulation du shift : Graisse (img_fat) décalée par rapport à l'eau (img_water)
            if bw == 220: px_shift = 0.0 
            else: px_shift = 220.0 / float(bw) # Formule arbitraire pour la démo
            
            # On force un décalage visible pour la pédagogie si BW est faible
            factor_visu = 5.0 # Pour rendre l'effet plus visible à l'écran
            effective_shift = px_shift * factor_visu

            sh = shift(img_fat, [0, effective_shift])
            res = img_water + sh
            
            fig_cs = plt.figure(figsize=(5,5))
            ax_cs = fig_cs.add_subplot(111)
            ax_cs.imshow(res, cmap='gray', vmin=0, vmax=1.3)
            ax_cs.set_title(f"Shift: {effective_shift:.1f} px")
            ax_cs.axis('off')
            st.pyplot(fig_cs)

    # --- 3. TRONCATURE (GIBBS) ---
    elif artefact_type == opt_trunc:
        with col_ctrl: 
            sm = st.select_slider(
                T("Matrice Simulée", "Simulated Matrix"), 
                [32, 64, 128, 256], 
                64
            )
            if sm <= 64: 
                st.warning(T("Oscillations visibles (Gibbs)", "Ringing visible (Gibbs)"))
        with col_visu:
            # Passage en K-Space -> Crop -> Retour
            ft = np.fft.fftshift(np.fft.fft2(final))
            c = S // 2
            k = sm // 2
            m = np.zeros_like(ft)
            # Masque carré simple
            m[c-k:c+k, c-k:c+k] = 1
            
            res = np.abs(np.fft.ifft2(np.fft.ifftshift(ft * m)))
            
            fig_g = plt.figure(figsize=(5,5))
            ax_g = fig_g.add_subplot(111)
            ax_g.imshow(res, cmap='gray', vmin=0, vmax=1.3)
            ax_g.set_title(T("Artefact de Troncature", "Truncation Artifact"))
            ax_g.axis('off')
            st.pyplot(fig_g)

    # --- 4. MOUVEMENT ---
    elif artefact_type == opt_motion:
        with col_ctrl: 
            it = st.slider(T("Intensité Mouvement", "Motion Intensity"), 0.0, 5.0, 0.5)
            st.markdown(T(
                "Perturbation de la phase dans l'espace K.",
                "Phase perturbation in K-space."
            ))
        with col_visu:
            ft = np.fft.fftshift(np.fft.fft2(final))
            if it > 0:
                # Ajout de bruit de phase aléatoire ligne par ligne
                ph = np.random.normal(0, it, S)
                for i in range(S): 
                    ft[i, :] *= np.exp(1j * ph[i])
            
            res = np.abs(np.fft.ifft2(np.fft.ifftshift(ft)))
            
            fig_m = plt.figure(figsize=(5,5))
            ax_m = fig_m.add_subplot(111)
            ax_m.imshow(res, cmap='gray', vmin=0, vmax=1.3)
            ax_m.set_title(T("Fantômes de Mouvement", "Motion Ghosts"))
            ax_m.axis('off')
            st.pyplot(fig_m)

    # --- 5. ZIPPER ---
    elif artefact_type == opt_zipper:
        with col_ctrl: 
            fr = st.slider(T("Fréquence (Ligne)", "Frequency (Line)"), 0, S-1, S//2)
            vol = st.slider("Volume / Amplitude", 0, 100, 10)
        with col_visu:
            ft = np.fft.fftshift(np.fft.fft2(final))
            if vol > 0:
                # Spike de RF dans l'espace K à une fréquence précise
                # Bruit blanc + Offset constant
                ns = np.random.normal(0, vol, S) + (vol * 5)
                # Alternance pour créer le motif en points
                alt = np.array([1 if i%2==0 else -1 for i in range(S)])
                ft[:, fr] += ns * alt * 50
            
            res = np.abs(np.fft.ifft2(np.fft.ifftshift(ft)))
            
            fig_z = plt.figure(figsize=(5,5))
            ax_z = fig_z.add_subplot(111)
            ax_z.imshow(res, cmap='gray', vmin=0, vmax=1.3)
            ax_z.set_title("Zipper Artifact")
            ax_z.axis('off')
            st.pyplot(fig_z)

with t9:
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
                fig_all, ax_all = plt.subplots(figsize=(8, 4))
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
                
                fig_full, ax_f = plt.subplots(figsize=(4, 4))
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
                fig_analog, ax_an = plt.subplots(figsize=(8, 4))
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
                
                fig_view, ax_v = plt.subplots(figsize=(3, 3))
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
        fig_lines, ax_lines = plt.subplots(figsize=(10, 1.5))
        for i in range(25): 
            if i % ipat_factor == 0:
                ax_lines.vlines(i, 0, 1, colors='green', linewidth=3)
            else:
                ax_lines.vlines(i, 0, 1, colors='red', linestyles='dotted', linewidth=1.5)
        
        ax_lines.set_xlim(-1, 26); ax_lines.set_ylim(0, 1); ax_lines.axis('off')
        
        txt_legend = T("Vert = Acquise\nRouge = Sautée", "Green = Acquired\nRed = Skipped")
        ax_lines.text(26, 0.5, txt_legend, va='center', fontsize=9)
        st.pyplot(fig_lines)
        plt.close(fig_lines)

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
        
        fig_s, ax_s = plt.subplots(figsize=(2, 2))
        ax_s.imshow(sens, cmap='jet', vmin=0, vmax=1); ax_s.axis('off')
        cols[i].pyplot(fig_s); plt.close(fig_s)

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

with t10:
    st.header(T("🧬 Théorie de la Diffusion (DWI)", "🧬 Diffusion Theory (DWI)"))
    st.markdown(T(
        "L'imagerie de diffusion est unique car elle sonde le **mouvement microscopique** des molécules d'eau.",
        "Diffusion imaging is unique because it probes the **microscopic movement** of water molecules."
    ))
    st.divider()
    
    # --- 1. CODE RESTAURÉ (ISOTROPIE & ADC) ---
    st.subheader(T("1. Isotropie vs Anisotropie", "1. Isotropy vs Anisotropy"))
    
    fig_iso, ax_iso = plt.subplots(1, 2, figsize=(6, 2))
    
    # Isotropie
    ax_iso[0].set_title(T("Isotrope (LCR)", "Isotropic (CSF)"))
    ax_iso[0].add_patch(patches.Circle((0.5, 0.5), 0.3, color='lightblue', alpha=0.3))
    ax_iso[0].text(0.5, 0.5, "H2O", ha='center', va='center', fontweight='bold')
    
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        rad = np.radians(angle)
        dx, dy = np.cos(rad)*0.25, np.sin(rad)*0.25
        ax_iso[0].arrow(0.5, 0.5, dx, dy, head_width=0.05, color='blue')
    ax_iso[0].axis('off')
    
    # Anisotropie
    ax_iso[1].set_title(T("Anisotrope (Fibre)", "Anisotropic (Fiber)"))
    ax_iso[1].add_patch(patches.Rectangle((0.1, 0.3), 0.8, 0.05, color='orange', alpha=0.5))
    ax_iso[1].add_patch(patches.Rectangle((0.1, 0.65), 0.8, 0.05, color='orange', alpha=0.5))
    
    ax_iso[1].text(0.5, 0.8, T("Fibre Nerveuse", "Nerve Fiber"), ha='center', color='orange')
    ax_iso[1].text(0.5, 0.5, "H2O", ha='center', va='center', fontweight='bold')
    
    ax_iso[1].arrow(0.5, 0.5, 0.3, 0, head_width=0.05, color='blue')
    ax_iso[1].arrow(0.5, 0.5, -0.3, 0, head_width=0.05, color='blue')
    ax_iso[1].arrow(0.5, 0.5, 0, 0.1, head_width=0.03, color='red', alpha=0.5)
    ax_iso[1].arrow(0.5, 0.5, 0, -0.1, head_width=0.03, color='red', alpha=0.5)
    ax_iso[1].axis('off')
    
    st.pyplot(fig_iso)
    plt.close(fig_iso)
    
    st.divider()
    
    st.subheader(T("2. Coefficient de Diffusion Apparent (ADC)", "2. Apparent Diffusion Coefficient (ADC)"))
    
    # Pour illustrer la différence clinique réelle :
    # 

    fig_adc, ax = plt.subplots(1, 2, figsize=(8, 1.5))
    
    # Variables de texte pour les scénarios
    txt_b1000 = "b=1000"
    txt_map = T("Map ADC", "ADC Map")
    txt_dwi = "DWI"
    
    # Scenario 1 : AVC
    ax[0].set_facecolor('black')
    ax[0].axis('off')
    ax[0].set_title(T("SCÉNARIO 1 : AVC (Restriction)", "SCENARIO 1: STROKE (Restriction)"), color='lime', weight='bold', fontsize=9)
    ax[0].text(0.3, 0.8, txt_b1000, color='black', ha='center', fontsize=8, fontweight='bold')
    ax[0].text(0.7, 0.8, txt_map, color='black', ha='center', fontsize=8, fontweight='bold')
    
    ax[0].add_patch(patches.Circle((0.3, 0.5), 0.15, edgecolor='red', facecolor='white', linewidth=4)) 
    ax[0].text(0.3, 0.25, txt_dwi, color='white', ha='center', fontweight='bold', fontsize=7)
    
    ax[0].text(0.5, 0.5, "➔", color='white', fontsize=12, ha='center', va='center')
    
    ax[0].add_patch(patches.Circle((0.7, 0.5), 0.15, edgecolor='red', facecolor='black', linewidth=4)) 
    ax[0].text(0.7, 0.25, T("ADC (Noir)", "ADC (Dark)"), color='white', ha='center', fontweight='bold', fontsize=7)
    
    # Scenario 2 : LCR
    ax[1].set_facecolor('black')
    ax[1].axis('off')
    ax[1].set_title(T("SCÉNARIO 2 : LCR (Liquide)", "SCENARIO 2: CSF (Liquid)"), color='red', weight='bold', fontsize=9)
    ax[1].text(0.3, 0.8, txt_b1000, color='black', ha='center', fontsize=8, fontweight='bold')
    ax[1].text(0.7, 0.8, txt_map, color='black', ha='center', fontsize=8, fontweight='bold')
    
    ax[1].add_patch(patches.Circle((0.3, 0.5), 0.15, edgecolor='red', facecolor='white', linewidth=4)) 
    ax[1].text(0.3, 0.25, txt_dwi, color='white', ha='center', fontweight='bold', fontsize=7)
    
    ax[1].text(0.5, 0.5, "➔", color='white', fontsize=12, ha='center', va='center')
    
    ax[1].add_patch(patches.Circle((0.7, 0.5), 0.15, edgecolor='red', facecolor='white', linewidth=4)) 
    ax[1].text(0.7, 0.25, T("ADC (Blanc)", "ADC (Bright)"), color='white', ha='center', fontweight='bold', fontsize=7)
    
    st.pyplot(fig_adc)
    plt.close(fig_adc)
    
    st.divider()

    # --- 2. FORMULE & GRAPHIQUE (IVIM / KURTOSIS) ---
    st.subheader(T("3. Comprendre la Décroissance (Avancé)", "3. Understanding Decay (Advanced)"))
    
    # Formule unique + Légende
    st.markdown(f"##### {T('La Formule de Base', 'The Basic Formula')}")
    st.latex(r"S = S_0 \cdot e^{-b \cdot ADC}")
    
    with st.expander(T("📖 Légende de la formule (Cliquez)", "📖 Formula Legend (Click)")):
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
        """))

    # Graphique Semi-Log
    col_plot, col_expl = st.columns([2, 1])
    
    with col_plot:
        b = np.linspace(0, 3000, 300)
        adc_pure = 0.8e-3
        
        # Courbes théoriques
        ln_S_adc = -b * adc_pure
        ivim_effect = 0.4 * np.exp(-b * 0.02)
        ln_S_ivim = np.log(np.exp(ln_S_adc) + ivim_effect)
        kurtosis_term = (1.0/6.0) * (b**2) * (adc_pure**2) * 1.5
        ln_S_kurt = ln_S_adc + kurtosis_term

        fig_decay, ax_d = plt.subplots(figsize=(8, 5))
        
        # Zones (Labels traduits)
        ax_d.fill_between(b, ln_S_adc, ln_S_ivim, where=(b < 800), color='#9b59b6', alpha=0.3, label=T('Effet IVIM', 'IVIM Effect'))
        ax_d.fill_between(b, ln_S_adc, ln_S_kurt, where=(b > 1000), color='#2ecc71', alpha=0.4, label=T('Effet Kurtosis', 'Kurtosis Effect'))
        
        # Droite ADC
        ax_d.plot(b, ln_S_adc, color='red', linewidth=3, label=T('ADC (Modèle Gaussien)', 'ADC (Gaussian Model)'))
        
        # Points simulés
        b_pts = np.arange(0, 3100, 200)
        y_pts = -b_pts * adc_pure
        y_pts[b_pts < 500] += np.log(1 + 0.4*np.exp(-b_pts[b_pts < 500]*0.02))
        y_pts[b_pts > 1500] += (1.0/6.0) * (b_pts[b_pts > 1500]**2) * (adc_pure**2) * 1.5
        ax_d.scatter(b_pts, y_pts, color='black', zorder=5, label=T('Données', 'Data'))

        # Annotations (Textes traduits)
        ax_d.text(300, -0.2, T("IVIM (Sang)", "IVIM (Blood)"), color='purple', fontweight='bold')
        ax_d.text(2200, -2.5, T("Kurtosis (Cellules)", "Kurtosis (Cells)"), color='green', fontweight='bold')
        
        txt_slope = T("Pente = -ADC", "Slope = -ADC")
        ax_d.text(1200, -1.2, txt_slope, color='red', rotation=-30, fontweight='bold')

        ax_d.set_xlabel(T("Facteur b", "b-Factor"))
        ax_d.set_ylabel("ln(Signal)")
        ax_d.set_xlim(0, 3000)
        ax_d.set_ylim(-4, 0.2)
        ax_d.legend()
        ax_d.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_decay)
        plt.close(fig_decay)

    with col_expl:
        st.info(f"### 🟣 {T('Zone IVIM (b < 200)', 'IVIM Zone (b < 200)')}")
        st.markdown(T("""
        **\"La Fausse Diffusion\"**
        Au début, le signal chute vite. Ce n'est pas de la diffusion, c'est le **sang** qui circule (Pseudo-diffusion).
        * *Utile pour voir la perfusion sans produit de contraste.*
        """, """
        **\"The Pseudo-Diffusion\"**
        At the start, signal drops fast. This isn't diffusion, it's circulating **blood**.
        * *Useful to see perfusion without contrast agent.*
        """))
        
        st.success(f"### 🟢 {T('Zone Kurtosis (b > 1000)', 'Kurtosis Zone (b > 1000)')}")
        st.markdown(T("""
        **\"L'Obstacle\"**
        À la fin, la courbe remonte. L'eau tape dans les murs des cellules (Membranes).
        * *Utile pour grader les tumeurs complexes.*
        """, """
        **\"The Obstacle\"**
        At the end, the curve rises. Water hits cell walls (Membranes).
        * *Useful for grading complex tumors.*
        """))

with t11:
    st.header(T("🎓 Cours Théorique", "🎓 Theoretical Course"))

    # --- DONNÉES DU COURS (STRUCTURE BILINGUE) ---
    # Pour rendre le cours utile, j'ai ajouté du vrai contenu simplifié.
    slides_data = [
        {
            "fr": "1. Le Spin Nucléaire", 
            "en": "1. Nuclear Spin",
            "body_fr": "Le proton H+ agit comme un petit aimant.\nEn l'absence de champ B0, ils sont orientés aléatoirement.\nSous B0, ils s'alignent (Parallèle / Anti-parallèle).",
            "body_en": "The H+ proton acts like a small magnet.\nWithout B0 field, they are randomly oriented.\nUnder B0, they align (Parallel / Anti-parallel)."
        },
        {
            "fr": "2. Résonance & Excitation", 
            "en": "2. Resonance & Excitation",
            "body_fr": "Pour basculer l'aimantation, on envoie une onde RF.\nLa fréquence doit être exactement la Fréquence de Larmor.\nF = γ * B0 (42.58 MHz/T pour l'Hydrogène).",
            "body_en": "To flip magnetization, an RF wave is sent.\nThe frequency must match the Larmor Frequency.\nF = γ * B0 (42.58 MHz/T for Hydrogen)."
        },
        {
            "fr": "3. Relaxation T1 & T2", 
            "en": "3. T1 & T2 Relaxation",
            "body_fr": "T1 (Longitudinal) : Repousse de l'aimantation (Graisse rapide, Eau lente).\nT2 (Transversal) : Déphasage des spins (Interaction spin-spin).\nC'est la base du contraste de l'image.",
            "body_en": "T1 (Longitudinal): Regrowth of magnetization (Fat fast, Water slow).\nT2 (Transverse): Dephasing of spins (Spin-spin interaction).\nThis is the basis of image contrast."
        },
        {
            "fr": "4. Espace K (Fourier)", 
            "en": "4. K-Space (Fourier)",
            "body_fr": "L'IRM n'acquiert pas l'image directement.\nElle remplit l'Espace K (fréquences spatiales).\nLe centre = Contraste. La périphérie = Détails.",
            "body_en": "MRI does not acquire the image directly.\nIt fills K-Space (spatial frequencies).\nCenter = Contrast. Periphery = Details."
        },
        {
            "fr": "5. Sécurité (SAR)", 
            "en": "5. Safety (SAR)",
            "body_fr": "Les ondes RF chauffent les tissus (Effet micro-onde).\nSAR = Taux d'Absorption Spécifique (W/kg).\nAttention aux implants, pacemakers et tatouages.",
            "body_en": "RF waves heat tissues (Microwave effect).\nSAR = Specific Absorption Rate (W/kg).\nBeware of implants, pacemakers, and tattoos."
        }
    ]

    # --- LOGIQUE DU SLIDER ---
    if 'slide_index' not in st.session_state: 
        st.session_state.slide_index = 0
    
    num_slides = len(slides_data)
    
    # Le slider retourne un index (0, 1, 2...)
    # format_func affiche le titre dans la bonne langue
    idx = st.select_slider(
        T("Navigation Diapositive", "Slide Navigation"), 
        options=range(num_slides), 
        value=st.session_state.slide_index, 
        format_func=lambda i: T(slides_data[i]["fr"], slides_data[i]["en"])
    )
    
    st.session_state.slide_index = idx
    
    # Récupération du contenu actuel
    current_data = slides_data[idx]
    title_current = T(current_data["fr"], current_data["en"])
    body_current = T(current_data["body_fr"], current_data["body_en"])

    st.markdown(f"### 📄 {title_current}")

    # --- AFFICHAGE GRAPHIQUE (SIMULATION PPT) ---
    fig_ppt, ax_ppt = plt.subplots(figsize=(10, 6))
    
    # Fond gris très clair pour faire "Diapo"
    ax_ppt.set_facecolor('#f0f0f5')
    
    # Titre dans l'image
    ax_ppt.text(0.5, 0.8, title_current.upper(), 
                ha='center', va='center', fontsize=16, fontweight='bold', color='#2c3e50')
    
    # Corps du texte
    ax_ppt.text(0.5, 0.5, body_current, 
                ha='center', va='center', fontsize=14, color='#34495e', wrap=True)
    
    # Pied de page (Numéro slide)
    footer_txt = T(f"Diapositive {idx+1}/{num_slides}", f"Slide {idx+1}/{num_slides}")
    ax_ppt.text(0.95, 0.05, footer_txt, 
                ha='right', va='bottom', fontsize=10, color='gray')

    ax_ppt.set_xlim(0, 1)
    ax_ppt.set_ylim(0, 1)
    ax_ppt.axis('off')
    
    st.pyplot(fig_ppt)
    plt.close(fig_ppt)

# [TAB 12 : SWI & DIPOLE]
with t12:
    st.header(T("🩸 Imagerie de Susceptibilité Magnétique (SWI)", "🩸 Susceptibility Weighted Imaging (SWI)"))
    
    # Onglets internes traduits
    swi_tab_names = [
        T("1. Physique (Phase & Vecteurs)", "1. Physics (Phase & Vectors)"), 
        T("2. Le Dipôle (Simulation)", "2. The Dipole (Simulation)"), 
        T("3. Imagerie Clinique", "3. Clinical Imaging")
    ]
    
    swi_tab1, swi_tab2, swi_tab3 = st.tabs(swi_tab_names)

    # --- SOUS-ONGLET 1 : PHYSIQUE & EXPLICATIONS ---
    with swi_tab1:
        st.subheader(T("1. Physique : L'Analogie de la Boussole", "1. Physics: The Compass Analogy"))
        
        col_ctrl, col_graph = st.columns([1, 2], gap="medium")
        
        with col_ctrl:
            st.markdown(f"#### {T('🎛️ Contrôles', '🎛️ Controls')}")
            st.caption(T("_Modifiez les valeurs pour faire tourner l'aiguille._", "_Modify values to rotate the needle._"))
            
            te_simu = st.slider(T("Temps d'Écho (TE)", "Echo Time (TE)"), 0, 80, 20, step=1, key="swi_te_p1_pedago")
            fa_simu = st.slider(T("Angle Bascule (°)", "Flip Angle (°)"), 5, 90, 30, key="swi_fa_p1_pedago")
            
            t2_star = 50.0; df = 8.0 
            mag = np.sin(np.radians(fa_simu)) * np.exp(-te_simu / t2_star)
            phase_visu = np.radians(max(10, min(80, 60 - (te_simu/2))))
            vec_visu = mag * np.exp(1j * phase_visu)
            
            st.divider()
            
            c_met1, c_met2 = st.columns(2)
            c_met1.metric(T("Réel (Ombre Sol)", "Real (Floor Shadow)"), f"{vec_visu.real:.2f}")
            c_met2.metric(T("Imag (Ombre Mur)", "Imag (Wall Shadow)"), f"{vec_visu.imag:.2f}")

        with col_graph:
            fig_v, ax_v = plt.subplots(figsize=(5, 5)) 
            fig_v.patch.set_alpha(0) 
            lim = 1.1; ax_v.set_xlim(-0.1, lim); ax_v.set_ylim(-0.1, lim)
            
            ax_v.axhline(0, color='white', lw=1); ax_v.axvline(0, color='white', lw=1)
            
            # Vecteur Principal
            ax_v.arrow(0, 0, vec_visu.real, vec_visu.imag, head_width=0.03, lw=4, fc='#3498db', ec='#3498db', length_includes_head=True, zorder=5)
            ax_v.text(vec_visu.real/2, vec_visu.imag/2 + 0.1, "SIGNAL", color='#3498db', fontweight='bold', ha='center', fontsize=12)
            
            # Projection Réelle
            ax_v.plot([vec_visu.real, vec_visu.real], [0, vec_visu.imag], color='gray', ls=':', lw=1)
            ax_v.arrow(0, 0, vec_visu.real, 0, head_width=0.02, lw=3, fc='#e74c3c', ec='#e74c3c', length_includes_head=True, zorder=4)
            ax_v.text(vec_visu.real/2, -0.08, T("Réel (X)", "Real (X)"), color='#e74c3c', ha='center', fontsize=10, fontweight='bold')
            
            # Projection Imaginaire
            ax_v.plot([0, vec_visu.real], [vec_visu.imag, vec_visu.imag], color='gray', ls=':', lw=1)
            ax_v.arrow(0, 0, 0, vec_visu.imag, head_width=0.02, lw=3, fc='#2ecc71', ec='#2ecc71', length_includes_head=True, zorder=4)
            ax_v.text(-0.02, vec_visu.imag/2, T("Imag (Y)", "Imag (Y)"), color='#2ecc71', ha='right', va='center', fontsize=10, fontweight='bold')
            
            # Arc Phase
            arc = patches.Arc((0,0), 0.4, 0.4, theta1=0, theta2=np.degrees(phase_visu), color='yellow', lw=2)
            ax_v.add_patch(arc)
            ax_v.text(0.25, 0.1, "Phase", color='yellow', fontsize=11, fontweight='bold')
            
            ax_v.set_title(T("Visualisation Vectorielle", "Vector Visualization"), color='white', fontsize=12)
            ax_v.set_aspect('equal'); ax_v.axis('off')
            st.pyplot(fig_v); plt.close(fig_v)

        st.markdown("---")
        with st.expander(T("📖 Comprendre l'Analogie (Cliquez)", "📖 Understand Analogy (Click)"), expanded=True):
            c_txt1, c_txt2 = st.columns(2)
            with c_txt1:
                st.info(T("""**🧭 L'Analogie de la Boussole** \n * **L'Aiguille (Bleue)** : C'est le Signal IRM total. \n * **Sa Longueur** : La force du signal (Magnitude). \n * **Sa Direction** : La nature du tissu (Phase).""",
                          """**🧭 The Compass Analogy** \n * **The Needle (Blue)**: It is the total MRI Signal. \n * **Its Length**: Signal Strength (Magnitude). \n * **Its Direction**: Tissue Nature (Phase)."""))
            with c_txt2:
                st.warning(T("""**💡 Pourquoi Réel & Imaginaire ?** \n L'ordinateur ne stocke pas une flèche. Il stocke ses ombres : \n * **Partie Réelle :** L'ombre au sol (Axe X). \n * **Partie Imaginaire :** L'ombre au mur (Axe Y).""",
                             """**💡 Why Real & Imaginary?** \n The computer doesn't store an arrow. It stores its shadows: \n * **Real Part:** Floor shadow (X Axis). \n * **Imaginary Part:** Wall shadow (Y Axis)."""))
    
    # --- SOUS-ONGLET 2 : LE DIPÔLE ---
    with swi_tab2:
        st.subheader(T("2. 🧲 Le Laboratoire du Dipôle", "2. 🧲 Dipole Laboratory"))
        
        col_dip_ctrl, col_dip_visu = st.columns([1, 3])
        
        with col_dip_ctrl:
            dipole_substance = st.radio(T("Substance :", "Substance:"), ["Hématome (Paramagnétique)", "Calcium (Diamagnétique)"], key="dip_sub_key")
            dipole_system = st.radio(T("Convention Phase :", "Phase Convention:"), ["RHS (GE/Philips/Canon)", "LHS (Siemens)"], key="dip_sys_key")
            st.divider()
            
            # Si on a généré un dipôle dans le main, on l'utilise pour l'affichage
            if "dipole_field" in locals() and np.max(np.abs(dipole_field)) > 0:
                st.success(T("✅ Champ Dipolaire Détecté (Fantôme)", "✅ Dipole Field Detected (Phantom)"))
                st.image(utils.apply_window_level(dipole_field, 1.0, 0.5), caption="Carte de Champ B (Simulée)", clamp=True)
            else:
                z_pos = st.slider(T("Coupe Axiale (Z)", "Axial Slice (Z)"), -1.5, 1.5, 0.0, 0.1, key="dip_z_key")

        with col_dip_visu:
            # Visualisation Schématique du Dipôle (Toujours utile pédagogiquement)
            fig_dip, axes_dip = plt.subplots(1, 2, figsize=(10, 4))
            fig_dip.patch.set_facecolor('#404040')
            
            is_rhs = "RHS" in dipole_system
            is_para = "Hématome" in dipole_substance
            combo = (1 if is_para else -1) * (1 if is_rhs else -1)
            
            col_eq_cen, col_eq_halo, col_poles = ('white', 'black', 'black') if combo > 0 else ('black', 'white', 'white')
            
            # Vue Coronale
            axes_dip[0].set_facecolor('#404040'); axes_dip[0].axis('off')
            axes_dip[0].add_patch(patches.Ellipse((0.5, 0.7), 0.25, 0.35, color=col_poles, alpha=0.9))
            axes_dip[0].add_patch(patches.Ellipse((0.5, 0.3), 0.25, 0.35, color=col_poles, alpha=0.9))
            axes_dip[0].add_patch(patches.Rectangle((0.35, 0.48), 0.3, 0.04, color=col_eq_cen))
            
            # Ligne de coupe
            z_val = z_pos if 'z_pos' in locals() else 0.0
            axes_dip[0].axhline(y=0.5 - (z_val * 0.2), color='yellow', linewidth=2, linestyle='--')
            
            # Vue Axiale (Résultat)
            axes_dip[1].set_facecolor('#404040'); axes_dip[1].axis('off')
            if abs(z_val) < 0.2:
                axes_dip[1].add_patch(patches.Circle((0.5, 0.5), 0.35, color=col_eq_halo, alpha=0.5))
                axes_dip[1].add_patch(patches.Circle((0.5, 0.5), 0.15, color=col_eq_cen))
            elif 0.2 <= abs(z_val) < 1.0:
                axes_dip[1].add_patch(patches.Circle((0.5, 0.5), 0.25 * (1.2 - abs(z_val)), color=col_poles))
                
            st.pyplot(fig_dip); plt.close(fig_dip)

    # --- SOUS-ONGLET 3 : IMAGERIE CLINIQUE ---
    with swi_tab3:
        st.subheader(T("3. Imagerie SWI Clinique", "3. Clinical SWI Imaging"))
        
        path_minip_fixe = os.path.join(current_dir, "minip_static.png") 
        
        if HAS_NILEARN and processor.ready:
            dims = processor.get_dims() 
            c1_swi, c2_swi = st.columns([1, 4])
            
            with c1_swi:
                 st.markdown(f"##### {T('🩻 Navigation', '🩻 Navigation')}")
                 
                 opt_ax = T("Axiale", "Axial")
                 opt_cor = T("Coronale", "Coronal")
                 opt_sag = T("Sagittale", "Sagittal")
                 
                 swi_view = st.radio(T("Plan de Coupe :", "Slice Plane:"), [opt_ax, opt_cor, opt_sag], key="swi_view_mode")
                 
                 if swi_view == opt_ax: 
                     swi_slice = st.slider(T("Position Z", "Position Z"), 0, dims[2]-1, 90, key="swi_z"); axis_code = 'z'
                 elif swi_view == opt_cor: 
                     swi_slice = st.slider(T("Position Y", "Position Y"), 0, dims[1]-1, 100, key="swi_y"); axis_code = 'y'
                 else: 
                     swi_slice = st.slider(T("Position X", "Position X"), 0, dims[0]-1, 90, key="swi_x"); axis_code = 'x'
                 
                 st.divider()
                 show_microbleeds_swi = st.checkbox(T("Simuler Micro-saignements", "Simulate Microbleeds"), False, key="swi_bleed_check")
                 show_dipole_test = st.checkbox(T("🧪 Dipôle (Test)", "🧪 Dipole (Test)"), False, key="swi_dip_test_check")
            
            with c2_swi:
                sys_arg = "RHS" if "RHS" in dipole_system else "LHS"
                sub_arg = dipole_substance 
                
                # Récupération image via Anatomie
                img_mag = processor.get_slice(axis_code, swi_slice, {}, swi_mode='mag', te=te_simu, with_bleeds=show_microbleeds_swi)
                img_phase = processor.get_slice(axis_code, swi_slice, {}, swi_mode='phase', with_bleeds=show_microbleeds_swi, swi_sys=sys_arg, swi_sub=sub_arg, with_dipole=show_dipole_test)
                
                c_mag, c_pha, c_min = st.columns(3)
                
                with c_mag: 
                    st.image(utils.apply_window_level(img_mag, 1.0, 0.5), caption=f"1. Magnitude ({swi_view})", use_container_width=True)
                with c_pha: 
                    st.image(utils.apply_window_level(img_phase, 1.0, 0.5), caption=f"2. Phase ({swi_view})", use_container_width=True)
                with c_min: 
                    if os.path.exists(path_minip_fixe): 
                        st.image(path_minip_fixe, caption=T("3. MinIP (Référence Axiale)", "3. MinIP (Axial Ref)"), use_container_width=True)
                    else: 
                        st.image(np.zeros((200,200)), caption=T("Image manquante", "Missing Image"), clamp=True)
        else:
            st.info(T("Module Anatomique non chargé. Utilisez le Fantôme Dipôle (Onglet 2) pour la démonstration.", 
                      "Anatomy Module not loaded. Use Dipole Phantom (Tab 2) for demo."))

with t13:
    st.header(T("🧠 Séquence 3D T1 Ultra-Rapide (MP-RAGE)", "🧠 Ultra-Fast 3D T1 Sequence (MP-RAGE)"))
    
    # --- PRÉPARATION DES TEXTES POUR LE TABLEAU HTML ---
    h_brand = T("Constructeur", "Manufacturer")
    h_name  = T("Nom Commercial", "Commercial Name")
    h_tech  = T("Signification Technique", "Technical Meaning")
    
    # Textes spécifiques aux cellules
    txt_philips = T("avec Pré-impulsion", "with Pre-pulse")
    txt_canon   = T("avec Inversion", "with Inversion")
    
    # Construction du tableau HTML dynamique via f-string
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
            T("Sélecteur Constructeur :", "Manufacturer Selector:"), 
            ["SIEMENS", "GE", "PHILIPS", "CANON"], 
            key="mp_const_select_final"
        )
        
        st.markdown(T(
            "**Pourquoi le TR diffère ?** Le TR affiché sur la console ne représente pas la même chose selon le constructeur.",
            "**Why does TR differ?** The TR displayed on the console does not represent the same thing depending on the manufacturer."
        ))

    with col_mp_plot:
        fig_mp, ax_mp = plt.subplots(figsize=(10, 4))
        
        ti_mp = 900
        train_len = 600
        tr_echo_val = 8 
        
        # Inversion Pulse
        ax_mp.bar(0, 1.2, width=40, color='#e74c3c', label='Inversion 180°', zorder=3)
        ax_mp.text(0, 1.35, "180°", color='#e74c3c', ha='center', fontweight='bold')
        
        # Echo Train
        echo_step = 60 
        for k in range(0, train_len, echo_step): 
            ax_mp.bar(ti_mp + k, 0.7, width=25, color='#3498db', alpha=0.7)
            
        # Enveloppe du train
        ax_mp.add_patch(patches.Rectangle((ti_mp - 20, 0), train_len + 10, 0.8, color='#3498db', alpha=0.1))
        
        # --- LOGIQUE D'AFFICHAGE DU TR SELON CONSTRUCTEUR ---
        if constructeur_mp == "SIEMENS":
            # Siemens définit le TR comme le temps entre deux pulses d'inversion
            ax_mp.annotate('', xy=(ti_mp + train_len + 100, -1.0), xytext=(0, -1.0), 
                           arrowprops=dict(arrowstyle='<->', color='green', lw=2.5))
            
            txt_tr_siemens = T("TR Siemens : Temps du Cycle Complet (~2300ms)", "TR Siemens: Full Cycle Time (~2300ms)")
            ax_mp.text((ti_mp + train_len)/2, -1.35, txt_tr_siemens, 
                       color='green', weight='bold', ha='center', fontsize=10)
        else:
            # Les autres définissent le TR comme le temps entre deux échos (ES)
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
        plt.close(fig_mp)
        
    st.divider()
    
    # =================================================================
    # PARTIE 3 : CONTRASTE T1 - MODULE vs PHASE (VERSION FINALE VALIDÉE)
    # =================================================================
    st.markdown(f"#### {T('Optimisation du Contraste (Substance Blanche vs Grise)', '3. Contrast Optimization (White vs Gray Matter)')}")

    col_mp_txt, col_mp_plot = st.columns([1, 2])

    with col_mp_txt:
        # 1. ANALYSE COMPARÉE
        st.info(T(
            "**Analyse Comparée (Même repère) :**\n\n"
            "Regardez les barres de gris à gauche :\n"
            "1. **Haut (Module + Phase) :** L'échelle est linéaire. -1 est **Noir**, +1 est **Blanc**. \nLa SG (négative) apparaît sombre, la SB (positive) apparaît claire. **Contraste Fort.**\n\n"
            "2. **Bas (Module Seul) :** L'échelle est en 'V'. 0 est **Noir**, mais -1 et +1 sont tous deux **Blancs (Hyper Signal)**.\nLa SG (négative) apparaît donc brillante, tout comme la SB. **Confusion & Perte de Contraste.**",
            
            "**Comparative Analysis (Same Frame):**\n\n"
            "Look at the grayscale bars on the left:\n"
            "1. **Top (Magnitude + Phase):** Linear scale. -1 is **Black**, +1 is **White**. \nGM (negative) appears dark, WM (positive) appears bright. **High Contrast.**\n\n"
            "2. **Bottom (Magnitude Only):** 'V' shape scale. 0 is **Black**, but both -1 and +1 are **White (Hyper Signal)**.\nGM (negative) thus appears bright, just like WM. **Confusion & Contrast Loss.**"
        ))
        
        # 2. EXPLICATION NULL POINT & RÔLE DU 180°
        st.success(T(
            "**💡 Pourquoi le TI annule-t-il le LCR ?**\n\n"
            "**Sans le 180° initial**, cette séquence rapide produirait une image 'plate' (type Densité de Protons) où le LCR serait gris clair.\n\n"
            "L'impulsion 180° force l'aimantation à partir de **-1**. En remontant vers **+1**, elle doit obligatoirement **croiser le Zéro**.\n"
            "👉 Si on fixe le **TI** exactement à cet instant, le LCR n'a plus de signal. Il apparaît **NOIR PUR**, ce qui est impossible sans cette préparation.",
            
            "**💡 Why does TI null the CSF?**\n\n"
            "**Without the initial 180°**, this fast sequence would yield a 'flat' image (PD type) where CSF would be light gray.\n\n"
            "The 180° pulse forces magnetization to start at **-1**. As it recovers towards **+1**, it must **cross Zero**.\n"
            "👉 If we set the **TI** exactly at this moment, the CSF has no signal. It appears **PURE BLACK**, which is impossible without this preparation."
        ))

    with col_mp_plot:
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig_mp, ax_mp = plt.subplots(2, 1, figsize=(8, 7), sharex=False)
        plt.subplots_adjust(hspace=0.4)

        # DONNÉES PÉDAGOGIQUES (Symétrie visuelle autour de 0 à 250ms)
        t = np.linspace(0, 2500, 500)
        t1_sb = 260   # SB : Passe juste au-dessus de 0
        t1_sg = 530   # SG : Reste juste en-dessous de 0
        TI_mp = 250   # Le temps de la démonstration

        # Calculs
        mz_sb_real = 1 - 2 * np.exp(-t / t1_sb)
        mz_sg_real = 1 - 2 * np.exp(-t / t1_sg)
        val_sb_real = 1 - 2 * np.exp(-TI_mp / t1_sb) 
        val_sg_real = 1 - 2 * np.exp(-TI_mp / t1_sg) 

        # =========================================================
        # GRAPHIQUE 1 : MODULE ET PHASE
        # =========================================================
        ax = ax_mp[0]
        ax.set_title(T("✅ IMAGE EN MODULE ET PHASE", "✅ MAGNITUDE AND PHASE IMAGE"), 
                     loc='center', color='green', fontweight='bold', pad=10)
        
        # Courbes
        ax.plot(t, mz_sb_real, color='black', lw=2, linestyle='-', label=T('SB', 'WM'))
        ax.plot(t, mz_sg_real, color='gray', lw=2, linestyle='--', label=T('SG', 'GM'))
        
        # Axes
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)
        ax.axvline(TI_mp, color='green', linestyle='--', alpha=0.8)
        
        # Annotation TI simple
        ax.text(TI_mp, 1.15, "TI", color='green', ha='center', fontweight='bold')

        # MARQUEURS PHASE
        col_sb_ph = (val_sb_real + 1) / 2
        col_sg_ph = (val_sg_real + 1) / 2
        ax.plot(TI_mp, val_sb_real, marker='s', markersize=14, markeredgecolor='green', color=str(col_sb_ph), zorder=10)
        ax.plot(TI_mp, val_sg_real, marker='s', markersize=14, markeredgecolor='green', color=str(col_sg_ph), zorder=10)
        
        # BARRE GAUCHE
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

        # =========================================================
        # GRAPHIQUE 2 : MODULE
        # =========================================================
        ax = ax_mp[1]
        ax.set_title(T("❌ IMAGE EN MODULE", "❌ MAGNITUDE IMAGE"), 
                     loc='center', color='red', fontweight='bold', pad=10)

        # Courbes
        ax.plot(t, mz_sb_real, color='black', lw=2, linestyle='-', label=T('SB', 'WM'))
        ax.plot(t, mz_sg_real, color='gray', lw=2, linestyle='--', label=T('SG', 'GM'))
        
        # Axes
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)
        ax.axvline(TI_mp, color='red', linestyle='--', alpha=0.8)

        # MARQUEURS MODULE
        col_sb_mag = abs(val_sb_real)
        col_sg_mag = abs(val_sg_real)
        ax.plot(TI_mp, val_sb_real, marker='s', markersize=14, markeredgecolor='red', color=str(col_sb_mag), zorder=10)
        ax.plot(TI_mp, val_sg_real, marker='s', markersize=14, markeredgecolor='red', color=str(col_sg_mag), zorder=10)
        
        # BARRE GAUCHE (En V)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("left", size="6%", pad=0.5)
        grad_v = np.abs(np.linspace(1, -1, 100)).reshape(-1, 1)
        cax.imshow(grad_v, aspect='auto', cmap='gray', extent=[0, 1, -1, 1])
        cax.set_xticks([]); cax.set_yticks([-1, 0, 1])
        cax.set_yticklabels(["-1 (Blanc)", "0", "+1 (Blanc)"])
        cax.yaxis.set_ticks_position('left')

        ax.set_xlabel("Temps (ms)")
        ax.set_ylim(-1.2, 1.2)
        ax.grid(False)
        ax.legend(loc='lower right', fontsize=8)

        st.pyplot(fig_mp)
        plt.close(fig_mp)

# [TAB 14 : ASL (PERFUSION)]
with t14:
    st.header(T("🩸 Perfusion ASL (Arterial Spin Labeling)", "🩸 ASL Perfusion (Arterial Spin Labeling)"))
    
    # --- 1. PRINCIPE & EXPLICATIONS ---
    c_principe, c_texte = st.columns([1, 1])
    
    with c_principe:
        image_asl_path = os.path.join(current_dir, "image_028fa1.jpg")
        if os.path.exists(image_asl_path): 
            st.image(
                image_asl_path, 
                caption=T("Principe ASL", "ASL Principle"), 
                use_container_width=True
            )
        else:
            # Fallback si l'image n'existe pas
            st.info(T("Image explicative non trouvée.", "Explanatory image not found."))
            
    with c_texte:
        # Markdown explicatif multilingue
        txt_fr = """
        ### Comment ça marche ?
        1.  **Marquage (Tag) :** Une impulsion "retourne" le sang au niveau du cou.
        2.  **Délai (PLD) :** On attend que le sang monte au cerveau.
        3.  **Acquisition :** On prend une image "Marquée".
        4.  **Soustraction :** Image Contrôle - Image Marquée = Perfusion.
        """
        txt_en = """
        ### How does it work?
        1.  **Labeling (Tag):** A pulse "flips" the blood at the neck level.
        2.  **Delay (PLD):** We wait for the blood to flow up to the brain.
        3.  **Acquisition:** We take a "Labeled" image.
        4.  **Subtraction:** Control Image - Labeled Image = Perfusion.
        """
        st.markdown(T(txt_fr, txt_en))
        
        # Petit focus physique
        with st.expander(T("⏱️ Focus Physique : Pourquoi TR > 4000ms ?", "⏱️ Physics Focus: Why TR > 4000ms?")):
            st.markdown(T(
                "**Cycle ASL :** Marquage (~2s) + Attente (~2s) + Acquisition (~0.5s) = TR ~4.5s",
                "**ASL Cycle:** Labeling (~2s) + Delay (~2s) + Acquisition (~0.5s) = TR ~4.5s"
            ))

    st.divider()

    # --- 2. GRAPHIQUE CHRONOGRAMME (MATPLOTLIB COMPACT) ---
    st.subheader(T("⏱️ Séquence Temporelle pCASL", "⏱️ pCASL Timing Diagram"))

    # Création de la figure (Taille compacte 7x2.2)
    fig_asl, ax_asl = plt.subplots(figsize=(7, 2.2))
    
    # Nettoyage du cadre
    ax_asl.set_xlim(0, 12)
    ax_asl.set_ylim(-1.5, 2.5)
    ax_asl.axis('off') 
    
    # Ligne de temps (axe X)
    ax_asl.plot([0.5, 11.5], [0, 0], color='black', linewidth=1)

    # Définitions des textes (Traduits)
    txt_sat_title = T("Saturation\n& Fond", "Saturation\n& Bkg")
    txt_pcasl = T("Marquage pCASL", "pCASL labeling")
    txt_bs = T("Suppression\nFond", "Bkg\nSuppression")
    txt_acq = T("Acquisition\n3D", "3D\nAcquisition")
    
    txt_dur_label = T("Durée (TL)", "Duration (TL)")
    txt_pld = "PLD"
    txt_readout = "Readout"

    # DESSIN DES ÉLÉMENTS

    # A. Flèche Saturation (Gauche)
    ax_asl.arrow(1.5, 0, 0, 1.2, head_width=0.2, head_length=0.15, color='#1565c0', lw=2, length_includes_head=True)
    ax_asl.text(1.5, 1.4, txt_sat_title, ha='center', va='bottom', fontsize=8, color='black')

    # B. Boîte pCASL Labeling (Rouge clair)
    rect_label = patches.Rectangle((3, 0), 3, 1, linewidth=1, edgecolor='black', facecolor='#ffcdd2') 
    ax_asl.add_patch(rect_label)
    ax_asl.text(4.5, 0.5, txt_pcasl, ha='center', va='center', fontsize=9, fontweight='bold')

    # C. Flèche Suppression Fond (Milieu)
    ax_asl.arrow(7, 0, 0, 1.2, head_width=0.2, head_length=0.15, color='#1565c0', lw=2, length_includes_head=True)
    ax_asl.text(7, 1.4, txt_bs, ha='center', va='bottom', fontsize=8, color='black')

    # D. Boîte Acquisition (Verte)
    rect_acq = patches.Rectangle((8.5, 0), 2.5, 1.5, linewidth=1, edgecolor='black', facecolor='#dcedc8')
    ax_asl.add_patch(rect_acq)
    ax_asl.text(9.75, 0.75, txt_acq, ha='center', va='center', fontsize=9, fontweight='bold')

    # COTATIONS DU BAS (Flèches bleues)
    def draw_double_arrow(x_start, x_end, y_pos, text_label):
        ax_asl.annotate('', xy=(x_start, y_pos), xytext=(x_end, y_pos), 
                        arrowprops=dict(arrowstyle='<->', color='#4472c4', lw=1.5))
        ax_asl.text((x_start + x_end) / 2, y_pos - 0.2, text_label, ha='center', va='top', fontsize=8)

    draw_double_arrow(3, 6, -0.3, txt_dur_label)
    draw_double_arrow(6, 8.5, -0.3, txt_pld)
    draw_double_arrow(8.5, 11, -0.3, txt_readout)

    # Affichage avec marges réduites
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    st.pyplot(fig_asl, use_container_width=False)
    plt.close(fig_asl)

    st.divider()
    
    # --- 3. SIMULATION CLINIQUE & PATHOLOGIES ---
    st.subheader(T("2. Simulation Clinique & Pathologies", "2. Clinical Simulation & Pathologies"))
    
    if HAS_NILEARN and processor.ready:
        c1_asl, c2_asl = st.columns([1, 4])
        
        with c1_asl:
            # Récupération des dimensions depuis le processeur
            dims = processor.get_dims()
            asl_slice = st.slider(T("Coupe Axiale (Z)", "Axial Slice (Z)"), 0, dims[2]-1, 90, key="asl_z")
            
            # Affichage informatif du PLD (doit être défini dans la boucle principale ou par défaut)
            pld_val = locals().get('pld', 1500) # Sécurité si pld n'est pas défini
            st.info(T(f"⏱️ **PLD Actuel : {pld_val} ms**", f"⏱️ **Current PLD: {pld_val} ms**"))
            
            if show_stroke: 
                st.error(T("⚠️ **AVC Ischémique**", "⚠️ **Ischemic Stroke**"))
            if show_atrophy: 
                st.warning(T("🧠 **Atrophie (Alzheimer)**", "🧠 **Atrophy (Alzheimer's)**"))
                
        with c2_asl:
            # Génération des cartes ASL via le module Anatomy
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
                        caption=T("1. Image Contrôle", "1. Control Image"), 
                        clamp=True, use_container_width=True
                    )
                with col_label: 
                    st.image(
                        utils.apply_window_level(label_img, 1.0, 0.5), 
                        caption=T("2. Image Marquée", "2. Labeled Image"), 
                        clamp=True, use_container_width=True
                    )
                with col_perf:
                    fig_perf, ax_perf = plt.subplots()
                    # Affichage carte perfusion (Jet colormap)
                    im = ax_perf.imshow(perf_map, cmap='jet', vmin=0, vmax=np.max(perf_map)*0.8)
                    ax_perf.axis('off')
                    st.pyplot(fig_perf)
                    st.caption(T("3. Carte de Perfusion", "3. Perfusion Map"))
            else:
                st.error("Erreur de calcul des cartes ASL.")
    else: 
        st.warning(T("Module Anatomique requis pour la simulation clinique.", "Anatomy Module required for clinical simulation."))
# [TAB : ANGIO TOF]
with t_tof:
    st.header(T("🩸 Angiographie TOF (Time of Flight)", "🩸 TOF Angiography"))

    # =========================================================
    # 1. PEDAGOGIE & SCIENCE (COMPLET)
    # =========================================================
    with st.expander(T("📘 Principe Physique : Phénomène d'Entrée de Coupe (Inflow)", "📘 Physics: Inflow Effect"), expanded=True):
        col_sci1, col_sci2 = st.columns([1, 1])
        
        with col_sci1:
            st.markdown(T("### 🌊 Le Mécanisme", "### 🌊 The Mechanism"))
            st.info(T(
                "Le TOF (Time Of Flight) est une séquence en **Écho de Gradient** qui sature les tissus fixes (Signal Noir) et laisse briller le sang frais (Signal Blanc).",
                "TOF (Time Of Flight) is a **Gradient Echo** sequence that saturates static tissues (Black Signal) and highlights fresh blood (White Signal)."
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
            """))

        with col_sci2:
            st.markdown(T("### ⚠️ Paramètres & Limites", "### ⚠️ Parameters & Limits"))
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
            """))
            
            # Schéma ASCII
            st.markdown("""
            <div style="background-color:#1e1e1e; padding:5px; border-radius:5px; color:white; font-family:monospace; text-align:center; font-size: 0.8em;">
                🌊 SANG FRAIS (Mz Max) ===> ⬛ TISSUS SATURÉS <br>
                ⬇ <br>
                ✨ SIGNAL HYPERINTENSE
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # =========================================================
    # 2. ATLAS INTERACTIF (MODE TURBO 🚀)
    # =========================================================
    
    # --- FONCTION CACHE OPTIMISÉE (Redimensionnement + RGB) ---
    @st.cache_data
    def load_tof_image_turbo(path):
        """Charge, convertit en RGB (pas d'alpha) et redimensionne pour la vitesse."""
        if os.path.exists(path):
            img = Image.open(path).convert("RGB") # RGB est plus léger que RGBA
            # Redimensionnement à 600px max (suffisant pour l'écran, très rapide à charger)
            img.thumbnail((600, 600)) 
            return img
        return None

    col_ctrl, col_view = st.columns([1.3, 2.7])

    # --- DONNÉES ANATOMIQUES ---
    # Structure : Key -> [X, Y, Groupe, Acronyme, Nom_Complet, Angle_Degré]
    
    points_axial = {
        "Com. Ant":   [0.525, 0.320, "ACA", "AComA", "AComA - Communicante Antérieure", 270],
        "ACA A2 (D)": [0.500, 0.380, "ACA", "ACA (D)", "ACA - Artère Cérébrale Ant. (D)", 250],
        "ACA A2 (G)": [0.550, 0.380, "ACA", "ACA (G)", "ACA - Artère Cérébrale Ant. (G)", 290],
        "ACM M1 (D)": [0.380, 0.400, "ACM", "ACM (D)", "ACM - Artère Cérébrale Moy. (D)", 215],
        "ACM M1 (G)": [0.640, 0.420, "ACM", "ACM (G)", "ACM - Artère Cérébrale Moy. (G)", 325],
        "Com. Post (D)": [0.470, 0.436, "PCOM", "AComP (D)", "AComP - Communicante Post. (D)", 195],
        "Com. Post (G)": [0.544, 0.436, "PCOM", "AComP (G)", "AComP - Communicante Post. (G)", 345],
        "Carotide (D)": [0.345, 0.540, "ICA", "ACI (D)", "ACI - Carotide Interne / Siphon (D)", 180],
        "Carotide (G)": [0.710, 0.540, "ICA", "ACI (G)", "ACI - Carotide Interne / Siphon (G)", 0],
        "Basilaire":  [0.525, 0.530, "BAS", "TB",      "TB - Tronc Basilaire", 90],
        "ACP P1 (D)": [0.400, 0.600, "ACP", "ACP (D)", "ACP - Artère Cérébrale Post. (D)", 160],
        "ACP P1 (G)": [0.600, 0.600, "ACP", "ACP (G)", "ACP - Artère Cérébrale Post. (G)", 20],
        "Vertébrale (D)": [0.450, 0.700, "BAS", "AV (D)", "AV - Artère Vertébrale (D)", 135],
        "Vertébrale (G)": [0.570, 0.700, "BAS", "AV (G)", "AV - Artère Vertébrale (G)", 45],
    }
    
    points_coronal = {
        "ACA (D)": [0.480, 0.250, "ACA", "ACA (D)", "ACA - Artère Cérébrale Ant. (D)", 260], 
        "ACA (G)": [0.520, 0.250, "ACA", "ACA (G)", "ACA - Artère Cérébrale Ant. (G)", 280],
        "ACM (D)": [0.350, 0.408, "ACM", "ACM (D)", "ACM - Artère Cérébrale Moy. (D)", 220], 
        "ACM (G)": [0.650, 0.422, "ACM", "ACM (G)", "ACM - Artère Cérébrale Moy. (G)", 320],
        "ACP (D)": [0.454, 0.482, "ACP", "ACP (D)", "ACP - Artère Cérébrale Post. (D)", 200], 
        "ACP (G)": [0.528, 0.482, "ACP", "ACP (G)", "ACP - Artère Cérébrale Post. (G)", 340],
        "Basilaire": [0.500, 0.626, "BAS", "TB",     "TB - Tronc Basilaire", 90], 
        "Carotide (D)": [0.390, 0.600, "ICA", "ACI (D)", "ACI - Carotide Interne (D)", 180], 
        "Carotide (G)": [0.584, 0.600, "ICA", "ACI (G)", "ACI - Carotide Interne (G)", 0],
        "Vertébrale (D)": [0.450, 0.850, "BAS", "AV (D)", "AV - Artère Vertébrale (D)", 150], 
        "Vertébrale (G)": [0.550, 0.850, "BAS", "AV (G)", "AV - Artère Vertébrale (G)", 30],
    }

    # --- UI DE CONTRÔLE ---
    with col_ctrl:
        st.subheader("Paramètres")
        view_mode = st.radio("Plan de Coupe :", ["AXIAL", "CORONAL"], label_visibility="collapsed")
        
        st.divider()
        
        active_dict = points_axial if view_mode == "AXIAL" else points_coronal
        all_keys = list(active_dict.keys())
        
        # Clé session unique
        ms_key = f"ms_sel_turbo_{view_mode}"
        if ms_key not in st.session_state:
            st.session_state[ms_key] = []

        # Boutons Rapides
        c1, c2 = st.columns(2)
        if c1.button("👁️ Tout Voir", use_container_width=True):
            st.session_state[ms_key] = all_keys
            st.rerun()
            
        if c2.button("❌ Cacher", use_container_width=True):
            st.session_state[ms_key] = []
            st.rerun()

        # Liste Repliable
        with st.expander("🔍 Sélection Individuelle", expanded=True):
            options_map = {k: v[4] for k, v in active_dict.items()}
            selected_keys = st.multiselect(
                "Cochez les structures :", 
                options=all_keys,
                format_func=lambda x: options_map[x],
                key=ms_key,
                label_visibility="collapsed"
            )

    # --- VISUALISATION ---
    with col_view:
        f_name = "tof_ax.png" if view_mode == "AXIAL" else "tof_coro.png"
        img_path = os.path.join(current_dir, f_name)
        
        # CHARGEMENT TURBO (Image 600px en cache)
        img_pil = load_tof_image_turbo(img_path)
        
        if img_pil:
            w_img, h_img = img_pil.size
            fig = px.imshow(img_pil, binary_string=True)
            
            # --- BOUSSOLE (Fixe) ---
            if view_mode == "AXIAL":
                lbl_top, lbl_bottom, lbl_left, lbl_right = "A", "P", "D", "G"
            else:
                lbl_top, lbl_bottom, lbl_left, lbl_right = "H", "B", "D", "G"
            
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

            # --- TRACÉ DES LÉGENDES (Disque Radière) ---
            cmap = { "ACA": "#00d2d3", "ACM": "#2ecc71", "ACP": "#e74c3c", "BAS": "#e67e22", "ICA": "#9b59b6", "PCOM": "#bdc3c7" }
            import math

            # Rayons ajustés
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
                
                # Correction AComA (Descendre)
                curr_rad_y = radius_y_std
                if name == "Com. Ant":
                    curr_rad_y = radius_y_std * 0.75
                
                # Calcul Position
                angle_rad = math.radians(angle_deg)
                dx = math.cos(angle_rad) * radius_x_std
                dy = math.sin(angle_rad) * curr_rad_y
                
                x_lbl = (center_x + dx) * w_img
                y_lbl = (center_y + dy) * h_img
                
                # Clamp (Bords écran)
                x_lbl = max(0.05*w_img, min(0.95*w_img, x_lbl))
                y_lbl = max(0.05*h_img, min(0.95*h_img, y_lbl))
                
                # Point
                fig.add_trace(go.Scatter(x=[x_anat], y=[y_anat], mode='markers', marker=dict(size=9, color=color, line=dict(width=1, color='white')), hoverinfo='text', text=full_desc, showlegend=False))
                # Ligne
                fig.add_trace(go.Scatter(x=[x_anat, x_lbl], y=[y_anat, y_lbl], mode='lines', line=dict(color=color, width=1, dash='dot'), hoverinfo='skip', showlegend=False))
                # Texte
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

            # Layout optimisé
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
with t15:
    st.header(T("🍔 Suppression de Graisse (Fat Sat)", "🍔 Fat Suppression (Fat Sat)"))
    
    # --- DÉFINITION DES NOMS DES ONGLETS ---
    fs_tabs_names = [
        T("1. Saturation Fréquentielle", "1. Frequency Saturation"),
        T("2. Séquence SPAIR", "2. SPAIR Sequence"),
        T("3. Séquence Dixon", "3. Dixon Sequence"),
        T("4. Excitation Eau", "4. Water Excitation"),
        T("5. Soustraction", "5. Subtraction"),
        T("6. Séquence STIR", "6. STIR Sequence"),
        T("7. Séquence PSIR", "7. PSIR Sequence")
    ]
    
    # Création des onglets
    fs_tabs = st.tabs(fs_tabs_names)
    
    # --- 1. SATURATION FRÉQUENTIELLE (FAT SAT CLASSIQUE) ---
    with fs_tabs[0]:
        st.subheader(T("1. Saturation Fréquentielle (Fat Sat)", "1. Frequency Selective Saturation (Fat Sat)"))

        # =================================================================
        # PARTIE A : LE CHRONOGRAMME (TIMING) - VALIDÉ
        # =================================================================
        st.markdown(f"#### {T('A. Séquence Temporelle', 'A. Timing Sequence')}")
        
        st.info(T(
            "1. **Au début du cycle :** Un 90° non sélectif (Bande Large). Pas de recueil de signal lors du premier TR.\n"
            "2. **À la fin du premier TR :** On utilise une bande de SAT Étroite sélective sur le pic de la Graisse.\n"
            "3. **Nouveau cycle :** Nouveau 90° (Bande Large), puis 180° (Bande Large) et Recueil du signal.",
            
            "1. **Cycle Start:** Non-selective 90° (Broad Band). No signal recording during first TR.\n"
            "2. **End of first TR:** Use of a Narrow SAT band selective on the Fat peak.\n"
            "3. **New Cycle:** New 90° (Broad Band), then 180° (Broad Band) and Signal recording."
        ))

        # --- DESSIN DU CHRONOGRAMME ---
        fig_time, ax_time = plt.subplots(figsize=(10, 6))
        
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

        ax_time.plot(time, mz_water, color='#3498db', lw=3, label=T("Eau", "Water"))
        ax_time.plot(time, mz_fat, color='#e67e22', lw=3, linestyle='--', label=T("Graisse", "Fat"))
        
        val_fat_at_sat = 1 - np.exp(-t_sat / T1_fat)
        ax_time.plot([t_sat, t_sat], [val_fat_at_sat, 0], color='#e67e22', lw=2, linestyle=':')

        y_pulse = -0.3
        h_pulse = 0.2
        
        # TR1 90
        ax_time.add_patch(patches.Rectangle((-20, y_pulse), 40, h_pulse, facecolor='#e74c3c', edgecolor='red'))
        ax_time.text(0, y_pulse + h_pulse + 0.05, "90°", ha='center', color='red', fontweight='bold')
        
        # SAT
        ax_time.add_patch(patches.Rectangle((t_sat-20, y_pulse), 40, h_pulse, facecolor='#2ecc71', edgecolor='green'))
        ax_time.text(t_sat, y_pulse + h_pulse + 0.05, "SAT", ha='center', color='green', fontweight='bold')
        ax_time.text(t_sat, y_pulse - 0.15, T("BP Étroite", "Narrow BW"), ha='center', color='green', fontsize=8)
        
        # TR2 90
        ax_time.add_patch(patches.Rectangle((t_exc-20, y_pulse), 40, h_pulse, facecolor='#e74c3c', edgecolor='red'))
        ax_time.text(t_exc, y_pulse + h_pulse + 0.05, "90°", ha='center', color='red', fontweight='bold')
        
        # 180
        t_180 = t_exc + 120
        ax_time.add_patch(patches.Rectangle((t_180-20, y_pulse), 40, h_pulse*1.2, facecolor='#e74c3c', edgecolor='red', alpha=0.6))
        ax_time.text(t_180, y_pulse + h_pulse*1.2 + 0.05, "180°", ha='center', color='#c0392b', fontsize=9, fontweight='bold')
        ax_time.text(t_180, y_pulse - 0.15, T("BP Large", "Broad BW"), ha='center', color='red', fontsize=8)
        
        # ECHO
        t_echo = t_180 + 120
        ts = np.linspace(t_echo-30, t_echo+30, 100)
        wave = np.exp(-0.005*(ts-t_echo)**2) * np.cos(0.3*(ts-t_echo)) * 0.3
        ax_time.plot(ts, wave + y_pulse + 0.1, color='black')
        ax_time.text(t_echo, y_pulse + 0.45, "ECHO", ha='center', fontweight='bold')

        y_tr_line = -0.6
        ax_time.annotate('', xy=(0, y_tr_line), xytext=(t_exc, y_tr_line), arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax_time.text(t_exc/2, y_tr_line - 0.1, T("TR 1", "TR 1"), ha='center', fontweight='bold')
        ax_time.annotate('', xy=(t_exc, y_tr_line), xytext=(t_end_plot, y_tr_line), arrowprops=dict(arrowstyle='->', color='black', lw=1.5)) 
        ax_time.text(t_exc + 150, y_tr_line - 0.1, T("TR 2", "TR 2"), ha='center', fontweight='bold')
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
        plt.close(fig_time)

        st.divider()

        # =================================================================
        # PARTIE B : SPECTRE (MODIFIÉE SELON DEMANDE)
        # =================================================================
        st.markdown(f"#### {T('B. Sélectivité Spectrale & Inhomogénéité', 'B. Spectral Selectivity & Inhomogeneity')}")

        col_ctrl, col_spec = st.columns([1, 2])
        
        with col_ctrl:
            st.write(T(
                "La Fat Sat est calibrée pour taper exactement à la fréquence de la graisse. Si le champ magnétique ($B_0$) est hétérogène, les pics se décalent vers la droite.",
                "Fat Sat is calibrated to hit exactly the fat frequency. If the magnetic field ($B_0$) is inhomogeneous, peaks shift to the right."
            ))
            st.write("")
            
            # BOUTON SIMPLE (Toggle)
            is_inhomogeneous = st.toggle(
                T("⚠️ Simuler Inhomogénéité B0", "⚠️ Simulate B0 Inhomogeneity"),
                value=False,
                key="fs_b0_toggle_simple"
            )
            
            # Logique de décalage
            if is_inhomogeneous:
                b0_shift = 80 # Décalage suffisant pour sortir de la bande
                st.error(T(
                    "❌ **ÉCHEC FAT SAT**\nL'impulsion BP étroite (verte) tire dans le vide. La graisse reste en hyper signal.",
                    "❌ **FAT SAT FAILURE**\nThe narrow BW pulse (green) misses the target. Fat remains hyperintense."
                ))
            else:
                b0_shift = 0
                st.success(T(
                    "✅ **SUCCÈS**\nChamp homogène. L'impulsion sature parfaitement la graisse.",
                    "✅ **SUCCESS**\nHomogeneous field. Pulse perfectly saturates fat."
                ))

        with col_spec:
            fig_s, ax_s = plt.subplots(figsize=(7, 4))
            
            freqs = np.linspace(-500, 200, 500)
            
            # Positions REELLES (dépendantes du bouton)
            center_water = 0 + b0_shift
            center_fat = -220 + b0_shift
            
            # Largeurs des pics
            sigma_water = 20
            sigma_fat = 25 
            
            # Pics
            peak_water = np.exp(-0.5 * ((freqs - center_water) / sigma_water)**2)
            peak_fat = np.exp(-0.5 * ((freqs - center_fat) / sigma_fat)**2)
            
            # Dessin Pics
            ax_s.fill_between(freqs, peak_water, color='#3498db', alpha=0.6, label=T('Eau', 'Water'))
            ax_s.fill_between(freqs, peak_fat, color='#e67e22', alpha=0.6, label=T('Graisse', 'Fat'))
            
            # BANDE SATURATION (FIXE & DOUBLÉE)
            # Ancienne largeur = 60. Nouvelle largeur doublée = 120.
            fixed_bw = 120 
            fixed_center = -220
            
            ax_s.axvspan(fixed_center - fixed_bw/2, fixed_center + fixed_bw/2, 
                         color='#2ecc71', alpha=0.5, label=T('Impulsion SAT (Fixe)', 'SAT Pulse (Fixed)'))
            
            # Annotation BP Etroite (On garde le texte comme demandé)
            ax_s.text(fixed_center, 1.05, T("BP Étroite", "Narrow BW"), color='green', ha='center', fontweight='bold')
            
            # Annotation 3.5 ppm (Entre les deux pics mobiles)
            mid_point = (center_water + center_fat) / 2
            ax_s.annotate('', xy=(center_water, 0.5), xytext=(center_fat, 0.5), 
                          arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
            ax_s.text(mid_point, 0.55, "3.5 ppm", ha='center', fontweight='bold')
            
            # Flèche rouge si décalage
            if b0_shift != 0:
                ax_s.annotate(T("Dérive B0", "B0 Drift"), 
                              xy=(center_water, 0.2), xytext=(0, 0.2),
                              arrowprops=dict(arrowstyle='->', color='red', lw=2), color='red', ha='center', fontsize=9, fontweight='bold')

            ax_s.set_xlim(-500, 200)
            ax_s.set_ylim(0, 1.2)
            ax_s.set_yticks([])
            ax_s.set_xlabel("Fréquence (Hz)")
            ax_s.legend(loc='upper left', fontsize='small')
            ax_s.grid(True, alpha=0.1)
            
            st.pyplot(fig_s)
            plt.close(fig_s)

    # --- 2. SPAIR (Adiabatique) ---
    with fs_tabs[1]:
        st.subheader(T("2. SPAIR (Spectral Adiabatic Inversion Recovery)", "2. SPAIR (Spectral Adiabatic Inversion Recovery)"))
        
        # =================================================================
        # PARTIE A : LE CHRONOGRAMME (FINALISÉ : PULSES ROUGES + TR DOUBLE FLÈCHE)
        # =================================================================
        st.markdown(f"#### {T('A. Séquence Temporelle (TI & Décroissance)', 'A. Timing Sequence (TI & Decay)')}")
        
        st.info(T(
            "1. **Impulsion Adiabatique (Verte) :** Inverse la Graisse (-Mz) mais laisse les autres tissus intacts (+Mz).\n"
            "2. **Attente (TI) :** La Graisse remonte vers 0.\n"
            "3. **Excitation (90°) :** À la fin du TI (ligne pointillée), on bascule. On observe alors la **décroissance exponentielle** du signal (Relaxation T2) des tissus A et B. La Graisse, elle, est éteinte.",
            
            "1. **Adiabatic Pulse (Green):** Inverts Fat (-Mz) but leaves other tissues intact (+Mz).\n"
            "2. **Wait (TI):** Fat recovers towards 0.\n"
            "3. **Excitation (90°):** At the end of TI (dotted line), flip occurs. We observe **exponential decay** (T2 Relaxation) of tissues A and B. Fat is nulled."
        ))

        # --- DESSIN DU CHRONOGRAMME ---
        fig_spair, ax_spair = plt.subplots(figsize=(10, 6.5)) 
        
        # 1. Paramètres Temporels
        TI = 180  
        TE = 60
        t_exc = TI
        t_echo = TI + TE
        t_total = t_echo + 100
        time = np.linspace(0, t_total, 1000)
        
        # Paramètres Physiques
        T1_Fat = 260   
        T2_Tissue_A = 100 
        T2_Tissue_B = 40  
        
        # 2. Calcul des Courbes
        mz_tissue_A = np.zeros_like(time)
        mz_tissue_B = np.zeros_like(time)
        mz_fat = np.zeros_like(time)
        
        for i, t in enumerate(time):
            # --- TISSUS (Bleus) ---
            if t < t_exc:
                mz_tissue_A[i] = 1.0 
                mz_tissue_B[i] = 1.0 
            else:
                dt = t - t_exc
                mz_tissue_A[i] = 1.0 * np.exp(-dt / T2_Tissue_A)
                mz_tissue_B[i] = 1.0 * np.exp(-dt / T2_Tissue_B)
                
            # --- GRAISSE (Orange) ---
            if t < t_exc:
                mz_fat[i] = 1 - 2 * np.exp(-t / T1_Fat)
            else:
                mz_fat[i] = 0
        
        # 3. Dessin des Courbes
        ax_spair.plot(time, mz_tissue_A, color='#3498db', lw=3, label=T("Tissu A (Décroissance lente)", "Tissue A (Slow decay)"))
        ax_spair.plot(time, mz_tissue_B, color='#5dade2', lw=2, linestyle=':', label=T("Tissu B (Décroissance rapide)", "Tissue B (Fast decay)"))
        ax_spair.plot(time, mz_fat, color='#e67e22', lw=3, label=T("Graisse (Inversée)", "Fat (Inverted)"))
        
        # Ligne Zéro
        ax_spair.axhline(0, color='black', lw=1)

        # LIGNE VERTICALE POINTILLÉE (FIN DU TI)
        ax_spair.axvline(TI, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax_spair.text(TI, 1.05, "TI", ha='center', fontweight='bold', fontsize=10)

        # 4. Impulsions (Bas du graphe)
        y_pulse = -1.3
        h_pulse = 0.3
        
        # A. IMPULSION ADIABATIQUE (Verte)
        ax_spair.add_patch(patches.Rectangle((-15, y_pulse), 30, h_pulse, facecolor='#2ecc71', edgecolor='green'))
        ax_spair.text(0, y_pulse + h_pulse + 0.1, "180° Adiabatic", ha='center', color='green', fontweight='bold', fontsize=9)
        ax_spair.text(0, y_pulse - 0.2, T("BP Étroite", "Narrow BW"), ha='center', color='green', fontsize=8)
        
        # B. EXCITATION (Rouge)
        x_90_start = t_exc - 10
        x_90_end = t_exc + 10
        ax_spair.add_patch(patches.Rectangle((x_90_start, y_pulse), 20, h_pulse, facecolor='#e74c3c', edgecolor='red'))
        ax_spair.text(t_exc, y_pulse + h_pulse + 0.1, "90°", ha='center', color='red', fontweight='bold')
        
        # C. REFOCALISATION (Rouge)
        t_180 = t_exc + TE/2
        x_180_start = t_180 - 10
        x_180_end = t_180 + 10
        ax_spair.add_patch(patches.Rectangle((x_180_start, y_pulse), 20, h_pulse*1.2, facecolor='#e74c3c', edgecolor='red', alpha=0.6))
        ax_spair.text(t_180, y_pulse + h_pulse*1.2 + 0.1, "180°", ha='center', color='#c0392b', fontsize=8)

        # D. GROUPEMENT "ACCOLADE" BANDE LARGE (Sous 90 et 180)
        y_bracket_top = y_pulse
        y_bracket_bottom = y_pulse - 0.15
        
        ax_spair.plot([x_90_start, x_90_start, x_180_end, x_180_end], 
                      [y_bracket_top, y_bracket_bottom, y_bracket_bottom, y_bracket_top], 
                      color='#e74c3c', lw=1.5)
        
        mid_bracket = (x_90_start + x_180_end) / 2
        ax_spair.text(mid_bracket, y_bracket_bottom - 0.1, T("Bande Large", "Broad BW"), 
                      ha='center', va='top', color='#c0392b', fontweight='bold', fontsize=9)

        # E. ECHO
        ts = np.linspace(t_echo-15, t_echo+15, 100)
        wave = np.exp(-0.01*(ts-t_echo)**2) * np.cos(0.4*(ts-t_echo)) * 0.4
        ax_spair.plot(ts, wave + y_pulse + 0.15, color='black')
        ax_spair.text(t_echo, y_pulse + 0.5, "ECHO", ha='center', fontweight='bold')
        
        # Liaison visuelle Tissu A -> Echo
        val_A_at_echo = mz_tissue_A[np.argmin(np.abs(time - t_echo))]
        ax_spair.plot([t_echo, t_echo], [val_A_at_echo, y_pulse + 0.5], color='#3498db', linestyle=':', alpha=0.5)

        # 5. Annotations
        ax_spair.annotate('', xy=(0, -1), xytext=(0, 1), arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2, ls='--'))
        ax_spair.annotate('', xy=(0, 0.1), xytext=(t_exc, 0.1), arrowprops=dict(arrowstyle='<->', color='black'))
        
        # 6. REPRÉSENTATION DU TR (DOUBLE FLÈCHE)
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
        plt.close(fig_spair)

        st.divider()

        # =================================================================
        # PARTIE B : SPECTRE (RÉTABLIE À L'IDENTIQUE SELON VOTRE DEMANDE)
        # =================================================================
        st.markdown(f"#### {T('B. Spectre : Robustesse au champ B0', 'B. Spectrum: B0 Field Robustness')}")

        # --- INTERFACE DE CONTRÔLE ---
        c_spair_ctrl, c_spair_plot = st.columns([1, 2])
        
        with c_spair_ctrl:
            
            # 1. Slider Position
            step_labels = [
                T("1. Flanc Gauche", "1. Left Slope"), 
                T("2. Centre du Pic", "2. Peak Center"), 
                T("3. Flanc Droit", "3. Right Slope")
            ]
            sweep_pos = st.select_slider(
                T("1. Position Impulsion", "1. Pulse Position"),
                options=step_labels,
                value=step_labels[0],
                key="spair_step_final_restored"
            )
            
            st.write("---")
            
            # 2. Slider B0 (Réduit à +/- 40 Hz comme demandé initialement)
            b0_shift = st.slider(
                T("2. Inhomogénéité B0 (Hz)", "2. B0 Inhomogeneity (Hz)"),
                min_value=-40, max_value=40, value=0, step=5, 
                key="spair_b0_shift_final_restored"
            )
            
            if b0_shift != 0:
                st.warning(T(f"⚠️ Pics décalés de {b0_shift} Hz.", f"⚠️ Peaks shifted by {b0_shift} Hz."))
            else:
                st.success(T("✅ Champs homogène.", "✅ Homogeneous field."))

        with c_spair_plot:
            fig_sp, ax_sp = plt.subplots(figsize=(8, 4))
            
            # A. Définition des Pics (Avec Décalage B0)
            freqs = np.linspace(-600, 200, 500)
            center_water_real = 0 + b0_shift
            center_fat_real = -220 + b0_shift
            
            water_peak = np.exp(-0.5 * ((freqs - center_water_real) / 20)**2) 
            fat_peak = np.exp(-0.5 * ((freqs - center_fat_real) / 35)**2) 
            
            # B. Logique Impulsion (Fixe & Jointive)
            pulse_width = 87.5 
            # MODIFICATION ICI : Pas de chevauchement => shift = width
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

            # C. Dessin Spectre
            ax_sp.fill_between(freqs, water_peak, color='#3498db', alpha=0.6, label=T('Eau', 'Water'))
            ax_sp.fill_between(freqs, fat_peak, color='#e67e22', alpha=0.6, label=T('Graisse', 'Fat'))
            
            # Impulsion Fixe
            ax_sp.axvspan(x_start, x_end, color='#2ecc71', alpha=0.5, label=T('Impulsion (Fixe)', 'Pulse (Fixed)'))
            
            # Flèche au dessus de l'impulsion
            ax_sp.arrow(current_center, 1.05, 0, -0.1, head_width=10, head_length=0.05, fc='green', ec='green')
            ax_sp.text(current_center, 1.1, label_txt, ha='center', color='green', fontsize=9, fontweight='bold')

            # D. Annotation 3.5 ppm (DOIT BOUGER AVEC B0)
            # Eau = 0 + shift, Graisse = -220 + shift
            # La flèche va du centre de l'eau au centre de la graisse
            
            # Coordonnées dynamiques
            x_water_arrow = 0 + b0_shift
            x_fat_arrow = -220 + b0_shift
            x_text = -110 + b0_shift
            
            # Flèche bidirectionnelle
            ax_sp.annotate('', xy=(x_water_arrow, 0.5), xytext=(x_fat_arrow, 0.5), 
                           arrowprops=dict(arrowstyle='<->', lw=1.5, color='black'))
            # Texte 3.5 ppm
            ax_sp.text(x_text, 0.55, r"$\delta = 3.5$ ppm", ha='center', fontweight='bold')

            # Indication visuelle du décalage B0 (si actif)
            if b0_shift != 0:
                ax_sp.annotate(T(f"Dérive", "Drift"), 
                               xy=(center_fat_real, 0.3), xytext=(center_fat_theo, 0.3),
                               arrowprops=dict(arrowstyle='->', color='red', lw=2), color='red', ha='center', fontsize=8)

            ax_sp.set_xlim(-600, 200)
            ax_sp.set_ylim(0, 1.35)
            ax_sp.set_xlabel("Fréquence (Hz)")
            ax_sp.set_yticks([]) 
            
            ax_sp.legend(loc='upper left', fontsize='small')
            ax_sp.grid(True, alpha=0.2)
            
            st.pyplot(fig_sp)
            plt.close(fig_sp)

    # --- 3. DIXON ---
    with fs_tabs[2]:
        st.subheader(T("3. Séquence Dixon (Chemical Shift Imaging)", "3. Dixon Sequence (Chemical Shift Imaging)"))
        st.markdown(f"#### {T('📡 A. L\'Acquisition (2 Échos)', '📡 A. Acquisition (2 Echoes)')}")
        
        c_dx1, c_dx2 = st.columns([1.2, 2])
        
        with c_dx1:
            # Sélecteur TE
            te_dixon = st.select_slider(
                T("Choisir le Temps d'Echo (TE)", "Select Echo Time (TE)"), 
                options=[2.2, 4.5], 
                key="dx_te_final_S_notation"
            )
            
            st.divider()
            
            # --- 1. S_OUT (OPPOSITION) ---
            lbl_oop = T("📉 Opposition (S_Out)", "📉 Out of Phase (S_Out)")
            # E/G en Français, W/F en Anglais
            form_oop = T(r"S_{Out} = E - G", r"S_{Out} = W - F")
            
            if te_dixon == 2.2:
                # Active : En rouge (Error style)
                with st.container():
                    st.error(f"**{lbl_oop}**")
                    st.latex(form_oop)
                    st.caption(T("Eau et Graisse s'opposent.", "Water and Fat oppose each other."))
            else:
                # Inactive : Texte simple
                st.markdown(f"**{lbl_oop}**") 
                st.latex(form_oop)

            st.write("") # Espace

            # --- 2. S_IN (PHASE) ---
            lbl_ip = T("📈 Phase (S_In)", "📈 In Phase (S_In)")
            form_ip = T(r"S_{In} = E + G", r"S_{In} = W + F")
            
            if te_dixon == 4.5:
                # Active : En vert (Success style)
                with st.container():
                    st.success(f"**{lbl_ip}**")
                    st.latex(form_ip)
                    st.caption(T("Eau et Graisse s'additionnent.", "Water and Fat sum up."))
            else:
                # Inactive : Texte simple
                st.markdown(f"**{lbl_ip}**")
                st.latex(form_ip)

        with c_dx2:
            fig_dx, ax_dx = plt.subplots(figsize=(8, 4))
            t_ms = np.linspace(0, 10, 500)
            
            # Graphique
            ax_dx.plot(t_ms, np.ones_like(t_ms), color='#3498db', label=T('Eau', 'Water'))
            ax_dx.plot(t_ms, np.cos(2 * np.pi * 220 * t_ms / 1000.0), color='#e67e22', label=T('Graisse', 'Fat'))
            
            # Point rouge interactif
            ax_dx.plot(te_dixon, np.cos(2 * np.pi * 220 * te_dixon / 1000.0), 'ro', markersize=12, label=T('Acquisition', 'Acquisition'))
            
            ax_dx.axvline(te_dixon, color='gray', linestyle='--')
            ax_dx.set_xlabel("TE (ms)")
            ax_dx.set_yticks([-1, 0, 1])
            ax_dx.set_yticklabels([T("Out", "Out"), "Quad", T("In", "In")])
            ax_dx.legend(loc='upper right'); ax_dx.grid(True, alpha=0.3)
            st.pyplot(fig_dx); plt.close(fig_dx)
            
        st.divider()
        st.markdown(f"#### {T('🧮 B. Le Calcul', '🧮 B. Calculation')}")
        c_calc1, c_calc2 = st.columns(2)
        
        # Mise à jour des formules de calcul avec S_In et S_Out
        with c_calc1: 
            st.markdown(f"##### {T('💧 Image EAU', '💧 WATER Image')}")
            st.latex(r"W = \frac{S_{In} + S_{Out}}{2}")
            
        with c_calc2: 
            st.markdown(f"##### {T('🧈 Image GRAISSE', '🧈 FAT Image')}")
            st.latex(r"F = \frac{S_{In} - S_{Out}}{2}")
    # --- 4. EXCITATION EAU ---
    with fs_tabs[3]:
        import pandas as pd 
        st.subheader(T("4. Excitation de l'Eau (Water Excitation / WE)", "4. Water Excitation (WE)"))
        st.markdown(f"#### {T('🌊 Principe : Sélection sans Saturation', '🌊 Principle: Selection without Saturation')}")
        
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
            """)
            st.markdown(txt_pulse)

        with c_we_acro:
            st.markdown(f"#### {T('🏷️ Noms Commerciaux', '🏷️ Commercial Names')}")
            # DataFrame bilingue
            col_brand = T("Constructeur", "Manufacturer")
            col_acro = T("Acronyme", "Acronym")
            
            df_names = pd.DataFrame({
                col_brand: ["Siemens / Fuji", "GE", "Philips", "Canon"], 
                col_acro: ["WE", "SSRF", "ProSET", "WET / PASTA"]
            })
            st.table(df_names.set_index(col_brand))
            
        st.divider()
        
        st.markdown(f"#### {T('🕹️ Visualisation Dynamique (Impulsion 1-1)', '🕹️ Dynamic Visualization (1-1 Pulse)')}")
        
        opt_step1 = T("1. Équilibre (M0)", "1. Equilibrium (M0)")
        opt_step2 = T("2. Premier Pulse (45°)", "2. First Pulse (45°)")
        opt_step3 = T("3. Délai (Opposition 180°)", "3. Delay (Opposition 180°)")
        opt_step4 = T("4. Second Pulse (45°)", "4. Second Pulse (45°)")
        
        step = st.select_slider(T("Étapes", "Steps"), options=[opt_step1, opt_step2, opt_step3, opt_step4], value=opt_step1)
        
        w_vec = np.array([0.0, 0.0, 1.0])
        f_vec = np.array([0.0, 0.0, 1.0])
        desc = ""
        
        if step == opt_step1:
            desc = T("Aimantation longitudinale (z).", "Longitudinal Magnetization (z).")
        elif step == opt_step2:
            val = np.sin(np.pi/4)
            w_vec = np.array([0.0, val, val])
            f_vec = np.array([0.0, val, val])
            desc = T("Pulse 45°. Tout bascule.", "Pulse 45°. Everything flips.")
        elif step == opt_step3:
            val = np.sin(np.pi/4)
            w_vec = np.array([0.0, val, val])
            f_vec = np.array([0.0, -val, val])
            desc = T("Délai : Opposition de phase.", "Delay: Phase Opposition.")
        elif step == opt_step4:
            w_vec = np.array([0.0, 1.0, 0.0])
            f_vec = np.array([0.0, 0.0, 1.0])
            desc = T("Pulse 45°. Eau à 90°, Graisse à 0°.", "Pulse 45°. Water at 90°, Fat at 0°.")
        
        c_visu1, c_visu2 = st.columns([1, 2])
        with c_visu1: 
            st.info(f"**{T('État', 'State')} :** {desc}")
            
        with c_visu2:
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot([0, 0], [0, 0], [-0.2, 1.2], 'k--', linewidth=1)
            ax.quiver(0, 0, 0, w_vec[0], w_vec[1], w_vec[2], color='#3498db', linewidth=4, arrow_length_ratio=0.1, label=T('Eau', 'Water'))
            offset = 0.05 if step in [opt_step1, opt_step2] else 0.0
            ax.quiver(offset, 0, 0, f_vec[0], f_vec[1], f_vec[2], color='#e67e22', linewidth=3, arrow_length_ratio=0.1, label=T('Graisse', 'Fat'))
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, 1.2)
            ax.view_init(elev=20, azim=20); ax.legend()
            st.pyplot(fig); plt.close(fig)

    # --- 5. SOUSTRACTION ---
    with fs_tabs[4]:
        st.subheader(T("5. Soustraction (Post - Pré)", "5. Subtraction (Post - Pre)"))
        c_sub1, c_sub2 = st.columns([1, 2])
        with c_sub1:
            move_x = st.slider(T("Mouvement Patient (px)", "Patient Motion (px)"), -10, 10, 0, 1, key="sub_move_clean")
            st.info(T("Le moindre mouvement crée des artefacts.", "The slightest movement creates artifacts."))
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
            c1.image(img_pre, caption=T("Pré", "Pre"), clamp=True)
            c2.image(img_post, caption="Post", clamp=True)
            c3.image(np.clip(img_post - img_pre, 0, 1), caption="Sub", clamp=True)

    # --- 6. STIR ---
    with fs_tabs[5]:
        st.subheader(T("6. STIR (Short Tau Inversion Recovery)", "6. STIR (Short Tau Inversion Recovery)"))
        st.markdown(f"#### {T('📡 1. Pourquoi \"Non-Sélectif\" ? (Bande Large)', '📡 1. Why \"Non-Selective\"? (Broadband)')}")
        
        col_ex1, col_ex2 = st.columns([2, 1])
        with col_ex1: 
            st.info(T("Le STIR utilise une impulsion courte qui tape **tout le spectre** (Eau, Graisse, Gado...).", 
                      "STIR uses a short pulse that hits **the entire spectrum** (Water, Fat, Gado...)."))
        with col_ex2:
            fig_bw, ax_bw = plt.subplots(figsize=(4, 2.5))
            ax_bw.fill_between(np.linspace(-500, 500, 100), 0, 1, color='purple', alpha=0.4)
            ax_bw.text(0, 0.5, T("Bande Large", "Broadband"), ha='center', color='purple')
            ax_bw.set_yticks([]); ax_bw.set_xlim(-500, 500)
            st.pyplot(fig_bw); plt.close(fig_bw)
            
        st.divider()
        # MODIFICATION TERMINOLOGIE ICI : On précise bien "Signal en Module"
        st.markdown(f"#### {T('📉 3. Visualisation (Signal en Module)', '📉 3. Visualization (Magnitude Signal)')}")
        
        c_st1, c_st2 = st.columns([1, 2])
        with c_st1:
            ti_stir = st.slider(T("Choisir le moment du 'CLIC' (TI)", "Select timing 'CLICK' (TI)"), 50, 800, 170, 10, key="st_ti_clean")
            mz_fat = 1 - 2 * np.exp(-ti_stir/260.0)
            mz_gado = 1 - 2 * np.exp(-ti_stir/280.0)
            
            # On affiche la valeur absolue (Module)
            st.metric(T("Signal Graisse (Module)", "Fat Signal (Magnitude)"), f"{abs(mz_fat):.2f}")
            
            if abs(mz_fat) < 0.1: 
                st.success(T("✅ **GRAISSE NOIRE**", "✅ **BLACK FAT**"))
            else: 
                st.warning(T("Graisse visible", "Fat visible"))
                
            if abs(mz_gado) < 0.2: 
                st.error(T("🚨 **GADO ANNULÉ**", "🚨 **GADO NULLIFIED**"))

        with c_st2:
            fig_st, (ax_st, ax_bar) = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={'width_ratios': [30, 1]})
            t_rng = np.linspace(0, 5000, 500)
            
            tissues = {
                T('Graisse (260ms)', 'Fat (260ms)'): (260, '#ff7f0e'), 
                T('Gado (280ms)', 'Gado (280ms)'): (280, 'red'), 
                T('SB (790ms)', 'WM (790ms)'): (790, '#bdc3c7'), 
                T('LCR (4000ms)', 'CSF (4000ms)'): (4000, 'cyan')
            }
            
            for name, (t1_val, col) in tissues.items():
                # TRACE EN MODULE (Valeur Absolue pour montrer le comportement STIR classique)
                # Mais on garde la courbe réelle en arrière plan ou on trace le module
                # Ici on trace la courbe réelle pour la compréhension physique, 
                # mais l'image finale (barre de droite) est en module.
                ax_st.plot(t_rng, 1 - 2 * np.exp(-t_rng / t1_val), label=name, color=col)
                
            ax_st.axhline(0, color='black')
            ax_st.axvline(ti_stir, color='green', linewidth=2, label=f'TI ({ti_stir}ms)')
            ax_st.set_xlim(0, 5000); ax_st.set_ylim(-1.1, 1.1)
            ax_st.legend(loc='lower right', fontsize=8); ax_st.grid(True, alpha=0.3)
            
            # BARRE DE DROITE : VISUALISATION EN MODULE (Valeur Absolue)
            y_grad = np.abs(np.linspace(1.1, -1.1, 200)).reshape(-1, 1)
            ax_bar.imshow(y_grad, aspect='auto', cmap='gray', vmin=0, vmax=1, extent=[0, 1, -1.1, 1.1])
            ax_bar.set_xticks([]); ax_bar.set_yticks([])
            ax_bar.set_title("Module", fontsize=8)
            # Le point suit la valeur absolue
            ax_bar.plot(0.5, 1 - 2 * np.exp(-ti_stir/260.0), 'o', color='orange', markeredgecolor='white')
            
            st.pyplot(fig_st); plt.close(fig_st)

    # --- 7. PSIR (Phase Sensitive Inversion Recovery) ---
    with fs_tabs[6]:
        st.subheader(T("7. PSIR : Robustesse vs TI Scout", "7. PSIR: Robustness vs TI Scout"))
        
        # EXPLICATION PÉDAGOGIQUE
        st.info(T(
            "**Démonstration de l'Insensibilité au TI :**\n"
            "👉 En imagerie classique (**Magnitude**), il faut un **TI Scout** précis pour annuler le myocarde (Le rendre noir). Si le TI est mal réglé (ex: Myocarde = -0.25, Fibrose = +0.25), les deux apparaissent gris identiques ($| -0.25 | = | +0.25 |$).\n"
            "👉 En **PSIR**, on garde le signe. -0.25 reste Noir, +0.25 reste Blanc. Le contraste est préservé sans réglage parfait.",
            
            "**Demonstrating TI Insensitivity:**\n"
            "👉 In classic (**Magnitude**) imaging, a precise **TI Scout** is needed to null myocardium. If TI is off (e.g. Myo = -0.25, Fib = +0.25), both appear identical grey ($| -0.25 | = | +0.25 |$).\n"
            "👉 In **PSIR**, sign is kept. -0.25 stays Black, +0.25 stays White. Contrast is preserved without perfect settings."
        ))
        
        col_psir_ctrl, col_psir_graph = st.columns([1, 2])
        
        with col_psir_ctrl:
            st.markdown(f"#### {T('🎛️ Réglage du TI', '🎛️ TI Settings')}")
            
            # Slider TI (100-500ms)
            # On met 280ms par défaut car c'est le "Piège" du Magnitude (Symétrie)
            ti_psir = st.slider(
                T("Temps d'Inversion (TI)", "Inversion Time (TI)"), 
                100, 500, 280, step=10, format="%d ms",
                key="psir_ti_slider"
            )
            
            st.divider()
            
            # Choix du mode
            mode_display = st.radio(
                T("Mode de Reconstruction", "Reconstruction Mode"),
                [T("A. Module (Nécessite TI Scout)", "A. Magnitude (Needs TI Scout)"), 
                 T("B. PSIR (Robuste)", "B. PSIR (Robust)")],
                index=1,
                key="psir_mode_radio"
            )
            
            # --- CALCUL DU CONTRASTE RÉEL (LA PREUVE) ---
            # T1 calibrés pour TI=280ms -> Myo=-0.25 / Fib=+0.25
            t1_myo_c = 596
            t1_fib_c = 286
            
            val_myo_c = 1 - 2 * np.exp(-ti_psir / t1_myo_c)
            val_fib_c = 1 - 2 * np.exp(-ti_psir / t1_fib_c)
            
            # Contraste Magnitude (Différence des valeurs absolues)
            con_mag = abs(abs(val_myo_c) - abs(val_fib_c))
            
            # Contraste PSIR (Différence réelle)
            con_psir = abs(val_myo_c - val_fib_c)
            
            st.divider()
            st.markdown(f"#### {T('📊 Score de Contraste', '📊 Contrast Score')}")
            
            # Affichage des scores
            c_score1, c_score2 = st.columns(2)
            c_score1.metric("Magnitude", f"{con_mag:.2f}", delta_color="off")
            c_score2.metric("PSIR", f"{con_psir:.2f}", delta="Robuste" if con_psir > 0.4 else "Faible")
            
            # ANALYSE AUTOMATIQUE (LA DÉMONSTRATION)
            if con_mag < 0.1 and con_psir > 0.4:
                st.error(T(
                    "🚨 **ÉCHEC MAGNITUDE !**\nLe contraste est nul (<0.1) car les signaux sont symétriques.\nC'est ici qu'il aurait fallu un **TI Scout** parfait.", 
                    "🚨 **MAGNITUDE FAIL!**\nContrast is null (<0.1) because signals are symmetric.\nThis is where a perfect **TI Scout** was needed."
                ))
                st.success(T(
                    "✅ **SUCCÈS PSIR !**\nMalgré le 'mauvais' TI, le contraste reste énorme (>0.4).\n👉 **Preuve de l'insensibilité au TI.**",
                    "✅ **PSIR SUCCESS!**\nDespite 'bad' TI, contrast remains huge (>0.4).\n👉 **Proof of TI insensitivity.**"
                ))
            elif con_mag > 0.3:
                 st.info(T("Bon TI pour le Module (Chance ou TI Scout réussi).", "Good TI for Magnitude (Luck or good TI Scout)."))


        with col_psir_graph:
            # Figure
            fig_psir, ax_psir = plt.subplots(figsize=(10, 6))
            
            # 1. PARAMÈTRES
            tr_sim = 1000 
            t = np.linspace(0, tr_sim, 1000)
            
            # Courbes PHYSIQUES (Signées)
            mz_myo_raw = 1 - 2 * np.exp(-t / t1_myo_c)
            mz_blood_raw = 1 - 2 * np.exp(-t / t1_fib_c)
            
            # Valeurs au TI
            pt_myo = val_myo_c
            pt_fib = val_fib_c
            
            # 2. LOGIQUE D'AFFICHAGE
            if mode_display.startswith("A"):
                # MODE MODULE
                y_myo = np.abs(mz_myo_raw)
                y_blood = np.abs(mz_blood_raw)
                pt_myo_plot = np.abs(pt_myo)
                pt_fib_plot = np.abs(pt_fib)
                
                y_label = "Signal |Mz| (Rectifié)"
                title_g = T("Reconstruction Module (Tout est Positif)", "Magnitude Reconstruction (All Positive)")
                
                # Echelle 0 à 1
                ymin_graph, ymax_graph = -0.2, 1.2 # Un peu de marge négative pour l'esthétique
                grad_min, grad_max = 1, 0
                grad_extent = [0, 1, 0, 1] 
                yticks = [0, 0.5, 1]
                
            else:
                # MODE PSIR
                y_myo = mz_myo_raw
                y_blood = mz_blood_raw
                pt_myo_plot = pt_myo
                pt_fib_plot = pt_fib
                
                y_label = "Aimantation Mz (Réelle)"
                title_g = T("Reconstruction PSIR (Signe Conservé)", "PSIR Reconstruction (Sign Preserved)")
                
                # Echelle -1 à 1
                ymin_graph, ymax_graph = -1.8, 1.3
                grad_min, grad_max = 1, -1
                grad_extent = [0, 1, -1, 1]
                yticks = [-1, -0.5, 0, 0.5, 1]

            # 3. TRACÉ DES COURBES
            ax_psir.plot(t, y_myo, label=T("Myocarde Sain", "Healthy Myocardium"), color='#3498db', lw=2)
            ax_psir.plot(t, y_blood, label=T("Fibrose (Gado)", "Fibrosis (Gado)"), color='#e74c3c', lw=2)
            
            # Ligne Zéro & TI
            ax_psir.axhline(0, color='black', lw=1)
            ax_psir.axvline(ti_psir, color='gray', linestyle='--', alpha=0.8)
            ax_psir.text(ti_psir, 1.15, f"TI = {ti_psir}ms", color='gray', ha='center', fontweight='bold')
            
            # Points interactifs
            ax_psir.plot(ti_psir, pt_myo_plot, 'o', color='#3498db', markersize=10, markeredgecolor='black', zorder=10)
            ax_psir.plot(ti_psir, pt_fib_plot, 'o', color='#e74c3c', markersize=10, markeredgecolor='black', zorder=10)

            # 4. CHRONOGRAMME (PARTIE BASSE - Uniquement si PSIR pour clarté ou toujours ?)
            # On l'affiche toujours pour montrer la séquence, mais on adapte la position
            y_seq_base = -1.5 if not mode_display.startswith("A") else -0.15 # Position différente si module
            if mode_display.startswith("A"): 
                # En mode module, on ne dessine pas le chronogramme complet pour ne pas écraser les courbes si elles sont basses
                # On simplifie ou on cache. Pour être pédagogique, on le cache en mode Module pour focus sur le "Crash" des courbes.
                pass 
            else:
                # DESSIN SÉQUENCE EN MODE PSIR
                h_pulse = 0.25
                # A. IMPULSION 180°
                rect_180 = patches.Rectangle((0, y_seq_base), 40, h_pulse, facecolor='#c0392b', edgecolor='black', linewidth=1)
                ax_psir.add_patch(rect_180)
                ax_psir.text(20, y_seq_base - 0.15, "180°", ha='center', color='#c0392b', fontsize=8, fontweight='bold')
                
                # B. LECTURE
                n_readout = 8
                for k in range(n_readout):
                    pos_x = ti_psir + (k * 15)
                    rect_read = patches.Rectangle((pos_x, y_seq_base), 10, h_pulse*0.7, facecolor='#f1c40f', edgecolor='orange')
                    ax_psir.add_patch(rect_read)
                ax_psir.text(ti_psir + (n_readout*15)/2, y_seq_base - 0.15, "Lecture", ha='center', color='#d35400', fontsize=8)

            # 5. MISE EN FORME
            ax_psir.set_ylim(ymin_graph, ymax_graph)
            ax_psir.set_xlim(-50, tr_sim + 50)
            ax_psir.set_yticks(yticks)
            ax_psir.set_xlabel("Temps (ms)")
            ax_psir.set_ylabel(y_label)
            ax_psir.set_title(title_g)
            ax_psir.legend(loc='upper right')
            ax_psir.grid(True, alpha=0.3)
            
            # --- BARRE DE GRIS ---
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
            cax.set_ylabel(T("Contraste", "Contrast"))
            
            st.pyplot(fig_psir)
            plt.close(fig_psir)
with t16:
    st.header(T("🔥 Sécurité RF : Console de Contrôle", "🔥 RF Safety: Control Console"))
    
    # --- 0. AVERTISSEMENT ---
    st.warning(T(
        "⚠️ **Simulateur Clinique :** Module reproduisant les contraintes réelles (IEC 60601-2-33). **Ce module est déconnecté du reste du simulateur et ne doit PAS être utilisé à des fins cliniques.**",
        "⚠️ **Clinical Simulator:** Module reproducing real constraints (IEC 60601-2-33). **This module is disconnected from the rest of the simulator and must NOT be used for clinical purposes.**"
    ))

    # --- 1. CONFIGURATION ---
    SAR_CALIB_FACTOR = 0.005
    
    # Définition dynamique pour la traduction des descriptions
    # Les clés restent techniques pour la logique, mais on peut les afficher traduites si besoin
    # Ici, je traduis les clés directement pour le menu déroulant
    
    k_sinc = T("Sinc (Standard 2D)", "Sinc (Standard 2D)")
    k_rect = T("Rect (Hard Pulse 3D)", "Rect (Hard Pulse 3D)")
    k_gauss = T("Gauss (Sélectif)", "Gauss (Selective)")

    PULSE_LIBRARY = {
        k_sinc:  {"factor": 1.0, "desc": T("Pour coupes 2D nettes (SE, TSE, GRE)", "For sharp 2D slices (SE, TSE, GRE)")},
        k_rect:  {"factor": 1.4, "desc": T("Pour volumes 3D rapides (MP-RAGE)", "For fast 3D volumes (MP-RAGE)")},
        k_gauss: {"factor": 0.7, "desc": T("Pour Saturation ou Inversion", "For Saturation or Inversion")}
    }
    
    RF_MODES = {
        "Low SAR": 0.8,
        "Normal": 1.0,
        "High Power": 1.2
    }

    # --- 2. ENTRÉES UTILISATEUR ---
    c_pat, c_seq, c_scan = st.columns(3)
    
    with c_pat:
        st.markdown(f"#### {T('👤 Patient', '👤 Patient')}")
        weight = st.number_input(T("Poids (kg)", "Weight (kg)"), 30.0, 150.0, 75.0, 5.0, key="sar_w_tabfinal")
        height = st.number_input(T("Taille (m)", "Height (m)"), 1.0, 2.2, 1.75, 0.05, key="sar_h_tabfinal")

    with c_seq:
        st.markdown(f"#### {T('📡 Séquence', '📡 Sequence')}")
        
        # Options traduites mais contenant les acronymes pour la logique (SE, TSE, GRE)
        opt_se = T("Spin Echo (SE)", "Spin Echo (SE)")
        opt_tse = T("Turbo Spin Echo (TSE)", "Turbo Spin Echo (TSE)")
        opt_gre = T("Echo de Gradient (GRE)", "Gradient Echo (GRE)")
        
        seq_type = st.selectbox(T("Type Séquence", "Sequence Type"), [opt_se, opt_tse, opt_gre], key="sar_type_tabfinal")
        
        b0_val = st.radio(T("Champ Magnétique (B0)", "Magnetic Field (B0)"), [1.5, 3.0], horizontal=True, key="sar_b0_tabfinal")
        
        pulse_shape = st.selectbox(T("Forme Onde", "Waveform"), list(PULSE_LIBRARY.keys()), index=0, key="sar_shape_tabfinal")
        
        # Presets et Labels Dynamiques
        if "GRE" in seq_type:
            def_etl, def_ang = 0, 20
            label_angle = T("Angle d'Excitation (α)", "Excitation Angle (α)")
            help_angle = T("Angle de bascule (5° à 90° en clinique)", "Flip angle (5° to 90° clinical)")
        elif "TSE" in seq_type: 
            def_etl, def_ang = 3, 180
            label_angle = T("Angle de Refoc (°)", "Refoc Angle (°)")
            help_angle = T("Angle des impulsions de refocalisation", "Refocusing pulse angle")
        else: 
            def_etl, def_ang = 0, 180
            label_angle = T("Angle de Refoc (°)", "Refoc Angle (°)")
            help_angle = T("Angle de l'impulsion de refocalisation", "Refocusing pulse angle")

        angle = st.slider(label_angle, 5, 180, def_ang, key="sar_angle_tabfinal", help=help_angle)
        
        if "TSE" in seq_type:
            etl = st.slider(T("ETL (Facteur Turbo)", "ETL (Turbo Factor)"), 2, 64, def_etl, key="sar_etl_tabfinal")
        else:
            etl = 0
            st.slider("ETL", 0, 1, 0, disabled=True, key="sar_etl_dis_tabfinal")

    with c_scan:
        st.markdown(f"#### {T('⚙️ Paramètres Scan', '⚙️ Scan Settings')}")
        tr = st.number_input("TR (ms)", 20, 10000, 600, 50, key="sar_tr_tabfinal", help=T("Temps de Répétition", "Repetition Time"))
        nb_slices = st.slider(T("Nombre de Coupes", "Number of Slices"), 1, 60, 20, key="sar_slices_tabfinal")
        
        rf_mode_name = st.select_slider("Mode RF", options=list(RF_MODES.keys()), value="Normal", key="sar_mode_tabfinal")
        rf_intensity = RF_MODES[rf_mode_name]
        
        nex = 1; matrix = 256
        scan_time_sec = (tr * matrix * nex) / 1000
        if etl > 1: scan_time_sec = scan_time_sec / etl
        
        st.caption(f"⏱️ Scan : {int(scan_time_sec//60)}min {int(scan_time_sec%60)}s")

    st.divider()

    # --- 3. MOTEUR PHYSIQUE ---
    factor_b0 = (b0_val / 1.5) ** 2 
    
    # Calcul Énergie
    energy_90 = 1.0 
    energy_angle_slider = (angle / 90.0) ** 2 
    
    if "GRE" in seq_type:
        total_energy_per_slice = energy_angle_slider 
    elif "TSE" in seq_type:
        total_energy_per_slice = energy_90 + (etl * energy_angle_slider) 
    else: # Spin Echo
        total_energy_per_slice = energy_90 + (1 * energy_angle_slider) 
    
    # Puissance SAR
    total_energy_per_tr = total_energy_per_slice * nb_slices
    power_factor = total_energy_per_tr / (tr / 1000.0)

    # Autres Facteurs
    factor_weight = 75.0 / weight
    factor_shape = PULSE_LIBRARY[pulse_shape]["factor"]
    
    sar_val = SAR_CALIB_FACTOR * factor_b0 * power_factor * factor_weight * rf_intensity * factor_shape
    
    # Calcul B1+rms
    peak_angle = angle 
    b1_peak_est = (peak_angle / 90.0) * 4.0 * rf_intensity 
    
    if "GRE" in seq_type: p_count = 1
    elif "TSE" in seq_type: p_count = 1 + etl
    else: p_count = 2
    
    duty_cycle = (p_count * nb_slices * 2.5) / tr 
    duty_cycle = min(duty_cycle, 1.0)
    
    b1_rms_ut = b1_peak_est * np.sqrt(duty_cycle) * factor_shape

    # --- 4. VISUALISATION ---
    st.subheader(T("📊 Moniteurs de Sécurité", "📊 Safety Monitors"))
    
    c_visu_g, c_visu_d = st.columns([1, 1])
    
    with c_visu_g:
        st.markdown(f"##### {T('📉 Profil RF & Charge', '📉 RF Profile & Load')}")
        
        fig_p, ax_p = plt.subplots(figsize=(5, 2.5))
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
        
        title_b1 = T(f"Pic B1: {b1_peak_est:.1f} µT (x{factor_b0:.0f} énergie à {b0_val}T)",
                     f"B1 Peak: {b1_peak_est:.1f} µT (x{factor_b0:.0f} energy at {b0_val}T)")
        ax_p.set_title(title_b1, fontsize=9, color='gray')
        ax_p.grid(True, alpha=0.3)
        st.pyplot(fig_p); plt.close(fig_p)
        
        if b0_val == 3.0:
            st.error(T("⚠️ **ATTENTION 3T** : Énergie x4 par rapport à 1.5T.", 
                       "⚠️ **WARNING 3T**: Energy x4 compared to 1.5T."))

    with c_visu_d:
        def draw_gauge_cursor(value, label, limit_norm, limit_first, max_scale=6.0):
            fig, ax = plt.subplots(figsize=(6, 2))
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

        st.pyplot(draw_gauge_cursor(sar_val, T("SAR Global (W/kg)", "Global SAR (W/kg)"), 2.0, 4.0))
        
        if sar_val > 4.0:
            st.error(T("🚨 **BLOCAGE** : SAR > 4 W/kg.", "🚨 **LOCKOUT**: SAR > 4 W/kg."))
        elif sar_val > 2.0:
            st.warning(T("⚠️ **MODE CONTRÔLÉ** : Surveillance requise.", "⚠️ **CONTROLLED MODE**: Monitoring required."))
        
        st.pyplot(draw_gauge_cursor(b1_rms_ut, "B1+rms (µT)", 2.8, 4.0))

    st.divider()
    
    # --- 5. FORMULES & GLOSSAIRES ---
    c_f1, c_f2 = st.columns(2)
    with c_f1:
        st.markdown(f"#### {T('🌡️ Calcul du SAR', '🌡️ SAR Calculation')}")
        # La formule est universelle (maths)
        st.latex(r"SAR \propto B_0^2 \times E_{totale} \times \frac{1}{TR \cdot Poids}")
        # Note : Poids est compréhensible en EN aussi ou traduit mentalement, sinon on peut mettre Weight
    with c_f2:
        st.markdown(f"#### {T('⚡ Calcul du B1+rms', '⚡ B1+rms Calculation')}")
        st.latex(r"B_{1}^{+rms} \propto B_{1,peak} \times \sqrt{DC}")

    c_exp1, c_exp2 = st.columns(2)
    with c_exp1:
        with st.expander(T("📖 Facteurs SAR (Détails)", "📖 SAR Factors (Details)")):
             if "GRE" in seq_type:
                st.markdown(T("""
                * **Type** : Écho de Gradient (GRE).
                * **Énergie** : Une seule impulsion d'excitation (Angle variable de **5° à 90°**).
                * **Analyse** : Le SAR est réduit car il n'y a **pas de train d'impulsions de refocalisation**.
                """, """
                * **Type**: Gradient Echo (GRE).
                * **Energy**: Single excitation pulse (Variable angle **5° to 90°**).
                * **Analysis**: SAR is low because there is **no refocusing pulse train**.
                """))
             else:
                st.markdown(T(f"""
                * **Type** : Spin Echo / TSE.
                * **Énergie** : Excitation 90° (Fixe) + Refocalisations {angle}° (Variable).
                * **Poids du 180°** : Un pulse 180° chauffe **4x** plus qu'un 90°.
                """, f"""
                * **Type**: Spin Echo / TSE.
                * **Energy**: Excitation 90° (Fixed) + Refocusing {angle}° (Variable).
                * **Weight of 180°**: A 180° pulse heats **4x** more than a 90°.
                """))
                
    with c_exp2:
        with st.expander(T("📖 Facteurs B1+rms (Détails)", "📖 B1+rms Factors (Details)")):
             if "GRE" in seq_type:
                st.markdown(T("""
                * **B1 Peak** : Dépend de l'angle $\\alpha$.
                * **Duty Cycle** : Très faible (1 pulse par TR).
                """, """
                * **B1 Peak**: Depends on angle $\\alpha$.
                * **Duty Cycle**: Very low (1 pulse per TR).
                """))
             else:
                st.markdown(T("""
                * **B1 Peak** : Intensité des pulses de refocalisation.
                * **Duty Cycle** : Élevé en TSE (mitraillage).
                """, """
                * **B1 Peak**: Refocusing pulse intensity.
                * **Duty Cycle**: High in TSE (rapid firing).
                """))
    
    st.divider()
    
    # --- 6. INFO & SEUILS COLORÉS ---
    with st.expander(T("📝 Seuils & Paramètres IEC", "📝 IEC Thresholds & Parameters"), expanded=False):
        st.markdown(T("""
        * 🟢 :green[**Mode Normal**] : **< 2.0 W/kg** (Routine Clinique, aucun risque).
        * 🟠 :orange[**Mode Contrôlé (Niveau 1)**] : **2.0 - 4.0 W/kg** (Surveillance médicale requise).
        * 🔴 :red[**Mode Restreint (Niveau 2)**] : **> 4.0 W/kg** (Blocage logiciel, risque d'échauffement > 1°C).
        """, """
        * 🟢 :green[**Normal Mode**]: **< 2.0 W/kg** (Clinical Routine, no risk).
        * 🟠 :orange[**First Level Mode**]: **2.0 - 4.0 W/kg** (Medical supervision required).
        * 🔴 :red[**Second Level Mode**]: **> 4.0 W/kg** (Software lockout, heating risk > 1°C).
        """))

    # --- RESTITUTION DU TABLEAU CLINIQUE DÉTAILLÉ (Markdown HTML String) ---
    with st.expander(T("🏥 Clinique : Formes d'Impulsions & Séquences", "🏥 Clinical: Pulse Shapes & Sequences"), expanded=True):
        
        # En-têtes et Contenu traduits
        h_shape = T("Forme", "Shape")
        h_usage = T("Usage Principal", "Main Usage")
        h_adv = T("Avantage", "Advantage")
        h_risk = T("Risque / Inconvénient", "Risk / Drawback")
        
        # Ligne Sinc
        sinc_usage = "TSE, SE (2D)"
        sinc_adv = T("Profil de coupe rectangulaire (Pas de croisement).", "Rectangular slice profile (No crosstalk).")
        sinc_risk = T("**SAR Élevé** (Impulsions longues & nombreuses).", "**High SAR** (Long & numerous pulses).")
        
        # Ligne Rect
        rect_name = T("Rectangulaire", "Rectangular")
        rect_usage = "MP-RAGE (3D)"
        rect_adv = T("Ultra-rapide (TR court).", "Ultra-fast (Short TR).")
        rect_risk = T("Coupe \"sale\" (bords flous) - corrigé par encodage 3D.", "\"Dirty\" slice (blurred edges) - corrected by 3D encoding.")
        
        # Ligne Gauss
        gauss_name = T("Gaussienne", "Gaussian")
        gauss_usage = "Fat Sat"
        gauss_adv = T("Très sélectif en fréquence.", "Frequency selective.")
        gauss_risk = T("**Pic B1 Élevé** (Stress sur l'ampli RF).", "**High B1 Peak** (RF Amp Stress).")

        st.markdown(f"""
        | {h_shape} | {h_usage} | {h_adv} | {h_risk} |
        | :--- | :--- | :--- | :--- |
        | **Sinc** | **{sinc_usage}** | {sinc_adv} | {sinc_risk} |
        | **{rect_name}** | **{rect_usage}** | {rect_adv} | {rect_risk} |
        | **{gauss_name}** | **{gauss_usage}** | {gauss_adv} | {gauss_risk} |
        """)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import streamlit as st

# ==============================================================================
# [TAB 17 : ARCHITECTURE - VERSION NETTOYÉE ET EXPLICATIVE]
# ==============================================================================
with t17:
    st.header(T("🏗️ Architecture : Structure et Composants IRM", "🏗️ Architecture: MRI Structure and Components"))
    
    col_view, col_desc = st.columns([2.5, 1])
    
    with col_desc:
        view_mode = st.radio(
            T("Progression pédagogique :", "Pedagogical progression:"),
            [
                T("1. Machine (Coque & Tunnel)", "1. Machine (Shell & Bore)"),
                T("2. Cryostat (Cylindre Hélium)", "2. Cryostat (Helium Cylinder)"),
                "3. Aimant (B0 & Supra)", 
                "4. Bobines de Shim (Orange)", 
                "5. GZ (Maxwell - Vert)", 
                "6. GY (Golay - Jaune)", 
                "7. GX (Golay - Bleu)", 
                T("8. Tout visualiser", "8. Show All")
            ],
            index=0, key="arch_final_clean"
        )
        
        st.divider()
        st.write(T("🔬 **Analyse** : Visualisation des couches internes de l'aimant.", 
                   "🔬 **Analysis**: Visualizing the magnet's internal layers."))

    with col_view:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black'); fig.patch.set_facecolor('black')

        mode_idx = ["1", "2", "3", "4", "5", "6", "7", "8"].index(view_mode[0])
        show_all = (mode_idx == 7)

        def get_poids(target_idx):
            if show_all:
                if target_idx >= 4: return 1.0  # Gradients à 100%
                return 0.3                      # Reste à 30%
            if target_idx == mode_idx: return 1.0
            if target_idx == 1 and mode_idx == 2: return 0.6 # Cryo pendant B0
            if target_idx < mode_idx: return 0.25
            return 0.0

        def draw_perfect_edges(r, l, alpha):
            """Rétablit les arêtes XYZ de la coque"""
            for y in [-r, r]:
                for z in [-r, r]:
                    ax.plot([-l, l], [y, y], [z, z], color='white', lw=1.5, alpha=alpha)
            for x in [-l, l]:
                for side in [-r, r]:
                    ax.plot([x, x], [-r, r], [side, side], color='white', lw=1, alpha=alpha*0.6)
                    ax.plot([x, x], [side, side], [-r, r], color='white', lw=1, alpha=alpha*0.6)

        def draw_bipolar_ramp(color, mode, alpha):
            """Rampes visibles uniquement en mode individuel"""
            if not show_all:
                pts = np.linspace(-1.1, 1.1, 40)
                for p in pts:
                    if abs(p) < 0.05: continue
                    amp = p * 0.4
                    if mode == "GZ": ax.plot([p, p], [0, 0], [0, amp], color=color, lw=3, alpha=alpha)
                    elif mode == "GX": ax.plot([0, 0], [p, p], [0, amp], color=color, lw=3, alpha=alpha)
                    elif mode == "GY": ax.plot([0, 0], [0, amp], [p, p], color=color, lw=3, alpha=alpha)
                l_s = 1.35
                ax.text(l_s if mode=="GZ" else 0, l_s if mode=="GX" else 0, l_s if mode=="GY" else 0, "+", color=color, fontsize=18, weight='bold', ha='center', alpha=alpha)
                ax.text(-l_s if mode=="GZ" else 0, -l_s if mode=="GX" else 0, -l_s if mode=="GY" else 0, "-", color=color, fontsize=18, weight='bold', ha='center', alpha=alpha)

        def draw_coil(z_r, t_r, color, alpha):
            t_g = np.deg2rad(np.linspace(t_r[0], t_r[1], 30)); r = 1.05
            ax.plot([z_r[0], z_r[1]], [r*np.cos(t_g[0])]*2, [r*np.sin(t_g[0])]*2, color=color, lw=5, alpha=alpha)
            ax.plot([z_r[0], z_r[1]], [r*np.cos(t_g[-1])]*2, [r*np.sin(t_g[-1])]*2, color=color, lw=5, alpha=alpha)
            ax.plot([z_r[0]]*30, r*np.cos(t_g), r*np.sin(t_g), color=color, lw=5, alpha=alpha)
            ax.plot([z_r[1]]*30, r*np.cos(t_g), r*np.sin(t_g), color=color, lw=5, alpha=alpha)

        # --- 1. MACHINE & TUNNEL ---
        p_m = get_poids(0)
        if p_m > 0 or show_all:
            draw_perfect_edges(1.9, 2.2, p_m)
            a_tun = 0.5 if show_all else 0.15
            z_t = np.linspace(-2.2, 2.2, 40); t_t = np.linspace(0, 2*np.pi, 40)
            if show_all: t_t = np.linspace(np.pi/2, 2*np.pi, 40)
            Z_t, T_t = np.meshgrid(z_t, t_t)
            ax.plot_surface(Z_t, 0.85*np.cos(T_t), 0.85*np.sin(T_t), color='white', alpha=a_tun)

        # --- 2. CRYOSTAT (CYLINDRE) ---
        p_c = get_poids(1)
        if p_c > 0:
            z_c = np.linspace(-2.0, 2.0, 40); t_c_s = np.linspace(0, 2*np.pi, 40)
            if show_all: t_c_s = np.linspace(np.pi/2, 2*np.pi, 40)
            Z_c, T_c = np.meshgrid(z_c, t_c_s)
            ax.plot_surface(Z_c, 1.75*np.cos(T_c), 1.75*np.sin(T_c), color='#00d2ff', alpha=p_c*0.2)

        # --- 3. AIMANT SUPRA (B0) ---
        p_s = get_poids(2)
        if p_s > 0:
            z_h = np.linspace(-1.9, 1.9, 800)
            ax.plot(z_h, 1.6*np.cos(40*np.pi*z_h), 1.6*np.sin(40*np.pi*z_h), color='#8e44ad', alpha=p_s, lw=1.5)
            ax.quiver(-2.4, 0, 0, 4.8, 0, 0, color='white', lw=1.5, alpha=p_s, arrow_length_ratio=0.06)
            ax.text(2.2, 0, 0.3, "B0", color='white', fontsize=12, weight='bold', alpha=p_s)

        # --- 4. SHIM (ORANGE) ---
        p_sh = get_poids(3)
        t_circ = np.linspace(0, 2*np.pi, 100)
        if p_sh > 0:
            for z_p in [-1.8, 1.8]:
                ax.plot([z_p]*100, 1.45*np.cos(t_circ), 1.45*np.sin(t_circ), color='orange', lw=6, alpha=p_sh)

        # --- 5, 6, 7. GRADIENTS ---
        # GZ
        p_gz = get_poids(4)
        if p_gz > 0:
            ax.plot([0.8]*100, np.cos(t_circ), np.sin(t_circ), color='#27ae60', lw=7, alpha=p_gz)
            ax.plot([-0.8]*100, np.cos(t_circ), np.sin(t_circ), color='#27ae60', lw=7, alpha=p_gz)
            if mode_idx == 4: draw_bipolar_ramp('#27ae60', "GZ", 1.0)
        # GY
        p_gy = get_poids(5)
        if p_gy > 0:
            for z in [[0.1, 0.75], [-0.75, -0.1]]:
                for t in [[65, 115], [245, 295]]: draw_coil(z, t, '#f1c40f', p_gy)
            if mode_idx == 5: draw_bipolar_ramp('#f1c40f', "GY", 1.0)
        # GX
        p_gx = get_poids(6)
        if p_gx > 0:
            for z in [[0.1, 0.75], [-0.75, -0.1]]:
                for t in [[-25, 25], [155, 205]]: draw_coil(z, t, '#2980b9', p_gx)
            if mode_idx == 6: draw_bipolar_ramp('#2980b9', "GX", 1.0)

        ax.view_init(elev=22, azim=-125)
        ax.set_axis_off()
        st.pyplot(fig)

    # --- EXPLICATIONS DÉTAILLÉES ---
    st.divider()
    cols = st.columns(3)
    with cols[0]:
        st.subheader("🧊 Cryogénie & B0")
        st.write("**Coque & Tunnel** : Structure mécanique et accueil du patient.")
        st.write("**Cryostat** : Enceinte thermique isolant l'hélium liquide (-269°C).")
        st.write("**Aimant Supra** : Bobinage principal générant le champ statique stable $B_0$.")
    with cols[1]:
        st.subheader("🎯 Homogénéité (Shim)")
        st.write("**Bobines de Shim** : Compensent les inhomogénéités locales du champ magnétique.")
        st.write("**Impact** : Essentiel pour obtenir une résonance précise sur l'ensemble de la zone imagée.")
    with cols[2]:
        st.subheader("📡 Codage Spatial")
        st.write("**GZ (Vert)** : Maxwell coils. Sélection de la coupe transversale.")
        st.write("**GY/GX (Jaune/Bleu)** : Golay coils. Codage de phase et de fréquence.")
        st.write("**Rampes +/-** : Illustrent la variation linéaire de champ induite par les gradients.")