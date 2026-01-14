# anatomie.py - VERSION 7.96 (GEOMETRIE FINE : CC ARQUÉ + CERVELET)
import numpy as np
import streamlit as st
from scipy.ndimage import gaussian_filter, zoom, binary_erosion, binary_dilation, sobel

# Gestion des imports optionnels
HAS_NILEARN = False
try:
    import nibabel as nib
    from nilearn import datasets, image
    HAS_NILEARN = True
except ImportError:
    pass

# --- FONCTION DE CHARGEMENT HORS CLASSE ---
@st.cache_resource
def load_nilearn_data():
    """
    Charge uniquement les bases indispensables (MNI + Harvard-Oxford).
    """
    if not HAS_NILEARN: return None
    
    print("--- Chargement des données Anatomiques ---")
    dataset_mni = datasets.fetch_icbm152_2009()
    mni_img = nib.load(dataset_mni['gm']) 
    
    gm = mni_img.get_fdata()
    wm = nib.load(dataset_mni['wm']).get_fdata()
    csf = nib.load(dataset_mni['csf']).get_fdata()
    rest = np.clip(1.0 - (gm + wm + csf), 0, 1)

    ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    res_cort = image.resample_to_img(ho_cort.maps, mni_img, interpolation='nearest')
    
    ho_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    res_sub = image.resample_to_img(ho_sub.maps, mni_img, interpolation='nearest')

    return {
        'gm': gm, 'wm': wm, 'csf': csf, 'rest': rest,
        'atlas_cort_data': res_cort.get_fdata(), 'atlas_cort_labels': ho_cort.labels,
        'atlas_sub_data': res_sub.get_fdata(), 'atlas_sub_labels': ho_sub.labels
    }

class AdvancedMRIProcessor:
    def __init__(self):
        self.ready = False
        self.static_vein_pattern = None 
        
        # Initialisation des données
        self.gm = None; self.wm = None; self.csf = None; self.rest = None
        self.atlas_cort_data = None; self.atlas_cort_labels = []
        self.atlas_sub_data = None; self.atlas_sub_labels = []
        
        # Masques Géométriques
        self.mask_cc = None       # Corps Calleux
        self.mask_cereb = None    # Cervelet
        
        if HAS_NILEARN:
            try: 
                data = load_nilearn_data()
                if data:
                    self.gm = data['gm']
                    self.wm = data['wm']
                    self.csf = data['csf']
                    self.rest = data['rest']
                    self.atlas_cort_data = data['atlas_cort_data']
                    self.atlas_cort_labels = data['atlas_cort_labels']
                    self.atlas_sub_data = data['atlas_sub_data']
                    self.atlas_sub_labels = data['atlas_sub_labels']
                    
                    # --- CONSTRUCTION DES MASQUES GÉOMÉTRIQUES ---
                    self._build_geometric_masks()
                    
                    self.ready = True
            except Exception as e:
                print(f"Erreur init nilearn: {e}")
                pass
        self._generate_deep_veins_base()

    def _build_geometric_masks(self):
        """Définit les zones anatomiques par coordonnées (MNI 1mm approx)."""
        # 1. CORPS CALLEUX (Forme Arquée)
        # On définit une zone centrale pour X, et on fait varier Z selon Y
        self.mask_cc = np.zeros_like(self.wm, dtype=bool)
        
        # Limites X (Gauche-Droite) : bande centrale étroite
        x_min, x_max = 88, 108 
        
        # A. Corps Central (Haut)
        # Y: 80 à 140 (Milieu), Z: 90 à 115 (Haut)
        self.mask_cc[x_min:x_max, 80:140, 90:115] = True
        
        # B. Genou (Avant - Plongeant)
        # Y: 140 à 165 (Avant), Z: 75 à 110 (Descend plus bas)
        self.mask_cc[x_min:x_max, 140:165, 75:110] = True
        
        # C. Splenium (Arrière - Plongeant)
        # Y: 50 à 80 (Arrière), Z: 75 à 110 (Descend plus bas)
        self.mask_cc[x_min:x_max, 50:80, 75:110] = True
        
        # Filtre : Doit être de la matière blanche
        self.mask_cc = self.mask_cc & (self.wm > 0.45)

        # 2. CERVELET (Fosse Postérieure)
        self.mask_cereb = np.zeros_like(self.gm, dtype=bool)
        
        # Boîte englobante large en bas à l'arrière
        # X: Large (20 à 175)
        # Y: Très Arrière (0 à 68) - Note: Le lobe temporal est plus en avant (Y > 70)
        # Z: Très Bas (0 à 55) - Sous le lobe occipital
        self.mask_cereb[20:175, 0:68, 0:55] = True
        
        # Exclusion du tronc cérébral central (approximatif) pour éviter de taguer le pont comme cervelet
        # On retire une petite boîte au milieu tout en bas
        self.mask_cereb[85:112, 30:70, 0:40] = False
        
        # Filtre : Doit être du tissu (Gris ou Blanc), pas de l'os ou de l'air
        self.mask_cereb = self.mask_cereb & ((self.gm + self.wm) > 0.2)

    def _generate_deep_veins_base(self):
        shape = (200, 200)
        np.random.seed(888)
        noise = np.random.normal(0, 1, shape)
        smooth = gaussian_filter(noise, sigma=2.0)
        sx = sobel(smooth, axis=0); sy = sobel(smooth, axis=1)
        magnitude = np.hypot(sx, sy)
        threshold = np.percentile(magnitude, 96) 
        mask = (magnitude > threshold).astype(float)
        self.static_vein_pattern = gaussian_filter(mask, sigma=0.5)

    def get_anatomical_veins(self, g, c, w, z_index_ratio):
        sulci_map = c * g 
        sulci_map = gaussian_filter(sulci_map, sigma=0.5)
        np.random.seed(int(z_index_ratio * 1000) + 42) 
        noise = np.random.normal(0, 1, g.shape)
        worms = gaussian_filter(noise, sigma=1.2)
        worms_edges = np.hypot(sobel(worms, 0), sobel(worms, 1))
        worms_mask = (worms_edges > np.percentile(worms_edges, 92)).astype(float)
        cortical_veins = worms_mask * sulci_map * 4.0 
        deep_veins_raw = self.static_vein_pattern
        zoom_factor = (g.shape[0] / deep_veins_raw.shape[0], g.shape[1] / deep_veins_raw.shape[1])
        deep_veins_res = zoom(deep_veins_raw, zoom_factor, order=1)
        center_mask = gaussian_filter(w, sigma=8)
        deep_veins = deep_veins_res * center_mask
        return np.clip(cortical_veins + deep_veins, 0, 1)

    # --- MÉTHODES UTILITAIRES ---
    def get_adc_map(self, s_gm, s_wm, s_csf): 
        return (s_csf * 3.0) + (s_gm * 0.8) + (s_wm * 0.7)
    def get_t2s_map(self, s_gm, s_wm, s_csf): 
        return (s_csf * 400.0) + (s_gm * 50.0) + (s_wm * 40.0)
    def get_t1_map(self, s_gm, s_wm, s_csf): 
        return (s_csf * 3500.0) + (s_gm * 1200.0) + (s_wm * 700.0)

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
            image[mask_center] = 0.5 + (0.4 * contrast_sign)
            image[mask_halo] = 0.5 - (0.1 * contrast_sign)
        else: 
            width = r * 3.0; height = r * 0.5
            mask_rect = (np.abs(x - cx) < width) & (np.abs(y - cy) < height)
            offset_y = r * 2.0; radius_x = r * 1.5; radius_y = r * 1.0
            mask_lobe_up = (((x - cx)**2) / radius_x**2 + ((y - (cy - offset_y))**2) / radius_y**2) <= 1
            mask_lobe_down = (((x - cx)**2) / radius_x**2 + ((y - (cy + offset_y))**2) / radius_y**2) <= 1
            image[mask_rect] = 0.5 + (0.4 * contrast_sign)
            image[mask_lobe_up] = 0.5 - (0.35 * contrast_sign)
            image[mask_lobe_down] = 0.5 - (0.35 * contrast_sign)
        return np.clip(image, 0, 1)

    def get_phase_map(self, s_gm, s_wm, s_csf, z_ratio, contrast_sign, with_bleeds=False, inject_dipole=False, axis='z'):
        vein_mask = self.get_anatomical_veins(s_gm, s_csf, s_wm, z_ratio)
        raw_susc_flat = (s_gm * 0.51) + (s_wm * 0.50) + (s_csf * 0.50)
        low_freq = gaussian_filter(raw_susc_flat, sigma=3.0)
        high_pass = raw_susc_flat - low_freq
        anat_structure = (s_gm * 0.85) + (s_wm * 0.65) + (s_csf * 0.25)
        anat_modulation = (anat_structure - 0.5) 
        phase_texture = 0.5 + (high_pass * 8.0 * contrast_sign) + (anat_modulation * 0.20 * contrast_sign)
        np.random.seed(42); noise_grain = np.random.normal(0, 0.015, s_gm.shape) 
        phase_texture += noise_grain
        brain_tissue = (s_gm + s_wm + s_csf) > 0.1
        brain_mask_strict = binary_erosion(brain_tissue, iterations=12)
        dilated_outer = binary_dilation(brain_tissue, iterations=3)
        skull_mask = dilated_outer ^ brain_mask_strict
        noise_air = np.random.normal(0.5, 0.035, s_gm.shape) 
        noise_skull = np.random.normal(0.35, 0.15, s_gm.shape)
        phase_final = noise_air
        phase_final = np.where(skull_mask, noise_skull, phase_final)
        phase_final = np.where(brain_mask_strict, phase_texture, phase_final)
        vein_intensity = 0.35 * contrast_sign
        phase_final = np.where(brain_mask_strict, phase_final - (vein_mask * vein_intensity), phase_final)
        if with_bleeds:
            np.random.seed(100); bleeds = np.random.choice([0, 1], size=s_gm.shape, p=[0.998, 0.002])
            bleeds = gaussian_filter(bleeds.astype(float), sigma=0.5)
            phase_final = np.where(brain_mask_strict, phase_final + (bleeds * 0.6 * contrast_sign), phase_final)
        if inject_dipole: 
            phase_final = self.inject_dipole_artifact(phase_final, axis, contrast_sign)
        return np.clip(phase_final, 0, 1)

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
        gm, wm, csf = self.gm, self.wm, self.csf
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
        T_BLOOD = 1650.0 
        decay = np.exp(-pld / T_BLOOD); arrival = np.clip((pld - att_map) / 500.0, 0, 1)
        asl_signal_visible = cbf * decay * arrival * 0.015 
        noise = np.random.normal(0, 0.01, g.shape)
        control_img = (g * 0.8) + (w * 0.6) + (c * 0.2) + noise
        label_img = control_img - asl_signal_visible
        perf_map_calc = control_img - label_img
        return control_img, label_img, perf_map_calc

    def get_slice(self, axis, index, w_vals, seq_type=None, te=0, tr=500, fa=90, b_val=0, adc_mode=False, with_stroke=False, swi_mode=None, with_bleeds=False, swi_sys="RHS", swi_sub="Hématome", with_dipole=False):
        if not self.ready: return None
        gm, wm, csf, rest = self.gm, self.wm, self.csf, self.rest
        ax_idx = {'x': 0, 'y': 1, 'z': 2}.get(axis, 2)
        idx_safe = max(0, min(index, gm.shape[ax_idx]-1))
        g = np.take(gm, idx_safe, axis=ax_idx).T; w = np.take(wm, idx_safe, axis=ax_idx).T; c = np.take(csf, idx_safe, axis=ax_idx).T; r = np.take(rest, idx_safe, axis=ax_idx).T
        if axis == 'z': g = np.flipud(g); w = np.flipud(w); c = np.flipud(c); r = np.flipud(r)
        elif axis == 'x': g = np.flipud(np.fliplr(g)); w = np.flipud(np.fliplr(w)); c = np.flipud(np.fliplr(c)); r = np.flipud(np.fliplr(r))
        elif axis == 'y': g = np.flipud(g); w = np.flipud(w); c = np.flipud(c); r = np.flipud(r)
        if st.session_state.get('atrophy_active', False) and axis == 'z': g, w, c, _ = self.apply_focal_atrophy(g, w, c)
        stroke_mask = None
        if with_stroke and axis == 'z' and (85 < index < 95): stroke_mask = self.create_lesion_mask(g.shape, center=(110, 70), radius=9)
        if swi_mode:
            is_rhs = "RHS" in swi_sys; is_para = "Hématome" in swi_sub
            contrast_sign = (1 if is_para else -1) * (1 if is_rhs else -1) 
            z_ratio = index / max(1, gm.shape[ax_idx])
            if swi_mode == 'phase': sim = self.get_phase_map(g, w, c, z_ratio, contrast_sign, with_bleeds, inject_dipole=with_dipole, axis=axis)
            elif swi_mode == 'mag':
                vein_mask = self.get_anatomical_veins(g, c, w, z_ratio); t2s = self.get_t2s_map(g, w, c)
                base_sig = (c * 0.25) + (g * 0.85) + (w * 0.65) + np.random.normal(0, 0.03, g.shape)
                decay = np.exp(-te / (t2s + 1e-6)); mag = base_sig * decay * 1.8 * (1 - vein_mask * 0.4) 
                if with_bleeds: 
                    bleeds = gaussian_filter(np.random.choice([0, 1], size=g.shape, p=[0.999, 0.001]).astype(float), sigma=0.8)
                    mag = mag * (1 - bleeds * 10)
                sim = mag
            elif swi_mode == 'minip': 
                vein_mask = self.get_anatomical_veins(g, c, w, z_ratio); vein_mask_thick = gaussian_filter(vein_mask, sigma=0.8)
                sim = (c * 0.9 + g*0.8 + w*0.7) * (1 - vein_mask_thick * 1.5) 
        elif adc_mode: 
            raw_adc = self.get_adc_map(g, w, c)
            if stroke_mask is not None: raw_adc[stroke_mask] = 0.4
            sim = (raw_adc + np.random.normal(0, 0.1, raw_adc.shape)) / 3.2 
        elif seq_type == 'dwi':
            adc_map = self.get_adc_map(g, w, c); s0_map = (c * 3.0) + (g * 1.5) + (w * 1.0) + (r * 0.1)
            if stroke_mask is not None: adc_map[stroke_mask] = 0.4; s0_map[stroke_mask] = 3.0
            sim = s0_map * np.exp(-b_val * adc_map * 0.001) / 3.0
        elif seq_type == 'gre':
             t1_map = self.get_t1_map(g, w, c); t2s_map = self.get_t2s_map(g, w, c)
             pd_map = (c * 1.0) + (g * 0.9) + (w * 0.7) + (r * 0.2); rad = np.radians(fa)
             e1 = np.exp(-tr / (t1_map + 1e-6)); sim = pd_map * ((np.sin(rad) * (1 - e1)) / (1 - np.cos(rad) * e1 + 1e-6)) * np.exp(-te / (t2s_map + 1e-6)) * 5.0
        else:
            if stroke_mask is not None and ('T2' in st.session_state.get('seq', '') or 'FLAIR' in st.session_state.get('seq', '')): w[stroke_mask] = 0; c[stroke_mask] = 1.0
            sim = (c * w_vals['csf']) + (g * w_vals['gm']) + (w * w_vals['wm']) + (r * (w_vals['fat']*0.2))
        return sim 
    
    def get_dims(self): return (197, 233, 189) if self.ready else (100, 100, 100)

    # --- IDENTIFICATION INTELLIGENTE (MULTI-ATLAS + GEOMETRIE) ---
    def get_anatomical_labels(self, axis, index):
        if not self.ready: return np.full((100, 100), "Chargement...")
        atlas_cort = self.atlas_cort_data; atlas_sub = self.atlas_sub_data
        mask_cc_vol = self.mask_cc; mask_cereb_vol = self.mask_cereb
        
        ax_idx = {'x': 0, 'y': 1, 'z': 2}.get(axis, 2)
        idx_safe = max(0, min(index, atlas_cort.shape[ax_idx]-1))
        
        sl_cort = np.take(atlas_cort, idx_safe, axis=ax_idx).T
        sl_sub = np.take(atlas_sub, idx_safe, axis=ax_idx).T
        sl_cc = np.take(mask_cc_vol, idx_safe, axis=ax_idx).T
        sl_cereb = np.take(mask_cereb_vol, idx_safe, axis=ax_idx).T

        if axis == 'z': sl_cort=np.flipud(sl_cort); sl_sub=np.flipud(sl_sub); sl_cc=np.flipud(sl_cc); sl_cereb=np.flipud(sl_cereb)
        elif axis == 'x': sl_cort=np.flipud(np.fliplr(sl_cort)); sl_sub=np.flipud(np.fliplr(sl_sub)); sl_cc=np.flipud(np.fliplr(sl_cc)); sl_cereb=np.flipud(np.fliplr(sl_cereb))
        elif axis == 'y': sl_cort=np.flipud(sl_cort); sl_sub=np.flipud(sl_sub); sl_cc=np.flipud(sl_cc); sl_cereb=np.flipud(sl_cereb)
            
        ids_cort = np.round(sl_cort).astype(int)
        ids_sub = np.round(sl_sub).astype(int)
        generic_sub_ids = [0, 1, 2, 12, 13] 
        
        def lookup_label(id_cort, id_sub, is_cc, is_cereb):
            # 1. GEOMETRIE FORTE
            if is_cc: return "Corps Calleux"
            if is_cereb: return "Cervelet (Hémisphère)"

            # 2. ATLAS SOUS-CORTICAL
            if id_sub not in generic_sub_ids and id_sub < len(self.atlas_sub_labels):
                lbl_sub = self.atlas_sub_labels[id_sub]
                if "Brain-Stem" in lbl_sub: return "Tronc Cérébral (Mésencéphale/Pont)"
                if "Thalamus" in lbl_sub: return "Thalamus"
                if "Caudate" in lbl_sub: return "Noyau Caudé"
                if "Putamen" in lbl_sub: return "Putamen"
                if "Pallidum" in lbl_sub: return "Pallidum"
                if "Hippocampus" in lbl_sub: return "Hippocampe"
                if "Amygdala" in lbl_sub: return "Amygdale"
                if "Accumbens" in lbl_sub: return "Noyau Accumbens"
                return lbl_sub 
            
            # 3. ATLAS CORTICAL
            if id_cort > 0 and id_cort < len(self.atlas_cort_labels): return self.atlas_cort_labels[id_cort] 
            
            # 4. FALLBACK
            if id_sub in [1, 12]: return "Matière Blanche Cérébrale"
            if id_sub in [2, 13]: return "Cortex (Non spécifié)"
            return "Liquide / Os / Air"

        vfunc = np.vectorize(lookup_label)
        return vfunc(ids_cort, ids_sub, sl_cc, sl_cereb)