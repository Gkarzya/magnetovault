# physique.py
import numpy as np

def calculate_signal(tr_val, te_val, ti_val, t1, t2, t2s, adc, pd, fa_deg, gre_mode, dwi_mode, b_val):
    """Calcule l'intensité du signal IRM selon les paramètres (Bloch Equations)."""
    val = 0.0
    # Mode Gradient Echo
    if gre_mode:
        rad = np.radians(fa_deg)
        e1 = np.exp(-tr_val / t1)
        if fa_deg < 40 and t1 > 2000: e1 = e1 * 0.1 
        e2s = np.exp(-te_val / t2s)
        num = np.sin(rad) * (1 - e1)
        den = 1 - np.cos(rad) * e1
        val = pd * (num / den) * e2s
    # Mode Spin Echo
    else:
        e2 = np.exp(-te_val / t2)
        if ti_val > 10: # Inversion Récupération
            val = pd * (1 - 2 * np.exp(-ti_val / t1) + np.exp(-tr_val / t1)) * e2
        else:
            val = pd * (1 - np.exp(-tr_val / t1)) * e2
            
    # Facteur Diffusion
    if dwi_mode and b_val > 0:
        diff_decay = np.exp(-b_val * adc * 0.001)
        val = val * diff_decay
        
    return np.abs(val)

def calculate_snr_relative(mat, nex, turbo, ipat, bw, fov, ep, v_wm, ref_wm):
    """
    Estime le SNR relatif.
    CORRECTION V7.42 : Cette fonction est indépendante du nombre de coupes (n_slices).
    Le SNR dépend du Volume Voxel, NEX, BW, Matrice et facteur d'antenne.
    """
    if ref_wm < 0.0001: ref_wm = 0.0001
    
    # 1. Facteur Volume Voxel (Taille du pixel * épaisseur)
    vol_factor = (fov/float(mat))**2 * ep
    
    # 2. Facteur Acquisition (Temps de lecture / moyennage)
    turbo_penalty = float(turbo)**0.25 
    ipat_penalty = (float(ipat)**0.25) * (1.2 + (0.1 * (ipat - 2))) if ipat > 1 else 1.0
    acq_factor = np.sqrt(float(mat)*float(nex)) / (turbo_penalty * ipat_penalty)
    
    # 3. Facteur Bande Passante (Plus BW est grand, plus le bruit augmente)
    bw_factor = np.sqrt(220.0 / float(bw))
    
    # Normalisation
    r_vol = vol_factor / ((240.0/256.0)**2 * 5.0)
    r_acq = acq_factor / np.sqrt(256.0)
    r_sig = v_wm / ref_wm
    
    return r_vol * r_acq * bw_factor * r_sig * 100.0