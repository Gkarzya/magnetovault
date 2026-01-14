# physique.py - VERSION 7.60 (CORRIGÉE : SNR & PROPORTIONNALITÉ CONCATS)
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

def calculate_acquisition_time(tr, mat, nex, turbo, ipat, concats, n_slices, is_3d):
    """
    Calcule le temps d'acquisition en ms.
    
    CORRECTION V7.60 :
    En 2D, si on utilise des concaténations, on divise les coupes en plusieurs paquets.
    Le TR diminue (car moins de coupes par paquet), MAIS on doit répéter la séquence 
    pour chaque paquet.
    Donc : Temps Total = Temps d'un paquet * Nombre de Concaténations.
    """
    
    # Sécurité pour éviter multiplication par 0 ou erreur
    if concats < 1: concats = 1
    if turbo < 1: turbo = 1
    if ipat < 1: ipat = 1

    # Base du calcul pour UNE passe (un paquet de coupes)
    # Formule : (TR * Lignes de Phase * NEX) / (Facteurs d'accélération)
    base_time = (tr * mat * nex) / (turbo * ipat)
    
    if is_3d:
        # En 3D, le nombre de coupes agit comme une seconde dimension de phase (Partition Encoding)
        return base_time * n_slices
    else:
        # En 2D :
        # Le temps dépend du nombre de "passes" nécessaires (concaténations).
        # Si concats = 1 : On fait tout en une fois.
        # Si concats = 2 : On fait 2 fois la séquence (avec un TR plus court géré dans main.py).
        return base_time * concats

def calculate_snr_relative(mat, nex, turbo, ipat, bw, fov, ep, v_wm, ref_wm):
    """
    Estime le SNR relatif.
    Indépendant du nombre de coupes et des concaténations.
    """
    if ref_wm < 0.0001: ref_wm = 0.0001
    
    # 1. Facteur Volume Voxel (Taille du pixel * épaisseur)
    vol_factor = (fov/float(mat))**2 * ep
    
    # 2. Facteur Acquisition (Temps de lecture / moyennage)
    turbo_penalty = float(turbo)**0.25 
    ipat_penalty = (float(ipat)**0.25) * (1.2 + (0.1 * (ipat - 2))) if ipat > 1 else 1.0
    acq_factor = np.sqrt(float(mat)*float(nex)) / (turbo_penalty * ipat_penalty)
    
    # 3. Facteur Bande Passante
    bw_factor = np.sqrt(220.0 / float(bw))
    
    # Normalisation
    r_vol = vol_factor / ((240.0/256.0)**2 * 5.0)
    r_acq = acq_factor / np.sqrt(256.0)
    r_sig = v_wm / ref_wm
    
    return r_vol * r_acq * bw_factor * r_sig * 100.0