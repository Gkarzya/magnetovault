# constantes.py

# TISSUS BIOLOGIQUES (T1, T2 en ms)
T_FAT = {'T1': 260.0, 'T2': 80.0, 'T2s': 30.0, 'ADC': 0.0, 'PD': 1.0, 'Label': 'Graisse'}
T_LCR = {'T1': 3607.0, 'T2': 2000.0, 'T2s': 400.0, 'ADC': 3.0, 'PD': 1.0, 'Label': 'Eau (LCR)'}
T_GM  = {'T1': 1300.0, 'T2': 140.0, 'T2s': 50.0, 'ADC': 0.8, 'PD': 0.95, 'Label': 'Subst. Grise'}
T_WM  = {'T1': 600.0,  'T2': 100.0, 'T2s': 40.0, 'ADC': 0.7, 'PD': 0.70, 'Label': 'Subst. Blanche'}
T_STROKE = {'T1': 900.0, 'T2': 200.0, 'T2s': 80.0, 'ADC': 0.4, 'PD': 0.90, 'Label': 'Ischémie (AVC)'}
T_BLOOD = 1650.0 

# LISTE DES SÉQUENCES
OPTIONS_SEQ = [
    "Pondération T1", "Pondération T2", "Écho de Gradient (T2*)", "Diffusion (DWI)", 
    "DP (Densité Protons)", "FLAIR (Eau -)", "Séquence STIR (Graisse)", "SWI (Susceptibilité)",
    "3D T1 (MP-RAGE)", "Perfusion ASL"
]

# PARAMÈTRES PAR DÉFAUT (Correctifs V7.42 appliqués : T2 TR=4000, TE=85)
STD_PARAMS = {
    "Pondération T1": {'tr': 500.0, 'te': 10.0, 'ti': 0.0},
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