import os
import ssl

# Bypass SSL pour éviter les erreurs de certificat
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("--- DÉBUT DU TÉLÉCHARGEMENT DES ATLAS (CORRECTIF) ---")

try:
    from nilearn import datasets
    
    data_dir = os.path.join(os.path.expanduser('~'), 'nilearn_data')
    print(f"Destination : {data_dir}")

    # 1. MNI
    print("\n1. Vérification MNI 152...")
    mni = datasets.fetch_icbm152_2009()
    print("✅ MNI OK")

    # 2. Harvard-Oxford
    print("\n2. Vérification Harvard-Oxford (Cortical)...")
    ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    print("✅ HO-Cortical OK")

    print("\n3. Vérification Harvard-Oxford (Sous-Cortical)...")
    ho_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
    print("✅ HO-Sub OK")

    # 3. JHU (CORRECTION DU NOM DE LA FONCTION)
    print("\n4. Téléchargement Atlas JHU (White Matter)...")
    # C'est ici que ça change : _white_matter
    jhu = datasets.fetch_atlas_jhu_white_matter('ICBM-DTI-81')
    print("✅ JHU OK")

    print("\n--- TOUT EST TÉLÉCHARGÉ AVEC SUCCÈS ! ---")

except Exception as e:
    print(f"\n❌ ERREUR : {e}")