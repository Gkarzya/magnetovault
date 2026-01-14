import plotly.graph_objects as go
import numpy as np

class MRIVisualizer:
    def create_figure(self, image, atlas_slice, atlas_labels, title="Vue IRM", show_legends=True):
        if image is None: return go.Figure()
        
        # Normalisation
        img_min, img_max = np.min(image), np.max(image)
        if img_max - img_min > 1e-6: img_disp = (image - img_min) / (img_max - img_min)
        else: img_disp = image

        fig = go.Figure()
        
        # Image (Visible)
        fig.add_trace(go.Heatmap(z=img_disp, colorscale='Gray', showscale=False, hoverinfo='skip'))

        # Légendes (Invisible mais interactive au survol)
        if show_legends and atlas_slice is not None and atlas_labels is not None:
            # On s'assure de ne pas dépasser les indices
            safe_slice = np.clip(atlas_slice, 0, len(atlas_labels)-1).astype(int)
            txt = np.array(atlas_labels)[safe_slice]
            
            # On rend transparent le fond (valeur 0)
            atl_nan = atlas_slice.astype(float)
            atl_nan[atlas_slice == 0] = np.nan
            
            fig.add_trace(go.Heatmap(
                z=atl_nan, colorscale='Jet', showscale=False, opacity=0.0,
                customdata=txt, hovertemplate="<b>%{customdata}</b><extra></extra>"
            ))

        fig.update_layout(
            title={'text': title, 'y': 0.95, 'x': 0.5, 'xanchor': 'center'},
            width=600, height=600, margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='black',
            xaxis=dict(visible=False),
            # autorange='reversed' pour corriger le sens dans Plotly
            yaxis=dict(visible=False, scaleanchor="x", scaleratio=1, autorange='reversed')
        )
        return fig