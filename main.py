from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import io
import base64
import webbrowser
import json


# -----------------------------
# Bild laden & vorbereiten
# -----------------------------
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    #image.show()
    image = image.rotate(-90, expand=True)
    array = np.array(image, dtype=np.uint8)
    return array


# -----------------------------
# Glättung mit Padding (stabil!)
# -----------------------------
def smooth(y, k):
    kernel = np.ones(k) / k
    y_padded = np.pad(y, (k, k), mode='edge')
    y_smooth = np.convolve(y_padded, kernel, mode='same')
    return y_smooth[k:-k]


# -----------------------------
# Kern: Breite bestimmen (vektorisiert)
# -----------------------------
def compute_width(array):
    # Gradient (ohne Overflow!)
    grad = array[:, 1:].astype(np.int16) - array[:, :-1].astype(np.int16)

    # stärkste positive & negative Kante
    high_gradient_at = np.argmax(grad, axis=1)
    low_gradient_at = np.argmin(grad, axis=1)

    # Breite
    width = np.abs(low_gradient_at - high_gradient_at)

    return width, high_gradient_at, low_gradient_at


# -----------------------------
# Visualisierung
# -----------------------------
def build_visualization(array, high_gradient_at, low_gradient_at):
    height, width = array.shape

    vis = np.zeros((height, width))
    vis = array.copy()

    left = np.minimum(high_gradient_at, low_gradient_at)
    right = np.maximum(high_gradient_at, low_gradient_at)

    for i in range(height):
        vis[i, left[i]:right[i]] = 200

    return vis


def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def annotate_analysis_image(image_path, high_gradient_at, low_gradient_at, width_smooth, min_idx, max_idx):
    image = Image.open(image_path).convert('RGB')
    image = image.rotate(-90, expand=True)
    draw = ImageDraw.Draw(image)

    left_min = int(np.minimum(high_gradient_at[min_idx], low_gradient_at[min_idx]))
    right_min = int(np.maximum(high_gradient_at[min_idx], low_gradient_at[min_idx]))
    left_max = int(np.minimum(high_gradient_at[max_idx], low_gradient_at[max_idx]))
    right_max = int(np.maximum(high_gradient_at[max_idx], low_gradient_at[max_idx]))

    h = image.height
    line_y_min = int(min_idx * image.height / len(width_smooth))
    line_y_max = int(max_idx * image.height / len(width_smooth))

    draw.rectangle([left_min, line_y_min - 2, right_min, line_y_min + 2], fill=(0, 255, 0))
    draw.rectangle([left_max, line_y_max - 2, right_max, line_y_max + 2], fill=(255, 0, 0))
    draw.line([(0, line_y_min), (image.width, line_y_min)], fill=(0, 255, 0), width=1)
    draw.line([(0, line_y_max), (image.width, line_y_max)], fill=(255, 0, 0), width=1)

    font = None
    try:
        font = ImageFont.truetype('arial.ttf', size=16)
    except Exception:
        font = ImageFont.load_default()

    draw.text((10, 10), f'Min width row: {min_idx}', fill=(0, 255, 0), font=font)
    draw.text((10, 30), f'Max width row: {max_idx}', fill=(255, 0, 0), font=font)
    draw.text((10, 50), f'Min width value: {width_smooth[min_idx]:.2f}', fill=(255, 255, 255), font=font)
    draw.text((10, 70), f'Max width value: {width_smooth[max_idx]:.2f}', fill=(255, 255, 255), font=font)

    return image_to_base64(image)


def plot_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def plot_to_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def create_dashboard_html(results, width_plot_html, position_plot_html, high_edge_plot_html, low_edge_plot_html, score_plot_html, best_result, best_interp_pressure, best_interp_score, output_path):
    list_items = ''
    image_details = ''
    for idx, result in enumerate(results):
        safe_id = result['name'].replace('.', '_').replace(' ', '_').replace('/', '_').replace('\\', '_')
        list_items += f"<button class='image-select' data-safe-id='{safe_id}' onclick=\"showImageDetail('{safe_id}')\">{result['name']} — {result['pressure']:.3f}</button>"
        image_details += f"""
        <div id='image-detail-{safe_id}' class='image-detail{' active' if idx == 0 else ''}'>
            <div class='image-info'>
                <div class='image-card'>
                    <h4>Image: {result['name']}</h4>
                    <div class='metrics'>
                        <div><strong>Pressure:</strong> {result['pressure']:.3f}</div>
                        <div><strong>Mean width:</strong> {result['width_mean']:.2f}</div>
                        <div><strong>Min width:</strong> {result['width_min']:.2f} @ {result['width_min_pos']}</div>
                        <div><strong>Max width:</strong> {result['width_max']:.2f} @ {result['width_max_pos']}</div>
                        <div><strong>Width diff:</strong> {result['width_diff']:.2f}</div>
                        <div><strong>Width std:</strong> {result['width_std']:.2f}</div>
                        <div><strong>High edge mean:</strong> {result['high_at_mean']:.2f}</div>
                        <div><strong>High edge std:</strong> {result['high_at_std']:.2f}</div>
                        <div><strong>Low edge mean:</strong> {result['low_at_mean']:.2f}</div>
                        <div><strong>Low edge std:</strong> {result['low_at_std']:.2f}</div>
                        <div><strong>Quality score:</strong> {result['score']:.3f}</div>
                    </div>
                </div>
            </div>
            <div class='image-preview'>
                <img class='preview-image' src='data:image/png;base64,{result['annotated_image']}' alt='annotated preview' />
            </div>
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang='de'>
    <head>
    <meta charset='UTF-8'>
    <title>Analysis Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f4f7fb; color: #222; }}
        .header {{ padding: 20px; background: #1f3b6f; color: white; }}
        .header h1 {{ margin: 0; font-size: 24px; }}
        .tabs {{ display: flex; background: #ffffff; border-bottom: 1px solid #ddd; }}
        .tab-button {{ padding: 14px 20px; cursor: pointer; border: none; border-bottom: 3px solid transparent; background: none; font-size: 16px; }}
        .tab-button.active {{ border-bottom-color: #1f3b6f; color: #1f3b6f; font-weight: bold; }}
        .tab-content {{ display: none; padding: 20px; }}
        .tab-content.active {{ display: block; }}
        .summary-card {{ background: white; padding: 18px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 18px; }}
        .image-card {{ background: white; padding: 14px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); max-width: 320px; margin: 0 auto; }}
        .image-card img {{ width: 100%; max-width: 280px; height: auto; border-radius: 8px; margin: 0 auto 10px; display: block; }}
        .metrics {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
        .metrics div {{ font-size: 14px; }}
        .image-list {{ display: grid; gap: 10px; max-width: 360px; margin-bottom: 18px; }}
        .image-select {{ text-align: left; padding: 12px 14px; border: 1px solid #ddd; border-radius: 10px; background: white; cursor: pointer; transition: background 0.2s, border-color 0.2s; }}
        .image-select:hover, .image-select.active {{ background: #e8f0ff; border-color: #1f3b6f; color: #1f3b6f; }}
        .image-detail {{ display: none; grid-template-columns: 1.1fr 0.9fr; gap: 20px; align-items: start; }}
        .image-detail.active {{ display: grid; }}
        .image-preview {{ background: white; padding: 14px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); text-align: center; }}
        .preview-image {{ width: 70%; max-width: 100%; border-radius: 8px; }}
        .slider-row {{ display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }}
        .slider-row label {{ font-weight: 600; }}
        .slider-row input[type='range'] {{ width: 100%; }}
        .image-details {{ display: grid; gap: 18px; }}
        .image-display {{ display: grid; gap: 18px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 12px; border: 1px solid #ddd; text-align: left; }}
        th {{ background: #f0f4f8; }}
    </style>
    </head>
    <body>
        <div class='header'>
            <h1>Analyse-Dashboard</h1>
            <p>Übersicht über alle analysierten Bilder mit erkannter Breite, Min/Max und Positionen.</p>
        </div>
        <div class='tabs'>
            <button class='tab-button active' onclick="openTab(event, 'overview')">Übersicht</button>
            <button class='tab-button' onclick="openTab(event, 'plots')">Plots</button>
            <button class='tab-button' onclick="openTab(event, 'images')">Bilder</button>
        </div>

        <div id='overview' class='tab-content active'>
            <div class='summary-card'>
                <h2>Globale Statistik</h2>
                <table>
                    <tr><th>Metric</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Median</th></tr>
                    <tr><td>Width mean</td><td>{np.mean([r['width_mean'] for r in results]):.2f}</td><td>{np.std([r['width_mean'] for r in results]):.2f}</td><td>{np.min([r['width_mean'] for r in results]):.2f}</td><td>{np.max([r['width_mean'] for r in results]):.2f}</td><td>{np.median([r['width_mean'] for r in results]):.2f}</td></tr>
                    <tr><td>Width min</td><td>{np.mean([r['width_min'] for r in results]):.2f}</td><td>{np.std([r['width_min'] for r in results]):.2f}</td><td>{np.min([r['width_min'] for r in results]):.2f}</td><td>{np.max([r['width_min'] for r in results]):.2f}</td><td>{np.median([r['width_min'] for r in results]):.2f}</td></tr>
                    <tr><td>Width max</td><td>{np.mean([r['width_max'] for r in results]):.2f}</td><td>{np.std([r['width_max'] for r in results]):.2f}</td><td>{np.min([r['width_max'] for r in results]):.2f}</td><td>{np.max([r['width_max'] for r in results]):.2f}</td><td>{np.median([r['width_max'] for r in results]):.2f}</td></tr>
                    <tr><td>Width diff</td><td>{np.mean([r['width_diff'] for r in results]):.2f}</td><td>{np.std([r['width_diff'] for r in results]):.2f}</td><td>{np.min([r['width_diff'] for r in results]):.2f}</td><td>{np.max([r['width_diff'] for r in results]):.2f}</td><td>{np.median([r['width_diff'] for r in results]):.2f}</td></tr>
                </table>
            </div>
            <div class='summary-card'>
                <h2>Allgemeine Auswertung</h2>
                {width_plot_html}
            </div>
            <div class='summary-card'>
                <h2>Bestes Pressure Advance</h2>
                <div class='metrics'>
                    <div><strong>Pressure:</strong> {best_result['pressure']:.3f}</div>
                    <div><strong>Width mean:</strong> {best_result['width_mean']:.2f}</div>
                    <div><strong>Width std:</strong> {best_result['width_std']:.2f}</div>
                    <div><strong>Width diff:</strong> {best_result['width_diff']:.2f}</div>
                    <div><strong>High edge std:</strong> {best_result['high_at_std']:.2f}</div>
                    <div><strong>Low edge std:</strong> {best_result['low_at_std']:.2f}</div>
                    <div><strong>Quality score:</strong> {best_result['score']:.3f}</div>
                    <div><strong>Interpolated best pressure:</strong> {best_interp_pressure:.3f}</div>
                    <div><strong>Interpolated best score:</strong> {best_interp_score:.3f}</div>
                </div>
            </div>
        </div>

        <div id='plots' class='tab-content'>
            <div class='summary-card'>
                <h2>Edge and width positions</h2>
                {position_plot_html}
            </div>
            <div class='summary-card'>
                <h2>High edge mean</h2>
                {high_edge_plot_html}
            </div>
            <div class='summary-card'>
                <h2>Low edge mean</h2>
                {low_edge_plot_html}
            </div>
            <div class='summary-card'>
                <h2>Quality score</h2>
                {score_plot_html}
            </div>
        </div>

        <div id='images' class='tab-content'>
            <div class='summary-card'>
                <h2>Bildauswahl</h2>
                <div class='image-display'>
                    <div class='image-list'>
                        {list_items}
                    </div>
                    <div class='slider-row'>
                        <label for='image-size-slider'>Bildgröße:</label>
                        <input id='image-size-slider' type='range' min='20' max='100' value='70' oninput='resizeSelectedImage(this.value)' />
                        <span id='image-size-value'>70%</span>
                    </div>
                    <div class='image-details'>
                        {image_details}
                    </div>
                </div>
            </div>
        </div>

        <script>
            function openTab(evt, tabId) {{
                document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
                evt.currentTarget.classList.add('active');
                document.getElementById(tabId).classList.add('active');
            }}

            function showImageDetail(safeId) {{
                document.querySelectorAll('.image-select').forEach(btn => btn.classList.remove('active'));
                document.querySelectorAll('.image-detail').forEach(detail => detail.classList.remove('active'));
                const selectedButton = document.querySelector(`.image-select[data-safe-id='${{safeId}}']`);
                if (selectedButton) selectedButton.classList.add('active');
                const detail = document.getElementById(`image-detail-${{safeId}}`);
                if (detail) detail.classList.add('active');
                resizeSelectedImage(document.getElementById('image-size-slider').value);
            }}

            function resizeSelectedImage(value) {{
                const image = document.querySelector('.image-detail.active .preview-image');
                if (image) {{
                    image.style.width = `${{value}}%`;
                }}
                const label = document.getElementById('image-size-value');
                if (label) {{
                    label.textContent = `${{value}}%`;
                }}
            }}

            document.addEventListener('DOMContentLoaded', function() {{
                const firstButton = document.querySelector('.image-select');
                if (firstButton) {{
                    firstButton.classList.add('active');
                }}
                const firstDetail = document.querySelector('.image-detail');
                if (firstDetail) {{
                    firstDetail.classList.add('active');
                }}
                resizeSelectedImage(document.getElementById('image-size-slider').value);
            }});
        </script>
    </body>
    </html>
    """

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def open_html_report(output_path):
    webbrowser.open_new_tab('file://' + os.path.abspath(output_path))


# -----------------------------
# MAIN
# -----------------------------
def main():
    np.set_printoptions(threshold=np.inf, precision=2)
    folder_path = "./img/Testimg/img2/"
    smooth_factor = 30

    image_files = sorted(
        [img for img in os.listdir(folder_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda fn: float(fn.rsplit('.', 1)[0])
    )

    results = []
    for img in image_files:
        image_path = os.path.join(folder_path, img)
        preasure_advance = np.float64(img.rsplit('.', 1)[0])

        array = preprocess_image(image_path)
        width, high_gradient_at, low_gradient_at = compute_width(array)
        width_smooth = smooth(width, smooth_factor)

        min_idx = int(np.argmin(width_smooth))
        max_idx = int(np.argmax(width_smooth))
        annotated_image = annotate_analysis_image(image_path, high_gradient_at, low_gradient_at, width_smooth, min_idx, max_idx)

        width_std = float(width_smooth.std())
        high_at_std = float(high_gradient_at.std())
        low_at_std = float(low_gradient_at.std())
        score = float(width_std + 0.5 * (width_smooth.max() - width_smooth.min()) + 0.25 * high_at_std + 0.25 * low_at_std)

        result = {
            'name': img,
            'pressure': preasure_advance,
            'width_mean': float(width_smooth.mean()),
            'width_std': width_std,
            'width_max': float(width_smooth.max()),
            'width_min': float(width_smooth.min()),
            'width_diff': float(width_smooth.max() - width_smooth.min()),
            'width_max_pos': max_idx,
            'width_min_pos': min_idx,
            'high_at_mean': float(high_gradient_at.mean()),
            'high_at_std': high_at_std,
            'low_at_mean': float(low_gradient_at.mean()),
            'low_at_std': low_at_std,
            'score': score,
            'annotated_image': annotated_image,
        }
        results.append(result)

        print(
            f"preasure advance={preasure_advance:.3f}: "
            f"width_mean={result['width_mean']:.2f}, "
            f"width_std={result['width_std']:.2f}, "
            f"min={result['width_min']:.2f} @ {result['width_min_pos']}, "
            f"max={result['width_max']:.2f} @ {result['width_max_pos']}, "
            f"diff={result['width_diff']:.2f}, "
            f"high_at={result['high_at_mean']:.2f} ± {result['high_at_std']:.2f}, "
            f"low_at={result['low_at_mean']:.2f} ± {result['low_at_std']:.2f}, "
            f"score={result['score']:.3f}"
        )

    pressure_advances = np.array([r['pressure'] for r in results])
    width_means = np.array([r['width_mean'] for r in results])
    width_stds = np.array([r['width_std'] for r in results])
    width_maxs = np.array([r['width_max'] for r in results])
    width_mins = np.array([r['width_min'] for r in results])
    width_diffs = np.array([r['width_diff'] for r in results])
    width_max_positions = np.array([r['width_max_pos'] for r in results])
    width_min_positions = np.array([r['width_min_pos'] for r in results])
    high_at_means = np.array([r['high_at_mean'] for r in results])
    high_at_stds = np.array([r['high_at_std'] for r in results])
    low_at_means = np.array([r['low_at_mean'] for r in results])
    low_at_stds = np.array([r['low_at_std'] for r in results])
    scores = np.array([r['score'] for r in results])

    def describe_statistics(name, values):
        values = np.array(values, dtype=np.float64)
        print(
            f"{name}: mean={values.mean():.2f}, std={values.std():.2f}, "
            f"min={values.min():.2f}, max={values.max():.2f}, median={np.median(values):.2f}"
        )

    print("\n=== Statistik über alle Bilder ===")
    describe_statistics("Width mean", width_means)
    describe_statistics("Width min", width_mins)
    describe_statistics("Width max", width_maxs)
    describe_statistics("Width std", width_stds)
    describe_statistics("Width diff", width_diffs)
    describe_statistics("Width max position", width_max_positions)
    describe_statistics("Width min position", width_min_positions)
    describe_statistics("High edge mean position", high_at_means)
    describe_statistics("High edge std", high_at_stds)
    describe_statistics("Low edge mean position", low_at_means)
    describe_statistics("Low edge std", low_at_stds)
    describe_statistics("Score", scores)

    fig_width = go.Figure()
    fig_width.add_trace(go.Scatter(x=pressure_advances, y=width_means, mode='lines+markers', name='width mean', line=dict(color='#1f77b4')))
    fig_width.add_trace(go.Scatter(x=pressure_advances, y=width_maxs, mode='lines+markers', name='width max', line=dict(color='#ff7f0e')))
    fig_width.add_trace(go.Scatter(x=pressure_advances, y=width_mins, mode='lines+markers', name='width min', line=dict(color='#2ca02c')))
    fig_width.add_trace(go.Scatter(x=pressure_advances, y=width_diffs, mode='lines+markers', name='width diff', line=dict(color='#d62728', dash='dash')))
    fig_width.add_trace(go.Scatter(x=pressure_advances.tolist() + pressure_advances.tolist()[::-1], y=width_mins.tolist() + width_maxs.tolist()[::-1], fill='toself', fillcolor='rgba(199, 217, 235, 0.25)', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
    fig_width.update_layout(title='Width metrics across pressure advance', xaxis_title='Pressure Advance', yaxis_title='Width')
    width_plot_html = plot_to_html(fig_width)

    fig_positions = go.Figure()
    fig_positions.add_trace(go.Scatter(x=pressure_advances, y=high_at_means, mode='lines+markers', name='high edge mean', line=dict(color='#9467bd')))
    fig_positions.add_trace(go.Scatter(x=pressure_advances, y=low_at_means, mode='lines+markers', name='low edge mean', line=dict(color='#8c564b')))
    fig_positions.add_trace(go.Scatter(x=pressure_advances, y=width_max_positions, mode='lines+markers', name='max width position', line=dict(color='#e377c2', dash='dash')))
    fig_positions.add_trace(go.Scatter(x=pressure_advances, y=width_min_positions, mode='lines+markers', name='min width position', line=dict(color='#7f7f7f', dash='dash')))
    fig_positions.update_layout(title='Edge and width positions across pressure advance', xaxis_title='Pressure Advance', yaxis_title='Row index')
    position_plot_html = plot_to_html(fig_positions)

    fig_high_edge = go.Figure()
    fig_high_edge.add_trace(go.Scatter(x=pressure_advances, y=high_at_means, mode='lines+markers', line=dict(color='#9467bd')))
    fig_high_edge.update_layout(title='High edge mean across pressure advance', xaxis_title='Pressure Advance', yaxis_title='Mean row index')
    high_edge_plot_html = plot_to_html(fig_high_edge)

    fig_low_edge = go.Figure()
    fig_low_edge.add_trace(go.Scatter(x=pressure_advances, y=low_at_means, mode='lines+markers', line=dict(color='#8c564b')))
    fig_low_edge.update_layout(title='Low edge mean across pressure advance', xaxis_title='Pressure Advance', yaxis_title='Mean row index')
    low_edge_plot_html = plot_to_html(fig_low_edge)

    dense_count = 500
    pressure_dense = np.linspace(pressure_advances.min(), pressure_advances.max(), dense_count)
    poly_degree = min(4, len(pressure_advances) - 1)
    coeff = np.polyfit(pressure_advances, scores, poly_degree)
    poly = np.poly1d(coeff)
    score_dense = poly(pressure_dense)

    fig_score = go.Figure()
    fig_score.add_trace(go.Scatter(x=pressure_advances, y=scores, mode='lines+markers', name='Measured score', line=dict(color='#17becf')))
    fig_score.add_trace(go.Scatter(x=pressure_dense, y=score_dense, mode='lines', name='Interpolated score', line=dict(color='rgba(23,190,207,0.5)')))

    critical_roots = np.roots(poly.deriv())
    candidate_pressures = [pressure_dense[0], pressure_dense[-1]]
    for root in critical_roots:
        if np.isreal(root):
            real_root = float(np.real(root))
            if pressure_dense[0] <= real_root <= pressure_dense[-1]:
                candidate_pressures.append(real_root)

    candidate_pressures = np.array(candidate_pressures)
    candidate_scores = poly(candidate_pressures)
    best_interp_index = int(np.argmin(candidate_scores))
    best_interp_pressure = float(candidate_pressures[best_interp_index])
    best_interp_score = float(candidate_scores[best_interp_index])

    fig_score.add_trace(go.Scatter(
        x=[best_interp_pressure],
        y=[best_interp_score],
        mode='markers',
        marker=dict(color='#d62728', size=12, symbol='x'),
        name='Interpolated minimum'
    ))
    fig_score.update_layout(title='Quality score across pressure advance', xaxis_title='Pressure Advance', yaxis_title='Score')
    score_plot_html = plot_to_html(fig_score)

    best_index = int(np.argmin(scores))
    best_result = results[best_index]
    print(f"\nBeste pressure advance: {best_result['pressure']:.3f} mit Score {best_result['score']:.3f}")
    print(f"Interpolated bester pressure advance: {best_interp_pressure:.3f} mit Score {best_interp_score:.3f}")

    output_path = os.path.join(os.getcwd(), 'dashboard.html')
    create_dashboard_html(results, width_plot_html, position_plot_html, high_edge_plot_html, low_edge_plot_html, score_plot_html, best_result, best_interp_pressure, best_interp_score, output_path)
    print(f"Dashboard gespeichert unter: {output_path}")
    open_html_report(output_path)


if __name__ == "__main__":
    main()