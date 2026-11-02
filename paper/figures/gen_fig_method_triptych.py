"""Render saved diagnostics as bars, selected-field ECDFs, and empirical ROC.

Adapted from the existing v12 plot and the CCFA annotated-matrix/paired-view
recipes. Only saved results are used. No model inference or bootstrap is run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def empirical_roc(scores, labels):
    """Threshold every distinct saved score; group ties before accumulating."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    assert np.isfinite(scores).all() and set(labels) == {0, 1}
    order = np.argsort(-scores, kind='stable')
    scores, labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    tp = np.r_[0, np.cumsum(labels)[ends]]
    fp = np.r_[0, (1 + ends) - np.cumsum(labels)[ends]]
    tpr, fpr = tp / labels.sum(), fp / (len(labels) - labels.sum())
    auc = float(np.trapezoid(tpr, fpr))
    # An independent rank/pair definition also handles ties explicitly.
    pos, neg = scores[labels == 1], scores[labels == 0]
    pair_auc = float(((pos[:, None] > neg).sum() + .5 * (pos[:, None] == neg).sum()) / (len(pos) * len(neg)))
    assert abs(auc - pair_auc) < 1e-12
    return fpr, tpr, auc


def remove_collinear_roc(fpr, tpr, positives, negatives):
    """Remove only exactly collinear vertices in integer (FP, TP) space."""
    fp = np.rint(np.asarray(fpr) * negatives).astype(int)
    tp = np.rint(np.asarray(tpr) * positives).astype(int)
    keep = []
    for i in range(len(fp)):
        while len(keep) >= 2:
            a, b = keep[-2:]
            cross = (fp[b] - fp[a]) * (tp[i] - tp[b]) - (tp[b] - tp[a]) * (fp[i] - fp[b])
            if cross:
                break
            keep.pop()
        keep.append(i)
    x, y = fp[keep] / negatives, tp[keep] / positives
    assert abs(np.trapezoid(y, x) - np.trapezoid(tpr, fpr)) < 1e-12
    assert keep[0] == 0 and keep[-1] == len(fp) - 1
    return x, y


def gaussian_display_guide(values, sigma_fraction=.02):
    """Smooth uniformly sampled display coordinates, not measurement scores.

    A positive symmetric kernel preserves monotonicity. Odd reflection at both
    ends preserves endpoint values. All series use the same fixed fractional
    bandwidth, without tuning to labels, AUC, or relative model performance.
    """
    values = np.asarray(values, dtype=float)
    assert (np.diff(values) >= -1e-12).all()
    sigma = sigma_fraction * (len(values)-1)
    radius = int(np.ceil(4*sigma))
    assert radius < len(values)-1
    offsets = np.arange(-radius, radius+1)
    kernel = np.exp(-.5*(offsets/sigma)**2)
    kernel /= kernel.sum()
    padded = np.pad(values, radius, mode='reflect', reflect_type='odd')
    smooth = np.convolve(padded, kernel, mode='valid')
    assert np.allclose(smooth[[0,-1]], values[[0,-1]], atol=1e-12)
    smooth[0], smooth[-1] = values[0], values[-1]
    assert (np.diff(smooth) >= -1e-12).all()
    assert smooth.min() >= values[0]-1e-12 and smooth.max() <= values[-1]+1e-12
    return smooth


def roc_visual_guide(x, y):
    """Smooth the whole empirical path, not its upper hull; retain raw AUC."""
    x, y = np.asarray(x), np.asarray(y)
    t = x+y
    query = np.linspace(t[0], t[-1], 2401)
    sampled_x, sampled_y = np.interp(query,t,x), np.interp(query,t,y)
    gx, gy = gaussian_display_guide(sampled_x), gaussian_display_guide(sampled_y)
    assert (np.diff(gx) >= -1e-12).all() and (np.diff(gy) >= -1e-12).all()
    assert min(gx.min(), gy.min()) >= -1e-12 and max(gx.max(), gy.max()) <= 1+1e-12
    return gx, gy, {'parameter': 'FPR+TPR', 'sigma_fraction': .02, 'parameter_sigma': .04,
                    'endpoint_rule': 'odd reflection', 'max_fpr_displacement': float(np.max(abs(gx-sampled_x))),
                    'max_tpr_displacement': float(np.max(abs(gy-sampled_y))),
                    'fpr': gx.tolist(), 'tpr': gy.tolist(), 'drawing_area_not_auroc': float(np.trapezoid(gy, gx))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=HERE.parents[3])
    parser.add_argument('--output', type=Path,
                        default=HERE.parents[3] / 'output/pdf/agent_care_opd_tmm/method_v21')
    args = parser.parse_args()
    source = args.repo_root / 'runs/mechanism_eval/qzbc_care_opd_v1'
    paths = [source / 'localization/localization_analysis.csv',
             source / 'recovery_margin/recovery_margin_summary.csv',
             source / 'margin_behavior/margin_behavior_examples.csv',
             source / 'margin_behavior/margin_behavior_summary.csv']
    overall = next(r for r in read_csv(paths[0]) if r['scope'] == 'overall')
    models = ['clean_sft', 'gkd_full', 'care_opd']
    region_keys = ['full_margin_mean', 'localized_margin_mean', 'outside_margin_mean']
    margins = {r['model_label']: r for r in read_csv(paths[1])}
    values = np.array([[float(margins[m][k]) for k in region_keys] for m in models])
    counts = [int(overall[k]) for k in ['full_response_tokens', 'selected_tokens', 'outside_selected_tokens']]
    assert counts[0] == counts[1] + counts[2]
    assert abs(counts[1] / counts[0] - float(overall['token_coverage_micro'])) < 1e-12
    examples, summaries = read_csv(paths[2]), read_csv(paths[3])
    assert len(examples) == 200 and len({r['id'] for r in examples}) == 200
    assert all(r['clean_failed'] == 'True' for r in examples)
    eligible = [r for r in examples if r['care_opd_localized'] and r['care_opd_outside']]
    assert len(eligible) == 189
    labels = [int(r['care_rescued'] == 'True') for r in eligible]
    assert sum(labels) == 164 and len(labels) - sum(labels) == 25
    curves = {}
    for region in ['localized', 'outside']:
        key = 'care_opd_' + region
        fpr, tpr, auc = empirical_roc([r[key] for r in eligible], labels)
        saved = next(r for r in summaries if r['section'] == 'prediction' and r['score'] == key)
        assert int(saved['n']) == len(eligible) and abs(auc - float(saved['auroc'])) < 1e-12
        curves[region] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auroc': auc}
    diff = next(r for r in summaries if r['section'] == 'prediction_difference' and r['metric'] == 'auroc')
    assert abs(curves['localized']['auroc'] - curves['outside']['auroc'] - float(diff['difference'])) < 1e-12
    for i, model in enumerate(models):
        assert int(margins[model]['examples']) == 200 and int(margins[model]['localized_examples']) == 189
        for j, region in enumerate(['full', 'localized', 'outside']):
            observed = [float(r[f'{model}_{region}']) for r in examples if r[f'{model}_{region}']]
            assert len(observed) == (200 if j == 0 else 189)
            assert abs(np.mean(observed) - values[i, j]) < 1e-12

    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.6, 'ytick.labelsize': 7.6,
        'axes.linewidth': .65, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'svg.hashsalt': 'care-method-triptych-v21', 'text.color': '#1F2D40',
        'path.simplify': False, 'lines.antialiased': True,
        'lines.solid_joinstyle': 'round', 'lines.solid_capstyle': 'round',
        'lines.dash_joinstyle': 'round', 'lines.dash_capstyle': 'round',
        'axes.labelcolor': '#1F2D40', 'xtick.color': '#1F2D40', 'ytick.color': '#1F2D40',
    })
    ink, muted = '#1F2D40', '#596779'
    colors = ['#5B8FD9', '#F2A64A', '#4FB28F']
    fig = plt.figure(figsize=(7.2, 2.6), facecolor='white')
    # Compose equal physical plot rectangles; wider categorical labels have
    # dedicated gutters, not a different data-region width.
    panel_left = np.array([.64, 3.06, 5.48])
    panel_width, panel_bottom, panel_height = 1.64, .40, 1.41
    a, b, c = [fig.add_axes([left / 7.2, panel_bottom / 2.6,
                            panel_width / 7.2, panel_height / 2.6]) for left in panel_left]
    axes = [a, b, c]
    for i, (left, title) in enumerate(zip(panel_left,
                        ['Responsibility coverage', 'Selected-field preference', 'CARE-OPD rescue'])):
        fig.text((left - .47) / 7.2, 2.45 / 2.6, f'({chr(97+i)})', weight='bold', size=9,
                 va='center', ha='left')
        fig.text((left + panel_width / 2) / 7.2, 2.45 / 2.6, title,
                 weight='bold', size=8.8, va='center', ha='center')

    # (a): retain the original full/partition counts, not disjoint groups.
    count_labels, bars = [], []
    for y, count, color in zip([2, 1, 0], counts, colors):
        bar = a.barh(y, count / 1000, height=.68, color=color, edgecolor='none', zorder=3)[0]
        bars.append(bar)
        count_labels.append(a.text(count / 1000 - 8, y, f'{count:,}', ha='right', va='center', size=7.4, color='#172638'))
    a.set_xlim(0, 410)
    a.set_ylim(-.5, 2.5)
    a.set_xticks([0, 200, 400])
    a.set_yticks([2, 1, 0], ['Full\n100%', f'Selected\n{100 * counts[1] / counts[0]:.2f}%', f'Outside\n{100 * counts[2] / counts[0]:.2f}%'])
    a.set_xlabel('Response tokens (×10³)', labelpad=4)
    a.tick_params(axis='y', pad=4)
    ratio = r'$N_{\rm full}\approx ' + f'{counts[0]/counts[1]:.3f}' + r'N_{\rm selected}$'
    fig.text((panel_left[0] + panel_width/2) / 7.2, 2.20 / 2.6, 'Pre-trust: ' + ratio,
             ha='center', va='center', size=7.3, color=muted)
    fig.text((panel_left[0] + panel_width/2) / 7.2, 2.035 / 2.6, '1,144 held-out failures',
             ha='center', va='center', size=7.2, color=muted)

    # (b): the full empirical distribution on the same paired localizable cases.
    # Model colors have their own explicit legend; a/c retain region colors.
    model_names = ['Clean-SFT', 'GKD-full', 'CARE-OPD']
    model_colors = ['#7D8795', '#9D75B2', '#0072B2']
    markers = ['o', 's', 'D']
    model_handles, ecdf_data = [], {}
    b.axvline(0, color='#909AA5', linewidth=.7, linestyle=(0, (2, 2)), zorder=1)
    for model, name, color, marker in zip(models, model_names, model_colors, markers):
        key = model + '_localized'
        assert all(r[key] for r in eligible)
        score = np.sort(np.array([float(r[key]) for r in eligible]))
        assert len(score) == 189 and np.isfinite(score).all()
        assert -5.4 < score.min() <= score.max() < 1.35
        unique, frequencies = np.unique(score, return_counts=True)
        cumulative = np.cumsum(frequencies) / len(score)
        xx = np.r_[-5.4, unique, 1.35]
        yy = np.r_[0, cumulative, 1]
        b.step(xx, yy, where='post', color=color, linewidth=.4, alpha=.20, zorder=2)
        guide_x = np.linspace(xx[0], xx[-1], 2401)
        empirical_y = np.searchsorted(score, guide_x, side='right')/len(score)
        guide_y = gaussian_display_guide(empirical_y)
        assert (np.diff(guide_y) >= -1e-12).all() and guide_y.min() >= -1e-12 and guide_y.max() <= 1+1e-12
        b.fill_between(guide_x, 0, guide_y, color=color, alpha=.11, linewidth=0, zorder=.7)
        b.plot(guide_x, guide_y, color=color, linestyle='-', linewidth=1.45,
               zorder=4 if model == 'care_opd' else 3)
        marked_ranks = np.array([47, 94, 141])
        b.plot(score[marked_ranks], (marked_ranks+1)/len(score), linestyle='none', marker=marker,
               markersize=2.6, color=color, markerfacecolor='white', markeredgewidth=.65, zorder=5)
        model_handles.append(Line2D([], [], color=color, linestyle='-', marker=marker,
                                   markersize=2.6, markerfacecolor='white', markeredgewidth=.65,
                                   linewidth=1.4, label=name))
        ecdf_data[model] = {'n': len(score), 'sorted_scores': score.tolist(), 'x': xx.tolist(), 'cdf': yy.tolist(),
                            'positive': int((score > 0).sum()), 'nonpositive': int((score <= 0).sum()),
                            'median': float(np.median(score)), 'positive_fraction': float((score > 0).mean()),
                            'visual_guide': {'x': guide_x.tolist(), 'cdf': guide_y.tolist(), 'sigma_fraction': .02,
                                             'score_axis_sigma': .135, 'endpoint_rule': 'odd reflection',
                                             'max_cdf_displacement': float(np.max(abs(guide_y-empirical_y))),
                                             'only_empirical_values_are_measurements': True},
                            'empirical_markers': {'zero_based_ranks': marked_ranks.tolist(),
                                                 'score': score[marked_ranks].tolist(),
                                                 'cdf': ((marked_ranks+1)/len(score)).tolist()}}
    b.set_xlim(-5.4, 1.35)
    b.set_ylim(0, 1.03)
    b.set_xticks([-4, -2, 0, 1])
    b.set_yticks([0, .5, 1], ['0', '0.5', '1'])
    b.set_xlabel('Selected-field margin', labelpad=4)
    b.set_ylabel('Cumulative fraction', labelpad=4)
    b.grid(axis='both', color='#E4E9ED', linewidth=.45, zorder=0)
    b.set_axisbelow(True)
    # Reference-style upper strip, never mixed with the CARE-OPD data row.
    fig.legend(handles=model_handles, loc='center', bbox_to_anchor=((panel_left[1] + panel_width/2) / 7.2, 2.20 / 2.6),
               bbox_transform=fig.transFigure, frameon=False, fontsize=7.3, ncol=3,
               handlelength=1.2, handletextpad=.38, columnspacing=.7, borderaxespad=0, borderpad=0)
    fig.text((panel_left[1] + panel_width/2) / 7.2, 2.035 / 2.6,
             '189 paired, localizable failures', ha='center', va='center', size=7.1, color=muted)

    # (c): faint exact empirical ROC and a bold, explicitly visual-only guide.
    c.plot([0, 1], [0, 1], color='#9BA4AD', linewidth=.75, linestyle=':', zorder=1)
    handles = []
    rendered_curves = {}
    for region, color, name, marker in [('localized', colors[1], 'Selected', 's'), ('outside', colors[2], 'Outside', 'o')]:
        curve = curves[region]
        x, y = remove_collinear_roc(curve['fpr'], curve['tpr'], sum(labels), len(labels) - sum(labels))
        c.plot(x, y, color=color, linestyle='-', linewidth=.4, alpha=.20, zorder=2)
        gx, gy, guide = roc_visual_guide(x, y)
        c.fill_between(gx, 0, gy, color=color, alpha=.13, linewidth=0, zorder=.7)
        c.plot(gx, gy, color=color, linestyle='-', linewidth=1.45, zorder=3)
        marker_indices = [int(np.argmin(abs(x-q))) for q in [.2,.5,.8]]
        c.plot(x[marker_indices], y[marker_indices], linestyle='none', marker=marker, markersize=2.7,
               color=color, markerfacecolor='white', markeredgewidth=.65, zorder=4)
        guide['area_difference_from_empirical'] = guide['drawing_area_not_auroc'] - curve['auroc']
        rendered_curves[region] = {'fpr': x.tolist(), 'tpr': y.tolist(), 'raw_vertices': len(curve['fpr']),
                                   'display_vertices': len(x), 'empirical_area_preserved': True, 'visual_guide': guide,
                                   'empirical_marker_indices': marker_indices}
        handles.append(Line2D([], [], color=color, linestyle='-', marker=marker, markersize=2.7,
                              markerfacecolor='white', markeredgewidth=.65, linewidth=1.45,
                              label=f'{name} (emp. AUC {curve["auroc"]:.3f})'))
    fig.legend(handles=handles, loc='center', bbox_to_anchor=((panel_left[2] + panel_width/2) / 7.2, 2.1175 / 2.6),
               bbox_transform=fig.transFigure, frameon=False, fontsize=7.3, ncol=1,
               handlelength=1.8, handletextpad=.45, labelspacing=.35, borderaxespad=0, borderpad=0)
    c.set_xlim(0, 1.02)
    c.set_ylim(0, 1.03)
    c.set_xticks([0, .5, 1], ['0', '0.5', '1'])
    c.set_yticks([0, .5, 1], ['0', '0.5', '1'])
    c.set_xlabel('False positive rate', labelpad=4)
    c.set_ylabel('True positive rate', labelpad=4)
    c.grid(color='#E4E9ED', linewidth=.45, zorder=0)
    c.set_axisbelow(True)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for side in ['left', 'bottom']:
            ax.spines[side].set_color('#6C747C')
        ax.tick_params(length=2.5, width=.65)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text, bar in zip(count_labels, bars):
        tbox, bbox = text.get_window_extent(renderer), bar.get_window_extent(renderer)
        assert bbox.contains(tbox.x0, tbox.y0) and bbox.contains(tbox.x1, tbox.y1), 'Count does not fit bar'
    assert all(abs(ax.get_position().y0 - a.get_position().y0) < 1e-12 for ax in axes)
    assert all(abs(ax.get_position().y1 - a.get_position().y1) < 1e-12 for ax in axes)
    assert all(abs(ax.get_position().width - a.get_position().width) < 1e-12 for ax in axes)
    assert np.allclose(np.diff(panel_left), np.diff(panel_left)[0], atol=1e-12)
    bbox = fig.get_tightbbox(renderer).padded(2 / 72)
    active_ticks = {id(t.label1) for ax in axes for axis in [ax.xaxis, ax.yaxis] for t in axis._update_ticks()}
    all_ticks = {id(t.label1) for ax in axes for axis in [ax.xaxis, ax.yaxis] for t in axis.get_major_ticks() + axis.get_minor_ticks()}
    text_artists = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()
                    and (id(t) not in all_ticks or id(t) in active_ticks)]
    export_px = bbox.transformed(fig.dpi_scale_trans)
    clipped = [t.get_text() for t in text_artists if not export_px.contains(*t.get_window_extent(renderer).p0)
               or not export_px.contains(*t.get_window_extent(renderer).p1)]
    assert not clipped, clipped
    # Text-to-text intersections identify label and panel-margin collisions.
    collisions = []
    for i, t in enumerate(text_artists):
        box = t.get_window_extent(renderer)
        for u in text_artists[i+1:]:
            if box.overlaps(u.get_window_extent(renderer)):
                collisions.append([t.get_text(), u.get_text()])
    assert not collisions, collisions

    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / 'method_diagnostics_abc_v21'
    for fmt in ['pdf', 'svg', 'png']:
        fig.savefig(stem.with_suffix('.' + fmt), dpi=400, facecolor='white', bbox_inches=bbox, pad_inches=0)
    with matplotlib.rc_context({'svg.fonttype': 'path'}):
        fig.savefig(stem.with_name(stem.name + '_outlined').with_suffix('.svg'), bbox_inches=bbox, pad_inches=0)
    manifest = {
        'figure': stem.name,
        'sources': [{'path': p.relative_to(args.repo_root).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
        'figure_size_inches': [bbox.width, bbox.height], 'working_size_inches': list(fig.get_size_inches()),
        'layout': {'equal_plot_size_inches': [panel_width, panel_height], 'plot_left_inches': panel_left.tolist(),
                   'upper_keys_outside_data': True, 'cohort_details_in_caption': True,
                   'user_reference_layout': 'separate panel letters; title row; legend/metadata strip; equal axes; no footer key'},
        'panel_a': {'failure_rows': int(overall['rows']), 'full_tokens': counts[0], 'selected_tokens': counts[1],
                    'outside_tokens': counts[2], 'selected_fraction_micro': counts[1]/counts[0], 'full_to_selected': counts[0]/counts[1]},
        'panel_b': {'models': models, 'region_drawn': 'selected fields only', 'n': 189, 'paired_case_ids': [r['id'] for r in eligible],
                    'chart_type': 'empirical cumulative distribution with a Gaussian-smoothed visual guide and area fill', 'x_axis': 'selected-field margin',
                    'y_axis': 'F(m) = count(M <= m) / 189', 'model_colors': model_colors, 'ecdf': ecdf_data,
                    'line_policy': 'faint observed empirical steps; bold fixed-bandwidth Gaussian display guide; no score truncation, jitter, or outcome-dependent tuning',
                    'original_mean_reference_not_drawn': {'regions': region_keys, 'values': values.tolist(), 'n_full': 200, 'n_localized': 189}},
        'panel_c': {'n': 189, 'rescued': 164, 'unrescued': 25, 'roc': curves,
                    'auroc_difference': float(diff['difference']), 'existing_paired_ci95': [float(diff['ci95_lower']), float(diff['ci95_upper'])],
                    'bootstrap_replicates_in_source': int(diff['bootstrap_valid_replicates']), 'new_bootstrap_run': False,
                    'rendered_roc': rendered_curves,
                    'display_policy': 'faint exact empirical ROC; bold Gaussian-smoothed parametric path is a visual guide only; no upper hull; all AUC/CI statistics use original empirical ROC'},
        'fill_policy': 'translucent area from zero to the display guides, not uncertainty or confidence intervals',
        'smoothing_specification': {'kind': 'Gaussian display smoothing', 'uniform_grid_points': 2401,
                                   'normalized_sigma': .02, 'kernel_radius_sigma': 4,
                                   'endpoint_extension': 'odd reflection', 'all_series_same_fractional_bandwidth': True,
                                   'markers': 'original empirical points only, never synthetic guide points'},
        'checks': {'source_auroc_match': True, 'pairwise_auroc_match': True, 'per_example_mean_matches_summary': True,
                   'token_partition': True, 'no_label_collisions': True, 'no_clipped_text': True, 'axes_top_bottom_aligned': True,
                   'axes_equal_width': True, 'equal_plot_gutters': True,
                   'guide_monotone_bounded': True, 'guide_endpoints_preserved': True,
                   'guide_sigma_fixed_across_series': True, 'shade_is_not_uncertainty': True,
                   'original_empirical_roc_retained': True, 'reported_auc_not_from_visual_guide': True},
        'scope': 'Existing frozen results, descriptive and failure-conditioned. Not a causal ablation or deployment-time confidence estimator.',
        'no_new_training_or_inference': True,
    }
    stem.with_suffix('.data.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    plt.close(fig)
    print(json.dumps({'output': str(stem), 'size': manifest['figure_size_inches'], 'checks': manifest['checks'],
                      'auroc': {k: v['auroc'] for k, v in curves.items()}}, indent=2))


if __name__ == '__main__':
    main()
