#!/usr/bin/env python3
"""Summarize reviewer-requested action statistics and physical outcomes.

Run after copying experiment_logs back from the offline industrial PC.
Only the Python standard library and NumPy are required.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


ACTION_COLUMNS = [
    *(f'mapped_{x}' for x in ('px','py','pz','rx','ry','timing','vx')),
    'executed_target_x', 'executed_target_y', 'executed_target_z',
    'k', 't_delay', 'cmd_vx', 'time_to_hit', 'fit_point_count',
]


def read_rows(path):
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def as_float(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return np.nan


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_actions(run_dirs, out_dir):
    grouped = defaultdict(list)
    for run_dir in run_dirs:
        for row in read_rows(run_dir / 'actions.csv'):
            # Timing representation is intentionally not a logging or
            # reporting factor; summarize only by observation method.
            grouped[row.get('method', '')].append(row)

    rows = []
    for method, items in sorted(grouped.items()):
        for column in ACTION_COLUMNS:
            values = np.asarray([as_float(r, column) for r in items], dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            rows.append({
                'method': method,
                'variable': column,
                'n': values.size,
                'mean': np.mean(values),
                'std': np.std(values, ddof=1) if values.size > 1 else 0.0,
                'min': np.min(values),
                'p05': np.percentile(values, 5),
                'median': np.median(values),
                'p95': np.percentile(values, 95),
                'max': np.max(values),
            })
        raw = np.asarray(
            [[as_float(r, f'raw_{x}') for x in ('px','py','pz','rx','ry','timing','vx')] for r in items], dtype=float
        )
        for i in range(7):
            vals = raw[:, i]
            vals = vals[np.isfinite(vals)]
            rows.append({
                'method': method,
                'variable': f"raw_{('px','py','pz','rx','ry','timing','vx')[i]}_saturation_rate_abs_ge_0.98",
                'n': vals.size,
                'mean': np.mean(np.abs(vals) >= 0.98) if vals.size else np.nan,
                'std': '', 'min': '', 'p05': '', 'median': '', 'p95': '', 'max': '',
            })
    write_csv(
        out_dir / 'action_statistics.csv',
        ['method','variable','n','mean','std','min','p05','median','p95','max'],
        rows,
    )


def summarize_outcomes(run_dirs, out_dir):
    counts = defaultdict(lambda: defaultdict(int))
    for run_dir in run_dirs:
        seen = set()
        for row in read_rows(run_dir / 'manual_outcomes.csv'):
            # If an operator corrected an episode, the latest record wins.
            key = (row.get('run_id', run_dir.name), row.get('episode', ''))
            seen.add(key)
        latest = {}
        for row in read_rows(run_dir / 'manual_outcomes.csv'):
            key = (row.get('run_id', run_dir.name), row.get('episode', ''))
            latest[key] = row
        for row in latest.values():
            counts[row.get('method', '')][row.get('outcome', 'unknown')] += 1

    rows = []
    for method, method_counts in sorted(counts.items()):
        total = sum(method_counts.values())
        for outcome, count in sorted(method_counts.items()):
            rows.append({
                'method': method, 'outcome': outcome, 'count': count,
                'total_labeled_trials': total,
                'rate': count / total if total else np.nan,
            })
    write_csv(
        out_dir / 'outcome_statistics.csv',
        ['method','outcome','count','total_labeled_trials','rate'], rows,
    )


def summarize_coverage(run_dirs, out_dir, bounds, linear_mapping=None):
    if bounds is None:
        return
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    grouped = defaultdict(list)
    grouped_axis = defaultdict(list)
    episode_outside = defaultdict(lambda: defaultdict(bool))
    for run_dir in run_dirs:
        for row in read_rows(run_dir / 'trajectory_points.csv'):
            method = row.get('method', '')
            xyz = np.asarray([as_float(row, x) for x in ('x','y','z')])
            if not np.all(np.isfinite(xyz)):
                continue
            if linear_mapping is not None:
                A, b = linear_mapping
                # trajectory_points.csv contains physical vision measurements,
                # while calibration bounds are defined in true coordinates.
                # Invert p_vision = A p_true + b to obtain an estimated true
                # coordinate for the coverage calculation.
                xyz = np.linalg.solve(A, xyz - b)
            outside_axis = np.asarray([
                not (xmin <= xyz[0] <= xmax),
                not (ymin <= xyz[1] <= ymax),
                not (zmin <= xyz[2] <= zmax),
            ], dtype=bool)
            outside = bool(np.any(outside_axis))
            grouped[method].append(outside)
            grouped_axis[method].append(outside_axis)
            episode_outside[method][(row.get('run_id',''), row.get('episode',''))] |= outside

    rows = []
    for method, values in sorted(grouped.items()):
        values = np.asarray(values, dtype=bool)
        axis_values = np.asarray(grouped_axis[method], dtype=bool)
        episode_values = np.asarray(list(episode_outside[method].values()), dtype=bool)
        rows.append({
            'method': method,
            'sample_count': values.size,
            'outside_sample_count': int(values.sum()),
            'outside_sample_rate': float(values.mean()) if values.size else np.nan,
            'episode_count': episode_values.size,
            'episodes_with_any_outside_sample': int(episode_values.sum()),
            'episode_outside_rate': float(episode_values.mean()) if episode_values.size else np.nan,
            'outside_x_count': int(axis_values[:, 0].sum()),
            'outside_x_rate': float(axis_values[:, 0].mean()),
            'outside_y_count': int(axis_values[:, 1].sum()),
            'outside_y_rate': float(axis_values[:, 1].mean()),
            'outside_z_count': int(axis_values[:, 2].sum()),
            'outside_z_rate': float(axis_values[:, 2].mean()),
            'coordinate_basis': 'inverse_mapped_true_estimate' if linear_mapping is not None else 'measured_proxy',
            'bounds': ','.join(str(x) for x in bounds),
        })
    write_csv(
        out_dir / 'calibration_coverage.csv',
        ['method','sample_count','outside_sample_count','outside_sample_rate','episode_count',
         'episodes_with_any_outside_sample','episode_outside_rate',
         'outside_x_count','outside_x_rate','outside_y_count','outside_y_rate',
         'outside_z_count','outside_z_rate','coordinate_basis','bounds'], rows,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_root', type=Path, help='experiment_logs directory')
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument(
        '--calibration-bounds', type=float, nargs=6, metavar=('XMIN','XMAX','YMIN','YMAX','ZMIN','ZMAX'),
        help='True calibrated region bounds in metres; omit until verified.',
    )
    parser.add_argument(
        '--linear-mapping', type=Path, default=None,
        help='NPZ containing A and b for p_vision=A*p_true+b; used to inverse-map physical measurements.',
    )
    args = parser.parse_args()
    run_dirs = sorted(p for p in args.log_root.iterdir() if p.is_dir())
    out_dir = args.out or (args.log_root / 'summary')
    summarize_actions(run_dirs, out_dir)
    summarize_outcomes(run_dirs, out_dir)
    linear_mapping = None
    if args.linear_mapping is not None:
        data = np.load(str(args.linear_mapping))
        linear_mapping = (
            np.asarray(data['A'], dtype=float),
            np.asarray(data['b'], dtype=float),
        )
    summarize_coverage(
        run_dirs, out_dir, args.calibration_bounds, linear_mapping=linear_mapping
    )
    print(f'Wrote summaries to {out_dir}')


if __name__ == '__main__':
    main()
