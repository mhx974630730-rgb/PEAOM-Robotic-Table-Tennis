#!/usr/bin/env python3
"""Quantify measurement-error propagation through quadratic trajectory fitting."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_rows(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def quadratic_fit(t, p):
    t0 = float(t[0])
    tr = t - t0
    design = np.column_stack((tr ** 2, tr, np.ones_like(tr)))
    coeff = np.column_stack([
        np.linalg.lstsq(design, p[:, axis], rcond=None)[0] for axis in range(3)
    ])
    return t0, coeff


def state_at(fit, t):
    t0, c = fit
    tr = float(t - t0)
    pos = c[0] * tr ** 2 + c[1] * tr + c[2]
    vel = 2.0 * c[0] * tr + c[1]
    return pos, vel


def descending_root(fit, axis, value, after):
    t0, c = fit
    a, b, cc = c[:, axis]
    roots = np.roots([a, b, cc - value]) if abs(a) > 1e-12 else np.roots([b, cc - value])
    candidates = []
    for root in roots:
        if abs(np.imag(root)) > 1e-8:
            continue
        t = t0 + float(np.real(root))
        if t <= after:
            continue
        _, vel = state_at(fit, t)
        if vel[axis] < 0:
            candidates.append(t)
    return min(candidates) if candidates else None


def predicted_hit_time(fit, after, hit_x, table_z, gravity, restitution_z, friction_xy):
    bounce_t = descending_root(fit, 2, table_z, after)
    if bounce_t is None:
        return None
    pos, vel = state_at(fit, bounce_t)
    vel = vel.copy()
    vel[2] = -vel[2] * restitution_z
    vel[:2] *= friction_xy
    if abs(vel[0]) < 0.1:
        return None
    dt = (hit_x - pos[0]) / vel[0]
    if dt <= 0:
        return None
    return float(bounce_t + dt)


def summarize(records, out_path):
    grouped = defaultdict(list)
    for r in records:
        grouped[(r['method'], r['window'])].append(r)
    fields = [
        'method','window','episodes','pos_rmse_x','pos_rmse_y','pos_rmse_z',
        'vel_rmse_x','vel_rmse_y','vel_rmse_z','hit_time_mae','hit_time_rmse',
        'valid_hit_time_pairs'
    ]
    rows = []
    for (method, window), items in sorted(grouped.items(), key=lambda x: (x[0][0], int(x[0][1]))):
        pos = np.asarray([x['pos_error'] for x in items], dtype=float)
        vel = np.asarray([x['vel_error'] for x in items], dtype=float)
        dt = np.asarray([x['hit_time_error'] for x in items], dtype=float)
        valid_dt = dt[np.isfinite(dt)]
        rows.append({
            'method': method, 'window': window, 'episodes': len(items),
            **{f'pos_rmse_{a}': np.sqrt(np.mean(pos[:, i] ** 2)) for i, a in enumerate('xyz')},
            **{f'vel_rmse_{a}': np.sqrt(np.mean(vel[:, i] ** 2)) for i, a in enumerate('xyz')},
            'hit_time_mae': np.mean(np.abs(valid_dt)) if valid_dt.size else np.nan,
            'hit_time_rmse': np.sqrt(np.mean(valid_dt ** 2)) if valid_dt.size else np.nan,
            'valid_hit_time_pairs': valid_dt.size,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_root', type=Path)
    parser.add_argument('--windows', type=int, nargs='+', default=[4, 6, 8, 10, 12, 16, 0],
                        help='0 means all samples up to the prediction plane')
    parser.add_argument('--prediction-plane-x', type=float, default=-0.3)
    parser.add_argument('--hit-plane-x', type=float, default=0.8)
    parser.add_argument('--table-z', type=float, default=0.0)
    parser.add_argument('--gravity', type=float, default=9.81)
    parser.add_argument('--restitution-z', type=float, default=0.85)
    parser.add_argument('--friction-xy', type=float, default=0.95)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    episodes = defaultdict(list)
    all_errors = defaultdict(list)
    for run_dir in sorted(p for p in args.log_root.iterdir() if p.is_dir()):
        path = run_dir / 'sim_observations.csv'
        if not path.exists():
            continue
        for row in read_rows(path):
            method = row['method']
            key = (row['run_id'], method, int(row['episode']))
            t = float(row['t_ros'])
            true_p = np.asarray([float(row[f'true_{a}']) for a in 'xyz'])
            obs_p = np.asarray([float(row[f'obs_{a}']) for a in 'xyz'])
            episodes[key].append((t, true_p, obs_p))
            all_errors[method].append(obs_p - true_p)

    records = []
    for (_, method, _), samples in episodes.items():
        samples.sort(key=lambda x: x[0])
        t = np.asarray([x[0] for x in samples], dtype=float)
        true_p = np.asarray([x[1] for x in samples], dtype=float)
        obs_p = np.asarray([x[2] for x in samples], dtype=float)
        crossing = np.flatnonzero(true_p[:, 0] >= args.prediction_plane_x)
        if crossing.size == 0:
            continue
        end = int(crossing[0]) + 1
        for window in args.windows:
            start = 0 if window == 0 else max(0, end - window)
            if end - start < 4:
                continue
            tw = t[start:end]
            true_fit = quadratic_fit(tw, true_p[start:end])
            obs_fit = quadratic_fit(tw, obs_p[start:end])
            eval_t = tw[-1]
            true_state = state_at(true_fit, eval_t)
            obs_state = state_at(obs_fit, eval_t)
            true_hit = predicted_hit_time(
                true_fit, eval_t, args.hit_plane_x, args.table_z, args.gravity,
                args.restitution_z, args.friction_xy,
            )
            obs_hit = predicted_hit_time(
                obs_fit, eval_t, args.hit_plane_x, args.table_z, args.gravity,
                args.restitution_z, args.friction_xy,
            )
            records.append({
                'method': method, 'window': window,
                'pos_error': obs_state[0] - true_state[0],
                'vel_error': obs_state[1] - true_state[1],
                'hit_time_error': (obs_hit - true_hit) if obs_hit is not None and true_hit is not None else np.nan,
            })

    out_dir = args.out or (args.log_root / 'summary')
    summarize(records, out_dir / 'state_sensitivity.csv')
    corr_rows = []
    for method, values in sorted(all_errors.items()):
        values = np.asarray(values, dtype=float)
        corr = np.corrcoef(values, rowvar=False) if len(values) > 1 else np.full((3,3), np.nan)
        corr_rows.append({
            'method': method, 'n': len(values),
            'corr_xy': corr[0,1], 'corr_xz': corr[0,2], 'corr_yz': corr[1,2],
        })
    with (out_dir / 'measurement_error_cross_axis.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method','n','corr_xy','corr_xz','corr_yz'])
        writer.writeheader(); writer.writerows(corr_rows)
    print(f'Wrote sensitivity summaries to {out_dir}')


if __name__ == '__main__':
    main()
