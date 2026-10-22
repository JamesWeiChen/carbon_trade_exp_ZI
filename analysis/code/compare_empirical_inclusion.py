"""Regenerate matched before/after figures without changing the stored ZI sample.

Run from the repository: python analysis/code/compare_empirical_inclusion.py
Requires the baseline commit in local git history. Three bounded original-model
draws verify stored ZI provenance; they never replace benchmark rows.
"""
from __future__ import annotations

import ast
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import types

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import matplotlib
import scipy
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
REL_SOURCE = "analysis/code/figure2_trading_dynamics_zi_v2.py"
SOURCE = ROOT / REL_SOURCE
BASELINE = "b06cfb712817fae45e4e14a965a28f520a14204c"
OUTPUT = ROOT / "analysis/output/figure2_zi_simultaneous_two_sided_resale"
COMPARE = ROOT / "analysis/output/empirical_inclusion_comparison"


def git_bytes(path):
    return subprocess.check_output(["git", "show", f"{BASELINE}:{path}"], cwd=ROOT)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_model(source, name):
    module = types.ModuleType(name)
    module.__file__ = str(SOURCE)
    exec(compile(source, str(SOURCE), "exec"), module.__dict__)
    module.ZI_PROGRESS_EVERY_QUOTES = 0
    return module


def equal_frame(a, b, columns=None):
    if columns is not None:
        a, b = a[columns], b[columns]
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, atol=1e-10, rtol=1e-10)


def manual_spearman(values, order):
    """Independent rank-then-Pearson calculation (production uses spearmanr)."""
    valid = np.isfinite(values) & np.isfinite(order)
    values, order = np.asarray(values)[valid], np.asarray(order)[valid]
    if len(values) < 3 or len(set(values)) < 2 or len(set(order)) < 2:
        return np.nan
    def ranks(a):
        return np.array([1 + sum(a < x) + (sum(a == x)-1)/2 for x in a])
    return float(np.corrcoef(ranks(values), ranks(order))[0, 1])


def independent_values(trades, subj, packages, units, markets):
    """Rebuild frontier, unit values, holdings and rhos without production helpers."""
    expected = []
    for key, tr in trades.groupby(["session_code", "round_raw"], sort=True):
        rows = subj.loc[subj.session_code.eq(key[0]) & subj.round_raw.eq(key[1])]
        holding = {int(r.id_in_group): int(r.permits) for r in rows.itertuples()}
        schedules = {}
        for r in rows.itertuples():
            cap, b, max_q = int(r.group_cap_total), int(r.carbon_emission_per_unit), int(r.max_production)
            disturbances = json.loads(r.disturbance_values)
            profits = [0.]
            for q in range(1, max_q+1):
                profits.append(profits[-1] + r.market_price - r.marginal_cost_coefficient*q - disturbances[q-1])
            frontier = [max(profits[:min(max_q, h//b)+1]) for h in range(cap+1)]
            mv = [0.] * (cap+1)
            for q in range(1, max_q+1):
                if q*b > cap:
                    break
                gain = (frontier[q*b] - frontier[(q-1)*b])/b
                for h in range((q-1)*b+1, q*b+1):
                    mv[h] = gain
            schedules[int(r.id_in_group)] = mv
        for t in tr.sort_values(["trade_index", "seconds"], kind="mergesort").itertuples():
            buyer, seller = int(t.buyer_id), int(t.seller_id)
            for unit in range(1, int(t.quantity)+1):
                bh, sh = holding[buyer], holding[seller]
                expected.append(dict(session_code=key[0], round_raw=key[1], trade_index=t.trade_index,
                    unit_number_within_package=unit, buyer_permit_value=schedules[buyer][bh+1],
                    seller_permit_cost=schedules[seller][sh], buyer_permit_ordinal=bh+1,
                    seller_permit_ordinal=sh, buyer_holding_before_unit=bh,
                    seller_holding_before_unit=sh, transaction_order=t.trade_index))
                holding[buyer] += 1; holding[seller] -= 1
        assert all(holding[int(r.id_in_group)] == int(r.current_permits) for r in rows.itertuples())
    oracle = pd.DataFrame(expected)
    equal_frame(oracle, units, list(oracle.columns))
    for key, u in units.groupby(["session_code", "round_raw", "trade_index"], sort=True):
        p = packages.loc[packages.session_code.eq(key[0]) & packages.round_raw.eq(key[1]) & packages.trade_index.eq(key[2])].iloc[0]
        for side, col in [("buyer", "buyer_permit_value"), ("seller", "seller_permit_cost")]:
            assert np.isclose(p[col+"_avg"], sum(u[col])/len(u))
            assert np.isclose(p[col+"_total"], sum(u[col]))
            assert p[side+"_n_eligible_units"] == p.quantity == len(u)
    for row in markets.itertuples():
        for level, panel in [("package", packages), ("unit", units)]:
            q = panel.loc[panel.session_code.eq(row.session_code) & panel.round_raw.eq(row.round_raw)]
            for side, col in [("buyer", "buyer_permit_value"), ("seller", "seller_permit_cost")]:
                value_col = col+"_avg" if level == "package" else col
                rho = manual_spearman(q[value_col].to_numpy(), q.transaction_order.to_numpy())
                assert np.isclose(rho, getattr(row, f"rho_{side}_{level}"), equal_nan=True)


def render(m, panels, zi, destination):
    m.OUTPUT_DIR = destination
    destination.mkdir(parents=True, exist_ok=True)
    market, packages, units, pairs = panels
    theory = m.theory_benchmark_table()
    results = []
    for level in ("package", "unit"):
        for mode in ("standardized", "raw"):
            results.append(m.create_combined_figure(market, pairs, zi, theory,
                rho_buyer_col=f"rho_buyer_{level}", rho_seller_col=f"rho_seller_{level}",
                price_change_mode=mode, observation_level="permit_unit" if level=="unit" else "package",
                title_suffix="", footer_text="", output_filename=f"figure_2abcd_{level}_{mode}_price_change.png"))
    stats = pd.concat([m.early_late_statistics(market, f"rho_buyer_{level}", f"rho_seller_{level}",
                     "permit_unit" if level=="unit" else "package") for level in ("package", "unit")], ignore_index=True)
    tables = {
        "figure_2_market_period_correlations_both_levels.csv": market,
        "figure_2_package_level_values_costs.csv": packages,
        "figure_2_unit_level_values_costs.csv": units,
        "figure_2_price_change_pairs.csv": pairs,
        "figure_2_early_late_statistics_both_levels.csv": stats,
        "figure_2_price_change_regression_results.csv": pd.DataFrame(results),
    }
    for name, frame in tables.items():
        frame.to_csv(destination/name, index=False, encoding="utf-8-sig")
    m.write_validation_report(market, packages, units, pairs, zi, theory)
    return stats, pd.DataFrame(results)


def composite(before, after, dest, title):
    a, b = Image.open(before).convert("RGB"), Image.open(after).convert("RGB")
    assert a.size == b.size
    width, height = a.size
    canvas = Image.new("RGB", (2*width, height+110), "white")
    canvas.paste(a, (0, 110)); canvas.paste(b, (width, 110))
    font_path = "/System/Library/Fonts/Helvetica.ttc"
    font = ImageFont.truetype(font_path, 40) if Path(font_path).exists() else ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 12), title, font=font, fill="black")
    draw.text((40, 60), "BEFORE: initial-endowment filters", font=font, fill="black")
    draw.text((width+40, 60), "AFTER: all purchases and sales", font=font, fill="black")
    canvas.save(dest)


def main():
    COMPARE.mkdir(parents=True, exist_ok=True)
    baseline_source, new_source = git_bytes(REL_SOURCE).decode(), SOURCE.read_text()
    old = load_model(baseline_source, "baseline_model")
    new = load_model(new_source, "revised_model")
    # No changes to model, RNG, sampling, frontier, correlation or plotting helpers.
    def nodes(source):
        return {node.name: ast.dump(node) for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    a, b = nodes(baseline_source), nodes(new_source)
    changed_functions = sorted(k for k in a if a[k] != b[k])
    assert set(changed_functions) == {"reconstruct_one_market", "write_validation_report", "main"}
    assert [ast.dump(n) for n in ast.parse(baseline_source).body if not isinstance(n, ast.FunctionDef)] == [ast.dump(n) for n in ast.parse(new_source).body if not isinstance(n, ast.FunctionDef)]
    input_paths = [ROOT/f"analysis/input/{name}_panel.csv" for name in ("trades", "subj_lv", "group_lv")]
    zi_paths = sorted(OUTPUT.glob("figure_2_zi_*.csv"))
    hashes = {}
    for path in input_paths + zi_paths:
        relative = str(path.relative_to(ROOT))
        data = path.read_bytes()
        assert data == git_bytes(relative), f"Input/ZI artifact changed since baseline: {relative}"
        hashes[relative] = sha(data)
    trades, subj, group = new.load_raw_data()
    before = old.build_analysis_data(trades, subj, group)
    after = new.build_analysis_data(trades, subj, group)
    # Verify old empirical exports against an actual fresh reconstruction.
    for panel, name in zip(before, ["market_period_correlations_both_levels", "package_level_values_costs", "unit_level_values_costs", "price_change_pairs"]):
        stored_path = OUTPUT/f"figure_2_{name}.csv"
        equal_frame(panel, pd.read_csv(io.BytesIO(git_bytes(str(stored_path.relative_to(ROOT))))))
    market_before, packages_before, units_before, pairs_before = before
    market_after, packages_after, units_after, pairs_after = after
    allowed_market = {"n_buyer_eligible_units", "n_seller_eligible_units", "rho_buyer_package", "rho_seller_package", "rho_buyer_unit", "rho_seller_unit"}
    equal_frame(market_before, market_after, [c for c in market_before if c not in allowed_market])
    allowed_package = {"buyer_permit_value_avg", "seller_permit_cost_avg", "buyer_permit_value_total", "seller_permit_cost_total", "buyer_n_eligible_units", "seller_n_eligible_units"}
    equal_frame(packages_before, packages_after, [c for c in packages_before if c not in allowed_package])
    allowed_unit = {"buyer_permit_value", "seller_permit_cost", "buyer_value_eligible", "seller_cost_eligible"}
    equal_frame(units_before, units_after, [c for c in units_before if c not in allowed_unit])
    equal_frame(pairs_before, pairs_after, [c for c in pairs_before if c not in allowed_package | allowed_unit])
    assert (units_after.buyer_permit_value == units_before.buyer_permit_value_raw).all()
    assert (units_after.seller_permit_cost == units_before.seller_permit_cost_raw).all()
    assert units_after.buyer_value_eligible.all() and units_after.seller_cost_eligible.all()
    assert market_after.reconstruction_ok.all()
    independent_values(trades, subj, packages_after, units_after, market_after)
    zi = pd.read_csv(OUTPUT/"figure_2_zi_simulation_benchmark.csv")
    assert pd.api.types.is_bool_dtype(zi["valid_for_95_region"]), "ZI eligibility must be boolean, never truthy strings"
    assert len(zi)==10000 and list(zi.simulation_id)==list(range(1,10001))
    assert zi.role_mode.eq(new.ZI_ROLE_MODE).all() and not zi.quote_event_cutoff_used.any()
    specs = [old.build_zi_market_spec(rows) for _, rows in subj.groupby(old.MARKET_KEYS, sort=True)]
    selected = np.random.default_rng(np.random.SeedSequence([old.ZI_RANDOM_SEED,0])).integers(0,len(specs),size=len(zi))
    np.testing.assert_array_equal(selected, zi.pool_index)
    assert all((str(r.session_code),int(r.round_raw)) == (specs[int(r.pool_index)]['session_code'],specs[int(r.pool_index)]['round_raw']) for r in zi.itertuples())
    # Bounded spot verification of existing records; never replace or redraw them.
    def timeout(*_):
        raise TimeoutError("Stored-ZI provenance check exceeded 10 seconds")
    signal.signal(signal.SIGALRM, timeout)
    for simulation in (1,2,3):
        signal.setitimer(signal.ITIMER_REAL,10)
        try:
            actual, *_ = old._simulate_zi_task(specs,(simulation,int(selected[simulation-1])))
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
        saved = zi.iloc[simulation-1]
        for key, value in actual.items():
            if isinstance(value,(float,int,np.number)):
                assert np.isclose(value,saved[key],equal_nan=True,atol=1e-10,rtol=1e-10),(simulation,key)
            else:
                assert value == saved[key],(simulation,key)
    diff = market_before.merge(market_after,on=new.MARKET_KEYS,suffixes=("_before","_after"),validate="one_to_one")
    rho_columns = [f"rho_{side}_{level}" for level in ("package","unit") for side in ("buyer","seller")]
    rho_summary = []
    for col in rho_columns:
        x,y=diff[col+"_before"],diff[col+"_after"]
        diff[col+"_delta"] = y-x
        common=x.notna() & y.notna()
        rho_summary.append(dict(measure=col,finite_before=int(x.notna().sum()),finite_after=int(y.notna().sum()),
            finite_to_undefined=int((x.notna()&y.isna()).sum()),undefined_to_finite=int((x.isna()&y.notna()).sum()),
            changed_finite=int((common & ~np.isclose(x,y,equal_nan=True)).sum()),
            common_market_mean_delta=float((y[common]-x[common]).mean()),max_absolute_delta=float(abs(y[common]-x[common]).max())))
    diff.to_csv(COMPARE/"market_comparison.csv",index=False)
    pd.DataFrame(rho_summary).to_csv(COMPARE/"rho_change_summary.csv",index=False)
    membership=[]; plotted=[]
    for level in ("package","unit"):
        cols=[f"rho_buyer_{level}",f"rho_seller_{level}",f"rho_price_{level}_raw"]
        mask_before=market_before[cols].notna().all(axis=1)
        mask_after=market_after[cols].notna().all(axis=1)
        common=mask_before & mask_after
        for idx in market_before.index:
            membership.append(dict(observation_level=level,session_code=market_before.loc[idx,"session_code"],round_raw=int(market_before.loc[idx,"round_raw"]),before=bool(mask_before[idx]),after=bool(mask_after[idx])))
        for col in cols:
            plotted.append(dict(observation_level=level,measure=col,n_before=int(mask_before.sum()),n_after=int(mask_after.sum()),
                entered=int((~mask_before&mask_after).sum()),exited=int((mask_before&~mask_after).sum()),
                mean_before=float(market_before.loc[mask_before,col].mean()),mean_after=float(market_after.loc[mask_after,col].mean()),
                common_n=int(common.sum()),common_before=float(market_before.loc[common,col].mean()),common_after=float(market_after.loc[common,col].mean())))
    pd.DataFrame(membership).to_csv(COMPARE/"plot_sample_membership.csv",index=False)
    pd.DataFrame(plotted).to_csv(COMPARE/"plotted_mean_comparison.csv",index=False)
    old_stats,old_price=render(old,before,zi,COMPARE/"before")
    new_stats,new_price=render(new,after,zi,OUTPUT)
    equal_frame(old_price,new_price)
    # Keep the unchanged regression export byte-for-byte, avoiding incidental
    # cross-version float formatting differences in an unrelated result table.
    regression_path=OUTPUT/"figure_2_price_change_regression_results.csv"
    original_regression=git_bytes(str(regression_path.relative_to(ROOT)))
    equal_frame(new_price,pd.read_csv(io.BytesIO(original_regression)))
    regression_path.write_bytes(original_regression)
    old_stats.merge(new_stats,on=["observation_level","measure","period_group"],suffixes=("_before","_after")).to_csv(COMPARE/"early_late_comparison.csv",index=False)
    panel_c_pixels=[]
    for level in ("package","unit"):
        for mode in ("standardized","raw"):
            filename=f"figure_2abcd_{level}_{mode}_price_change.png"
            image_before=np.asarray(Image.open(COMPARE/"before"/filename))
            image_after=np.asarray(Image.open(OUTPUT/filename))
            assert image_before.shape==image_after.shape
            h,w=image_before.shape[:2]
            # Right-hand upper panel and its labels/colorbar, above panel d.
            roi=(slice(int(.07*h),int(.66*h)),slice(int(.48*w),w))
            identical=bool(np.array_equal(image_before[roi],image_after[roi]))
            assert identical, filename
            panel_c_pixels.append(filename)
    for level in ("package","unit"):
        filename=f"figure_2abcd_{level}_standardized_price_change.png"
        composite(COMPARE/"before"/filename,OUTPUT/filename,COMPARE/f"before_after_{level}_standardized.png",f"{level}: same inputs and same 10,000 ZI outcomes")
    for relative,digest in hashes.items():
        assert sha((ROOT/relative).read_bytes())==digest
    report=dict(baseline_commit=BASELINE,source_sha256=sha(SOURCE.read_bytes()),unchanged_input_and_zi_sha256=hashes,
        environment=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__),
        changed_functions=changed_functions,zi_reused=10000,zi_provenance_spot_checks=[1,2,3],
        zi_not_replaced=True,markets=len(market_after),packages=len(packages_after),units=len(units_after),
        buyer_units_before=int(units_before.buyer_value_eligible.sum()),buyer_units_after=len(units_after),
        seller_units_before=int(units_before.seller_cost_eligible.sum()),seller_units_after=len(units_after),
        unchanged_price_pairs=True,unchanged_price_rhos=True,unchanged_price_ols=True,unchanged_zi_and_theory=True,
        independent_value_and_rank_oracle_pass=True,old_exports_reproduced=True,panel_c_pixel_identical=panel_c_pixels,
        rho_summary=rho_summary,plotted_means=plotted)
    (COMPARE/"verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!='unchanged_input_and_zi_sha256'},indent=2),flush=True)


if __name__ == "__main__":
    main()
