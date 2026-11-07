"""Pipeline monitor: shows the newest data-pipeline run, read from the files that run wrote."""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.serving.monitor import (checks_table, cleaning_table, find_run_dir, load_run, stage_table,  # noqa: E402
                                 summary, timing_table)

st.set_page_config(page_title='Pipeline monitor', layout='wide')
st.title('Data pipeline monitor')
st.caption('Every number on this page is read from the files a pipeline run wrote (manifest, cleaning log, quality gate, '
           'leakage report). Nothing is typed in by hand.')

run_dir = find_run_dir(ROOT)
if run_dir is None:
    st.error('No pipeline run found. Expected pipeline_runs/<timestamp>/run_status.json or docs/data_run/manifest.json.')
    st.stop()

run = load_run(run_dir)
s = summary(run)

status = s['status'] or 'UNKNOWN'
(st.success if status in ('PASS', 'RUNNING') else st.error)(f"Latest run: {status} ({s['source']})")
c1, c2, c3, c4 = st.columns(4)
c1.metric('Last run (UTC)', s['last_run_utc'] or 'n/a')
c2.metric('Quality gate', s['quality_status'] or 'n/a')
c3.metric('Leakage checks passed', f"{s['leakage_passed']} / {s['leakage_total']}")
c4.metric('Final interactions', f"{s['final_interactions']:,}" if s['final_interactions'] else 'n/a')

st.subheader('Data volume at each stage')
stages = stage_table(run)
if len(stages):
    st.bar_chart(stages, x='stage', y='interactions', sort=False)          # keep pipeline order, not alphabetical
    st.dataframe(stages.style.format({'users': '{:,}', 'items': '{:,}', 'interactions': '{:,}', 'avg_seq_len': '{:.2f}'}),
                 hide_index=True, width='stretch')

st.subheader('Cleaning rules and rows removed')
st.dataframe(cleaning_table(run).style.format({'rows_before': '{:,}', 'rows_after': '{:,}', 'removed': '{:,}'}),
             hide_index=True, width='stretch')

if run['kcore'] is not None:
    st.subheader('Iterative 5-core filter')
    st.line_chart(run['kcore'].set_index('iteration')[['rows_after']].rename(columns={'rows_after': 'interactions remaining'}))

st.subheader('Checks')
st.dataframe(checks_table(run), hide_index=True, width='stretch')

timings = timing_table(run)
if len(timings):
    st.subheader('Notebook execution (automated run)')
    st.dataframe(timings, hide_index=True, width='stretch')
    if run['status_file'].get('note'):
        st.info(run['status_file']['note'])
    if run['status_file'].get('failed_step'):
        st.error(f"Failed at: {run['status_file']['failed_step']}. {run['status_file'].get('error', '')}")

with st.expander('Run details'):
    st.write({'run folder': str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir),
              'software versions': s['versions']})
