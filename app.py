import html
import json
import pandas as pd
import streamlit as st
from modelwatch.demo import datasets
from modelwatch.engine import analyze, evaluate, read_csv, serialize
from modelwatch.store import clear, history, save_run

st.set_page_config(page_title='ModelWatch', page_icon='◉', layout='wide')
st.markdown('''<style>
.stApp {background:#17161d;color:#efedf5} [data-testid="stSidebar"] {background:#211f29;border-right:1px solid #383442}
h1,h2,h3 {letter-spacing:-.035em} h1 {font-weight:800} [data-testid="stMetric"] {background:#24222e;padding:18px;border:1px solid #393443;border-radius:12px}
.stButton button[kind="primary"] {background:#a349f3;border:0;color:white} .eyebrow {color:#bd83f7;font-size:.75rem;font-weight:800;letter-spacing:.18em}
.caption {color:#b4afc0} .banner {border-left:3px solid #a349f3;padding:12px 16px;background:#25212e;border-radius:4px}
</style>''', unsafe_allow_html=True)
with st.sidebar:
    st.markdown('## ◉ ModelWatch')
    st.caption('OBSERVE / DETECT / UNDERSTAND')
    page = st.radio('Workspace', ['Overview', 'Datasets & analysis', 'Run history', 'Methodology'], label_visibility='collapsed')
    st.divider()
    st.caption('BATCH MONITORING  •  v0.1')
    st.caption('Data drift is a signal, not proof of model degradation.')

st.markdown('<div class="eyebrow">MODEL OBSERVABILITY / WORKSPACE</div>', unsafe_allow_html=True)
if page == 'Datasets & analysis':
    st.title('Make the shift visible.')
    st.caption('Compare two datasets, investigate changes, and evaluate predictions when labels are available.')
    source = st.radio('Data source', ['Synthetic demo', 'Upload CSV'], horizontal=True)
    if source == 'Synthetic demo':
        ref, cur = datasets()
        st.info('Synthetic demo data — generated locally with a fixed seed.')
    else:
        a, b = st.columns(2)
        with a: reference_file = st.file_uploader('Reference CSV', type='csv')
        with b: current_file = st.file_uploader('Current CSV', type='csv')
        if not reference_file or not current_file:
            st.info('Upload both CSV files to continue. Limit: 20 MB per file.')
            st.stop()
        try:
            ref, cur = read_csv(reference_file.getvalue()), read_csv(current_file.getvalue())
        except ValueError as exc:
            st.error(str(exc)); st.stop()
    missing = sorted(set(ref.columns)-set(cur.columns)); added = sorted(set(cur.columns)-set(ref.columns))
    a,b = st.columns(2)
    with a:
        st.subheader('Reference'); st.caption(f'{len(ref):,} rows · {len(ref.columns)} columns'); st.dataframe(ref.head(10), use_container_width=True)
    with b:
        st.subheader('Current'); st.caption(f'{len(cur):,} rows · {len(cur.columns)} columns'); st.dataframe(cur.head(10), use_container_width=True)
    if missing or added: st.warning(f'Schema changes — missing from current: {missing or "none"}; added: {added or "none"}.')
    common = [x for x in ref.columns if x in cur.columns]
    default = [x for x in common if x.lower() not in ('actual','target','prediction','id')]
    selected = st.multiselect('Feature columns', common, default=default, help='Exclude identifiers, targets and predictions unless explicitly analyzing their distributions.')
    alpha = st.slider('Statistical significance threshold', .001, .2, .05, .001)
    with st.expander('Optional model evaluation'):
        task = st.selectbox('Task', ['None', 'classification', 'regression'])
        target = st.selectbox('Actual label column', common, index=None, placeholder='Select a column')
        pred = st.selectbox('Prediction column', common, index=None, placeholder='Select a column')
    if st.button('Run analysis ↗', type='primary'):
        try:
            result = analyze(ref, cur, selected, alpha)
            result['source'] = 'synthetic demo' if source == 'Synthetic demo' else 'uploaded CSV'
            result['evaluation'] = evaluate(cur, target, pred, task) if task != 'None' else None
            result['warnings'] = ['Drift does not prove performance degradation.', 'Results are affected by sample size; p-values are not corrected for multiple comparisons.']
            save_run(result)
            st.session_state['latest'] = result
            st.success('Analysis complete and summary saved to local history.')
        except (ValueError, TypeError) as exc: st.error(str(exc))
    result = st.session_state.get('latest')
    if result:
        st.subheader('Latest analysis')
        a,b,c = st.columns(3)
        a.metric('Features analyzed', f"{result['analyzed']} / {result['selected']}")
        b.metric('Drifted features', f"{result['drift_pct']:.1f}%" if result['drift_pct'] is not None else 'Unavailable')
        c.metric('Current missing values', f"{result['current_quality']['missing_pct']:.1f}%")
        if not result['coverage_complete']: st.warning('Some selected features could not be analyzed; overall drift is incomplete.')
        st.dataframe(pd.DataFrame(result['features']), use_container_width=True, hide_index=True)
        st.subheader('Data quality')
        st.dataframe(pd.DataFrame(result['current_quality']['features']), use_container_width=True, hide_index=True)
        if result['evaluation']:
            st.subheader('Model evaluation'); st.json(result['evaluation'])
        else: st.info('Model performance is unavailable without actual labels and predictions. Drift alone cannot establish degradation.')
        st.download_button('Download JSON report', serialize(result), file_name='modelwatch-report.json', mime='application/json')
        st.download_button('Download feature CSV', pd.DataFrame(result['features']).to_csv(index=False), file_name='modelwatch-features.csv', mime='text/csv')
        report = '<!doctype html><meta charset="utf-8"><title>ModelWatch report</title><h1>ModelWatch report</h1><p>Generated from analysis output. See warnings and methodology below.</p><pre>' + html.escape(serialize(result)) + '</pre>'
        st.download_button('Download HTML report', report, file_name='modelwatch-report.html', mime='text/html')
elif page == 'Overview':
    st.title('Know when things shift.')
    st.markdown('<div class="banner">A clearer view of your data, one monitoring run at a time.</div>', unsafe_allow_html=True)
    runs = history()
    if not runs:
        st.subheader('Your workspace is quiet.')
        st.write('Start with the synthetic demo or upload reference and current CSV files in **Datasets & analysis**. No metrics are invented.')
    else:
        r = runs[0][2]; a,b,c = st.columns(3)
        a.metric('Monitoring runs', len(runs)); b.metric('Latest drifted features', f"{r['drift_pct']:.1f}%" if r['drift_pct'] is not None else 'Unavailable'); c.metric('Current missing values', f"{r['current_quality']['missing_pct']:.1f}%")
        if not r['coverage_complete']: st.warning('Latest drift result has incomplete feature coverage.')
        st.subheader('Latest feature results'); st.dataframe(pd.DataFrame(r['features']), hide_index=True, use_container_width=True)
        st.caption('Latest source: ' + r['source'])
elif page == 'Run history':
    st.title('Monitoring history')
    runs = history()
    if not runs: st.info('No saved runs yet.')
    for run_id, date, result in runs:
        with st.expander(f"Run #{run_id} · {date} · {result.get('source', 'unknown')}"):
            st.dataframe(pd.DataFrame(result['features']), hide_index=True, use_container_width=True)
            st.download_button('Download report', serialize(result), file_name=f'modelwatch-{run_id}.json', key=f'dl{run_id}')
    if runs and st.checkbox('I understand this deletes all saved summaries'):
        if st.button('Delete history'): clear(); st.rerun()
else:
    st.title('How to read the signals')
    st.write('Numeric features use a two-sample Kolmogorov–Smirnov test. Categorical features use a chi-square test. A p-value below the chosen threshold is flagged; small expected counts and incomplete coverage are reported.')
    st.write('Statistical significance depends on sample size. No correction for multiple comparisons is applied. Inspect feature distributions and practical impact before making decisions.')
    st.write('Classification metrics are weighted by class support. Regression uses MAE, MSE, RMSE and R² when defined. Missing label/prediction pairs are excluded and counted.')
    st.warning('Data drift, prediction drift, and model performance are different concepts. Drift does not establish performance degradation.')
