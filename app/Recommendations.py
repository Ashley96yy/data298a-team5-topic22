"""Demo interface: pick products (or load a real user's history) and see live top-10 recommendations.

Run locally from the repository root:   streamlit run app/Recommendations.py
Recommendations are computed live from small artifacts built from the frozen dataset (see src/serving/recommend.py).
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.serving.recommend import Recommender  # noqa: E402

st.set_page_config(page_title='Skin care recommender', layout='wide')


@st.cache_resource
def load_recommender():
    return Recommender(ROOT / 'app' / 'artifacts')


rec = load_recommender()
if 'history' not in st.session_state:
    st.session_state.history = []


def add_item(item_idx):
    if item_idx not in st.session_state.history:
        st.session_state.history.append(int(item_idx))


def remove_item(item_idx):
    st.session_state.history = [i for i in st.session_state.history if i != item_idx]


def load_example():
    row = rec.examples.sample(1).iloc[0]
    st.session_state.history = [int(i) for i in row['train_items']]
    st.session_state.example_user = int(row['user_idx'])


st.title('Personalised skin-care recommendations')
st.caption('Team 5, Topic 22: Personalization and Recommendation Under Privacy Constraints. '
           'Scores are computed live from the frozen Amazon Reviews 2023 Skin Care dataset '
           '(56,017 users, 25,903 products).')

with st.sidebar:
    st.header('Settings')
    method = st.radio('Recommender', rec.methods, index=rec.methods.index('markov'),
                      help='popularity: most bought products. markov: what people most often buy right after your last product.')
    k = st.slider('Number of recommendations', 3, 20, 10)
    st.button('Load a real example user', on_click=load_example)
    st.button('Clear my history', on_click=lambda: st.session_state.update(history=[]))

left, right = st.columns([1, 1.4])
with left:
    st.subheader('1. Pick products you have bought')
    query = st.text_input('Search by name or brand', placeholder='for example: serum, CeraVe, sunscreen')
    for item_idx, row in rec.search(query, limit=8).iterrows():
        cols = st.columns([5, 1])
        cols[0].write(f"**{row['title'][:90]}**  \n{row['brand']} · {row['cat_l2']}")
        cols[1].button('Add', key=f'add_{item_idx}', on_click=add_item, args=(item_idx,))

    st.subheader('Your history (oldest to newest)')
    if not st.session_state.history:
        st.info('Nothing picked yet. Search above, or load a real example user in the sidebar.')
    for item_idx in st.session_state.history:
        cols = st.columns([5, 1])
        cols[0].write(rec.items.loc[item_idx, 'title'][:90])
        cols[1].button('Remove', key=f'rm_{item_idx}', on_click=remove_item, args=(item_idx,))

with right:
    st.subheader('2. Recommendations')
    history = st.session_state.history
    if not history:
        st.caption('With no history the system shows the most popular products (cold start).')
    result = rec.recommend(history, method if history else 'popularity', k)
    table = result.rename(columns={'cat_l2': 'category', 'price_num': 'price'})[['rank', 'title', 'brand', 'category', 'price']]
    table['price'] = table['price'].map(lambda p: f'${p:.2f}' if p == p else 'n/a')   # NaN means no price in the data
    st.dataframe(table, hide_index=True, width='stretch', column_config={
        'rank': st.column_config.NumberColumn('rank', width='small'),
        'title': st.column_config.TextColumn('title', width='medium'),
        'price': st.column_config.TextColumn('price', width='small'),
    })
    st.caption(f'Method: {method if history else "popularity"} · history length: {len(history)}')
