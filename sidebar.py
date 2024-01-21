from typing import Tuple,List
import datetime as dt

import streamlit as st

from _const import (
    MIN_DATE,
    BANKS
)




def selectors() -> Tuple[dt.date, dt.date, List[int], List[int], bool]:
    today = dt.date.today()
    selecter_kokyaku = st.sidebar.selectbox(
        label='顧客',
        options=BANKS.values(),
        index=0
    )
    st.sidebar.subheader('修理受付期間の選択')
    start_date = st.sidebar.date_input(
        label='開始日',
        min_value=MIN_DATE,
        max_value=today,
        value=dt.date(today.year,today.month,1),
        format="YYYY-MM-DD"
    )
    end_date = st.sidebar.date_input(
        label='終了日',
        min_value=MIN_DATE,
        max_value=today,
        value=today,
        format="YYYY-MM-DD"
    )
    customer = selecter_kokyaku
    st.sidebar.header('店舗を選択')
    selecter_shop = st.sidebar.selectbox(
        label='店舗',
        options=[f"{selecter_kokyaku}_{i}" for i in range(1,11)],
        index=None
    )
    return customer, start_date, end_date, selecter_shop


