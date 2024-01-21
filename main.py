import random
random.seed(0)

import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib

from st_config import setting
setting()
from sidebar import selectors
from _const import (
    M_GAS_LEAK_KUBUN,
    M_SINDAN_KEKKA
)

if 'meta_button' not in st.session_state:
    st.session_state.meta_button = False
if 'accuracy_button' not in st.session_state:
    st.session_state.accuracy_button = True

def download_df(df: pd.DataFrame, start, end):
    file_name = f'gas_leak_{start}_{end}.csv'
    if isinstance(df, pd.DataFrame) and df.empty == False:
        st.download_button(
            label='csvファイルに保存',
            data=convert(df),
            file_name=file_name,
            mime='text/csv',
            use_container_width=True
        )


@st.cache_data
def convert(df: pd.DataFrame):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('cp932')


def button_calc_accuracy_content():
    st.session_state.accuracy_button = not st.session_state.accuracy_button
    st.session_state.meta_button = False


def main():
    customer, start_date, end_date, selecter_shop = selectors()

    data = []
    devi = (end_date -start_date).days
    if devi <= 0:
        st.warning('開始日が終了日よりも後になっています')
    else:
        for _ in range(int(devi*0.7)):
            uketuke_date = dt.datetime(start_date.year, start_date.month, start_date.day) + dt.timedelta(days=random.randint(0, devi))
            start_time = uketuke_date + dt.timedelta(hours=random.randint(1, 24))
            end_time = start_time + dt.timedelta(hours=random.randint(1, 8))
            data.append([
                uketuke_date,
                random.choice([f"{customer}_{i}" for i in range(1,11)]),
                random.choice(list(M_GAS_LEAK_KUBUN.values())),
                start_time,
                end_time,
                random.choice(list(M_SINDAN_KEKKA.values())),
                random.choice([i for i in range(1,11)]),
            ])

        repair_info = pd.DataFrame(data, columns=['修理受付日', '店舗', '漏えい箇所', '作業開始時間', '作業終了時間', '診断結果', '検知速度(days)'])
        if selecter_shop is not None:
            repair_info = repair_info[repair_info['店舗']==selecter_shop]
        st.header(f'AI診断の効果・精度確認のためのサンプル画面')
        _bool_1 = st.toggle('表示',value=False, key='effect')
        if _bool_1:
            st.subheader(f'修理履歴: {start_date} ~ {end_date}')
            st.data_editor(repair_info)
            #########################################
            col_paragraph_2_2, col_paragraph_2_3, _ = st.columns([4,4,2])
            with col_paragraph_2_2:
                accuracy_button = st.button(label='精度算出',use_container_width=True)
            with col_paragraph_2_3:
                download_df(repair_info, start_date, end_date)
            if accuracy_button:
                button_calc_accuracy_content()
            if st.session_state.accuracy_button:
                calc_accuracy(repair_info)
        #########################################
        st.header('センサーデータの時系列グラフを表示しアノテーション行うサンプル画面')
        _bool_2 = st.toggle('表示',value=False, key='sensor')
        if _bool_2:
            generate_sensor_data(start_date, end_date)


def generate_sensor_data(start_date, end_date):
    date_range = pd.date_range(start_date, end_date, freq='H')

    # ダミーデータの生成
    data2 = {
        'Sensor1': np.random.normal(10, 2, len(date_range)),
        'Sensor2': np.random.normal(5, 1.5, len(date_range)),
        'Sensor3': np.random.normal(3, 1, len(date_range)),
        'Sensor4': np.random.normal(1, 0.5, len(date_range)),
        'Sensor5': np.random.normal(0, 0.2, len(date_range)),
    }
    df = pd.DataFrame(data2, index=date_range)

    min_date = df.index.min().to_pydatetime()
    max_date = df.index.max().to_pydatetime()
    start_date = max_date - dt.timedelta(days=5)
    end_date = min_date + dt.timedelta(days=5)
    if start_date < min_date:
        start_date = min_date
    if end_date > max_date:
        end_date = max_date

    selected_normal_period = st.slider(
        "正常期間の選択",
        min_value=min_date,
        max_value=max_date,
        value=(start_date, max_date)
    )

    selected_abnormal_period = st.slider(
        "異常期間の選択",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, end_date)
    )

    # 特徴量の選択
    options = st.multiselect(
        label='表示するセンサーデータを選択',
        options=df.columns.tolist(),
        default=['Sensor1', 'Sensor2', 'Sensor3']
    )

    # グラフの表示
    if options:
        fig = px.line(df, y=options, title='センサーデータの時系列グラフ')
        fig.update_layout(width=1200, height=600)
        # 選択された期間の背景色を変更
        fig = fig_add(fig, selected_normal_period, "LightSkyBlue")
        fig = fig_add(fig, selected_abnormal_period, "rgba(255, 0, 0, 0.6)")
        st.plotly_chart(fig)


def fig_add(fig, selected_period, color):
    fig.add_shape(
        # 長方形の追加
        type="rect",
        x0=selected_period[0], y0=0,
        x1=selected_period[1], y1=1,
        xref='x', yref='paper',
        fillcolor=color,  # 背景色
        opacity=0.5,
        layer="below",
        line_width=0,
    )
    return fig

def calc_accuracy(df):
    tp = df[(df['漏えい箇所']!='正常') & (df['診断結果']=='漏えい')].shape[0]
    fp = df[(df['漏えい箇所']=='正常') & (df['診断結果']=='漏えい')].shape[0]
    fn = df[(df['漏えい箇所']!='正常') & (df['診断結果'].isin(['漏えいなし','その他異常']))].shape[0]
    tn = df[(df['漏えい箇所']=='正常') & (df['診断結果'].isin(['漏えいなし','その他異常']))].shape[0]
    recall = tp / (tp + fn) * 100
    precision = tp / ( tp + fp) * 100
    class_names = ["Yes(True)", "No(False)"]
    retu = np.array([tp, fn, fp, tn]).reshape(2,2)
    # Matplotlibを使って混同行列を表示
    fig, ax = plt.subplots(figsize=(10,1))
    sns.heatmap(retu, annot=True, ax=ax, cmap='Blues', fmt='g', xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('予測が冷媒漏えい')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_ylabel('実際が冷媒漏えい')
    col_calc_accuracy1, col_calc_accuracy2, col_calc_accuracy3, col_calc_accuracy4 = st.columns(4)
    col_calc_accuracy1.metric('再現率(実際の漏えいを検知できる確率)',f"{recall:.1f}%")
    col_calc_accuracy2.metric('適合率(漏えい検知の結果が本当に漏えいである確率)',f"{precision:.1f}%")
    if recall != 0 and precision != 0:
        col_calc_accuracy3.metric('f1スコア(再現率と適合率を加味した総合的な指標)',f"{2*precision*recall/(precision+recall):.1f}%")
        col_calc_accuracy4.metric('修理受付日と漏えい検知日の差', f"{df['検知速度(days)'].mean():.1f}日")

    st.pyplot(fig)

if __name__=='__main__':
    main()