import os
import datetime as dt
from typing import List,Dict,Tuple,Union

# from pycaret.anomaly import *
import seaborn as sns
import matplotlib.pyplot as plt
# import japanize_matplotlib
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np


from _const import (
    MIN_DATE,
    COLUMNS,
    M_GAS_LEAK_KUBUN
)




@st.cache_data
def append_gas_leak_kubun_mei(gas_leak_kubun_id):
    kubun = M_GAS_LEAK_KUBUN.copy()
    kubun = kubun[kubun['id']==gas_leak_kubun_id].reset_index(drop=True)

    return (None if kubun.empty else kubun['gas_leak_kubun_mei'].iloc[0],)


@st.cache_data
def get_gas_leak_data(name: str, start: dt.date, end: dt.date) -> Tuple[pd.DataFrame, Dict[str,str]]:
    """
    Get gas leak cases according to start and end
    The argument 'name' is used to cache the process
    
    :args1 name: Customer name 
    
    """
    print('get_gas_leak_data')
    if not df_merge.empty:
        df_merge[['speed','can']] = df_merge.apply(lambda row: append_sensing_speed(row['kansi_no'],row['uketuke_date'],row['reitouki_no'],row['uketuke_no']),axis=1,result_type='expand')
        df_merge[['gas_leak_kubun_mei']] = df_merge.apply(lambda row: append_gas_leak_kubun_mei(row['gas_leak_kubun_id'],row['gas_leak_detail_id']),axis=1,result_type='expand')
        df_merge.drop(columns=['gas_leak_kubun_id','gas_leak_detail_id'],inplace=True)
    return df_merge, meta_data


def shaping(df, kansi_no: List[int], kokyaku_id: List[int]):
    if kansi_no and kokyaku_id:
        df = df[(df['kansi_no'].isin(kansi_no)) | (df['kokyaku_id'].isin(kokyaku_id))]
    elif kansi_no:
        df = df[df['kansi_no'].isin(kansi_no)]
    elif kokyaku_id:
        df = df[df['kokyaku_id'].isin(kokyaku_id)]
    else:
        pass

    return df


@st.cache_data
def append_sensing_speed(kansi_no, uketuke_date, cc, uketuke_no) -> Tuple[Union[None, int],bool]:
    """
    calculate sensing speed and returned deviation ('int')
    The argument 'uketuke_no' is used to cache the process 
    """
    deviation, can = None, True
    _date = dt.date(uketuke_date.year, uketuke_date.month, uketuke_date.day)
    end = _date + dt.timedelta(days=5)
    res = get_sindankekka(kansi_no=kansi_no, cs=0, cc=cc, end=end, on=True, n = 13)
    if isinstance(res,bool):
        can = False
    else:
        target_df = res[(res['sindan_kekka'].str.contains('冷媒漏えい')) | (res['sindan_kekka'].str.contains('ガス漏れ')) | (res['sindan_kekka'].str.contains('冷媒漏洩'))]
        if not target_df.empty:
            deviation = (_date - target_df['sindan_taisyou_date'].min()).days

    return deviation, can


def custom_format_d(x):
    return 'None' if x == None else x


def custom_format_1f(x):
    return 'None' if x == None else f'{x:.1f}'


def draw_gas_leak_table(df: pd.DataFrame) -> pd.DataFrame:
    st.write(f'length: {len(df)}')
    with st.form('gas_leak_table'):
        edited_df = st.data_editor(
            data=df.rename(columns=COLUMNS).style.format({
                '監視番号':'{:d}',
                '検知速度':custom_format_d,
                'フロン充填量':custom_format_1f
            }),
            column_config={
                '診断可能':st.column_config.Column(required=True)
            }
        )
        st.form_submit_button(label='表の更新')
    edited_df.columns = df.columns

    return edited_df


@st.cache_data
def select_gl_meta_from(df: pd.DataFrame, uketuke_no: int) -> Tuple[pd._libs.tslibs.timestamps.Timestamp, np.int32, np.int32]:
    target_index = df[df['uketuke_no']==uketuke_no].index[0]
    kansi_no = df.loc[target_index, 'kansi_no']
    _date = df.loc[target_index,'uketuke_date']
    default_cs = df.loc[target_index,'kiki_no']
    default_cc = df.loc[target_index,'reitouki_no']

    return kansi_no, _date, default_cs, default_cc


@st.cache_data
def get_shop_meta_data(code: int, date: dt.datetime, set_file = False) -> Tuple[List[str], List[int]]:
    shop = Shop(code,date,set_file)
    cases = shop.showcases.copy()
    refrigerators = shop.refrigerators.copy()

    return cases, refrigerators


def draw_can_diagnose_table(df):
    col1_1,col1_2 = st.columns([2,8])
    with col1_1:
        shop_code = st.selectbox(label='店舗番号',options=df['SHOP_CODE'].unique(),index=None)
        # shop_name = st.text_input(label=('店舗名'),max_chars=100)
    with col1_2:
        if shop_code is not None:
            df = df[df['SHOP_CODE']==shop_code]
        st.dataframe(df.rename(columns=COLUMNS))



def paragraph_2(name, start, end, kansi_no_list, kokyaku_id_list) -> pd.DataFrame:
    """
    サイドバーで指定した条件(顧客、開始日、終了日、監視番号、顧客ID)の冷媒漏えい事例を確認するディスプレイ
    """
    df, meta_data = get_gas_leak_data(name,start, end)
    df = shaping(df,kansi_no_list, kokyaku_id_list)
    if not df.empty:
        edited_df = draw_gas_leak_table(df)
    else:
        st.success('Gas leak data is None')
        return df
    col_paragraph_2_1, col_paragraph_2_2, col_paragraph_2_3, _ = st.columns([3,3,3,1])
    with col_paragraph_2_1:
        meta_button = st.button(label='メタ情報',use_container_width=True)
    with col_paragraph_2_2:
        accuracy_button = st.button(label='精度算出',use_container_width=True)
    with col_paragraph_2_3:
        download_df(edited_df, start, end)
    if meta_button:
        button_meta_info_button_exe_content()
    if accuracy_button:
        button_calc_accuracy_content()
    if st.session_state.meta_button:
        of_call = len(df[(df['on_call']==False) & ~(df['flon_zyuten_ryou1'].isin([None]))])
        can_not = len(df[df['can']==False])
        meta_info_button_exe(meta=meta_data,of_call=of_call,can_not=can_not)
    elif st.session_state.accuracy_button:
        calc_accuracy(edited_df)

    return edited_df


def button_meta_info_button_exe_content():
    st.session_state.meta_button = not st.session_state.meta_button
    st.session_state.accuracy_button = False


def button_calc_accuracy_content():
    st.session_state.accuracy_button = not st.session_state.accuracy_button
    st.session_state.meta_button = False


def meta_info_button_exe(meta: dict, of_call: int, can_not: int):
    col_meta_info_button_exe1, col_meta_info_button_exe2 = st.columns([3,7])
    df = pd.DataFrame(meta)
    df = df[df.index==0].T.reset_index().rename(columns={0:'length','index':'meta_name'})
    df['length'] = df['length'] + of_call - can_not
    with col_meta_info_button_exe1:
        st.dataframe(df)
    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y('meta_name', title='Count',sort=alt.EncodingSortField(field='length', order='descending')),
            x=alt.X('length',title='length'),
            # color=alt.condition(click, color, alt.value('lightgray')),
        )
        .properties(width=550)
    )
    with col_meta_info_button_exe2:
        st.altair_chart(bars, theme=None, use_container_width=True)


def calc_accuracy(df: pd.DataFrame):
    df = df[df['can']==True]
    tp = len(df[(df['on_call']==True) & (df['speed'].isnull()==False)]) + len(df[(df['on_call']==False) & ~(df['flon_zyuten_ryou1'].isin([None]))])
    fp = len(df[(df['on_call']==False) & (df['flon_zyuten_ryou1'].isin([None]))])
    fn = len(df[(df['on_call']==True) & (df['speed'].isnull()==True)])
    recall = tp / (tp + fn) * 100
    precision = tp / ( tp + fp) * 100
    class_names = ["Yes(True)", "No(False)"]
    retu = np.array([tp, fn, fp, 0]).reshape(2,2)
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
        col_calc_accuracy4.metric('修理受付日と漏えい検知日の差', f"{df['speed'].mean():.1f}日")

    st.pyplot(fig)


def pragraph_3(df):
    """
    セレクトボックスにて指定された条件（受付番号、冷凍機番号、診断実施日）の診断結果を確認するディスプレイ
    """
    col2_1, col2_2, col2_3, _ = st.columns([2,2,2,2])
    with col2_1:
        uketuke_no = st.selectbox(
            label='受付番号',
            options=df['uketuke_no'],
            index=0
        )
    kansi_no, uketuke_date, default_cs, default_cc = select_gl_meta_from(df, uketuke_no)
    shop = Shop(kansi_no,uketuke_date)
    cases, refrigerators = get_shop_meta_data(kansi_no, uketuke_date, shop.set_file)
    with col2_2:
        cc = st.selectbox(
            label='冷凍機番号',
            options=refrigerators,
            index=refrigerators.index(default_cc) if default_cc in refrigerators else 0
        )
    with col2_3:
        target_date = st.date_input(
            label='診断実施日',
            min_value=MIN_DATE,
            max_value=dt.date.today(),
            value=dt.date(uketuke_date.year, uketuke_date.month, uketuke_date.day),
            format="YYYY-MM-DD"
        )
    use_col = [
        'kansi_no','showcase_kiki_no','sindan_taisyou_date','kisyumei','sindan_kekka','sindan1',
        'sindan2','sindan3','sindan4','sindan5','sindan6','sindan7','sindan8'
    ]
    sindan_kekka_bunch = get_sindankekka(kansi_no=kansi_no, cs = 0, cc=cc, end=target_date, on=True)
    on_sindan_kekka_bunch_display = st.toggle('Activate 診断結果リスト',value=False)
    if isinstance(sindan_kekka_bunch,pd.DataFrame):
        if on_sindan_kekka_bunch_display:
            edited_df = st.data_editor(
                data=sindan_kekka_bunch[use_col].rename(columns=COLUMNS),
                column_config={
                    '診断結果':st.column_config.Column(width='medium')
                }
            )
    else:
        edited_df = None
        st.warning(f'Data is None')

    return uketuke_no, kansi_no, cc, uketuke_date, shop, default_cs


def paragraph_4( kansi_no, cc, date, shop, default_cs, customer: Customer):
    render_dinamic_graph(kansi_no, cc, date, shop, default_cs, customer)


def get_founder():
    return Untrain()


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


def main(customer: Customer, start: dt.date, end: dt.date, kansi_no_list: List[int], kokyaku_id_list: List[int]):
    ########################
    print('-----start gas leak analysis-----')
    ########################

    if 'meta_button' not in st.session_state:
        st.session_state.meta_button = False
    if 'accuracy_button' not in st.session_state:
        st.session_state.accuracy_button = False
    if 'create_model_button' not in st.session_state:
        st.session_state.create_model_button = False
    df = paragraph_2(customer.name, start, end, kansi_no_list, kokyaku_id_list)
    if not df.empty:
        uketuke_no, kansi_no, cc, date, shop, default_cs = pragraph_3(df)
        if cc is not None:
            paragraph_4(kansi_no, cc, date, shop, default_cs, customer)


if __name__=='__main__':
    main()