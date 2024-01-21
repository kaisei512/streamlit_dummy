import os
import datetime as dt
import pathlib


ABS_PATH: os.path = str(pathlib.Path(__file__).resolve().parents[2])

TODAY = dt.date.today()
MIN_DATE = dt.date(2020,1,1)

VERSION_MAP = {'MV':'m_v_leak','SH':'other','FPC':'m_v_leak_FPC'}
COLUMNS = {
    'showcase_kiki_no':'ケース番号','sindan_taisyou_date':'診断対象日','kisyumei':'機種名',
    'sindan_kekka':'診断結果','sindan1':'診断1','sindan2':'診断2','sindan3':'診断3',
    'sindan4':'診断4','sindan5':'診断5','sindan6':'診断6','sindan7':'診断7','sindan8':'診断8',
    'uketuke_date':'受付日','kansi_no':'監視番号','kokyaku_mei':'顧客名','kiki_no':'機器番号',
    'kisyu_mei':'機種名','kiban':'製造番号','setubi_mei':'設備名','reitouki_no':'冷凍機番号',
    'uketuke_no':'受付番号','flon_zyuten_ryou1':'フロン充填量','speed':'検知速度',
    'sagyou_naiyou_houkokusyo_muke':'作業内容','sagyou_start_time':'作業開始時間','診断可能':'can',
    'sagyou_end_time':'作業終了時間','SHOP_CODE':'監視番号','KNO_CC':'冷凍機番号','on_call':'コールあり',
    'gas_leak_kubun_mei':'漏えい製品区分','gas_leak_detail_mei':'漏えい個所詳細',
}
M_GAS_LEAK_KUBUN = {
    1:'冷凍機本体',2:'ショーケース本体',3:'冷凍機配管',4:'ショーケース配管',5:'冷凍機配管部品',6:'正常'
}

BANKS = {
    1: '顧客A', 2: '顧客B', 3: '顧客C', 4: '顧客D', 5: '顧客E', 6: '顧客F', 7: '顧客G', 8: '顧客H', 9: '顧客I', 10: '顧客J'
}
M_SINDAN_KEKKA = {1:'漏えい',2:'漏えいなし',3:'その他異常'}



if __name__=='__main__':
    print(M_GAS_LEAK_KUBUN)
