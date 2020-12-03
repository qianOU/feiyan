# -*- coding: utf-8 -*-
from gevent import monkey; monkey.patch_all()
import gevent
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from pyquery import pyquery as pq
import re
from collections import Counter
from lxml import etree
from  concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")



#python -W ignore yourscript.py
data = pd.read_csv(r'D:\作业\非参数统计\期末\MedMNIST\mygan\USstate_count.csv', encoding='utf-8-sig')

data.date1 = data.date1.apply(lambda x: datetime.strptime(x, '%d%b%Y'))

data2 = data.loc[data.date1>datetime(2013, 12, 31)]
data2 = data2.sort_values(['CIK', 'type'])
data2 = data2.drop_duplicates('CIK', keep='first')

areas = data2.columns[4:]

#新数据覆写data2 需要CIK字段
data2 = pd.read_csv(r'D:\作业\非参数统计\期末\MedMNIST\mygan\new_cik_list.csv',encoding='utf-8-sig')





timeout = 30


class askexception(Exception):
    def __init__(self, *args):
        super(askexception, self).__init__(*args)
        


def choice_table(temp:list):
    dates = [item["_source"]["file_date"][:4] for item in temp]
    types = [item["_source"]["form"] for item in temp]
    years = [int(i) for i in dates]
    one = pd.DataFrame(list(zip(years, types)), columns = ['year', 'type'])
    one = one.drop_duplicates(['year', 'type'], keep='first')
    #在前的年份得1分
    prior_year = np.array([0] * one.shape[0])
    year_order = one.year.values.tolist()
    for i in set(one.year.values):
        prior_year[year_order.index(i)] = 1
    #type为 10-k的得2分
    prior_type  = (np.array(one.type) == '10-K').astype(int) *2 
    #组成得分形式
    priority = prior_type + prior_year
    one = one.copy()
    one.loc[:,'prior'] = priority
    one.reset_index(inplace=True)
    two = [tuple(i) for i in one.groupby('year')['prior'].max().reset_index().values]
    condition = one.apply(lambda x:tuple(x[['year', 'prior']].values) in two , axis=1).values
    assert sum(condition) == sum(prior_year) #确保每一年取出一个
    indexes = one['index'][condition].values
    
    return [temp[i] for i in indexes]
    
# In[查询接口]
def ask_page(cik):
    # cik2 = cik
    cik = '{:0>10d}'.format(cik)
    headers = {
        'authority': 'efts.sec.gov',
        'accept': 'application/json, text/javascript, */*; q=0.01',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.106 Safari/537.36',
        'content-type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'origin': 'https://www.sec.gov',
        'sec-fetch-site': 'same-site',
        'sec-fetch-mode': 'cors',
        'sec-fetch-dest': 'empty',
        'referer': 'https://www.sec.gov/edgar/search/',
        'accept-language': 'zh-CN,zh;q=0.9',
    }
    
    data = '{"dateRange":"custom","category":"custom","ciks":["%s"],"entityName":"AAR CORP (AIR) (CIK %s)","startdt":"2015-01-01","enddt":"2020-11-01","forms":["10-K"]}' %(cik, cik)
    response = requests.post('https://efts.sec.gov/LATEST/search-index', headers=headers, data=data, timeout=timeout)
    try:
        temp = response.json()['hits']['hits']
    except KeyError:
        # print(cik2, response.text)
        raise askexception('查询错误')
    #若查询到报表,选取需要的temp
    if len(temp) > 0:
        temp = choice_table(temp)
    return temp



def get_document(cik):
    try:
        answer = ask_page(cik)
    except AssertionError:
        print('获取的报表数目不对！')
        assert 1==0 
    except askexception:
        return 0
    print(cik, '查询成功！')
    cik2 = '{:0>10d}'.format(cik)
    first_url = 'https://www.sec.gov/Archives/edgar/data/%s/' % cik2
    
    detail_ = []
    if len(answer) == 0:
        # print('{} 没有报表！'.format(cik))
        return None
    else:
        for item in answer:
            type_ = item["_source"]['form']
            time1 = item["_source"]["file_date"]
            head_, tail = item['_id'].split(':')
            page = head_.replace('-', '') + '/'+ tail
            detail_.append({'date1':time1, 'url':first_url+page, 'type':type_})
    
    df = pd.DataFrame(detail_)
    return df
    

#计数
def word_count(text):
    result = dict.fromkeys(areas, 0)
    words = re.findall(r'\w+', text).lower()
    words = Counter(words)
    for i, I in zip(areas, [i for i in areas]):
        result[i] = words.get(I, 0)
    return result



def get_detail(df, cik):
    result = df.date1.to_frame().copy()
    
    info = list()
    for i, (date, url, type_) in enumerate(df.values):
        headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.106 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            print('{}的{}的报表获取成功!'.format(cik, date))
            if '<?xml' in response.text[:400]:
                xml = etree.fromstring(response.content)
                text = etree.tostring(xml, encoding='unicode')
            else:
                text = str(pq.PyQuery(response.text).text())
            temp = word_count(text)
            temp['date1'] = date
            temp['link'] = url
            temp['type'] = type_
            one = pd.DataFrame(temp, index=[i])
            info.append(one)
    if len(info)>0:
        agg = pd.concat(info, axis=0)
        answer = pd.merge(result, agg, how='left', on='date1')
        return answer
    return cik

# for cik in data3.CIK:
#     df = get_document(cik)
#     if df is None:
#         print('----------%d完成！-----------' % cik)
#         continue
#     answer = get_detail(df, cik)
#     answer['CIK'] = cik
#     answer['type'] = '10-K'
#     time.sleep(random.random())
#     print('----------%d完成！-----------' % cik)
    # for i in areas:
    #     result[i] = len(re.findall(i, text))
    
    
def map_main(cik):
    try:
        df = get_document(cik)
    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
        print('-----------%d查询失败！---------------'  % cik)
        return cik
    if isinstance(df, int):
        print( '-----------%d查询失败！---------------'  % cik)
        return cik
    if df is None:
        print('----------%d（无报表）完成！-----------' % cik)
        return None
    try:
        answer = get_detail(df, cik)
        #若详细页查询没有结果
        if isinstance(answer, int):
            return cik
    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
        print('-----------%d详细内容，查询失败！---------------'  % cik)
        return cik
    answer['CIK'] = cik
    # time.sleep(random.random())
    print('----------%d（有报表）完成！-----------' % cik)
    return answer


if __name__ == '__main__':
    store = pd.DataFrame(columns=data.columns)
    data3 = data2
    threshold = data3.shape[0]
    wanted = data3.CIK.values.tolist()
    round_ = 0
    no_tablbes = [] #没有报表的集合
    # with ThreadPoolExecutor(max_workers=5) as executor:
    while len(wanted):
        round_ += 1
        length = len(wanted)
        # pool = ProcessPoolExecutor(4) #必须在if __name__ == '_main__':下运行
        print(len(wanted))
        futures = [gevent.spawn(map_main, cik) for cik in wanted]
        gevent.joinall(futures)
        results = [future.value for future in futures]
        # results = .
        assert len(results) == len(wanted) #结果个数与查询个数相等
        result = [i for i in results if isinstance(i, pd.DataFrame)]
        store = pd.concat([store, *result], axis=0)
        store = store.apply(lambda x:pd.to_numeric(x, errors='ignore'), axis=0)
        useless = [wanted[i] for i in range(len(wanted)) if results[i] is None]
        no_tablbes.extend(useless)
        wanted = [i for i in results if isinstance(i, int)]
        
        print('已完成第%d轮,共%d个需要查询，成功查询%d个，需要重新查询%d个, 退市公司%d个' %(round_, length, 
                                                     len(result), len(wanted), len(useless)))
        
        #控制循环，强行退出
        if round_ > (threshold//2+10):
            break
    ######################################################
    #保存位置
    store.to_excel(r'D:\store.xlsx', index=False) #每次需要
    #无报表的cik集合
    ######################################################
    with open(r'D:\no_talbes.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(str(i) for i  in no_tablbes))
