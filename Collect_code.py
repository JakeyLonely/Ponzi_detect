from bs4 import BeautifulSoup
import collections
import Levenshtein
from urllib.parse import urljoin
import requests
from sklearn.model_selection import train_test_split
import re
import lightgbm as lgb
import pandas as pd
import numpy as np
import time
start = time.clock()
data1 = []
label = []
headers = {'Connection':'keep-alive',
           'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.108 Safari/537.36',
           #'Host':'maoyan.com'
          }
info_list = []
opcode_list = []
normal_list = []
inner_list = []
i = '0xf41624c6465e57a0dca498ef0b62f07cbaab09ca'
def delete(li, index):
    li = li[:index] + li[index+1:]
    return li
def get_response(url):
    #response = requests.get(url,headers = headers)
    response = requests.get(url)
    if response.status_code == 200:
        response.encoding = 'utf8'
        return response.text
    return None

def get_byte(i):#爬取字节码信息并计算字节码相似度
    Ai_data=pd.read_csv(open(r'D:\Thinkpad文件备份\区块链数据分析\TByteData.csv')
    ,usecols=[1,2],encoding = "utf-8")
    max_value = 0.0
    data = np.array(Ai_data)
    data = data.tolist()
    url = 'https://etherscan.io/address/'+ str(i) +'#code'
    html = get_response(url)
    bsObj = BeautifulSoup(html,features="html.parser")
    byte_list = bsObj.findAll(id = 'verifiedbytecode2')
    #temp = []
    #temp.append(i)

    if i in data:    
        data1 = delete(data,data.index(i))
    else:
        data1 = data
    for j in data1:
        similarRate = Levenshtein.ratio(str(byte_list[0].get_text()),j[1])
        if similarRate > max_value:
            max_value = similarRate
    #temp.append(max_value)
    #info_list.append(temp)
    return max_value
def get_opcode(i):#爬取指定智能合约的操作码并清洗
    Ai_data=pd.read_csv(open(r'D:\Thinkpad文件备份\区块链数据分析\AI data4.csv'),encoding = "utf-8")
    labels = list(Ai_data.columns.values)
    labels.remove('Contract')
    labels.remove('Ponzi')
    labels.remove('Similarty')
    url = 'https://etherscan.io/api?module=opcode&action=getopcode&address=' + str(i)
    html = get_response(url)
    bsObj = BeautifulSoup(html,features="html.parser")
    temp = []
    temp.append(i)
    temp.append(bsObj)
    opcode_list.append(temp)
    dict1 = dict()
    a = collections.Counter(re.findall('[A-Z][A-Z]+',str(bsObj)))
    if len(a):
        dict1['address']=i
        for key in a:
            if key == 'OK':
                continue
            else:
                dict1[key]=a[key]
    demo_x = []
    for j in labels:
        if  j  in dict1:
            demo_x.append(dict1[j])
            #print(dict1[j])
        else:
            demo_x.append(0)
    return demo_x
def trans_info(i):#爬取指定智能合约的交易记录并清洗
    url1 = 'http://api.etherscan.io/api?module=account&action=txlist&address='+ str(i) + '&startblock=0&endblock=99999999&sort=asc&apikey=YourApiKeyToken'
    html1 = get_response(url1)
    bsObj1 = BeautifulSoup(html1,features="html.parser")
    url2 = 'http://api.etherscan.io/api?module=account&action=txlistinternal&address=' + str(i) +'&startblock=0&endblock=2702578&sort=asc&apikey=YourApiKeyToken'
    html2 = get_response(url2)
    bsObj2 = BeautifulSoup(html2,features="html.parser")
    N_pay = 0
    N_maxpay = 0
    N_inv = 0
    V_pay = 0
    V_inv = 0
    bal = 0
    paid_rate = 0
    trans_list = []
    temp = []
    temp.append(i)
    bsObj1 = eval(str(bsObj1))
    bsObj2 = eval(str(bsObj2))
    pay_adr = set()
    if bsObj1['result']:
        for i in bsObj1['result']:
            if int(i['value']) > 0:
                N_inv = N_inv + 1    #合约投资总次数
                V_inv = V_inv + int(i['value'])   #合约投资总金额
            if i['to'] not in pay_adr:
                pay_adr.add(i['to'])        #所有收到回报的地址
    inv_adr = set()
    if bsObj2['result']:
        for i in bsObj2['result']:
            N_pay = N_pay + 1
            V_pay = V_pay + int(i['value'])      #合约投资金额数
            if N_maxpay<=int(i['value']):
                N_maxpay=int(i['value'])       #投资者最大一笔回报
            if i['from'] not in inv_adr:
                inv_adr.add(i['from'])   #所有投资者   
    bal = V_inv - V_pay #合约余额
    count = 0
    for i in inv_adr:
        if i in pay_adr:
            count = count + 1
    if len(inv_adr):
        paid_rate = count/len(inv_adr)     #投资者收到至少一笔回报的比例
    else:
        paid_rate = 0
    bal = bal/1000000000000000000            #将单位改成以太
    N_maxpay = N_maxpay/1000000000000000000
    temp.append(bal)                #
    temp.append(N_maxpay)
    temp.append(N_inv)    #投资总次数
    temp.append(N_pay)    #付款总次数
    temp.append(paid_rate)
    trans_list.append(temp)
    return trans_list
def detect(i):
    Ai_data=pd.read_csv(open(r'D:\Thinkpad文件备份\区块链数据分析\AI data4.csv'),encoding = "utf-8")
    data=Ai_data.iloc[:,2:]
    temp = np.array(Ai_data['Ponzi'])
    train_x, test_x, train_y, test_y = train_test_split(data, temp, test_size=0.2,random_state = 9)
    demo_x = get_opcode(i)
    demo_x.append(get_byte(i))    
    demo_y = []
    demo_y.append(demo_x)
    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_eval = lgb.Dataset(test_x, test_y, reference=lgb_train)
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss', 'auc'},
        'num_leaves': 13,
        'max_depth': 17,
        'learning_rate': 0.059,
        'bagging_fraction': 0.7,   
        'seed':100,
        'lambda_l1': 0,  
        'nfold':5,   
        'verbose': 5,
        'scale_pos_weight':310  #不同类别权重比值
    }
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10000,   #最大运行轮次
                    valid_sets=lgb_eval,
                    early_stopping_rounds=100 #最早停止轮次，连续几轮无优化则提前停止
                    )
    
    ypred = gbm.predict(demo_y, num_iteration=gbm.best_iteration)
    y_pred = (ypred >= 0.5)*1
    if y_pred >0.5:
        print('this is  a scam')
    else:
        print('this is not a scam')
    return y_pred
def summer(i):
    demo = ['address','bal','N_maxpay','N_inv','N_pay','paid_rate','Simliarty','Ponzi']
    test2 = []
    test = []
    #for j in trans_info(i):
    #    for k in detect(i):
    #        j.append(k)
    #    test.append(j)
    #test2.append(demo)
    #test2.append(test)
    test = ['0x258d778e4771893758dfd3e7dd1678229320eeb5','0','10','35','27','0.666666667','0.876453488','1']
    test2.append(demo)
    test2.append(test)
    return test2
def getTransData(i):
    #Ai_data=pd.read_csv(open(r'D:\Thinkpad文件备份\区块链数据分析\AI data4.csv'),encoding = "utf-8")
#    labels = list(Ai_data.columns.values)
#    labels.remove('Contract')
#    labels.remove('Ponzi')
#    labels.remove('Similarty')
    #count = 2
    url = 'https://etherscan.io/txs?a=' + str(i)
    html = get_response(url)
    bsObj = BeautifulSoup(html,'lxml')
    #byte_list = bsObj.findAll( table class="table table-hover")
   # data1 = bsObj.select('address')
   # print(html)
    #print(bsObj)
    #bsObj = eval(str(bsObj)) 
    data1 = bsObj.find(name = 'tbody')
    print(data1)
    
#    while(html):
#        html = get_response(url)
#        url = 'https://etherscan.io/txs?a=' + str(i) + '&p=' + str(count)
#        bsObj = BeautifulSoup(html,features="html.parser")
#        count = count + 1
#       # bsObj = eval(str(bsObj))
#    print(count)
    #bsObj2 = eval(str(bsObj2))
#    temp = []
#    temp.append(i)
#    temp.append(bsObj)
#    opcode_list.append(temp)

#    demo_x = []
#    for j in labels:
#        if  j  in dict1:
#            demo_x.append(dict1[j])
#            #print(dict1[j])
#        else:
#            demo_x.append(0)
#    return demo_x
#print(get_opcode(0xe82719202e5965cf5d9b6673b7503a3b92de20be))
#getTransData(i)
print(get_opcode(i))
#print(get_byte(i))  
#print(trans_info(i))
#print(summer(i))
#info_pd = pd.DataFrame(info_list,columns=['address','byte'])
#info_pd.to_csv(r'C:\Users\小强.DESKTOP-077EIE0\Desktop\区块链数据分析\ByteData.csv')
#print(detect(i))
print((time.clock() - start))
