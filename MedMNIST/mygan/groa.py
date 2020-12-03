# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 22:22:57 2020

@author: 28659
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime

filename1 = 'FAR_Finidx.xlsx'
filename2 = 'AF_Actual.xlsx'

#读入数据
finidx = pd.read_excel(filename1,skiprows=[1, 2])
actual = pd.read_excel(filename2,skiprows=[1, 2])

#使用外连接将表格进行拼接
data = pd.merge(actual, finidx, left_on=['Stkcd', 'Ddate'], right_on=['Stkcd', 'Accper'], how='outer')
data = data.drop(columns=['Accper'])
## 变量说明
#Stkcd [证券代码]
# A100000 总资产
# 
#T40801 [净资产收益率] - 净利润

#取出只需要的列
wanted = data[['S']]