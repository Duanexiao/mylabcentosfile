#coding = utf-8
import pandas as pd
import numpy as np
from pandas.io.excel import ExcelWriter
import datetime
import os
import itertools

#os.chdir('/home/duane/promoter_low')
print os.getcwd()

localfiles = os.listdir(os.getcwd())
for i in localfiles:
    if i.startswith(datetime.datetime.now().strftime('%Y%m%d')):
	excelfilename = i		#获取进行操作的excel文件名
print excelfilename
df = pd.read_excel(excelfilename)	#使用pandas读取excel文件
df = df.fillna(u'空')


#df['o_lqo_primary_reason'] = df['o_lqo_primary_reason'].astype(int)
all_cols = df.columns[1:-1]#['order_yearandmonth', 'order_time', 'promotionway', 'order_type', 'o_department', 'o_lqo_primary_reason', 'oid_count']
all_cols_num = len(all_cols)
new_colsnum = 1
new_cols_name_list = []
for i, j in enumerate(all_cols):
    locals()['all_cols_'+str(i)] = df[j].drop_duplicates()
    new_colsnum *= len(locals()['all_cols_'+str(i)])
    new_cols_name_list.append(locals()['all_cols_'+str(i)])
new_cols = []
new_cols_tuple = []
for cols in itertools.product(*new_cols_name_list):
    new_cols.append(''.join(cols))
    new_cols_tuple.append(cols)
print new_cols
first_col_time = [i for i in df.iloc[:,0].drop_duplicates()]

#promotionway = df['promotionway'].drop_duplicates()		#将promotionway此列去重
#promotionwaylabel = [i for i in promotionway]		#将去重后内容写进列表中
#order_type = df['order_type'].drop_duplicates()	
#order_typelabel = [i for i in order_type]		
#o_department = df['o_department'].drop_duplicates()	
#o_departmentlabel = [i for i in o_department]
#o_lqo_primary_reason = df['o_lqo_primary_reason'].drop_duplicates()
#o_lqo_primary_reasonlabel = [i for i in o_lqo_primary_reason]		
#new_colsnum = len(promotionway)*len(order_type)*len(o_department)*len(o_lqo_primary_reason)	#新建后数据框列数
#new_cols = []	#新建列表保存新建数据框列名
#for i in range(len(promotionwaylabel)):		#填充新建数据框列表名称   
#    for j in range(len(order_typelabel)):
#        for k in range(len(o_departmentlabel)):
	    #for q in range(len(o_lqo_primary_reasonlabel)):
           #     newcolname = 'w'+str(promotionwaylabel[i])+'t'+str(order_typelabel[j])+'d'+str(o_departmentlabel[k])+'r'+str(o_lqo_primary_reasonlabel[q])	#渠道加推广来源加终端类型
           # 	new_cols.append(newcolname)

#yearandmonth = [i for i in df['order_yearandmonth'].drop_duplicates()]	
#ordertime = [i for i in df['order_time'].drop_duplicates()]		
newdfindex = []		#为新建的数据框创建索引
for j in range(len(first_col_time)):	#时间作为索引，时间去掉了年月信息
    newindexname = str(first_col_time[j])
    newdfindex.append(newindexname)
newdf = pd.DataFrame(np.zeros((len(first_col_time), new_colsnum)), index=newdfindex, columns = new_cols)	#建立新的数据框
newdftran = newdf.T		#将新建的数据框转置以便对数据框赋值
print new_cols[0]
for i in range(len(newdf)):
    testfirst = []
    for j in new_cols_tuple:
        q = 0
	tem_df = df[df.iloc[:,0]==first_col_time[i]]
        while q < len(all_cols):
	    tem_df = tem_df[tem_df[all_cols[q]]==j[q]]
            q += 1
        if len(tem_df) == 0:
            count_num = 0
        else:
            count_num = tem_df[df.columns[-1]]
	    print count_num
            count_num = count_num[count_num.index[0]]
        testfirst.append(count_num)
    newdftran[newdfindex[i]] = testfirst
newdftran = newdftran.T
print newdftran
#print 'ok'
'''for i in range(len(newdf)):
    testfirst = []
    for j in new_cols:
        selected_row = df[(df.iloc[:,0] == first_col_time[i]) & df]
	separate_w = j.split('w')
	separate_w_t = separate_w[1].split('t')
	separate_w_t_d = separate_w_t[1].split('d')
	separate_w_t_d_r = separate_w_t_d[1].split('r')
        dfdatetime = df[df['order_time']==ordertime[i]]		#提取旧的数据框中满足log_time等于logtime[i]条件后的数据框
        wayjudge = dfdatetime[dfdatetime['promotionway']==int(separate_w_t[0])]
        wayandtypejudge = wayjudge[wayjudge['order_type']==int(separate_w_t_d[0])]
        wayandtypeanddepartmentjudge = wayandtypejudge[wayandtypejudge['o_department']==int(separate_w_t_d_r[0])]
	wayandtypeanddepartmentandreasonjudge = wayandtypeanddepartmentjudge[wayandtypeanddepartmentjudge['o_lqo_primary_reason']==int(separate_w_t_d_r[1])]
	
        if len(wayandtypeanddepartmentandreasonjudge) == 0:
            order_count = 0 
        else:
            order_count = wayandtypeanddepartmentandreasonjudge['oid_count']
            order_count = order_count[order_count.index[0]]
        testfirst.append(order_count)
    newdftran[newdfindex[i]] = testfirst
newdftran = newdftran.T'''
#timecols = []
#for i in range(len(newdftran)):
##    selecteddf = df[df['order_time']==eval(newdfindex[i])]
#    if len(str(selecteddf['order_time'].values[0])) == 1:
#        timecolsname = str(selecteddf['order_yearandmonth'].values[0])+'0'+str(selecteddf['order_time'].values[0])
#    else:
#        timecolsname = str(selecteddf['order_yearandmonth'].values[0])+str(selecteddf['order_time'].values[0])
#    timecols.append(timecolsname)
#newdftran = newdftran.T
#newdftran.columns = timecols
#newdftran = newdftran.T
#new_colsname = newdftran.columns
#infodict = {'w0': u'上海','w1':u'北京','w2':u'广州','w3':u'成都','w4':u'楼程程','w5':u'窝客','w6':u'妈妈帮','w888888888':u'自然','t0':u'快速预诊','t2':u'点名预诊','t888888888':u'无','d0':u'客服接诊','d1':u'妇产科','d2':u'儿科','d3':u'内科','d888888888':u'无','r1':u'问题过于简单','r2':u'相同提问人','r3':u'一个来回','r4':u'重复问题','r5':u'疑似录音','r6':u'长时间没有回复','r888888888':u'无', 'r0':u'正常订单'}
#translatecolnames = []
#for i in new_colsname:
   # separate_w = i.split('w')
   # separate_w_t = separate_w[1].split('t')
   # separate_w_t_d = separate_w_t[1].split('d')
   # separate_w_t_d_r = separate_w_t_d[1].split('r')
  #  promotionway = 'w' + separate_w_t[0]
  #  order_type = 't' + separate_w_t_d[0]
 #   order_department = 'd' + separate_w_t_d_r[0]
 #   reason = 'r' + separate_w_t_d_r[1]
#    
#    translatecolname = infodict[promotionway] + infodict[order_type] + infodict[order_department] + infodict[reason]
#    translatecolnames.append(translatecolname)
#newdftran.columns = translatecolnames
newdftran = newdftran[newdftran.columns[(newdftran != 0).any()]]
excelname = newdftran.index[0]+'-'+newdftran.index[-1]
excelname = datetime.datetime.now().strftime('%Y%m%d')
excelname += '上周低质量推广员订单数据.xls'
writer = ExcelWriter(excelname)	
newdftran.to_excel(writer,'sheet1')
writer.save()
print 'ok'
#print os.listdir(os.getcwd())
#os.remove(excelname)
#os.remove(excelfilename)




    
    
