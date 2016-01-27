---
layout: post
title: Solve the time problem in python
category: tech
tags: Essay
keywords: python,timezone
---


##时间的转化
>我们的代码往往需要进行简单的时间转化工作，比如将日转化成秒，将小时间转化成分钟等。

>在python中我们可以利用datetime模块来完成不同时间单位的转化。例如，需要表示一个时间间隔，可以像这样创建一个timedelta实例：


```python
from datetime import timedelta
a=timedelta(days=2,hours=6)
b=timedelta(hours=4.5)
c=a+b
c.days 
```




    2




```python
c.seconds
```




    37800




```python
c.total_seconds()
```




    210600.0



##如果需要表示特定的日期和时间，可以创建datetime实例并使用标准的数学运算来操纵它们。示例如下：


```python
from datetime import datetime
a=datetime(2012,9,23)
a+timedelta(days=10)
```




    datetime.datetime(2012, 10, 3, 0, 0)




```python
print a+timedelta(days=10)
```

    2012-10-03 00:00:00



```python
type(a)
```




    datetime.datetime




```python
b=datetime(2012,12,21)
d=b-a
d.days
```




    89




```python
now=datetime.today()
print now
```

    2016-01-20 10:35:01.035368



```python
print now+timedelta(minutes=10)
```

    2016-01-20 10:45:01.035368



```python
#当执行计算时候，应当注意的是datetime模块是可以正确处理闰年的，实例如下：
a=datetime(2012,3,1)
b=datetime(2012,2,28)
(a-b).days
```




    2




```python
c=datetime(2013,3,1)
d=datetime(2013,2,28)
(c-d).days
```




    1



##help(timedelta)可以解决不了months的增加减少，因此需要引入dteutil.relativedelta()来解决，其中dateutil的一个显著特别就是处理有关月份的问题时候能补充一些datetime模块的缺失



```python
from dateutil.relativedelta import relativedelta
a=datetime(2012,9,23)
a+relativedelta(months=+1)
```




    datetime.datetime(2012, 10, 23, 0, 0)




```python
print a+relativedelta(months=+4)
```

    2013-01-23 00:00:00



```python
b=datetime(2012,12,21)
d=b-a
d
```




    datetime.timedelta(89)




```python
d=relativedelta(b,a)
d
```




    relativedelta(months=+2, days=+28)




```python
print (d.days,d.months)
```

    (28, 2)



```python
#将字符串转化为日期
text='2012-09-20'
type(text)
```




    str




```python
y=datetime.strptime(text,'%Y-%m-%d') #strptime支持很多格式代码，具体可以看帮助文档
print y
```

    2012-09-20 00:00:00



```python
type(y)
```




    datetime.datetime



##同样也可以将datetime.datetime类型的转化为字符型


```python
nice_z=datetime.strftime(y,'%Y-%m-%d') #%Y %m %d 代表输入格式，其中-代表text中的-
print nice_z
```

    2012-09-20



```python
z=datetime(2012,9,23,21,37,4,177393)
nice_b=datetime.strftime(z,'%A %B %d,%Y')  #%A,%B.%d 分别代表输出格式
nice_b
```




    'Sunday September 23,2012'



##处理设计时区的日期问题设计下面几个问题

 1. 时区的设置
 2. 时区的转化


```python
from pytz import timezone #pytz.timezone来设置
d=datetime(2012,12,21,9,20,0)
print d
```

    2012-12-21 09:20:00



```python
central=timezone('US/Central')
loc_d=central.localize(d)
print loc_d #芝加哥表示时间来表示日期
```

    2012-12-21 09:20:00-06:00



```python
#将一个本地时间转化为其他区的时间，比如想知道同一个时间在班加罗尔是几点
#Convert to bangalore time
bang_d=loc_d.astimezone(timezone('Asia/Kolkata'))
print bang_d
```

    2012-12-21 20:50:00+05:30


##将所有时间转化为UTC(世界统一时间)。


```python
print loc_d
help(loc_d.astimezone)
```

    2012-12-21 09:20:00-06:00
    Help on built-in function astimezone:
    
    astimezone(...)
        tz -> convert to local time in new timezone tz
    



```python
import pytz
utc_d=loc_d.astimezone(pytz.utc)
print utc_d
```

    2012-12-21 15:20:00+00:00

