
---
layout: post
title:  Using python access web data
category: tool,homework
tags: Essay
keywords: python,re
---


```python
import re
```


```python
f=open(r'/Users/dal/Desktop/regex_sum_215309.txt','r')
```


```python
k=re.findall('[0-9]+',f.read())
```


```python
print(k)
```

    ['5804', '6080', '2625', '2282', '4202', '1236', '7754', '6343', '9514', '1502', '3216', '3156', '7911', '607', '3712', '7940', '4131', '5122', '9446', '2277', '1774', '7065', '4644', '7345', '576', '4780', '7816', '6754', '2148', '3563', '2996', '369', '2163', '6075', '9656', '422', '5587', '3066', '4176', '2666', '2236', '9917', '2742', '2723', '3592', '4781', '4493', '874', '3801', '4196', '7043', '5229', '9589', '6012', '7846', '9440', '8105', '8820', '5635', '8340', '3456', '5533', '8543', '4748', '42']



```python
len(k)
```




    65




```python
sum(int(item) for item in k)
```




    316237




```python
print (sum( int(item) for item in re.findall('[0-9]+',open(r'/Users/dal/Desktop/regex_sum_215309.txt').read())))
```

    316237



```python
re.findall('[0-9]+',open(r'/Users/dal/Desktop/regex_sum_215309.txt').read())
```




    ['5804',
     '6080',
     '2625',
     '2282',
     '4202',
     '1236',
     '7754',
     '6343',
     '9514',
     '1502',
     '3216',
     '3156',
     '7911',
     '607',
     '3712',
     '7940',
     '4131',
     '5122',
     '9446',
     '2277',
     '1774',
     '7065',
     '4644',
     '7345',
     '576',
     '4780',
     '7816',
     '6754',
     '2148',
     '3563',
     '2996',
     '369',
     '2163',
     '6075',
     '9656',
     '422',
     '5587',
     '3066',
     '4176',
     '2666',
     '2236',
     '9917',
     '2742',
     '2723',
     '3592',
     '4781',
     '4493',
     '874',
     '3801',
     '4196',
     '7043',
     '5229',
     '9589',
     '6012',
     '7846',
     '9440',
     '8105',
     '8820',
     '5635',
     '8340',
     '3456',
     '5533',
     '8543',
     '4748',
     '42']




```python
import urllib.request
from bs4 import BeautifulSoup
url = input('Enter - ')
html = urllib.request.urlopen(url).read()

soup = BeautifulSoup(html,"html.parser")

# Retrieve all of the anchor tags
tags = soup('span')
m=0
for tag in tags:
    m=m+int(tag.contents[0])
print(m)
```

    Enter - http://python-data.dr-chuck.net/comments_215314.html 
    3133



```python
https://pr4e.dr-chuck.com/tsugi/mod/python-data/data/known_by_Fikret.html
```


```python
import urllib.request
from bs4 import BeautifulSoup
url = input('Enter - ')
count=int(input('Enter count:'))
position=int(input('Enter position: '))-1
i=1
while (count > 0 ):
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html,"html.parser")
    tags = soup('a')
    url=tags[position].get('href', None)
    print (url)
    print(tags[position].contents[0])
    count=count-1
    
print("the last names is ",tags[position].contents[0])
```

    Enter - https://pr4e.dr-chuck.com/tsugi/mod/python-data/data/known_by_Shae.html
    Enter count:7
    Enter position: 18
    https://pr4e.dr-chuck.com/tsugi/mod/python-data/data/known_by_Nabeeha.html
    Nabeeha
    https://pr4e.dr-chuck.com/tsugi/mod/python-data/data/known_by_Cator.html
    Cator
    https://pr4e.dr-chuck.com/tsugi/mod/python-data/data/known_by_Kelsiee.html
    Kelsiee
    https://pr4e.dr-chuck.com/tsugi/mod/python-data/data/known_by_Jeannie.html
    Jeannie
    https://pr4e.dr-chuck.com/tsugi/mod/python-data/data/known_by_Nidhi.html
    Nidhi
    https://pr4e.dr-chuck.com/tsugi/mod/python-data/data/known_by_Romi.html
    Romi
    https://pr4e.dr-chuck.com/tsugi/mod/python-data/data/known_by_Artemis.html
    Artemis
    the last names is  Artemis



```python
import urllib.request
import xml.etree.ElementTree as ET
url = input('Enter - ')
data = urllib.request.urlopen(url).read()
tree = ET.fromstring(data)
results = tree.findall('.//count')
print(len(results))
m=0
for i in range(0,len(results)):
    m=m+int(results[i].text)
print(m)
```

    Enter - http://python-data.dr-chuck.net/comments_215311.xml
    50
    2492



```python
http://python-data.dr-chuck.net/comments_42.xml 
```


      File "<ipython-input-2-9db62134317e>", line 1
        http://python-data.dr-chuck.net/comments_42.xml
            ^
    SyntaxError: invalid syntax




```python
def lamb(x):
    return int(results[x].text)
k=list(map(lamb,list(range(0,40))))
```


```python
sum(k)
```




    2378




```python
import urllib.request
import xml.etree.ElementTree as ET
url = input('Enter - ')
data = urllib.request.urlopen(url).read()
tree = ET.fromstring(data)
results = tree.findall('.//count')
print(len(results))
def lamb(x):
    return int(results[x].text)
k=list(map(lamb,list(range(0,len(results)))))
print(sum(k))
```

    Enter - http://python-data.dr-chuck.net/comments_42.xml
    50
    2482



```python
import json
import urllib.request
url=input('Enter location:')
#http://python-data.dr-chuck.net/comments_42.json
print("Retrieving",url)
data=urllib.request.urlopen(url)
data1=str(data.read(),encoding = "utf-8")
print("Retrived:",len(data1))
info=json.loads(data1)
print('User count:', len(info))
a=0;i=0
for item in info["comments"]:
    #print('name', item['name'])
    i=i+1
    a=a+item['count']
print(i)
print(a)
```

    Enter location:http://python-data.dr-chuck.net/comments_42.json
    Retrieving http://python-data.dr-chuck.net/comments_42.json
    Retrived: 2733
    User count: 2
    50
    2482



```python
except: js = None
    if 'status' not in js or js['status'] != 'OK':
        print('==== Failure To Retrieve ====')
        print(data)
        continue
```


```python
import urllib
import json
serviceurl = 'http://maps.googleapis.com/maps/api/geocode/json?'
while True:
    address = input('Enter location: ')
    if len(address) < 1 : break
    url = serviceurl + urllib.parse.urlencode({'sensor':'false', 'address': address})
    print('Retrieving', url)
    uh = urllib.request.urlopen(url)
    data = uh.read()
    print('Retrieved',len(data),'characters')
    js = json.loads(str(data,encoding = "utf-8"))
    print(json.dumps(js, indent=4))
    lat = js["results"][0]["geometry"]["location"]["lat"]
    lng = js["results"][0]["geometry"]["location"]["lng"]
    print('lat',lat,'lng',lng)
    id = js['results'][0]['"place_id"']
    print('place id:',id)
```

    Enter location: oxford
    Retrieving http://maps.googleapis.com/maps/api/geocode/json?sensor=false&address=oxford



    ---------------------------------------------------------------------------

    OSError                                   Traceback (most recent call last)

    /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/urllib/request.py in do_open(self, http_class, req, **http_conn_args)
       1239             try:
    -> 1240                 h.request(req.get_method(), req.selector, req.data, headers)
       1241             except OSError as err: # timeout error


    /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/http/client.py in request(self, method, url, body, headers)
       1082         """Send a complete request to the server."""
    -> 1083         self._send_request(method, url, body, headers)
       1084 


    /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/http/client.py in _send_request(self, method, url, body, headers)
       1127             body = body.encode('iso-8859-1')
    -> 1128         self.endheaders(body)
       1129 


    /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/http/client.py in endheaders(self, message_body)
       1078             raise CannotSendHeader()
    -> 1079         self._send_output(message_body)
       1080 


    /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/http/client.py in _send_output(self, message_body)
        910 
    --> 911         self.send(msg)
        912         if message_body is not None:


    /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/http/client.py in send(self, data)
        853             if self.auto_open:
    --> 854                 self.connect()
        855             else:


    /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/http/client.py in connect(self)
        825         self.sock = self._create_connection(
    --> 826             (self.host,self.port), self.timeout, self.source_address)
        827         self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


    /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/socket.py in create_connection(address, timeout, source_address)
        710     if err is not None:
    --> 711         raise err
        712     else:


    /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/socket.py in create_connection(address, timeout, source_address)
        701                 sock.bind(source_address)
    --> 702             sock.connect(sa)
        703             return sock


    OSError: [Errno 65] No route to host

    
    During handling of the above exception, another exception occurred:


    URLError                                  Traceback (most recent call last)

    <ipython-input-14-9c1ea0781dea> in <module>()
          7     url = serviceurl + urllib.parse.urlencode({'sensor':'false', 'address': address})
          8     print('Retrieving', url)
    ----> 9     uh = urllib.request.urlopen(url)
         10     data = uh.read()
         11     print('Retrieved',len(data),'characters')


    /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/urllib/request.py in urlopen(url, data, timeout, cafile, capath, cadefault, context)
        160     else:
        161         opener = _opener
    --> 162     return opener.open(url, data, timeout)
        163 
        164 def install_opener(opener):


    /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/urllib/request.py in open(self, fullurl, data, timeout)
        463             req = meth(req)
        464 
    --> 465         response = self._open(req, data)
        466 
        467         # post-process response


    /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/urllib/request.py in _open(self, req, data)
        481         protocol = req.type
        482         result = self._call_chain(self.handle_open, protocol, protocol +
    --> 483                                   '_open', req)
        484         if result:
        485             return result


    /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/urllib/request.py in _call_chain(self, chain, kind, meth_name, *args)
        441         for handler in handlers:
        442             func = getattr(handler, meth_name)
    --> 443             result = func(*args)
        444             if result is not None:
        445                 return result


    /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/urllib/request.py in http_open(self, req)
       1266 
       1267     def http_open(self, req):
    -> 1268         return self.do_open(http.client.HTTPConnection, req)
       1269 
       1270     http_request = AbstractHTTPHandler.do_request_


    /Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/urllib/request.py in do_open(self, http_class, req, **http_conn_args)
       1240                 h.request(req.get_method(), req.selector, req.data, headers)
       1241             except OSError as err: # timeout error
    -> 1242                 raise URLError(err)
       1243             r = h.getresponse()
       1244         except:


    URLError: <urlopen error [Errno 65] No route to host>



```python

```
