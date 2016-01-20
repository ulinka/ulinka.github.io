---
layout: post
title: week one Using python with database
category: homework
tags: Essay
keywords: python,SQLlite
---

```python
import sqlite3
import re
conn = sqlite3.connect('emaildb.sqlite')
cur = conn.cursor()

cur.execute('''
DROP TABLE IF EXISTS Counts''')

cur.execute('''
CREATE TABLE Counts (org TEXT, count INTEGER)''')

fname = input('Enter file name: ')
if ( len(fname) < 1 ) : fname = 'mbox-short.txt'
fh = open(fname)
for line in fh:
    if not line.startswith('From: ') : continue
    pieces = line.split()
    org1= pieces[1]
    org=re.findall(r'@((?:[A-Za-z0-9]+\.)+[A-Za-z]+$)',org1)[0]
    #print email
    cur.execute('SELECT count FROM Counts WHERE org = ? ', (org, ))
    row = cur.fetchone()
    if row is None:
        cur.execute('''INSERT INTO Counts (org, count)
                VALUES ( ?, 1 )''', ( org, ) )
    else :
        cur.execute('UPDATE Counts SET count=count+1 WHERE org = ?',
            (org, ))
    # This statement commits outstanding changes to disk each
    # time through the loop - the program can be made faster
    # by moving the commit so it runs only after the loop completes
    conn.commit()

# https://www.sqlite.org/lang_select.html
sqlstr = 'SELECT org, count FROM Counts ORDER BY count DESC LIMIT 10'

print ("Counts:")
for row in cur.execute(sqlstr) :
    print(str(row[0]), row[1])

cur.close()
```

    Enter file name: mbox.txt
    Counts:
    iupui.edu 536
    umich.edu 491
    indiana.edu 178
    caret.cam.ac.uk 157
    vt.edu 110
    uct.ac.za 96
    media.berkeley.edu 56
    ufp.pt 28
    gmail.com 25
    et.gatech.edu 17



```python

```
