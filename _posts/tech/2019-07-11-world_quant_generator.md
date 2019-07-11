---
layout: post
title:  Using python to generate expression
category: quant
tags: Essay
keywords: python,re
---

定义运算符
```
two_tuple_operator = ['+','-','*','/']
two_param_operator = ['ts_corr','ts_cov']
two_operator = two_tuple_operator + two_param_operator
one_tuple_operator = ['rank']
one_param_operator = ['delay','decay','ts_mean']
one_operator = one_tuple_operator + one_param_operator
```

核心思想是波兰表达式，二元-1，一元0，变量1
生成一个path 1开始 1 结束 最大是3 最小是1
```
def is_operate(symbol):
    if symbol in two_operator:
        return -1
    elif symbol in one_operator:
        return 0
    else:
        return 1

```
## 美化 去掉多余的括号
def add_brackets(s):
    l1 = sum(list(map(lambda x: x =='(',s)))
    l2 = sum(list(map(lambda x: x ==')',s)))
    if l1 == 0:
        return 1
    elif (l1-l2) ==0:
        return 0
    else:
        return 1

```

```
def generate_expression(list_data):
    token =[]
    param_num = 0
    param_dict ={}
    for i in list_data:
        if i in two_operator:
            first, second=token.pop(), token.pop()
            if i in two_tuple_operator:
                token.append(str('('+second+i+first+')'))
            else:
                token.append(str(i+'('+second+','+first+','+'param'+str(param_num)+')'))
                param_dict[i] = 'param'+str(param_num)
                param_num +=1
                
        elif i in one_operator:
            last=token.pop()
            if add_brackets(last):    
                if i in one_tuple_operator:
                    token.append(i+'('+last+')')
                else:
                    token.append(i+'('+last+','++'param'+str(param_num) +')')
                    param_dict[i] = 'param'+str(param_num)
                    param_num +=1
            else:
                if i in one_tuple_operator:
                    token.append(i+last) 
                else:
                    token.append(i+'('+last+','+'param'+str(param_num)+')')
                    param_dict[i] = 'param'+str(param_num)
                    param_num +=1             
        else:
            token.append(i)
#         print(token)
    return (token.pop(),param_dict) if token else 0

```

效果
```
data = [1,2,"+", '3', 4, "ts_corr" ,"*",'delay']
data_stred = list(map(str,data))
generate_expression(data_stred)
# output
('delay(((1+2)*ts_corr(3,4,param0)),param1)',
 {'ts_corr': 'param0', 'delay': 'param1'})
```