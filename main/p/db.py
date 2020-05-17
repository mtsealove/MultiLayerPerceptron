import pymysql
from main.p.pattern import pattern
import numpy as np

pw = 'Fucker0916!'


# db 에서 입력 패턴
def get_input_output():
    conn = pymysql.connect(host='localhost', db='AI', passwd=pw, port=3306, user='root', charset='utf8')
    cursor = conn.cursor()
    cursor.execute('select * from Pattern')
    fetch = cursor.fetchall()
    result = []
    for data in fetch:
        result.append(pattern(data[2], data[1]))
    cursor.close()
    conn.close()
    return result


# 기존 가중치 삭제
def remove_weight():
    conn = pymysql.connect(host='localhost', db='AI', passwd=pw, port=3306, user='root', charset='utf8')
    cursor = conn.cursor()
    cursor.execute('delete from Weights')
    conn.commit()
    cursor.close()
    conn.close()


# 학습된 가중치 저장
def save_weight(t, weight):
    conn = pymysql.connect(host='localhost', db='AI', passwd=pw, port=3306, user='root', charset='utf8')
    cursor = conn.cursor()
    teach = ''
    for w in t:
        teach = teach + str(w)
    print(teach)
    weight_str = ''
    for x in range(len(weight)):
        line = weight[x]
        for i in range(len(line)):
            weight_str += str(line[i])
            if i != len(line) - 1:
                weight_str += ','
        if x != len(weight) - 1:
            weight_str += ';;'
    # print(weight_str)
    sql = "insert into Weights set ID='" + teach + "', Learn='" + weight_str + "'"
    cursor.execute(sql)
    conn.commit()
    cursor.close()
    conn.close()


# 학습된 출력 가중치 반환
def get_weights():
    conn = pymysql.connect(host='localhost', db='AI', passwd=pw, port=3306, user='root', charset='utf8')
    cursor = conn.cursor()
    sql = 'select * from Weights'
    cursor.execute(sql)
    result = []
    fetch = cursor.fetchall()
    for data in fetch:
        lines = data[1].split(';;')
        learn = []
        for line in lines:
            nums = line.split(',')
            a = []

            for num in nums:
                a.append(float(num))
            learn.append(a)
        result.append(learn)
    cursor.close()
    conn.close()
    return np.array(result)
