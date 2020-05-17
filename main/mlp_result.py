from main.p import db
import numpy as np

# inputs = [1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1]
inputs = [1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1]

inputs = np.array(inputs)
inout = db.get_input_output()

weights = db.get_weights()
# 0번 패턴을 기존 최대값으로 설정
max_sum = np.dot(weights[0], inputs).sum()
max_idx = 0

for i in range(len(weights)):
    weight = weights[i]
    # 값을 곱해서 최대값을 구함
    weight_sum = (weight * inputs).sum()
    # 최대 값 구하기
    if weight_sum > max_sum:
        max_sum = weight_sum
        max_idx = i
# 최종 결과값 출력
print(inout[max_idx].outputs)
