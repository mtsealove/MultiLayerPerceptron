import numpy as np
import main.p.db as db


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def diff(a, b):
    return abs((a - b).sum())


# 7개의 은닉충
num_of_hidden = 7
# 10개의 출력층
num_of_output = 10
# 25개의 은닉층
num_of_input = 25
eta = 0.1

# 입출력 값들을 가져옴
inout = db.get_input_output()

input_hidden_weights = []
hidden_output_weights = []
prev_input_hidden_weights = []
prev_hidden_output_weights = []

# 초기 가중치 설정
for i in range(len(inout)):
    input_hidden_weights.append(np.ones((num_of_hidden, num_of_input)) * 0.4)
    hidden_output_weights.append(np.ones((num_of_output, num_of_hidden)) * 0.4)

# 비교용 이전 가중치 설정
for i in range(len(inout)):
    prev_hidden_output_weights.append(hidden_output_weights[i].copy())
    prev_input_hidden_weights.append(input_hidden_weights[i].copy())
prev_diff = 10000000000
switch = 1

# 각각의 학습 패턴에 대해서 수행
for epoch in range(100000):
    idx = 0
    # 각 클래스별로
    for io in inout:
        # 전방향 학습
        # 입력층 -> 은닉층의 출력값 계산
        hidden_output = []
        for i in range(num_of_hidden):
            hidden_output.append(sigmoid((io.inputs * input_hidden_weights[idx][i]).sum()))
        # 은닉층 -> 출력층의 출력값
        # print(hidden_output)
        output_output = []
        for i in range(num_of_output):
            output_output.append(sigmoid(hidden_output * hidden_output_weights[idx][i]).sum())
        # 출력층 오차 계산
        output_delta = []
        for i in range(num_of_output):
            output_delta.append(output_output[i] * (1 - output_output[i]) * (io.outputs[i] - output_output[i]))
        # 은닉층 오차 계산
        hidden_delta = []
        for i in range(num_of_hidden):
            x = 0
            for j in range(num_of_output):
                x = x + output_delta[j] + hidden_output_weights[idx][j][i]
            hidden_delta.append(hidden_output[i] * (1 - hidden_output[i]) * x)

        # 역방향 학습
        # 은닉층 -> 출력층 가중치 수정
        for i in range(num_of_output):
            for j in range(num_of_hidden):
                hidden_output_weights[idx][i][j] = hidden_output_weights[idx][i][j] + eta * output_delta[i] * \
                                                   output_output[i]
        #  입력층 -> 출력층 가중치 수정
        for i in range(num_of_input):
            for h in range(num_of_hidden):
                # print(hidden_delta[h])
                # print(hidden_output[h])
                input_hidden_weights[idx][h][i] = input_hidden_weights[idx][h][i] + eta * hidden_delta[h] * \
                                                  hidden_output[h]
        # print(input_hidden_weight)
        idx = idx + 1
    # 이전의 가중치와 비교
    total = 0
    for i in range(len(input_hidden_weights)):
        total = total + abs((prev_input_hidden_weights[i] - input_hidden_weights[i]).sum())
        total = total + abs((prev_hidden_output_weights[i] - hidden_output_weights[i]).sum())

    # 1000번의 epoch 마다 출력
    if epoch % 1000 == 0:
        print('epoch: ' + str(epoch))

    # 이전 변화량과 비료
    if abs(switch - abs(prev_diff - total)) < 0.000001:
        print('epoch: ' + str(epoch))
        break
    switch = (abs(prev_diff - total))
    prev_diff = total

db.remove_weight()
for i in range(len(input_hidden_weights)):
    # 학습된 가중치 저장
    db.save_weight(inout[i].outputs, np.dot(hidden_output_weights[i], input_hidden_weights[i]))
    print('-------------------------')


