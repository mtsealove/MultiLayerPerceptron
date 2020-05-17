import numpy as np


# 입력값 및 출력값을 가지는 클래스
class pattern:
    def __init__(self, teach, input_pattern):
        self.outputs = self.set_teach(teach)
        self.inputs = self.set_pattern(input_pattern)

    def set_pattern(self, input_pattern):
        nums = input_pattern.split(',')
        result = []
        for num in nums:
            n = int(num)
            if n == -1:
                n = 0
            result.append(n)
        return np.array(result)

    def set_teach(self, teach):
        result = []
        for i in teach:
            result.append(int(i))
        return np.array(result)
