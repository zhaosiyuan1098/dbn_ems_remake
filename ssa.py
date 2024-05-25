import numpy as np

from option import Option

option = Option()


class SSA():
    def __init__(self, option:Option):
        self.window_length = option.ssa_window_length
        self.save_length = option.ssa_save_length
        self.k = int(option.num_row_perpage - option.ssa_window_length + 1)

    def ssa_on_segment(self, segment):
        # 初始化SSA处理结果的存储
        U, sigma, VT = np.linalg.svd(segment, full_matrices=False)
        sigma_matrix = np.diag(sigma)
        # 重建信号
        rec = np.dot(U[:, :self.save_length], sigma_matrix[:self.save_length, :self.save_length]).dot(VT[:self.save_length, :])
        return rec

    def ssa_3d(self, x):
        # 初始化输出数组
        a, b, c = x.shape
        result = np.zeros((a, b * self.save_length, c))

        # 对每一段信号逐个应用SSA
        for i in range(a):
            for j in range(b):
                segment = x[i, j, :]
                segment = segment.reshape(-1, segment.shape[-1])  # 确保二维
                ssa_result = self.ssa_on_segment(segment)
                # 存储结果
                result[i, j * self.save_length:(j + 1) * self.save_length, :] = ssa_result

        return result
