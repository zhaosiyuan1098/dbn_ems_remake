import numpy as np

from option import Option

option = Option()


class SSA():
    def __init__(self, option:Option):
        self.window_length = option.ssa_window_length
        self.save_length = option.ssa_save_length
        self.k = int(option.num_row_perpage - option.ssa_window_length + 1)

    # def ssa_3d(self, x):
    #     ssa_result = np.zeros((x.shape[0], x.shape[1], self.save_length, x.shape[2]))
    #     ssa_x = np.zeros((self.window_length, self.k))
    #     for i in range(x.shape[0]):
    #         for j in range(x.shape[1]):
    #             for k in range(self.k):
    #                 ssa_x[:, k] = x[i, j, k:k + self.window_length]
    #             U, sigma, VT = np.linalg.svd(ssa_x, full_matrices=False)

    #             for k in range(VT.shape[0]):
    #                 VT[k, :] *= sigma[k]
    #             A = VT

    #             rec = np.zeros((self.window_length, x.shape[2]))

    #             for k in range(self.window_length):
    #                 for n in range(self.window_length - 1):
    #                     for m in range(n + 1):
    #                         rec[k, n] += A[k, n - m] * U[m, k]
    #                     rec[k, n] /= (n + 1)
    #                 for n in range(self.window_length - 1, x.shape[2] - self.window_length + 1):
    #                     for m in range(self.window_length):
    #                         rec[k, n] += A[k, n - m] * U[m, k]
    #                     rec[k, n] /= self.window_length
    #                 for n in range(x.shape[2] - self.window_length + 1, x.shape[2]):
    #                     for m in range(n - x.shape[2] + self.window_length, self.window_length):
    #                         rec[k, n] += A[k, n - m] * U[m, k]
    #                     rec[k, n] /= (x.shape[2] - n)

    #             saved_rec = rec[:self.save_length, :]
    #             ssa_result[i, j, :, :] = saved_rec
    #             a,b,c,d=ssa_result.shape
    #             resized_result=ssa_result.reshape((a,b*c,d))
                
    #     return resized_result
    
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
