from option import Option
import numpy as np
option=Option()

class FFT :
    def fft_3d(self, x):
        fft_data_3d = np.zeros(x.shape)
        for i in range(x.shape[0]):
            fft_temp = np.fft.fft(x[i:i + 1, :, :], axis=1)
            fft_data_3d[i:i + 1, :, :] = fft_temp
        return fft_data_3d
    
    def fft_transform_multidimensional(self,input_array):
        # Ensure the input is a numpy array
        input_array = np.array(input_array)
        
        # Check if the input array has three dimensions
        if input_array.ndim != 3:
            raise ValueError("Input array must be three-dimensional")
        
        # Apply FFT along the last dimension of each segment
        transformed_array = np.fft.fft(input_array, axis=2)
        
        # Separate real and imaginary parts
        real_part = transformed_array.real
        imag_part = transformed_array.imag
        
        # Stack real and imaginary parts along a new dimension
        combined_array = np.stack((real_part, imag_part), axis=1)
        
        # Reshape to ensure the shape is (a, 2*b, c)
        final_array = combined_array.reshape(input_array.shape[0], 2 * input_array.shape[1], input_array.shape[2])
        
        return final_array

