"""Inference wrapper submitted to the challenge platform (autoregressive 3-step model -> 18-step forecast)."""
import os
import tensorflow as tf
import numpy as np
class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'SubmissionModel'))

    def predict(self, X, categories):
        
        # Autoregressive forecasting: predict 3 steps, append them to the input window, repeat 6 times.
        reg_predictions = np.array([])
        X_temp = X
        batch_size = X.shape[0]
        for reg in range(0,18,3):
            pred_temp = self.model.predict(X_temp)
            if(len(reg_predictions)==0):
                reg_predictions = pred_temp
            else:
                reg_predictions = np.concatenate((reg_predictions,pred_temp),axis=1)
            X_temp = np.concatenate((X_temp[:, 3:], pred_temp.reshape(batch_size, 3)), axis=1)
        out = reg_predictions  # Shape [BSx9] for Phase 1 and [BSx18] for Phase 2
        out = out.reshape(batch_size, 18)
        return out