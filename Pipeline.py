import torch
class AbstractPipelineClass:
    def __init__(self, model=None):
        if model:
            self.model = model
        raise NotImplementedError
    
    def train(self):
        raise NotImplementedError
    
    def preprocess(self, x):
        raise NotImplementedError
    
    def predict(self, x):
        preprocessed = self.preprocess(x)
        return self.decorate_output(self.model(preprocessed.to(torch.float32)))

    def decorate_output(self):
        raise NotImplementedError