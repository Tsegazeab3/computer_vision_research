import numpy as np

class Texture:
    def value(self, u: float, v: float, p: np.ndarray) -> np.ndarray:
        pass

class ConstantTexture(Texture):
    def __init__(self, color: np.ndarray):
        self.color = color
    def value(self, u: float, v: float, p: np.ndarray) -> np.ndarray:
        return self.color