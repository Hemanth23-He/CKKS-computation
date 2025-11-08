class BarrettReducer:
    def __init__(self, modulus):
        self.modulus = modulus
        self.k = modulus.bit_length() * 2
        self.mu = (1 << self.k) // modulus

    def reduce(self, x):
        q = ((x * self.mu) >> self.k)
        r = x - q * self.modulus
        if r >= self.modulus:
            r -= self.modulus
        if r < 0:
            r += self.modulus
        return r
    
    def inverse(self, x):
        # Modular inverse using extended Euclidean algorithm
        # Returns y such that (x * y) % modulus == 1
        a, b, u = x, self.modulus, 1
        while b != 0:
            t = a // b
            a, b = b, a - t * b
            u = u - t * u
        return u % self.modulus
