"""A module to handle polynomial arithmetic in the quotient ring Z_a[x]/f(x)."""

from util.ntt import NTTContext, FFTContext
from util.barrette import BarrettReducer        # Use MontgomeryReducer if preferred

class Polynomial:
    """A polynomial in the ring R_a.
    Here, R is the quotient ring Z[x]/f(x), where f(x) = x^d + 1.
    The polynomial keeps track of the ring degree d, the coefficient
    modulus a, and the coefficients in an array.
    Attributes:
        ring_degree (int): Degree d of polynomial that determines the
            quotient ring R.
        coeffs (array): Array of coefficients of polynomial, where coeffs[i]
            is the coefficient for x^i.
    """
    def __init__(self, degree, coeffs):
        """Inits Polynomial in the ring R_a with the given coefficients.
        Args:
            degree (int): Degree of quotient polynomial for ring R_a.
            coeffs (array): Array of integers of size degree, representing
                coefficients of polynomial.
        """
        self.ring_degree = degree
        assert len(coeffs) == degree, 'Size of polynomial array %d is not equal to degree %d of ring' %(len(coeffs), degree)
        self.coeffs = coeffs

    def add(self, poly, coeff_modulus=None):
        assert isinstance(poly, Polynomial)
        poly_sum = Polynomial(self.ring_degree, [self.coeffs[i] + poly.coeffs[i] for i in range(self.ring_degree)])
        if coeff_modulus:
            reducer = BarrettReducer(coeff_modulus)
            poly_sum.coeffs = [reducer.reduce(c) for c in poly_sum.coeffs]
        return poly_sum

    def subtract(self, poly, coeff_modulus=None):
        assert isinstance(poly, Polynomial)
        poly_diff = Polynomial(self.ring_degree, [self.coeffs[i] - poly.coeffs[i] for i in range(self.ring_degree)])
        if coeff_modulus:
            reducer = BarrettReducer(coeff_modulus)
            poly_diff.coeffs = [reducer.reduce(c) for c in poly_diff.coeffs]
        return poly_diff

    def multiply(self, poly, coeff_modulus, ntt=None, crt=None):
        if crt:
            return self.multiply_crt(poly, crt)
        if ntt:
            a = ntt.ftt_fwd(self.coeffs)
            b = ntt.ftt_fwd(poly.coeffs)
            ab = [a[i] * b[i] for i in range(self.ring_degree)]
            prod = ntt.ftt_inv(ab)
            reducer = BarrettReducer(coeff_modulus)
            prod = [reducer.reduce(p) for p in prod]
            return Polynomial(self.ring_degree, prod)

        return self.multiply_naive(poly, coeff_modulus)

    def multiply_naive(self, poly, coeff_modulus=None):
        assert isinstance(poly, Polynomial)
        poly_prod = Polynomial(self.ring_degree, [0] * self.ring_degree)
        for d in range(2 * self.ring_degree - 1):
            index = d % self.ring_degree
            sign = int(d < self.ring_degree) * 2 - 1
            coeff = 0
            for i in range(self.ring_degree):
                if 0 <= d - i < self.ring_degree:
                    coeff += self.coeffs[i] * poly.coeffs[d - i]
            poly_prod.coeffs[index] += sign * coeff

        if coeff_modulus:
            reducer = BarrettReducer(coeff_modulus)
            poly_prod.coeffs = [reducer.reduce(c) for c in poly_prod.coeffs]
        return poly_prod

    def scalar_multiply(self, scalar, coeff_modulus=None):
        if coeff_modulus:
            reducer = BarrettReducer(coeff_modulus)
            new_coeffs = [reducer.reduce(scalar * c) for c in self.coeffs]
        else:
            new_coeffs = [(scalar * c) for c in self.coeffs]
        return Polynomial(self.ring_degree, new_coeffs)

    def scalar_integer_divide(self, scalar, coeff_modulus=None):
        if coeff_modulus:
            reducer = BarrettReducer(coeff_modulus)
            new_coeffs = [reducer.reduce(c // scalar) for c in self.coeffs]
        else:
            new_coeffs = [(c // scalar) for c in self.coeffs]
        return Polynomial(self.ring_degree, new_coeffs)

    def rotate(self, r):
        k = 5 ** r
        new_coeffs = [0] * self.ring_degree
        for i in range(self.ring_degree):
            index = (i * k) % (2 * self.ring_degree)
            if index < self.ring_degree:
                new_coeffs[index] = self.coeffs[i]
            else:
                new_coeffs[index - self.ring_degree] = -self.coeffs[i]
        return Polynomial(self.ring_degree, new_coeffs)

    def conjugate(self):
        new_coeffs = [0] * self.ring_degree
        new_coeffs[0] = self.coeffs[0]
        for i in range(1, self.ring_degree):
            new_coeffs[i] = -self.coeffs[self.ring_degree - i]
        return Polynomial(self.ring_degree, new_coeffs)

    def round(self):
        if type(self.coeffs[0]) == complex:
            new_coeffs = [round(c.real) for c in self.coeffs]
        else:
            new_coeffs = [round(c) for c in self.coeffs]
        return Polynomial(self.ring_degree, new_coeffs)

    def floor(self):
        new_coeffs = [int(c) for c in self.coeffs]
        return Polynomial(self.ring_degree, new_coeffs)

    def mod(self, coeff_modulus):
        reducer = BarrettReducer(coeff_modulus)
        new_coeffs = [reducer.reduce(c) for c in self.coeffs]
        return Polynomial(self.ring_degree, new_coeffs)

    def mod_small(self, coeff_modulus):
        reducer = BarrettReducer(coeff_modulus)
        new_coeffs = [reducer.reduce(c) for c in self.coeffs]
        new_coeffs = [c - coeff_modulus if c > coeff_modulus // 2 else c for c in new_coeffs]
        return Polynomial(self.ring_degree, new_coeffs)

    def base_decompose(self, base, num_levels):
        decomposed = [Polynomial(self.ring_degree, [0] * self.ring_degree) for _ in range(num_levels)]
        poly = self
        for i in range(num_levels):
            decomposed[i] = poly.mod(base)
            poly = poly.scalar_multiply(1 / base).floor()
        return decomposed

    def evaluate(self, inp):
        result = self.coeffs[-1]
        for i in range(self.ring_degree - 2, -1, -1):
            result = result * inp + self.coeffs[i]
        return result

    def __str__(self):
        s = ''
        for i in range(self.ring_degree - 1, -1, -1):
            if self.coeffs[i] != 0:
                if s != '':
                    s += ' + '
                if i == 0 or self.coeffs[i] != 1:
                    s += str(int(self.coeffs[i]))
                if i != 0:
                    s += 'x'
                if i > 1:
                    s += '^' + str(i)
        return s
