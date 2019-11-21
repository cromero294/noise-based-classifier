import numpy as np
import numpy.matlib

def create_dataset(n, model, ymargin=0.0, noise=None, output_boundary=False):
    x = np.random.rand(n, 1) * 2.0 * np.pi
    xbnd = np.linspace(0, 2.0 * np.pi, 100)

    if model == 'sine':
        y = (np.random.rand(n, 1) - 0.5) * 2.2
        c = y > np.sin(x)
        ybnd = np.sin(xbnd)
    elif model == 'linear':
        y = np.random.rand(n, 1) * 2.0 * np.pi3
        c = y > x
        ybnd = xbnd
    elif model == 'square':
        y = np.random.rand(n, 1) * 4.0 * np.pi * np.pi
        c = y > x * x
        ybnd = xbnd * xbnd
    elif model == 'ringnorm':
        a = 2 / np.sqrt(2.0)
        n2 = int(n / 2)
        x[:n2] = np.random.normal(0, 4.0, (n2, 1))
        x[n2:] = np.random.normal(a, 1.0, (n2, 1))
        y = np.array(x)
        y[:n2] = np.random.normal(0, 4.0, (n2, 1))
        y[n2:] = np.random.normal(a, 1.0, (n2, 1))
        c = np.zeros_like(x)
        c[:n2] = 1
        ybnd = xbnd * xbnd
    elif model == 'reringnorm':
        a = 2 / np.sqrt(2.0)
        n2 = int(n / 2)
        n4 = int(n / 4)
        x[:n4] = np.random.normal(0, 4.0, (n4, 1))
        x[n4:3 * n4] = np.random.normal(a, 1.0, (n2, 1))
        x[3 * n4:] = np.random.normal(a, 0.25, (n4, 1))
        y = np.array(x)
        y[:n4] = np.random.normal(0, 4.0, (n4, 1))
        y[n4:3 * n4] = np.random.normal(a, 1.0, (n2, 1))
        y[n2 + n4:] = np.random.normal(a, 0.25, (n4, 1))
        c = np.zeros_like(x)
        c[n4:3 * n4] = 1
        ybnd = xbnd * xbnd
    elif model == 'xnorm':
        a = 2 / np.sqrt(2.0)
        n2 = int(n / 2)
        n4 = int(n / 4)
        x[:n2] = np.random.normal(a, 1.0, (n2, 1))
        x[n2:] = np.random.normal(-a, 1.0, (n2, 1))
        y = np.array(x)
        y[:n4] = np.random.normal(a, 1.0, (n4, 1))
        y[n4:3 * n4] = np.random.normal(-a, 1.0, (n2, 1))
        y[3 * n4:] = np.random.normal(a, 1.0, (n4, 1))
        c = np.ones_like(x)
        c[:n4] = 0
        c[n2:3 * n4] = 0
        ybnd = xbnd * xbnd
    elif model == 'threenorm':
        a = 2 / np.sqrt(2.0)
        n2 = int(n / 2)
        n4 = int(n / 4)
        x[:n4] = np.random.normal(a, 1.0, (n4, 1))
        x[n4:n2] = np.random.normal(-a, 1.0, (n4, 1))
        x[n2:] = np.random.normal(a, 1.0, (n2, 1))
        y = np.array(x)
        y[:n4] = np.random.normal(a, 1.0, (n4, 1))
        y[n4:n2] = np.random.normal(-a, 1.0, (n4, 1))
        y[n2:] = np.random.normal(-a, 1.0, (n2, 1))
        c = np.ones_like(x)
        c[:n2] = 0
        ybnd = xbnd * xbnd
    elif model == 'twonorm':
        a = 2 / np.sqrt(2.0)
        n2 = int(n / 2)
        x[:n2] = np.random.normal(a, 1.0, (n2, 1))
        x[n2:] = np.random.normal(-a, 1.0, (n2, 1))
        y = np.array(x)
        y[:n2] = np.random.normal(a, 1.0, (n2, 1))
        y[n2:] = np.random.normal(-a, 1.0, (n2, 1))
        c = np.ones_like(x)
        c[:n2] = 0
        ybnd = xbnd * xbnd
    else:
        y = np.random.rand(n, 1) * 2.0 * np.pi
        c = y > x
        ybnd = xbnd

    y[c == True] = y[c == True] + ymargin
    y[c == False] = y[c == False] - ymargin

    if noise is not None:
        y = y + noise * np.random.randn(n, 1)
        x = x + noise * np.random.randn(n, 1)

    if output_boundary == True:
        return np.matlib.matrix(x), np.matlib.matrix(y), np.matlib.matrix(c * 1), xbnd, ybnd
    else:
        return np.matlib.matrix(x), np.matlib.matrix(y), np.matlib.matrix(c * 1)


def create_full_dataset(n, dimm, model, noise=None):
    x = np.random.rand(n, dimm) * 2.0 * np.pi

    if model == 'threenorm':
        a = 2 / np.sqrt(2.0)

        n2 = int(n / 2)
        n4 = int(n / 4)

        for i in range(dimm):
            x[:n4, i] = np.random.normal(a, 1.0, (n4, 1)).transpose()
            x[n4:n2, i] = np.random.normal(-a, 1.0, (n4, 1)).transpose()

            if i % 2 == 0:
                x[n2:, i] = np.random.normal(a, 1.0, (n2, 1)).transpose()
            else:
                x[n2:, i] = np.random.normal(-a, 1.0, (n2, 1)).transpose()

        # x[:n4, :] = np.random.normal(a, 1.0, (n4, 1))
        # x[n4:n2, :] = np.random.normal(-a, 1.0, (n4, 1))
        # x[n2:, ::2] = np.random.normal(a, 1.0, (n2, 1))
        # x[n2:, 1::2] = np.random.normal(-a, 1.0, (n2, 1))

        c = np.ones((1, n))
        c[:, :n2] = 0

    elif model == "ringnorm":
        a = 2 / np.sqrt(2.0)

        n2 = int(n / 2)

        for i in range(dimm):
            x[:n2, i] = np.random.normal(0, 4.0, (n2, 1)).transpose()
            x[n2:, i] = np.random.normal(a, 1.0, (n2, 1)).transpose()

        c = np.ones((1, n))
        c[:, :n2] = 0

    if noise is not None:
        x = x + noise * np.random.randn(n, 1)

    return np.matlib.matrix(x), np.matlib.matrix(c * 1)
