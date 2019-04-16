
# %% Test kernels

if __name__ == "__main__":
    X = np.array([[1, 5],
                  [2, 6],
                  [3, 7],
                  [4, 8],
                  [5, 9],
                  [6, 10],
                  [7, 11]])
    wt = np.array([-0.5, 0.5])

    print(Ks.linear(X, wt))
    print(Ks.polynomial(X, wt))
    print(Ks.gaussian(X, wt))
    print(Ks.RBF(X, wt))
