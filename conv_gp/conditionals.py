import tensorflow as tf
from gpflow import settings

logger = settings.logger()

def conditional(Kmn, Kmm, Knn, f, *, full_cov=False, q_sqrt=None, white=False):
    """
    Given a g1 and g2, and distribution p and q such that
      p(g2) = N(g2;0,Kmm)
      p(g1) = N(g1;0,Knn)
      p(g1|g2) = N(g1;0,Knm)
    And
      q(g2) = N(g2;f,q_sqrt*q_sqrt^T)
    This method computes the mean and (co)variance of
      q(g1) = \int q(g2) p(g1|g2)
    :param Kmn: P x M x N
    :param Kmm: M x M
    :param Knn: P x N x N  or P x N
    :param f: M x R
    :param full_cov: bool
    :param q_sqrt: R x M x M (lower triangular)
    :param white: bool
    :return: N x R  or R x N x N
    """
    logger.debug("base conditional")
    # compute kernel stuff
    num_func = tf.shape(f)[1]  # R

    Lm = tf.cholesky(Kmm)

    def solve_A(MN_Kmn):
        return tf.matrix_triangular_solve(Lm, MN_Kmn, lower=True) # M x M @ M x N -> M x N
    A = tf.map_fn(solve_A, Kmn) # P x M x N

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = Knn - tf.tensordot(A, A, [[1], [1]]) # P x N x N
        fvar = tf.tile(fvar[None, :, :, :], [num_func, 1, 1, 1])  # R x N x N
    else:
        fvar = Knn - tf.reduce_sum(tf.square(A), 1) # P x N
        fvar = tf.tile(fvar[None, :, :], [num_func, 1, 1])  # R x P x N

    # another backsubstitution in the unwhitened case
    if not white:
        def backsub(MN_A):
            return tf.matrix_triangular_solve(tf.transpose(Lm), MN_A, lower=False)
        A = tf.map_fn(backsub, A) # P x M x N

    # construct the conditional mean
    fmean = tf.tensordot(A, f, [[1], [0]]) # P x N x R
    fmean = tf.transpose(fmean, [1, 0, 2]) # N x P x R

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 3:
            L = tf.matrix_band_part(q_sqrt, -1, 0)  # R x M x M

            # A: P x M x N
            LTA = tf.tensordot(L, A, [[1], [1]]) # R x M x P x N
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" %
                             str(q_sqrt.get_shape().ndims))
        if full_cov:
            fvar = fvar + tf.tensordot(LTA, LTA, [[1], [1]]) # R x P x N x N
        else:
            fvar = fvar + tf.reduce_sum(tf.square(LTA), 1) # R x P x N

    return fmean, fvar # N x P x R, R x P x N or R x P x N x N

