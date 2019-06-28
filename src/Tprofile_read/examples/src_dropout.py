

"""
.########..########...#######..########...#######..##.....##.########
.##.....##.##.....##.##.....##.##.....##.##.....##.##.....##....##...
.##.....##.##.....##.##.....##.##.....##.##.....##.##.....##....##...
.##.....##.########..##.....##.########..##.....##.##.....##....##...
.##.....##.##...##...##.....##.##........##.....##.##.....##....##...
.##.....##.##....##..##.....##.##........##.....##.##.....##....##...
.########..##.....##..#######..##.........#######...#######.....##...
"""




@tf_export("nn.dropout", v1=[])
def dropout_v2(x, rate, noise_shape=None, seed=None, name=None):
  """Computes dropout.

  With probability `rate`, drops elements of `x`. Input that are kept are
  scaled up by `1 / (1 - rate)`, otherwise outputs `0`.  The scaling is so that
  the expected sum is unchanged.

  **Note:** The behavior of dropout has changed between TensorFlow 1.x and 2.x.
  When converting 1.x code, please use named arguments to ensure behavior stays
  consistent.

  By default, each element is kept or dropped independently.  If `noise_shape`
  is specified, it must be
  [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
  to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
  will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
  and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
  kept independently and each row and column will be kept or not kept together.

  Args:
    x: A floating point tensor.
    rate: A scalar `Tensor` with the same type as x. The probability
      that each element is dropped. For example, setting rate=0.1 would drop
      10% of input elements.
    noise_shape: A 1-D `Tensor` of type `int32`, representing the
      shape for randomly generated keep/drop flags.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    name: A name for this operation (optional).

  Returns:
    A Tensor of the same shape of `x`.

  Raises:
    ValueError: If `rate` is not in `(0, 1]` or if `x` is not a floating point
      tensor.
  """
  with ops.name_scope(name, "dropout", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    # if not x.dtype.is_floating:
    #   raise ValueError("x has to be a floating point tensor since it's going to"
    #                    " be scaled. Got a %s tensor instead." % x.dtype)
    # if isinstance(rate, numbers.Real):
    #   if not (rate >= 0 and rate < 1):
    #     raise ValueError("rate must be a scalar tensor or a float in the "
    #                      "range [0, 1), got %g" % rate)
    #   if rate > 0.5:
    #     logging.log_first_n(
    #         logging.WARN, "Large dropout rate: %g (>0.5). In TensorFlow "
    #         "2.x, dropout() uses dropout rate instead of keep_prob. "
    #         "Please ensure that this is intended.", 5, rate)

    # # Early return if nothing needs to be dropped.
    # if isinstance(rate, numbers.Real) and rate == 0:
    #   return x
    # if context.executing_eagerly():
    #   if isinstance(rate, ops.EagerTensor):
    #     if rate.numpy() == 0:
    #       return x
    # else:
    #   rate = ops.convert_to_tensor(
    #       rate, dtype=x.dtype, name="rate")
    #   rate.get_shape().assert_is_compatible_with(tensor_shape.scalar())

    #   # Do nothing if we know rate == 0
    #   if tensor_util.constant_value(rate) == 0:
    #     return x

    noise_shape = _get_noise_shape(x, noise_shape)
    # Sample a uniform distribution on [0.0, 1.0) and select values larger than
    # rate.
    #
    # NOTE: Random uniform actually can only generate 2^23 floats on [1.0, 2.0)
    # and subtract 1.0.
    random_tensor = random_ops.random_uniform(
        noise_shape, seed=seed, dtype=x.dtype)
    keep_prob = 1 - rate
    scale = 1 / keep_prob
    # NOTE: if (1.0 + rate) - 1 is equal to rate, then we want to consider that
    # float to be selected, hence we use a >= comparison.
    keep_mask = random_tensor >= rate
    ret = x * scale * math_ops.cast(keep_mask, x.dtype)
    if not context.executing_eagerly():
      ret.set_shape(x.get_shape())
    return ret
