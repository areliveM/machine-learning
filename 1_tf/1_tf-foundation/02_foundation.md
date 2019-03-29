# TensorFlow 基本用法

### 综述

TensorFlow 是一个编程系统, 使用图来表示计算任务. 图中的节点被称之为 op (operation 的缩写). 一个 op 获得 0 个或多个 `Tensor`, 执行计算, 产生 0 个或多个 `Tensor`. 每个 `Tensor` 是一个类型化的多维数组. 例如, 你可以将一小组图像集表示为一个四维浮点数数组, 这四个维度分别是 `[batch, height, width, channels]`.

一个 TensorFlow 图描述了计算的过程. 为了进行计算, 图必须在 `会话` 里被启动. `会话` 将图的 op 分发到诸如 CPU 或 GPU 之类的 `设备` 上, 同时提供执行 op 的方法. 这些方法执行后, 将产生的 `tensor` 返回. 在 Python 语言中, 返回的 `tensor` 是 `numpy ndarray` 对象; 在 C 和 C++ 语言中, 返回的 `tensor `是 `tensorflow::Tensor` 实例.

### 基本使用

使用 TensorFlow, 你必须明白 TensorFlow:

+ 使用图 (graph) 来表示计算任务.
+ 在被称之为 `会话 (Session)` 的上下文 (context) 中执行图.
+ 使用 tensor 表示数据.
+ 通过 `变量 (Variable)` 维护状态.
+ 使用 `feed` 和 `fetch` 可以为任意的操作(arbitrary operation) 赋值或者从其中获取数据.

### 计算图

TensorFlow 程序通常被组织成一个构建阶段和一个执行阶段. 在构建阶段, op 的执行步骤被描述成一个图. 在执行阶段, 使用会话执行执行图中的 op.

例如, 通常在构建阶段创建一个图来表示和训练神经网络, 然后在执行阶段反复执行图中的训练 op.

TensorFlow 支持 C, C++, Python 编程语言. 目前, TensorFlow 的 Python 库更加易用, 它提供了大量的辅助函数来简化构建图的工作, 这些函数尚未被 C 和 C++ 库支持.

三种语言的会话库 (session libraries) 是一致的.

### 构建图

构建图的第一步, 是创建源 op (source op). 源 op 不需要任何输入, 例如 `常量 (Constant)`. 源 op 的输出被传递给其它 op 做运算.

Python 库中, op 构造器的返回值代表被构造出的 op 的输出, 这些返回值可以传递给其它 op 构造器作为输入.

TensorFlow Python 库有一个*默认图 (default graph)*, op 构造器可以为其增加节点. 这个默认图对 许多程序来说已经足够用了. 阅读 

[Graph 类](http://www.tensorfly.cn/tfdoc/api_docs/python/framework.html#Graph) （需要修改）

文档 来了解如何管理多个图.

```python
import tensorflow as tf

# 创建一个常量 op, 产生一个 1x2 矩阵. 这个 op 被作为一个节点
# 加到默认图中.
#
# 构造器的返回值代表该常量 op 的返回值.
matrix1 = tf.constant([[3., 3.]])

# 创建另外一个常量 op, 产生一个 2x1 矩阵.
matrix2 = tf.constant([[2.],[2.]])

# 创建一个矩阵乘法 matmul op , 把 'matrix1' 和 'matrix2' 作为输入.
# 返回值 'product' 代表矩阵乘法的结果.
product = tf.matmul(matrix1, matrix2)
```

默认图现在有三个节点, 两个 `constant()` op, 和一个`matmul()` op. 为了真正进行矩阵相乘运算, 并得到矩阵乘法的 结果, 你必须在会话里启动这个图.

### 在一个会话中启动图

构造阶段完成后, 才能启动图. 启动图的第一步是创建一个 `Session` 对象, 如果无任何创建参数, 会话构造器将启动默认图.

欲了解完整的会话 API, 请阅读

[Session 类](http://www.tensorfly.cn/tfdoc/api_docs/python/client.html#session-management).（需要修改）

```python
# 启动默认图.
sess = tf.Session()

# 调用 sess 的 'run()' 方法来执行矩阵乘法 op, 传入 'product' 作为该方法的参数. 
# 上面提到, 'product' 代表了矩阵乘法 op 的输出, 传入它是向方法表明, 我们希望取回
# 矩阵乘法 op 的输出.
#
# 整个执行过程是自动化的, 会话负责传递 op 所需的全部输入. op 通常是并发执行的.
# 
# 函数调用 'run(product)' 触发了图中三个 op (两个常量 op 和一个矩阵乘法 op) 的执行.
#
# 返回值 'result' 是一个 numpy `ndarray` 对象.
result = sess.run(product)
print result
# ==> [[ 12.]]

# 任务完成, 关闭会话.
sess.close()
```

`Session` 对象在使用完后需要关闭以释放资源. 除了显式调用 close 外, 也可以使用 "with" 代码块 来自动完成关闭动作.

```python
with tf.Session() as sess:
  result = sess.run([product])
  print result
```

在实现上, TensorFlow 将图形定义转换成分布式执行的操作, 以充分利用可用的计算资源(如 CPU 或 GPU). 一般你不需要显式指定使用 CPU 还是 GPU, TensorFlow 能自动检测. 如果检测到 GPU, TensorFlow 会尽可能地利用找到的第一个 GPU 来执行操作.

如果机器上有超过一个可用的 GPU, 除第一个外的其它 GPU 默认是不参与计算的. 为了让 TensorFlow 使用这些 GPU, 你必须将 op 明确指派给它们执行. `with...Device` 语句用来指派特定的 CPU 或 GPU 执行操作:

```python
with tf.Session() as sess:
  with tf.device("/gpu:1"):
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.],[2.]])
    product = tf.matmul(matrix1, matrix2)
    ...
```

设备用字符串进行标识. 目前支持的设备包括:

+ `"/cpu:0"`: 机器的 CPU.
+ `"/gpu:0"`: 机器的第一个 GPU, 如果有的话.
+ `"/gpu:1"`: 机器的第二个 GPU, 以此类推.

阅读

[使用GPU](http://www.tensorfly.cn/tfdoc/how_tos/using_gpu.html)（需修改）

章节, 了解 TensorFlow GPU 使用的更多信息.

### 交互式使用

文档中的 Python 示例使用一个会话 [`Session`](http://www.tensorfly.cn/tfdoc/api_docs/python/client.html#Session) 来 启动图, 并调用 [`Session.run()`](http://www.tensorfly.cn/tfdoc/api_docs/python/client.html#Session.run) 方法执行操作.

为了便于使用诸如 `IPython` 之类的 Python 交互环境, 可以使用

 [`InteractiveSession`](http://www.tensorfly.cn/tfdoc/api_docs/python/client.html#InteractiveSession) （需修改）

代替 `Session` 类, 使用 

[`Tensor.eval()`](http://www.tensorfly.cn/tfdoc/api_docs/python/framework.html#Tensor.eval) （需修改）

和 

[`Operation.run()`](http://www.tensorfly.cn/tfdoc/api_docs/python/framework.html#Operation.run)（需修改）

 方法代替 `Session.run()`. 这样可以避免使用一个变量来持有会话.

```python
# 进入一个交互式 TensorFlow 会话.
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# 使用初始化器 initializer op 的 run() 方法初始化 'x' 
x.initializer.run()

# 增加一个减法 sub op, 从 'x' 减去 'a'. 运行减法 op, 输出结果 
sub = tf.sub(x, a)
print sub.eval()
# ==> [-2. -1.]
```



