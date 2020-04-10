### A simple three-layer neural net, implemented in Rust

This is a simple, three-layer neural net implemented with only basic linear algebra operations. Not including the input matrix, there are two hidden layers (two weights vectors), and the resulting output layer. The activation of both layers is the [sigmoid logistic function](https://en.wikipedia.org/wiki/Logistic_function).

This is more or less supposed to be a minmal example of a multiple layer neural net with clear linear algebra that could generalized for more layers. At the very least, it seems to be working based on some pretty limited testing. If you see something wrong either in the math or the implementation, please file an issue to let me know! This was a learning exercise for me, so any constructive feedback would certainly be welcome!
