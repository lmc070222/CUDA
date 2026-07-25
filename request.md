
1.你加的两个 compare 是把 naiveSgemm2D 和 CublasSgemm 的结果分别和 CPU 版本的结果比较，但是这两个实现正确性都是没问题的，不需要你去验证，只是作为评价性能的参照。
你要做的是保留 naiveSgemm2D 和 CublasSgemm 作为 baseline，自己写一个优化的 gemm kernel，把优化 kernel 的结果跟原有三个版本中的任意一个比较，并测量它需要多少时间。因为马上要到登分的时候了，我建议你直接去做 shared memory tiling。它的思想大概是这样的：
在 naive GEMM 中，一个线程负责计算结果矩阵 C 里的一个元素，每一次乘加都要从 global memory 读取 A 和 B。注意到相邻线程会重复读取很多相同数据，global memory 访问又比较慢，那这就给你提供了优化空间。
shared memory tiling 是把矩阵分成若干小块（你可以用 16×16，如果出现 M/N/K 不能被 tilesize 整除的情况，记得给边界填 0 以防越界访问）。然后在一小块的计算中，一个 block 中的线程先协作把在 A 和 B 中需要的对应区域从 global memory 加载到 shared memory（速度更快），接着同步。接下来，每个线程就可以使用 shared memory 中的数据完成这一小块对应的乘加计算。
计算完当前小块后，再加载下一组小块，重复过程最后得到 C。




2.你把memcpy的时间算进来了，测的不是实际运行时间，记得改一下