module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduction_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %5 = tt.splat %arg2 : i32 -> tensor<1024xi32, #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32, #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<1024xi32, #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %10 = tt.reshape %9 allow_reorder : tensor<1024xf32, #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>> -> tensor<1024xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %11 = "tt.reduce"(%10) <{axis = 0 : i32}> ({
    ^bb0(%arg3: f32, %arg4: f32):
      %13 = arith.addf %arg3, %arg4 : f32
      tt.reduce.return %13 : f32
    }) : (tensor<1024xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>) -> f32
    %12 = tt.atomic_rmw fadd, acq_rel, gpu, %arg1, %11, %true : (!tt.ptr<f32>, f32, i1) -> f32
    tt.return
  }
}
