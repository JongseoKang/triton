module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:89", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduction_2d_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %c32_i32 = arith.constant 32 : i32
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c31_i32 = arith.constant 31 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x4xf32, #ttg.slice<{dim = 2, parent = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [32, 1, 1], warpsPerCTA = [1, 4, 1], order = [2, 0, 1]}>}>>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %3 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %4 = tt.splat %1 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %5 = arith.addi %4, %2 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %6 = arith.addi %arg3, %c31_i32 : i32
    %7 = arith.divsi %6, %c32_i32 : i32
    %8:2 = scf.for %arg4 = %c0_i32 to %7 step %c1_i32 iter_args(%arg5 = %cst_0, %arg6 = %cst_1) -> (tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>}>>, tensor<32x4xf32, #ttg.slice<{dim = 2, parent = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [32, 1, 1], warpsPerCTA = [1, 4, 1], order = [2, 0, 1]}>}>>)  : i32 {
      %14 = arith.muli %arg4, %c32_i32 : i32
      %15 = tt.splat %14 : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
      %16 = arith.addi %15, %3 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
      %17 = tt.expand_dims %5 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>}>> -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>>
      %18 = tt.expand_dims %16 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>}>> -> tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>>
      %19 = tt.broadcast %17 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>> -> tensor<32x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>>
      %20 = tt.broadcast %18 : tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>> -> tensor<32x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>>
      %21 = arith.addi %19, %20 : tensor<32x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>>
      %22 = tt.splat %arg2 : i32 -> tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>>
      %23 = arith.cmpi slt, %17, %22 : tensor<32x1xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>>
      %24 = tt.splat %arg3 : i32 -> tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>>
      %25 = arith.cmpi slt, %18, %24 : tensor<1x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>>
      %26 = tt.broadcast %23 : tensor<32x1xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>> -> tensor<32x32xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>>
      %27 = tt.broadcast %25 : tensor<1x32xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>> -> tensor<32x32xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>>
      %28 = arith.andi %26, %27 : tensor<32x32xi1, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>>
      %29 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>>
      %30 = tt.addptr %29, %21 : tensor<32x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>>, tensor<32x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>>
      %31 = tt.load %30, %28, %cst : tensor<32x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>>
      %32 = "tt.reduce"(%31) <{axis = 1 : i32}> ({
      ^bb0(%arg7: f32, %arg8: f32):
        %37 = arith.addf %arg7, %arg8 : f32
        tt.reduce.return %37 : f32
      }) : (tensor<32x32xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
      %33 = tt.reshape %31 allow_reorder efficient_layout : tensor<32x32xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>> -> tensor<32x4x8xf32, #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [32, 1, 1], warpsPerCTA = [1, 4, 1], order = [2, 0, 1]}>>
      %34 = "tt.reduce"(%33) <{axis = 2 : i32}> ({
      ^bb0(%arg7: f32, %arg8: f32):
        %37 = arith.addf %arg7, %arg8 : f32
        tt.reduce.return %37 : f32
      }) : (tensor<32x4x8xf32, #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [32, 1, 1], warpsPerCTA = [1, 4, 1], order = [2, 0, 1]}>>) -> tensor<32x4xf32, #ttg.slice<{dim = 2, parent = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [32, 1, 1], warpsPerCTA = [1, 4, 1], order = [2, 0, 1]}>}>>
      %35 = arith.addf %arg6, %34 : tensor<32x4xf32, #ttg.slice<{dim = 2, parent = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [32, 1, 1], warpsPerCTA = [1, 4, 1], order = [2, 0, 1]}>}>>
      %36 = arith.addf %arg5, %32 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
      scf.yield %arg5, %35 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>}>>, tensor<32x4xf32, #ttg.slice<{dim = 2, parent = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [32, 1, 1], warpsPerCTA = [1, 4, 1], order = [2, 0, 1]}>}>>
    }
    %9 = "tt.reduce"(%8#1) <{axis = 1 : i32}> ({
    ^bb0(%arg4: f32, %arg5: f32):
      %14 = arith.addf %arg4, %arg5 : f32
      tt.reduce.return %14 : f32
    }) : (tensor<32x4xf32, #ttg.slice<{dim = 2, parent = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [32, 1, 1], warpsPerCTA = [1, 4, 1], order = [2, 0, 1]}>}>>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [32, 1, 1], warpsPerCTA = [1, 4, 1], order = [2, 0, 1]}>}>}>>
    %10 = ttg.convert_layout %9 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [32, 1, 1], warpsPerCTA = [1, 4, 1], order = [2, 0, 1]}>}>}>> -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %11 = arith.addf %cst_0, %10 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>}>>
    %12 = "tt.reduce"(%11) <{axis = 0 : i32}> ({
    ^bb0(%arg4: f32, %arg5: f32):
      %14 = arith.addf %arg4, %arg5 : f32
      tt.reduce.return %14 : f32
    }) : (tensor<32xf32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>}>>) -> f32
    %13 = tt.atomic_rmw fadd, acq_rel, gpu, %arg1, %12, %true : (!tt.ptr<f32>, f32, i1) -> f32
    tt.return
  }
}
