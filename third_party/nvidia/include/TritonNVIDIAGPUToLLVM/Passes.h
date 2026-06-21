#ifndef TRITONGPU_CONVERSION_TRITONNVIDIAGPUTOLLVM_PASSES_H
#define TRITONGPU_CONVERSION_TRITONNVIDIAGPUTOLLVM_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cstdint>
#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

struct HelionCacheTritonGPUToLLVMStats {
  uint64_t macroSignatureTotal = 0;
  uint64_t macroSignatureHits = 0;
  uint64_t macroSignatureUnique = 0;
  uint64_t merkleNodeTotal = 0;
  uint64_t merkleNodeHits = 0;
  uint64_t merkleNodeUnique = 0;
  uint64_t macroRecipeTotal = 0;
  uint64_t macroRecipeHits = 0;
  uint64_t macroRecipeMisses = 0;
  uint64_t macroRecipeStored = 0;
  uint64_t macroRecipeMaterializeFail = 0;
  uint64_t macroRecipeVerifyFail = 0;
  uint64_t macroRecipeUnsupported = 0;
  uint64_t macroRecipeEntries = 0;
  uint64_t macroRecipeBytes = 0;
};

HelionCacheTritonGPUToLLVMStats getHelionCacheTritonGPUToLLVMStats();
void resetHelionCacheTritonGPUToLLVMStats();

#define GEN_PASS_DECL
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonGPUToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(int32_t computeCapability);
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(int32_t computeCapability, int32_t ptxVersion);
std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonGPUToLLVMPass(
    int32_t computeCapability, int32_t ptxVersion, int32_t cacheLevel);
std::unique_ptr<OperationPass<ModuleOp>>
createAllocateSharedMemoryNvPass(int32_t computeCapability, int32_t ptxVersion);

#define GEN_PASS_REGISTRATION
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h.inc"

} // namespace triton

} // namespace mlir

#endif
