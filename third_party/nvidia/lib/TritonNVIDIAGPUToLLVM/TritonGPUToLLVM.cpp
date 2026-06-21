#include "Dialect/NVGPU/IR/Dialect.h"
#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "TritonNVIDIAGPUToLLVM/Utility.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "Allocation.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTRITONGPUTOLLVM
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton::NVIDIA;

namespace {

std::mutex helionCacheStatsMutex;
llvm::StringSet<> helionCacheMacroSignatures;
llvm::StringSet<> helionCacheMerkleNodes;
mlir::triton::HelionCacheTritonGPUToLLVMStats helionCacheStats;

struct HelionCacheRecipeEntry {
  std::string templateText;
  unsigned liveInCount = 0;
  unsigned liveOutCount = 0;
  uint64_t sizeBytes = 0;
};

std::mutex helionCacheRecipeMutex;
std::unordered_map<std::string, HelionCacheRecipeEntry> helionCacheRecipes;
std::deque<std::string> helionCacheRecipeLru;
uint64_t helionCacheRecipeBytes = 0;

struct HelionCacheLoweringContext {
  ModuleOp mod;
  int32_t computeCapability;
  int32_t ptxVersion;
  int32_t cacheLevel;
  std::string moduleContext;
  DenseMap<Value, uint64_t> merkleNodeIds;
};

thread_local HelionCacheLoweringContext *activeHelionCacheContext = nullptr;

int64_t getEnvInt64(const char *name, int64_t defaultValue) {
  const char *value = std::getenv(name);
  if (!value || !*value)
    return defaultValue;
  char *end = nullptr;
  long long parsed = std::strtoll(value, &end, 10);
  if (end == value)
    return defaultValue;
  return parsed;
}

void updateRecipeCacheFootprintLocked() {
  helionCacheStats.macroRecipeEntries = helionCacheRecipes.size();
  helionCacheStats.macroRecipeBytes = helionCacheRecipeBytes;
}

void evictRecipesLocked() {
  int64_t maxEntries =
      getEnvInt64("HELIONCACHE_TTGIR_LLIR_RECIPE_MAX_ENTRIES", 4096);
  int64_t maxBytes =
      getEnvInt64("HELIONCACHE_TTGIR_LLIR_RECIPE_MAX_BYTES", 128LL << 20);
  if (maxEntries <= 0 || maxBytes <= 0) {
    helionCacheRecipes.clear();
    helionCacheRecipeLru.clear();
    helionCacheRecipeBytes = 0;
    updateRecipeCacheFootprintLocked();
    return;
  }
  while (!helionCacheRecipeLru.empty() &&
         (static_cast<int64_t>(helionCacheRecipes.size()) > maxEntries ||
          static_cast<int64_t>(helionCacheRecipeBytes) > maxBytes)) {
    std::string key = helionCacheRecipeLru.front();
    helionCacheRecipeLru.pop_front();
    auto it = helionCacheRecipes.find(key);
    if (it == helionCacheRecipes.end())
      continue;
    helionCacheRecipeBytes -= it->second.sizeBytes;
    helionCacheRecipes.erase(it);
  }
  updateRecipeCacheFootprintLocked();
}

void touchRecipeLocked(StringRef key) {
  helionCacheRecipeLru.erase(
      std::remove(helionCacheRecipeLru.begin(), helionCacheRecipeLru.end(),
                  key.str()),
      helionCacheRecipeLru.end());
  helionCacheRecipeLru.push_back(key.str());
}

void incrementRecipeStat(uint64_t mlir::triton::HelionCacheTritonGPUToLLVMStats::*field) {
  std::lock_guard<std::mutex> lock(helionCacheStatsMutex);
  ++(helionCacheStats.*field);
}

std::string stringify(Type type) {
  std::string result;
  llvm::raw_string_ostream os(result);
  type.print(os);
  return result;
}

std::string stringify(Attribute attr) {
  std::string result;
  llvm::raw_string_ostream os(result);
  attr.print(os);
  return result;
}

std::string stringifyValueType(Value value) { return stringify(value.getType()); }

std::string makeModuleContext(ModuleOp mod, int32_t computeCapability,
                              int32_t ptxVersion) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << "cc=" << computeCapability << "|ptx=" << ptxVersion;
  for (StringRef attrName :
       {"ttg.num-warps", "ttg.num-ctas", "ttg.threads-per-warp"}) {
    os << "|" << attrName << "=";
    if (Attribute attr = mod->getAttr(attrName))
      attr.print(os);
  }
  return result;
}

bool isMacroSignatureOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "ttg.convert_layout" || name == "tt.dot" ||
         name == "tt.load" || name == "tt.store" ||
         name.starts_with("ttg.local_") ||
         name.starts_with("ttng.tc_gen5_mma") ||
         name.starts_with("ttng.warp_group_dot") ||
         name.starts_with("ttng.async_") ||
         name.starts_with("ttg.async_") || name.contains("tma") ||
         name.contains("barrier");
}

bool isMerklePureOp(Operation *op) {
  if (op->getNumRegions() != 0)
    return false;

  StringRef name = op->getName().getStringRef();
  if (name == "tt.get_program_id" || name == "tt.make_range" ||
      name == "tt.splat" || name == "tt.expand_dims" ||
      name == "tt.broadcast" || name == "tt.reshape" ||
      name == "tt.trans" || name == "tt.join" || name == "tt.split" ||
      name == "tt.addptr" || name == "tt.bitcast")
    return true;

  if (name == "arith.constant" || name == "arith.addi" ||
      name == "arith.subi" || name == "arith.muli" ||
      name == "arith.divsi" || name == "arith.divui" ||
      name == "arith.remsi" || name == "arith.remui" ||
      name == "arith.andi" || name == "arith.ori" ||
      name == "arith.xori" || name == "arith.shli" ||
      name == "arith.shrsi" || name == "arith.shrui" ||
      name == "arith.cmpi" || name == "arith.select" ||
      name == "arith.index_cast" || name == "arith.extsi" ||
      name == "arith.extui" || name == "arith.trunci" ||
      name == "arith.maxsi" || name == "arith.maxui" ||
      name == "arith.minsi" || name == "arith.minui")
    return true;

  return false;
}

void recordMacroSignature(StringRef signature) {
  std::lock_guard<std::mutex> lock(helionCacheStatsMutex);
  bool inserted = helionCacheMacroSignatures.insert(signature).second;
  ++helionCacheStats.macroSignatureTotal;
  if (inserted)
    ++helionCacheStats.macroSignatureUnique;
  else
    ++helionCacheStats.macroSignatureHits;
}

void recordMerkleNode(StringRef nodeId) {
  std::lock_guard<std::mutex> lock(helionCacheStatsMutex);
  bool inserted = helionCacheMerkleNodes.insert(nodeId).second;
  ++helionCacheStats.merkleNodeTotal;
  if (inserted)
    ++helionCacheStats.merkleNodeUnique;
  else
    ++helionCacheStats.merkleNodeHits;
}

std::string makeMacroSignature(Operation *op, StringRef context) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << context << "|op=" << op->getName().getStringRef();
  os << "|operands=[";
  llvm::interleaveComma(op->getOperands(), os,
                        [&](Value value) { os << stringifyValueType(value); });
  os << "]|results=[";
  llvm::interleaveComma(op->getResults(), os,
                        [&](Value value) { os << stringifyValueType(value); });
  os << "]|attrs=[";
  llvm::interleaveComma(op->getAttrs(), os, [&](NamedAttribute attr) {
    os << attr.getName().strref() << "=";
    attr.getValue().print(os);
  });
  os << "]|regions=" << op->getNumRegions();
  return result;
}

uint64_t hashString(StringRef value) { return llvm::xxh3_64bits(value); }

uint64_t makeExternalValueHash(Value value, StringRef context) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << context << "|external-value|type=" << stringifyValueType(value);
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    os << "|block-arg=" << arg.getArgNumber();
    if (Operation *parentOp = arg.getOwner()->getParentOp())
      os << "|parent=" << parentOp->getName().getStringRef();
  } else if (auto resultValue = dyn_cast<OpResult>(value)) {
    Operation *owner = resultValue.getOwner();
    os << "|result=" << resultValue.getResultNumber();
    os << "|owner=" << owner->getName().getStringRef();
  }
  return hashString(result);
}

uint64_t getValueNodeId(Value value, DenseMap<Value, uint64_t> &nodeIds,
                        StringRef context) {
  auto it = nodeIds.find(value);
  if (it != nodeIds.end())
    return it->second;
  uint64_t id = makeExternalValueHash(value, context);
  nodeIds[value] = id;
  return id;
}

uint64_t makeMerkleNodeId(Operation *op, DenseMap<Value, uint64_t> &nodeIds,
                          StringRef context) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << context << "|op=" << op->getName().getStringRef();
  os << "|operands=[";
  llvm::interleaveComma(op->getOperands(), os, [&](Value operand) {
    os << getValueNodeId(operand, nodeIds, context);
  });
  os << "]|results=[";
  llvm::interleaveComma(op->getResults(), os, [&](Value value) {
    os << stringifyValueType(value);
  });
  os << "]|attrs=[";
  llvm::interleaveComma(op->getAttrs(), os, [&](NamedAttribute attr) {
    os << attr.getName().strref() << "=";
    attr.getValue().print(os);
  });
  os << "]";
  return hashString(result);
}

void observeMerkleNodes(HelionCacheLoweringContext &loweringContext) {
  ModuleOp mod = loweringContext.mod;
  StringRef context = loweringContext.moduleContext;
  DenseMap<Value, uint64_t> &nodeIds = loweringContext.merkleNodeIds;
  mod.walk([&](Operation *op) {
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments())
          getValueNodeId(arg, nodeIds, context);
      }
    }
  });

  mod.walk([&](Operation *op) {
    if (!isMerklePureOp(op))
      return;
    uint64_t nodeId = makeMerkleNodeId(op, nodeIds, context);
    std::string nodeIdText = std::to_string(nodeId);
    recordMerkleNode(nodeIdText);
    for (Value result : op->getResults())
      nodeIds[result] = nodeId;
  });
}

void observeHelionCacheSignatures(HelionCacheLoweringContext &loweringContext) {
  ModuleOp mod = loweringContext.mod;
  StringRef context = loweringContext.moduleContext;
  mod.walk([&](Operation *op) {
    if (isMacroSignatureOp(op))
      recordMacroSignature(makeMacroSignature(op, context));
  });
  observeMerkleNodes(loweringContext);
}

bool isRecipeEligibleRoot(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "ttg.convert_layout" || name == "tt.load" ||
         name == "tt.store" || name == "ttg.local_load" ||
         name == "ttg.local_store";
}

bool isRecipeAllowedPayloadOp(Operation *op) {
  if (op->getNumRegions() != 0 || op->getNumSuccessors() != 0)
    return false;
  StringRef name = op->getName().getStringRef();
  return name.starts_with("llvm.") || name.starts_with("nvvm.") ||
         name.starts_with("arith.") ||
         name == "builtin.unrealized_conversion_cast";
}

std::string makeRecipeKey(Operation *op, ValueRange adaptedOperands) {
  HelionCacheLoweringContext *ctx = activeHelionCacheContext;
  if (!ctx)
    return {};

  std::string result;
  llvm::raw_string_ostream os(result);
  os << "helioncache.recipe.v2|";
  os << makeMacroSignature(op, ctx->moduleContext);
  os << "|operand-merkle=[";
  llvm::interleaveComma(op->getOperands(), os, [&](Value value) {
    auto it = ctx->merkleNodeIds.find(value);
    if (it != ctx->merkleNodeIds.end())
      os << it->second;
    else
      os << "type:" << stringifyValueType(value);
  });
  os << "]|adapted=[";
  llvm::interleaveComma(adaptedOperands, os, [&](Value value) {
    os << stringifyValueType(value);
  });
  os << "]";
  return std::to_string(hashString(result));
}

std::optional<HelionCacheRecipeEntry> lookupRecipe(StringRef key) {
  std::lock_guard<std::mutex> statsLock(helionCacheStatsMutex);
  ++helionCacheStats.macroRecipeTotal;
  {
    std::lock_guard<std::mutex> recipeLock(helionCacheRecipeMutex);
    auto it = helionCacheRecipes.find(key.str());
    if (it == helionCacheRecipes.end()) {
      ++helionCacheStats.macroRecipeMisses;
      updateRecipeCacheFootprintLocked();
      return std::nullopt;
    }
    touchRecipeLocked(key);
    ++helionCacheStats.macroRecipeHits;
    updateRecipeCacheFootprintLocked();
    return it->second;
  }
}

void storeRecipe(StringRef key, HelionCacheRecipeEntry entry) {
  std::lock_guard<std::mutex> statsLock(helionCacheStatsMutex);
  std::lock_guard<std::mutex> recipeLock(helionCacheRecipeMutex);
  auto it = helionCacheRecipes.find(key.str());
  if (it != helionCacheRecipes.end()) {
    helionCacheRecipeBytes -= it->second.sizeBytes;
    it->second = std::move(entry);
  } else {
    helionCacheRecipes.emplace(key.str(), std::move(entry));
  }
  helionCacheRecipeBytes += helionCacheRecipes[key.str()].sizeBytes;
  touchRecipeLocked(key);
  ++helionCacheStats.macroRecipeStored;
  evictRecipesLocked();
}

void noteRecipeUnsupported() {
  incrementRecipeStat(&mlir::triton::HelionCacheTritonGPUToLLVMStats::
                          macroRecipeUnsupported);
}

void noteRecipeMaterializeFail() {
  incrementRecipeStat(&mlir::triton::HelionCacheTritonGPUToLLVMStats::
                          macroRecipeMaterializeFail);
}

void noteRecipeVerifyFail() {
  incrementRecipeStat(&mlir::triton::HelionCacheTritonGPUToLLVMStats::
                          macroRecipeVerifyFail);
}

bool isDefinedByCapturedOp(Value value, const llvm::SetVector<Operation *> &ops) {
  Operation *def = value.getDefiningOp();
  return def && ops.contains(def);
}

FailureOr<HelionCacheRecipeEntry>
buildRecipeTemplate(MLIRContext *ctx, Location loc, ValueRange adaptedOperands,
                    ArrayRef<Operation *> insertedOps,
                    ArrayRef<Value> replacements) {
  if (!activeHelionCacheContext || insertedOps.empty())
    return failure();

  llvm::SetVector<Operation *> capturedOps;
  for (Operation *op : insertedOps) {
    if (!op || !op->getBlock() || !isRecipeAllowedPayloadOp(op))
      return failure();
    capturedOps.insert(op);
  }

  DenseMap<Value, unsigned> liveInIndex;
  SmallVector<Type> liveInTypes;
  for (Value value : adaptedOperands) {
    if (!value)
      return failure();
    liveInIndex.try_emplace(value, liveInTypes.size());
    liveInTypes.push_back(value.getType());
  }

  SmallVector<Value> liveOuts(replacements.begin(), replacements.end());
  SmallVector<Type> liveOutTypes;
  for (Value value : liveOuts) {
    if (!value)
      return failure();
    if (!isDefinedByCapturedOp(value, capturedOps) &&
        !liveInIndex.contains(value))
      return failure();
    liveOutTypes.push_back(value.getType());
  }

  for (Operation *op : capturedOps) {
    for (Value operand : op->getOperands()) {
      if (isDefinedByCapturedOp(operand, capturedOps))
        continue;
      if (!liveInIndex.contains(operand))
        return failure();
    }
  }

  OpBuilder builder(ctx);
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());
  auto funcType = builder.getFunctionType(liveInTypes, liveOutTypes);
  auto func = builder.create<func::FuncOp>(loc, "__helioncache_recipe",
                                           funcType);
  Block *body = func.addEntryBlock();
  builder.setInsertionPointToStart(body);

  IRMapping mapper;
  for (auto [value, index] : liveInIndex)
    mapper.map(value, body->getArgument(index));

  for (Operation *op : capturedOps) {
    Operation *cloned = op->clone(mapper);
    builder.insert(cloned);
    for (auto [oldResult, newResult] :
         llvm::zip(op->getResults(), cloned->getResults()))
      mapper.map(oldResult, newResult);
  }

  SmallVector<Value> mappedLiveOuts;
  for (Value value : liveOuts)
    mappedLiveOuts.push_back(mapper.lookupOrDefault(value));
  builder.create<func::ReturnOp>(loc, mappedLiveOuts);

  std::string text;
  llvm::raw_string_ostream os(text);
  OpPrintingFlags flags;
  flags.enableDebugInfo(false);
  module.print(os, flags);
  os.flush();

  HelionCacheRecipeEntry entry;
  entry.templateText = std::move(text);
  entry.liveInCount = adaptedOperands.size();
  entry.liveOutCount = liveOuts.size();
  entry.sizeBytes = entry.templateText.size();
  return entry;
}

LogicalResult materializeRecipe(const HelionCacheRecipeEntry &entry,
                                Operation *root, ValueRange adaptedOperands,
                                ConversionPatternRewriter &rewriter) {
  if (entry.liveInCount != adaptedOperands.size())
    return failure();

  auto parsed =
      parseSourceString<ModuleOp>(entry.templateText, root->getContext());
  if (!parsed)
    return failure();

  func::FuncOp func;
  for (func::FuncOp candidate : parsed->getOps<func::FuncOp>()) {
    func = candidate;
    break;
  }
  if (!func || func.empty())
    return failure();
  Block &body = func.getBody().front();
  auto returnOp = dyn_cast<func::ReturnOp>(body.getTerminator());
  if (!returnOp || returnOp.getNumOperands() != entry.liveOutCount)
    return failure();

  IRMapping mapper;
  for (auto [arg, value] : llvm::zip(body.getArguments(), adaptedOperands))
    mapper.map(arg, value);

  for (Operation &op : body.without_terminator()) {
    if (!isRecipeAllowedPayloadOp(&op))
      return failure();
    Operation *cloned = op.clone(mapper);
    cloned->setLoc(root->getLoc());
    rewriter.insert(cloned);
    for (auto [oldResult, newResult] :
         llvm::zip(op.getResults(), cloned->getResults()))
      mapper.map(oldResult, newResult);
  }

  SmallVector<Value> replacements;
  for (Value value : returnOp.getOperands())
    replacements.push_back(mapper.lookupOrDefault(value));

  if (replacements.empty())
    rewriter.eraseOp(root);
  else
    rewriter.replaceOp(root, replacements);
  return success();
}

struct RecipeGuardImpl {
  RecipeGuardImpl(Operation *root, ValueRange adaptedOperands,
                  ConversionPatternRewriter &rewriter)
      : root(root), adaptedOperands(adaptedOperands.begin(),
                                    adaptedOperands.end()),
        rewriter(&rewriter), rootContext(root->getContext()),
        rootLoc(root->getLoc()), rootResultCount(root->getNumResults()),
        rootBlock(root->getBlock()) {
    HelionCacheLoweringContext *ctx = activeHelionCacheContext;
    active = ctx && ctx->cacheLevel >= 2 && isRecipeEligibleRoot(root);
    if (active) {
      key = makeRecipeKey(root, this->adaptedOperands);
      if (rootBlock) {
        for (Operation &op : *rootBlock)
          preexistingOps.insert(&op);
      }
    }
  }

  ~RecipeGuardImpl() = default;

  LogicalResult tryReplay() {
    if (!active || key.empty())
      return failure();
    if (std::optional<HelionCacheRecipeEntry> entry = lookupRecipe(key)) {
      if (succeeded(materializeRecipe(*entry, root, adaptedOperands, *rewriter))) {
        replayed = true;
        return success();
      }
      noteRecipeMaterializeFail();
    }
    return failure();
  }

  LogicalResult finish(LogicalResult result) {
    return finish(result, ValueRange());
  }

  LogicalResult finish(LogicalResult result, ValueRange replacements) {
    if (failed(result) || !active || replayed) {
      return result;
    }
    if (!rootBlock) {
      noteRecipeUnsupported();
      return result;
    }
    if (rootResultCount != replacements.size()) {
      noteRecipeUnsupported();
      return result;
    }
    SmallVector<Operation *> insertedOps;
    for (Operation &op : *rootBlock) {
      if (!preexistingOps.contains(&op))
        insertedOps.push_back(&op);
    }
    SmallVector<Value> replacementValues(replacements.begin(),
                                         replacements.end());
    FailureOr<HelionCacheRecipeEntry> entry = buildRecipeTemplate(
        rootContext, rootLoc, adaptedOperands, insertedOps, replacementValues);
    if (failed(entry)) {
      noteRecipeUnsupported();
      return result;
    }
    storeRecipe(key, std::move(*entry));
    return result;
  }

  Operation *root;
  SmallVector<Value> adaptedOperands;
  ConversionPatternRewriter *rewriter;
  MLIRContext *rootContext;
  Location rootLoc;
  unsigned rootResultCount;
  Block *rootBlock;
  DenseSet<Operation *> preexistingOps;
  std::string key;
  bool active = false;
  bool replayed = false;
};

struct ActiveHelionCacheScope {
  explicit ActiveHelionCacheScope(HelionCacheLoweringContext *ctx) {
    previous = activeHelionCacheContext;
    activeHelionCacheContext = ctx;
  }
  ~ActiveHelionCacheScope() { activeHelionCacheContext = previous; }

  HelionCacheLoweringContext *previous = nullptr;
};

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addLegalDialect<cf::ControlFlowDialect>();
    addLegalDialect<mlir::triton::nvgpu::NVGPUDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();

    // Warp specialization is lowered later.
    addLegalOp<triton::gpu::WarpSpecializeOp>();
    addLegalOp<triton::gpu::WarpYieldOp>();
    addLegalOp<triton::gpu::WarpSpecializePartitionsOp>();
    addLegalOp<triton::gpu::WarpReturnOp>();
  }
};

struct ConvertTritonGPUToLLVM
    : public triton::impl::ConvertTritonGPUToLLVMBase<ConvertTritonGPUToLLVM> {
  using ConvertTritonGPUToLLVMBase::ConvertTritonGPUToLLVMBase;

  ConvertTritonGPUToLLVM(int32_t computeCapability)
      : ConvertTritonGPUToLLVMBase({computeCapability}) {}
  ConvertTritonGPUToLLVM(int32_t computeCapability, int32_t ptxVersion)
      : ConvertTritonGPUToLLVMBase({computeCapability, ptxVersion}) {}
  ConvertTritonGPUToLLVM(int32_t computeCapability, int32_t ptxVersion,
                         int32_t cacheLevel)
      : ConvertTritonGPUToLLVMBase(
            {computeCapability, ptxVersion, cacheLevel}) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    TargetInfo targetInfo(computeCapability, ptxVersion);
    HelionCacheLoweringContext helionCacheContext{
        mod, computeCapability, ptxVersion, cacheLevel,
        makeModuleContext(mod, computeCapability, ptxVersion),
        DenseMap<Value, uint64_t>()};
    ActiveHelionCacheScope helionCacheScope(
        cacheLevel > 0 ? &helionCacheContext : nullptr);
    if (cacheLevel > 0)
      observeHelionCacheSignatures(helionCacheContext);

    // Allocate shared memory and set barrier
    ModuleAllocation allocation(
        mod, mlir::triton::nvidia_gpu::getNvidiaAllocationAnalysisScratchSizeFn(
                 targetInfo));
    ModuleMembarAnalysis membarPass(&allocation);
    membarPass.run();

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);

    // Lower functions
    TritonLLVMFunctionConversionTarget funcTarget(*context);
    RewritePatternSet funcPatterns(context);
    mlir::triton::populateFuncOpConversionPattern(
        typeConverter, funcPatterns, targetInfo, patternBenefitDefault);
    if (failed(
            applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
      return signalPassFailure();

    // initSharedMemory is run before the conversion of call and ret ops,
    // because the call op has to know the shared memory base address of each
    // function
    initSharedMemory(typeConverter);
    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    RewritePatternSet patterns(context);
    int benefit = patternBenefitPrioritizeOverLLVMConversions;
    mlir::triton::NVIDIA::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);
    mlir::triton::NVIDIA::populateTensorMemorySubviewOpToLLVMPattern(
        typeConverter, patterns, patternBenefitNvidiaTensorCoreSubviewPattern);
    mlir::triton::NVIDIA::populateTMAToLLVMPatterns(typeConverter, targetInfo,
                                                    patterns, benefit);
    populateDotOpToLLVMPatterns(typeConverter, patterns, computeCapability,
                                benefit);
    populateElementwiseOpToLLVMPatterns(typeConverter, patterns,
                                        axisInfoAnalysis, computeCapability,
                                        targetInfo, benefit);
    populateClampFOpToLLVMPattern(typeConverter, patterns, axisInfoAnalysis,
                                  computeCapability,
                                  patternBenefitClampOptimizedPattern);
    populateLoadStoreOpToLLVMPatterns(typeConverter, targetInfo,
                                      computeCapability, patterns,
                                      axisInfoAnalysis, benefit);
    mlir::triton::populateReduceOpToLLVMPatterns(typeConverter, patterns,
                                                 targetInfo, benefit);
    mlir::triton::populateScanOpToLLVMPatterns(typeConverter, patterns,
                                               targetInfo, benefit);
    mlir::triton::populateGatherOpToLLVMPatterns(typeConverter, patterns,
                                                 targetInfo, benefit);
    populateBarrierOpToLLVMPatterns(typeConverter, patterns, benefit,
                                    targetInfo);
    populateTensorPtrOpsToLLVMPatterns(typeConverter, patterns, benefit);
    populateClusterOpsToLLVMPatterns(typeConverter, patterns, benefit);
    mlir::triton::populateHistogramOpToLLVMPatterns(typeConverter, patterns,
                                                    targetInfo, benefit);
    mlir::triton::populatePrintOpToLLVMPattern(typeConverter, patterns,
                                               targetInfo, benefit);
    mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                     targetInfo, benefit);
    mlir::triton::NVIDIA::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                                      benefit);
    mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                              targetInfo, benefit);
    // TODO(thomas): this should probably be done in a separate step to not
    // interfere with our own lowering of arith ops. Add arith/math's patterns
    // to help convert scalar expression to LLVM.
    mlir::arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateGpuToNVVMConversionPatterns(typeConverter, patterns);
    mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);
    mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                               benefit);
    mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns,
                                                targetInfo, benefit);
    mlir::triton::NVIDIA::populateMemoryOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);
    mlir::triton::NVIDIA::populateTensorMemoryOpToLLVMPattern(
        typeConverter, patterns, benefit);
    mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                                   patterns, benefit);
    mlir::triton::NVIDIA::populateTCGen5MMAOpToLLVMPattern(typeConverter,
                                                           patterns, benefit);
    mlir::triton::NVIDIA::populateFp4ToFpToLLVMPatterns(typeConverter, patterns,
                                                        benefit);
    mlir::triton::populateInstrumentationToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);

    TritonLLVMConversionTarget convTarget(*context);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();

    // Lower CF ops separately to avoid breaking analysis.
    TritonLLVMFunctionConversionTarget cfTarget(*context);
    cfTarget.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return op->getDialect() !=
             context->getLoadedDialect<cf::ControlFlowDialect>();
    });
    RewritePatternSet cfPatterns(context);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          cfPatterns);
    if (failed(applyPartialConversion(mod, cfTarget, std::move(cfPatterns))))
      return signalPassFailure();

    // Fold CTAId when there is only 1 CTA.
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    if (numCTAs == 1) {
      mod.walk([](triton::nvgpu::ClusterCTAIdOp id) {
        OpBuilder b(id);
        Value zero = LLVM::createConstantI32(id->getLoc(), b, 0);
        id.replaceAllUsesWith(zero);
      });
    }
    fixUpLoopAnnotation(mod);

    // Ensure warp group code is isolated from above.
    makeAllWarpGroupsIsolatedFromAbove(mod);
  }

private:
  void initSharedMemory(LLVMTypeConverter &typeConverter) {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    // Set array size 0 and external linkage indicates that we use dynamic
    // shared allocation to allow a larger shared memory size for each kernel.
    //
    // Ask for 16B alignment on global_smem because that's the largest we should
    // ever need (4xi32).
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    b.create<LLVM::GlobalOp>(
        loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::External,
        "global_smem", /*value=*/Attribute(), /*alignment=*/16,
        // Add ROCm support.
        static_cast<unsigned>(NVVM::NVVMMemorySpace::kSharedMemorySpace));
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {
namespace NVIDIA {

HelionCacheRecipeGuard::HelionCacheRecipeGuard(
    Operation *root, ValueRange adaptedOperands,
    ConversionPatternRewriter &rewriter)
    : impl(new RecipeGuardImpl(root, adaptedOperands, rewriter)) {}

HelionCacheRecipeGuard::~HelionCacheRecipeGuard() {
  delete static_cast<RecipeGuardImpl *>(impl);
  impl = nullptr;
}

LogicalResult HelionCacheRecipeGuard::tryReplay() {
  if (!impl)
    return failure();
  return static_cast<RecipeGuardImpl *>(impl)->tryReplay();
}

LogicalResult HelionCacheRecipeGuard::finish(LogicalResult result) {
  if (!impl)
    return result;
  return static_cast<RecipeGuardImpl *>(impl)->finish(result);
}

LogicalResult HelionCacheRecipeGuard::finish(LogicalResult result,
                                             ValueRange replacements) {
  if (!impl)
    return result;
  return static_cast<RecipeGuardImpl *>(impl)->finish(result, replacements);
}

} // namespace NVIDIA
} // namespace triton
} // namespace mlir

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonGPUToLLVMPass() {
  return std::make_unique<ConvertTritonGPUToLLVM>();
}
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(int32_t computeCapability) {
  return std::make_unique<ConvertTritonGPUToLLVM>(computeCapability);
}
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToLLVMPass(int32_t computeCapability,
                                 int32_t ptxVersion) {
  return std::make_unique<ConvertTritonGPUToLLVM>(computeCapability,
                                                  ptxVersion);
}
std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonGPUToLLVMPass(
    int32_t computeCapability, int32_t ptxVersion, int32_t cacheLevel) {
  return std::make_unique<ConvertTritonGPUToLLVM>(computeCapability,
                                                  ptxVersion, cacheLevel);
}

HelionCacheTritonGPUToLLVMStats getHelionCacheTritonGPUToLLVMStats() {
  std::lock_guard<std::mutex> lock(helionCacheStatsMutex);
  return helionCacheStats;
}

void resetHelionCacheTritonGPUToLLVMStats() {
  std::lock_guard<std::mutex> lock(helionCacheStatsMutex);
  helionCacheMacroSignatures.clear();
  helionCacheMerkleNodes.clear();
  {
    std::lock_guard<std::mutex> recipeLock(helionCacheRecipeMutex);
    helionCacheRecipes.clear();
    helionCacheRecipeLru.clear();
    helionCacheRecipeBytes = 0;
  }
  helionCacheStats = {};
}

bool NVIDIA::canSkipBarSync(Operation *before, Operation *after) {
  // Multiple init barriers on the same allocation would usually not happen but
  // that allows us to avoid barriers between multiple subslice of an array of
  // mbarriers. This is still correct even if the inits happen on the same
  // allocation.
  if (isa<triton::nvidia_gpu::InitBarrierOp>(before) &&
      isa<triton::nvidia_gpu::InitBarrierOp>(after))
    return true;

  if (isa<triton::nvidia_gpu::InvalBarrierOp>(before) &&
      isa<triton::nvidia_gpu::InvalBarrierOp>(after))
    return true;

  //  We can't have a warp get ahead when we have a chain of mbarrier wait so we
  //  need a barrier in between two WaitBarrierOp.
  if (isa<triton::nvidia_gpu::WaitBarrierOp>(before) &&
      isa<triton::nvidia_gpu::WaitBarrierOp>(after))
    return false;

  // Even though WaitBarrierOp, AsyncTMACopyGlobalToLocalOp and
  // AsyncTMACopyGlobalToLocalOp read and write to the mbarrier allocation it is
  // valid for them to happen in different order on different threads, therefore
  // we don't need a barrier between those operations.
  if (isa<triton::nvidia_gpu::WaitBarrierOp,
          triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp,
          triton::nvidia_gpu::AsyncTMAGatherOp,
          triton::nvidia_gpu::BarrierExpectOp>(before) &&
      isa<triton::nvidia_gpu::WaitBarrierOp,
          triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp,
          triton::nvidia_gpu::AsyncTMAGatherOp,
          triton::nvidia_gpu::BarrierExpectOp>(after))
    return true;

  // A mbarrier wait is released only when the whole operations is done,
  // therefore any thread can access the memory after the barrier even if some
  // threads haven't reached the mbarrier wait.
  if (isa<triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp,
          triton::nvidia_gpu::AsyncTMAGatherOp,
          triton::nvidia_gpu::WaitBarrierOp>(before) &&
      !isa<triton::nvidia_gpu::InvalBarrierOp>(after))
    return true;

  return false;
}

} // namespace triton
} // namespace mlir
