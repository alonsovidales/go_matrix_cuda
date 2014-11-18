// Packae mt
// Linear algebra functions to work with matrix

package mt

import (
	"github.com/barnex/cuda5/cu"
	"math"
	"unsafe"
)

// CudaMatrix This struct represents a matrix allocated in the memory of the CUDA device
type CudaMatrix struct {
	m cu.DevicePtr
	w int
	h int
}

// Set the ptx files to be used from the cuda_modules directory if true, or the
// go kernels if false
const DEBUG = false

var cudaDevice = 0

var currentBuff = "default"
var usedMem = map[string]map[cu.DevicePtr]bool{
	currentBuff: make(map[cu.DevicePtr]bool),
}

var addBiasTopMod, multMod, subMod, addMod, multAllMod, negMatrixMod, setBiasToZeroMod,
	multTransMod, multByMod, removeBiasTopMod, transMod, sigmoidMatrixMod,
	logMatrixMod, oneMinusMod, addBiasMod, removeBias, powTwoMod, sigmoidGradMod,
	sumAll cu.Function
var maxNumThreads int
var cudaInitialized = false
var ctx cu.Context
var dev cu.Device

// InitCuda Initialize the CUDA driver, and loads all the CUDA kernels on the
// graphic card
func InitCuda() {
	if !cudaInitialized {
		var mod cu.Module

		cu.Init(0)
		dev = cu.DeviceGet(cudaDevice)
		maxNumThreads = dev.Attribute(cu.MAX_THREADS_PER_BLOCK)

		ctx = cu.CtxCreate(cu.CTX_SCHED_AUTO, dev)
		ctx.SetCurrent()

		if DEBUG {
			mod = cu.ModuleLoad("/cuda_modules/matrix_mult.ptx")
			multMod = mod.GetFunction("matrixMul")
			mod = cu.ModuleLoad("/cuda_modules/matrix_sub.ptx")
			subMod = mod.GetFunction("matrixSub")
			mod = cu.ModuleLoad("/cuda_modules/matrix_add.ptx")
			addMod = mod.GetFunction("matrixAdd")
			mod = cu.ModuleLoad("/cuda_modules/matrix_mult_all.ptx")
			multAllMod = mod.GetFunction("matrixMultAll")
			mod = cu.ModuleLoad("/cuda_modules/matrix_neg.ptx")
			negMatrixMod = mod.GetFunction("matrixNeg")
			mod = cu.ModuleLoad("/cuda_modules/matrix_mult_trans.ptx")
			multTransMod = mod.GetFunction("matrixMulTrans")
			mod = cu.ModuleLoad("/cuda_modules/matrix_sigmoid.ptx")
			sigmoidMatrixMod = mod.GetFunction("matrixSigmoid")
			mod = cu.ModuleLoad("/cuda_modules/matrix_log.ptx")
			logMatrixMod = mod.GetFunction("matrixLog")
			mod = cu.ModuleLoad("/cuda_modules/matrix_one_minus.ptx")
			oneMinusMod = mod.GetFunction("matrixOneMinus")
			mod = cu.ModuleLoad("/cuda_modules/matrix_add_bias.ptx")
			addBiasMod = mod.GetFunction("matrixAddBias")
			mod = cu.ModuleLoad("/cuda_modules/matrix_remove_bias.ptx")
			removeBias = mod.GetFunction("matrixRemoveBias")
			mod = cu.ModuleLoad("/cuda_modules/matrix_pow_two.ptx")
			powTwoMod = mod.GetFunction("matrixPowTwo")
			mod = cu.ModuleLoad("/cuda_modules/matrix_sigmoid_gradient.ptx")
			sigmoidGradMod = mod.GetFunction("matrixSigmoidGrad")
			mod = cu.ModuleLoad("/cuda_modules/matrix_mult_by.ptx")
			multByMod = mod.GetFunction("matrixMultBy")
			mod = cu.ModuleLoad("/cuda_modules/matrix_add_bias_top.ptx")
			addBiasTopMod = mod.GetFunction("matrixAddBiasTop")
			mod = cu.ModuleLoad("/cuda_modules/matrix_remove_bias_top.ptx")
			removeBiasTopMod = mod.GetFunction("matrixRemoveBiasTop")
			mod = cu.ModuleLoad("/cuda_modules/matrix_trans.ptx")
			transMod = mod.GetFunction("matrixTrans")
			mod = cu.ModuleLoad("/cuda_modules/matrix_sum_all.ptx")
			sumAll = mod.GetFunction("matrixSumAll")
			mod = cu.ModuleLoad("/cuda_modules/matrix_set_bias_to_zero.ptx")
			setBiasToZeroMod = mod.GetFunction("matrixSetBiasToZero")
		} else {
			mod = cu.ModuleLoadData(KER_MATRIX_MULT)
			multMod = mod.GetFunction("matrixMul")
			mod = cu.ModuleLoadData(KER_MATRIX_SUB)
			subMod = mod.GetFunction("matrixSub")
			mod = cu.ModuleLoadData(KER_MATRIX_ADD)
			addMod = mod.GetFunction("matrixAdd")
			mod = cu.ModuleLoadData(KER_MATRIX_MULT_ALL)
			multAllMod = mod.GetFunction("matrixMultAll")
			mod = cu.ModuleLoadData(KER_MATRIX_NEG)
			negMatrixMod = mod.GetFunction("matrixNeg")
			mod = cu.ModuleLoadData(KER_MATRIX_MULT_TRANS)
			multTransMod = mod.GetFunction("matrixMulTrans")
			mod = cu.ModuleLoadData(KER_MATRIX_SIGMOID)
			sigmoidMatrixMod = mod.GetFunction("matrixSigmoid")
			mod = cu.ModuleLoadData(KER_MATRIX_LOG)
			logMatrixMod = mod.GetFunction("matrixLog")
			mod = cu.ModuleLoadData(KER_MATRIX_ONE_MINUS)
			oneMinusMod = mod.GetFunction("matrixOneMinus")
			mod = cu.ModuleLoadData(KER_MATRIX_ADD_BIAS)
			addBiasMod = mod.GetFunction("matrixAddBias")
			mod = cu.ModuleLoadData(KER_MATRIX_REMOVE_BIAS)
			removeBias = mod.GetFunction("matrixRemoveBias")
			mod = cu.ModuleLoadData(KER_MATRIX_POW_TWO)
			powTwoMod = mod.GetFunction("matrixPowTwo")
			mod = cu.ModuleLoadData(KER_MATRIX_SIGMOID_GRADIENT)
			sigmoidGradMod = mod.GetFunction("matrixSigmoidGrad")
			mod = cu.ModuleLoadData(KER_MATRIX_MULT_BY)
			multByMod = mod.GetFunction("matrixMultBy")
			mod = cu.ModuleLoadData(KER_MATRIX_ADD_BIAS_TOP)
			addBiasTopMod = mod.GetFunction("matrixAddBiasTop")
			mod = cu.ModuleLoadData(KER_MATRIX_REMOVE_BIAS_TOP)
			removeBiasTopMod = mod.GetFunction("matrixRemoveBiasTop")
			mod = cu.ModuleLoadData(KER_MATRIX_TRANS)
			transMod = mod.GetFunction("matrixTrans")
			mod = cu.ModuleLoadData(KER_MATRIX_SUM_ALL)
			sumAll = mod.GetFunction("matrixSumAll")
			mod = cu.ModuleLoadData(KER_MATRIX_SET_BIAS_TO_ZERO)
			setBiasToZeroMod = mod.GetFunction("matrixSetBiasToZero")
		}

		cudaInitialized = true
	}

	// Ugly hack to prevent problems with the libraries and the context
	// handling
	if cu.CtxGetCurrent() == 0 && ctx != 0 {
		ctx.SetCurrent()
	}
}

// SetDevice Specify the CUDA device to be used, 0 by default
func SetDevice(dev int) {
	cudaDevice = dev
}

// StartBufferingMem Specifies a namespace to be used in order to define a
// space in memory, this namespace will be used by the garbage collector in
// order to release memory. @see FreeAllMem, FreeMem methods
func StartBufferingMem(buff string) {
	if buff != "" {
		currentBuff = buff
		usedMem[buff] = make(map[cu.DevicePtr]bool)
	}
}

// SetDefaultBuff Sets the default buffer to be used
func SetDefaultBuff() {
	currentBuff = "default"
}

// AddToBuff Adds a memory allocated by cu.MemAlloc to the memory used on this
// memory space
func AddToBuff(ptr cu.DevicePtr) {
	usedMem[currentBuff][ptr] = true
}

// FreeAllMem Releases all the memory allocated in all the namespaces
func FreeAllMem() {
	for _, buff := range usedMem {
		for m := range buff {
			cu.MemFree(m)
		}
	}
	usedMem = map[string]map[cu.DevicePtr]bool{
		currentBuff: make(map[cu.DevicePtr]bool),
	}
}

// FreeMem Releases the memory allocated in the current namespace
func FreeMem() {
	for m := range usedMem[currentBuff] {
		cu.MemFree(m)
		delete(usedMem[currentBuff], m)
	}
}

// Free Removes all the memory used to allocate a CudaMatrix
func (m *CudaMatrix) Free() {
	delete(usedMem[currentBuff], m.m)
	cu.MemFree(m.m)
}

// CudaMemAlloc Allocated memory in the CUDA device and returns a pointer to
// the allocated memory space
func CudaMemAlloc(bytes int64) cu.DevicePtr {
	p := cu.MemAlloc(bytes)
	AddToBuff(p)

	return p
}

// InitCudaMatrix Initializes a CUDA matrix struct allocating all the necesary memory on the CUDA device
func InitCudaMatrix(w int, h int) (m *CudaMatrix) {
	size := int64(w*h) * cu.SIZEOF_FLOAT64
	InitCuda()
	m = &CudaMatrix{
		w: w,
		h: h,
		m: CudaMemAlloc(size),
	}
	// Initialize this var to zeros
	aux := make([]float64, w*h)
	cu.MemcpyHtoD(m.m, unsafe.Pointer(&aux[0]), size)

	return
}

// H Returns the number of rows on a cuda matrix
func (m *CudaMatrix) H() int {
	return m.h
}

// W Returns the number of columns on a cuda matrix
func (m *CudaMatrix) W() int {
	return m.w
}

// CopyTo Copy a cuda matrix to a previously memory allocated cuda matrix, this
// copy operation is performed internally in the CUDA device
func (m *CudaMatrix) CopyTo(t *CudaMatrix) *CudaMatrix {
	size := int64(m.w*m.h) * cu.SIZEOF_FLOAT64
	InitCuda()
	if t.w == 0 && t.h == 0 {
		t.m = CudaMemAlloc(size)
		t.w = m.w
		t.h = m.h
	}
	cu.MemcpyDtoD(t.m, m.m, size)

	return t
}

// Copy Returns a copy of the cuda matrix in a new memory allocated space
// inside the same device. This operation is performed internally in the CUDA
// device
func (m *CudaMatrix) Copy() (r *CudaMatrix) {
	size := int64(m.w*m.h) * cu.SIZEOF_FLOAT64

	InitCuda()
	r = &CudaMatrix{
		m: CudaMemAlloc(size),
		w: m.w,
		h: m.h,
	}
	cu.MemcpyDtoD(r.m, m.m, size)

	return
}

// MoveToCuda Moves a two dimensional array from the memory of the machine to
// the memory of the CUDA device on a previously initialized CudaMatrix struct
func MoveToCuda(m [][]float64, p *CudaMatrix) {
	linealM := make([]float64, len(m)*len(m[0]))
	for i := 0; i < len(m); i++ {
		for j := 0; j < len(m[0]); j++ {
			linealM[(i*p.w)+j] = m[i][j]
		}
	}
	size := int64(len(linealM)) * cu.SIZEOF_FLOAT64

	InitCuda()
	if p.w == 0 && p.h == 0 {
		size := int64(len(m[0])*len(m)) * cu.SIZEOF_FLOAT64
		p.m = CudaMemAlloc(size)
		p.w = len(m[0])
		p.h = len(m)
	}
	cu.MemcpyHtoD(p.m, unsafe.Pointer(&linealM[0]), size)

	return
}

// GetCudaMatrix Initilizes and returns a CudaMatrix struct allocating the
// necessary memory on the cuda device, and copying from the machine memory to
// the CUDA device memory the especified matrix
func GetCudaMatrix(m [][]float64) (p *CudaMatrix) {
	p = &CudaMatrix{
		w: len(m[0]),
		h: len(m),
	}

	linealM := make([]float64, len(m)*len(m[0]))
	for i := 0; i < len(m); i++ {
		for j := 0; j < len(m[0]); j++ {
			linealM[(i*p.w)+j] = m[i][j]
		}
	}
	size := int64(len(linealM)) * cu.SIZEOF_FLOAT64

	InitCuda()
	p.m = CudaMemAlloc(size)
	cu.MemcpyHtoD(p.m, unsafe.Pointer(&linealM[0]), size)

	return
}

// TransOneDimMatrix Transposes a one dimensional CudaMatrix, this method
// doesn't performs any operation in the CUDA device, and is recomended to
// transform 1D matrices
// ex:
//   1 2 3 --> 1
//         --> 2
//         --> 3
func (m *CudaMatrix) TransOneDimMatrix() *CudaMatrix {
	m.w ^= m.h
	m.h ^= m.w
	m.w ^= m.h

	return m
}

// TransTo Transposes a CudaMatrix and stores the result in the provided
// CudaMatrix, if the CudaMatrix points to a previously allocated memory on the
// CUDA device, doesn't allocate new memory for the result
func (m *CudaMatrix) TransTo(rm *CudaMatrix) *CudaMatrix {
	InitCuda()

	if rm.w+rm.h == 0 {
		rm.w = m.h
		rm.h = m.w
		rm.m = CudaMemAlloc(int64(m.w*m.h) * cu.SIZEOF_FLOAT64)
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&rm.w),
		unsafe.Pointer(&rm.h),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(transMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return rm
}

// Trans Transposes a CudaMatrix and stores the result in a new CudaMatrix
// allocating the necessary memory on the CUDA device to it
func (m *CudaMatrix) Trans() (rm *CudaMatrix) {
	InitCuda()

	rm = &CudaMatrix{
		w: m.h,
		h: m.w,
		m: CudaMemAlloc(int64(m.w*m.h) * cu.SIZEOF_FLOAT64),
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&rm.w),
		unsafe.Pointer(&rm.h),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(transMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return
}

// CudaSync Blocks until the CUDA device has completed all preceding requested
// tasks
func CudaSync() {
	cu.CtxSynchronize()
}

// GetMatrixFromCuda Returns a two domensional array that represents a matrix
// allocated on the machine memory from a CudaMatrix. Downloads a matrix from
// the CUDA device
func (m *CudaMatrix) GetMatrixFromCuda() (r [][]float64) {
	buff := make([]float64, m.h*m.w)
	r = make([][]float64, m.h)

	InitCuda()
	cu.MemcpyDtoH(unsafe.Pointer(&buff[0]), m.m, int64(len(buff))*cu.SIZEOF_FLOAT64)
	for i := 0; i < m.h; i++ {
		r[i] = buff[i*m.w : (i+1)*m.w]
	}

	return
}

// MultAllElemsTo Multiplies one by one all the elements of two matrices,
// and uses the allocated memory on the third specified matrix to store the
// result, this method is much more faster than allocate new memory for the
// result, and can reutilize previously allocated memory on the CUDA device
// ex:
//   1 2 3 4 5 * 1 2 3 4 5 --> 1 4 9 16 25
//   1 2 3 4 5 * 1 2 3 4 5 --> 1 4 9 16 25
//   1 2 3 4 5 * 1 2 3 4 5 --> 1 4 9 16 25
func MultAllElemsTo(m1 *CudaMatrix, m2 *CudaMatrix, rm *CudaMatrix) *CudaMatrix {
	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(m1.w, m1.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),

		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	InitCuda()
	cu.CtxSynchronize()
	launchKernelSync(multAllMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)
	cu.CtxSynchronize()

	return rm
}

// MultAllElems Multiplies one by one all the elements of two matrices,
// and returns the result in a new allocated matrix on the CUDA device memory
// ex:
//   1 2 3 4 5 * 1 2 3 4 5 --> 1 4 9 16 25
//   1 2 3 4 5 * 1 2 3 4 5 --> 1 4 9 16 25
//   1 2 3 4 5 * 1 2 3 4 5 --> 1 4 9 16 25
func MultAllElems(m1 *CudaMatrix, m2 *CudaMatrix) (rm *CudaMatrix) {
	InitCuda()
	rm = &CudaMatrix{
		w: m2.w,
		h: m1.h,
		m: CudaMemAlloc(int64(m2.w*m1.h) * cu.SIZEOF_FLOAT64),
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(m1.w, m1.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),

		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(multAllMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return
}

// SetPosTo Modifies the value on a position given by the x and y coordinates
// by the given value on a cuda matrix
func (m *CudaMatrix) SetPosTo(val float64, x int, y int) *CudaMatrix {
	buff := make([]float64, m.h*m.w)
	buffPoint := unsafe.Pointer(&buff[0])
	size := int64(len(buff)) * cu.SIZEOF_FLOAT64
	InitCuda()
	cu.MemcpyDtoH(buffPoint, m.m, size)
	buff[(y*m.w)+x] = val
	cu.MemcpyHtoD(m.m, buffPoint, size)

	return m
}

// MultBy Multiply all the elements of a matrix by a given float and returns a
// new CudaMatrix allocating memory on the CUDA device to store the result
func (m *CudaMatrix) MultBy(by float64) *CudaMatrix {
	InitCuda()
	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(m.w, m.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&by),
		unsafe.Pointer(&m.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(multByMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return m
}

// SumAll Returns the sum of all the elements in a CudaMatrix
// ex:
// 1 2 3 4 5
// 6 7 8 9 10
// --> 55
func (m *CudaMatrix) SumAll() float64 {
	// Note: Don't split in blocks
	InitCuda()

	cu.CtxSynchronize()
	size := m.w * m.h
	matrixSplits := int(math.Ceil(float64(size) / float64(maxNumThreads)))

	sumP := cu.MemAllocHost(cu.SIZEOF_FLOAT64)
	args := []unsafe.Pointer{
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&matrixSplits),
		unsafe.Pointer(&size),
		unsafe.Pointer(&sumP),
	}

	threads := maxNumThreads
	if size < maxNumThreads {
		threads = maxNumThreads
	}
	launchKernelSync(sumAll, 1, 1, 1, threads, 1, 1, 0, 0, args)
	cu.CtxSynchronize()

	return *(*float64)(sumP)
}

// Log Apply the log to all the elements of the CudaMatrix
func (m *CudaMatrix) Log() *CudaMatrix {
	m.applyFunc(logMatrixMod)

	return m
}

// SigmoidGradient Applies the sigmoid gradient to all the elements of the
// CudaMatrix
// with z as element: sigmoid(z) * (1 - sigmoid(z));
func (m *CudaMatrix) SigmoidGradient() *CudaMatrix {
	m.applyFunc(sigmoidGradMod)

	return m
}

// Sigmoid Applies the sigmoid function to all the elements of the
// CudaMatrix
// with z as element: 1.0 / (1.0 + exp(-z));
func (m *CudaMatrix) Sigmoid() *CudaMatrix {
	m.applyFunc(sigmoidMatrixMod)

	return m
}

// OneMinus Applies 1 - n where n is each element of the matrix on the same
// given CudaMatrix
// ex:
//    1 2 3 4 5 --> 0 -1 -2 -3 -4
//    1 2 3 4 5 --> 0 -1 -2 -3 -4
func (m *CudaMatrix) OneMinus() *CudaMatrix {
	m.applyFunc(oneMinusMod)

	return m
}

// PowTwo Multiplies all the elements of a CudaMatrix by themselves
func (m *CudaMatrix) PowTwo() *CudaMatrix {
	m.applyFunc(powTwoMod)

	return m
}

// Neg Negates all the elements of the CudaMatrix
// ex:
//    1 2 3 4 5 --> -1 -2 -3 -4 -5
//    1 2 3 4 5 --> -1 -2 -3 -4 -5
func (m *CudaMatrix) Neg() *CudaMatrix {
	m.applyFunc(negMatrixMod)

	return m
}

// Mult Multiplies two matrices, and returns the result in a new CudaMatrix
// allocating memory to store the result on the CUDA device memory
func Mult(m1 *CudaMatrix, m2 *CudaMatrix) (rm *CudaMatrix) {
	var resH, resW int
	InitCuda()
	rm = &CudaMatrix{
		w: m2.w,
		h: m1.h,
		m: CudaMemAlloc(int64(m2.w*m1.h) * cu.SIZEOF_FLOAT64),
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),
		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&m2.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(multMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return
}

// MultTo Multiplies two matrices, and returns the result in the given
// CudaMatrix without allocate memory on the CUDA device
func MultTo(m1 *CudaMatrix, m2 *CudaMatrix, rm *CudaMatrix) *CudaMatrix {
	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),
		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&m2.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	InitCuda()
	launchKernelSync(multMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return rm
}

// CudaSubTo Matrix subtraction, returns the result in the provided Cuda Matrix
// without need to allocate new memory to store it
func CudaSubTo(m1 *CudaMatrix, m2 *CudaMatrix, rm *CudaMatrix) *CudaMatrix {
	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),
		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	InitCuda()
	cu.CtxSynchronize()
	launchKernelSync(subMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)
	cu.CtxSynchronize()

	return rm
}

// CudaSub Matrix subtraction, returns the result in a new CudaMatrix
// allocating memory on the CUDA device to store it
func CudaSub(m1 *CudaMatrix, m2 *CudaMatrix) (rm *CudaMatrix) {
	InitCuda()
	rm = &CudaMatrix{
		w: m1.w,
		h: m1.h,
		m: CudaMemAlloc(int64(m1.w*m1.h) * cu.SIZEOF_FLOAT64),
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),
		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(subMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return
}

// CudaSumTo Sums two matrices and returns the result in the provided Cuda
// Matrix without need to allocate new memory to store it
func CudaSumTo(m1 *CudaMatrix, m2 *CudaMatrix, rm *CudaMatrix) *CudaMatrix {
	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),
		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	InitCuda()
	launchKernelSync(addMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return rm
}

// CudaSum Sums two matrices, returns the result in a new CudaMatrix
// allocating memory on the CUDA device to store it
func CudaSum(m1 *CudaMatrix, m2 *CudaMatrix) (rm *CudaMatrix) {
	InitCuda()
	rm = &CudaMatrix{
		w: m1.w,
		h: m1.h,
		m: CudaMemAlloc(int64(m1.w*m1.h) * cu.SIZEOF_FLOAT64),
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),
		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(addMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return
}

// MultTransTo Multiply a CudaMatrix by the transpose of another CudaMatrix and
// returns the result in the provided CudaMatrix without need to allocate new
// memory to store it
func MultTransTo(m1 *CudaMatrix, m2 *CudaMatrix, rm *CudaMatrix) *CudaMatrix {
	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	InitCuda()
	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),

		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&rm.w),

		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(multTransMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return rm
}

// MultTrans Multiply a CudaMatrix by the transpose of another CudaMatrix and
// returns the result in a new CudaMatrix allocating memory on the CUDA device
// to store it
func MultTrans(m1 *CudaMatrix, m2 *CudaMatrix) (rm *CudaMatrix) {
	InitCuda()
	rm = &CudaMatrix{
		w: m2.h,
		h: m1.h,
		m: CudaMemAlloc(int64(m2.h*m1.h) * cu.SIZEOF_FLOAT64),
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),

		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&rm.w),

		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(multTransMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return
}

// Det Calculates the determinant of the matrix
// TODO Implement using CUDA
func Det(m [][]float64) (rm float64) {
	// Sum diagonals
	ml := len(m)
	sums := make([]float64, ml*2)
	for i := 0; i < len(sums); i++ {
		sums[i] = 1
	}

	for r := 0; r < ml; r++ {
		for c := 0; c < ml; c++ {
			if c-r < 0 {
				sums[ml+c-r] *= m[c][r]
			} else {
				sums[c-r] *= m[c][r]
			}

			if c+r >= ml {
				sums[c+r] *= m[c][r]
			} else {
				sums[c+r+ml] *= m[c][r]
			}
		}
	}

	to := len(sums)
	if ml == 2 {
		to = 2
		ml = 1
	}
	for i := 0; i < to; i++ {
		if i >= ml {
			rm -= sums[i]
		} else {
			rm += sums[i]
		}
	}
	return
}

// Minors Returns the minors matrix
// TODO Implement using CUDA
func Minors(m [][]float64) (rm [][]float64) {
	ml := len(m)
	rm = make([][]float64, ml)
	for r := 0; r < ml; r++ {
		rm[r] = make([]float64, ml)
		for c := 0; c < ml; c++ {
			auxM := [][]float64{}
			for ra := 0; ra < ml; ra++ {
				if ra != r {
					auxR := []float64{}
					for ca := 0; ca < ml; ca++ {
						if ca != c {
							auxR = append(auxR, m[ra][ca])
						}
					}
					auxM = append(auxM, auxR)
				}
			}
			rm[r][c] = Det(auxM)
		}
	}

	return
}

// Cofactors Returns the cofactors matrix
// TODO Implement using CUDA
func Cofactors(m [][]float64) (rm [][]float64) {
	rm = make([][]float64, len(m))
	for r := 0; r < len(m); r++ {
		rm[r] = make([]float64, len(m[0]))
		for c := 0; c < len(m[0]); c++ {
			if (c+r)%2 == 0 {
				rm[r][c] = m[r][c]
			} else {
				rm[r][c] = -m[r][c]
			}
		}
	}

	return
}

// Inv Calculates the inverse matrix
// TODO Implement using CUDA
func Inv(m [][]float64) (rm [][]float64) {
	dm := Det(m)
	adj := trans(Cofactors(Minors(m)))

	rm = multBy(adj, 1.0/dm)

	return
}

// Div Divide the first matrix by the second one
// TODO Implement using CUDA
func Div(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	return mult(m1, Inv(m2))
}

// SumAll all the elements in a matrix
func SumAll(m [][]float64) (rm float64) {
	for i := 0; i < len(m); i++ {
		for j := 0; j < len(m[0]); j++ {
			rm += m[i][j]
		}
	}

	return
}

// Apply a function to all the elements of a matrix, the function will receive a
// float64 as param and returns a float64 too
func Apply(m [][]float64, f func(x float64) float64) (rm [][]float64) {
	rm = make([][]float64, len(m))

	// Initialize the matrix
	for x := 0; x < len(m); x++ {
		rm[x] = make([]float64, len(m[0]))
		for y := 0; y < len(m[0]); y++ {
			rm[x][y] = f(m[x][y])
		}
	}

	return
}

// Concat Concatenates two matrix elements, ex:
// m1 = (M111, M112, M113)
//      (M121, M122, M123)
//      (M131, M132, M133)
// m2 = (M211, M212, M213)
//      (M221, M222, M223)
//      (M231, M232, M233)
// rm = (M111, M112, M113, M221, M222, M223)
//      (M121, M122, M123, M221, M222, M223)
//      (M131, M132, M133, M231, M232, M233)
func Concat(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	rm = make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		rm[i] = make([]float64, len(m1[i])+len(m2[i]))
		for j := 0; j < len(m1[i]); j++ {
			rm[i][j] = m1[i][j]
		}
		for j := 0; j < len(m2[i]); j++ {
			rm[i][j+len(m1[i])] = m2[i][j]
		}
	}

	return
}

// RemoveBiasTo Method for machine learning: Removes the last column of a
// matrix and uses the given CudaMatrix to allocate the result, if the
// CudaMatrix CUDA memory was previously allocates, reuses the memory without
// need to perform an alocation
// ex:
//   1 2 3 4 5 1  --> 1 2 3 4 5
//   1 2 3 4 5 1  --> 1 2 3 4 5
//   1 2 3 4 5 1  --> 1 2 3 4 5
func (m *CudaMatrix) RemoveBiasTo(rm *CudaMatrix) *CudaMatrix {
	if rm.w == 0 && rm.h == 0 {
		rm.w = m.w - 1
		rm.h = m.h
		rm.m = CudaMemAlloc(int64(rm.w*rm.h) * cu.SIZEOF_FLOAT64)
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&rm.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(removeBias, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return rm
}

// RemoveBias Method for machine learning: Removes the last column of a
// matrix and returns a new CudaMatrix performing the memory allocation in the
// CUDA device
// ex:
//   1 2 3 4 5 1  --> 1 2 3 4 5
//   1 2 3 4 5 1  --> 1 2 3 4 5
//   1 2 3 4 5 1  --> 1 2 3 4 5
func (m *CudaMatrix) RemoveBias() (rm *CudaMatrix) {
	InitCuda()

	rm = &CudaMatrix{
		w: m.w - 1,
		h: m.h,
		m: CudaMemAlloc(int64((m.w-1)*m.h) * cu.SIZEOF_FLOAT64),
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&rm.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(removeBias, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return
}

// AddBias Method for machine learning: Adds a bias column setted to one to a
// given CudaMatrix and returns a new CudaMatrix allocating memory on the CUDA
// device to store the resulting matrix
// ex:
//   1 2 3 4 5  --> 1 2 3 4 5 1
//   1 2 3 4 5  --> 1 2 3 4 5 1
//   1 2 3 4 5  --> 1 2 3 4 5 1
func (m *CudaMatrix) AddBias() (rm *CudaMatrix) {
	InitCuda()

	rm = &CudaMatrix{
		w: m.w + 1,
		h: m.h,
		m: CudaMemAlloc(int64((m.w+1)*m.h) * cu.SIZEOF_FLOAT64),
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&rm.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(addBiasMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return
}

// AddBiasTo Method for machine learning: Adds a bias column setted to one to a
// given CudaMatrix and stores the result in the given matrix withouth allocate
// new memory on the CUDA device
// ex:
//   1 2 3 4 5  --> 1 2 3 4 5 1
//   1 2 3 4 5  --> 1 2 3 4 5 1
//   1 2 3 4 5  --> 1 2 3 4 5 1
func (m *CudaMatrix) AddBiasTo(rm *CudaMatrix) *CudaMatrix {
	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&rm.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&rm.w),
		unsafe.Pointer(&resultSize),
	}

	InitCuda()
	launchKernelSync(addBiasMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return rm
}

// RemoveBiasTopTo Method for machine learning: Removes a bias row from the top
// of the given matrix, and stores the result on the given CudaMatrix without
// allocate new memory on the CUDA device
// ex:
//   1 1 1 1 1
//   1 2 3 4 5  --> 1 2 3 4 5
//   1 2 3 4 5  --> 1 2 3 4 5
func (m *CudaMatrix) RemoveBiasTopTo(rm *CudaMatrix) *CudaMatrix {
	InitCuda()

	if rm.w+rm.h == 0 {
		rm.w = m.w
		rm.h = m.h - 1
		rm.m = CudaMemAlloc(int64(m.w*(m.h+1)) * cu.SIZEOF_FLOAT64)
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&rm.w),

		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(removeBiasTopMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return rm
}

// RemoveBiasTop Method for machine learning: Removes a bias row from the top
// of the given matrix, and stores the result on the given CudaMatrix without
// allocate new memory on the CUDA device
// ex:
//   1 1 1 1 1  -->
//   1 2 3 4 5  --> 1 2 3 4 5
//   1 2 3 4 5  --> 1 2 3 4 5
func (m *CudaMatrix) RemoveBiasTop() (rm *CudaMatrix) {
	InitCuda()

	rm = &CudaMatrix{
		w: m.w,
		h: m.h - 1,
		m: CudaMemAlloc(int64(m.w*(m.h+1)) * cu.SIZEOF_FLOAT64),
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&rm.w),

		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(removeBiasTopMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return
}

// AddBiasTopTo Method for machine learning:  Adds a new row at the top of the
// matrix filled by 1's, stores the result in the given CudaMatrix without
// allocate memory on the CUDA device if is not necessary
// ex:
//              --> 1 1 1 1 1
//   1 2 3 4 5  --> 1 2 3 4 5
//   1 2 3 4 5  --> 1 2 3 4 5
func (m *CudaMatrix) AddBiasTopTo(rm *CudaMatrix) *CudaMatrix {
	InitCuda()

	if rm.w == 0 && rm.h == 0 {
		rm.h = m.h + 1
		rm.w = m.w
		rm.m = CudaMemAlloc(int64(m.w*(m.h+1)) * cu.SIZEOF_FLOAT64)
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&rm.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(addBiasTopMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return rm
}

// AddBiasTop Method for machine learning:  Adds a new row at the top of the
// matrix filled by 1's, and returns a new CudaMatrix allocating the necessary
// memory on the CUDA device
// ex:
//              --> 1 1 1 1 1
//   1 2 3 4 5  --> 1 2 3 4 5
//   1 2 3 4 5  --> 1 2 3 4 5
func (m *CudaMatrix) AddBiasTop() (rm *CudaMatrix) {
	InitCuda()

	rm = &CudaMatrix{
		w: m.w,
		h: m.h + 1,
		m: CudaMemAlloc(int64(m.w*(m.h+1)) * cu.SIZEOF_FLOAT64),
	}

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(rm.w, rm.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&rm.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(addBiasTopMod, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)

	return
}

// SetBiasToZero Method for machine learning: Sets the right column to zero,
// this transformation is applied directly in the same CudaMatrix
// ex:
//    1 2 3 4 5 1 --> 1 2 3 4 5 0
//    1 2 3 4 5 1 --> 1 2 3 4 5 0
//    1 2 3 4 5 1 --> 1 2 3 4 5 0
func (m *CudaMatrix) SetBiasToZero() *CudaMatrix {
	InitCuda()
	var gridsH, resH int

	if m.h > maxNumThreads {
		gridsH = int(math.Ceil(float64(m.h) / float64(maxNumThreads)))
		resH = maxNumThreads
	} else {
		gridsH = 1
		resH = m.h
	}

	args := []unsafe.Pointer{
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&m.h),
		unsafe.Pointer(&m.w),
		unsafe.Pointer(&resH),
	}

	launchKernelSync(setBiasToZeroMod, 1, gridsH, 1, 1, resH, 1, 0, 0, args)

	return m
}

func multElems(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	cm1 := GetCudaMatrix(m1)
	cm2 := GetCudaMatrix(m2)

	result := MultAllElems(cm1, cm2)

	rm = result.GetMatrixFromCuda()

	return
}

func (m *CudaMatrix) applyFunc(function cu.Function) {
	InitCuda()

	resW, resH, gridsW, gridsH, resultSize := getGridThreadsFromSize(m.w, m.h)

	args := []unsafe.Pointer{
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&m.w),
		unsafe.Pointer(&resultSize),
	}

	launchKernelSync(function, gridsW, gridsH, 1, resW, resH, 1, 0, 0, args)
}

func launchKernelSync(f cu.Function, gridDimX, gridDimY, gridDimZ int, blockDimX, blockDimY, blockDimZ int, sharedMemBytes int, stream cu.Stream, kernelParams []unsafe.Pointer) {
	cu.LaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, stream, kernelParams)
	cu.CtxSynchronize()
}

func getGridThreadsFromSize(w, h int) (resW, resH, gridsW, gridsH, size int) {
	if w > h {
		if w > maxNumThreads {
			resW = maxNumThreads
			resH = 1
		} else {
			resW = w
			resH = int(float64(maxNumThreads) / float64(resW))
		}
	} else {
		if h > maxNumThreads {
			resH = maxNumThreads
			resW = 1
		} else {
			resH = h
			resW = int(float64(maxNumThreads) / float64(resH))
		}
	}
	if resW > w {
		resW = w
	}
	if resH > w {
		resH = h
	}

	gridsW = int(math.Ceil(float64(w) / float64(resW)))
	gridsH = int(math.Ceil(float64(h) / float64(resH)))

	size = w * h

	return
}

// Methods for test proposals only

func neg(m [][]float64) (rm [][]float64) {
	cm := GetCudaMatrix(m)

	cm.Neg()

	rm = cm.GetMatrixFromCuda()

	return
}

func mult(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	cm1 := GetCudaMatrix(m1)
	cm2 := GetCudaMatrix(m2)

	result := Mult(cm1, cm2)

	rm = result.GetMatrixFromCuda()

	return
}

func multNoCuda(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	rm = make([][]float64, len(m1))
	// Initialize the matrix
	for x := 0; x < len(m1); x++ {
		rm[x] = make([]float64, len(m2[0]))

		for y := 0; y < len(m2[0]); y++ {
			for k := 0; k < len(m2); k++ {
				rm[x][y] += m1[x][k] * m2[k][y]
			}
		}
	}

	return
}

func sub(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	cm1 := GetCudaMatrix(m1)
	cm2 := GetCudaMatrix(m2)

	result := CudaSub(cm1, cm2)

	rm = result.GetMatrixFromCuda()

	return
}

func subNoCuda(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	rm = make([][]float64, len(m1))
	// Initialize the matrix
	for x := 0; x < len(m1); x++ {
		rm[x] = make([]float64, len(m1[0]))

		for y := 0; y < len(m1[0]); y++ {
			rm[x][y] = m1[x][y] - m2[x][y]
		}
	}

	return
}

func sum(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	cm1 := GetCudaMatrix(m1)
	cm2 := GetCudaMatrix(m2)

	result := CudaSum(cm1, cm2)

	rm = result.GetMatrixFromCuda()

	return
}

// Returns the rm of multiply all the elements of a matrix by a float number
func multBy(m1 [][]float64, n float64) (rm [][]float64) {
	rm = make([][]float64, len(m1))
	// Initialize the matrix
	for x := 0; x < len(m1); x++ {
		rm[x] = make([]float64, len(m1[0]))

		for y := 0; y < len(m1[0]); y++ {
			rm[x][y] = m1[x][y] * n
		}
	}

	return
}

// Matrix Transpose, returns the transpose of the given square matrix
func trans(m1 [][]float64) (rm [][]float64) {
	if len(m1) == 0 {
		return [][]float64{}
	}
	rm = make([][]float64, len(m1[0]))

	// Initialize the matrix
	for x := 0; x < len(m1[0]); x++ {
		rm[x] = make([]float64, len(m1))

		for y := 0; y < len(m1); y++ {
			rm[x][y] = m1[y][x]
		}
	}

	return
}

// Returns the sum of the given two matrix
func sumNoCuda(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	rm = make([][]float64, len(m1))
	// Initialize the matrix
	for x := 0; x < len(m1); x++ {
		rm[x] = make([]float64, len(m1[0]))

		for y := 0; y < len(m1[0]); y++ {
			rm[x][y] = m1[x][y] + m2[x][y]
		}
	}

	return
}

// Multiply on matrix by the transpose of the second matrix
func multTrans(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	cm1 := GetCudaMatrix(m1)
	cm2 := GetCudaMatrix(m2)

	result := MultTrans(cm1, cm2)

	rm = result.GetMatrixFromCuda()

	return
}

// Copy Returns a copy of the matrix
func copyMatrix(m [][]float64) (rm [][]float64) {
	rm = make([][]float64, len(m))

	for i := 0; i < len(m); i++ {
		rm[i] = make([]float64, len(m[i]))
		for j := 0; j < len(m[i]); j++ {
			rm[i][j] = m[i][j]
		}
	}

	return
}
