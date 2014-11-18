package mt

import (
	"fmt"
	"testing"
)

func TestAddBiasTop(t *testing.T) {
	fmt.Println("TestAddBiasTop")
	m1 := GetCudaMatrix([][]float64{
		[]float64{1, 5, 5, 6},
		[]float64{1, 9, 8, 3},
		[]float64{-2, 3.5, 2, 4},
	})

	result := m1.AddBiasTop().GetMatrixFromCuda()

	expectedRes := [][]float64{
		[]float64{1, 1, 1, 1},
		[]float64{1, 5, 5, 6},
		[]float64{1, 9, 8, 3},
		[]float64{-2, 3.5, 2, 4},
	}

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result[0]); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestMultTrans(t *testing.T) {
	fmt.Println("TestMultTrans")
	height := 35
	width := 20
	m1 := make([][]float64, height)
	for i := 0; i < height; i++ {
		m1[i] = make([]float64, width)
		for j := 0; j < width; j++ {
			m1[i][j] = float64(i + j)
		}
	}
	height = 35
	width = 20
	m2 := make([][]float64, height)
	for i := 0; i < height; i++ {
		m2[i] = make([]float64, width)
		for j := 0; j < width; j++ {
			m2[i][j] = float64(i + j)
		}
	}

	expectedRes := mult(m1, trans(m2))
	result := multTrans(m1, m2)

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result[0]); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", j, i, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestMatrixBigSumAll(t *testing.T) {
	fmt.Println("TestMatrixBigSumAll")
	height := 2500
	width := 3
	m1 := make([][]float64, height)
	for i := 0; i < height; i++ {
		m1[i] = make([]float64, width)
		for j := 0; j < width; j++ {
			m1[i][j] = float64(i + j)
		}
	}

	cudaRes := GetCudaMatrix(m1).SumAll()
	expected := SumAll(m1)

	if cudaRes != expected {
		t.Error("Expected result for SumAll:", expected, "obtained:", cudaRes)
	}
}

func TestCudaTrans(t *testing.T) {
	fmt.Println("TestCudaTrans")
	m1 := GetCudaMatrix([][]float64{
		[]float64{1, 5, 5, 6},
		[]float64{1, 9, 8, 3},
		[]float64{-2, 3.5, 2, 4},
	})

	result := m1.Trans().GetMatrixFromCuda()

	expectedRes := [][]float64{
		[]float64{1, 1, -2},
		[]float64{5, 9, 3.5},
		[]float64{5, 8, 2},
		[]float64{6, 3, 4},
	}

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result[0]); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestCudaSetBiasToZero(t *testing.T) {
	fmt.Println("TestCudaSetBiasToZero")
	m1 := GetCudaMatrix([][]float64{
		[]float64{1, 5},
		[]float64{1, 9},
		[]float64{-2, 3.5},
	})

	result := m1.SetBiasToZero().GetMatrixFromCuda()

	expectedRes := [][]float64{
		[]float64{0, 5},
		[]float64{0, 9},
		[]float64{0, 3.5},
	}

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result[0]); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestCudaAddBias(t *testing.T) {
	fmt.Println("TestCudaAddBias")
	m1 := GetCudaMatrix([][]float64{
		[]float64{1, 5},
		[]float64{1, 9},
		[]float64{-2, 3.5},
	})

	result := m1.AddBias().GetMatrixFromCuda()

	expectedRes := [][]float64{
		[]float64{1, 1, 5},
		[]float64{1, 1, 9},
		[]float64{1, -2, 3.5},
	}

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result[0]); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}

	result = m1.AddBias().RemoveBias().GetMatrixFromCuda()

	expectedRes = [][]float64{
		[]float64{1, 5},
		[]float64{1, 9},
		[]float64{-2, 3.5},
	}

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result[0]); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestCudaNegMatrix(t *testing.T) {
	fmt.Println("TestCudaNegMatrix")
	m1 := [][]float64{
		[]float64{1, 5},
		[]float64{1, 9},
		[]float64{-2, 3.5},
	}

	result := neg(m1)

	expectedRes := [][]float64{
		[]float64{-1, -5},
		[]float64{-1, -9},
		[]float64{2, -3.5},
	}

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result[0]); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestCudaNoCudaMult(t *testing.T) {
	fmt.Println("TestCudaNoCudaMult")

	m1 := [][]float64{
		[]float64{1, 5},
		[]float64{1, 9},
		[]float64{1, 6},
		[]float64{1, 7},
		[]float64{1, 8},
		[]float64{1, 8},
		[]float64{1, 5},
		[]float64{1, 4},
		[]float64{1, 1},
		[]float64{1, 4},
		[]float64{1, 7},
		[]float64{1, 2},
	}
	m2 := [][]float64{
		[]float64{1},
		[]float64{3},
	}

	cudaRes := mult(m1, m2)
	expected := multNoCuda(m1, m2)
	for i := 0; i < len(cudaRes); i++ {
		for j := 0; j < len(cudaRes[0]); j++ {
			if cudaRes[i][j] != expected[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expected[i][j], "but obtained:", cudaRes[i][j])
			}
		}
	}
}

func TestMatrixBigMultiplication(t *testing.T) {
	fmt.Println("TestMatrixBigMultiplication")
	height := 40
	width := 60
	m1 := make([][]float64, height)
	for i := 0; i < height; i++ {
		m1[i] = make([]float64, width)
		for j := 0; j < width; j++ {
			m1[i][j] = float64(i + j)
		}
	}
	m2 := make([][]float64, width)
	for i := 0; i < width; i++ {
		m2[i] = make([]float64, height)
		for j := 0; j < height; j++ {
			m2[i][j] = float64(i + j)
		}
	}

	cudaRes := mult(m2, m1)
	expected := multNoCuda(m2, m1)

	for i := 0; i < width; i++ {
		for j := 0; j < width; j++ {
			if cudaRes[i][j] != expected[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expected[i][j], "but obtained:", cudaRes[i][j])
			}
		}
	}
}

func TestMatrixBigSub(t *testing.T) {
	fmt.Println("TestMatrixBigSub")
	height := 40
	width := 60
	m1 := make([][]float64, height)
	for i := 0; i < height; i++ {
		m1[i] = make([]float64, width)
		for j := 0; j < width; j++ {
			m1[i][j] = float64(i + j)
		}
	}
	m2 := make([][]float64, height)
	for i := 0; i < height; i++ {
		m2[i] = make([]float64, width)
		for j := 0; j < width; j++ {
			m2[i][j] = float64(i+j) + 0.5
		}
	}

	cudaRes := sub(m1, m2)
	expected := subNoCuda(m1, m2)

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			if cudaRes[i][j] != expected[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expected[i][j], "but obtained:", cudaRes[i][j])
			}
		}
	}
}

func TestCudaMatrixMultiplication(t *testing.T) {
	fmt.Println("TestCudaMatrixMultiplication")
	m1 := GetCudaMatrix([][]float64{
		[]float64{3, 2, 1},
		[]float64{9, 5, 7},
	})
	m2 := GetCudaMatrix([][]float64{
		[]float64{2, 3},
		[]float64{1, 4},
		[]float64{2, 9},
	})

	expectedRes := [][]float64{
		[]float64{10, 26},
		[]float64{37, 110},
	}

	result := Mult(m1, m2).GetMatrixFromCuda()

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestMatrixMultiplication(t *testing.T) {
	fmt.Println("TestMatrixMultiplication")
	m1 := [][]float64{
		[]float64{3, 2, 1},
		[]float64{9, 5, 7},
	}
	m2 := [][]float64{
		[]float64{2, 3},
		[]float64{1, 4},
		[]float64{2, 9},
	}

	expectedRes := [][]float64{
		[]float64{10, 26},
		[]float64{37, 110},
	}

	result := mult(m1, m2)

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestMatrixSum(t *testing.T) {
	fmt.Println("TestMatrixSum")
	m1 := [][]float64{
		[]float64{3, 2, 1},
		[]float64{9, 5, 7},
	}
	m2 := [][]float64{
		[]float64{2, 3, 4},
		[]float64{1, 4, 7},
	}

	expectedRes := [][]float64{
		[]float64{5, 5, 5},
		[]float64{10, 9, 14},
	}

	result := sum(m1, m2)

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestMultElems(t *testing.T) {
	fmt.Println("TestMultElems")
	m1 := [][]float64{
		[]float64{3, 2, 1},
		[]float64{9, 5, 7},
	}
	m2 := [][]float64{
		[]float64{2, 3, 4},
		[]float64{1, 4, 7},
	}

	expectedRes := [][]float64{
		[]float64{6, 6, 4},
		[]float64{9, 20, 49},
	}

	result := multElems(m1, m2)

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestMatrixSub(t *testing.T) {
	fmt.Println("TestMatrixSub")
	m1 := [][]float64{
		[]float64{3, 2, 1},
		[]float64{9, 5, 7},
	}
	m2 := [][]float64{
		[]float64{2, 3, 4},
		[]float64{1, 4, 7},
	}

	expectedRes := [][]float64{
		[]float64{1, -1, -3},
		[]float64{8, 1, 0},
	}

	result := sub(m1, m2)

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestMatrixTrans(t *testing.T) {
	fmt.Println("TestMatrixTrans")
	m1 := [][]float64{
		[]float64{3, 2, 1},
		[]float64{9, 5, 7},
	}

	expectedRes := [][]float64{
		[]float64{3, 9},
		[]float64{2, 5},
		[]float64{1, 7},
	}

	result := trans(m1)

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result[0]); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestSumAll(t *testing.T) {
	fmt.Println("TestSumAll")
	m := [][]float64{
		[]float64{3, 2, 1},
		[]float64{9, 5, 7},
	}

	result := SumAll(m)

	if result != 27 {
		t.Error("Expected result for SumAll: 27 but obtained:", result)
	}
}

func TestApply(t *testing.T) {
	fmt.Println("TestApply")
	m := [][]float64{
		[]float64{4, 2, 1},
		[]float64{8, 3, 6},
	}

	expectedRes := [][]float64{
		[]float64{2, 1, 0.5},
		[]float64{4, 1.5, 3},
	}

	f := func(x float64) float64 {
		return x / 2
	}

	result := Apply(m, f)

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result[0]); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestDet(t *testing.T) {
	fmt.Println("TestDet")
	m := [][]float64{
		[]float64{1, 3, 1},
		[]float64{1, 1, 2},
		[]float64{2, 3, 4},
	}

	d := Det(m)

	if d != -1 {
		t.Error("Expected Det result -1,", d, "obtained")
	}

	m = [][]float64{
		[]float64{3, 3},
		[]float64{5, 2},
	}

	d = Det(m)

	if d != -9 {
		t.Error("Expected Det result -4,", d, "obtained")
	}

	m = [][]float64{
		[]float64{3, 2, 1, 9},
		[]float64{2, 7, 5, 4},
		[]float64{3, 2, 9, 7},
		[]float64{4, 3, 3, 1},
	}

	d = Det(m)

	if d != -176 {
		t.Error("Expected Det result -176,", d, "obtained")
	}
}

func TestMinors(t *testing.T) {
	fmt.Println("TestMinors")
	m := [][]float64{
		[]float64{1, 3, 1},
		[]float64{1, 1, 2},
		[]float64{2, 3, 4},
	}

	expectedRes := [][]float64{
		[]float64{-2, 0, 1},
		[]float64{9, 2, -3},
		[]float64{5, 1, -2},
	}

	result := Minors(m)

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result[0]); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestCofactor(t *testing.T) {
	fmt.Println("TestCofactor")
	m := [][]float64{
		[]float64{1, 3, 1},
		[]float64{1, -1, -2},
		[]float64{2, 3, 4},
	}

	expectedRes := [][]float64{
		[]float64{1, -3, 1},
		[]float64{-1, -1, 2},
		[]float64{2, -3, 4},
	}

	result := Cofactors(m)

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result[0]); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestInv(t *testing.T) {
	fmt.Println("TestInv")
	m := [][]float64{
		[]float64{1, 3, 1},
		[]float64{1, 1, 2},
		[]float64{2, 3, 4},
	}

	expectedRes := [][]float64{
		[]float64{2, 9, -5},
		[]float64{0, -2, 1},
		[]float64{-1, -3, 2},
	}

	result := Inv(m)

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result[0]); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestDiv(t *testing.T) {
	fmt.Println("TestDiv")
	m1 := [][]float64{
		[]float64{1, 2, 3},
		[]float64{3, 2, 1},
		[]float64{2, 1, 3},
	}

	m2 := [][]float64{
		[]float64{4, 5, 6},
		[]float64{6, 5, 4},
		[]float64{4, 6, 5},
	}

	expectedRes := [][]float64{
		[]float64{7.0 / 10.0, 3.0 / 10.0, 0},
		[]float64{-3.0 / 10.0, 7.0 / 10.0, 0},
		[]float64{6.0 / 5.0, 1.0 / 5.0, -1},
	}

	result := Div(m1, m2)

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result); j++ {
			if result[i][j]-expectedRes[i][j] > 0.000001 {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestConcat(t *testing.T) {
	fmt.Println("TestConcat")
	m1 := [][]float64{
		[]float64{1, 2, 3},
		[]float64{3, 2, 1},
		[]float64{2, 1, 3},
	}

	m2 := [][]float64{
		[]float64{5, 6},
		[]float64{5, 4},
		[]float64{6, 5},
	}

	expectedRes := [][]float64{
		[]float64{1, 2, 3, 5, 6},
		[]float64{3, 2, 1, 5, 4},
		[]float64{2, 1, 3, 6, 5},
	}

	result := Concat(m1, m2)

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result); j++ {
			if result[i][j]-expectedRes[i][j] > 0.000001 {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}
