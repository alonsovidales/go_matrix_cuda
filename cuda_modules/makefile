ll:
	rm *.ptx
	nvcc -arch=sm_30 -ptx -Xptxas -allow-expensive-optimizations -fmad=false -ftz=false -prec-div=true -prec-sqrt=true *.cu

	cp empty_files/* ../

	cat matrix_add_bias.ptx >> ../cuda_kernel_add_bias.go
	cat matrix_add_bias_top.ptx >> ../cuda_kernel_matrix_add_bias_top.go
	cat matrix_add.ptx >> ../cuda_kernel_matrix_add.go
	cat matrix_log.ptx >> ../cuda_kernel_matrix_log.go
	cat matrix_mult_all.ptx >> ../cuda_kernel_matrix_mult_all.go
	cat matrix_mult_by.ptx >> ../cuda_kernel_matrix_mult_by.go
	cat matrix_mult.ptx >> ../cuda_kernel_matrix_mult.go
	cat matrix_mult_trans.ptx >> ../cuda_kernel_matrix_mult_trans.go
	cat matrix_neg.ptx >> ../cuda_kernel_matrix_neg.go
	cat matrix_one_minus.ptx >> ../cuda_kernel_matrix_one_minus.go
	cat matrix_pow_two.ptx >> ../cuda_kernel_matrix_pow_two.go
	cat matrix_remove_bias.ptx >> ../cuda_kernel_remove_bias.go
	cat matrix_remove_bias_top.ptx >> ../cuda_kernel_remove_bias_top.go
	cat matrix_set_bias_to_zero.ptx >> ../cuda_kernel_matrix_set_bias_to_zero.go
	cat matrix_sigmoid_gradient.ptx >> ../cuda_kernel_matrix_sigmoid_gradient.go
	cat matrix_sigmoid.ptx >> ../cuda_kernel_matrix_sigmoid.go
	cat matrix_sub.ptx >> ../cuda_kernel_matrix_sub.go
	cat matrix_sum_all.ptx >> ../cuda_kernel_sum_all.go
	cat matrix_trans.ptx >> ../cuda_kernel_matrix_trans.go

	echo "\`" >> ../cuda_kernel_add_bias.go
	echo "\`" >> ../cuda_kernel_matrix_add_bias_top.go
	echo "\`" >> ../cuda_kernel_matrix_add.go
	echo "\`" >> ../cuda_kernel_matrix_log.go
	echo "\`" >> ../cuda_kernel_matrix_mult_all.go
	echo "\`" >> ../cuda_kernel_matrix_mult_by.go
	echo "\`" >> ../cuda_kernel_matrix_mult.go
	echo "\`" >> ../cuda_kernel_matrix_mult_trans.go
	echo "\`" >> ../cuda_kernel_matrix_neg.go
	echo "\`" >> ../cuda_kernel_matrix_one_minus.go
	echo "\`" >> ../cuda_kernel_matrix_pow_two.go
	echo "\`" >> ../cuda_kernel_remove_bias.go
	echo "\`" >> ../cuda_kernel_remove_bias_top.go
	echo "\`" >> ../cuda_kernel_matrix_set_bias_to_zero.go
	echo "\`" >> ../cuda_kernel_matrix_sigmoid_gradient.go
	echo "\`" >> ../cuda_kernel_matrix_sigmoid.go
	echo "\`" >> ../cuda_kernel_matrix_sub.go
	echo "\`" >> ../cuda_kernel_sum_all.go
	echo "\`" >> ../cuda_kernel_matrix_trans.go
