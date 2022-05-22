//
// Created by jonas on 13.05.22.
//

#include "task.h"
#include "../support/math.c"

#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <barrier.h>


#include "../../VN/support/common.h"

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;
BARRIER_INIT(my_barrier, NR_TASKLETS);
/*
void vector_norm_host(double *v[], unsigned int norm, unsigned int numbers, double *result){
    *result = 0;
    for(int i= 0; i < numbers; i++){
        *result += x_to_the_power_of_n(v[i], norm);
    }
    *result = x_to_the_power_of_z(result, 1/((double )norm));
}*/
extern int main_kernel1();

int (*kernels[nr_kernels])(void) = {main_kernel1};

int main(void){
    return kernels[DPU_INPUT_ARGUMENTS.kernel]();

}
//TODO: Adjust the copied code
int main_kernel1(){
    unsigned int tasklet_id = me();
#if PRINT
    printf("tasklet_id = %u\n", tasklet_id);
#endif
    if (tasklet_id == 0){ // Initialize once the cycle counter
        mem_reset(); // Reset the heap
    }
    // Barrier
    barrier_wait(&my_barrier);

    uint32_t input_size_dpu_bytes = DPU_INPUT_ARGUMENTS.size; // Input size per DPU in bytes
    uint32_t input_size_dpu_bytes_transfer = DPU_INPUT_ARGUMENTS.transfer_size; // Transfer input size per DPU in bytes

    // Address of the current processing block in MRAM
    uint32_t base_tasklet = tasklet_id << BLOCK_SIZE_LOG2;
    uint32_t mram_base_addr_A = (uint32_t)DPU_MRAM_HEAP_POINTER;
    uint32_t mram_base_addr_B = (uint32_t)(DPU_MRAM_HEAP_POINTER + input_size_dpu_bytes_transfer);

    // Initialize a local cache to store the MRAM block
    double *cache_A = (double *) mem_alloc(BLOCK_SIZE);
    double *result = (double *) mem_alloc(BLOCK_SIZE);

    for(unsigned int byte_index = base_tasklet; byte_index < input_size_dpu_bytes; byte_index += BLOCK_SIZE * NR_TASKLETS){

        // Bound checking
        uint32_t l_size_bytes = (byte_index + BLOCK_SIZE >= input_size_dpu_bytes) ? (input_size_dpu_bytes - byte_index) : BLOCK_SIZE;

        // Load cache with current MRAM block
        mram_read((__mram_ptr void const*)(mram_base_addr_A + byte_index), cache_A, l_size_bytes);
        //mram_read((__mram_ptr void const*)(mram_base_addr_B + byte_index), cache_B, l_size_bytes);

        // Computer vector addition
        vector_norm_host(cache_A, 2, 2, result);

        // Write cache to current MRAM block
        mram_write(result, (__mram_ptr void*)(mram_base_addr_B + byte_index), l_size_bytes);

    }
    return 0;
}