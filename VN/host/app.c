//
// Created by jonas on 13.05.22.
//
//shamelessly inspired by VA
#include "app.h"
#include <math.h>
#include <stdio.h>
#include "../support/math.c"
#include "dpu.h"


#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <bits/stdint-uintn.h>
#include <time.h>

#include "../../VN/support/common.h"
#include "../../VN/support/timer.h"
#include "../../VN/support/params.h"




// Define the DPU Binary path as DPU_BINARY here
#ifndef DPU_BINARY
#define DPU_BINARY "./bin/dpu_code"
#endif

#if ENERGY
#include <dpu_probe.h>
#endif


void vector_norm_host(double *v[], unsigned int norm, unsigned int numbers, double *result){
    *result = 0;
    for(int i= 0; i < numbers; i++){
        *result += x_to_the_power_of_n(v[i], norm);
    }
    *result = x_to_the_power_of_z(result, 1/((double )norm));
}


void fill_up_vector(double *v[], unsigned int length){
    for(int i= 0;i<length; i++){
        *v[i] = rand();
    }
}
int main(int argc, char **argv) {
    printf("hi");
    double t = 2;
    double n = 0.5;
    double result = x_to_the_power_of_z(&t, n);
    double t2 = x_to_the_power_of_n(&t, 10);
    struct Params p = input_params(argc, argv);

    struct dpu_set_t dpu_set, dpu;
    uint32_t nr_of_dpus;

#if ENERGY
    struct dpu_probe_t probe;
    DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
#endif
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));
    printf("Allocated %d DPU(s)\n", nr_of_dpus);
    unsigned int i = 0;
    const unsigned int vector_length = 1000;
    const unsigned int input_size =
            p.exp == 0 ? p.input_size * nr_of_dpus : p.input_size; // Total input size (weak or strong scaling)
    const unsigned int input_size_bytes =
            ((input_size) * sizeof(double) * vector_length) % != 0 ? roundup(input_size, 8) : input_size;
    const unsigned int input_size_dpu = divceil(input_size, nr_of_dpus);
    const unsigned int input_size_dpu_8bytes =
        ((input_size_dpu * sizeof(double)*vector_length) % 8) != 0 ? roundup(input_size_dpu, 8) : input_size_dpu; // Input size per DPU (max.), 8-byte aligned
    double *v[vector_length];
    double *output_host;
    double *output_dpu;


    //Alocate the vector
    srand(time(NULL));

    Timer timer;
    // Loop over main kernel

    for(int rep = 0; rep <p.n_warmup +p.n_reps; rep++){
        fill_up_vector(v,vector_length);
        if(rep >= p.n_warmup)
            start(&timer, 0, rep - p.n_warmup);
        vector_norm_host(v, 2,vector_length,output_host );
        if(rep >= p.n_warmup)
            stop(&timer, 0);

        unsigned int kernel = 0;
        dpu_arguments_t  input_arguments[NR_DPUS];
        for(i=0; i<nr_of_dpus -1; i++){
            input_arguments[i].size=input_size_dpu_8bytes* sizeof(double)*vector_length;
            input_arguments[i].transfer_size=input_size_dpu_8bytes* sizeof(double )*vector_length;
            input_arguments[i].kernel=kernel;
        }
        //Todo: Make copied code fitting this scenario here.
        DPU_FOREACH(dpu_set, dpu, i){
            DPU_ASSERT(dpu_prepare_xfer(dpu, &input_arguments[i]));
        }
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &input_arguments[i]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(input_arguments[0]), DPU_XFER_DEFAULT));

        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, bufferA + input_size_dpu_8bytes * i));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, input_size_dpu_8bytes * sizeof(T), DPU_XFER_DEFAULT));

        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, bufferB + input_size_dpu_8bytes * i));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, input_size_dpu_8bytes * sizeof(T), input_size_dpu_8bytes * sizeof(T), DPU_XFER_DEFAULT));
        if(rep >= p.n_warmup)
            stop(&timer, 1);

                printf("Run program on DPU(s) \n");
        // Run DPU kernel
        if(rep >= p.n_warmup) {
            start(&timer, 2, rep - p.n_warmup);
            #if ENERGY
            DPU_ASSERT(dpu_probe_start(&probe));
            #endif
        }
        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
        if(rep >= p.n_warmup) {
            stop(&timer, 2);
            #if ENERGY
            DPU_ASSERT(dpu_probe_stop(&probe));
            #endif
        }

#if PRINT
        {
            unsigned int each_dpu = 0;
            printf("Display DPU Logs\n");
            DPU_FOREACH (dpu_set, dpu) {
                printf("DPU#%d:\n", each_dpu);
                DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
                each_dpu++;
            }
        }
#endif

        printf("Retrieve results\n");
        if(rep >= p.n_warmup)
            start(&timer, 3, rep - p.n_warmup);
        i = 0;
        // PARALLEL RETRIEVE TRANSFER
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, bufferC + input_size_dpu_8bytes * i));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, input_size_dpu_8bytes * sizeof(T), input_size_dpu_8bytes * sizeof(T), DPU_XFER_DEFAULT));
        if(rep >= p.n_warmup)
            stop(&timer, 3);
    }

}