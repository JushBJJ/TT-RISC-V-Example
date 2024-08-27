#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_custom_buffer_manager.h"
#include <iostream>

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char **argv) {
    /* Silicon accelerator setup */
    Device *device = CreateDevice(0);

    /* Setup program to execute along with its buffers and kernels to use */
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t single_tile = 32;
    BufferManager buffer_manager = BufferManager(device, single_tile);

    buffer_manager.create_buffer("src0", 1);
    buffer_manager.create_buffer("src1", 1);
    buffer_manager.create_buffer("dst", 1);

    buffer_manager.create_cb(program, core, "src0", CB::c_in0);
    buffer_manager.create_cb(program, core, "src1", CB::c_in1);
    buffer_manager.create_cb(program, core, "dst", CB::c_out0);

    KernelHandle risc_kernel = CreateKernel(
        program,
        "../kernels/mul.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, 
            .noc = NOC::RISCV_0_default
        }
    );

    SetRuntimeArgs(program, risc_kernel, core, 
    {
        buffer_manager.get_buffer("src0").buffer->address(),
        buffer_manager.get_buffer("src1").buffer->address(),
        buffer_manager.get_buffer("dst").buffer->address(),
        buffer_manager.get_buffer("src0").noc_x,
        buffer_manager.get_buffer("src0").noc_y,
        buffer_manager.get_buffer("src1").noc_x,
        buffer_manager.get_buffer("src1").noc_y,
        buffer_manager.get_buffer("dst").noc_x,
        buffer_manager.get_buffer("dst").noc_y
    });

    auto src0_cpu_buffer = buffer_manager.get_buffer("src0").cpu_buffer;
    auto src1_cpu_buffer = buffer_manager.get_buffer("src1").cpu_buffer;
    auto dst_cpu_buffer = buffer_manager.get_buffer("dst").cpu_buffer;
    auto src0_dram_buffer = buffer_manager.get_buffer("src0").buffer;
    auto src1_dram_buffer = buffer_manager.get_buffer("src1").buffer;
    auto dst_dram_buffer = buffer_manager.get_buffer("dst").buffer;

    src0_cpu_buffer[0] = 4;
    src1_cpu_buffer[0] = 2;

    EnqueueWriteBuffer(cq, src0_dram_buffer, src0_cpu_buffer, false);
    EnqueueWriteBuffer(cq, src1_dram_buffer, src1_cpu_buffer, false);
    EnqueueProgram(cq, program, false);
    Finish(cq);
    EnqueueReadBuffer(cq, dst_dram_buffer, dst_cpu_buffer, true);
   
    int expected_result = src0_cpu_buffer[0] * src1_cpu_buffer[0];

    std::cout<<"Result: " << dst_cpu_buffer[0] << std::endl;
    std::cout<<"Expected: " << expected_result << std::endl;

    if(dst_cpu_buffer[0] == expected_result){
        std::cout<<"Test Passed";
    }
    else{
        std::cout<<"Test Failed";
    }
    CloseDevice(device);
}
