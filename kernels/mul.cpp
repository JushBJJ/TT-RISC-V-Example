void kernel_main() {
    uint32_t src0_address =  get_arg_val<uint32_t>(0);
    uint32_t src1_addres =  get_arg_val<uint32_t>(1);
    uint32_t dst_address =  get_arg_val<uint32_t>(2);
    uint32_t src0_noc_x =  get_arg_val<uint32_t>(3);
    uint32_t src0_noc_y =  get_arg_val<uint32_t>(4);
    uint32_t src1_noc_x =  get_arg_val<uint32_t>(5);
    uint32_t src1_noc_y =  get_arg_val<uint32_t>(6);
    uint32_t dst_noc_x =  get_arg_val<uint32_t>(7);
    uint32_t dst_noc_y =  get_arg_val<uint32_t>(8);

    uint64_t src0_noc_addr = get_noc_addr(src0_noc_x, src0_noc_y, src0_address);
    uint64_t src1_noc_addr = get_noc_addr(src1_noc_x, src1_noc_y, src1_addres);
    uint64_t dst_noc_addr = get_noc_addr(dst_noc_x, dst_noc_y, dst_address);

    uint32_t c_in0_size = get_tile_size(tt::CB::c_in0);
    uint32_t c_in1_size = get_tile_size(tt::CB::c_in1);
    uint32_t c_out0_size = get_tile_size(tt::CB::c_out0);
    uint32_t c_in0_write_address = get_write_ptr(tt::CB::c_in0);
    uint32_t c_in1_write_address = get_write_ptr(tt::CB::c_in1);
    uint32_t c_out0_read_address = get_read_ptr(tt::CB::c_out0);

    noc_async_read(src0_noc_addr, c_in0_write_address, c_in0_size);
    noc_async_read(src1_noc_addr, c_in1_write_address, c_in1_size);
    noc_async_read_barrier();

    // RISC-V Compute
    ///////////////////////////////

    uint32_t* src0 = (uint32_t*)c_in0_write_address;
    uint32_t* src1 = (uint32_t*)c_in1_write_address;

    src0[0] = src0[0] * src1[0];

    ///////////////////////////////

    noc_async_write(c_in0_write_address, dst_noc_addr, c_out0_size);
    noc_async_write_barrier();
}
