#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include <memory>
#include <vector>
#include <unordered_map>

using namespace tt;
using namespace tt::tt_metal;

class BufferManager {
public:
    BufferManager(Device* device, uint32_t single_tile_size) : single_tile_size_(single_tile_size) {
        dram_config = tt_metal::InterleavedBufferConfig{
            .device = device,
            .size = single_tile_size,
            .page_size = single_tile_size,
            .buffer_type = tt_metal::BufferType::DRAM
        };
    }

    struct BufferInfo {
        std::shared_ptr<tt::tt_metal::Buffer> buffer;
        std::vector<uint32_t> cpu_buffer;
        uint32_t noc_x;
        uint32_t noc_y;
    };

    BufferInfo create_buffer(const std::string& name, uint32_t actual_size) {
        /* Create DRAM Buffer */
	    std::shared_ptr<tt::tt_metal::Buffer> buffer = tt_metal::CreateBuffer(dram_config);
        auto noc_coord = buffer->noc_coordinates();
        std::vector<uint32_t> cpu_buffer(actual_size, 0);

        BufferInfo info = {
            buffer,
            cpu_buffer,
            noc_coord.x,
            noc_coord.y
        };

        dram_buffers_[name] = info;
        return info;
    }

    void create_cb(tt_metal::Program& program, const CoreCoord& core, 
                              const std::string& name, uint32_t cb_index) {
        CircularBufferConfig cb_config = CircularBufferConfig(
            single_tile_size_,
            {{cb_index, tt::DataFormat::Float16_b}}
        ).set_page_size(cb_index, single_tile_size_);

        tt_metal::CreateCircularBuffer(program, core, cb_config);
        cb_configs_.emplace(name, std::move(cb_config));
    }

    BufferInfo get_buffer(const std::string& name) {
        return dram_buffers_[name];
    }

    CircularBufferConfig get_cb_config(const std::string& name) {
        return cb_configs_.at(name);
    }

private:
    uint32_t single_tile_size_;
    std::unordered_map<std::string, BufferInfo> dram_buffers_;
    std::unordered_map<std::string, CircularBufferConfig> cb_configs_;
    tt_metal::InterleavedBufferConfig dram_config;
};
