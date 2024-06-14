/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *//*
 */

/** @file   common.cu
 *  @author Thomas Müller and Nikolaus Binder, NVIDIA
 *  @brief  Common utilities that are needed by pretty much every component of this framework.
 */

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/multi_stream.h>

#include <cuda.h>

#include <algorithm>
#include <cctype>
#include <unordered_map>

TCNN_NAMESPACE_BEGIN

static_assert(
	__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2),
	"tiny-cuda-nn requires at least CUDA 10.2"
);

int cuda_device() {
	int device;
	CUDA_CHECK_THROW(cudaGetDevice(&device));
	return device;
}

void set_cuda_device(int device) {
	CUDA_CHECK_THROW(cudaSetDevice(device));
}

int cuda_device_count() {
	int device_count;
	CUDA_CHECK_THROW(cudaGetDeviceCount(&device_count));
	return device_count;
}

bool cuda_supports_virtual_memory(int device) {
	int supports_vmm;
	CU_CHECK_THROW(cuDeviceGetAttribute(&supports_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, device));
	return supports_vmm != 0;
}

uint32_t cuda_compute_capability(int device) {
	cudaDeviceProp props;
	CUDA_CHECK_THROW(cudaGetDeviceProperties(&props, device));
	return props.major * 10 + props.minor;
}

size_t cuda_memory_granularity(int device) {
	size_t granularity;
	CUmemAllocationProp prop = {};
	prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	prop.location.id = 0;
	CUresult granularity_result = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
	if (granularity_result == CUDA_ERROR_NOT_SUPPORTED) {
		return 1;
	}
	CU_CHECK_THROW(granularity_result);
	return granularity;
}

MemoryInfo cuda_memory_info() {
	MemoryInfo info;
	CUDA_CHECK_THROW(cudaMemGetInfo(&info.free, &info.total));
	info.used = info.total - info.free;
	return info;
}

std::string to_lower(std::string str) {
	std::transform(std::begin(str), std::end(str), std::begin(str), [](unsigned char c) { return (char)std::tolower(c); });
	return str;
}

std::string to_upper(std::string str) {
	std::transform(std::begin(str), std::end(str), std::begin(str), [](unsigned char c) { return (char)std::toupper(c); });
	return str;
}

template <>
std::string type_to_string<float>() {
	return "float";
}

template <>
std::string type_to_string<__half>() {
	return "__half";
}

struct StreamOwnedObjects {
	std::unordered_map<cudaStream_t, std::shared_ptr<GPUMemoryArena>> stream_gpu_memory_arenas;
	std::unordered_map<int, std::shared_ptr<GPUMemoryArena>> global_gpu_memory_arenas;

	struct MultiStreamWrapper {
		~MultiStreamWrapper() {
			// Avoids free_multi_stream being called in the middle of multi_streams's destruction.
			// Note, that we don't use `.clear()` intentionally, because it would also have issues
			// with recursive calling of free_multi_stream by ~StreamAndEvent.
			while (!multi_streams.empty()) {
				free_multi_streams(multi_streams.begin()->first);
			}
		}

		std::unordered_map<cudaStream_t, std::stack<std::shared_ptr<MultiStream>>> multi_streams;
	} stream_multi_streams;
	std::unordered_map<int, std::stack<std::shared_ptr<MultiStream>>> global_multi_streams;
};

StreamOwnedObjects& stream_owned_objects() {
	static StreamOwnedObjects s_stream_owned_objects;
	return s_stream_owned_objects;
}

std::unordered_map<cudaStream_t, std::shared_ptr<GPUMemoryArena>>& stream_gpu_memory_arenas() {
	return stream_owned_objects().stream_gpu_memory_arenas;
}

std::unordered_map<int, std::shared_ptr<GPUMemoryArena>>& global_gpu_memory_arenas() {
	return stream_owned_objects().global_gpu_memory_arenas;
}

std::unordered_map<cudaStream_t, std::stack<std::shared_ptr<MultiStream>>>& stream_multi_streams() {
	return stream_owned_objects().stream_multi_streams.multi_streams;
}

std::unordered_map<int, std::stack<std::shared_ptr<MultiStream>>>& global_multi_streams() {
	return stream_owned_objects().global_multi_streams;
}

TCNN_NAMESPACE_END
