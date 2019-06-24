#include "vkrpch.h"

#define PANIC_BAD_RESULT(result)                              \
    if (result != VK_SUCCESS) {                               \
        spdlog::error("PANIC AT: {}:{}", __LINE__, __FILE__); \
        std::exit(EXIT_FAILURE);                              \
    }

namespace vkc {
PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT;
PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT;

static VKAPI_ATTR VkBool32 validation_layer_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
    spdlog::debug("validation layer: {}", pCallbackData->pMessage);
    return VK_FALSE;
}

VkDebugUtilsMessengerEXT create_debug_messenger(const VkInstance instance) {
    VkDebugUtilsMessengerCreateInfoEXT messenger_create_info = {};
    messenger_create_info.sType =
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    messenger_create_info.pfnUserCallback = validation_layer_callback;
    messenger_create_info.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    messenger_create_info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

    vkCreateDebugUtilsMessengerEXT =
        (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance, "vkCreateDebugUtilsMessengerEXT");

    vkDestroyDebugUtilsMessengerEXT =
        (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance, "vkDestroyDebugUtilsMessengerEXT");

    VkDebugUtilsMessengerEXT debug_messenger;
    PANIC_BAD_RESULT(vkCreateDebugUtilsMessengerEXT(
        instance, &messenger_create_info, nullptr, &debug_messenger));

    return debug_messenger;
}

class MoveOnly {
   public:
    MoveOnly() = default;
    MoveOnly(MoveOnly&& other) = default;
    MoveOnly& operator=(MoveOnly&& other) = default;

    MoveOnly(MoveOnly& other) = delete;
    MoveOnly& operator=(MoveOnly) = delete;
};

struct Buffer {
    VkBuffer handle_;

    VmaAllocator allocator_;
    VmaAllocation allocation_;
    VmaAllocationInfo allocation_info_;
    VmaMemoryUsage mem_usage_;

    uint32_t size_;
    uint32_t family_index_;
    VkBufferCreateFlags buffer_usage_;

    char* map_ptr_;
    bool is_mapped_;

    Buffer() = default;
    Buffer(const VmaAllocator& allocator, uint32_t size,
           VkBufferCreateFlags buffer_usage, VmaMemoryUsage mem_usage,
           uint32_t family_index);
    ~Buffer();

    char* map();
    void unmap();
};

Buffer::Buffer(const VmaAllocator& allocator, uint32_t size,
               VkBufferCreateFlags buffer_usage, VmaMemoryUsage mem_usage,
               uint32_t family_index)
    : allocator_(allocator),
      mem_usage_(mem_usage),
      size_(size),
      family_index_(family_index),
      buffer_usage_(buffer_usage),
      is_mapped_(false),
      map_ptr_(nullptr) {
    VkBufferCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    create_info.size = size;
    create_info.usage = buffer_usage;
    create_info.queueFamilyIndexCount = 1;
    create_info.pQueueFamilyIndices = &family_index;
    create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocation_create_info = {};
    allocation_create_info.usage = mem_usage;

    PANIC_BAD_RESULT(vmaCreateBuffer(allocator_, &create_info,
                                     &allocation_create_info, &handle_,
                                     &allocation_, &allocation_info_));
}
Buffer::~Buffer() {}

char* Buffer::map() {
    if (is_mapped_) return map_ptr_;

    if (mem_usage_ == VMA_MEMORY_USAGE_GPU_ONLY) {
        spdlog::error("GPU only memory can't be mapped");
        std::exit(EXIT_FAILURE);
    }

    PANIC_BAD_RESULT(vmaMapMemory(allocator_, allocation_,
                                  reinterpret_cast<void**>(&map_ptr_)));

    is_mapped_ = true;
    return map_ptr_;
}

void Buffer::unmap() {
    if (!is_mapped_) {
        spdlog::error("Buffer isn't mapped");
        std::exit(EXIT_FAILURE);
    }

    vmaUnmapMemory(allocator_, allocation_);
    is_mapped_ = false;
    map_ptr_ = nullptr;
}

struct DescriptorSet {
    VkDescriptorSet handle_;
    VkDescriptorSetLayout set_layout_;
    std::vector<VkDescriptorSetLayoutBinding> bindings_;

    VkDevice device_;

    DescriptorSet() = default;
    DescriptorSet(const VkDevice& device, VkDescriptorSet descriptor_set,
                  VkDescriptorSetLayout descriptor_set_layout,
                  std::vector<VkDescriptorSetLayoutBinding> bindings);

    void update(uint32_t binding, uint32_t start_element,
                uint32_t descriptor_count, uint32_t offset, uint64_t range,
                const VkBuffer& buffer);
};

DescriptorSet::DescriptorSet(const VkDevice& device, VkDescriptorSet handle,
                             VkDescriptorSetLayout set_layout,
                             std::vector<VkDescriptorSetLayoutBinding> bindings)
    : device_(device),
      handle_(handle),
      set_layout_(set_layout),
      bindings_(bindings) {}

void DescriptorSet::update(uint32_t binding, uint32_t start_element,
                           uint32_t descriptor_count, uint32_t offset,
                           uint64_t range, const VkBuffer& buffer) {
    VkDescriptorBufferInfo buffer_info = {};
    buffer_info.buffer = buffer;
    buffer_info.offset = offset;
    buffer_info.range = range;

    VkWriteDescriptorSet write_set = {};
    write_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_set.descriptorCount = descriptor_count;
    write_set.descriptorType =
        bindings_.at(binding)
            .descriptorType;  // Only works if the bindings are sorted
    write_set.dstBinding = binding;
    write_set.dstArrayElement = start_element;
    write_set.dstBinding = binding;
    write_set.dstSet = handle_;
    write_set.pBufferInfo = &buffer_info;

    write_set.pImageInfo = nullptr;
    write_set.pTexelBufferView = nullptr;

    vkUpdateDescriptorSets(device_, 1, &write_set, 0, nullptr);
}

struct DescriptorPool {
    VkDescriptorPool handle_;
    std::vector<VkDescriptorPoolSize> pool_sizes_;
    uint32_t max_sets_;

    VkDevice device_;

    DescriptorPool() = default;
    DescriptorPool(const VkDevice& device, uint32_t max_sets,
                   const std::vector<VkDescriptorPoolSize>& pool_sizes);

    void create();
    DescriptorSet allocate_descriptor_set(
        const std::vector<VkDescriptorSetLayoutBinding>& bindings);
};

DescriptorPool::DescriptorPool(
    const VkDevice& device, uint32_t max_sets,
    const std::vector<VkDescriptorPoolSize>& pool_sizes)
    : device_(device), max_sets_(max_sets), pool_sizes_(pool_sizes) {
    create();
}

void DescriptorPool::create() {
    VkDescriptorPoolCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    create_info.maxSets = max_sets_;
    create_info.poolSizeCount = pool_sizes_.size();
    create_info.pPoolSizes = pool_sizes_.data();

    PANIC_BAD_RESULT(
        vkCreateDescriptorPool(device_, &create_info, nullptr, &handle_));
}

DescriptorSet DescriptorPool::allocate_descriptor_set(
    const std::vector<VkDescriptorSetLayoutBinding>& bindings) {
    // Keep the bindings sorted
    auto bindings_sorted = bindings;
    std::sort(bindings_sorted.begin(), bindings_sorted.end(),
              [](const VkDescriptorSetLayoutBinding& b1,
                 const VkDescriptorSetLayoutBinding& b2) -> bool {
                  return (b1.binding < b2.binding);
              });

    VkDescriptorSetLayoutCreateInfo set_layout_info = {};
    set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_layout_info.bindingCount = bindings.size();
    set_layout_info.pBindings = bindings.data();

    VkDescriptorSetLayout set_layout;
    PANIC_BAD_RESULT(vkCreateDescriptorSetLayout(device_, &set_layout_info,
                                                 nullptr, &set_layout));

    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = handle_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &set_layout;

    VkDescriptorSet set;
    PANIC_BAD_RESULT(vkAllocateDescriptorSets(device_, &alloc_info, &set));
    return DescriptorSet(device_, set, set_layout, bindings);
}

struct Pipeline {
    VkShaderModule shader_;
    VkPipeline pipeline_;
    VkPipelineLayout pipeline_layout_;

    VkDevice device_;

    Pipeline() = default;
    Pipeline(const VkDevice& device);

    void create(const std::vector<uint32_t>& program_src,
                const VkDescriptorSetLayout& descriptor_set_layout);
};

Pipeline::Pipeline(const VkDevice& device) : device_(device) {}

void Pipeline::create(const std::vector<uint32_t>& program_src,
                      const VkDescriptorSetLayout& descriptor_set_layout) {
    VkShaderModuleCreateInfo shader_info_ = {};
    shader_info_.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info_.pCode = program_src.data();
    shader_info_.codeSize = program_src.size();

    PANIC_BAD_RESULT(
        vkCreateShaderModule(device_, &shader_info_, nullptr, &shader_));

    VkPipelineLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &descriptor_set_layout;

    PANIC_BAD_RESULT(vkCreatePipelineLayout(device_, &layout_info, nullptr,
                                            &pipeline_layout_));

    VkPipelineShaderStageCreateInfo stage_info = {};
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.pName = "main";
    stage_info.module = shader_;

    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.layout = pipeline_layout_;
    pipeline_info.stage = stage_info;

    PANIC_BAD_RESULT(vkCreateComputePipelines(
        device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_));
}

struct PhysicalDevice {
    VkPhysicalDevice handle_;
    VkPhysicalDeviceProperties properties_;
    VkPhysicalDeviceMemoryProperties memory_properties_;
    std::vector<VkQueueFamilyProperties> family_properties_;
};

struct Instance {
    VkInstance handle_;
    std::vector<const char*> extensions_;

    bool use_validation_;
    VkDebugUtilsMessengerEXT debug_messenger_;

    std::vector<PhysicalDevice> physical_devices_;

    void init(bool use_validation);
    void query_physical_devices();
};

void Instance::init(bool use_validation) {
    use_validation_ = use_validation;

    spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("[%T] [%^%l%$] %v");

    /*
        Using volk to minimize API calls overhead
        More info at: https://gpuopen.com/reducing-vulkan-api-call-overhead/
    */
    spdlog::info("Initializing volk");
    if (volkInitialize() != VK_SUCCESS) {
        spdlog::error("Failed to initialize volk, abort");
        std::exit(EXIT_FAILURE);
    }

    std::vector<const char*> validation_layers;

    if (use_validation_) {
        extensions_.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        validation_layers.push_back("VK_LAYER_LUNARG_standard_validation");
        // validation_layers.push_back("VK_LAYER_LUNARG_api_dump");
    }

    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.apiVersion = VK_API_VERSION_1_1;
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "Null Engine";
    app_info.pApplicationName = "Compute App";

    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.enabledExtensionCount = extensions_.size();
    create_info.ppEnabledExtensionNames = extensions_.data();
    create_info.enabledLayerCount = validation_layers.size();
    create_info.ppEnabledLayerNames = validation_layers.data();
    create_info.pApplicationInfo = &app_info;

    PANIC_BAD_RESULT(vkCreateInstance(&create_info, nullptr, &handle_));

    volkLoadInstance(handle_);

    if (use_validation_) debug_messenger_ = create_debug_messenger(handle_);
}

void Instance::query_physical_devices() {
    std::vector<VkPhysicalDevice> physical_devices;
    uint32_t physical_device_count = 0;
    vkEnumeratePhysicalDevices(handle_, &physical_device_count, nullptr);
    physical_devices.resize(physical_device_count);
    vkEnumeratePhysicalDevices(handle_, &physical_device_count,
                               physical_devices.data());

    physical_devices_.reserve(physical_devices.size());

    for (auto& phys_dev : physical_devices) {
        std::vector<VkQueueFamilyProperties> family_properties;
        uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &family_count,
                                                 nullptr);
        family_properties.resize(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &family_count,
                                                 family_properties.data());

        VkPhysicalDeviceProperties physical_device_properties;
        vkGetPhysicalDeviceProperties(phys_dev, &physical_device_properties);
        VkPhysicalDeviceMemoryProperties physical_device_memory_properties;
        vkGetPhysicalDeviceMemoryProperties(phys_dev,
                                            &physical_device_memory_properties);

        physical_devices_.push_back({phys_dev, physical_device_properties,
                                     physical_device_memory_properties,
                                     family_properties});
    }
}

VmaAllocator create_vma_allocator(VkDevice device,
                                  VkPhysicalDevice physical_device) {
    /*
            Vulkan Memory Allocator
            More info at:
           https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
         */
    spdlog::info("Initializing VMA Allocator");

    VmaVulkanFunctions vma_functions = {vkGetPhysicalDeviceProperties,
                                        vkGetPhysicalDeviceMemoryProperties,
                                        vkAllocateMemory,
                                        vkFreeMemory,
                                        vkMapMemory,
                                        vkUnmapMemory,
                                        vkFlushMappedMemoryRanges,
                                        vkInvalidateMappedMemoryRanges,
                                        vkBindBufferMemory,
                                        vkBindImageMemory,
                                        vkGetBufferMemoryRequirements,
                                        vkGetImageMemoryRequirements,
                                        vkCreateBuffer,
                                        vkDestroyBuffer,
                                        vkCreateImage,
                                        vkDestroyImage,
                                        vkCmdCopyBuffer};

    VmaAllocatorCreateInfo allocator_create_info = {};
    allocator_create_info.physicalDevice = physical_device;
    allocator_create_info.device = device;
    allocator_create_info.pVulkanFunctions = &vma_functions;

    VmaAllocator allocator;
    PANIC_BAD_RESULT(vmaCreateAllocator(&allocator_create_info, &allocator));
    return allocator;
}

struct CommandPool {
    VkCommandPool handle_;
    uint32_t family_index_;
    std::vector<VkCommandBuffer> command_buffers_;

    VkDevice device_;

    CommandPool() = default;
    CommandPool(VkDevice device, uint32_t family_index);

    VkCommandBuffer allocate_command_buffer(VkCommandBufferLevel level);
};

CommandPool::CommandPool(VkDevice device, uint32_t family_index)
    : family_index_(family_index), device_(device) {
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = family_index_;

    spdlog::info("Creating command pool");
    PANIC_BAD_RESULT(
        vkCreateCommandPool(device_, &pool_info, nullptr, &handle_));
}

VkCommandBuffer CommandPool::allocate_command_buffer(
    VkCommandBufferLevel level) {
    VkCommandBufferAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandBufferCount = 1;
    alloc_info.level = level;
    alloc_info.commandPool = handle_;

    VkCommandBuffer command_buffer;
    PANIC_BAD_RESULT(
        vkAllocateCommandBuffers(device_, &alloc_info, &command_buffer));

    command_buffers_.push_back(command_buffer);
    return command_buffer;
}

class Compute {
   public:
    Instance instance_;
    PhysicalDevice physical_device_;

    VkDevice device_;
    std::vector<const char*> device_extensions_;

    VkQueue compute_queue_;
    uint32_t compute_queue_index_;

    VmaAllocator allocator_;
    std::vector<Buffer> buffers_;
    uint32_t vec_size_;

    DescriptorPool descriptor_pool_;
    DescriptorSet descriptor_set_;

    Pipeline pipeline_;

    CommandPool command_pool_;
    VkCommandBuffer command_buffer_;

    void init(bool use_validation);
    void get_physical_device();
    void init_device();
    void create_buffers();
    void create_descriptors();
    void create_pipeline();
    void create_command_buffer();

    void dispatch();

    void fill_buffer();
    void dump_buffer();
};

void Compute::init(bool use_validation) {
    instance_.init(use_validation);
    instance_.query_physical_devices();

    get_physical_device();
}

void Compute::get_physical_device() {
    bool found_device = false;
    for (auto& phys_dev : instance_.physical_devices_) {
        for (uint32_t i = 0; i < phys_dev.family_properties_.size(); ++i) {
            auto& family_property = phys_dev.family_properties_.at(i);
            if (family_property.queueFlags & VK_QUEUE_COMPUTE_BIT) {
                found_device = true;
                physical_device_ = phys_dev;
                compute_queue_index_ = i;
            }
        }
    }

    if (!found_device) {
        spdlog::error("Failed to find a compute device, abort");
        std::exit(EXIT_FAILURE);
    }
}

void Compute::init_device() {
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info = {};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.pQueuePriorities = &queue_priority;
    queue_create_info.queueCount = 1;
    queue_create_info.queueFamilyIndex = compute_queue_index_;

    VkDeviceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.enabledExtensionCount = device_extensions_.size();
    create_info.ppEnabledExtensionNames = device_extensions_.data();
    create_info.queueCreateInfoCount = 1;
    create_info.pQueueCreateInfos = &queue_create_info;

    PANIC_BAD_RESULT(vkCreateDevice(physical_device_.handle_, &create_info,
                                    nullptr, &device_));

    volkLoadDevice(device_);
    allocator_ = create_vma_allocator(device_, physical_device_.handle_);

    vkGetDeviceQueue(device_, compute_queue_index_, 0, &compute_queue_);
}

void Compute::create_buffers() {
    spdlog::info("Creating buffers");
    buffers_.emplace_back(allocator_,
                          vec_size_ * sizeof(float) + sizeof(uint32_t),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VMA_MEMORY_USAGE_CPU_TO_GPU, compute_queue_index_);
    buffers_.emplace_back(allocator_,
                          vec_size_ * sizeof(float) + sizeof(uint32_t),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VMA_MEMORY_USAGE_GPU_TO_CPU, compute_queue_index_);
}

void Compute::create_descriptors() {
    spdlog::info("Creatring descriptors");

    std::vector<VkDescriptorPoolSize> pool_sizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}};

    spdlog::info("Creating descriptor pool");
    descriptor_pool_ = DescriptorPool(device_, 1, pool_sizes);

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
         nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
         nullptr}};

    spdlog::info("Creating descriptor set");
    descriptor_set_ = descriptor_pool_.allocate_descriptor_set(bindings);

    spdlog::info("Updating descriptor set");
    descriptor_set_.update(0, 0, 1, 0, VK_WHOLE_SIZE, buffers_.at(0).handle_);
    descriptor_set_.update(1, 0, 1, 0, VK_WHOLE_SIZE, buffers_.at(1).handle_);
}

void Compute::create_pipeline() {
    pipeline_ = Pipeline(device_);

    spdlog::info("Loading SPIR-V binary file");

    std::vector<uint32_t> shader_binary;
    std::ifstream fs("shaders/sum.comp.spv");
    if (!fs.is_open()) {
        spdlog::error("Failed to open shader binary, abort");
        std::exit(EXIT_FAILURE);
    }
    fs.seekg(0, fs.end);
    shader_binary.resize(fs.tellg());
    fs.seekg(0, fs.beg);
    fs.read(reinterpret_cast<char*>(shader_binary.data()),
            shader_binary.size());

    spdlog::info("Binary Size: {}", shader_binary.size());

    spdlog::info("Creating compute pipeline");
    pipeline_.create(shader_binary, descriptor_set_.set_layout_);
}

void Compute::create_command_buffer() {
    command_pool_ = CommandPool(device_, compute_queue_index_);
    command_buffer_ =
        command_pool_.allocate_command_buffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY);
}

void Compute::dispatch() {
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    spdlog::info("Recording command buffer");
    PANIC_BAD_RESULT(vkBeginCommandBuffer(command_buffer_, &begin_info));

    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipeline_.pipeline_);

    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_.pipeline_layout_, 0, 1,
                            &descriptor_set_.handle_, 0, nullptr);

    vkCmdDispatch(command_buffer_, 1, 1, 1);

    PANIC_BAD_RESULT(vkEndCommandBuffer(command_buffer_));

    spdlog::info("Submitting");
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer_;

    PANIC_BAD_RESULT(
        vkQueueSubmit(compute_queue_, 1, &submit_info, VK_NULL_HANDLE));

    spdlog::info("Wainting on queue");
    vkQueueWaitIdle(compute_queue_);
}

void Compute::fill_buffer() {
    std::srand(std::time(nullptr));

    for (auto& buffer : buffers_) {
        char* m = buffer.map();

        *reinterpret_cast<uint32_t*>(m) = vec_size_;
        float* v = reinterpret_cast<float*>(m + sizeof(uint32_t));
        std::stringstream ss;
        for (uint32_t i = 0; i < vec_size_; ++i) {
            v[i] = std::rand() % 10;
            ss << v[i] << " ";
        }
        spdlog::info("vector: size {} [ {}]", vec_size_, ss.str());

        buffer.unmap();
    }
}

void Compute::dump_buffer() {
    char* m = buffers_.at(1).map();

    uint32_t vec_size = *reinterpret_cast<uint32_t*>(m);
    float* v = reinterpret_cast<float*>(m + sizeof(uint32_t));
    std::stringstream ss;
    for (uint32_t i = 0; i < vec_size_; ++i) {
        ss << v[i] << " ";
    }
    spdlog::info("vector: size {} [ {}]", vec_size, ss.str());

    buffers_.at(1).unmap();
}

}  // namespace vkc

int main() {
    vkc::Compute compute;
    compute.init(true);
    compute.get_physical_device();
    compute.init_device();

    compute.create_command_buffer();

    compute.vec_size_ = 10;
    compute.create_buffers();
    compute.create_descriptors();
    compute.create_pipeline();

    compute.fill_buffer();
    compute.dispatch();
    compute.dump_buffer();

    return 0;
}
