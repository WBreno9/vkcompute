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
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    messenger_create_info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT;

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

class Buffer {
   public:
    const VmaAllocator allocator_;
    VmaAllocation allocation_;
    VmaAllocationInfo allocation_info_;
    const VmaMemoryUsage mem_usage_;

    VkBuffer buffer_;
    uint32_t size_;
    uint32_t family_index_;
    VkBufferCreateFlags buffer_usage_;

    Buffer() = default;
    Buffer(const VmaAllocator& allocator, uint32_t size,
           VkBufferCreateFlags buffer_usage, VmaMemoryUsage mem_usage,
           uint32_t family_index);
    ~Buffer();
};

Buffer::Buffer(const VmaAllocator& allocator, uint32_t size,
               VkBufferCreateFlags buffer_usage, VmaMemoryUsage mem_usage,
               uint32_t family_index)
    : allocator_(allocator),
      mem_usage_(mem_usage),
      size_(size),
      family_index_(family_index),
      buffer_usage_(buffer_usage) {
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
                                     &allocation_create_info, &buffer_,
                                     &allocation_, &allocation_info_));
}
Buffer::~Buffer() {}

class DescriptorSet {
   public:
    VkDescriptorSet descriptor_set_;
    VkDescriptorSetLayout descriptor_set_layout_;
    std::vector<VkDescriptorSetLayoutBinding> bindings_;

    VkDevice device_;

    DescriptorSet() = default;
    DescriptorSet(VkDevice device, VkDescriptorSet descriptor_set,
                  VkDescriptorSetLayout descriptor_set_layout,
                  std::vector<VkDescriptorSetLayoutBinding> bindings);

    void update(uint32_t binding, uint32_t start_element,
                uint32_t descriptor_count, uint32_t offset, uint64_t range,
                const VkBuffer& buffer);
};

DescriptorSet::DescriptorSet(VkDevice device, VkDescriptorSet descriptor_set,
                             VkDescriptorSetLayout descriptor_set_layout,
                             std::vector<VkDescriptorSetLayoutBinding> bindings)
    : descriptor_set_(descriptor_set),
      descriptor_set_layout_(descriptor_set_layout),
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
    write_set.dstSet = descriptor_set_;
    write_set.pBufferInfo = &buffer_info;

    vkUpdateDescriptorSets(device_, 1, &write_set, 0, nullptr);
}

class DescriptorPool {
   public:
    VkDescriptorPool descriptor_pool_;
    std::vector<DescriptorSet> descriptor_sets_;
    std::vector<VkDescriptorPoolSize> pool_sizes_;
    uint32_t max_sets_;

    VkDevice device_;

    DescriptorPool() = default;
    DescriptorPool(const VkDevice& device, uint32_t max_sets,
                   const std::vector<VkDescriptorPoolSize>& pool_sizes);

    void create();
    DescriptorSet* allocate_descriptor_set(
        const std::vector<VkDescriptorSetLayoutBinding>& bindings);
};

DescriptorPool::DescriptorPool(
    const VkDevice& device, uint32_t max_sets,
    const std::vector<VkDescriptorPoolSize>& pool_sizes)
    : device_(device), max_sets_(max_sets), pool_sizes_(pool_sizes) {}

void DescriptorPool::create() {
    VkDescriptorPoolCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    create_info.maxSets = max_sets_;
    create_info.poolSizeCount = pool_sizes_.size();
    create_info.pPoolSizes = pool_sizes_.data();

    PANIC_BAD_RESULT(vkCreateDescriptorPool(device_, &create_info, nullptr,
                                            &descriptor_pool_));
}

DescriptorSet* DescriptorPool::allocate_descriptor_set(
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
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &set_layout;

    VkDescriptorSet set;
    PANIC_BAD_RESULT(vkAllocateDescriptorSets(device_, &alloc_info, &set));

    descriptor_sets_.emplace_back(device_, set, set_layout, bindings);
    return &(descriptor_sets_.at(descriptor_sets_.size() - 1));
}

class Pipeline {
   public:
    VkShaderModule shader_;
    VkPipeline pipeline_;
    VkPipelineLayout pipeline_layout_;

    VkDevice device_;

    Pipeline() = default;
    Pipeline(const VkDevice& device);

    void create(const std::vector<uint32_t>& program_src,
                const DescriptorSet& descriptor_set);
};

Pipeline::Pipeline(const VkDevice& device) : device_(device) {}

void Pipeline::create(const std::vector<uint32_t>& program_src,
                      const DescriptorSet& descriptor_set) {
    VkShaderModuleCreateInfo shader_info_ = {};
    shader_info_.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info_.pCode = program_src.data();
    shader_info_.codeSize = program_src.size();

    PANIC_BAD_RESULT(
        vkCreateShaderModule(device_, &shader_info_, nullptr, &shader_));

    VkPipelineLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &descriptor_set.descriptor_set_layout_;

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

class Compute {
   public:
    VkInstance instance_;
    bool use_validation_;

    VkDebugUtilsMessengerEXT debug_messenger_;

    VkPhysicalDevice physical_device_;
    VkPhysicalDeviceProperties physical_device_properties_;
    VkPhysicalDeviceMemoryProperties physical_device_memory_properties_;

    VkDevice device_;
    VkQueue compute_queue_;
    uint32_t compute_queue_index_;

    VmaAllocator allocator_;

    std::vector<const char*> instance_extensions_;
    std::vector<const char*> device_extensions_;

    std::vector<Buffer> buffers_;
    DescriptorPool descriptor_pool_;

    Pipeline pipeline_;

    void init_instance(bool use_validation);
    void get_physical_device();
    void init_device();
    void init_vma_allocator();
    void create_pipeline();
};

void Compute::init_instance(bool use_validation) {
    use_validation_ = use_validation;

    spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("[%T] [%^%l%$] %v");

    spdlog::debug("Initializing volk");
    if (volkInitialize() != VK_SUCCESS) {
        spdlog::error("Failed to initialize volk, abort");
        std::exit(EXIT_FAILURE);
    }

    std::vector<const char*> validation_layers;

    if (use_validation_) {
        instance_extensions_.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        validation_layers.push_back("VK_LAYER_LUNARG_standard_validation");
    }

    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.apiVersion = VK_VERSION_1_1;
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "Null Engine";
    app_info.pApplicationName = "Compute App";

    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.enabledExtensionCount = instance_extensions_.size();
    create_info.ppEnabledExtensionNames = instance_extensions_.data();
    create_info.enabledLayerCount = validation_layers.size();
    create_info.ppEnabledLayerNames = validation_layers.data();
    create_info.pApplicationInfo = &app_info;

    PANIC_BAD_RESULT(vkCreateInstance(&create_info, nullptr, &instance_));

    volkLoadInstance(instance_);

    if (use_validation_) debug_messenger_ = create_debug_messenger(instance_);
}

void Compute::get_physical_device() {
    std::vector<VkPhysicalDevice> physical_devices;
    uint32_t physical_device_count = 0;
    vkEnumeratePhysicalDevices(instance_, &physical_device_count, nullptr);
    physical_devices.resize(physical_device_count);
    vkEnumeratePhysicalDevices(instance_, &physical_device_count,
                               physical_devices.data());

    bool found_device = false;
    for (auto& phys_dev : physical_devices) {
        std::vector<VkQueueFamilyProperties> family_properties;
        uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &family_count,
                                                 nullptr);
        family_properties.resize(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &family_count,
                                                 family_properties.data());

        for (uint32_t i = 0; i < family_properties.size(); ++i) {
            if (family_properties.at(i).queueFlags & VK_QUEUE_COMPUTE_BIT) {
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

    vkGetPhysicalDeviceProperties(physical_device_,
                                  &physical_device_properties_);
    vkGetPhysicalDeviceMemoryProperties(physical_device_,
                                        &physical_device_memory_properties_);
}

void Compute::init_device() {
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info = {};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    queue_create_info.pQueuePriorities = &queue_priority;
    queue_create_info.queueCount = 1;
    queue_create_info.queueFamilyIndex = compute_queue_index_;

    VkDeviceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.enabledExtensionCount = device_extensions_.size();
    create_info.ppEnabledExtensionNames = device_extensions_.data();
    create_info.queueCreateInfoCount = 1;
    create_info.pQueueCreateInfos = &queue_create_info;

    PANIC_BAD_RESULT(
        vkCreateDevice(physical_device_, &create_info, nullptr, &device_));

    volkLoadDevice(device_);

    vkGetDeviceQueue(device_, compute_queue_index_, 0, &compute_queue_);
}

void Compute::init_vma_allocator() {
    spdlog::debug("Initializing VMA Allocator");

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
    allocator_create_info.physicalDevice = physical_device_;
    allocator_create_info.device = device_;
    allocator_create_info.pVulkanFunctions = &vma_functions;

    PANIC_BAD_RESULT(vmaCreateAllocator(&allocator_create_info, &allocator_));
}

void Compute::create_pipeline() {
    pipeline_ = Pipeline(device_);
    // pipeline_.create();
}
}  // namespace vkc

int main() {
    vkc::Compute compute;
    compute.init_instance(true);
    compute.get_physical_device();
    compute.init_device();
    compute.init_vma_allocator();

    return 0;
}
