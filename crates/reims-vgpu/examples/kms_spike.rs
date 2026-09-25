//! M3 gate D1: can this process take a DRM connector through `VK_KHR_display`
//! and present on it, with no window system at all?
//!
//! Standalone on purpose: plain `ash`, none of the device's engine, so the only
//! question answered is the display one. It opens the card, chooses the
//! connector and mode the way `host_display` does, acquires the connector with
//! `vkAcquireDrmDisplayEXT`, builds a display plane surface and a FIFO
//! swapchain, and clears the images red, green and blue in turn, one colour a
//! second, for `KMS_SPIKE_SECONDS` (default 20).
//!
//! Output and exit codes:
//! - `kms_spike presented frames=<n> mode=<w>x<h>@<hz>`, exit 0;
//! - `kms_spike refused reason=<no_connector|no_mode|drm_open>`, exit 3;
//! - `kms_spike refused reason=acquire_refused vk=<VkResult>`, exit 4;
//! - `kms_spike failed step=<step> vk=<VkResult>`, exit 5.
//!
//! Environment: `REIMS_VGPU_DRM_CARD` (default `/dev/dri/card1`),
//! `REIMS_VGPU_CONNECTOR` (default: the first connected one).

#[cfg(target_os = "linux")]
fn main() {
    std::process::exit(linux::run());
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("kms_spike: DRM/KMS is Linux-only");
    std::process::exit(2);
}

#[cfg(target_os = "linux")]
mod linux {
    use std::ffi::CStr;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use ash::vk;
    use reims_vgpu::host_display::drm::Card;
    use reims_vgpu::host_display::select::choose;

    const COLOURS: [[f32; 4]; 3] = [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ];

    /// A Vulkan step that failed, for the `failed` line.
    struct Failed(&'static str, vk::Result);

    trait Step<T> {
        fn step(self, name: &'static str) -> Result<T, Failed>;
    }

    impl<T> Step<T> for Result<T, vk::Result> {
        fn step(self, name: &'static str) -> Result<T, Failed> {
            self.map_err(|r| Failed(name, r))
        }
    }

    pub fn run() -> i32 {
        let card_path = PathBuf::from(
            std::env::var("REIMS_VGPU_DRM_CARD").unwrap_or_else(|_| "/dev/dri/card1".into()),
        );
        let wanted = std::env::var("REIMS_VGPU_CONNECTOR").ok();
        let card = match Card::open(&card_path) {
            Ok(card) => card,
            Err(error) => {
                println!(
                    "kms_spike refused reason=drm_open card={} error={error}",
                    card_path.display()
                );
                return 3;
            }
        };
        let connectors = match card.connectors() {
            Ok(list) => list,
            Err(error) => {
                println!("kms_spike refused reason=drm_open error={error}");
                return 3;
            }
        };
        let choice = match choose(&connectors, wanted.as_deref()) {
            Ok(choice) => choice,
            Err(error) => {
                println!("kms_spike refused reason={} ({error})", error.slug());
                return 3;
            }
        };
        println!(
            "kms_spike chose connector={} id={} mode={}x{}@{}mHz",
            choice.connector,
            choice.connector_id,
            choice.mode.width,
            choice.mode.height,
            choice.mode.refresh_mhz
        );
        let seconds = std::env::var("KMS_SPIKE_SECONDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(20u64);
        // SAFETY: every Vulkan call below uses handles created in this function
        // and destroyed in reverse order before it returns.
        match unsafe { present(card.fd(), &choice, Duration::from_secs(seconds)) } {
            Ok(Outcome::Presented { frames, hz }) => {
                println!(
                    "kms_spike presented frames={frames} mode={}x{}@{hz}",
                    choice.mode.width, choice.mode.height
                );
                0
            }
            Ok(Outcome::AcquireRefused(result)) => {
                println!("kms_spike refused reason=acquire_refused vk={result:?}");
                4
            }
            Err(Failed(step, result)) => {
                println!("kms_spike failed step={step} vk={result:?}");
                5
            }
        }
    }

    enum Outcome {
        Presented { frames: u64, hz: u32 },
        AcquireRefused(vk::Result),
    }

    unsafe fn present(
        drm_fd: i32,
        choice: &reims_vgpu::host_display::select::Choice,
        duration: Duration,
    ) -> Result<Outcome, Failed> {
        let entry = ash::Entry::load()
            .map_err(|_| Failed("load", vk::Result::ERROR_INITIALIZATION_FAILED))?;
        let extensions: [&CStr; 4] = [
            ash::khr::surface::NAME,
            ash::khr::display::NAME,
            ash::ext::direct_mode_display::NAME,
            ash::ext::acquire_drm_display::NAME,
        ];
        let extension_ptrs: Vec<_> = extensions.iter().map(|e| e.as_ptr()).collect();
        let app = vk::ApplicationInfo::default()
            .application_name(c"kms_spike")
            .api_version(vk::API_VERSION_1_1);
        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app)
                    .enabled_extension_names(&extension_ptrs),
                None,
            )
            .step("create_instance")?;
        let result = with_instance(&entry, &instance, drm_fd, choice, duration);
        instance.destroy_instance(None);
        result
    }

    unsafe fn with_instance(
        entry: &ash::Entry,
        instance: &ash::Instance,
        drm_fd: i32,
        choice: &reims_vgpu::host_display::select::Choice,
        duration: Duration,
    ) -> Result<Outcome, Failed> {
        let pd = *instance
            .enumerate_physical_devices()
            .step("enumerate_physical_devices")?
            .first()
            .ok_or(Failed(
                "physical_device",
                vk::Result::ERROR_INITIALIZATION_FAILED,
            ))?;
        let drm = ash::ext::acquire_drm_display::Instance::new(entry, instance);
        let direct = ash::ext::direct_mode_display::Instance::new(entry, instance);
        let display_ext = ash::khr::display::Instance::new(entry, instance);
        let surface_ext = ash::khr::surface::Instance::new(entry, instance);

        let display = match drm.get_drm_display(pd, drm_fd, choice.connector_id) {
            Ok(display) => display,
            Err(result) => return Ok(Outcome::AcquireRefused(result)),
        };
        if let Err(result) = drm.acquire_drm_display(pd, drm_fd, display) {
            return Ok(Outcome::AcquireRefused(result));
        }
        let result = with_display(
            instance,
            &display_ext,
            &surface_ext,
            pd,
            display,
            choice,
            duration,
        );
        let _ = (direct.fp().release_display_ext)(pd, display);
        result
    }

    unsafe fn with_display(
        instance: &ash::Instance,
        display_ext: &ash::khr::display::Instance,
        surface_ext: &ash::khr::surface::Instance,
        pd: vk::PhysicalDevice,
        display: vk::DisplayKHR,
        choice: &reims_vgpu::host_display::select::Choice,
        duration: Duration,
    ) -> Result<Outcome, Failed> {
        let (w, h) = (u32::from(choice.mode.width), u32::from(choice.mode.height));
        let modes = display_ext
            .get_display_mode_properties(pd, display)
            .step("get_display_mode_properties")?;
        let mode = modes
            .iter()
            .filter(|m| {
                (
                    m.parameters.visible_region.width,
                    m.parameters.visible_region.height,
                ) == (w, h)
            })
            .min_by_key(|m| m.parameters.refresh_rate.abs_diff(choice.mode.refresh_mhz))
            .ok_or(Failed(
                "display_mode",
                vk::Result::ERROR_FORMAT_NOT_SUPPORTED,
            ))?;
        let planes = display_ext
            .get_physical_device_display_plane_properties(pd)
            .step("get_physical_device_display_plane_properties")?;
        let mut plane = None;
        for index in 0..planes.len() as u32 {
            let supported = display_ext
                .get_display_plane_supported_displays(pd, index)
                .step("get_display_plane_supported_displays")?;
            if supported.contains(&display) {
                plane = Some(index);
                break;
            }
        }
        let plane = plane.ok_or(Failed(
            "display_plane",
            vk::Result::ERROR_INITIALIZATION_FAILED,
        ))?;
        let extent = vk::Extent2D {
            width: w,
            height: h,
        };
        let surface = display_ext
            .create_display_plane_surface(
                &vk::DisplaySurfaceCreateInfoKHR::default()
                    .display_mode(mode.display_mode)
                    .plane_index(plane)
                    .plane_stack_index(planes[plane as usize].current_stack_index)
                    .transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                    .global_alpha(1.0)
                    .alpha_mode(vk::DisplayPlaneAlphaFlagsKHR::OPAQUE)
                    .image_extent(extent),
                None,
            )
            .step("create_display_plane_surface")?;
        let hz = mode.parameters.refresh_rate.div_ceil(1000);
        let result = with_surface(instance, surface_ext, pd, surface, extent, duration)
            .map(|frames| Outcome::Presented { frames, hz });
        surface_ext.destroy_surface(surface, None);
        result
    }

    unsafe fn with_surface(
        instance: &ash::Instance,
        surface_ext: &ash::khr::surface::Instance,
        pd: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        extent: vk::Extent2D,
        duration: Duration,
    ) -> Result<u64, Failed> {
        let families = instance.get_physical_device_queue_family_properties(pd);
        let mut family = None;
        for (index, props) in families.iter().enumerate() {
            let index = index as u32;
            if props.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && surface_ext
                    .get_physical_device_surface_support(pd, index, surface)
                    .step("get_physical_device_surface_support")?
            {
                family = Some(index);
                break;
            }
        }
        let family = family.ok_or(Failed(
            "present_queue",
            vk::Result::ERROR_INITIALIZATION_FAILED,
        ))?;
        let priorities = [1.0];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];
        let device = instance
            .create_device(
                pd,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queue_info)
                    .enabled_extension_names(&device_extensions),
                None,
            )
            .step("create_device")?;
        let result = with_device(
            instance,
            &device,
            surface_ext,
            pd,
            surface,
            family,
            extent,
            duration,
        );
        device.destroy_device(None);
        result
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn with_device(
        instance: &ash::Instance,
        device: &ash::Device,
        surface_ext: &ash::khr::surface::Instance,
        pd: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        family: u32,
        extent: vk::Extent2D,
        duration: Duration,
    ) -> Result<u64, Failed> {
        let caps = surface_ext
            .get_physical_device_surface_capabilities(pd, surface)
            .step("get_physical_device_surface_capabilities")?;
        let formats = surface_ext
            .get_physical_device_surface_formats(pd, surface)
            .step("get_physical_device_surface_formats")?;
        let format = formats
            .iter()
            .find(|f| f.format == vk::Format::B8G8R8A8_UNORM)
            .or(formats.first())
            .copied()
            .ok_or(Failed(
                "surface_format",
                vk::Result::ERROR_FORMAT_NOT_SUPPORTED,
            ))?;
        let mut images_wanted = caps.min_image_count.max(3);
        if caps.max_image_count > 0 {
            images_wanted = images_wanted.min(caps.max_image_count);
        }
        let swapchain_ext = ash::khr::swapchain::Device::new(instance, device);
        let swapchain = swapchain_ext
            .create_swapchain(
                &vk::SwapchainCreateInfoKHR::default()
                    .surface(surface)
                    .min_image_count(images_wanted)
                    .image_format(format.format)
                    .image_color_space(format.color_space)
                    .image_extent(extent)
                    .image_array_layers(1)
                    .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                    .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                    .present_mode(vk::PresentModeKHR::FIFO)
                    .clipped(true),
                None,
            )
            .step("create_swapchain")?;
        let result = with_swapchain(device, &swapchain_ext, swapchain, family, duration);
        let _ = device.device_wait_idle();
        swapchain_ext.destroy_swapchain(swapchain, None);
        result
    }

    unsafe fn with_swapchain(
        device: &ash::Device,
        swapchain_ext: &ash::khr::swapchain::Device,
        swapchain: vk::SwapchainKHR,
        family: u32,
        duration: Duration,
    ) -> Result<u64, Failed> {
        let images = swapchain_ext
            .get_swapchain_images(swapchain)
            .step("get_swapchain_images")?;
        let queue = device.get_device_queue(family, 0);
        let pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .step("create_command_pool")?;
        let cmd = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .step("allocate_command_buffers")?[0];
        let acquired = device
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            .step("create_semaphore")?;
        // One "cleared" semaphore per image: the present engine may still hold
        // the previous one when the next frame signals.
        let mut cleared = Vec::with_capacity(images.len());
        for _ in &images {
            cleared.push(
                device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .step("create_semaphore")?,
            );
        }
        let fence = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .step("create_fence")?;

        let result = frames(
            device,
            swapchain_ext,
            swapchain,
            queue,
            cmd,
            &images,
            acquired,
            &cleared,
            fence,
            duration,
        );

        let _ = device.device_wait_idle();
        device.destroy_fence(fence, None);
        for semaphore in cleared {
            device.destroy_semaphore(semaphore, None);
        }
        device.destroy_semaphore(acquired, None);
        device.destroy_command_pool(pool, None);
        result
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn frames(
        device: &ash::Device,
        swapchain_ext: &ash::khr::swapchain::Device,
        swapchain: vk::SwapchainKHR,
        queue: vk::Queue,
        cmd: vk::CommandBuffer,
        images: &[vk::Image],
        acquired: vk::Semaphore,
        cleared: &[vk::Semaphore],
        fence: vk::Fence,
        duration: Duration,
    ) -> Result<u64, Failed> {
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let start = Instant::now();
        let mut frames = 0u64;
        while start.elapsed() < duration {
            let colour = COLOURS[(start.elapsed().as_secs() % 3) as usize];
            let (index, _) = swapchain_ext
                .acquire_next_image(swapchain, u64::MAX, acquired, vk::Fence::null())
                .step("acquire_next_image")?;
            let image = images[index as usize];
            device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .step("reset_command_buffer")?;
            device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .step("begin_command_buffer")?;
            let to_clear = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_clear],
            );
            device.cmd_clear_color_image(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue { float32: colour },
                &[range],
            );
            let to_present = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::empty())
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_present],
            );
            device.end_command_buffer(cmd).step("end_command_buffer")?;
            let wait = [acquired];
            let stages = [vk::PipelineStageFlags::TRANSFER];
            let signal = [cleared[index as usize]];
            let cmds = [cmd];
            device
                .queue_submit(
                    queue,
                    &[vk::SubmitInfo::default()
                        .wait_semaphores(&wait)
                        .wait_dst_stage_mask(&stages)
                        .command_buffers(&cmds)
                        .signal_semaphores(&signal)],
                    fence,
                )
                .step("queue_submit")?;
            let swapchains = [swapchain];
            let indices = [index];
            swapchain_ext
                .queue_present(
                    queue,
                    &vk::PresentInfoKHR::default()
                        .wait_semaphores(&signal)
                        .swapchains(&swapchains)
                        .image_indices(&indices),
                )
                .step("queue_present")?;
            device
                .wait_for_fences(&[fence], true, u64::MAX)
                .step("wait_for_fences")?;
            device.reset_fences(&[fence]).step("reset_fences")?;
            frames += 1;
        }
        Ok(frames)
    }
}
