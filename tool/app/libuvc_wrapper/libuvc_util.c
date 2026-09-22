#include "libuvc/libuvc.h"

#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <direct.h>
#include <fcntl.h>
#include <io.h>
#else
#include <sys/stat.h>
#endif

//cmake --build /Users/ge83nax/Desktop/code/cpp_opencv/libuvc/build --target libuvc_util

static volatile sig_atomic_t keep_running = 1;

typedef struct capture_options {
  const char *name;
  const char *out_dir;
  int debug;
  int width;
  int height;
  int fps;
  int stream;
  int list;
  int trigger;
  int trigger_specified;
} capture_options_t;

static void debug_log(const capture_options_t *options, const char *message) {
  if (options->debug)
    fprintf(stderr, "[debug] %s\n", message);
}

typedef struct stream_record_header {
  char magic[4];
  uint32_t frame_id;
  uint64_t timestamp_seconds;
  uint32_t timestamp_milliseconds;
  uint32_t width;
  uint32_t height;
  uint32_t frame_format;
  uint64_t frame_bytes;
  uint64_t metadata_bytes;
} stream_record_header_t;

static void stop_handler(int signal_number) {
  (void)signal_number;
  keep_running = 0;
}

static int create_output_directory(const char *path) {
#ifdef _WIN32
  return _mkdir(path);
#else
  return mkdir(path, 0777);
#endif
}

static void usage(const char *program) {
  fprintf(stderr,
      "Usage: %s -name <camera_name> [options]\n"
      "  -list                List available cameras\n"
      "  -resx <pixels>       Requested width\n"
      "  -resy <pixels>       Requested height\n"
      "  -fps <fps>           Requested frame rate\n"
      "  -out <folder_path>   Save frames to this folder\n"
      "  -stream              Write binary frame records to stdout\n"
      "  -debug               Print initialization and streaming diagnostics\n"
      "  -trigger <true|false> Set autofocus/trigger control state\n"
      "\n"
      "Camera name is matched exactly against the product or serial name.\n",
      program);
}

static int parse_positive_int(const char *value, int *result) {
  char *end;
  long parsed;

  errno = 0;
  parsed = strtol(value, &end, 10);
  if (errno || *value == '\0' || *end != '\0' || parsed <= 0 || parsed > INT32_MAX)
    return 0;
  *result = (int)parsed;
  return 1;
}

static int parse_options(int argc, char **argv, capture_options_t *options) {
  int index;

  memset(options, 0, sizeof(*options));
  for (index = 1; index < argc; ++index) {
    if (!strcmp(argv[index], "-name") || !strcmp(argv[index], "-out") ||
        !strcmp(argv[index], "-resx") || !strcmp(argv[index], "-resy") ||
        !strcmp(argv[index], "-fps")) {
      if (index + 1 >= argc) {
        fprintf(stderr, "Missing value for %s\n", argv[index]);
        return 0;
      }
      if (!strcmp(argv[index], "-name")) {
        options->name = argv[++index];
      } else if (!strcmp(argv[index], "-out")) {
        options->out_dir = argv[++index];
      } else if (!strcmp(argv[index], "-resx")) {
        if (!parse_positive_int(argv[++index], &options->width)) return 0;
      } else if (!strcmp(argv[index], "-resy")) {
        if (!parse_positive_int(argv[++index], &options->height)) return 0;
      } else {
        if (!parse_positive_int(argv[++index], &options->fps)) return 0;
      }
    } else if (!strcmp(argv[index], "-debug") || !strcmp(argv[index], "--debug")) {
      options->debug = 1;
    } else if (!strcmp(argv[index], "-stream")) {
      options->stream = 1;
    } else if (!strcmp(argv[index], "-list")) {
      options->list = 1;
    } else if (!strcmp(argv[index], "-trigger")) {
      if (index + 1 >= argc) {
        fprintf(stderr, "Missing value for -trigger (expected true or false)\n");
        return 0;
      }
      if (!strcmp(argv[++index], "true")) {
        options->trigger = 1;
      } else if (!strcmp(argv[index], "false")) {
        options->trigger = 0;
      } else {
        fprintf(stderr, "Invalid value for -trigger: %s (expected true or false)\n",
            argv[index]);
        return 0;
      }
      options->trigger_specified = 1;
    } else {
      fprintf(stderr, "Unknown argument: %s\n", argv[index]);
      return 0;
    }
  }

  if (!options->name && !options->list) {
    fprintf(stderr, "-name is required\n");
    return 0;
  }
  return 1;
}

static uvc_error_t list_devices(uvc_context_t *ctx) {
  uvc_device_t **devices;
  uvc_error_t result;
  size_t index;

  result = uvc_get_device_list(ctx, &devices);
  if (result < 0) return result;

  printf("Available UVC devices:\n");
  for (index = 0; devices[index]; ++index) {
    uvc_device_descriptor_t *descriptor = NULL;

    if (uvc_get_device_descriptor(devices[index], &descriptor) != UVC_SUCCESS)
      continue;
    printf("[%zu] product=\"%s\" serial=\"%s\" manufacturer=\"%s\" vid=0x%04x pid=0x%04x\n",
      index, descriptor->product ? descriptor->product : "[none]",
      descriptor->serialNumber ? descriptor->serialNumber : "[none]",
      descriptor->manufacturer ? descriptor->manufacturer : "[none]",
        descriptor->idVendor, descriptor->idProduct);
    uvc_free_device_descriptor(descriptor);
  }

  uvc_free_device_list(devices, 1);
  return UVC_SUCCESS;
}

static uvc_error_t find_named_device(uvc_context_t *ctx, const char *name,
    uvc_device_t **device) {
  uvc_device_t **devices;
  uvc_error_t result;
  size_t index;

  result = uvc_get_device_list(ctx, &devices);
  if (result < 0) return result;

  for (index = 0; devices[index]; ++index) {
    uvc_device_descriptor_t *descriptor = NULL;
    int matches = 0;

    if (uvc_get_device_descriptor(devices[index], &descriptor) == UVC_SUCCESS) {
      matches = (descriptor->product && !strcmp(descriptor->product, name)) ||
          (descriptor->serialNumber && !strcmp(descriptor->serialNumber, name));
      uvc_free_device_descriptor(descriptor);
    }
    if (matches) {
      uvc_ref_device(devices[index]);
      *device = devices[index];
      uvc_free_device_list(devices, 1);
      return UVC_SUCCESS;
    }
  }

  uvc_free_device_list(devices, 1);
  return UVC_ERROR_NO_DEVICE;
}

static enum uvc_frame_format format_for_descriptor(const uvc_format_desc_t *format) {
  switch (format->bDescriptorSubtype) {
    case UVC_VS_FORMAT_MJPEG: return UVC_COLOR_FORMAT_MJPEG;
    case UVC_VS_FORMAT_FRAME_BASED: return UVC_FRAME_FORMAT_H264;
    default: return UVC_FRAME_FORMAT_YUYV;
  }
}

static unsigned frame_timestamp_seconds(const uvc_frame_t *frame) {
  return (unsigned)(frame->capture_time_finished.tv_sec % 10000);
}

static unsigned frame_timestamp_milliseconds(const uvc_frame_t *frame) {
  return (unsigned)(frame->capture_time_finished.tv_nsec / 1000000);
}

static int save_frame(const capture_options_t *options, const uvc_frame_t *frame) {
  char filename[512];
  FILE *file;
  unsigned seconds = frame_timestamp_seconds(frame);
  unsigned milliseconds = frame_timestamp_milliseconds(frame);

  if (!options->out_dir) return 1;
  if (snprintf(filename, sizeof(filename), "%s/frame_%06u__%04u_%03u.jpg",
      options->out_dir, frame->sequence, seconds, milliseconds) >= (int)sizeof(filename)) {
    fprintf(stderr, "Output path is too long\n");
    return 0;
  }
  file = fopen(filename, "wb");
  if (!file) {
    perror(filename);
    return 0;
  }
  if (fwrite(frame->data, 1, frame->data_bytes, file) != frame->data_bytes) {
    perror(filename);
    fclose(file);
    return 0;
  }
  fclose(file);
  return 1;
}

static int stream_frame(const uvc_frame_t *frame) {
  stream_record_header_t header = {{'U', 'V', 'C', 'F'}, frame->sequence,
  (uint64_t)frame->capture_time_finished.tv_sec,
  frame_timestamp_milliseconds(frame), frame->width, frame->height,
      (uint32_t)frame->frame_format, frame->data_bytes, frame->metadata_bytes};

  if (fwrite(&header, sizeof(header), 1, stdout) != 1 ||
      fwrite(frame->data, 1, frame->data_bytes, stdout) != frame->data_bytes ||
      (frame->metadata_bytes && fwrite(frame->metadata, 1, frame->metadata_bytes, stdout) != frame->metadata_bytes))
    return 0;
  return fflush(stdout) == 0;
}

int main(int argc, char **argv) {
  capture_options_t options;
  uvc_context_t *context = NULL;
  uvc_device_t *device = NULL;
  uvc_device_handle_t *device_handle = NULL;
  uvc_stream_handle_t *stream = NULL;
  uvc_stream_ctrl_t control;
  const uvc_input_terminal_t *camera_terminal;
  const uvc_format_desc_t *format;
  const uvc_frame_desc_t *frame_desc;
  enum uvc_frame_format frame_format;
  uvc_error_t result;
  int width = 0;
  int height = 0;
  int fps = 0;
  int exit_code = 1;

#ifdef _WIN32
  _setmode(_fileno(stdout), _O_BINARY);
#endif

  if (!parse_options(argc, argv, &options)) {
    usage(argv[0]);
    return 2;
  }

  debug_log(&options, "arguments parsed");

  if (options.out_dir && create_output_directory(options.out_dir) && errno != EEXIST) {
    perror(options.out_dir);
    return 2;
  }

  signal(SIGINT, stop_handler);
  signal(SIGTERM, stop_handler);

  debug_log(&options, "initializing libuvc");
  result = uvc_init(&context, NULL);
  if (result < 0) {
    fprintf(stderr, "uvc_init failed with error code %d\n", result);
    uvc_perror(result, "uvc_init");
    return 1;
  }
  debug_log(&options, "libuvc initialized");
  if (options.list) {
    debug_log(&options, "enumerating cameras");
    result = list_devices(context);
    if (result < 0) {
      uvc_perror(result, "list_devices");
      goto cleanup;
    }
    if (!options.name) {
      exit_code = 0;
      goto cleanup;
    }
  }
  debug_log(&options, "looking up requested camera");
  result = find_named_device(context, options.name, &device);
  if (result < 0) {
    fprintf(stderr, "camera not found: %s\n", options.name);
    uvc_perror(result, "find_named_device");
    goto cleanup;
  }
  debug_log(&options, "opening camera");
  result = uvc_open(device, &device_handle);
  if (result < 0) {
    fprintf(stderr, "uvc_open failed with error code %d\n", result);
    uvc_perror(result, "uvc_open");
    goto cleanup;
  }
  camera_terminal = uvc_get_camera_terminal(device_handle);
  if (options.trigger_specified && camera_terminal &&
      (camera_terminal->bmControls & (1ULL << (UVC_CT_FOCUS_AUTO_CONTROL - 1)))) {
    debug_log(&options, "enabling autofocus/trigger control");
    result = uvc_set_focus_auto(device_handle, options.trigger);
    if (result < 0) {
      fprintf(stderr, "Could not enable autofocus/trigger control (error %d)\n", result);
      uvc_perror(result, "uvc_set_focus_auto");
      goto cleanup;
    }
    debug_log(&options, "autofocus/trigger control set");
  } else if (options.trigger_specified) {
    fprintf(stderr, "Camera does not advertise autofocus/trigger control; skipping request\n");
  }

  format = uvc_get_format_descs(device_handle);
  if (!format || !format->frame_descs) {
    fprintf(stderr, "Camera has no stream format\n");
    goto cleanup;
  }
  frame_desc = format->frame_descs;
  width = options.width ? options.width : frame_desc->wWidth;
  height = options.height ? options.height : frame_desc->wHeight;
  fps = options.fps ? options.fps : (int)(10000000 / frame_desc->dwDefaultFrameInterval);
  frame_format = format_for_descriptor(format);
  if (options.debug) {
    fprintf(stderr, "[debug] requested stream: %dx%d at %d fps, format %d\n",
        width, height, fps, frame_format);
  }

  result = uvc_get_stream_ctrl_format_size(device_handle, &control, frame_format,
      width, height, fps);
  if (result < 0) {
    uvc_perror(result, "uvc_get_stream_ctrl_format_size");
    goto cleanup;
  }
  result = uvc_stream_open_ctrl(device_handle, &stream, &control);
  if (result < 0) {
    uvc_perror(result, "uvc_stream_open_ctrl");
    goto cleanup;
  }
  result = uvc_stream_start(stream, NULL, NULL, 0);
  if (result < 0) {
    fprintf(stderr, "uvc_stream_start failed with error code %d\n", result);
    uvc_perror(result, "uvc_stream_start");
    goto cleanup;
  }
  debug_log(&options, "stream started");

  while (keep_running) {
    uvc_frame_t *frame = NULL;
    result = uvc_stream_get_frame(stream, &frame, 100000);
    if (result == UVC_ERROR_TIMEOUT) continue;
    if (result < 0) {
      uvc_perror(result, "uvc_stream_get_frame");
      break;
    }
    if (!frame) continue;
    debug_log(&options, "frame received");
    if (!save_frame(&options, frame) || (options.stream && !stream_frame(frame))) {
      keep_running = 0;
      break;
    }
  }
  exit_code = 0;

cleanup:
  if (stream) {
    uvc_stream_stop(stream);
    uvc_stream_close(stream);
  }
  if (device_handle) uvc_close(device_handle);
  if (device) uvc_unref_device(device);
  if (context) uvc_exit(context);
  return exit_code;
}
