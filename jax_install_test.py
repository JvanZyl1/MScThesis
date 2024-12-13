import jax

devices = jax.devices()
print("Available devices:", devices)

if any(device.device_kind == 'gpu' for device in devices):
    print("GPU is available.")
else:
    print("No GPU found.")