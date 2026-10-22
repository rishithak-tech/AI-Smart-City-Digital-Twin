import torch

from unet_model import LightweightUNet


# ============================================
# CREATE MODEL
# ============================================

model = LightweightUNet(
    num_classes=7
)


# ============================================
# CREATE DUMMY IMAGE
# ============================================

image = torch.randn(
    1,
    3,
    227,
    320
)


# ============================================
# FORWARD PASS
# ============================================

output = model(image)


# ============================================
# PRINT RESULTS
# ============================================

print("\n====================================")
print("U-NET TEST")
print("====================================")

print("\nInput shape:")
print(image.shape)

print("\nOutput shape:")
print(output.shape)

print("\nExpected:")
print(
    "[1, 7, height, width]"
)

print("\nNumber of parameters:")

print(
    sum(
        parameter.numel()
        for parameter in model.parameters()
    )
)

print("\n====================================")
print("U-NET FORWARD PASS PASSED")
print("====================================")