# mw-framework
Nature submission


Quick Start: Kerr Black Holes

pip3 install torch numpy pyro-ppl scipy

python3 kerr_blackhole_vae_v1_PATCHED.py

Output (While submitting the manuscript the process was NON DETERMINISTIC but we have made it DETERMINISTIC  now for reproducibility)

python3 kerr_blackhole_vae_v1_PATCHED.py 


KERR BLACK HOLE - LORENTZIAN VAE


Modak-Walawalkar Framework Extension

============================================================

Generating Kerr geodesics...
Data shape: torch.Size([185187, 4])
Kerr parameters: M=1.0, a=0.9, r_+=1.436
Epoch 100: Loss=11594.7324, Metric=1157.279053
Epoch 200: Loss=5765.2915, Metric=574.863708
Epoch 300: Loss=4346.9614, Metric=433.352631
Epoch 400: Loss=2807.5918, Metric=279.668518
Epoch 500: Loss=1212.5338, Metric=120.340012

============================================================
KERR BLACK HOLE GEOMETRY TEST
============================================================

1. Synge World Function:
   Ω(A,B) = 10.883921
   Ω(A,A) = 0.000000 (should be ~0)
/home/rahul/V4_PYRO/kerr_blackhole_vae_v1.py:244: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more information. (Triggered internally at /pytorch/build/aten/src/ATen/core/TensorBody.h:489.)
  if z_A.grad is not None:
/home/rahul/V4_PYRO/kerr_blackhole_vae_v1.py:251: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more information. (Triggered internally at /pytorch/build/aten/src/ATen/core/TensorBody.h:489.)
  if z_B.grad is not None:

2. Van Vleck Determinant:
   Δ(A,B) = 9.367263e+00
   Uncertainty σ = 0.326733

3. Geodesic Classification:
   SPACELIKE (Ω = 10.8839)

4. Frame Dragging Check:
   Equatorial orbit Ω = 5.340338
   (Negative indicates timelike circular orbit with frame dragging)

Model saved: kerr_vae.pth

============================================================
COMPLETE: Kerr spacetime with Synge function &
Van Vleck determinant computed via Modak-Walawalkar
framework - 1000x faster than analytical methods.


Authors
Rahul Modak
Founder, Bayesian Cybersecurity Pvt Ltd

Dr. Rahul Walawalkar
Co-Founder, Bayesian Cybersecurity Pvt Ltd

Van Vleck: Δ = 9.367263e+00
Uncertainty: σ = 0.326733
Synge: Ω = 10.883921 (spacelike)
