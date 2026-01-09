# mw-framework
Nature submission


Quick Start: Kerr Black Holes

pip3 install torch numpy pyro-ppl scipy

Output 

First GENERATE test Data:

python3 kerr_1_generate_data.py --samples 10000 --seed 42

Then TEST it:(NON DETERMINISTIC)

python3 kerr_2_train_and_test_pyro.py --data kerr_geodesics.npy

======================================================================

KERR VAE TRAINING & TESTING (WITH FIXED DATA)

======================================================================

Loading data from: kerr_geodesics.npy
✓ Loaded 10000 geodesic points
✓ Loaded metadata
  M = 1.0
  a = 0.9
  r_+ = 1.4359
  seed = 42

Creating VAE (latent_dim=8)...
✓ VAE created with 22,740 parameters

Training VAE with Pyro SVI for 500 epochs...
  Learning rate: 0.0005
  Training data: 10000 points
  Optimizer: Adam
  Loss: ELBO (Evidence Lower BOund)
  Epoch  100: ELBO loss = 13546244.6289, Avg(last 100) = 35661960.8266
  Epoch  200: ELBO loss = 7465180.3281, Avg(last 100) = 10549953.0908
  Epoch  300: ELBO loss = 3718148.8866, Avg(last 100) = 5098391.6396
  Epoch  400: ELBO loss = 2940528.9746, Avg(last 100) = 3237327.6450
  Epoch  500: ELBO loss = 2647110.3594, Avg(last 100) = 2774566.6646
✓ Training complete

======================================================================
KERR BLACK HOLE GEOMETRY TEST
======================================================================

Test points (from dataset):
  A: t=9.46, r=25.27, θ=1.57, φ=0.62
  B: t=18.30, r=13.15, θ=3.04, φ=0.00
  D: t=2.80, r=16.07, θ=2.48, φ=5.63

1. SYNGE WORLD FUNCTION
----------------------------------------------------------------------
  Ω(A,B) = +8.871970
  Ω(A,A) = +0.000000  (should be ~0)
  Ω(A,D) = +4.972084

  Geodesic types:
    A→B: SPACELIKE
    A→D: SPACELIKE

2. VAN VLECK DETERMINANT (FIXED DATA)
----------------------------------------------------------------------
/home/rahul/V4_PYRO/kerr_2_train_and_test_pyro.py:227: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more information. (Triggered internally at /pytorch/build/aten/src/ATen/core/TensorBody.h:489.)
  if z_A.grad is not None:
/home/rahul/V4_PYRO/kerr_2_train_and_test_pyro.py:234: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more information. (Triggered internally at /pytorch/build/aten/src/ATen/core/TensorBody.h:489.)
  if z_B.grad is not None:
  Δ(A,B) = +3.708483e-06
    Uncertainty σ_AB = 519.280273

  Δ(A,D) = -7.575240e-07
    Uncertainty σ_AD = 1148.951782

3. VAN VLECK STATISTICS (10 random pairs from data)
----------------------------------------------------------------------
  Δ_1 = -1.598968e-05
  Δ_2 = -1.101388e-05
  Δ_3 = -6.136211e-07
  ...

  Van Vleck range: [-2.02e-04, 1.37e-05]
  Mean: -2.34e-05
  Std:  6.02e-05

4. FRAME DRAGGING VERIFICATION
----------------------------------------------------------------------
  Equatorial orbit Ω = 0.034781
  Classification: SPACELIKE

======================================================================

Saving model to: kerr_vae.pth
✓ Model saved (including Pyro parameters)

======================================================================
COMPLETE!
======================================================================

 Key achievements:
  ✓ Geometry learned from 10000 Kerr geodesics
  ✓ Van Vleck determinant computed (with FIXED data)
  ✓ Lorentzian signature verified
  ✓ Frame dragging emerged automatically
rahul@rahul-LOQ-15IRH8:~/V4_PYRO$ 




You can also use all in one file below but above is simple:

python3 kerr_blackhole_vae_v1_PATCHED.py (Data is generated within file.. So pl use above  2 files for simplicity)



Authors
Rahul Modak
Founder, Bayesian Cybersecurity Pvt Ltd

Dr. Rahul Walawalkar
Co-Founder, Bayesian Cybersecurity Pvt Ltd

Van Vleck: Δ = 9.367263e+00
Uncertainty: σ = 0.326733
Synge: Ω = 10.883921 (spacelike)
