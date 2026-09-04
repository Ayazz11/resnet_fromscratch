# Robust Image Classification Under Distribution Shift

This project trains a deep learning model to classify small (32×32 pixel) color images into 10 categories : airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck, while staying accurate even when the images are blurry, noisy, or have shifted colors. It's built entirely from scratch: no pretrained models are used.

## Why this is harder than normal image classification

Most beginner image classifiers are trained and tested on clean, well-behaved pictures. This project is deliberately harder in two ways:

1. **The test images are "shifted."** They're not exactly like the training images — they might be blurrier, noisier, or have different lighting/color than what the model trained on. A model that just memorizes the training data's exact look will fall apart here. A model that learns the *actual shape and structure* of a cat or a truck, not just its typical colors and sharpness, will win.
2. **Some classes have way fewer examples than others.** In the training data:

   | Class | Training Images |
   |---|---|
   | airplane, automobile, bird, cat | ~4,250 each |
   | deer, dog | ~3,400 each |
   | frog, horse | ~425 each |
   | ship | 255 |
   | truck | 85 |

   That means there are **50 times more automobile pictures than truck pictures.** Without special handling, a model would basically ignore truck entirely, since guessing "not truck" is almost always right and barely hurts overall accuracy. This project uses several techniques specifically to prevent that.

## How the model is trained

At a high level, this project:
1. Builds a deep neural network from scratch (no pretrained weights).
2. Trains it with a bag of techniques designed to (a) handle the rare classes fairly and (b) make the model robust to corrupted/shifted images, not just clean ones.
3. Checks its progress honestly, using a validation set that's deliberately corrupted the same way the real test set is expected to be.
4. Fine-tunes its decision-making after training (a step called "calibration").
5. At the very end, makes its final predictions by looking at each test image from several different angles and averaging its guesses, this is known "test-time augmentation," explained below.

## The model: Wide ResNet

The model is a **Wide ResNet** (specifically, "WRN-40-8"), a well-studied type of convolutional neural network :

- **Convolutional neural network (CNN):** a network that scans an image with small filters to detect patterns — edges, textures, shapes - building up from simple patterns to whole-object understanding.
- **Residual ("Res") connections:** shortcuts that let information skip over some layers. This makes very deep networks easier to train, since gradients can flow backward without fading out.
- **"Wide":** instead of making the network extremely deep, this design makes each layer wider (more filters per layer), which tends to work very well for smaller images like these 32×32 pictures.
- **40 / 8:** "40" is the network's depth (40 layers), "8" is how many times wider it is than a baseline version.

This is a real architecture from published deep learning research (WRN - Zagoruyko & Komodakis, 2016), reimplemented here from scratch in plain PyTorch — no external library or pretrained checkpoint involved.

## Training techniques.

This project doesn't just train the network in the most basic way — it stacks several proven techniques together. Here's what each one does and why it's here:

- **MixUp:** occasionally blends two training images (and their labels) together. This teaches the model to be less overconfident and generalize better to unusual inputs.
- **CutMix:** occasionally cuts a rectangular patch from one image and pastes it onto another, mixing the labels proportionally to the patch size. Similar goal to MixUp, different mechanism.
- **RandAugment:** randomly applies a mix of standard image tweaks (rotation, color changes, sharpness, etc.) during training so the model sees a wider variety of looks for the same object.
- **Class-balanced sampling ("square-root inverse" sampling):** instead of showing the model images in their natural (very imbalanced) proportions, the training loop oversamples the rare classes (frog, horse, ship, truck) so they show up far more often per epoch than their raw counts would suggest — without going so far that they're *artificially* as common as the majority classes, which can backfire.
- **Label smoothing:** instead of training the model to be 100% certain about every prediction, it's trained to be, say, 95% confident and spread a little uncertainty across the other classes. This makes the model less brittle and less prone to being confidently wrong.
- **Alternating "clean" and "robustness" training batches:** every other epoch, the model trains on images with heavier corruption-style augmentation (blur, noise, color shifts) instead of the standard augmentation. This directly practices the skill the model is being tested on — handling distorted images — without abandoning normal training entirely.
- **EMA (Exponential Moving Average) model:** alongside the main model, a second "shadow" copy is kept that's a slowly-updated running average of the main model's weights over time. EMA models are usually smoother and more reliable than the raw model at any single training snapshot, similar to judging a runner's pace by their average speed over the last mile rather than their single fastest step.
- **Cosine learning rate schedule with warmup:** the learning rate (how big a step the model takes when correcting its mistakes) starts small, ramps up over the first few epochs, then gradually decreases in a smooth curve for the rest of training. This avoids instability early on and lets the model settle into a good solution late in training.
- **Mixed-precision training (AMP):** uses lower-precision number formats where safe to speed up training on the GPU without meaningfully hurting accuracy.
- **Gradient clipping:** caps how large a single training update can be, preventing rare unstable spikes from throwing off training.

## validation

Two separate validation sets are used, and the difference between them matters a lot:

- **Clean validation set:** held-out images with no extra corruption — like an ordinary vision benchmark.
- **Robust validation set:** the *same* held-out images, but with blur, noise, and color distortion applied — deliberately mimicking what the real test set is expected to look like. This is created **once, at the very start of training**, and reused for every single epoch after that — not re-randomized each time. If it were re-randomized every epoch, the score used to judge the model would itself be a moving target, making it impossible to tell whether the model was actually improving or the corruption just happened to be easier that epoch.

The model that ends up saved as the "best" one isn't just whichever model scored highest on a single lucky epoch — the training loop averages the robust score over the last several epochs first, since any single epoch's score can be noisy (especially for the very-rare classes with few validation examples).

**Early stopping:** if the model goes 60 epochs without improving on this smoothed robust score (and only after at least 100 epochs have already happened, so it doesn't stop before the training schedule has had a chance to mature), training stops automatically to avoid wasting time.

## Post-training calibration

After training finishes, there's one more optimization step: **class-bias calibration**. In plain terms, the model's raw output for each class gets a small hand-tuned nudge (up or down) to squeeze a little more performance out of the exact same trained model, without changing any of its learned weights. This is found by trying small adjustments for each class one at a time and keeping whichever adjustment helps the most, repeating a few passes over all classes.

*A note on scientific honesty:* this calibration step is tuned directly against the validation set it's also being scored on, so the *reported* improvement from calibration should be treated with mild caution — it may not transfer perfectly to genuinely new data (like the real competition test set). The pre-calibration numbers are the more trustworthy, "clean" measurement.

## Test-time augmentation (TTA)

When it's finally time to make predictions on the test set, the model doesn't just look at each image once. It looks at each image from up to 8 different "views" — the original, a horizontally flipped version, a slightly cropped version, versions with mild brightness/contrast/saturation/blur changes, and so on — and averages its confidence across all of them before making a final decision. This tends to smooth out occasional mistakes that come from the model being thrown off by one specific detail in one specific version of the image.

## Results

On the held-out validation data:

| Metric | Clean images | Corrupted ("robust") images |
|---|---|---|
| Accuracy | 93.45% | 89.71% |
| Macro F1 score* | 88.31% | 85.23% |

*Macro F1 score treats every class equally important, regardless of how many training images it had — so a model can't hide poor performance on rare classes behind strong performance on common ones. This is the primary metric this project is judged on.

The relatively small gap between clean and corrupted performance (roughly 3–4 points) is a good sign — it suggests the model learned features that generalize, rather than memorizing the exact look of the clean training images.

**Per-class performance on the corrupted validation set:**

| Class | F1 Score | Training Images |
|---|---|---|
| automobile | 0.98 | 4,250 |
| airplane | 0.93 | 4,250 |
| deer | 0.91 | 3,400 |
| bird | 0.90 | 4,250 |
| horse | 0.88 | 425 |
| dog | 0.83 | 3,400 |
| frog | 0.83 | 425 |
| cat | 0.84 | 4,250 |
| ship | 0.81 | 255 |
| **truck** | **0.61** | **85** |

Every class except one clears 0.80+ F1 — a genuinely strong, well-rounded result. **Truck is the clear exception**, and the reason is visible right in the table: it has drastically fewer training examples than everything else. No training technique fully overcomes a real shortage of source data; the class-balancing techniques in this project meaningfully lift truck's performance compared to a naive training run, but they can't manufacture information that isn't there in only 85 images.

## limitations

- **The "robust" validation set is a synthetic stand-in**, built using blur/noise/color-jitter that the project's authors chose — it's a reasonable proxy for the real test set's distribution shift, but it isn't guaranteed to match the real shift exactly. The true measure of success is the model's score on the actual competition test set.
- **Truck remains a weak point**, capped by data scarcity rather than model quality.
- **The calibration step's reported gain may be optimistic**, since it was tuned and measured on the same validation data.

## Project files

- `train_images/`, `test_images/`, `train_labels.csv`, `classes.txt` — the dataset.
- `best_model.pth` — the saved weights of the best-performing model checkpoint found during training.
- `class_bias.npy` — the small per-class calibration adjustments learned after training.
