---
layout: post
title: "Train a tiny generative multimodal language model"
date: 2025-01-04
categories: vision-language-model, multimodal
---

In this post, we will train a tiny multimodal language model that can generate both text and images, also called a vision language model. Vision language models that can operate on both text and image data have shown very good performance on text/image generation, understanding, question-answering, and other tasks. With the larger multimodal (mostly closed-source) models, these capabilites are extended to include audio and video as well. Vision language models can be trained in different ways (see overview in [this review paper](https://arxiv.org/abs/2405.17247)), we will make some simplifying choices for our tiny experiment. Code for the whole implementation is at [https://github.com/suyashss/mnist_vlm](https://github.com/suyashss/mnist_vlm).

### Design choices
* Dataset: MNIST - We will use the MNIST dataset of digit images for our experiment. This is a small dataset, and will allow faster iteration. We can also make our model smaller and train it on less powerful GPUs.

* Tasks: Image generation and classification - The "text" information we will use is the digit label for each image. So our model will be capable of two tasks (1) generation: label -> image (2) classification: image -> label.

* Image representation - We will represent each image as a fixed number of discrete tokens. For this, we will use a vector-quantized variational autoencoder (VQVAE) that we will train on the MNIST images. VQVAEs are an extension of variational autoencoders that provide an additional encoding of the input as a set of indices from a codebook (more details [here](https://arxiv.org/abs/1711.00937)).  

* Training pipeline: nanoGPT - We will use the [nanoGPT training pipeline](https://github.com/karpathy/nanoGPT) from Andrej Karpathy. This is designed for text data, so we should be able to use this to inspect our image representations. We will use 0 through 9, lowercase a through z, and punctuations to represent our text, and uppercase A through Z to represent our image tokens.

## Implementation

Code for the implementation is at [https://github.com/suyashss/mnist_vlm](https://github.com/suyashss/mnist_vlm). For nanoGPT, I created a fork of Andrei Karpathy's original repo with some modifications (described below), at [https://github.com/suyashss/nanoGPT](https://github.com/suyashss/nanoGPT). 

To implement this as you read along, click on the "how to run" buttons like the one below.

<details>
  <summary>how to run</summary>

{% highlight shell %}
git clone https://github.com/suyashss/mnist_vlm
cd mnist_vlm
git clone https://github.com/suyashss/nanoGPT
{% endhighlight %}

</details>

### Step 0: Train/validation/test splits
Set up train/test splits to avoid data leakage. For the VQVAE stage, we will split the MNIST training data into two unequal parts, using the larger part for the VQVAE training, and the smaller for validation. For the full VLM training, we will similarly use the MNIST training split as training and validation, and MNIST validation as the held-out test set.

### Step 1: Tokenization

To train our model, we need to convert both our text and images into tokens. For text, we will use a simple character-based tokenizer, every character is a token. For images, we will train a VQVAE to get a discrete token representation of the image.

#### VQVAE training
Ww will train a VQVAE to reconstruct MNIST images. From this training, we will extract two components (1) an encoder that can represent an MNIST image as a sequence of 8 tokens from a vocabulary of size 26, (2) a decoder that can take a sequence of 8 tokens from a vocabulary of size 26, and return a 28x28 image.

The VQVAE model code is described in `vqvae_model.py`. The encoder initially produces a 32-dimensional embedding of each 28x28 image. It then cuts up this embedding into 8 chunks of size 4. Each 4-d chunk is approximated by its closest neighbor from a codebook of 26 4-d vectors. Thus, each image is encoded as a sequence of 8 indices, each between 0-25. The definition of the VQVAE class is below.

```python
class VQVAE(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_embeddings, quantization_embedding_dim, commitment_cost, learning_rate):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)  # Encoder to produce latent representation
        self.quantizer = VectorQuantizer(num_embeddings, quantization_embedding_dim, commitment_cost)  # Vector quantizer to discretize the latent representation
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)  # Decoder to reconstruct the input from the quantized representation
        self.learning_rate = learning_rate
        self.latents = [] # Store some encoder output for visualization
        self.labels = [] # Store some labels for visualization
```

The VQAE model training is described in `vqvae_train.py`. WandB training logs for the runs are available [here](https://api.wandb.ai/links/suyashss-123test/zd8hixwz). The final model is saved in `vqvae_mnist_final_model.pth`.

<details>
  <summary>how to run</summary>

{% highlight shell %}
python vqvae_train.py
{% endhighlight %}

</details>

We can see the clustering of image based on the un-quantized VAE embedding below. The model can separate most digits well, with 4 and 9 being hard to separate (other pairs with overlapping representations are 3 and 5, 7 and 9). Since the LLM will see an approximation to these embeddings as input, it might also face problems separating the same pairs.

![Plot of t-SNE embedding of the VAE embeddings a validation batch](/assets/images/tiny-mnist-vlm/vqvae_clustering.png)


### Step 2: Data preprocessing

#### Construct VLM datasets as text

We will take the VLM training and validation sets, and construct a text examples corresponding to one of our two tasks. For each (image i,number n) pair, we will construct a classification example with 50% probability, or an image generation example with 50% probability. For each image, we will get the encoding from the VQVAE, and represent the tokens 1-26 as uppercase letters A-Z. Assuming image `i` for number `n` is represented by the 8-character string `"ABCDEFGH"` (corresponding to codebook indices `[0,1,2,3,4,5,6,7]` from the VQVAE), the converted examples will look like:
* Classification -> `image:ABCDEFGH, number:n.######`
* Generation -> `number:n, image:ABCDEFGH.######`

We use "#" to pad every sequence to 32 characters.

Code for creating the text datasets is in `format_vlm_inputs.py`. The main loop is below.

```python
 for idx, (image, label) in enumerate(data_loader):
    # Get the output from the model (encode/decode or latent representation)
    with torch.no_grad():
        # Get image representation as codebook indices
        _,_,_,representation = model(image)  
        img_tokens = img2text(representation)
    # Generate a sentence with 50% probability for each type
    if random.random() < 0.5:
        sentence = f"number:{label.item()}, image:{img_tokens}."
    else:
        sentence = f"image:{img_tokens}, number:{label.item()}."

    if len(sentence) < maxlen:
        sentence += ("#"*(maxlen - len(sentence)))
```
<details>
  <summary>how to run</summary>

{% highlight shell %}
python format_vlm_inputs.py
{% endhighlight %}

</details>


#### Create nanoGPT input
Prepare the data for nanoGPT training by converting the text sequences to their token indices in the VLM vocabulary. 

Code for the nanoGPT data preparation is in `data/mnist_vlm/prepare.py` in my nanoGPT fork.

<details>
  <summary>how to run</summary>

{% highlight shell %}
mkdir ./nanoGPT/data/mnist_vlm/
mv *.txt ./nanoGPT/data/mnist_vlm/
cd nanoGPT
python ./data/mnist_vlm/prepare.py
{% endhighlight %}

</details>

Before we start training, let's look at our image tokens in the text. Each image is now represented as 8-character string containing A-Z. To visualize these, we can create histograms of the letters at each position, and look for patterns. This is only a partial visualization since it looks at each position separately, but could still be informative. Such plots of frequencies from character strings are called [sequence logos](https://en.wikipedia.org/wiki/Sequence_logo) and are commonly used in analysis of DNA sequences. We can use the `logomaker` python package to easily create these plots. The visualization of the images from the first 1,000 lines in the training file for nanoGPT is below.

![Plot of letter histograms](/assets/images/tiny-mnist-vlm/image_token_plot.png)

Here, we have 10 plots, one for each digit. Let's take the digit 0 plot from the top left for example - it has 8 histograms represented by columns, one for each position. In each histogram column, the size of the letter shows how frequently the letter appears in the image tokenization. More frequent letters are at the top, less frequent letters are at the bottom. Each letter is colored differently to get an impression of the histograms. 

Overall, by the letter and color patterns, we can see that the digits have different token/text representations. By looking at just the first position, we can differentiate many of the digits from each other. We can also see that 4 and 9 have somewhat similar first positions.

### Step 3: nanoGPT training
We train nanoGPT to predict next tokens from the training sequences. We will need to make two minor modifications to the original pipeline for our use case:
* Sampling at sequence starts - In the original code, the input is assumed to be a single large piece of text, and so training chunks are sampled at random positions in the text. In our setting, training across two sequences doesn't make sense, so we require that each training chunk start a valid sequence start position.
* Saving intermediate checkpoints - We want to see how model performance changes during training, so we'll save some intermediate checkpoints.

We will use a small model with ~15M parameters, so we can train it on a laptop. The training config I used for my laptop is defined in `config/train_mnist_vlm.py` in my nanoGPT fork. The training reached a train/validation loss of about 0.6 in an hour on my M1 Macbook Air with 8 GB RAM. In a test on Google colab with a T4 GPU, I was able to reach a train/validation loss of about 0.58 in 35 minutes using a larger batch size of 2048.

<details>
  <summary>how to run</summary>

{% highlight shell %}
python train.py config/train_mnist_vlm.py
{% endhighlight %}

</details>


### Step 4: Evaluation
To evaluate the model, we will create masked prompts from the held-out test set, and evaluate performance on the classification task and image generation task. For classification, we can use prediction accuracy (proportion of examples where the model predicts the number label correctly). For image generation, we will perform a visual inspection of generated outputs. Code for the evaluation is in `mnist_vlm_eval.ipynb`.

#### Classification evaluation
Let's check the models prediction accuracy over iterations. We will load checkpoints for various iterations, use those to generate completions for prompts like `image:ABCDEFGH, number:`, and ask the model to predict what number label comes next. We can then compare it to the true number label. The code is shown below. In addition to comparing prediction accuracy, it also computes the standard error of the accuracy estimate, so we can be sure the differences we see across iterations are statistically meaningful.

```python
ckpts = [x*500 for x in [1] + list(range(2,11,2))]
accuracy_list = []
se_list = []
for ckpt in ckpts:
    print(ckpt)
    model,encode,decode = setup_model_and_meta("./nanoGPT/out-mnist-vlm/",f"ckpt_{ckpt}.pt")
    classifn_preds_raw = minimal_multi_sample(model,masked_classifn_egs,encode,decode)
    classifn_preds = np.array([x.split(".")[0].split(":")[-1] for x in classifn_preds_raw])
    prediction_accuracy = np.mean(labels_classifn_egs == classifn_preds)
    accuracy_list.append(prediction_accuracy)
    se_list.append(np.sqrt(prediction_accuracy*(1-prediction_accuracy)/len(labels_classifn_egs)))
```

![Plot of prediction accuracy over iterations](/assets/images/tiny-mnist-vlm/prediction_accuracy_plot.png)
**We can clearly see the prediction accuracy increase from about 0.78 at iteration 500 to about 0.93 at iteration 5000. So the model has learnt to classify MNIST images better over the training iterations.**

Looking more carefully at the error patterns, we see that the most common misclassification types are 4 to 9 and vice versa (around 5-7% of the time). The next most common is 7 to 9 and vice versa (around 4% of the time), and the last one is from 5 to 3, also 4% of the time. These match well with what we expected from our clustering visualization. These could be improved by improving our VQVAE to improve the quality of input to the model. 

#### Generation evaluation 
For evaluating generation, let's get the model to generate image completions from the prompts like `number:n, image:` and examine the generated images. To do this, we will have to take the generated output (letters), convert them back to indices from 0 to 25, and pass them to the VQVAE decoder. The function to do this is:

```python
def get_reconstruction(vqvae_model, letterstr):
    try:
        with torch.no_grad():
            indices = torch.tensor([ord(x) - ord('A') for x in list(letterstr)])
            quantized = vqvae_model.quantizer.embedding(indices).view(1,-1)
            recon = vqvae_model.decoder(quantized).numpy().squeeze()
    except IndexError as e: # handle case when a letter is not between A and Z
        recon = np.zeros((28,28))
    return recon
```

First, here are the generations from the model at iteration 500. The generations for digits 0 and 1 are pretty good, showing the model has started learning image patterns. But most other generations are quite fuzzy. For digit 7, the 7th generated image is all dark, indicating the model output a character outside the allowed range for image tokens. A number of the generated images are hard to identify as any digit at all.
![Plot of generated images at iteration 500](/assets/images/tiny-mnist-vlm/generation_plot_ckpt_500.png)


Now, here are the generations from the model at iteration 5000. These look much sharper overall, and most digits are clearly identifiable. Some error patterns are 4 to 9 and vice versa, 5 to 6 but not the reverse, and the occasional blurry image (the 2nd image for 3, and the 1st image for 8). There is still room for improvement, but this is much better.
![Plot of generated images at iteration 5000](/assets/images/tiny-mnist-vlm/generation_plot_ckpt_5000.png)

**Clearly, the model has learned to improve performance at this task too, producing sharper images that are mostly identfiable as the correct digit .**

## Summary

We have now built a tiny (multimodal) vision language model on the MNIST data. Our model can classify images to numbers, as well as draw numbers. To do this, we implemented a VQVAE to tokenize images into a set of discrete tokens, created sequences from our MNIST examples, and trained an autoregressive LLM on this sequence data. In principle, this approach could be applied to any number of modalities to jointly model them in a single autoregressive model, as done by [CM3Leon](https://arxiv.org/abs/2309.02591) and [Chameleon](https://arxiv.org/abs/2405.09818) from Meta. Alternatively, once all modalities are tokenized, one could train a single masked language model on the data, as done by [4M](https://arxiv.org/abs/2312.06647) from Apple.