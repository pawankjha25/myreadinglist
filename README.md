**July Reading list**

2014–2017: Foundations of Generative Models

1. Generative Adversarial Nets (GANs)
Ian Goodfellow et al., 2014
🔹 Introduced adversarial learning framework.
🔹 Kickstarted a whole field of generative modeling.

📌 Citation: https://arxiv.org/abs/1406.2661

2. Auto-Encoding Variational Bayes (VAE)
Kingma & Welling, 2014
🔹 Introduced the Variational Autoencoder (VAE).
🔹 Key technique in probabilistic generative modeling.

📌 Citation: https://arxiv.org/abs/1312.6114

3. Pixel Recurrent Neural Networks
van den Oord et al., 2016
🔹 Introduced autoregressive models for images.
🔹 Basis for PixelCNN, PixelRNN.

📌 Citation: https://arxiv.org/abs/1601.06759

4. WaveNet: A Generative Model for Raw Audio
van den Oord et al., DeepMind, 2016
🔹 High-quality speech generation.
🔹 Basis for audio models and later neural codecs.

📌 Citation: https://arxiv.org/abs/1609.03499

🧠 2018–2020: Rise of Transformers and Language Generation
5. Attention is All You Need
Vaswani et al., 2017
🔹 Introduced the Transformer architecture.
🔹 Foundation for all LLMs and multimodal transformers.

📌 Citation: https://arxiv.org/abs/1706.03762

6. BERT: Pre-training of Deep Bidirectional Transformers
Devlin et al., 2018 (Google AI)
🔹 Masked language modeling + fine-tuning.
🔹 Not generative per se, but a turning point in transformer-based modeling.

📌 Citation: https://arxiv.org/abs/1810.04805

7. GPT-2: Language Models are Unsupervised Multitask Learners
OpenAI, 2019
🔹 Showed impressive few-shot/few-label performance.
🔹 Marked the beginning of generative LLMs with practical value.

📌 Blog: https://openai.com/research/language-unsupervised

8. Taming Transformers for High-Resolution Image Synthesis
Esser et al., 2020 (VQ-VAE-2 + Transformers)
🔹 Combined vector quantization and transformers.
🔹 Influenced DALL·E and image tokenization.

📌 Citation: https://arxiv.org/abs/2012.09841

🧠 2021–2022: Diffusion Models and Foundation Models
9. DDPM: Denoising Diffusion Probabilistic Models
Ho et al., 2020 (published 2021)
🔹 Introduced the modern diffusion model framework.
🔹 Key breakthrough in image synthesis, used in DALLE-2, Stable Diffusion.

📌 Citation: https://arxiv.org/abs/2006.11239

10. DALL·E: Zero-Shot Text-to-Image Generation
OpenAI, 2021
🔹 Introduced multimodal generation at scale (text → image).
🔹 Used discrete VAE and transformer decoder.

📌 Blog: https://openai.com/research/dall-e

11. Imagen: Photorealistic Text-to-Image Diffusion Models
Saharia et al., Google Research, 2022
🔹 Achieved SOTA photorealism in text-to-image generation.
🔹 Uses T5 encoder + cascaded diffusion decoders.

📌 Citation: https://arxiv.org/abs/2205.11487

12. GLIDE: Guided Language to Image Diffusion for Generation
Nichol et al., OpenAI, 2021
🔹 Introduced CLIP guidance into diffusion models.
🔹 Predecessor to DALL·E 2.

📌 Citation: https://arxiv.org/abs/2112.10741

🧠 2023–2024: LLMs, Alignment, and Agentic AI
13. GPT-4 Technical Report
OpenAI, 2023
🔹 First multimodal foundation model (image + text).
🔹 No architecture details, but sparked widespread real-world use.

📌 Citation: https://openai.com/research/gpt-4

14. PaLM-E: Embodied Multimodal Language Model
Google DeepMind, 2023
🔹 Combines robotics, vision, and language.
🔹 Early example of embodied and agentic generative AI.

📌 Citation: https://arxiv.org/abs/2303.03378

15. Self-Rewarding Language Agents
Google DeepMind, 2024
🔹 Introduced language agents that generate their own reward signals.
🔹 Foundation for autonomous agentic systems.

📌 Citation: https://arxiv.org/abs/2403.07691

16. LLM-as-a-Judge: Automatic Evaluation of Generated Content
OpenAI, Anthropic, Meta-style labs, 2023–2024
🔹 Popularized use of LLMs to evaluate generative model outputs.
🔹 Now common in RAG, alignment, safety pipelines.

🧠 Bonus: Infrastructure & Training Papers
🔸 LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
https://arxiv.org/abs/2106.09685

🔸 RLHF: Deep RL from Human Preferences (Christiano et al., 2017)
https://arxiv.org/abs/1706.03741

🔸 Transformer Interpretability & Scaling Laws (Kaplan et al., 2020; Anthropic 2022–24)
